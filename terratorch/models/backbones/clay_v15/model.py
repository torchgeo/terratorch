import math
import random

import timm
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn
from torchvision.transforms import v2

from terratorch.models.backbones.clay_v15.backbone import Transformer
from terratorch.models.backbones.clay_v15.factory import DynamicEmbedding
from terratorch.models.backbones.clay_v15.utils import posemb_sincos_2d_with_gsd

# NOTE: Do not mutate global torch/env state at import time. Both
# ``torch.set_float32_matmul_precision`` and ``TORCH_CUDNN_V8_API_DISABLED``
# are process-wide, so merely importing this module changed numerics for
# every other model in the same process. Set the ``terratorch_FLOAT_32_PRECISION``
# env var (see ``terratorch.cli_tools``) to opt into reduced precision instead.


class Encoder(nn.Module):
    def __init__(
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.patch_embedding = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            fused_attn=True,
        )

    def to_patch_embed(self, cube, waves):
        """Split the input cube into patches & create embeddings per patch"""
        patches, waves_encoded = self.patch_embedding(cube, waves)  # [B L D]
        return patches, waves_encoded  # ([B L D], [N D])

    def add_encodings(self, patches, time, latlon, gsd):
        """Add position encoding to the patches"""
        B, L, D = patches.shape

        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2

        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size,
                w=grid_size,
                dim=(self.dim - 8),
                gsd=gsd,
            )
            .to(patches.device)
            .detach()
        )  # [L (D - 8)]

        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)  # [B L 8]
        pos_metadata_encoding = torch.cat((pos_encoding, time_latlon), dim=-1)  # [B L D]

        patches = patches + pos_metadata_encoding  # [B L D] + [B L D] -> [B L D]
        return patches  # [B L D]

    def mask_out(self, patches):
        """
        Mask out patches randomly by shuffling the patches & masking out the
        first N patches

        Parameters
        ----------
        patches : torch.Tensor A tensor of shape (B, L, D)

        Returns
        -------
        unmasked_patches : torch.Tensor
            A tensor of shape (B, L:(1 - mask_ratio), D) containing the
            embeddings of the unmasked patches.
        unmasked_indices : torch.Tensor
            A tensor of shape (B, (1 - mask_ratio)) containing the indices of
            the unmasked patches.
        masked_indices : torch.Tensor
            A tensor of shape (B, mask_ratio) containing the indices of the
            masked patches.
        masked_matrix : torch.Tensor
            A tensor of shape (B, L) containing the mask matrix, 1 indicates a masked
            patch & 0 indicates an unmasked patch.
        """
        B, L, D = patches.shape
        # assert (
        #     L == self.num_patches
        # ), f"Expected {self.num_patches} patches, got {L} patches."

        if self.shuffle:  # Shuffle the patches
            noise = torch.randn((B, L), device=patches.device)  # [B L]
        else:  # Don't shuffle, useful for interpolation & inspection of embeddings
            noise = rearrange(torch.arange(B * L, device=patches.device), "(B L) -> B L", B=B, L=L)

        random_indices = torch.argsort(noise, dim=-1)  # [B L]
        reverse_indices = torch.argsort(random_indices, dim=-1)  # [B L]

        num_masked_patches = int(self.mask_ratio * self.num_patches)  # Number of patches to be masked out
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],  # [B mask_ratio * L]
            random_indices[:, num_masked_patches:],  # [B (1 - mask_ratio) * L]
        )

        # create a mask of shape B L, where 1 indicates a masked patch
        # and 0 indicates an unmasked patch
        masked_matrix = torch.zeros((B, L), device=patches.device)  # [B L] = 0
        masked_matrix[:, :num_masked_patches] = 1  # [B mask_ratio * L] = 1
        masked_matrix = torch.gather(
            masked_matrix, dim=1, index=reverse_indices
        )  # [B L] -> [B L] - reorder the patches

        # mask out the patches
        batch_indices = rearrange(torch.arange(B, device=patches.device), "B -> B 1")  # [B 1]
        unmasked_patches = patches[batch_indices, unmasked_indices, :]  # [B L:(1 - mask_ratio) D]
        _ = patches[batch_indices, masked_indices, :]  # [B L:mask_ratio D]

        return (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B L:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B L]

    def forward(self, datacube):
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )  # [B C H W]

        B, C, H, W = cube.shape

        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D] - patchify & create embeddings per patch
        # TODO: Add time & latlon as encoding to patches
        patches = self.add_encodings(
            patches,
            time,
            latlon,
            gsd,
        )  # [B L D] - add position encoding to the embeddings

        # mask out patches
        (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.mask_out(patches)  # [B L:(1 - mask_ratio) D], [(1-mask_ratio)], [mask_ratio], [B L]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        unmasked_patches = torch.cat((cls_tokens, unmasked_patches), dim=1)  # [B (1 + L) D]

        # pass the unmasked patches through the transformer
        encoded_unmasked_patches = self.transformer(unmasked_patches)  # [B ((1 + L)):(1 - mask_ratio)) D]

        return (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )  # [B ((1 + L):(1 - mask_ratio)) D], [(1-mask_ratio)], [mask_ratio], [B L]


class Decoder(nn.Module):
    def __init__(
        self,
        mask_ratio,
        patch_size,
        encoder_dim,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.dim = dim

        self.enc_to_dec = nn.Linear(encoder_dim, dim) if encoder_dim != dim else nn.Identity()
        self.mask_patch = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            fused_attn=True,
        )
        self.embed_to_pixels = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=True,
        )

    def reconstruct_and_add_encoding(
        self,
        unmasked_patches,
        unmasked_indices,
        masked_indices,
        masked_matrix,
        time,
        latlon,
        gsd,
    ):
        B, L = masked_matrix.shape
        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2
        cls_tokens, unmasked_patches = (
            unmasked_patches[:, :1, :],
            unmasked_patches[:, 1:, :],
        )  # [B 1 D], [B L:(1 - mask_ratio) D]

        pos_encoding = (
            posemb_sincos_2d_with_gsd(h=grid_size, w=grid_size, dim=(self.dim - 8), gsd=gsd)
            .to(unmasked_patches.device)
            .detach()
        )  # [L D]
        time_latlon = torch.hstack((time, latlon)).to(unmasked_patches.device).detach()  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)  # [B L 8]
        pos_metadata_encoding = torch.cat((pos_encoding, time_latlon), dim=-1)  # [B L D]

        batch_indices = rearrange(torch.arange(B, device=unmasked_patches.device), "B -> B 1")  # [B 1]

        num_masked_patches = int(self.mask_ratio * self.num_patches)
        masked_patches = repeat(self.mask_patch, "D -> B L D", B=B, L=num_masked_patches)  # [B L:mask_ratio D]

        # Add position encoding
        masked_patches = (
            masked_patches + pos_metadata_encoding[batch_indices, masked_indices, :]
        )  # [B L:mask_ratio D] + [B L:mask_ratio D]
        unmasked_patches = (
            unmasked_patches + pos_metadata_encoding[batch_indices, unmasked_indices, :]
        )  # [B GL:(1 - masked_ratio) D] + [B GL:(1 - mask_ratio) D]

        # Concatenate the masked & unmasked patches
        decoder_patches = torch.zeros((B, self.num_patches, self.dim), device=unmasked_patches.device)  # [B L D]
        decoder_patches[batch_indices, unmasked_indices, :] = unmasked_patches  # [B L:(1 - mask_ratio) D])
        decoder_patches[batch_indices, masked_indices, :] = masked_patches  # [B L:mask_ratio D])

        decoder_patches = torch.cat((cls_tokens, decoder_patches), dim=1)  # [B (1 + L) D]

        return decoder_patches  # [B (1 + L) D]

    def forward(
        self,
        encoded_unmasked_patches,
        unmasked_indices,
        masked_indices,
        masked_matrix,
        time,
        latlon,
        gsd,
        waves,
    ):
        # Change the embedding dimension from encoder to decoder
        encoded_unmasked_patches = self.enc_to_dec(encoded_unmasked_patches)  # [B (1 + L) D]

        # Reconstruct the patches to feed into the decoder transformer
        decoder_patches = self.reconstruct_and_add_encoding(
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
            time,
            latlon,
            gsd,
        )  # [B (1 + L) D]

        # Pass the decoder patches through the transformer
        decoded_patches = self.transformer(decoder_patches)  # [B (1 + L) D]

        pixels, waves = self.embed_to_pixels(decoded_patches, waves)  # [B (1 + L) (C P P)]
        # Remove the class token
        pixels = pixels[:, 1:, :]
        return pixels, waves  # [B L (C P P)], [B N]


class ClayMAE(nn.Module):
    def __init__(
        self,
        mask_ratio,
        patch_size,
        norm_pix_loss,
        shuffle,
        metadata,
        teacher,
        dolls,
        doll_weights,
        # ENCODER
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        # DECODER
        decoder_dim,
        decoder_depth,
        decoder_heads,
        decoder_dim_head,
        decoder_mlp_ratio,
        **kwargs,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.shuffle = shuffle
        self.metadata = metadata
        self.teacher = timm.create_model(teacher, pretrained=True, num_classes=0)
        self.teacher_chip_size = 518
        self.teacher_resize = v2.Resize(size=(self.teacher_chip_size, self.teacher_chip_size))
        # self.mrl = MRL(features=self.teacher.num_features, dolls=dolls)
        # self.mrl_loss = MRLLoss(weights=doll_weights)
        self.proj = nn.Linear(dim, self.teacher.num_features)

        self.encoder = Encoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            shuffle=shuffle,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
        )

        self.decoder = Decoder(
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            encoder_dim=dim,
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_ratio=decoder_mlp_ratio,
        )

        self.freeze_teacher()

    def freeze_teacher(self):
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def per_pixel_loss(self, cube, pixels, masked_matrix):
        """
        cube: [B C H W]
        pixels: [B L (C P P)]
        masked_matrix: [B L], 0 is unmasked, 1 is masked
        """
        patches = rearrange(
            cube,
            "B C (h p1) (w p2) -> B (h w) (C p1 p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )  # [B L (C P P)]

        if self.norm_pix_loss:
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, keepdim=True)
            patches = (patches - mean) / (var + 1e-6) ** 0.5

        loss = F.l1_loss(patches, pixels, reduction="none")  # loss per pixel
        loss = reduce(loss, "B L D -> B L", reduction="mean")  # loss per patch

        loss = (loss * masked_matrix).sum() / masked_matrix.sum()  # loss on masked patches only

        return loss

    def forward(self, datacube):
        """
        datacube: dict containing the following keys:
            - pixels: [B C H W]
            - time: [B 4] # week hour
            - latlon: [B 4] # lat lon
            - platform: [B 1]
            - date: [B 1]
        """
        platform = datacube["platform"][0]
        waves = torch.tensor(list(self.metadata[platform].bands.wavelength.values()))
        gsd = torch.tensor(self.metadata[platform].gsd)

        # Drop channels randomly
        _pixels = datacube["pixels"].clone()
        batch_size, channels, _, _ = _pixels.size()

        # Define probabilities for dropping channels
        prob_drop_all = 0.10  # 10% probability to drop all channels
        prob_drop_half = 0.20  # 20% probability to drop half the channels

        for i in range(batch_size):
            if torch.any(datacube["latlon"][i] != 0):  # Check if latlon is not all zeros
                rand_val = random.random()
                if rand_val < prob_drop_all:
                    _pixels[i, :, :, :] = 0  # Drop all channels
                elif rand_val < prob_drop_all + prob_drop_half:
                    channel_indices = torch.randperm(channels)[: channels // 2]  # Get 50% of channel indices
                    _pixels[i, channel_indices, :, :] = 0  # Drop 50% of channels

        # ENCODER
        (
            encoded_unmasked_patches,  # [B (1 + L):(1 - mask_ratio) D]
            unmasked_indices,  # [(1-mask_ratio)]
            masked_indices,  # [mask_ratio]
            masked_matrix,  # [B L]
        ) = self.encoder(
            {
                "pixels": _pixels,
                "time": datacube["time"],
                "latlon": datacube["latlon"],
                "gsd": gsd,
                "waves": waves,
            }
        )

        # DECODER
        pixels, waves = self.decoder(
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
            datacube["time"],
            datacube["latlon"],
            gsd,
            waves,
        )  # [B L (C P P)]

        # MAE
        reconstruction_loss = self.per_pixel_loss(datacube["pixels"], pixels, masked_matrix)
        # MODIS has a 10x reconstruction loss compared to all the other sensors,
        # so we need to scale it down to improve the learning capability.
        if platform == "modis":
            reconstruction_loss /= 10

        # # MRL
        # representations = self.mrl(encoded_unmasked_patches[:, 0, :])  # [(B D') ...]

        # PROJ
        representations = self.proj(encoded_unmasked_patches[:, 0, :])  # [B D']

        with torch.no_grad():
            if platform == "sentinel-1-rtc":
                r = datacube["pixels"][:, 0, :, :]
                g = datacube["pixels"][:, 1, :, :]
                b = (r + g) / 2
                rgb = torch.stack((r, g, b), dim=1)
            else:
                # Read RGB bands from the sensor to feed the teacher model
                indices = self.metadata[platform].rgb_indices
                rgb = datacube["pixels"][:, indices, :, :]
            rgb = self.teacher_resize(rgb)
            target = self.teacher(rgb)
            # target = self.teacher(rgb)

        # representation_loss = self.mrl_loss(representations, target)
        representation_loss = 1.0 - F.cosine_similarity(representations, target).mean()

        loss = 0.9 * reconstruction_loss + 0.1 * representation_loss
        return (loss, reconstruction_loss, representation_loss)


def clay_mae_tiny(**kwargs):
    args = {
        # ENCODER
        "dim": 192,
        "depth": 6,
        "heads": 4,
        "dim_head": 48,
        "mlp_ratio": 2,
        # DECODER
        "decoder_dim": 96,
        "decoder_depth": 3,
        "decoder_heads": 2,
        "decoder_dim_head": 48,
        "decoder_mlp_ratio": 2,
    }
    args.update(kwargs)
    return ClayMAE(**args)


def clay_mae_small(**kwargs):
    args = {
        # ENCODER
        "dim": 384,
        "depth": 6,
        "heads": 6,
        "dim_head": 64,
        "mlp_ratio": 2,
        # DECODER
        "decoder_dim": 192,
        "decoder_depth": 4,
        "decoder_heads": 4,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 2,
    }
    args.update(kwargs)
    return ClayMAE(**args)


def clay_mae_base(**kwargs):
    args = {
        # ENCODER
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "dim_head": 64,
        "mlp_ratio": 4,
        # DECODER
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 4,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return ClayMAE(**args)


def clay_mae_large(**kwargs):
    args = {
        # ENCODER
        "dim": 1024,
        "depth": 24,
        "heads": 16,
        "dim_head": 64,
        "mlp_ratio": 4,
        # DECODER
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 4,
        "decoder_dim_head": 64,
        "decoder_mlp_ratio": 4,
    }
    args.update(kwargs)
    return ClayMAE(**args)
