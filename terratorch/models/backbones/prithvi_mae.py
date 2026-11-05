# Copyright (c) IBM Corp. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# transformers: https://github.com/huggingface/transformers
# --------------------------------------------------------

import logging
import warnings
from functools import reduce
from operator import mul

import numpy as np
import torch
from einops import rearrange
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block
from torch import nn

logger = logging.getLogger(__name__)


def get_3d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 3D sin/cos positional embeddings.

    Args:
        embed_dim (int):
            Embedding dimension.
        grid_size (tuple[int, int, int] | list[int]):
            The grid depth, height and width.
        add_cls_token (bool, *optional*, defaults to False):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size[0]*grid_size[1]*grid_size[2], embed_dim) or
        (1+grid_size[0]*grid_size[1]*grid_size[2], embed_dim): the position embeddings (with or without cls token)
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _get_1d_sincos_embed_from_grid_torch(embed_dim: int, pos: torch.Tensor):
    """Modified torch version of *get_1d_sincos_pos_embed_from_grid()*.

    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,) - must be float dtype!
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    assert pos.dtype in [torch.float32, torch.float16, torch.bfloat16]

    omega = torch.arange(embed_dim // 2, dtype=pos.dtype).to(pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

    return emb


def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def _interpolate_pos_encoding(
    pos_embed: torch.Tensor,
    grid_size: tuple[int, int, int] | list[int],
    patch_size: tuple[int, int, int] | list[int],
    shape: tuple[int, int, int],
    embed_dim: int,
):
    """
    Adapted from:
    - transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding,
    - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194
    """
    t, h, w = shape
    t_patches = t // patch_size[0]
    h_patches = h // patch_size[1]
    w_patches = w // patch_size[2]

    if [t_patches, h_patches, w_patches] == grid_size:
        # No interpolation needed
        return pos_embed
    if t_patches != grid_size[0]:
        # Re-compute pos embedding to handle changed num_frames
        new_grid_size = (t_patches, *grid_size[1:])
        new_pos_embed = get_3d_sincos_pos_embed(pos_embed.shape[-1], new_grid_size, add_cls_token=True)
        new_pos_embed = torch.from_numpy(new_pos_embed).float().unsqueeze(0).to(pos_embed.device)
    else:
        new_grid_size = grid_size
        new_pos_embed = pos_embed

    class_pos_embed, patch_pos_embed = new_pos_embed[:, :1], new_pos_embed[:, 1:]

    patch_pos_embed = patch_pos_embed.reshape(*new_grid_size, embed_dim).permute(0, 3, 1, 2)

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        size=(h_patches, w_patches),
        mode="bicubic",
        align_corners=True,
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)

    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


class PatchEmbed(nn.Module):
    """3D version of timm.models.vision_transformer.PatchEmbed"""

    def __init__(
        self,
        input_size: tuple[int, int, int] = (1, 224, 224),
        patch_size: tuple[int, int, int] = (1, 16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = [s // p for s, p in zip(self.input_size, self.patch_size)]
        assert self.grid_size >= [1, 1, 1], "Patch size is bigger than input size."
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape

        if T / self.patch_size[0] % 1 or H / self.patch_size[1] % 1 or W / self.patch_size[2] % 1:
            warnings.warn(
                f"Input {x.shape[-3:]} is not divisible by patch size {self.patch_size}."
                f"The border will be ignored, add backbone_padding for pixel-wise tasks."
            )

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.year_embed_dim = embed_dim // 2
        self.julian_day_embed_dim = embed_dim - self.year_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer("scale", torch.ones(1))

    def forward(self, temporal_coords: torch.Tensor, tokens_per_frame: int | None = None):
        """
        Args:
            temporal_coords: year and day-of-year info with shape (B, T, 2).
            tokens_per_frame: number of tokens for each frame in the sample. If provided, embeddings will be
                repeated over T dimension, and final shape is (B, T*tokens_per_frame, embed_dim).
        """
        shape = temporal_coords.shape[:2] + (-1,)  # B, T, -1

        year = _get_1d_sincos_embed_from_grid_torch(self.year_embed_dim, temporal_coords[:, :, 0].flatten()).reshape(
            shape
        )
        julian_day = _get_1d_sincos_embed_from_grid_torch(
            self.julian_day_embed_dim, temporal_coords[:, :, 1].flatten()
        ).reshape(shape)

        embedding = self.scale * torch.cat([year, julian_day], dim=-1)

        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)

        return embedding  # B, T*tokens_per_frame, embed_dim


class LocationEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer("scale", torch.ones(1))

    def forward(self, location_coords: torch.Tensor):
        """
        location_coords: lat and lon info with shape (B, 2).
        """
        shape = location_coords.shape[:1] + (1, -1)  # B, 1, -1

        lat = _get_1d_sincos_embed_from_grid_torch(self.lat_embed_dim, location_coords[:, 0].flatten()).reshape(shape)
        lon = _get_1d_sincos_embed_from_grid_torch(self.lon_embed_dim, location_coords[:, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([lat, lon], dim=-1)

        return embedding  # B, 1, embed_dim


class PrithviViT(nn.Module):
    """Prithvi ViT Encoder"""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int, int] = (1, 16, 16),
        num_frames: int = 1,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        coords_encoding: list[str] | None = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
        vpt: bool = False,
        vpt_n_tokens: int | None = None,
        vpt_dropout: float = 0,
        **kwargs,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.img_size = to_2tuple(img_size)
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)

        # 3D patch embedding
        self.patch_embed = PatchEmbed(
            input_size=(num_frames,) + self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.out_channels = [embed_dim * self.patch_embed.grid_size[0]] * depth

        # Optional temporal and location embedding
        coords_encoding = coords_encoding or []
        self.temporal_encoding = "time" in coords_encoding
        self.location_encoding = "location" in coords_encoding
        if self.temporal_encoding:
            assert patch_size[0] == 1, f"With temporal encoding, patch_size[0] must be 1, received {patch_size[0]}"
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        # Transformer layers
        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                )
            )
        self.blocks = nn.ModuleList(self.blocks)

        self.norm = norm_layer(embed_dim)

        self.vpt = vpt
        self.vpt_n_tokens = vpt_n_tokens
        self.vpt_dropout = vpt_dropout
        if self.vpt:
            if self.vpt_n_tokens is None:
                msg = "vpt_n_tokens must be provided when using VPT"
                raise ValueError(msg)
            self.vpt_prompt_embeddings = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, self.vpt_n_tokens, embed_dim)) for _ in range(depth)]
            )
            self.vpt_dropout_layers = nn.ModuleList([nn.Dropout(vpt_dropout) for _ in range(depth)])

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, add_cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

        # initialize VPT prompt embeddings
        if self.vpt:
            # extracted from https://github.com/KMnP/vpt/blob/4410440ec1b489f24f66b9fad3d9b10ff3443567/src/models/vit_prompt/vit.py#L57
            val = np.sqrt(6.0 / float(3 * reduce(mul, self.patch_embed.patch_size[1:], 1) + self.embed_dim))
            for emb in self.vpt_prompt_embeddings:
                nn.init.uniform_(emb, -val, val)

    def random_masking(self, sequence, mask_ratio, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.FloatTensor` of shape `(batch_size, sequence_length, dim)`)
            mask_ratio (float): mask ratio to use.
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def interpolate_pos_encoding(self, sample_shape: tuple[int, int, int]):

        pos_embed = _interpolate_pos_encoding(
            pos_embed=self.pos_embed,
            grid_size=self.patch_embed.grid_size,
            patch_size=self.patch_embed.patch_size,
            shape=sample_shape,
            embed_dim=self.embed_dim,
        )
        return pos_embed

    def forward(
        self,
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        mask_ratio=0.75,
    ):
        if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
            # add time dim
            x = x.unsqueeze(2)
        sample_shape = x.shape[-3:]

        # embed patches
        x = self.patch_embed(x)

        pos_embed = self.interpolate_pos_encoding(sample_shape)
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding and location_coords is not None:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        bs = x.shape[0]
        for idx, block in enumerate(self.blocks):
            if self.vpt:
                x = torch.cat(
                    (
                        x[:, :1, :],
                        self.vpt_dropout_layers[idx](self.vpt_prompt_embeddings[idx].expand(bs, -1, -1)),
                        x[:, 1:, :],
                    ),
                    dim=1,
                )  # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
            x = block(x)
            if self.vpt:
                x = torch.cat(
                    (x[:, :1, :], x[:, (1 + self.vpt_n_tokens) :, :]),
                    dim=1,
                )
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_features(
        self,
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
    ) -> list[torch.Tensor]:
        if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
            # add time dim
            x = x.unsqueeze(2)
        sample_shape = x.shape[-3:]

        # embed patches
        x = self.patch_embed(x)

        pos_embed = self.interpolate_pos_encoding(sample_shape)
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding and location_coords is not None:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        bs = x.shape[0]
        out = []
        for idx, block in enumerate(self.blocks):
            if self.vpt:
                x = torch.cat(
                    (
                        x[:, :1, :],
                        self.vpt_dropout_layers[idx](self.vpt_prompt_embeddings[idx].expand(bs, -1, -1)),
                        x[:, 1:, :],
                    ),
                    dim=1,
                )  # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
            x = block(x)
            if self.vpt:
                x = torch.cat(
                    (x[:, :1, :], x[:, (1 + self.vpt_n_tokens) :, :]),
                    dim=1,
                )
            out.append(x.clone())

        x = self.norm(x)
        out[-1] = x
        return out

    def prepare_features_for_image_model(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        effective_time_dim = self.patch_embed.input_size[0] // self.patch_embed.patch_size[0]
        for x in features:
            x_no_token = x[:, 1:, :]
            number_of_tokens = x_no_token.shape[1]
            tokens_per_timestep = number_of_tokens // effective_time_dim
            h = int(np.sqrt(tokens_per_timestep))
            encoded = rearrange(
                x_no_token,
                "batch (t h w) e -> batch (t e) h w",
                e=self.embed_dim,
                t=effective_time_dim,
                h=h,
            )
            out.append(encoded)
        return out

    def freeze(self):
        for n, param in self.named_parameters():
            if "vpt_prompt_embeddings" not in n:
                param.requires_grad_(False)


class MAEDecoder(nn.Module):
    """Transformer Decoder used in the Prithvi MAE"""

    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = (1, 16, 16),
        grid_size: list[int] | tuple[int, int, int] = (3, 14, 14),
        in_chans: int = 3,
        encoder_embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        coords_encoding: list[str] | None = None,
        coords_scale_learn: bool = False,
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_dim = decoder_embed_dim
        self.grid_size = grid_size
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        self.patch_size = patch_size
        self.num_frames = self.grid_size[0] * patch_size[0]
        num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Optional temporal and location embedding
        coords_encoding = coords_encoding or []
        self.temporal_encoding = "time" in coords_encoding
        self.location_encoding = "location" in coords_encoding
        if self.temporal_encoding:
            self.temporal_embed_dec = TemporalEncoder(decoder_embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_dec = LocationEncoder(decoder_embed_dim, coords_scale_learn)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.register_buffer("decoder_pos_embed", torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size[0] * patch_size[1] * patch_size[2] * in_chans, bias=True
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.grid_size, add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def interpolate_pos_encoding(self, sample_shape: tuple[int, int, int]):

        pos_embed = _interpolate_pos_encoding(
            pos_embed=self.decoder_pos_embed,
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            shape=sample_shape,
            embed_dim=self.decoder_embed_dim,
        )

        return pos_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        ids_restore: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        input_size: list[int] = None,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)
        cls_token = x[:, :1, :]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x.device))

        # add pos embed
        decoder_pos_embed = self.interpolate_pos_encoding(input_size[-3:])
        cls_token = cls_token + decoder_pos_embed[:, :1, :]
        x = x + decoder_pos_embed[:, 1:, :]

        if self.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_dec(temporal_coords, num_tokens_per_frame)
            # Add temporal encoding w/o cls token
            x = x + temporal_encoding
        if self.location_encoding and location_coords is not None:
            location_encoding = self.location_embed_dec(location_coords)
            # Add location encoding w/o cls token
            x = x + location_encoding

        # append cls token
        x = torch.cat([cls_token, x], dim=1)

        # apply Transformer layers (blocks)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        pred = pred[:, 1:, :]

        return pred


class PrithviMAE(nn.Module):
    """Prithvi Masked Autoencoder"""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int, int] = (1, 16, 16),
        num_frames: int = 4,
        in_chans: int = 6,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
        coords_encoding: list[str] | None = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
        mask_ratio: float = 0.75,
        **kwargs,
    ):
        super().__init__()

        self.encoder = PrithviViT(
            img_size=img_size,
            num_frames=num_frames,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
            drop_path=drop_path,
        )

        self.decoder = MAEDecoder(
            patch_size=patch_size,
            grid_size=self.encoder.patch_embed.grid_size,
            in_chans=in_chans,
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
        )

        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.out_channels = self.encoder.out_channels

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (torch.FloatTensor of shape `(batch_size, num_channels, time, height, width)`):
                Pixel values.

        Returns:
            torch.FloatTensor of shape
                `(batch_size, num_patches, patch_size[0]*patch_size[1]*patch_size[2] * num_channels)`:
                Patchified pixel values.
        """
        patch_size_t, patch_size_h, patch_size_w = self.encoder.patch_embed.patch_size
        num_channels = self.encoder.in_chans

        # patchify
        patchified_pixel_values = rearrange(
            pixel_values,
            "b c (t s) (h p) (w q) -> b (t h w) (s p q c)",
            c=num_channels,
            s=patch_size_t,
            p=patch_size_h,
            q=patch_size_w,
        )

        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values, image_size: tuple[int, int] | None = None):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape
                `(batch_size, num_patches, patch_size[0]*patch_size[1]*patch_size[2] * num_channels))`:
                Patchified pixel values.
            image_size (`tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        batch_size = patchified_pixel_values.shape[0]
        patch_size_t, patch_size_h, patch_size_w = self.encoder.patch_embed.patch_size
        image_size = to_2tuple(image_size) if image_size is not None else self.encoder.img_size
        original_height, original_width = image_size
        num_patches_h = original_height // patch_size_h
        num_patches_w = original_width // patch_size_w

        pixel_values = rearrange(
            patchified_pixel_values,
            "b (t h w) (s p q c) -> b c (t s) (h p) (w q)",
            b=batch_size,
            h=num_patches_h,
            w=num_patches_w,
            s=patch_size_t,
            p=patch_size_h,
            q=patch_size_w,
        )
        return pixel_values

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad_(False)

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, time, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape
                `(batch_size, num_patches, patch_size[0]*patch_size[1]*patch_size[2] * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        pixel_values: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
        mask_ratio: float | None = None,
    ):

        if len(pixel_values.shape) == 4 and self.encoder.patch_embed.input_size[0] == 1:
            # add time dim
            pixel_values = pixel_values.unsqueeze(2)
            time_dim_added = True
        else:
            time_dim_added = False

        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        if mask_ratio == 0.0:
            raise ValueError("mask_ratio=0.0 results in no masked patches and an undefined reconstruction loss. Use a mask_ratio > 0.0 for training.")
        latent, mask, ids_restore = self.encoder(pixel_values, temporal_coords, location_coords, mask_ratio)
        pred = self.decoder(latent, ids_restore, temporal_coords, location_coords, input_size=pixel_values.shape)
        loss = self.forward_loss(pixel_values, pred, mask)

        # Prepare output format in TerraTorch
        loss = {"loss": loss}
        mask = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])
        mask = self.unpatchify(mask, image_size=pixel_values.shape[-2:])
        mask = mask[:, 0]  # Remove channel dim
        pred = self.unpatchify(pred, image_size=pixel_values.shape[-2:])
        if time_dim_added:
            # Remove time dim to match input data
            pred = pred.squeeze(2)
            mask = mask.squeeze(1)
        return loss, pred, mask

    def forward_features(
        self,
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
    ) -> list[torch.Tensor]:
        return self.encoder.forward_features(x, temporal_coords, location_coords)
