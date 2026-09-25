# Copyright contributors to the Terratorch project
#
# Modified encoder-only port of ExPLoRA's grouped-channel SatMAE implementation
# (https://github.com/samar-khanna/ExPLoRA, commit
# 5b8cdcb704eead1b4cfe7ba1d6c870fb58ec8afd). ExPLoRA is Apache-2.0 and
# ships no NOTICE file.
"""Native static SatMAE backbones; weights remain separate CC-BY-4.0 artifacts."""

import hashlib
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import torch
from timm.layers import PatchEmbed
from timm.models.vision_transformer import Block
from torch import Tensor, nn

from terratorch.datasets import HLSBands
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

SATMAE_BANDS = (
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.RED_EDGE_1,
    HLSBands.RED_EDGE_2,
    HLSBands.RED_EDGE_3,
    HLSBands.NIR_BROAD,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
)
SATMAE_CHANNEL_GROUPS = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))
CHANNEL_EMBED_DIM = 256
INPUT_SHAPE = (10, 96, 96)
ARCHITECTURES = {"base": (768, 12, 12), "large": (1024, 24, 16)}
WEIGHTS = {
    "base": ("pretrain-vit-base-e199.pth", "1269e6a8255cee0affdeb6bb75776c86"),
    "large": ("pretrain-vit-large-e199.pth", "436bcd815194afdae38f3e023c74e9cd"),
}


def _sincos_1d(dim: int, positions: Tensor) -> Tensor:
    if dim % 2:
        msg = f"Sin/cos embedding dimension must be even, got {dim}."
        raise ValueError(msg)
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2))
    values = positions.reshape(-1, 1).float() * omega.reshape(1, -1)
    return torch.cat((values.sin(), values.cos()), dim=1)


def _sincos_2d(dim: int) -> Tensor:
    y, x = torch.meshgrid(torch.arange(12), torch.arange(12), indexing="ij")
    return torch.cat((torch.zeros(1, dim), torch.cat((_sincos_1d(dim // 2, x), _sincos_1d(dim // 2, y)), 1))).unsqueeze(
        0
    )


class GroupedTokenToFeatureMap(nn.Module):
    def forward(self, tokens: Tensor) -> Tensor:
        batch, token_count, width = tokens.shape
        per_group = (token_count - 1) // 3
        side = int(per_group**0.5)
        if token_count - 1 != 3 * per_group or side * side != per_group:
            msg = f"Invalid SatMAE token layout {tuple(tokens.shape)}."
            raise ValueError(msg)
        return tokens[:, 1:].reshape(batch, 3, side, side, width).mean(1).permute(0, 3, 1, 2).contiguous()


class SatMAEEncoder(nn.Module):
    """Static grouped-channel SatMAE encoder compatible with official weights.

    Inputs must already be native-radiometry Sentinel-2 data in the fixed ten-band
    SatMAE order. Preprocessing intentionally stays in the data pipeline.
    """

    def __init__(
        self,
        *,
        variant="base",
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_indices=None,
        bands: Sequence[HLSBands | str] | None = None,
        pretrained=False,
        ckpt_path: str | Path | None = None,
    ):
        super().__init__()
        if variant not in WEIGHTS:
            msg = f"Unsupported SatMAE variant {variant!r}."
            raise ValueError(msg)
        self._validate_bands(bands)
        if embed_dim <= CHANNEL_EMBED_DIM or (embed_dim - CHANNEL_EMBED_DIM) % 2:
            msg = f"embed_dim must exceed {CHANNEL_EMBED_DIM} and leave an even spatial embedding dimension."
            raise ValueError(msg)
        self.variant, self.embed_dim, self.depth, self.num_heads = variant, embed_dim, depth, num_heads
        self.out_indices = list(range(depth)) if out_indices is None else list(out_indices)
        if not self.out_indices or any(index < -depth or index >= depth for index in self.out_indices):
            msg = f"Invalid out_indices {self.out_indices} for depth {depth}."
            raise ValueError(msg)
        self.out_channels = [embed_dim] * len(self.out_indices)
        self.patch_embed = nn.ModuleList([PatchEmbed(96, 8, len(group), embed_dim) for group in SATMAE_CHANNEL_GROUPS])
        self.pos_embed = nn.Parameter(_sincos_2d(embed_dim - CHANNEL_EMBED_DIM), requires_grad=False)
        self.channel_embed = nn.Parameter(
            _sincos_1d(CHANNEL_EMBED_DIM, torch.arange(3)).unsqueeze(0), requires_grad=False
        )
        # Official SatMAE MAE checkpoints contain this encoder parameter.
        self.channel_cls_embed = nn.Parameter(torch.zeros(1, 1, CHANNEL_EMBED_DIM))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
        )
        self.norm = norm_layer(embed_dim)
        indices = (2, 5, 8, 11) if variant == "base" else (5, 11, 17, 23)
        self.feature_taps = nn.ModuleDict({str(i): GroupedTokenToFeatureMap() for i in indices if i < depth})
        for patch_embed in self.patch_embed:
            nn.init.xavier_uniform_(patch_embed.proj.weight.flatten(1))
            if patch_embed.proj.bias is not None:
                nn.init.zeros_(patch_embed.proj.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)
        elif pretrained:
            self.load_checkpoint(self._download_weights())

    @staticmethod
    def _validate_bands(bands):
        if bands is None:
            return
        try:
            converted = tuple(HLSBands(b.value if isinstance(b, HLSBands) else str(b)) for b in bands)
        except ValueError as error:
            msg = "SatMAE received an unsupported band."
            raise ValueError(msg) from error
        if converted != SATMAE_BANDS:
            msg = f"SatMAE requires exact band order {[b.value for b in SATMAE_BANDS]}."
            raise ValueError(msg)

    @staticmethod
    def _md5(path: Path) -> str:
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _download_weights(self) -> Path:
        filename, expected_md5 = WEIGHTS[self.variant]
        path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and self._md5(path) == expected_md5:
            return path
        if path.exists():
            path.unlink()
        partial_path = path.with_suffix(path.suffix + ".part")
        url = f"https://zenodo.org/api/records/7338613/files/{filename}/content"
        try:
            torch.hub.download_url_to_file(url, str(partial_path), progress=True)
            if self._md5(partial_path) != expected_md5:
                msg = "MD5 verification failed"
                raise ValueError(msg)
            partial_path.replace(path)
        except Exception as error:
            partial_path.unlink(missing_ok=True)
            msg = f"Unable to obtain official SatMAE {self.variant} weights; pass ckpt_path for offline use."
            raise RuntimeError(msg) from error
        return path

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        path = Path(ckpt_path)
        if not path.is_file():
            msg = f"SatMAE checkpoint does not exist: {path}"
            raise FileNotFoundError(msg)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            msg = "SatMAE checkpoint must be a state dict or contain a 'model' state dict."
            raise ValueError(msg)
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        expected = self.state_dict()
        unexpected = sorted(
            key for key in state if key not in expected and not key.startswith(("decoder_", "mask_token"))
        )
        if unexpected:
            msg = f"Unexpected non-decoder SatMAE checkpoint keys: {unexpected[:5]}"
            raise ValueError(msg)
        encoder = {key: value for key, value in state.items() if key in expected}
        missing = sorted(key for key in expected if key not in encoder)
        mismatched = sorted(
            key for key, value in encoder.items() if not isinstance(value, Tensor) or value.shape != expected[key].shape
        )
        if missing or mismatched:
            msg = f"Incompatible SatMAE encoder checkpoint: missing={missing[:5]}, mismatched={mismatched[:5]}"
            raise ValueError(msg)
        self.load_state_dict(encoder, strict=True)

    def _tokens(self, x: Tensor) -> Tensor:
        if x.ndim != len(INPUT_SHAPE) + 1 or x.shape[1:] != INPUT_SHAPE:
            msg = f"SatMAE expects native (B, 10, 96, 96) input, got {tuple(x.shape)}."
            raise ValueError(msg)
        pieces = [embed(x[:, group]) for embed, group in zip(self.patch_embed, SATMAE_CHANNEL_GROUPS, strict=True)]
        tokens = torch.stack(pieces, 1)
        batch, groups, patches, width = tokens.shape
        channel = self.channel_embed.unsqueeze(2).expand(-1, -1, patches, -1)
        position = self.pos_embed[:, 1:].unsqueeze(1).expand(-1, groups, -1, -1)
        tokens = (tokens + torch.cat((position, channel), -1)).reshape(batch, groups * patches, width)
        cls = torch.cat((self.pos_embed[:, :1], self.channel_cls_embed), -1)
        return torch.cat((cls + self.cls_token.expand(batch, -1, -1), tokens), 1)

    def forward_features(self, x: Tensor) -> list[Tensor]:
        tokens, features = self._tokens(x), []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if str(index) in self.feature_taps:
                # Taps are parameter-free modules intended for external hooks.
                self.feature_taps[str(index)](tokens)
            features.append(tokens)
        features[-1] = self.norm(features[-1])
        return features

    def forward(self, x: Tensor) -> list[Tensor]:
        features = self.forward_features(x)
        return [features[index] for index in self.out_indices]

    def prepare_features_for_image_model(self, features: list[Tensor]) -> list[Tensor]:
        return [GroupedTokenToFeatureMap()(feature) for feature in features]


def _build(variant: str, **kwargs) -> SatMAEEncoder:
    embed_dim, depth, num_heads = ARCHITECTURES[variant]
    return SatMAEEncoder(
        variant=variant,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        **kwargs,
    )


@TERRATORCH_BACKBONE_REGISTRY.register
def satmae_base(**kwargs) -> SatMAEEncoder:
    return _build("base", **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def satmae_large(**kwargs) -> SatMAEEncoder:
    return _build("large", **kwargs)
