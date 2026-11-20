# Copyright contributors to the Terratorch project
# Licensed under the Apache License 2.0

"""Galileo backbone integration for TerraTorch.

Galileo (NASA Harvest) is a multimodal, multi-temporal Earth observation encoder. Its bands are
grouped into four families:

    space-time (B, H, W, T, 13)  Sentinel-1 VV/VH, 11 Sentinel-2 bands, NDVI
    space      (B, H, W, 16)     SRTM elevation/slope, Dynamic World, WorldCover
    time       (B, T, 6)         ERA5, TerraClimate, VIIRS
    static     (B, 18)           LandScan, location, static Dynamic World/WorldCover

The encoder is vendored in ``single_file_galileo.py`` (MIT). Weights are loaded from the
Hugging Face Hub (``nasaharvest/galileo``).

References:
    - GitHub: https://github.com/nasaharvest/galileo
    - Paper: https://arxiv.org/abs/2502.09356
"""

import logging
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn

from terratorch.models.backbones.galileo.single_file_galileo import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    Encoder,
)
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

logger = logging.getLogger(__name__)

HF_REPO_ID = "nasaharvest/galileo"
# Encoder configs of the released checkpoints (models/<size>/config.json on the Hub).
GALILEO_CONFIGS = {
    "nano": {"embedding_size": 128, "depth": 4, "num_heads": 8},
    "tiny": {"embedding_size": 192, "depth": 12, "num_heads": 3},
    "base": {"embedding_size": 768, "depth": 12, "num_heads": 12},
}
COMMON_CONFIG = {"mlp_ratio": 4, "max_sequence_length": 24, "max_patch_size": 8, "drop_path": 0.1}


class GalileoBackbone(nn.Module):
    """Galileo encoder returning TerraTorch feature maps (B, C, H', W').

    Inputs:
        - a tensor (B, C, H, W) or a time series (B, C, T, H, W). Without ``bands``, it holds the
          13 space-time bands in Galileo's order (``SPACE_TIME_BANDS``);
        - a tensor whose channels are named by ``bands``: space-time bands are reordered into
          Galileo's layout, space bands (e.g. ``elevation``) go to the space family, averaged over
          time. Bands Galileo has no slot for (e.g. Sentinel-2 B1, B9) are ignored. Band groups with
          no given band are masked, i.e. treated as absent; missing bands of a partly given group
          are zero;
        - a dict with ``space_time`` and optionally ``space`` (B, H, W, 16), ``time`` (B, T, 6),
          ``static`` (B, 18) and ``months``, in Galileo's native layout.

    Families that are not given are zero-filled and masked, as in Galileo.

    Args:
        encoder: Galileo ``Encoder``.
        patch_size: Patch size for tokenization. Patch sizes other than the base one (8) are handled by
            resizing the input, FlexiViT-style.
        bands: Names of the input channels, see above. Galileo band names, case-insensitive.
        temporal_pooling: "mean" averages the time steps (default). "concat" stacks them into
            channels, giving (B, n_timestamps * D, H', W').
        n_timestamps: Number of time steps, required for "concat".
        out_indices: Encoder blocks whose outputs are returned, for multi-scale decoders.
            Defaults to the final encoder output only.
    """

    def __init__(
        self,
        encoder: Encoder,
        patch_size: int = 4,
        bands: list[str] | None = None,
        temporal_pooling: str = "mean",
        n_timestamps: int = 1,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        if temporal_pooling not in ("mean", "concat"):
            msg = f"temporal_pooling must be 'mean' or 'concat', got {temporal_pooling!r}"
            raise ValueError(msg)
        self.encoder = encoder
        self.patch_size = patch_size
        self.embed_dim = encoder.embedding_size
        self.temporal_pooling = temporal_pooling
        self.n_timestamps = n_timestamps

        self.band_split = None
        if bands is not None:
            names = [str(getattr(b, "value", b)).lower() for b in bands]
            space_time = [b.lower() for b in SPACE_TIME_BANDS]
            space = [b.lower() for b in SPACE_BANDS]
            # (target index, source channel) per family
            self.band_split = {
                "space_time": [(space_time.index(n), i) for i, n in enumerate(names) if n in space_time],
                "space": [(space.index(n), i) for i, n in enumerate(names) if n in space],
            }
            ignored = [b for b, n in zip(bands, names, strict=True) if n not in space_time + space]
            if ignored:
                warnings.warn(f"Galileo has no slot for bands {ignored}, they are ignored.", stacklevel=2)
            # groups with no given band are masked, i.e. treated as absent
            self.masked_groups = {
                family: [
                    g for g, idx in enumerate(groups.values()) if not {t for t, _ in self.band_split[family]} & set(idx)
                ]
                for family, groups in (("space_time", SPACE_TIME_BANDS_GROUPS_IDX), ("space", SPACE_BAND_GROUPS_IDX))
            }

        self.out_indices = None
        if out_indices:
            depth = len(encoder.blocks)
            self.out_indices = [i % depth for i in out_indices]
            self._captured: dict[int, Tensor] = {}
            for i in set(self.out_indices):
                encoder.blocks[i].register_forward_hook(self._capture_hook(i))

    def _capture_hook(self, idx: int) -> Callable:
        def hook(_module, _inputs, output):
            self._captured[idx] = output[0] if isinstance(output, tuple) else output

        return hook

    @property
    def out_channels(self) -> list[int]:
        channels = self.embed_dim * (self.n_timestamps if self.temporal_pooling == "concat" else 1)
        return [channels] * (len(self.out_indices) if self.out_indices else 1)

    def forward(self, x: Tensor | dict[str, Tensor], months: Tensor | None = None) -> list[Tensor]:
        """
        Args:
            x: Tensor (B, C, H, W) or (B, C, T, H, W), or a dict of Galileo families (see class docs).
            months: Optional (B, T) zero-indexed acquisition months. Defaults to 0.

        Returns:
            List of feature maps (B, out_channels, H / patch_size, W / patch_size), one per output level.
        """
        space = time = static = None
        if isinstance(x, dict):
            months = x.get("months", months)
            space, time, static = x.get("space"), x.get("time"), x.get("static")
            x = x["space_time"]
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, C, H, W) -> (B, C, T=1, H, W)

        B, _, T, H, W = x.shape
        device, dtype = x.device, x.dtype
        from_bands = False
        s_t_m = torch.zeros(B, H, W, T, len(SPACE_TIME_BANDS_GROUPS_IDX), dtype=torch.long, device=device)
        if self.band_split is not None:
            s_t_x = x.new_zeros(B, len(SPACE_TIME_BANDS), T, H, W)
            for target, source in self.band_split["space_time"]:
                s_t_x[:, target] = x[:, source]
            s_t_m[..., self.masked_groups["space_time"]] = 1
            from_bands = bool(self.band_split["space"]) and space is None
            if from_bands:
                space = x.new_zeros(B, H, W, len(SPACE_BANDS))
                for target, source in self.band_split["space"]:
                    space[..., target] = x[:, source].mean(dim=1)  # the space family is time-invariant
            x = s_t_x
        elif x.shape[1] != len(SPACE_TIME_BANDS):
            msg = f"Galileo expects {len(SPACE_TIME_BANDS)} space-time bands {SPACE_TIME_BANDS}, got {x.shape[1]}"
            raise ValueError(msg)

        def family(data, shape, n_groups):
            """Given families are observed (mask 0); absent ones are zero-filled and masked (mask 1)."""
            if data is None:
                return torch.zeros(*shape, dtype=dtype, device=device), torch.ones(
                    *shape[:-1], n_groups, dtype=torch.long, device=device
                )
            return data.to(device=device, dtype=dtype), torch.zeros(
                *shape[:-1], n_groups, dtype=torch.long, device=device
            )

        sp_x, sp_m = family(space, (B, H, W, len(SPACE_BANDS)), len(SPACE_BAND_GROUPS_IDX))
        if self.band_split is not None and from_bands:
            sp_m[..., self.masked_groups["space"]] = 1
        t_x, t_m = family(time, (B, T, len(TIME_BANDS)), len(TIME_BAND_GROUPS_IDX))
        st_x, st_m = family(static, (B, len(STATIC_BANDS)), len(STATIC_BAND_GROUPS_IDX))
        if months is None:
            months = torch.zeros(B, T, dtype=torch.long, device=device)

        out = self.encoder(
            x.permute(0, 3, 4, 2, 1), sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months, patch_size=self.patch_size
        )
        # out[0]: space-time tokens (B, H', W', T, band_groups, D)
        if not self.out_indices:
            return [self._to_feature_map(out[0])]

        # Block outputs are the pruned token sequence: the encoder drops masked tokens with a stable
        # sort before the blocks and restores them after the last one. Undo the same sort to put each
        # block's tokens back on the grid, then keep the space-time part. out[4:8] are the token masks.
        grid = out[0].shape[1:5]
        n_space_time = grid.numel()
        masks = torch.cat([m.flatten(1) for m in out[4:8]], dim=1)
        order = torch.sort((masks == 0).int(), dim=1, descending=True, stable=True)[1]
        captured = [self._captured[i] for i in self.out_indices]
        self._captured = {}
        return [
            self._to_feature_map(
                torch.zeros(B, masks.shape[1], self.embed_dim, dtype=block.dtype, device=device)
                .scatter(1, order[:, : block.shape[1], None].expand_as(block), block)[:, :n_space_time]
                .reshape(B, *grid, self.embed_dim)
            )
            for block in captured
        ]

    def _to_feature_map(self, tokens: Tensor) -> Tensor:
        """(B, H', W', T, band_groups, D) -> (B, out_channels, H', W')."""
        tokens = tokens.mean(dim=4)  # average band groups -> (B, H', W', T, D)
        if self.temporal_pooling == "mean":
            tokens = tokens.mean(dim=3)
        else:
            if tokens.shape[3] != self.n_timestamps:
                msg = f"expected {self.n_timestamps} time steps, got {tokens.shape[3]}"
                raise ValueError(msg)
            tokens = tokens.flatten(start_dim=3)  # (B, H', W', T * D)
        return tokens.permute(0, 3, 1, 2)


def _create_galileo_backbone(size: str, pretrained: bool = True, **kwargs) -> GalileoBackbone:
    if pretrained:
        from huggingface_hub import hf_hub_download

        files = [hf_hub_download(HF_REPO_ID, f"models/{size}/{f}") for f in ("config.json", "encoder.pt")]
        encoder = Encoder.load_from_folder(Path(files[0]).parent, device=torch.device("cpu"))
    else:
        encoder = Encoder(**COMMON_CONFIG, **GALILEO_CONFIGS[size])
    wrapper_args = ("patch_size", "bands", "temporal_pooling", "n_timestamps", "out_indices")
    return GalileoBackbone(encoder, **{k: v for k, v in kwargs.items() if k in wrapper_args})


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_nano(pretrained: bool = True, **kwargs) -> GalileoBackbone:
    """Galileo Nano (embedding size 128, depth 4)."""
    return _create_galileo_backbone("nano", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_tiny(pretrained: bool = True, **kwargs) -> GalileoBackbone:
    """Galileo Tiny (embedding size 192, depth 12)."""
    return _create_galileo_backbone("tiny", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def galileo_base(pretrained: bool = True, **kwargs) -> GalileoBackbone:
    """Galileo Base (embedding size 768, depth 12)."""
    return _create_galileo_backbone("base", pretrained=pretrained, **kwargs)
