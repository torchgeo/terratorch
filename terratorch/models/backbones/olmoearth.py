# Copyright contributors to the Terratorch project
# Licensed under the Apache License 2.0

"""OlmoEarth backbone integration for TerraTorch.

This module registers OlmoEarth (Allen AI) foundation models as backbones
in the TerraTorch backbone registry. OlmoEarth is a spatio-temporal, multimodal
foundation model for Earth observations supporting Sentinel-1, Sentinel-2, and Landsat.

Requires the `olmoearth-pretrain` package:
    pip install terratorch[olmoearth]

References:
    - GitHub: https://github.com/allenai/olmoearth_pretrain
    - Paper: https://arxiv.org/abs/2511.13655
"""

import logging
from collections.abc import Callable

import torch
from torch import Tensor, nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

logger = logging.getLogger(__name__)


# OlmoEarth model configurations
# Maps terratorch model names -> (ModelID enum value, encoder embedding size)
OLMOEARTH_CONFIGS: dict[str, dict] = {
    "olmoearth_v1_nano": {
        "model_id": "OlmoEarth-v1-Nano",
        "embed_dim": 128,
    },
    "olmoearth_v1_tiny": {
        "model_id": "OlmoEarth-v1-Tiny",
        "embed_dim": 128,
    },
    "olmoearth_v1_base": {
        "model_id": "OlmoEarth-v1-Base",
        "embed_dim": 768,
    },
    "olmoearth_v1_large": {
        "model_id": "OlmoEarth-v1-Large",
        "embed_dim": 1024,
    },
    "olmoearth_v1_1_nano": {
        "model_id": "OlmoEarth-v1_1-Nano",
        "embed_dim": 128,
    },
    "olmoearth_v1_1_tiny": {
        "model_id": "OlmoEarth-v1_1-Tiny",
        "embed_dim": 192,
    },
    "olmoearth_v1_1_base": {
        "model_id": "OlmoEarth-v1_1-Base",
        "embed_dim": 768,
    },
    "olmoearth_v1_2_nano": {
        "model_id": "OlmoEarth-v1_2-Nano",
        "embed_dim": 128,
    },
    "olmoearth_v1_2_tiny": {
        "model_id": "OlmoEarth-v1_2-Tiny",
        "embed_dim": 192,
    },
    "olmoearth_v1_2_small": {
        "model_id": "OlmoEarth-v1_2-Small",
        "embed_dim": 384,
    },
    "olmoearth_v1_2_base": {
        "model_id": "OlmoEarth-v1_2-Base",
        "embed_dim": 768,
    },
}


class OlmoEarthBackbone(nn.Module):
    """Wrapper around OlmoEarth encoder for use as a TerraTorch backbone.

    This wrapper loads an OlmoEarth model (encoder only) and adapts its
    forward pass to produce a list of feature maps compatible with TerraTorch
    decoders. The encoder outputs per-patch token embeddings which are reshaped
    into (B, D, H', W') spatial feature maps.

    OlmoEarth's native interface expects a MaskedOlmoEarthSample input with
    multi-modal data in [B, H, W, T, C] format. This wrapper provides a
    simplified interface that accepts a standard image tensor (B, C, H, W) and
    wraps it as Sentinel-2 L2A input.
    """

    def __init__(
        self,
        model_id: str,
        embed_dim: int,
        pretrained: bool = True,
        patch_size: int = 8,
        input_res: int = 10,
        modality: str = "sentinel2_l2a",
        ckpt_path: str | None = None,
    ):
        """Initialize OlmoEarth backbone.

        Args:
            model_id: OlmoEarth model identifier (e.g. "OlmoEarth-v1-Base")
            embed_dim: Embedding dimension of the encoder
            pretrained: Whether to load pretrained weights from HuggingFace
            patch_size: Patch size for tokenization (default 8)
            input_res: Input resolution in meters (default 10m for Sentinel-2)
            modality: Input modality name matching MaskedOlmoEarthSample fields
                (default "sentinel2_l2a")
            ckpt_path: Optional path to a local checkpoint file
        """
        super().__init__()

        from olmoearth_pretrain.data.constants import Modality as ModalitySpec
        from olmoearth_pretrain.model_loader import ModelID, load_model_from_id, load_model_from_path

        self.patch_size = patch_size
        self.input_res = input_res
        self.modality = modality

        # Load the full model (encoder + decoder)
        if ckpt_path is not None:
            full_model = load_model_from_path(ckpt_path, load_weights=pretrained)
        else:
            model_id_enum = ModelID(model_id)
            full_model = load_model_from_id(model_id_enum, load_weights=pretrained)

        # Extract only the encoder
        self.encoder = full_model.encoder

        # Use the encoder's actual embedding size as the authoritative embed_dim
        self.embed_dim = self.encoder.embedding_size

        # Determine number of band sets for this modality (needed for mask shape)
        modality_spec = ModalitySpec.get(self.modality)
        self.num_band_sets = len(modality_spec.band_sets)

        # Expose embedding dimension for downstream components
        self.num_features = self.embed_dim

    @property
    def out_channels(self) -> list[int]:
        """Return the output channel dimensions for each feature level.

        OlmoEarth encoder produces a single feature level.
        """
        return [self.embed_dim]

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass through the OlmoEarth encoder.

        Takes a standard (B, C, H, W) image tensor and returns a list of
        spatial feature maps [(B, D, H', W')] suitable for TerraTorch decoders.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List containing a single feature map of shape (B, embed_dim, H', W')
            where H' = H // patch_size and W' = W // patch_size.
        """
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

        B, C, H, W = x.shape

        # Compute spatial grid dimensions after patchification
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # OlmoEarth expects raw pixel data in [B, H, W, T, C] format.
        # Reshape from (B, C, H, W) -> (B, H, W, T=1, C)
        x_bhwtc = x.permute(0, 2, 3, 1).unsqueeze(3)  # (B, H, W, 1, C)

        # Create mask at pixel resolution: [B, H, W, T, num_band_sets]
        # All tokens visible (ONLINE_ENCODER)
        mask = torch.full(
            (B, H, W, 1, self.num_band_sets),
            fill_value=MaskValue.ONLINE_ENCODER.value,
            device=x.device,
            dtype=torch.long,
        )

        # Construct timestamps: [B, T, 3] with [day, month, year] as long
        timestamps = torch.zeros(B, 1, 3, device=x.device, dtype=torch.long)

        # Build the MaskedOlmoEarthSample with the correct field names
        sample_kwargs = {
            "timestamps": timestamps,
            self.modality: x_bhwtc,
            f"{self.modality}_mask": mask,
        }
        sample = MaskedOlmoEarthSample(**sample_kwargs)

        # Run encoder
        encoder_output = self.encoder(
            sample,
            patch_size=self.patch_size,
            input_res=self.input_res,
        )

        # Extract spatial token embeddings from tokens_and_masks.
        # Shape: (B, H_patches, W_patches, T, band_sets, D)
        tokens_and_masks = encoder_output["tokens_and_masks"]
        modality_tokens = getattr(tokens_and_masks, self.modality)

        # Average over T and band_sets dims to get (B, H', W', D)
        features = modality_tokens.mean(dim=(3, 4))

        # Reshape to (B, D, H', W') for compatibility with TerraTorch decoders
        features = features.permute(0, 3, 1, 2)

        return [features]


def _create_olmoearth_backbone(
    variant: str,
    pretrained: bool = True,
    patch_size: int = 8,
    input_res: int = 10,
    modality: str = "sentinel2_l2a",
    ckpt_path: str | None = None,
    **kwargs,
) -> OlmoEarthBackbone:
    """Create an OlmoEarth backbone model.

    Args:
        variant: Model variant name (e.g. "olmoearth_v1_base")
        pretrained: Whether to load pretrained weights
        patch_size: Patch size for tokenization
        input_res: Input resolution in meters
        modality: Input modality name
        ckpt_path: Optional path to local checkpoint
        **kwargs: Additional keyword arguments (unused)

    Returns:
        OlmoEarthBackbone instance
    """
    if variant not in OLMOEARTH_CONFIGS:
        msg = f"Unknown OlmoEarth variant '{variant}'. Available: {list(OLMOEARTH_CONFIGS.keys())}"
        raise ValueError(msg)

    cfg = OLMOEARTH_CONFIGS[variant]

    model = OlmoEarthBackbone(
        model_id=cfg["model_id"],
        embed_dim=cfg["embed_dim"],
        pretrained=pretrained,
        patch_size=patch_size,
        input_res=input_res,
        modality=modality,
        ckpt_path=ckpt_path,
    )

    return model


# Register all OlmoEarth variants with the TerraTorch backbone registry


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_nano(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Nano (1.4M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_nano", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_tiny(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Tiny (6.2M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_tiny", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_base(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Base (89M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_base", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_large(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Large (308M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_large", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_nano(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.1 Nano (5.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_1_nano", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_tiny(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.1 Tiny (12.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_1_tiny", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_base(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.1 Base (114M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_1_base", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_nano(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Nano (5.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_nano", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_tiny(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Tiny (12.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_tiny", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_small(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Small (35.6M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_small", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_base(pretrained: bool = True, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Base (114M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_base", pretrained=pretrained, **kwargs)
