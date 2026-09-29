import logging
import math
from collections.abc import Callable
from pathlib import Path

import torch
from einops import repeat
from torch import nn
from box import Box

import terratorch.models.decoders as decoder_registry
from terratorch.models.backbones.clay_v15.model import Encoder
from terratorch.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
)
from terratorch.models.pixel_wise_model import PixelWiseModel
from terratorch.models.scalar_output_model import ScalarOutputModel
from terratorch.models.utils import DecoderNotFoundError, extract_prefix_keys
from terratorch.registry import MODEL_FACTORY_REGISTRY

PIXEL_WISE_TASKS = ["segmentation", "regression"]
SCALAR_TASKS = ["classification", "scalar_regression"]
SUPPORTED_TASKS = PIXEL_WISE_TASKS + SCALAR_TASKS

# ClayMAE checkpoint keys that collide with the terratorch decoder_ prefix convention.
# These are ignored when building the terratorch decoder kwargs.
_CLAY_DECODER_KEYS = {"dim", "depth", "heads", "dim_head", "mlp_ratio", "ratio"}


def _strip_clay_keys(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k not in _CLAY_DECODER_KEYS}


logger = logging.getLogger("terratorch")


class ClayMAEBackbone(nn.Module):
    """Wraps the Clay v1.5 Encoder as a feature extractor for downstream tasks.

    Runs the encoder without masking to produce full spatial feature maps,
    compatible with Terratorch's PixelWiseModel / ScalarOutputModel pipelines.
    """

    def __init__(self, encoder: nn.Module, platform: str | list[str], metadata: Box):
        super().__init__()
        self.encoder_module = encoder
        self.platform = platform[0] if isinstance(platform, list) else platform
        self.metadata = metadata
        self.patch_size = encoder.patch_size
        self.dim = encoder.dim

    def forward(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        B, C, H, W = x.shape

        waves = torch.tensor(
            list(self.metadata[self.platform].bands.wavelength.values()),
            device=x.device,
        )
        gsd = torch.tensor(self.metadata[self.platform].gsd, device=x.device)
        time = torch.zeros(B, 4, device=x.device)
        latlon = torch.zeros(B, 4, device=x.device)

        # Patch embedding + positional encoding (no masking for downstream tasks)
        patches, _ = self.encoder_module.to_patch_embed(x, waves)
        patches = self.encoder_module.add_encodings(patches, time, latlon, gsd)

        # Prepend CLS token and pass through transformer
        cls_tokens = repeat(self.encoder_module.cls_token, "1 1 D -> B 1 D", B=B)
        patches = torch.cat((cls_tokens, patches), dim=1)
        encoded = self.encoder_module.transformer(patches)  # [B, 1+L, D]

        # Strip CLS token and reshape to spatial [B, D, H', W']
        features = encoded[:, 1:, :]  # [B, L, D]
        grid_size = int(math.sqrt(features.shape[1]))
        features = features.transpose(1, 2).reshape(B, self.dim, grid_size, grid_size)

        return [features]


@MODEL_FACTORY_REGISTRY.register
class Clay1_5ModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        in_channels: int,
        bands: list[int] = [],
        num_classes: int | None = None,
        pretrained: bool = True,  # noqa: FBT001, FBT002
        num_frames: int = 1,
        prepare_features_for_image_model: Callable | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,  # noqa: FBT002, FBT001
        checkpoint_path: str | None = None,
        **kwargs,
    ) -> Model:
        """Model factory for Clay v1.5 downstream tasks.

        Builds a full encoder-decoder model using the Clay v1.5 encoder as backbone,
        wrapped in Terratorch's standard PixelWiseModel or ScalarOutputModel pipeline.

        Required kwargs:
            metadata (dict): Platform metadata containing band wavelengths and GSD.
            platform (str | list[str]): Platform name matching a key in metadata.
            dim (int): Encoder embedding dimension.
            depth (int): Number of transformer layers.
            heads (int): Number of attention heads.
            dim_head (int): Dimension per attention head.
            mlp_ratio (float): MLP hidden dim multiplier.
            patch_size (int): Patch size for the convolutional embedding.

        Optional kwargs:
            mask_ratio (float): Unused for inference; kept for compatibility. Default 0.75.
            shuffle (bool): Unused for inference; kept for compatibility. Default False.
            padding (str): Padding mode for PixelWiseModel. Default "reflect".
            batch_size (int): Ignored; accepted for backward compatibility.
            decoder_* (kwargs): Passed to the terratorch decoder constructor.
                Note: Clay v1.5 internal keys (decoder_dim, decoder_depth, decoder_heads,
                decoder_dim_head, decoder_mlp_ratio) are silently ignored.

        Args:
            task: One of "segmentation", "regression", "classification", "scalar_regression".
            backbone: Ignored; kept for API compatibility with other factories.
            decoder: Decoder class name (e.g. "FCNDecoder") or an nn.Module instance.
            in_channels: Number of input channels.
            bands: Band names; informational only.
            num_classes: Number of output classes. Required for segmentation/classification.
            pretrained: If True, loads weights from checkpoint_path or downloads from HuggingFace.
            num_frames: Ignored; kept for API compatibility.
            aux_decoders: Optional list of auxiliary decoder heads.
            rescale: Rescale model output to input spatial size via bilinear interpolation.
            checkpoint_path: Path to a local Clay v1.5 checkpoint (.ckpt or .pt).
        """
        task = task.lower()
        if task not in SUPPORTED_TASKS:
            msg = f"Task {task} not supported. Please choose one of {SUPPORTED_TASKS}"
            raise NotImplementedError(msg)

        platform = kwargs.get("platform")
        if platform is None:
            raise ValueError("'platform' is required in kwargs for Clay1_5ModelFactory")
        metadata_raw = kwargs.get("metadata")
        if metadata_raw is None:
            raise ValueError("'metadata' is required in kwargs for Clay1_5ModelFactory")
        metadata = Box(metadata_raw)
        platform_str = platform[0] if isinstance(platform, list) else platform
        if platform_str not in metadata:
            raise KeyError(f"Platform '{platform_str}' not found in metadata. Available: {list(metadata.keys())}")

        patch_size = kwargs.get("patch_size", 8)
        padding = kwargs.get("padding", "reflect")

        encoder = Encoder(
            mask_ratio=kwargs.get("mask_ratio", 0.75),
            patch_size=patch_size,
            shuffle=kwargs.get("shuffle", False),
            dim=kwargs["dim"],
            depth=kwargs["depth"],
            heads=kwargs["heads"],
            dim_head=kwargs["dim_head"],
            mlp_ratio=kwargs["mlp_ratio"],
        )

        if pretrained:
            ckpt_path = _resolve_checkpoint(checkpoint_path)
            if ckpt_path:
                _load_encoder_weights(encoder, ckpt_path)

        backbone_module = ClayMAEBackbone(encoder, platform, metadata)
        feature_channels = [backbone_module.dim]

        decoder_cls = _get_decoder(decoder)
        decoder_kwargs, _ = extract_prefix_keys(kwargs, "decoder_")
        decoder_instance = decoder_cls(feature_channels, **_strip_clay_keys(decoder_kwargs))

        head_kwargs, _ = extract_prefix_keys(kwargs, "head_")
        if num_classes:
            head_kwargs["num_classes"] = num_classes

        if aux_decoders is None:
            return _build_appropriate_model(
                task, backbone_module, decoder_instance, head_kwargs,
                patch_size=patch_size, padding=padding, rescale=rescale,
            )

        to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_cls = _get_decoder(aux_decoder.decoder)
            aux_decoder_kwargs, _ = extract_prefix_keys(args, "decoder_")
            aux_decoder_instance = aux_decoder_cls(feature_channels, **_strip_clay_keys(aux_decoder_kwargs))
            aux_head_kwargs, _ = extract_prefix_keys(args, "head_")
            if num_classes:
                aux_head_kwargs["num_classes"] = num_classes
            to_be_aux_decoders.append(
                AuxiliaryHeadWithDecoderWithoutInstantiatedHead(
                    aux_decoder.name, aux_decoder_instance, aux_head_kwargs
                )
            )

        return _build_appropriate_model(
            task, backbone_module, decoder_instance, head_kwargs,
            patch_size=patch_size, padding=padding, rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _resolve_checkpoint(checkpoint_path: str | None) -> str | None:
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return str(path)
    # No explicit path provided — try HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading Clay v1.5 checkpoint from HuggingFace...")
        return hf_hub_download(repo_id="made-with-clay/Clay", filename="v1.5/clay-v1.5.ckpt")
    except Exception as e:
        logger.warning(f"Could not load pretrained weights: {e}")
        return None


def _load_encoder_weights(encoder: nn.Module, ckpt_path: str) -> None:
    logger.info(f"Loading encoder weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Clay v1.5 checkpoints store the full ClayMAE under "model.encoder.*"
    prefix = "model.encoder."
    encoder_state = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    if not encoder_state:
        logger.warning(
            f"No keys with prefix '{prefix}' found in checkpoint. "
            "The encoder weights were NOT loaded. Keys found: "
            f"{list(state_dict.keys())[:10]} ..."
        )
        return
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        logger.warning(f"Missing encoder keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected encoder keys: {unexpected}")
    if not missing:
        logger.info("Encoder weights loaded successfully")


def _build_appropriate_model(
    task: str,
    backbone: nn.Module,
    decoder: nn.Module,
    head_kwargs: dict,
    patch_size: int | None,
    padding: str,
    rescale: bool = True,
    auxiliary_heads=None,
):
    if task in PIXEL_WISE_TASKS:
        return PixelWiseModel(
            task, backbone, decoder, head_kwargs,
            patch_size=patch_size, padding=padding, rescale=rescale,
            auxiliary_heads=auxiliary_heads,
        )
    if task in SCALAR_TASKS:
        # rescale is not passed: ScalarOutputModel does not do spatial output
        return ScalarOutputModel(
            task, backbone, decoder, head_kwargs,
            patch_size=patch_size, padding=padding,
            auxiliary_heads=auxiliary_heads,
        )
    raise AssertionError(f"Unreachable: unexpected task '{task}'")


def _get_decoder(decoder: str | nn.Module) -> nn.Module:
    if isinstance(decoder, nn.Module):
        return decoder
    if isinstance(decoder, str):
        try:
            return getattr(decoder_registry, decoder)
        except AttributeError as e:
            msg = f"Decoder {decoder} was not found in the registry."
            raise DecoderNotFoundError(msg) from e
    msg = "Decoder must be str or nn.Module"
    raise Exception(msg)
