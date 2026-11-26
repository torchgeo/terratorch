# Copyright contributors to the Terratorch project
# Licensed under the Apache License 2.0

"""OlmoEarth backbone integration for TerraTorch.

This module registers OlmoEarth (Allen AI) foundation models as backbones
in the TerraTorch backbone registry. OlmoEarth is a spatio-temporal, multimodal
foundation model for Earth observations supporting Sentinel-1, Sentinel-2, and Landsat.

Requires the `olmoearth-pretrain` package:
    pip install terratorch[olmoearth]
or the lighter `olmoearth-pretrain-minimal` (the loader without training/evaluation dependencies), which is
used if installed. Both ship the same model code.

References:
    - GitHub: https://github.com/allenai/olmoearth_pretrain
    - GitHub (minimal loader): https://github.com/allenai/olmoearth_pretrain_minimal
    - Paper: https://arxiv.org/abs/2511.13655
"""

import logging
import warnings
from types import SimpleNamespace

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
        "embed_dim": 192,
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
    "olmoearth_v1_3_base": {  # olmoearth-pretrain-minimal >= 0.1.5
        "model_id": "OlmoEarth-v1_3-Base",
        "embed_dim": 768,
    },
}


def _olmoearth_api() -> SimpleNamespace:
    """OlmoEarth loader and data types, from ``olmoearth-pretrain-minimal`` or else the full ``olmoearth-pretrain``.

    Both packages ship the same model code under different module paths.
    """
    try:
        from olmoearth_pretrain_minimal import ModelID, load_model_from_id, load_model_from_path
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample, MaskValue
    except ImportError:
        from olmoearth_pretrain.data.normalize import load_computed_config
        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
        from olmoearth_pretrain.model_loader import ModelID, load_model_from_id, load_model_from_path
    return SimpleNamespace(
        ModelID=ModelID,
        load_model_from_id=load_model_from_id,
        load_model_from_path=load_model_from_path,
        Modality=Modality,
        MaskedOlmoEarthSample=MaskedOlmoEarthSample,
        MaskValue=MaskValue,
        load_computed_config=load_computed_config,
    )


def olmoearth_normalization(
    bands: dict[str, list[str]], std_multiplier: float = 2.0
) -> tuple[list[float], list[float]]:
    """Means and stds for TerraTorch datamodules that reproduce OlmoEarth's pre-training normalization.

    OlmoEarth maps raw values (Sentinel-2 L2A digital numbers, Sentinel-1 in dB) from ``mean ± 2·std`` to [0, 1],
    with the statistics shipped in the OlmoEarth package. ``(x - (mean - 2·std)) / (4·std)`` is the same, so pass
    the returned values as the datamodule's ``means`` and ``stds``.

    Args:
        bands: The backbone's ``bands``, ``{modality: [band, ...]}`` in input channel order.
        std_multiplier: Width of the normalized range in standard deviations, 2 in pre-training.

    Returns:
        ``(means, stds)``, one value per input channel.
    """
    stats = _olmoearth_api().load_computed_config()
    means, stds = [], []
    for name, names in bands.items():
        for band in names:
            band_stats = {k.lower(): v for k, v in stats[name].items()}[band.lower()]
            means.append(band_stats["mean"] - std_multiplier * band_stats["std"])
            stds.append(2 * std_multiplier * band_stats["std"])
    return means, stds


class OlmoEarthBackbone(nn.Module):
    """Wrapper around OlmoEarth encoder for use as a TerraTorch backbone.

    This wrapper loads an OlmoEarth model (encoder only) and adapts its
    forward pass to produce a list of feature maps compatible with TerraTorch
    decoders. The encoder outputs per-patch token embeddings which are reshaped
    into (B, D, H', W') spatial feature maps.

    OlmoEarth's native interface expects a MaskedOlmoEarthSample input with
    multi-modal data in [B, H, W, T, C] format. This wrapper accepts:

    - a tensor (B, C, H, W), or a time series (B, C, T, H, W), carrying ``modality``
      (default Sentinel-2 L2A), with bands in OlmoEarth's order for that modality;
    - a tensor whose channels are described by ``bands``, e.g. a Sentinel-2 + Sentinel-1
      stack. It is split into OlmoEarth's native modalities and each is reordered;
    - a dict ``{modality: tensor}``, optionally with a ``"timestamps"`` entry.

    All modalities are encoded jointly. The output averages the band sets of each modality, then
    the modalities (or concatenates them, ``modality_pooling="concat"``). Non-multitemporal
    modalities (e.g. ``srtm``) are averaged over time.
    """

    def __init__(
        self,
        model_id: str,
        embed_dim: int,
        pretrained: bool = False,
        patch_size: int = 8,
        input_res: int = 10,
        modality: str = "sentinel2_l2a",
        ckpt_path: str | None = None,
        bands: dict[str, list[str]] | None = None,
        temporal_pooling: str = "mean",
        n_timestamps: int = 1,
        fast_pass: bool = True,
        modality_pooling: str = "mean",
    ):
        """Initialize OlmoEarth backbone.

        Args:
            model_id: OlmoEarth model identifier (e.g. "OlmoEarth-v1-Base")
            embed_dim: Embedding dimension of the encoder
            pretrained: Whether to load pretrained weights from HuggingFace (default False)
            patch_size: Patch size for tokenization (default 8)
            input_res: Input resolution in meters (default 10m for Sentinel-2)
            modality: Input modality name matching MaskedOlmoEarthSample fields
                (default "sentinel2_l2a")
            ckpt_path: Optional path to a local checkpoint file
            bands: Channel layout of a stacked input tensor, modality by modality in channel
                order, e.g. ``{"sentinel2_l2a": ["B01", ..., "B12"], "sentinel1": ["vv", "vh"]}``.
                Any subset of a modality's bands, in any order: band sets without any given band are masked
                as missing, missing bands of a partly given band set are zero.
            temporal_pooling: "mean" averages the time steps (default). "concat" stacks them into
                channels, giving (B, n_timestamps * D, H', W') for a temporal neck or decoder.
            n_timestamps: Number of time steps, required for "concat".
            fast_pass: Pass no attention mask in training (default True), so SDPA can use its memory-efficient
                kernels; with a mask it expands to (B, heads, N, N), which runs out of memory at small patch
                sizes. Exact here, since this wrapper never masks tokens; in eval the encoder skips the mask anyway.
                With missing band sets (see ``bands``), the masked tokens are removed instead and the attention
                mask is skipped as long as it masks nothing, which keeps the memory benefit.
            modality_pooling: "mean" averages the modalities (default). "concat" stacks them into channels in
                ``bands`` order, giving (B, n_modalities * C, H', W'); needs ``bands``.
        """
        super().__init__()

        oe = _olmoearth_api()
        ModelID, load_model_from_id, load_model_from_path = oe.ModelID, oe.load_model_from_id, oe.load_model_from_path
        ModalitySpec = self._modality_spec = oe.Modality  # also used by _to_feature_map

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

        for name, value in (("temporal_pooling", temporal_pooling), ("modality_pooling", modality_pooling)):
            if value not in ("mean", "concat"):
                msg = f"{name} must be 'mean' or 'concat', got {value!r}"
                raise ValueError(msg)
        if modality_pooling == "concat" and not bands:
            msg = "modality_pooling='concat' needs `bands` to know the modalities"
            raise ValueError(msg)
        self.modality_pooling = modality_pooling
        self.temporal_pooling = temporal_pooling
        self.n_timestamps = n_timestamps
        self.fast_pass = fast_pass

        # bands -> {modality: input channel per band in OlmoEarth's band order, -1 if not given} and
        # {modality: band sets without any given band, masked as missing}
        self.band_split, self.missing_band_sets = {}, {}
        offset = 0
        for name, names in (bands or {}).items():
            spec = ModalitySpec.get(name)
            given = [b.lower() for b in names]
            unknown = [b for b in names if b.lower() not in [o.lower() for o in spec.band_order]]
            if unknown or len(set(given)) != len(given):
                msg = f"bands for {name} must be distinct bands of {spec.band_order}, got {names}"
                raise ValueError(msg)
            self.band_split[name] = [
                offset + given.index(b.lower()) if b.lower() in given else -1 for b in spec.band_order
            ]
            offset += len(names)
            band_sets = [[b.lower() for b in band_set.bands] for band_set in spec.band_sets]
            self.missing_band_sets[name] = [i for i, bs in enumerate(band_sets) if not set(bs) & set(given)]
            zeros = [b for bs in band_sets if set(bs) & set(given) for b in bs if b not in given]
            if zeros:
                warnings.warn(f"OlmoEarth {name}: bands {zeros} are not given and set to zero.", stacklevel=2)
        if any(len(m) == len(ModalitySpec.get(n).band_sets) for n, m in self.missing_band_sets.items()):
            msg = f"every modality in `bands` needs at least one band, got {bands}"
            raise ValueError(msg)
        self.has_missing = any(self.missing_band_sets.values())
        if self.has_missing:
            # Missing band sets are the same for every sample, so after removing their tokens the attention mask
            # is all True: skip it, as fast_pass does, so SDPA keeps its memory-efficient kernels.
            get_attn_mask = self.encoder._maybe_get_attn_mask

            def attn_mask_unless_vacuous(new_mask, fast_pass):
                return (
                    None if new_mask is None or bool(new_mask.all()) else get_attn_mask(new_mask, fast_pass=fast_pass)
                )

            self.encoder._maybe_get_attn_mask = attn_mask_unless_vacuous

        # Return every block's output, like TerraMind and Prithvi (pick with SelectIndices); the last is the
        # encoder's final, normed output. The encoder regroups its flat token sequence per modality with a dims
        # dict local to its forward, and without fast_pass removes masked tokens; record both so the captured
        # blocks are treated the same way.
        self._captured: list[Tensor] = []
        for block in self.encoder.blocks:
            block.register_forward_hook(
                lambda _module, _inputs, out: self._captured.append(out[0] if isinstance(out, tuple) else out)
            )
        regroup = self.encoder.split_and_expand_per_modality
        self._regroup = regroup

        def recording_regroup(tokens, modalities_to_dims_dict, *args, **kwargs):
            self._dims = modalities_to_dims_dict
            return regroup(tokens, modalities_to_dims_dict, *args, **kwargs)

        self.encoder.split_and_expand_per_modality = recording_regroup

        remove = self.encoder._maybe_remove_masked_tokens
        self._removal = (None, None)

        def recording_remove(tokens, mask, fast_pass):
            out = remove(tokens, mask, fast_pass)
            self._removal = (out[1], out[2])  # indices, new_mask
            return out

        self.encoder._maybe_remove_masked_tokens = recording_remove

    @property
    def out_channels(self) -> list[int]:
        """Return the output channel dimensions for each feature level."""
        channels = self.embed_dim * (self.n_timestamps if self.temporal_pooling == "concat" else 1)
        channels *= len(self.band_split) if self.modality_pooling == "concat" else 1
        return [channels] * len(self.encoder.blocks)

    def forward(
        self, x: Tensor | dict[str, Tensor], timestamps: Tensor | None = None, **kwargs: object
    ) -> list[Tensor]:
        """Forward pass through the OlmoEarth encoder.

        Args:
            x: Tensor (B, C, H, W) or (B, C, T, H, W), or a dict of such tensors keyed by
                OlmoEarth modality name, optionally with a "timestamps" entry.
            timestamps: Optional (B, T, 3) acquisition dates as [day, month, year], months
                zero-indexed. Defaults to zeros.
            **kwargs: Other batch entries passed on by TerraTorch tasks, ignored.

        Returns:
            List of feature maps of shape (B, out_channels, H', W'), one per output level,
            where H' = H // patch_size and W' = W // patch_size.
        """
        oe = _olmoearth_api()
        ModalitySpec, MaskedOlmoEarthSample, MaskValue = oe.Modality, oe.MaskedOlmoEarthSample, oe.MaskValue

        if isinstance(x, dict):
            inputs = {k: v for k, v in x.items() if k != "timestamps"}
            timestamps = x.get("timestamps", timestamps)
        elif self.band_split:
            inputs = {}
            for name, idx in self.band_split.items():
                if -1 in idx:  # bands that are not given are zero
                    data = x.new_zeros(x.shape[0], len(idx), *x.shape[2:])
                    given = [i for i, j in enumerate(idx) if j >= 0]
                    data[:, given] = x[:, [idx[i] for i in given]]
                    inputs[name] = data
                else:
                    inputs[name] = x[:, idx]
        else:
            inputs = {self.modality: x}

        # OlmoEarth expects raw pixel data in [B, H, W, T, C] format, with a mask of
        # [B, H, W, T, num_band_sets]. All tokens visible (ONLINE_ENCODER), except missing band sets.
        sample_kwargs, steps = {}, 1
        for name, data in inputs.items():
            if data.dim() == 4:
                data = data.unsqueeze(2)  # (B, C, H, W) -> (B, C, T=1, H, W)
            spec = ModalitySpec.get(name)
            if spec.is_multitemporal:
                steps = data.shape[2]
            else:
                data = data.mean(dim=2, keepdim=True)
            B, _, T, H, W = data.shape
            sample_kwargs[name] = data.permute(0, 3, 4, 2, 1)
            sample_kwargs[f"{name}_mask"] = torch.full(
                (B, H, W, T, len(spec.band_sets)),
                fill_value=MaskValue.ONLINE_ENCODER.value,
                device=data.device,
                dtype=torch.long,
            )
            sample_kwargs[f"{name}_mask"][..., self.missing_band_sets.get(name, [])] = MaskValue.MISSING.value

        # Construct timestamps: [B, T, 3] with [day, month, year] as long
        if timestamps is None:
            timestamps = torch.zeros(B, steps, 3, device=data.device, dtype=torch.long)
        sample = MaskedOlmoEarthSample(timestamps=timestamps, **sample_kwargs)

        encoder_output = self.encoder(
            sample,
            patch_size=self.patch_size,
            input_res=self.input_res,
            fast_pass=self.fast_pass and not self.has_missing,  # missing band sets need token removal
        )

        captured, self._captured = self._captured[:-1], []  # the last block is replaced by the final output
        indices, new_mask = self._removal
        if indices is not None:
            captured = [self.encoder.add_removed_tokens(tokens, indices, new_mask)[0] for tokens in captured]
        maps = [self._to_feature_map(self._regroup(tokens, self._dims), inputs) for tokens in captured]
        return maps + [self._to_feature_map(encoder_output["tokens_and_masks"], inputs)]

    def _to_feature_map(self, tokens_and_masks, inputs: dict[str, Tensor]) -> Tensor:
        """(B, H', W', T, band_sets, D) per modality -> (B, out_channels, H', W')."""
        if self.modality_pooling == "concat" and list(inputs) != list(self.band_split):
            msg = f"modality_pooling='concat' expects the modalities {list(self.band_split)}, got {list(inputs)}"
            raise ValueError(msg)
        maps = []
        for name in inputs:
            tokens = tokens_and_masks[name] if isinstance(tokens_and_masks, dict) else getattr(tokens_and_masks, name)
            missing = self.missing_band_sets.get(name)
            if missing:  # average only the given band sets
                tokens = tokens[..., [i for i in range(tokens.shape[4]) if i not in missing], :]
            tokens = tokens.mean(dim=4)  # average band sets -> (B, H', W', T, D)
            if self.temporal_pooling == "mean":
                tokens = tokens.mean(dim=3)
            else:
                if tokens.shape[3] == 1 and not self._modality_spec.get(name).is_multitemporal:  # e.g. srtm
                    tokens = tokens.expand(-1, -1, -1, self.n_timestamps, -1)
                if tokens.shape[3] != self.n_timestamps:
                    msg = f"{name}: expected {self.n_timestamps} time steps, got {tokens.shape[3]}"
                    raise ValueError(msg)
                tokens = tokens.flatten(start_dim=3)  # (B, H', W', T * D)
            maps.append(tokens)
        if len(maps) == 1:
            features = maps[0]
        elif self.modality_pooling == "concat":
            features = torch.cat(maps, dim=-1)  # (B, H', W', n_modalities * C)
        else:
            features = torch.stack(maps).mean(dim=0)
        # Reshape to (B, D, H', W') for compatibility with TerraTorch decoders
        return features.permute(0, 3, 1, 2)


def _create_olmoearth_backbone(
    variant: str,
    pretrained: bool = False,
    patch_size: int = 8,
    input_res: int = 10,
    modality: str = "sentinel2_l2a",
    ckpt_path: str | None = None,
    bands: dict[str, list[str]] | None = None,
    temporal_pooling: str = "mean",
    n_timestamps: int = 1,
    fast_pass: bool = True,
    modality_pooling: str = "mean",
    **kwargs,
) -> OlmoEarthBackbone:
    """Create an OlmoEarth backbone model.

    Args:
        variant: Model variant name (e.g. "olmoearth_v1_base")
        pretrained: Whether to load pretrained weights (default False)
        **kwargs: Additional keyword arguments (unused)
        Other arguments: see ``OlmoEarthBackbone``.

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
        bands=bands,
        temporal_pooling=temporal_pooling,
        n_timestamps=n_timestamps,
        fast_pass=fast_pass,
        modality_pooling=modality_pooling,
    )

    return model


# Register all OlmoEarth variants with the TerraTorch backbone registry


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_nano(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Nano (1.4M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_nano", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_tiny(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Tiny (6.2M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_tiny", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_base(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Base (89M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_base", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_large(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1 Large (308M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_large", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_nano(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.1 Nano (5.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_1_nano", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_tiny(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.1 Tiny (12.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_1_tiny", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_1_base(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.1 Base (114M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_1_base", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_nano(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Nano (5.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_nano", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_tiny(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Tiny (12.5M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_tiny", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_small(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Small (35.6M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_small", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_2_base(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.2 Base (114M encoder params)."""
    return _create_olmoearth_backbone("olmoearth_v1_2_base", pretrained=pretrained, **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def olmoearth_v1_3_base(pretrained: bool = False, **kwargs) -> OlmoEarthBackbone:
    """OlmoEarth v1.3 Base (needs olmoearth-pretrain-minimal >= 0.1.5)."""
    return _create_olmoearth_backbone("olmoearth_v1_3_base", pretrained=pretrained, **kwargs)
