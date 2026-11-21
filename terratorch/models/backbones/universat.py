# Copyright contributors to the Terratorch project

"""UniverSat backbone.

UniverSat (https://github.com/gastruc/UniverSat, weights at https://huggingface.co/g-astruc/UniverSat) is a
modality- and resolution-agnostic Earth observation encoder. One forward pass fuses any set of sensors (optical,
multispectral, hyperspectral, SAR, elevation; single images or time series; centimetre to hundreds of metres per
pixel) covering the same tile, and returns tokens on an output grid chosen at inference time. Sensors are
described by metadata (wavelengths or band codes, ground sampling distance, sub-patch factor), read from UniverSat's
modality registry or given explicitly for unseen sensors.

This module wraps UniverSat's ``encode`` API. The model is loaded with ``torch.hub`` from the official repository
(pinned to a tested commit by default), or from a local checkout of it. UniverSat's modality registry is read from
the loaded code, so use one UniverSat version per process.

Reference: Y. Perron, G. Astruc, N. Gonthier, C. Mallet, L. Landrieu. "UniverSat: Resolution- and Modality-Agnostic
Transformers for Earth Observation". Advances in Neural Information Processing Systems (NeurIPS), 2026.
"""

import logging
import math
import re
import sys
import warnings

import torch
from torch import nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

logger = logging.getLogger(__name__)

# Tested commit of the official repository. torch.hub only validates branch and tag names, so pinned commits skip
# that validation (see `_load_universat`).
UNIVERSAT_HUB_REPO = "gastruc/UniverSat:4f55ee696ae384f7a07ae93a016f78e326f7f5eb"
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
UNIVERSAT_HF_REPO_ID = "g-astruc/UniverSat"
UNIVERSAT_EMBED_DIM = 768  # released Base model
UNIVERSAT_REGISTRY_MODULE = "modality_registry"  # imported by UniverSat's hubconf.py
DATES_SUFFIX = "_dates"
IMAGE_NDIM = 4  # (B, C, H, W)
SERIES_NDIM = 5  # (B, C, T, H, W), TerraTorch temporal layout
TOKEN_NDIM = 3  # (B, N, D)
TILE_EXTENT_RTOL = 0.01  # tolerance when checking that all modalities cover the same tile
# Ratio bands (e.g. "Ratio_VV_VH" for Sentinel-1, "Ratio_HH_HV" for ALOS), computed as in UniverSat's data loaders.
RATIO_PREFIX = "Ratio_"
RATIO_EPS = 1e-10
RATIO_CLIP = 1e4
STD_MIN = 1e-6  # floor on standard deviations, as in UniverSat's normalization
OUTPUT_RES_RTOL = 0.05  # warn when output_res does not divide the tile extent
# Arguments set by the wrapper, which encode_kwargs must not repeat.
WRAPPER_ENCODE_ARGS = {"patch_size", "latent_grid", "output_grid", "wavelengths", "input_res", "subpatches"}
# State dict prefixes stripped by `_load_checkpoint`: TerraTorch task, wrapper, Lightning training, bare encoder.
CHECKPOINT_PREFIXES = ("model.encoder.model.", "encoder.model.", "model.model.", "model.encoder.", "encoder.", "model.")

# TerraTorch modality names mapped to UniverSat modality names. Other names are passed through unchanged.
MODALITY_ALIASES = {
    "S2L2A": "s2",
    "S2L1C": "s2l1c",
    "S1GRD": "s1",
    "S1RTC": "s1",
}
# Band selections for common TerraTorch layouts of aliased modalities, keyed by (alias, number of channels).
# 12-band S2L2A (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12) -> UniverSat s2 (no B01, B09).
ALIAS_BAND_SELECTION = {
    ("S2L2A", 12): [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
}


def _universat_registry() -> dict[str, dict] | None:
    """Return UniverSat's modality metadata (wavelengths, input_res, subpatches), if its registry is loaded."""
    registry = sys.modules.get(UNIVERSAT_REGISTRY_MODULE)
    if registry is None or not hasattr(registry, "WAVELENGTHS"):
        return None
    return {
        "wavelengths": registry.WAVELENGTHS,
        "input_res": getattr(registry, "INPUT_RES", {}),
        "subpatches": getattr(registry, "SUBPATCHES", {}),
    }


class UniverSatEncoderWrapper(nn.Module):
    """Encoder wrapper returning fused UniverSat tokens for any combination of modalities.

    Inputs:
        - a tensor, encoded as ``modality``;
        - or a dict ``{modality: tensor}`` with one or more modalities covering the same tile, encoded jointly.
          Modalities may have different ground sampling distances, so different pixel sizes.

    Each tensor is a single image ``(B, C, H, W)`` or a time series ``(B, C, T, H, W)`` (TerraTorch layout), with
    square ``H = W``. Time series need acquisition dates ``(B, T)`` in the convention of UniverSat (day of year),
    given as ``"<modality>_dates"`` entries in the input dict or with the ``dates`` argument. A single image with
    dates is encoded as a one-date time series.

    Modality names are UniverSat registry names (``s2``, ``s1``, ``s2l1c``, ``l7``, ``l8``, ``hls``, ``alos``,
    ``spot``, ``naip``, ``aerial``, ``enmap``, ``dem``, ...) or TerraTorch aliases. Sensors outside the registry are
    supported by giving their ``wavelengths``, ``input_res`` and, if needed, ``subpatches``. The number of channels
    must match the modality's band list (``wavelengths``), in the same order.

    TerraTorch aliases: ``S2L2A`` takes UniverSat's 10 ``s2`` bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11,
    B12), or TerraTorch's 12-band layout (B01 to B12 without B10), from which B01 and B09 are dropped. ``S2L1C`` takes
    13 bands. ``S1GRD`` and ``S1RTC`` take VV and VH, with the VV/VH ratio band added (see below), or all three.
    For other band sets, pass ``wavelengths`` for the modality in your band order.

    Bands that UniverSat expects as ratios of other bands (``Ratio_VV_VH`` for ``s1``, ``Ratio_HH_HV`` for ``alos``)
    are computed from the raw input if missing, e.g. for Sentinel-1 with VV and VH only, with the same clipping as
    UniverSat's data loaders. Inputs must then be in the representation the ratio is meant for (as in UniverSat's
    training data), and ``mean`` / ``std`` must include the ratio band.

    The output is a one-element list with tokens ``(B, G * G, D)`` in row-major order over a ``G x G`` grid
    covering the tile, without class or register tokens. Use the ``ReshapeTokensToImage`` neck with
    ``remove_cls_token: false`` to obtain ``(B, D, G, G)``. The grid side ``G`` is
    ``output_grid`` if given, else the tile extent divided by ``output_res`` if given, else dense: one token per
    sub-patch, i.e. the latent grid side times the largest sub-patch factor of the inputs. With the default 40 m
    patches, this is 40 m for Sentinel-2 (sub-patch factor 1) and 4 m for aerial images (factor 10). Use
    ``output_res`` for one token per pixel, which is costly for very high resolution inputs.

    Only the final tokens are returned, as a single feature level: decoders that need several levels (e.g. UPerNet)
    are not recommended with UniverSat.

    Use with TerraTorch tasks:
        - Time series are fused natively: pass ``(B, C, T, H, W)`` tensors with ``"<modality>_dates"`` in the input
          dict, and do not set ``backbone_use_temporal``, which encodes each date separately and drops the dates.
        - With modalities of different pixel sizes, TerraTorch's pixel-wise models size their output on the largest
          input. Set ``image_size_out`` to the label grid.
        - ``modis`` (one token per tile) is in UniverSat's registry but not supported by ``encode`` at the tested
          UniverSat version.

    Args:
        model: UniverSat model exposing ``encode``.
        modality: Modality name used for tensor inputs.
        patch_size_meters: Latent patch size in meters.
        latent_grid: Number of latent tokens (a perfect square). Inferred from the inputs if None.
        output_grid: Output grid side ``G``.
        output_res: Output resolution in meters per token, e.g. 10 for one token per Sentinel-2 pixel. Used if
            ``output_grid`` is None. Requires the input resolution of the modalities.
        wavelengths: Per-modality wavelengths in micrometers, or band codes for radar and elevation (e.g.
            ``{"s1": ["VV", "VH"]}``). Override the registry, and are required for unseen sensors.
        input_res: Per-modality ground sampling distance in meters.
        subpatches: Per-modality number of sub-patches per latent patch side.
        mean: Optional per-modality band means. With ``std``, inputs are standardized in the backbone. UniverSat
            does not normalize its inputs: like UniverSat, use statistics of the dataset, either here or in the
            datamodule.
        std: Optional per-modality band standard deviations.
        add_ratio_bands: If True, compute missing ratio bands (see above).
        encode_kwargs: Extra keyword arguments for ``UniverSat.encode``, other than those set by the wrapper.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        modality: str = "s2",
        patch_size_meters: float = 40,
        latent_grid: int | None = None,
        output_grid: int | None = None,
        output_res: float | None = None,
        wavelengths: dict[str, list[float | str]] | None = None,
        input_res: dict[str, float] | None = None,
        subpatches: dict[str, int] | None = None,
        mean: dict[str, list[float]] | None = None,
        std: dict[str, list[float]] | None = None,
        add_ratio_bands: bool = True,
        encode_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if (mean is None) != (std is None):
            msg = "mean and std must be given together."
            raise ValueError(msg)
        if (mean or {}).keys() != (std or {}).keys():
            msg = "mean and std must be given for the same modalities."
            raise ValueError(msg)

        self.model = model
        self.modality = self._resolve(modality)
        self.patch_size_meters = patch_size_meters
        self.latent_grid = latent_grid
        self.output_grid = output_grid
        self.output_res = output_res
        self.overrides = {
            "wavelengths": {self._resolve(k): list(v) for k, v in (wavelengths or {}).items()},
            "input_res": {self._resolve(k): float(v) for k, v in (input_res or {}).items()},
            "subpatches": {self._resolve(k): int(v) for k, v in (subpatches or {}).items()},
        }
        self.add_ratio_bands = add_ratio_bands
        self.encode_kwargs = encode_kwargs or {}
        duplicates = sorted(WRAPPER_ENCODE_ARGS & set(self.encode_kwargs))
        if duplicates:
            msg = f"encode_kwargs repeats arguments set by the wrapper: {duplicates}. Use the wrapper arguments."
            raise ValueError(msg)
        self.out_channels = [getattr(model, "embed_dim", UNIVERSAT_EMBED_DIM)]

        self.standardized = set()
        for stat, values_per_modality in (("mean", mean), ("std", std)):
            for name, values in (values_per_modality or {}).items():
                stat_tensor = torch.tensor(values).float()[None, :, None, None]
                self.register_buffer(f"_{self._resolve(name)}_{stat}", stat_tensor, persistent=False)
                self.standardized.add(self._resolve(name))

    @staticmethod
    def _resolve(modality: str) -> str:
        return MODALITY_ALIASES.get(modality, modality)

    def _split_input(
        self, x: torch.Tensor | dict[str, torch.Tensor], dates: torch.Tensor | dict[str, torch.Tensor] | None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, list[int]]]:
        """Return ``{modality: image}``, ``{modality: dates}`` and alias band selections, with UniverSat names."""
        if isinstance(x, torch.Tensor):
            x = {self.modality: x}
            if isinstance(dates, torch.Tensor):
                dates = {self.modality: dates}
        images, all_dates, selections = {}, {}, {}
        for key, value in x.items():
            if key.endswith(DATES_SUFFIX):
                all_dates[self._resolve(key.removesuffix(DATES_SUFFIX))] = value
                continue
            name = self._resolve(key)
            selection = ALIAS_BAND_SELECTION.get((key, value.shape[1] if value.ndim > 1 else None))
            if selection is not None and name not in self.overrides["wavelengths"]:
                selections[name] = selection
            images[name] = value
        if isinstance(dates, torch.Tensor):
            msg = "Pass dates as a dict {modality: dates} for dict inputs."
            raise TypeError(msg)
        all_dates.update({self._resolve(k): v for k, v in (dates or {}).items()})
        if not images:
            msg = "UniverSatEncoderWrapper received no image modality."
            raise ValueError(msg)
        unmatched = sorted(set(all_dates) - set(images))
        if unmatched:
            msg = f"Dates given for modalities without images: {unmatched}."
            raise ValueError(msg)
        return images, all_dates, selections

    def _metadata(self, key: str, modality: str) -> float | int | list | None:
        if modality in self.overrides[key]:
            return self.overrides[key][modality]
        registry = _universat_registry()
        return None if registry is None else registry[key].get(modality)

    def _check_known(self, modalities: list[str]) -> None:
        # UniverSat ignores dict keys it does not know, so a misspelled modality would be dropped silently.
        registry = _universat_registry()
        if registry is None:
            return
        known = set(registry["wavelengths"]) | set(self.overrides["wavelengths"])
        unknown = sorted(set(modalities) - known)
        if unknown:
            msg = (
                f"Unknown UniverSat modalities {unknown}. Use a registry name ({sorted(registry['wavelengths'])}), "
                "a TerraTorch alias, or give the sensor's wavelengths, input_res and subpatches."
            )
            raise ValueError(msg)

    def _with_ratio_bands(self, modality: str, image: torch.Tensor) -> torch.Tensor:
        """Append the ratio bands of ``modality`` (``Ratio_<A>_<B>`` = A / B) if the input lacks them."""
        bands = self._metadata("wavelengths", modality)
        if not self.add_ratio_bands or not bands:
            return image
        ratios = [band for band in bands if isinstance(band, str) and band.startswith(RATIO_PREFIX)]
        base = [band for band in bands if band not in ratios]
        if not ratios or image.shape[1] != len(base):
            return image
        extra = []
        for ratio in ratios:
            numerator, denominator = ratio.removeprefix(RATIO_PREFIX).split("_", 1)
            value = image[:, base.index(numerator)] / (image[:, base.index(denominator)] + RATIO_EPS)
            extra.append(value.clamp(-RATIO_CLIP, RATIO_CLIP))
        return torch.cat([image, torch.stack(extra, dim=1)], dim=1)

    def _standardize(self, modality: str, image: torch.Tensor) -> torch.Tensor:
        mean, std = getattr(self, f"_{modality}_mean"), getattr(self, f"_{modality}_std")
        if mean.shape[1] != image.shape[1]:
            bands = self._metadata("wavelengths", modality)
            ratios = [band for band in bands or [] if isinstance(band, str) and band.startswith(RATIO_PREFIX)]
            hint = (
                f" The ratio bands {ratios} are added before standardization, so give their mean/std too."
                if ratios and self.add_ratio_bands
                else ""
            )
            msg = (
                f"{modality}: mean/std have {mean.shape[1]} values, but the input has {image.shape[1]} channels "
                f"(UniverSat bands {list(bands) if bands else 'unknown'}).{hint}"
            )
            raise ValueError(msg)
        if image.ndim == SERIES_NDIM:
            mean, std = mean[:, :, None], std[:, :, None]
        return (image - mean) / std.clamp_min(STD_MIN)

    def _check_channels(self, modality: str, image: torch.Tensor) -> None:
        bands = self._metadata("wavelengths", modality)
        if bands is not None and image.shape[1] != len(bands):
            msg = (
                f"{modality}: got {image.shape[1]} channels, but UniverSat describes {len(bands)} bands {list(bands)}. "
                f"Pass wavelengths={{'{modality}': [...]}} matching your band order."
            )
            raise ValueError(msg)

    def _output_grid(self, images: dict[str, torch.Tensor]) -> int | None:
        if self.output_grid is not None:
            return self.output_grid
        if self.output_res is None:
            return self._subpatch_grid(images)
        extents = {}
        for modality, image in images.items():
            res = self._metadata("input_res", modality)
            if res is None:
                msg = f"output_res needs the input resolution of {modality!r}; pass it in input_res."
                raise ValueError(msg)
            extents[modality] = image.shape[-1] * float(res)
        extent = next(iter(extents.values()))
        if any(not math.isclose(e, extent, rel_tol=TILE_EXTENT_RTOL) for e in extents.values()):
            msg = f"All modalities must cover the same tile, got extents in meters {extents}."
            raise ValueError(msg)
        side = max(round(extent / self.output_res), 1)
        if not math.isclose(extent / side, self.output_res, rel_tol=OUTPUT_RES_RTOL):
            warnings.warn(
                f"output_res={self.output_res} m does not divide the {extent} m tile: using {side} tokens of "
                f"{extent / side:.2f} m.",
                stacklevel=3,
            )
        return side

    def _subpatch_grid(self, images: dict[str, torch.Tensor]) -> int | None:
        """Grid side with one token per sub-patch, as computed by ``UniverSat.encode`` by default."""
        if self.latent_grid is not None:
            side = math.isqrt(self.latent_grid)
        else:
            name, image = next(iter(images.items()))
            res = self._metadata("input_res", name)
            if res is None:
                return None  # left to UniverSat
            side = max(image.shape[-1] // max(int(self.patch_size_meters / float(res)), 1), 1)
        factors = [self._metadata("subpatches", modality) or 1 for modality in images]
        return side * max(int(factor) for factor in factors)

    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        dates: torch.Tensor | dict[str, torch.Tensor] | None = None,
        **kwargs,  # noqa: ARG002
    ) -> list[torch.Tensor]:
        images, all_dates, selections = self._split_input(x, dates)
        self._check_known(list(images))
        batch = next(iter(images.values())).shape[0]

        inputs, encode_meta = {}, {key: {} for key in self.overrides}
        for modality, raw in images.items():
            image = raw
            if image.ndim not in {IMAGE_NDIM, SERIES_NDIM}:
                msg = f"{modality}: expected (B, C, H, W) or (B, C, T, H, W), got shape {tuple(image.shape)}."
                raise ValueError(msg)
            if image.shape[0] != batch:
                msg = f"{modality}: batch size {image.shape[0]} differs from {batch}."
                raise ValueError(msg)
            if image.shape[-2] != image.shape[-1]:
                msg = f"{modality}: UniverSat expects square inputs, got {tuple(image.shape[-2:])}."
                raise ValueError(msg)

            image = image.float()
            standardize = modality in self.standardized
            if modality in selections:
                # Statistics may be given for the full TerraTorch layout or for the selected bands.
                if standardize and getattr(self, f"_{modality}_mean").shape[1] == image.shape[1]:
                    image, standardize = self._standardize(modality, image), False
                image = image[:, selections[modality]]
            image = self._with_ratio_bands(modality, image)
            self._check_channels(modality, image)
            if standardize:
                image = self._standardize(modality, image)

            modality_dates = all_dates.get(modality)
            if image.ndim == SERIES_NDIM or modality_dates is not None:
                if image.ndim == IMAGE_NDIM:
                    image = image[:, :, None]  # one-date series
                steps = image.shape[2]
                if modality_dates is None or tuple(modality_dates.shape) != (batch, steps):
                    msg = f"{modality}: time series input needs dates of shape ({batch}, {steps})."
                    raise ValueError(msg)
                # UniverSat expects (B, T, C, H, W).
                inputs[modality] = image.permute(0, 2, 1, 3, 4)
                inputs[f"{modality}{DATES_SUFFIX}"] = modality_dates.to(image.device).long()
            else:
                inputs[modality] = image

            for key, values in self.overrides.items():
                if modality in values:
                    encode_meta[key][modality] = values[modality]

        output_grid = self._output_grid(images)
        tokens, _ = self.model.encode(
            inputs,
            patch_size=self.patch_size_meters,
            latent_grid=self.latent_grid,
            output_grid=output_grid,
            **{key: values for key, values in encode_meta.items() if values},
            **self.encode_kwargs,
        )
        if tokens.ndim != TOKEN_NDIM or tokens.shape[0] != batch:
            msg = f"UniverSat returned tokens of shape {tuple(tokens.shape)}, expected ({batch}, N, D)."
            raise ValueError(msg)
        if output_grid is not None and tokens.shape[1] != output_grid**2:
            msg = f"UniverSat returned {tokens.shape[1]} tokens, expected {output_grid**2}."
            raise ValueError(msg)
        return [tokens]


def _load_universat(
    hub_repo: str,
    *,
    pretrained: bool,
    source: str,
    trust_repo: bool | str,
    repo_id: str,
    compile_model: bool | str,
    hub_kwargs: dict,
) -> nn.Module:
    if pretrained:
        entrypoint, kwargs = "from_pretrained", {"repo_id": repo_id, **hub_kwargs}
    else:
        entrypoint, kwargs = "universat", {"pretrained": False, **hub_kwargs}
    # torch.hub validates GitHub refs against branch and tag names only, so a pinned commit must skip validation.
    ref = hub_repo.partition(":")[2]
    skip_validation = source == "github" and COMMIT_SHA.fullmatch(ref) is not None
    model = torch.hub.load(
        hub_repo,
        entrypoint,
        source=source,
        trust_repo=trust_repo,
        skip_validation=skip_validation,
        compile=compile_model,
        **kwargs,
    )
    if not hasattr(model, "encode"):
        msg = "The loaded UniverSat model does not expose `encode`."
        raise TypeError(msg)
    return model


def _load_checkpoint(model: nn.Module, ckpt_path: str, *, weights_only: bool) -> None:
    """Load UniverSat weights from a state dict or a checkpoint, whatever the key prefix."""
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=weights_only)
    state_dict = state_dict.get("state_dict", state_dict)
    target = set(model.state_dict())

    def candidate(prefix: str) -> dict[str, torch.Tensor]:
        stripped = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
        with_model = {f"model.{k}": v for k, v in stripped.items()}  # hub module keys start with "model."
        return max(stripped, with_model, key=lambda d: len(target & set(d)))

    best = max((candidate(prefix) for prefix in ("", *CHECKPOINT_PREFIXES)), key=lambda d: len(target & set(d)))
    matched = len(target & set(best))
    if matched == 0:
        msg = (
            f"No key of {ckpt_path} matches the UniverSat model. Expected a UniverSat state dict, possibly prefixed "
            f"with one of {list(CHECKPOINT_PREFIXES)}."
        )
        raise ValueError(msg)
    incompatible = model.load_state_dict(best, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(
            f"Loaded {matched}/{len(target)} UniverSat tensors from {ckpt_path}: {len(incompatible.missing_keys)} "
            f"missing (e.g. {incompatible.missing_keys[:3]}), {len(incompatible.unexpected_keys)} unexpected "
            f"(e.g. {incompatible.unexpected_keys[:3]})."
        )


@TERRATORCH_BACKBONE_REGISTRY.register
def universat(
    *,
    pretrained: bool = True,
    ckpt_path: str | None = None,
    ckpt_weights_only: bool = True,
    hub_repo: str = UNIVERSAT_HUB_REPO,
    hub_source: str = "github",
    trust_repo: bool | str = True,
    repo_id: str = UNIVERSAT_HF_REPO_ID,
    compile_model: bool | str = "fast",
    hub_kwargs: dict | None = None,
    modality: str = "s2",
    patch_size_meters: float = 40,
    latent_grid: int | None = None,
    output_grid: int | None = None,
    output_res: float | None = None,
    wavelengths: dict[str, list[float | str]] | None = None,
    input_res: dict[str, float] | None = None,
    subpatches: dict[str, int] | None = None,
    mean: dict[str, list[float]] | None = None,
    std: dict[str, list[float]] | None = None,
    add_ratio_bands: bool = True,
    encode_kwargs: dict | None = None,
    **kwargs,
) -> UniverSatEncoderWrapper:
    """Build a UniverSat Base encoder.

    Args:
        pretrained: Load the released weights from the Hugging Face Hub. If False, build the model with random
            weights (e.g. for training from scratch, or before loading ``ckpt_path``).
        ckpt_path: Optional local weights loaded last (e.g. fine-tuned): a UniverSat state dict, or a checkpoint with
            a ``state_dict`` entry (TerraTorch or Lightning), with any of the usual key prefixes.
        ckpt_weights_only: Passed to ``torch.load``. Lightning checkpoints with non-tensor content need False; only
            use False for files you trust.
        hub_repo: ``torch.hub`` repository, ``"owner/repo[:ref]"`` on GitHub, or a local checkout directory. The
            default pins the tested commit of the official repository; ``"gastruc/UniverSat"`` tracks its default
            branch, and ``ref`` can be a branch, a tag or a full commit hash.
        hub_source: ``"github"`` or ``"local"`` (for a local checkout in ``hub_repo``).
        trust_repo: Passed to ``torch.hub.load``. True (default) trusts ``hub_repo`` without asking; ``"check"`` asks
            interactively for repositories not yet trusted, which fails in batch jobs.
        repo_id: Hugging Face model ID or local directory with the released weights.
        compile_model: UniverSat compilation policy: ``"fast"`` (default, recommended for fine-tuning), ``"max"``
            (recommended for inference only, e.g. embedding generation), or False (eager).
        hub_kwargs: Extra arguments for the hub entry point, e.g. ``revision`` or ``cache_dir`` when pretrained,
            ``modalities_dict`` when not.
        **kwargs: Unused TerraTorch arguments, ignored. Other arguments: see ``UniverSatEncoderWrapper``.
    """
    if kwargs:
        logger.debug(f"Ignoring UniverSat arguments: {sorted(kwargs)}")
    model = _load_universat(
        hub_repo,
        pretrained=pretrained,
        source=hub_source,
        trust_repo=trust_repo,
        repo_id=repo_id,
        compile_model=compile_model,
        hub_kwargs=hub_kwargs or {},
    )
    if ckpt_path is not None:
        _load_checkpoint(model, ckpt_path, weights_only=ckpt_weights_only)
    return UniverSatEncoderWrapper(
        model,
        modality=modality,
        patch_size_meters=patch_size_meters,
        latent_grid=latent_grid,
        output_grid=output_grid,
        output_res=output_res,
        wavelengths=wavelengths,
        input_res=input_res,
        subpatches=subpatches,
        mean=mean,
        std=std,
        add_ratio_bands=add_ratio_bands,
        encode_kwargs=encode_kwargs,
    )
