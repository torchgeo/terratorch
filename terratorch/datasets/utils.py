# Copyright contributors to the Terratorch project

import os
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


class HLSBands(Enum):
    COASTAL_AEROSOL = "COASTAL_AEROSOL"
    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"
    RED_EDGE_1 = "RED_EDGE_1"
    RED_EDGE_2 = "RED_EDGE_2"
    RED_EDGE_3 = "RED_EDGE_3"
    NIR_BROAD = "NIR_BROAD"
    NIR_NARROW = "NIR_NARROW"
    SWIR_1 = "SWIR_1"
    SWIR_2 = "SWIR_2"
    WATER_VAPOR = "WATER_VAPOR"
    CIRRUS = "CIRRUS"
    THERMAL_INFRARED_1 = "THERMAL_INFRARED_1"
    THERMAL_INFRARED_2 = "THERMAL_INFRARED_2"

    @classmethod
    def try_convert_to_hls_bands_enum(cls, x: Any):
        try:
            return cls(x)
        except ValueError:
            return x


class OpticalBands(Enum):
    COASTAL_AEROSOL = "COASTAL_AEROSOL"
    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"
    RED_EDGE_1 = "RED_EDGE_1"
    RED_EDGE_2 = "RED_EDGE_2"
    RED_EDGE_3 = "RED_EDGE_3"
    NIR_BROAD = "NIR_BROAD"
    NIR_NARROW = "NIR_NARROW"
    SWIR_1 = "SWIR_1"
    SWIR_2 = "SWIR_2"
    WATER_VAPOR = "WATER_VAPOR"
    CIRRUS = "CIRRUS"
    THERMAL_INFRARED_1 = "THERMAL_INFRARED_1"
    THERMAL_INFRARED_2 = "THERMAL_INFRARED_2"

    @classmethod
    def try_convert_to_optical_bands_enum(cls, x: Any):
        try:
            return cls(x)
        except ValueError:
            return x


class SARBands(Enum):
    VV = "VV"
    VH = "VH"
    ASC_VV = "ASC_VV"
    ASC_VH = "ASC_VH"
    DSC_VV = "DSC_VV"
    DSC_VH = "DSC_VH"
    VV_VH = "VV_VH"

    @classmethod
    def try_convert_to_optical_bands_enum(cls, x: Any):
        try:
            return cls(x)
        except ValueError:
            return x


class MetadataBands(Enum):
    DEM = "DEM"
    NDVI = "NDVI"
    LULC = "LULC"


class LULCclasses(Enum):
    LULC = "LULC"


class Modalities(Enum):
    S1GRD = "S1GRD"
    S1RTC = "S1RTC"
    S2L1C = "S2L1C"
    S2L2A = "S2L2A"
    S2RGB = "S2RGB"
    DEM = "DEM"
    NDVI = "NDVI"
    LULC = "LULC"


def default_transform(**batch):
    return to_tensor(batch)


def generate_bands_intervals(bands_intervals: list[int | str | HLSBands | tuple[int]] | None = None):
    if bands_intervals is None:
        return None
    bands = []
    for element in bands_intervals:
        # if its an interval
        if isinstance(element, tuple):
            if len(element) != 2:  # noqa: PLR2004
                msg = "When defining an interval, a tuple of two integers should be passed,\
                        defining start and end indices inclusive"
                raise Exception(msg)
            expanded_element = list(range(element[0], element[1] + 1))
            bands.extend(expanded_element)
        else:
            bands.append(element)
    return bands


def filter_valid_files(
    files, valid_files: Iterator[str] | None = None, ignore_extensions: bool = False, allow_substring: bool = True
):
    if valid_files is None:
        return sorted(files)
    valid_files = list(valid_files)
    if ignore_extensions:
        valid_files = [os.path.splitext(sub)[0] for sub in valid_files]
    filter_function = partial(
        _split_filter_function,
        valid_files=valid_files,
        ignore_extensions=ignore_extensions,
        allow_substring=allow_substring,
    )
    # TODO fix this
    filtered = filter(filter_function, files)

    return sorted(filtered)


def _split_filter_function(file_name, valid_files: list[str], ignore_extensions=False, allow_substring=True):
    base_name = os.path.basename(file_name)
    if ignore_extensions:
        base_name = os.path.splitext(base_name)[0]
    if not allow_substring:
        return base_name in valid_files

    for valid_file in valid_files:
        if valid_file in base_name:
            return True
    return False


def to_tensor(d, transpose=True):
    new_dict = {}
    for k, v in d.items():
        if not isinstance(v, np.ndarray):
            new_dict[k] = v
        else:
            if k == "image" and v.ndim > 2 and transpose:
                v = np.moveaxis(v, -1, 0)
            new_dict[k] = torch.from_numpy(v)
    return new_dict


def pad_numpy(x, target_length, pad_value=0):
    padlen = target_length - x.shape[0]
    if padlen <= 0:
        return x

    pad_width = [(padlen, 0)] + [(0, 0) for _ in range(len(x.shape) - 1)]

    return np.pad(x, pad_width=pad_width, mode="constant", constant_values=pad_value)


def pad_dates_numpy(dates, target_length, pad_value=-1):
    padlen = target_length - dates.shape[0]
    if padlen <= 0:
        return dates

    pad_width = [(padlen, 0)]

    return np.pad(dates, pad_width=pad_width, mode="constant", constant_values=pad_value)


def validate_bands(bands: Sequence[str], bands_default: Sequence[str]):
    assert isinstance(bands, Sequence), "'bands' must be a sequence"
    set_diff = set(bands) - set(bands_default)
    if set_diff != set():
        raise ValueError(f"'{set_diff}' are invalid band names.")


def clip_image(img: np.ndarray) -> np.ndarray:
    """Clip image between (0, 1) considering min and max values.

    Args:
        img (np.ndarray): image in the format HWC.

    Returns:
        clipped ndarray image
    """
    img = (img - img.min(axis=(0, 1))) * (1 / img.max(axis=(0, 1)))
    img = np.clip(img, 0, 1)
    return img


def clip_image_percentile(img: np.ndarray, q_lower: float = 1, q_upper: float = 99) -> np.ndarray:
    """Remove values outside percentile range [lower, upper] and rescale image.

    Standalone numpy implementation based on ``np.percentile``; intentionally
    independent of torchgeo's normalization helpers.

    Args:
        img (np.ndarray): image in the format HWC.
        q_lower (float): lower percentile in [0,100].
        q_upper (float): upper percentile in [0,100].

    Returns:
        clipped ndarray image
    """
    assert q_lower < q_upper
    lower = np.percentile(img, q_lower)
    upper = np.percentile(img, q_upper)
    img = (img - lower) / (upper - lower + 1e-5)
    img = np.clip(img, 0, 1)

    return img


def to_rgb(
    image_chw: np.ndarray,
    rgb_indices: list[int],
    p_low: float = 0.0,
    p_high: float = 99.0,
    gamma: float = 0.7,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Convert a channel-first image (C, H, W) to an RGB image for visualization.

    Args:
        image_chw: Input image in (C, H, W) format.
        rgb_indices: Indices of channels to use as R, G, B.
        p_low: Lower percentile for contrast stretching.
        p_high: Upper percentile for contrast stretching.
        gamma: Gamma correction applied after normalization.
        eps: Small constant to avoid division by zero.

    Returns:
        RGB image in (H, W, 3) with values in [0, 1].
    """

    img = image_chw.take(rgb_indices, axis=0)
    img = np.transpose(img, (1, 2, 0))
    lo = np.percentile(img, p_low, axis=(0, 1), keepdims=True)
    hi = np.percentile(img, p_high, axis=(0, 1), keepdims=True)

    img = (img - lo) / (hi - lo + eps)

    if gamma is not None:
        img = np.power(img, gamma)

    return np.clip(img, 0, 1)


def to_pca_rgb(image_chw: np.ndarray, step: int = 4) -> np.ndarray:
    """Convert channel-first embedding (C, H, W) to a 3-channel PCA visualization.

    Args:
        image_chw: Spatial embeddings in (C, H, W) format.
        step (int): Spatial subsampling factor for PCA fitting in embedding visualizations.
            PCA components are estimated using only every pca_step-th spatial embedding
            (e.g. pca_step=4 uses 1/4 of embeddings), then applied to all embeddings. Defaults to 4.
    """
    if image_chw.ndim != 3:
        raise ValueError(f"Unsupported embedding shape {tuple(image_chw.shape)}")

    C, H, W = image_chw.shape
    emb_flat = image_chw.reshape(C, -1).T

    emb_fit = emb_flat[::step].astype(np.float32, copy=True)
    mean = emb_fit.mean(axis=0, keepdims=True)
    emb_fit -= mean

    pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
    pca.fit(emb_fit)

    proj = pca.transform(emb_flat.astype(np.float32, copy=False) - mean)

    min = proj.min(axis=0, keepdims=True)
    max = proj.max(axis=0, keepdims=True)
    proj = (proj - min) / np.maximum(max - min, 1e-8)

    return proj.reshape(H, W, 3), H, W


def resize_hwc(img_hwc: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Resize an image in (H, W, C) format to a target spatial size."""
    target_h, target_w = size_hw

    img_chw = np.transpose(img_hwc, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float()

    img_resized = F.interpolate(
        img_tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    return img_resized.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)


def extract_georeference(path: Path) -> dict[str, Any] | None:
    """
    Returns georeference needed to align rasters for embedding generation in a collate-friendly format.
    """
    try:
        with rasterio.open(path) as ds:
            if ds.crs is None or ds.transform is None:
                return None

            t = ds.transform
            left, bottom, right, top = ds.bounds
            return {
                "crs": ds.crs.to_string(),
                "geotransform": np.array([t.c, t.a, t.b, t.f, t.d, t.e], dtype=np.float32),
                "raster_shape": np.array([int(ds.height), int(ds.width)], dtype=np.float32),
                "bounds": np.array([left, bottom, right, top], dtype=np.float32),
            }
    except Exception:
        return None


def unbind_samples(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reverse of ``stack_samples``, tolerant of non-tensor values.

    torchgeo 0.10 made :func:`torchgeo.datasets.utils.unbind_samples` call
    ``torch.unbind`` unconditionally, which raises ``TypeError`` on non-tensor batch
    entries such as the list of strings under ``filename``. Sequence values are indexed
    positionally here instead, matching torchgeo <= 0.9 behaviour.

    Args:
        sample: a mini-batch of samples.

    Returns:
        A list of samples.
    """
    uncollated = {
        key: torch.unbind(values) if isinstance(values, torch.Tensor) else values
        for key, values in sample.items()
    }
    keys = [
        key
        for key, values in uncollated.items()
        if isinstance(values, Sequence) and not isinstance(values, str | bytes)
    ]
    n = min((len(uncollated[key]) for key in keys), default=0)
    return [{key: uncollated[key][i] for key in keys} for i in range(n)]
