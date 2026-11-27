# Copyright contributors to the Terratorch project

"""Module containing generic multimodal dataset classes"""

import glob
import logging
import os
import random
import re
import warnings
from abc import ABC
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import torch
import xarray as xr
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.transforms import MultimodalToTensor, MultimodalTransforms
from terratorch.datasets.utils import default_transform, generate_bands_intervals

logger = logging.getLogger("terratorch")


def load_table_data(file_path: str | Path) -> pd.DataFrame:
    file_path = str(file_path)
    if file_path.endswith("parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith("csv"):
        df = pd.read_csv(file_path, index_col=0)
    else:
        raise Exception(f"Unrecognized file type: {file_path}. Only parquet and csv are supported.")
    return df


class GenericMultimodalDataset(NonGeoDataset, ABC):
    """
    This is a generic dataset class initialized by
    [GenericMultiModalDataModule][terratorch.datamodules.GenericMultiModalDataModule].
    """

    def __init__(
        self,
        data_root: dict[str, Path | str],
        label_data_root: Path | str | list[Path | str] | None = None,
        image_grep: dict[str, str] | None = "*",
        label_grep: str | None = "*",
        prefix: str | None = "*",
        split: Path | None = None,
        image_modalities: list[str] | None = None,
        rgb_indices: dict[str, list[int]] | None = None,
        allow_missing_modalities: bool = False,
        allow_substring_file_names: bool = True,
        skip_file_checks: bool = False,
        dataset_bands: dict[str, list] | None = None,
        output_bands: dict[str, list] | None = None,
        constant_scale: dict[str, float] = None,
        transform: A.Compose | dict | None = None,
        no_data_replace: float | None = None,
        no_label_replace: float | None = -1,
        expand_temporal_dimension: bool = False,
        temporal_channel_major: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -3,
        scalar_label: bool = False,
        data_with_sample_dim: bool = False,
        concat_bands: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            data_root (dict[Path]): Dictionary of paths to data root directory or csv/parquet files with image-level
                data, with modalities as keys.
            label_data_root (Path, optional): Path to data root directory with labels or csv/parquet files with
                image-level labels. Needs to be specified for supervised tasks. Set to None for prediction mode.
            image_grep (dict[str], optional): Dictionary with regular expression appended to data_root to find input
                images, with modalities as keys. Defaults to "*". Ignored when allow_substring_file_names is False.
            label_grep (str, optional): Regular expression appended to label_data_root to find labels or mask files.
                Defaults to "*". Ignored when allow_substring_file_names is False.
             prefix (str, optional): Prefix for filenames and/ or in case of data in subdirs (e.g. "part-xxx). Defaults to "*".
            split (Path, optional): Path to file containing samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            image_modalities(list[str], optional): List of pixel-level raster modalities. Defaults to data_root.keys().
                The difference between all modalities and image_modalities are non-image modalities which are treated
                differently during the transforms and are not modified but only converted into a tensor if possible.
            rgb_indices (dict[str, list[int]], optional): Indices of RGB channels for plotting with the format
                {<modality>: [<band indices>]}. Defaults to {image_modalities[0]: [0, 1, 2]} if not provided.
            allow_missing_modalities (bool, optional): Allow missing modalities during data loading. Defaults to False.
            allow_substring_file_names (bool, optional): Allow substrings during sample identification using
                wildcards (*). If False, treats sample prefix + image_grep as full file name. Defaults to True.
            dataset_bands (dict[list], optional): Bands present in the dataset, provided in a dictionary with modalities
                as keys. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so
                that they can then be referred to by output_bands. Needs to be superset of output_bands. Can be a subset
                of all modalities. Defaults to None.
            output_bands (dict[list], optional): Bands that should be output by the dataset as named by dataset_bands,
                provided as a dictionary with modality keys. Can be subset of all modalities. Defaults to None.
            constant_scale (dict[str, float]): Factor to multiply data values by, provided as a dictionary with
                modalities as keys. Can be subset of all modalities. Defaults to None.
            transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities (transformation are shared between image modalities, e.g., similar crop or rotation).
                Should end with albumentations.ToTensorV2(). If used through the generic_data_module, should not include
                normalization. The transform is not applied to non-image data, which is only converted to tensors if
                possible. If dict with modalities as keys, transforms are applied seperatly per modality without shared
                parameters. Defaults to terratorch.datasets.transforms.MultimodalToTensor().
            no_data_replace (float | None): Replace nan values in input data with this value.
                If None, does no replacement. Defaults to None.
            no_label_replace (float | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to -1.
            skip_file_checks (bool, optional): Skips the check if a sample path exists. Only works
                with allow_missing_modalities=False and allow_substring_file_names=False. Samples are expected in the
                format <prefix><image_grep> without any wildcards (*), e.g. sample1_s2l2a.tif. Defaults to False.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Only works with image modalities. Is only applied to modalities with defined dataset_bands.
                Defaults to False. Assumes bands are grouped by time (all bands of one timestep are stacked together), otherwise set temporal_channel_major=True.
            temporal_channel_major: Used for expand_temporal_dimension, set True if bands are grouped by channel (all timesteps of one band are stacked together).
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            channel_position (int): Position of the channel dimension in the image modalities. Defaults to -3.
            scalar_label (bool): Returns a image mask if False or otherwise the raw labels. Defaults to False.
            concat_bands (bool): Concatenate all image modalities along the band dimension into a single "image", so
                that it can be processed by single-modal models. Concatenate in the order of provided modalities.
                Works with image modalities only. Does not work with allow_missing_modalities. Defaults to False.
        """

        super().__init__()

        self.split_file = split
        self.modalities = list(data_root.keys())
        assert "mask" not in self.modalities, "Modality cannot be called 'mask'."
        self.image_modalities = image_modalities or self.modalities
        self.non_image_modalities = list(set(self.modalities) - set(image_modalities))
        self.modalities = self.image_modalities + self.non_image_modalities  # Ensure image modalities to be first

        # Order by modalities and convert path strings to lists as the code expects a list of paths per modality
        data_root = {m: data_root[m] for m in self.modalities}

        # Default behaviour (no subdirs)
        self.subdir_data = False
        self.part_dirs = None
        self.prefix = prefix if prefix is not None else "*"

        self.constant_scale = constant_scale or {}
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.reduce_zero_label = reduce_zero_label
        self.expand_temporal_dimension = expand_temporal_dimension
        self.temporal_channel_major = temporal_channel_major
        self.channel_position = channel_position
        self.scalar_label = scalar_label
        self.data_with_sample_dim = data_with_sample_dim
        self.concat_bands = concat_bands
        assert not self.concat_bands or len(self.non_image_modalities) == 0, (
            f"concat_bands can only be used with image modalities, "
            f"but non-image modalities are given: {self.non_image_modalities}"
        )
        assert not self.concat_bands or not allow_missing_modalities, (
            "concat_bands cannot be used with allow_missing_modalities."
        )

        if self.expand_temporal_dimension:
            if not self.temporal_channel_major:
                warnings.warn(
                    "expand_temporal_dimension=True assumes bands are grouped by time "
                    "(all bands of one timestep are stacked together). "
                    "If instead bands are grouped by channel "
                    "(all timesteps of one band are stacked together), "
                    "set temporal_channel_major=True."
                )

            if dataset_bands is None:
                raise ValueError("Please provide dataset_bands when expand_temporal_dimension=True.")

        if scalar_label:
            self.non_image_modalities += ["label"]

        # Load samples based on split file
        if self.split_file is not None:
            if not os.path.isfile(self.split_file):
                raise FileNotFoundError(f"Split file {self.split_file} does not exist.")
            elif str(self.split_file).endswith(".txt"):
                with open(self.split_file) as f:
                    split = f.readlines()
                valid_files = [rf"{substring.strip()}" for substring in split]
            else:
                valid_files = list(load_table_data(self.split_file).index)
            if len(valid_files) == 0:
                raise ValueError(f"No sample candidates (file prefixes) found in split file {self.split_file}.")

            # Prepend prefix to all filenames, and warn if wildcards remain in it
            prefix = getattr(self, "prefix", "*")
            if prefix != "*":
                # Warn if prefix contains a wildcard pattern
                if "*" in prefix:
                    warnings.warn(
                        f"Prefix={prefix!r} contains a wildcard pattern. Prepending it to split-file entries may result in a heavy glob search later. "
                        f"If the split file entries already contain the prefix, do not pass the `prefix` argument. Otherwise, provide a non-glob prefix "
                        f"(e.g. 'part-000013/', 'files_') or consider adding prefix into split file entries.",
                        stacklevel=2,
                    )

                # Prepend prefix, make sure v is not an absolute path
                valid_files = [
                    (os.path.join(prefix, v.lstrip("/\\")) if v and not os.path.isabs(v) else v) for v in valid_files
                ]

        else:
            image_files = {}
            for m, m_paths in data_root.items():
                image_files[m] = sorted(glob.glob(os.path.join(m_paths, self.prefix + image_grep[m])))
                if len(image_files[m]) > 10_000:
                    warnings.warn("Found large data folder, consider providing split files to speed up dataset build.")

            def get_file_id(file_name, mod):
                base = os.path.basename(file_name)
                glob_as_regex = "^(.*?)" + "".join(re.escape(ch) for ch in image_grep[mod].strip("*")) + "$"
                stem = re.match(glob_as_regex, base).group(1)

                if "." not in image_grep[mod] and allow_substring_file_names:
                    stem = os.path.splitext(stem)[0]

                if "/" in self.prefix:
                    parent = os.path.basename(os.path.dirname(file_name))  # direct parent dir

                    # warn if we have more than one subdirectory level
                    if self.prefix.count("/") > 1:
                        warnings.warn(
                            f"Multiple subdirectories in {self.prefix!r}; only a single parent directory "
                            f"is supported. Using parent={parent!r}.",
                            stacklevel=2,
                        )
                    return f"{parent}/{stem}"

                return stem

            if allow_missing_modalities:
                valid_files = list(
                    set([get_file_id(file, mod) for mod, files in image_files.items() for file in files])
                )
            else:
                valid_files = [get_file_id(file, self.modalities[0]) for file in image_files[self.modalities[0]]]

        self.samples = []
        num_modalities = len(self.modalities) + int(label_data_root is not None)
        if len(valid_files) == 0:
            # Provide additional information if no candidates are found
            image_files = {m: f[:3] for m, f in image_files.items()}
            raise ValueError(
                f"No sample candidates (file prefixes) found for multimodal dataset. "
                f"Please review files and parameters.\n"
                f"data_root: {data_root}\n"
                f"image_grep: {image_grep}\n"
                f"allow_missing_modalities: {allow_missing_modalities}\n"
                f"File examples in data_root: {image_files}\n"
            )

        # Check for parquet and csv files with modality data and read the file
        for m, m_path in data_root.items():
            if os.path.isfile(m_path):
                data_root[m] = load_table_data(m_path)
                # Check for some sample keys
                if not any(f in data_root[m].index for f in valid_files[:100]):
                    warnings.warn(
                        f"Sample key expected in table index (first column) for {m} (file: {m_path}). "
                        f"{valid_files[:3] + ['...']} are not in index {list(data_root[m].index[:3]) + ['...']}."
                    )
        if label_data_root is not None:
            if os.path.isfile(label_data_root):
                label_data_root = load_table_data(label_data_root)
                # Check for some sample keys
                if not any(f in label_data_root.index for f in valid_files[:100]):
                    warnings.warn(
                        f"Keys expected in table index (first column) for labels (file: {label_data_root}). "
                        f"The keys {valid_files[:3] + ['...']} are not in the index."
                    )

        # Iterate over all files in split
        failed_candidates = []
        for file in valid_files:
            sample = {}
            # Iterate over all modalities
            for m, m_path in data_root.items():
                if isinstance(m_path, pd.DataFrame):
                    # Add tabular data to sample
                    sample[m] = m_path.loc[file].values
                elif allow_substring_file_names:
                    # Substring match with image_grep
                    if os.path.exists(os.path.join(m_path, file + image_grep[m].strip("*"))):
                        # Avoid glob if possible to speed up the dataset build
                        m_files = [os.path.join(m_path, file + image_grep[m].strip("*"))]
                    else:
                        m_files = sorted(glob.glob(os.path.join(m_path, file + image_grep[m])))
                        if len(valid_files) > 10_000:
                            warnings.warn(
                                "Found large data folder. You can speed up the dataset build by "
                                "providing split files with sample ids and suffixes without wildcards. E.g. "
                                "sample id 'sample1' and suffix '_s2l2a.tif' for file 'sample1_s2l2a.tif'."
                            )

                    if m_files:
                        sample[m] = m_files[-1]
                        if len(m_files) > 1:
                            warnings.warn(
                                f"Found multiple matching files for sample {file} and grep {image_grep[m]}: "
                                f"{m_files}. Selecting last one. "
                                f"Consider changing data structure or parameters for unique selection."
                            )
                else:
                    # Exact match
                    file_path = os.path.join(m_path, file + image_grep[m].strip("*"))
                    if skip_file_checks or os.path.exists(file_path):
                        sample[m] = file_path

            if label_data_root is not None:
                if isinstance(label_data_root, pd.DataFrame):
                    # Add tabular data to sample
                    sample["mask"] = label_data_root.loc[file].values
                elif allow_substring_file_names:
                    if os.path.exists(os.path.join(label_data_root, file + label_grep.strip("*"))):
                        # Avoid glob if possible to speed up the dataset build
                        l_files = [os.path.join(label_data_root, file + label_grep.strip("*"))]
                    else:
                        # Substring match with label_grep
                        l_files = sorted(glob.glob(os.path.join(label_data_root, file + label_grep)))
                    if l_files:
                        sample["mask"] = l_files[-1]
                else:
                    # Exact match
                    file_path = os.path.join(label_data_root, file + label_grep.strip("*"))
                    if skip_file_checks or os.path.exists(file_path):
                        sample["mask"] = file_path
                if "mask" not in sample:
                    # Only add sample if mask is present
                    failed_candidates.append(sample)
                    continue

            if len(sample) == num_modalities or allow_missing_modalities:
                self.samples.append(sample)
            else:
                failed_candidates.append(sample)

        if len(self.samples) == 0:
            # Provide additional information if no multi-modal samples are found
            idx = random.sample(range(len(valid_files)), min(5, len(valid_files)))
            raise ValueError(
                f"No samples found for multimodal dataset. Please review files, path, and grep params.\n"
                f"data_root: {data_root}\n"
                f"image_grep: {image_grep}\n"
                f"allow_substring_file_names: {allow_substring_file_names}\n"
                f"allow_missing_modalities: {allow_missing_modalities}\n"
                f"Candidate prefixes: {', '.join([valid_files[i] for i in idx])}\n"
                f"Sample candidate paths: {', '.join([str(failed_candidates[i]) for i in idx])}"
            )

        self.rgb_indices = rgb_indices or {image_modalities[0]: [0, 1, 2]}

        if dataset_bands is not None:
            self.dataset_bands = {m: generate_bands_intervals(m_bands) for m, m_bands in dataset_bands.items()}
        else:
            self.dataset_bands = None
        if output_bands is not None:
            self.output_bands = {m: generate_bands_intervals(m_bands) for m, m_bands in output_bands.items()}
            for modality in self.modalities:
                if modality in self.output_bands and modality not in self.dataset_bands:
                    msg = f"If output bands are provided, dataset_bands must also be provided (modality: {modality})"
                    raise Exception(msg)
        else:
            self.output_bands = {}

        self.filter_indices = {}
        if self.output_bands:
            for m in self.output_bands.keys():
                if m not in self.output_bands or self.output_bands[m] == self.dataset_bands[m]:
                    continue
                if len(set(self.output_bands[m]) & set(self.dataset_bands[m])) != len(self.output_bands[m]):
                    msg = f"Output bands must be a subset of dataset bands (Modality: {m})"
                    raise Exception(msg)

                self.filter_indices[m] = [self.dataset_bands[m].index(band) for band in self.output_bands[m]]

            if not self.channel_position:
                logger.warning(
                    "output_bands is defined but no channel_position is provided. "
                    "Channels must be in the last dimension, otherwise provide channel_position."
                )

        # If no transform is given, apply only to transform to torch tensor
        if isinstance(transform, A.Compose):
            self.transform = MultimodalTransforms(
                transform,
                non_image_modalities=self.non_image_modalities + ["label"]
                if scalar_label
                else self.non_image_modalities,
            )
        elif isinstance(transform, dict):
            # Modality-specific transforms
            transform = {m: transform.get(m) or default_transform for m in self.modalities}
            self.transform = MultimodalTransforms(transform, shared=False)
        elif transform is None:
            self.transform = MultimodalToTensor(self.modalities)
        else:
            raise ValueError(f"Unknown transform type: {type(transform)}, expect A.Compose, dict or None.")

        # Ignore rasterio warning for not geo-referenced files
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        warnings.filterwarnings("ignore", message="Dataset has no geotransform")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        output = {}
        if isinstance(index, tuple):
            # Load only sampled modalities instead of all modalities
            # (see sample_num_modalities in GenericMultiModalDataModule for details)
            index, modalities = index
            sample = {m: self.samples[index][m] for m in modalities}
        else:
            sample = self.samples[index]

        for modality, file in sample.items():
            data = self._load_file(
                file,
                nan_replace=self.no_label_replace if modality == "mask" else self.no_data_replace,
                modality=modality,
            )

            # Expand temporal dim
            if modality in self.filter_indices and self.expand_temporal_dimension:
                if self.temporal_channel_major:
                    data = rearrange(
                        data, "(channels time) h w -> channels time h w", channels=len(self.dataset_bands[modality])
                    )
                else:
                    data = rearrange(
                        data, "(time channels) h w -> channels time h w", channels=len(self.dataset_bands[modality])
                    )

            if modality == "mask" and not self.scalar_label:
                # tasks expect image masks without channel dim
                data = data[0]

            if modality in self.image_modalities and len(data.shape) >= 3 and self.channel_position:
                # to channels last (required by albumentations)
                data = np.moveaxis(data, self.channel_position, -1)

            if modality in self.filter_indices:
                data = data[..., self.filter_indices[modality]]

            if modality in self.constant_scale:
                data = data.astype(np.float32) * self.constant_scale[modality]

            if data.dtype == np.float64:
                data = data.astype(np.float32)

            output[modality] = data

        if "mask" in output:
            if self.reduce_zero_label:
                output["mask"] -= 1
            if self.scalar_label:
                output["label"] = output.pop("mask")

        if self.transform:
            output = self.transform(output)

        if self.concat_bands:
            # Concatenate bands of all image modalities
            data = [output.pop(m) for m in self.image_modalities if m in output]
            output["image"] = torch.cat(data, dim=1 if self.data_with_sample_dim else 0)
        else:
            # Tasks expect data to be stored in "image", moving modalities to image dict
            output["image"] = {m: output.pop(m) for m in self.modalities if m in output}

        output["filename"] = self.samples[index]

        return output

    def _load_file(self, path, nan_replace: int | float | None = None, modality: str | None = None) -> xr.DataArray:
        if isinstance(path, np.ndarray):
            # data was loaded from table and is saved in memory
            data = path
        elif path.endswith(".zarr") or path.endswith(".zarr.zip"):
            data = xr.open_zarr(path, mask_and_scale=True)
            data_var = modality if modality in data.data_vars else list(data.data_vars)[0]
            data = data[data_var].to_numpy()
        elif path.endswith(".npy"):
            data = np.load(path)
        else:
            data = rioxarray.open_rasterio(path, masked=True).to_numpy()

        if nan_replace is not None:
            data = np.nan_to_num(data, nan=nan_replace)
        return data

    def plot(self, sample: dict[str, torch.Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        if suptitle is None:
            # Infer suitable subtitle from filename
            filename = sample.get("filename", "")
            if isinstance(filename, str):
                suptitle = filename.split("/")[-1].split(".")[0]
            elif isinstance(filename, dict):
                # Can be dict if user plots dataset samples
                suptitle = list(filename.values())[0].split("/")[-1].split(".")[0]
        images = {}
        for mod, indices in self.rgb_indices.items():
            if "image" in sample and isinstance(sample["image"], dict) and mod in sample["image"]:
                # Move modality to sample dict
                sample[mod] = sample["image"][mod]
            if mod in sample.keys():
                image = sample[mod][indices]
                # Per modality processing
                if isinstance(image, torch.Tensor):
                    image = image.numpy()
                # Normalize to 0 - 1
                image = image - image.min(axis=(-1, -2), keepdims=True)
                image = image / np.quantile(image, q=0.99, axis=(-1, -2), keepdims=True)
                image = np.clip(image, 0, 1)
                # Channel last
                image = np.moveaxis(image, 0, -1)
                if image.ndim == 4:
                    warnings.warn("Found time series data. Plotting only supports images, selecting the first one.")
                    image = image[0]
                images[mod] = image

        if len(images) == 0:
            warnings.warn(
                f"No RGB modalities found ({suptitle}). Sample keys: {list(sample.keys())}, "
                f"Dataset rgb_indices modalities: {list(self.rgb_indices.keys())}"
            )
            raise ValueError("No RGB images found.")

        if "mask" in sample:
            mask = sample["mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
        else:
            mask = None

        if "prediction" in sample:
            prediction = sample["prediction"]
            if isinstance(prediction, dict):
                raise ValueError("Multiple outputs not yet supported")
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.numpy()
            while prediction.ndim < 2:
                prediction = np.expand_dims(prediction, -1)
        else:
            prediction = None

        # Scalar label
        if "label" in sample:
            label = sample["label"]
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            while label.ndim < 2:
                label = np.expand_dims(label, -1)
        else:
            label = None

        if hasattr(self, "num_classes"):
            # Classification, Segmentation
            vmin, vmax = 0, self.num_classes - 1
            cmap = plt.get_cmap("rainbow")
            cmap = ListedColormap(cmap(np.linspace(0.10, 1.0, self.num_classes)))  # Start with blue
            class_names = self.class_names or list(range(0, self.num_classes))
            handles = [Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(self.num_classes)]

            if self.no_label_replace is not None:
                if mask is not None:
                    mask[mask == self.no_label_replace] = -1
                elif label is not None:
                    label[label == self.no_label_replace] = -1
                vmin = -1
                cmap = cmap(np.arange(self.num_classes))
                cmap = ListedColormap(np.vstack(([0, 0, 0, 1], cmap)), N=len(cmap) + 1)
                class_names = ["No label"] + class_names
                handles = [Rectangle((0, 0), 1, 1, color=(0, 0, 0, 1))] + handles
        else:
            # Regression
            vmax = np.max(
                [
                    mask.max() if mask is not None else 0,
                    prediction.max() if prediction is not None else 0,
                    label.max() if label is not None else 0,
                ]
            )
            vmin = np.min(
                [
                    mask.min() if mask is not None else 0,
                    prediction.min() if prediction is not None else 0,
                    label.min() if label is not None else 0,
                ]
            )
            cmap = "viridis"
            class_names = handles = None

        # Plot images
        num_images = len(images) + int(mask is not None or label is not None) + int(prediction is not None)
        fig, ax = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

        for i, (mod, image) in enumerate(images.items()):
            ax[i].imshow(image)
            ax[i].axis("off")
            ax[i].set_title(mod)

        image = list(images.values())[0]  # First RGB modality as base image for mask

        mask_i = -1 if prediction is None else -2
        if mask is not None:
            ax[mask_i].imshow(image)
            ax[mask_i].imshow(mask, alpha=0.7, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
            ax[mask_i].axis("off")
            ax[mask_i].set_title("GT Mask")

            if class_names is not None:
                # Segmentation task
                ax[-1].legend(handles, class_names, loc="upper left", bbox_to_anchor=(1, 1))

        if prediction is not None:
            ax[-1].imshow(image)
            ax[-1].imshow(prediction, alpha=0.7, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
            ax[-1].axis("off")
            ax[-1].set_title("Prediction")

        if label is not None:
            # Plot scalar values
            ax[mask_i].imshow(image)
            ax[mask_i].imshow(label, alpha=0.7, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
            ax[mask_i].axis("off")
            ax[mask_i].set_title(f"GT Label:\n{label[:, 0]}")
            if prediction is not None:
                ax[-1].set_title(f"Prediction:\n{prediction[:, 0]}")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class GenericMultimodalSegmentationDataset(GenericMultimodalDataset):
    """GenericNonGeoSegmentationDataset"""

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        image_modalities: list[str] | None = None,
        rgb_indices: list[str] | None = None,
        allow_missing_modalities: bool = False,
        allow_substring_file_names: bool = False,
        skip_file_checks: bool = False,
        dataset_bands: dict[str, list] | None = None,
        output_bands: dict[str, list] | None = None,
        class_names: list[str] | None = None,
        constant_scale: dict[str, float] = None,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = -1,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -3,
        concat_bands: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            data_root (dict[Path]): Dictionary of paths to data root directory or csv/parquet files with image-level
                data, with modalities as keys.
            num_classes (int): Number of classes.
            label_data_root (Path): Path to data root directory with mask files. Set to None for prediction mode.
            image_grep (dict[str], optional): Dictionary with regular expression appended to data_root to find input
                images, with modalities as keys. Defaults to "*". Ignored when allow_substring_file_names is False.
            label_grep (str, optional): Regular expression appended to label_data_root to find mask files.
                Defaults to "*". Ignored when allow_substring_file_names is False.
            split (Path, optional): Path to file containing samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            image_modalities(list[str], optional): List of pixel-level raster modalities. Defaults to data_root.keys().
                The difference between all modalities and image_modalities are non-image modalities which are treated
                differently during the transforms and are not modified but only converted into a tensor if possible.
            rgb_indices (dict[str, list[int]], optional): Indices of RGB channels for plotting with the format
                {<modality>: [<band indices>]}. Defaults to {image_modalities[0]: [0, 1, 2]} if not provided.
            allow_missing_modalities (bool, optional): Allow missing modalities during data loading. Defaults to False.
            allow_substring_file_names (bool, optional): Allow substrings during sample identification using
                wildcards (*). If False, treats sample prefix + image_grep as full file name. Defaults to True.
            skip_file_checks (bool, optional): Skips the check if a sample path exists. Only works
                with allow_missing_modalities=False and allow_substring_file_names=False. Samples are expected in the
                format <prefix><image_grep> without any wildcards (*), e.g. sample1_s2l2a.tif. Defaults to False.
            dataset_bands (dict[list], optional): Bands present in the dataset, provided in a dictionary with modalities
                as keys. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so
                that they can then be referred to by output_bands. Needs to be superset of output_bands. Can be a subset
                of all modalities. Defaults to None.
            output_bands (dict[list], optional): Bands that should be output by the dataset as named by dataset_bands,
                provided as a dictionary with modality keys. Can be subset of all modalities. Defaults to None.
            class_names (list[str], optional): Names of the classes. Defaults to None.
            constant_scale (dict[str, float]): Factor to multiply data values by, provided as a dictionary with
                modalities as keys. Can be subset of all modalities. Defaults to None.
            transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities (transformation are shared between image modalities, e.g., similar crop or rotation).
                Should end with albumentations.ToTensorV2(). If used through the generic_data_module, should not include
                normalization. The transform is not applied to non-image data, which is only converted to tensors if
                possible. If dict with modalities as keys, transforms are applied seperatly per modality without shared
                parameters. Defaults to terratorch.datasets.transforms.MultimodalToTensor().
            no_data_replace (float | None): Replace nan values in input data with this value.
                If None, does no replacement. Defaults to None.
            no_label_replace (float | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to -1.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Only works with image modalities. Is only applied to modalities with defined dataset_bands.
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            channel_position (int): Position of the channel dimension in the image modalities. Defaults to -3.
            concat_bands (bool): Concatenate all image modalities along the band dimension into a single "image", so
                that it can be processed by single-modal models. Concatenate in the order of provided modalities.
                Works with image modalities only. Does not work with allow_missing_modalities. Defaults to False.
        """

        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            image_modalities=image_modalities,
            rgb_indices=rgb_indices,
            allow_missing_modalities=allow_missing_modalities,
            allow_substring_file_names=allow_substring_file_names,
            skip_file_checks=skip_file_checks,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
            channel_position=channel_position,
            concat_bands=concat_bands,
            *args,
            **kwargs,
        )
        self.num_classes = num_classes
        self.class_names = class_names

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)

        if "mask" in item:
            item["mask"] = item["mask"].long()

        return item


class GenericMultimodalPixelwiseRegressionDataset(GenericMultimodalDataset):
    """GenericNonGeoPixelwiseRegressionDataset"""

    def __init__(
        self,
        data_root: Path,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        image_modalities: list[str] | None = None,
        rgb_indices: list[int] | None = None,
        allow_missing_modalities: bool = False,
        allow_substring_file_names: bool = False,
        skip_file_checks: bool = False,
        dataset_bands: dict[str, list] | None = None,
        output_bands: dict[str, list] | None = None,
        constant_scale: dict[str, float] = None,
        transform: A.Compose | dict | None = None,
        no_data_replace: float | None = None,
        no_label_replace: float | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -3,
        concat_bands: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            data_root (dict[Path]): Dictionary of paths to data root directory or csv/parquet files with image-level
                data, with modalities as keys.
            label_data_root (Path): Path to data root directory with ground truth files. Set to None for predictions.
            image_grep (dict[str], optional): Dictionary with regular expression appended to data_root to find input
                images, with modalities as keys. Defaults to "*". Ignored when allow_substring_file_names is False.
            label_grep (str, optional): Regular expression appended to label_data_root to find ground truth files.
                Defaults to "*". Ignored when allow_substring_file_names is False.
            split (Path, optional): Path to file containing samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            image_modalities(list[str], optional): List of pixel-level raster modalities. Defaults to data_root.keys().
                The difference between all modalities and image_modalities are non-image modalities which are treated
                differently during the transforms and are not modified but only converted into a tensor if possible.
            rgb_indices (dict[str, list[int]], optional): Indices of RGB channels for plotting with the format
                {<modality>: [<band indices>]}. Defaults to {image_modalities[0]: [0, 1, 2]} if not provided.
            allow_missing_modalities (bool, optional): Allow missing modalities during data loading. Defaults to False.
            allow_substring_file_names (bool, optional): Allow substrings during sample identification using
                wildcards (*). If False, treats sample prefix + image_grep as full file name. Defaults to True.
            skip_file_checks (bool, optional): Skips the check if a sample path exists. Only works
                with allow_missing_modalities=False and allow_substring_file_names=False. Samples are expected in the
                format <prefix><image_grep> without any wildcards (*), e.g. sample1_s2l2a.tif. Defaults to False.
            dataset_bands (dict[list], optional): Bands present in the dataset, provided in a dictionary with modalities
                as keys. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so
                that they can then be referred to by output_bands. Needs to be superset of output_bands. Can be a subset
                of all modalities. Defaults to None.
            output_bands (dict[list], optional): Bands that should be output by the dataset as named by dataset_bands,
                provided as a dictionary with modality keys. Can be subset of all modalities. Defaults to None.
            constant_scale (dict[float]): Factor to multiply data values by, provided as a dictionary with modalities as
                keys. Can be subset of all modalities. Defaults to None.
            transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities (transformation are shared between image modalities, e.g., similar crop or rotation).
                Should end with albumentations.ToTensorV2(). If used through the generic_data_module, should not include
                normalization. The transform is not applied to non-image data, which is only converted to tensors if
                possible. If dict with modalities as keys, transforms are applied seperatly per modality without shared
                parameters. Defaults to terratorch.datasets.transforms.MultimodalToTensor().
            no_data_replace (float | None): Replace nan values in input data with this value.
                If None, does no replacement. Defaults to None.
            no_label_replace (float | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to None.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Only works with image modalities. Is only applied to modalities with defined dataset_bands.
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            channel_position (int): Position of the channel dimension in the image modalities. Defaults to -3.
            concat_bands (bool): Concatenate all image modalities along the band dimension into a single "image", so
                that it can be processed by single-modal models. Concatenate in the order of provided modalities.
                Works with image modalities only. Does not work with allow_missing_modalities. Defaults to False.
        """

        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            image_modalities=image_modalities,
            rgb_indices=rgb_indices,
            allow_missing_modalities=allow_missing_modalities,
            allow_substring_file_names=allow_substring_file_names,
            skip_file_checks=skip_file_checks,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
            channel_position=channel_position,
            concat_bands=concat_bands,
            *args,
            **kwargs,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)

        if "mask" in item:
            item["mask"] = item["mask"].float()

        return item


class GenericMultimodalScalarDataset(GenericMultimodalDataset):
    """GenericMultimodalClassificationDataset"""

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        image_modalities: list[str] | None = None,
        rgb_indices: list[int] | None = None,
        allow_missing_modalities: bool = False,
        allow_substring_file_names: bool = False,
        skip_file_checks: bool = False,
        dataset_bands: dict[str, list] | None = None,
        output_bands: dict[str, list] | None = None,
        class_names: list[str] | None = None,
        constant_scale: dict[str, float] = None,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        channel_position: int = -3,
        concat_bands: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Constructor

        Args:
            data_root (dict[Path]): Dictionary of paths to data root directory or csv/parquet files with image-level
                data, with modalities as keys.
            num_classes (int): Number of classes.
            label_data_root (Path, optional): Path to data root directory with labels or csv/parquet files with labels.
                Set to None for prediction mode.
            image_grep (dict[str], optional): Dictionary with regular expression appended to data_root to find input
                images, with modalities as keys. Defaults to "*". Ignored when allow_substring_file_names is False.
            label_grep (str, optional): Regular expression appended to label_data_root to find labels files.
                Defaults to "*". Ignored when allow_substring_file_names is False.
            split (Path, optional): Path to file containing samples prefixes to be used for this split.
                The file can be a csv/parquet file with the prefixes in the index or a txt file with new-line separated
                sample prefixes. File names must be exact matches if allow_substring_file_names is False. Otherwise,
                files are searched using glob with the form Path(data_root).glob(prefix + [image or label grep]).
                If not specified, search samples based on files in data_root. Defaults to None.
            image_modalities(list[str], optional): List of pixel-level raster modalities. Defaults to data_root.keys().
                The difference between all modalities and image_modalities are non-image modalities which are treated
                differently during the transforms and are not modified but only converted into a tensor if possible.
            rgb_indices (dict[str, list[int]], optional): Indices of RGB channels for plotting with the format
                {<modality>: [<band indices>]}. Defaults to {image_modalities[0]: [0, 1, 2]} if not provided.
            allow_missing_modalities (bool, optional): Allow missing modalities during data loading. Defaults to False.
            allow_substring_file_names (bool, optional): Allow substrings during sample identification using
                wildcards (*). If False, treats sample prefix + image_grep as full file name. Defaults to True.
            skip_file_checks (bool, optional): Skips the check if a sample path exists. Only works
                with allow_missing_modalities=False and allow_substring_file_names=False. Samples are expected in the
                format <prefix><image_grep> without any wildcards (*), e.g. sample1_s2l2a.tif. Defaults to False.
            dataset_bands (dict[list], optional): Bands present in the dataset, provided in a dictionary with modalities
                as keys. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so
                that they can then be referred to by output_bands. Needs to be superset of output_bands. Can be a subset
                of all modalities. Defaults to None.
            output_bands (dict[list], optional): Bands that should be output by the dataset as named by dataset_bands,
                provided as a dictionary with modality keys. Can be subset of all modalities. Defaults to None.
            class_names (list[str], optional): Names of the classes. Defaults to None.
            constant_scale (dict[str, float]): Factor to multiply data values by, provided as a dictionary with
                modalities as keys. Can be subset of all modalities. Defaults to None.
            transform (Albumentations.Compose | dict | None): Albumentations transform to be applied to all image
                modalities (transformation are shared between image modalities, e.g., similar crop or rotation).
                Should end with albumentations.ToTensorV2(). If used through the generic_data_module, should not include
                normalization. The transform is not applied to non-image data, which is only converted to tensors if
                possible. If dict with modalities as keys, transforms are applied seperatly per modality without shared
                parameters. Defaults to terratorch.datasets.transforms.MultimodalToTensor().
            no_data_replace (float | None): Replace nan values in input data with this value.
                If None, does no replacement. Defaults to None.
            no_label_replace (float | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to -1.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Only works with image modalities. Is only applied to modalities with defined dataset_bands.
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            channel_position (int): Position of the channel dimension in the image modalities. Defaults to -3.
            concat_bands (bool): Concatenate all image modalities along the band dimension into a single "image", so
                that it can be processed by single-modal models. Concatenate in the order of provided modalities.
                Works with image modalities only. Does not work with allow_missing_modalities. Defaults to False.
        """

        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            image_modalities=image_modalities,
            rgb_indices=rgb_indices,
            allow_missing_modalities=allow_missing_modalities,
            allow_substring_file_names=allow_substring_file_names,
            skip_file_checks=skip_file_checks,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
            channel_position=channel_position,
            scalar_label=True,
            concat_bands=concat_bands,
            *args,
            **kwargs,
        )

        self.num_classes = num_classes
        self.class_names = class_names

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        return item
