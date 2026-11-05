# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Module containing generic dataset classes"""

import glob
import os
import warnings
from abc import ABC
from collections.abc import Hashable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    try:
        from tacoreader.dataset import TacoDataset
    except ImportError:
        TacoDataset = Any  # type: ignore[assignment,misc]

import albumentations as A
import matplotlib as mpl
import numpy as np
import pandas as pd
import rioxarray
import torch
import xarray as xr
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from rasterio.errors import NotGeoreferencedWarning
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import (
    HLSBands,
    default_transform,
    extract_georeference,
    filter_valid_files,
    generate_bands_intervals,
    resize_hwc,
    to_pca_rgb,
    to_rgb,
)


class GenericPixelWiseDataset(NonGeoDataset, ABC):
    """
    This is a generic dataset class to be used for instantiating datasets from arguments.
    Ideally, one would create a dataset class specific to a dataset.
    """

    def __init__(
        self,
        data_root: Path | None = None,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[int] | None = None,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        embedding_input: bool = False,
        pca_step: int = 4,
        expand_temporal_dimension: bool = False,
        temporal_channel_major: bool = False,
        reduce_zero_label: bool = False,
        tortilla_df: "pd.DataFrame | TacoDataset | None" = None,
        tortilla_indices: list[Hashable] | None = None,
        return_georeference: bool = False,
    ) -> None:
        """Constructor

        Args:
            data_root (Path): Path to data root directory
            label_data_root (Path, optional): Path to data root directory with labels.
                If not specified, will use the same as for images.
            image_grep (str, optional): Regular expression appended to data_root to find input images.
                Defaults to "*".
            label_grep (str, optional): Regular expression appended to data_root to find ground truth masks.
                Defaults to "*".
            split (Path, optional): Path to file containing files to be used for this split.
                The file should be a new-line separated prefixes contained in the desired files.
                Files will be seached using glob with the form Path(data_root).glob(prefix + [image or label grep])
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            dataset_bands (list[HLSBands | int | tuple[int, int] | str] | None): Bands present in the dataset. This parameter names input channels (bands) using HLSBands, ints, int ranges, or strings, so that they can then be refered to by output_bands. Defaults to None.
            output_bands (list[HLSBands | int | tuple[int, int] | str] | None): Bands that should be output by the dataset as named by dataset_bands.
            constant_scale (float): Factor to multiply image values by. Defaults to 1.
            transform (Albumentations.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value. If none, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value. If none, does no replacement. Defaults to -1.
            embedding_input (bool): Whether the input represents embeddings rather than an image, used for plotting. Defaults to False.
            pca_step (int): Spatial subsampling factor for PCA fitting in embedding visualizations.
                PCA components are estimated using only every pca_step-th spatial embedding
                (e.g. pca_step=4 uses 1/4 of embeddings), then applied to all embeddings. Defaults to 4.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False. Assumes bands are grouped by time (all bands of one timestep are stacked together), otherwise set temporal_channel_major=True.
            temporal_channel_major: Used for expand_temporal_dimension, set True if bands are grouped by channel (all timesteps of one band are stacked together).
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            tortilla_df (pd.DataFrame | TacoDataset | None): Tortilla DataFrame (tacoreader v1) or
                TacoDataset (tacoreader v2) to use for loading data. Defaults to None.
                If provided, data_root is ignored.
            tortilla_indices (list[Hashable] | None): List of indices to use from tortilla_df. Defaults to None, which uses all indices.
            return_georeference (bool): Whether to return georeference metadata info (CRS, Bounds, ...). Defaults to False.
        """
        super().__init__()

        # Resolve TacoDataset once for all v2 isinstance checks below
        try:
            from tacoreader.dataset import TacoDataset as _TacoDataset
        except ImportError:
            _TacoDataset = None  # type: ignore[assignment]

        self.split_file = split

        label_data_root = label_data_root if label_data_root is not None else data_root

        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.reduce_zero_label = reduce_zero_label
        self.expand_temporal_dimension = expand_temporal_dimension
        self.temporal_channel_major = temporal_channel_major
        self.return_georeference = return_georeference

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

        self.tortilla_df = tortilla_df
        if tortilla_indices is None and self.tortilla_df is not None:
            # v2 TacoDataset: use row count from materialised data
            if _TacoDataset is not None and isinstance(self.tortilla_df, _TacoDataset):
                self.tortilla_indices = list(range(len(self.tortilla_df.data)))
            else:
                # v1 TortillaDataFrame (pandas DataFrame subclass)
                self.tortilla_indices = self.tortilla_df.index.tolist()
        else:
            self.tortilla_indices = tortilla_indices

        if data_root is None and self.tortilla_df is None:
            msg = "Please provide either data_root or tortilla_df"
            raise Exception(msg)

        self.image_files = []
        self.segmentation_mask_files = []
        if self.tortilla_df is not None:
            for sample_index in self.tortilla_indices:
                sample_data = self.tortilla_df.read(sample_index)
                # v2 TacoDataset: .read() on a FOLDER returns a TacoDataFrame;
                # .read() on a FILE returns the GDAL VSI path as a str.
                if _TacoDataset is not None and isinstance(self.tortilla_df, _TacoDataset):
                    if isinstance(sample_data, str):
                        # Top-level FILE — treat the single VSI path as the image
                        self.image_files.append(sample_data)
                    else:
                        # FOLDER: iterate child rows (TacoDataFrame backed by PyArrow)
                        arrow_table = sample_data.to_arrow()
                        for row in arrow_table.to_pylist():
                            sample_id = row.get("id", "")
                            vsi_path = row.get("internal:gdal_vsi")
                            if sample_id == "image":
                                self.image_files.append(vsi_path)
                            elif sample_id == "label":
                                self.segmentation_mask_files.append(vsi_path)
                elif isinstance(sample_data, pd.DataFrame):
                    # v1 TortillaDataFrame child
                    for _, row in sample_data.iterrows():
                        if row.get("tortilla:id") == "image":
                            self.image_files.append(row.get("internal:subfile"))
                        elif row.get("tortilla:id") == "label":
                            self.segmentation_mask_files.append(row.get("internal:subfile"))
        else:
            self.image_files = sorted(glob.glob(os.path.join(data_root, image_grep)))
            self.segmentation_mask_files = sorted(glob.glob(os.path.join(label_data_root, label_grep)))

        if self.expand_temporal_dimension and output_bands is None:
            msg = "Please provide output_bands when expand_temporal_dimension is True"
            raise Exception(msg)

        if self.split_file is not None and self.tortilla_df is None:
            with open(self.split_file) as f:
                split = f.readlines()
            valid_files = {rf"{substring.strip()}" for substring in split}
            self.image_files = filter_valid_files(
                self.image_files,
                valid_files=valid_files,
                ignore_extensions=ignore_split_file_extensions,
                allow_substring=allow_substring_split_file,
            )
            self.segmentation_mask_files = filter_valid_files(
                self.segmentation_mask_files,
                valid_files=valid_files,
                ignore_extensions=ignore_split_file_extensions,
                allow_substring=allow_substring_split_file,
            )

        # We don't define a split file for prediction
        if not self.split_file:
            # When prediction is enabled, we don't have mask files, so
            # we need to provide a way to run the dataloder in these cases.
            if not self.segmentation_mask_files:
                self.segmentation_mask_files = self.image_files
                # The masks can be `None` since they won't be used in fact.

        self.rgb_indices = [0, 1, 2] if rgb_indices is None else rgb_indices

        self.dataset_bands = generate_bands_intervals(dataset_bands)
        self.output_bands = generate_bands_intervals(output_bands)
        self.embedding_input = embedding_input
        self.pca_step = pca_step

        if self.output_bands and not self.dataset_bands:
            msg = "If output bands provided, dataset_bands must also be provided"
            return Exception(msg)  # noqa: PLE0101

        # There is a special condition if the bands are defined as simple strings.
        if self.output_bands:
            if len(set(self.output_bands) & set(self.dataset_bands)) != len(self.output_bands):
                msg = "Output bands must be a subset of dataset bands"
                raise Exception(msg)

            self.filter_indices = [self.dataset_bands.index(band) for band in self.output_bands]

        else:
            self.filter_indices = None

        # If no transform is given, apply only to transform to torch tensor
        self.transform = transform if transform else default_transform
        # self.transform = transform if transform else ToTensorV2()

        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, georef = self._load_file(self.image_files[index], nan_replace=self.no_data_replace)
        image = image.to_numpy()
        # to channels last
        if self.expand_temporal_dimension:
            if self.temporal_channel_major:
                image = rearrange(image, "(channels time) h w -> channels time h w", channels=len(self.dataset_bands))
            else:
                image = rearrange(image, "(time channels) h w -> channels time h w", channels=len(self.dataset_bands))
        image = np.moveaxis(image, 0, -1)
        if self.filter_indices:
            image = image[..., self.filter_indices]
        output = {
            "image": image.astype(np.float32) * self.constant_scale,
        }
        if georef:
            output["metadata"] = georef
        if self.segmentation_mask_files:
            mask, _ = self._load_file(self.segmentation_mask_files[index], nan_replace=self.no_label_replace)
            output["mask"] = mask.to_numpy()[0]
            if self.reduce_zero_label:
                output["mask"] -= 1
        if self.transform:
            output = self.transform(**output)
        output["filename"] = self.image_files[index]

        return output

    def _load_file(self, path, nan_replace: int | float | None = None) -> tuple[xr.DataArray, dict | None]:
        data = rioxarray.open_rasterio(path, masked=True)
        if nan_replace is not None:
            data = data.fillna(nan_replace)
        if getattr(self, "return_georeference", False):
            georef = extract_georeference(path)
            return data, georef
        else:
            return data, None

    def plot(
        self,
        sample: dict[str, Tensor],
        suptitle: str | None = None,
        show_axes: bool = False,
        embedding_input: bool | None = None,
        pca_step: int | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            suptitle: Optional string to use as a suptitle.
            show_axes: Whether to show axes.
            embedding_input: Whether the input represents embedding rather than an image.
                If ``True``, embedding is visualized as a 3-channel PCA image and resized
                to match the spatial shape of the target or prediction. If ``None``, the
                value configured at dataset initialization is used.
            pca_step: Spatial downsampling factor used when fitting PCA on embeddings.
                Larger values increase efficiency at the cost of stability. If ``None``,
                the value configured at dataset initialization is used.

        .. versionadded:: 0.2
        """

        image = sample["image"]
        embedding_input = self.embedding_input if embedding_input is None else embedding_input
        pca_step = self.pca_step if pca_step is None else pca_step

        if len(image.shape) == 5:
            return
        if len(image.shape) == 4:
            image = torch.squeeze(image[:, -1, :, :], 1)
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()

        label_mask = sample["mask"]
        if isinstance(label_mask, Tensor):
            label_mask = label_mask.detach().cpu().numpy()

        if embedding_input:
            if image.ndim == 2:
                warnings.warn(
                    "Embedding plotting is only supported for spatially arranged embeddings. "
                    "For example, in TerraTorch you can use a spatial output format such as TIFF, "
                    "or add a ReshapeTokensToImage neck during Embedding Generation."
                )
                return  # Plotting requires (reshaped) spatial (C, H, W) input embeddings.

            image_for_plot, H_emb, W_emb = to_pca_rgb(image_chw=image, step=pca_step)  # Get 3-channel PCA image
            target_h, target_w = label_mask.shape[-2], label_mask.shape[-1]

            if (H_emb, W_emb) != (target_h, target_w):
                image_for_plot = resize_hwc(image_for_plot, (target_h, target_w))  # Resize to target size

        else:
            image_for_plot = to_rgb(image_chw=image, rgb_indices=self.rgb_indices)

        showing_predictions = "prediction" in sample
        prediction_mask = None
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, Tensor):
                prediction_mask = prediction_mask.detach().cpu().numpy()

        kwargs = {}
        if hasattr(self, "num_classes"):
            kwargs["num_classes"] = self.num_classes
            kwargs["class_names"] = getattr(self, "class_names", None)

        return self._plot_sample(
            image=image_for_plot,
            label=label_mask,
            prediction=prediction_mask,
            suptitle=suptitle,
            show_axes=show_axes,
            embedding_input=embedding_input,
            **kwargs,
        )


class GenericNonGeoSegmentationDataset(GenericPixelWiseDataset):
    """GenericNonGeoSegmentationDataset"""

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[str] | None = None,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        class_names: list[str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        embedding_input: bool = False,
        pca_step: int = 4,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        tortilla_df: "pd.DataFrame | TacoDataset | None" = None,
        tortilla_indices: list[Hashable] | None = None,
        return_georeference: bool = False,
    ) -> None:
        """See :class:`GenericPixelWiseDataset` for shared args.

        Adds:
            num_classes (int): Number of classes in the dataset
            class_names (list[str], optional): Class names. Defaults to None.
        """

        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            ignore_split_file_extensions=ignore_split_file_extensions,
            allow_substring_split_file=allow_substring_split_file,
            rgb_indices=rgb_indices,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            embedding_input=embedding_input,
            pca_step=pca_step,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
            tortilla_df=tortilla_df,
            tortilla_indices=tortilla_indices,
            return_georeference=return_georeference,
        )
        self.num_classes = num_classes
        self.class_names = class_names

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        if "mask" in item:
            item["mask"] = item["mask"].long()
        return item

    @staticmethod
    def _plot_sample(
        image,
        label,
        num_classes: int,
        prediction=None,
        suptitle=None,
        class_names=None,
        show_axes=False,
        embedding_input=False,
    ):
        num_images = 5 if prediction is not None else 4
        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
        axes_visibility = "on" if show_axes else "off"

        # for legend
        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=num_classes - 1)
        ax[1].axis(axes_visibility)
        ax[1].set_title("Embedding PCA" if embedding_input else "RGB Image")
        ax[1].imshow(image)

        ax[2].axis(axes_visibility)
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(label, cmap="jet", norm=norm)

        ax[3].axis(axes_visibility)
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(label, cmap="jet", alpha=0.5 if embedding_input else 0.4, norm=norm)

        if prediction is not None:
            ax[4].axis(axes_visibility)
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction, cmap="jet", norm=norm)

        cmap = plt.get_cmap("jet")
        legend_data = []
        for i, _ in enumerate(range(num_classes)):
            class_name = class_names[i] if class_names else str(i)
            data = [i, cmap(norm(i)), class_name]
            legend_data.append(data)
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [n for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class GenericNonGeoPixelwiseRegressionDataset(GenericPixelWiseDataset):
    """GenericNonGeoPixelwiseRegressionDataset"""

    def __init__(
        self,
        data_root: Path,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*",
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[int] | None = None,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        embedding_input: bool = False,
        pca_step: int = 4,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
    ) -> None:
        """See :class:`GenericPixelWiseDataset` for args."""
        super().__init__(
            data_root,
            label_data_root=label_data_root,
            image_grep=image_grep,
            label_grep=label_grep,
            split=split,
            ignore_split_file_extensions=ignore_split_file_extensions,
            allow_substring_split_file=allow_substring_split_file,
            rgb_indices=rgb_indices,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            embedding_input=embedding_input,
            pca_step=pca_step,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        if "mask" in item:
            item["mask"] = item["mask"].float()
        return item

    @staticmethod
    def _plot_sample(image, label, prediction=None, suptitle=None, show_axes=False, embedding_input=False, **kwargs):
        num_images = 4 if prediction is not None else 3
        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
        axes_visibility = "on" if show_axes else "off"

        norm = mpl.colors.Normalize(vmin=label.min(), vmax=label.max())
        ax[0].axis(axes_visibility)
        ax[0].title.set_text("Embedding PCA" if embedding_input else "RGB Image")
        ax[0].imshow(image)

        ax[1].axis(axes_visibility)
        ax[1].title.set_text("Ground Truth Mask")
        ax[1].imshow(label, cmap="Greens", norm=norm)

        ax[2].axis(axes_visibility)
        ax[2].title.set_text("GT Mask on Image")
        ax[2].imshow(image)
        ax[2].imshow(label, cmap="Greens", alpha=0.3, norm=norm)
        # ax[2].legend()

        if prediction is not None:
            ax[3].axis(axes_visibility)
            ax[3].title.set_text("Predicted Mask")
            ax[3].imshow(prediction, cmap="Greens", norm=norm)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
