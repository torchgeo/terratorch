# Copyright contributors to the Terratorch project

"""
This module contains generic data modules for instantiation at runtime.
"""

import logging
import os
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional

import albumentations as A
import kornia.augmentation as K
import numpy as np
import torch
from albumentations.core.composition import BaseCompose, Compose

try:
    import tacoreader
    import tacoreader.v1 as _tacoreader_v1
    from tacoreader._legacy import is_legacy_format as _is_legacy_format
    from tacoreader.dataset import TacoDataset as _TacoDataset

    HAS_TACOREADER = True
except ImportError:
    HAS_TACOREADER = False
    tacoreader = None  # type: ignore
    _tacoreader_v1 = None  # type: ignore
    _is_legacy_format = None  # type: ignore
    _TacoDataset = None  # type: ignore
from kornia.augmentation import AugmentationSequential
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import MisconfigurationException, NonGeoDataModule

from terratorch.datamodules.utils import Normalize, wrap_in_compose_is_list
from terratorch.datasets import GenericNonGeoPixelwiseRegressionDataset, GenericNonGeoSegmentationDataset, HLSBands
from terratorch.io.file import load_from_file_or_attribute

from .utils import check_dataset_stackability

logger = logging.getLogger("terratorch")


# def collate_fn_list_dicts(batch):
#     metadata = []
#     for i in range(len(batch)):
#         metadata.append(batch[i].pop("metadata"))
#     batch = default_collate(batch)
#     # print(len(batch["image"]), len(batch["mask"]))
#     batch["metadata"] = metadata
#     # print(len(batch["metadata"]))
#     if len(batch["image"]) != len(batch["metadata"]):
#         print(len(batch["image"]), len(batch["metadata"]))
#     return batch


class GenericNonGeoSegmentationDataModule(NonGeoDataModule):
    """
    This is a generic datamodule class for instantiating data modules at runtime.
    Composes several [GenericNonGeoSegmentationDatasets][terratorch.datasets.GenericNonGeoSegmentationDataset]
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_classes: int,
        train_data_root: Path | None = None,
        val_data_root: Path | None = None,
        test_data_root: Path | None = None,
        img_grep: str = "*",
        label_grep: str = "*",
        means: list[float] | str | None = None,
        stds: list[float] | str | None = None,
        predict_data_root: Path | None = None,
        train_label_data_root: Path | None = None,
        val_label_data_root: Path | None = None,
        test_label_data_root: Path | None = None,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        predict_dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        predict_output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        rgb_indices: list[int] | None = None,
        train_transform: list[Any] | None = None,
        val_transform: list[Any] | None = None,
        test_transform: list[Any] | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        embedding_input: bool = False,
        pca_step: int = 4,
        drop_last: bool = True,
        pin_memory: bool = False,
        check_stackability: bool = True,
        tortilla_file: Path | None = None,
        return_georeference: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructor

        Args:
            batch_size (int): _description_
            num_workers (int): _description_
            train_data_root (Path): _description_
            val_data_root (Path): _description_
            test_data_root (Path): _description_
            predict_data_root (Path): _description_
            img_grep (str): _description_
            label_grep (str): _description_
            means (list[float]): _description_
            stds (list[float]): _description_
            num_classes (int): _description_
            train_label_data_root (Path | None, optional): _description_. Defaults to None.
            val_label_data_root (Path | None, optional): _description_. Defaults to None.
            test_label_data_root (Path | None, optional): _description_. Defaults to None.
            train_split (Path | None, optional): _description_. Defaults to None.
            val_split (Path | None, optional): _description_. Defaults to None.
            test_split (Path | None, optional): _description_. Defaults to None.
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            dataset_bands (list[HLSBands | int] | None): Bands present in the dataset. Defaults to None.
            output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
                Naming must match that of dataset_bands. Defaults to None.
            predict_dataset_bands (list[HLSBands | int] | None): Overwrites dataset_bands
                with this value at predict time.
                Defaults to None, which does not overwrite.
            predict_output_bands (list[HLSBands | int] | None): Overwrites output_bands
                with this value at predict time. Defaults to None, which does not overwrite.
            constant_scale (float, optional): _description_. Defaults to 1.
            rgb_indices (list[int] | None, optional): _description_. Defaults to None.
            train_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            val_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            test_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value. If none, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value. If none, does no replacement. Defaults to None.
            embedding_input (bool): Whether the input represents embeddings rather than an image, used for plotting. Defaults to False.
            pca_step (int): Spatial subsampling factor for PCA fitting in embedding visualizations.
                PCA components are estimated using only every pca_step-th spatial embedding
                (e.g. pca_step=4 uses 1/4 of embeddings), then applied to all embeddings. Defaults to 4.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            drop_last (bool): Drop the last batch if it is not complete. Defaults to True.
            pin_memory (bool): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them. Defaults to False.
            check_stackability (bool): Check if all the files in the dataset has the same size and can be stacked.
            tortilla_file (Path | None): Path to a tortilla file. If provided, the dataset will be loaded from the tortilla file. Defaults to None.
            return_georeference (bool): Whether to return georeference metadata info (CRS, Bounds, ...). Defaults to False.

        """
        super().__init__(GenericNonGeoSegmentationDataset, batch_size, num_workers, **kwargs)
        self.num_classes = num_classes
        self.img_grep = img_grep
        self.label_grep = label_grep
        self.train_root = train_data_root
        self.val_root = val_data_root
        self.test_root = test_data_root
        self.predict_root = predict_data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.ignore_split_file_extensions = ignore_split_file_extensions
        self.allow_substring_split_file = allow_substring_split_file
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.drop_last = drop_last
        self.pin_memory = pin_memory

        self.train_label_data_root = train_label_data_root
        self.val_label_data_root = val_label_data_root
        self.test_label_data_root = test_label_data_root

        self.dataset_bands = dataset_bands
        self.predict_dataset_bands = predict_dataset_bands if predict_dataset_bands else dataset_bands
        self.predict_output_bands = predict_output_bands if predict_output_bands else output_bands
        self.output_bands = output_bands
        self.rgb_indices = rgb_indices
        self.embedding_input = embedding_input
        self.pca_step = pca_step
        self.expand_temporal_dimension = expand_temporal_dimension
        self.reduce_zero_label = reduce_zero_label
        self.return_georeference = return_georeference

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        # self.aug = AugmentationSequential(
        #     K.Normalize(means, stds),
        #     data_keys=["image"],
        # )
        if means and stds:
            means = load_from_file_or_attribute(means)
            stds = load_from_file_or_attribute(stds)

            self.aug = Normalize(means, stds)
        else:
            self.aug = lambda x: x

        # self.aug = Normalize(means, stds)
        # self.collate_fn = collate_fn_list_dicts

        self.check_stackability = check_stackability

        if tortilla_file is not None and any([train_data_root, val_data_root, test_data_root]):
            raise MisconfigurationException(
                "Cannot provide both tortilla_file and train_data_root/val_data_root/test_data_root. "
                "Use tortilla_file OR data roots (not both)."
            )

        if tortilla_file is not None and not HAS_TACOREADER:
            raise ImportError(
                "tacoreader is required to use tortilla files. Install it with: pip install terratorch[tortilla]"
            )

        if tortilla_file is not None:
            path_str = str(tortilla_file)
            # Legacy v1 formats (.tortilla, .taco) require tacoreader.v1.load
            if _is_legacy_format is not None and _is_legacy_format(path_str):
                if _tacoreader_v1 is None:
                    raise ImportError(
                        f"Legacy tortilla file format detected ({path_str!r}). "
                        "tacoreader v2 is installed but tacoreader.v1 is not available. "
                        "Install tacoreader<1.0 to read legacy files, or migrate to .tacozip format."
                    )
                logger.warning(
                    "Loading legacy v1 tortilla file %r. Consider migrating to .tacozip format using tacotoolbox.",
                    path_str,
                )
                self.tortilla_df = _tacoreader_v1.load(path_str)  # type: ignore[union-attr]
            else:
                self.tortilla_df = tacoreader.load(path_str)  # type: ignore
        else:
            self.tortilla_df = None

    @staticmethod
    def _is_v2_taco_dataset(ds: Any) -> bool:
        """Return True if *ds* is a tacoreader v2 TacoDataset (not a pandas DataFrame)."""
        return _TacoDataset is not None and isinstance(ds, _TacoDataset)

    def _get_tortilla_indices(self, stage: str) -> list[Hashable] | None:
        if self.tortilla_df is None:
            return None

        # tacoreader v2: TacoDataset — materialise the top-level rows and filter
        # by the 'tortilla:data_split' column when it is present.
        if self._is_v2_taco_dataset(self.tortilla_df):
            import pyarrow.compute as pc

            stage_map = {"fit": "train", "validate": "validation", "test": "test"}
            split_label = stage_map.get(stage)
            if split_label is None:
                return None

            arrow_table = self.tortilla_df.data.to_arrow()  # type: ignore[union-attr]
            if "tortilla:data_split" not in arrow_table.column_names:
                # Dataset has no data_split column — return all row positions
                return list(range(arrow_table.num_rows))
            mask = pc.equal(arrow_table.column("tortilla:data_split"), split_label)
            return [i for i, keep in enumerate(mask.to_pylist()) if keep]

        # tacoreader v1: TortillaDataFrame (pandas DataFrame subclass)
        if stage in ["fit"]:
            return [i for i, row in self.tortilla_df.iterrows() if row["tortilla:data_split"] == "train"]
        if stage in ["validate"]:
            return [i for i, row in self.tortilla_df.iterrows() if row["tortilla:data_split"] == "validation"]
        if stage in ["test"]:
            return [i for i, row in self.tortilla_df.iterrows() if row["tortilla:data_split"] == "test"]

        return None

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                self.train_root,
                self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.train_label_data_root,
                split=self.train_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                embedding_input=self.embedding_input,
                pca_step=self.pca_step,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                tortilla_df=self.tortilla_df,
                tortilla_indices=self._get_tortilla_indices(stage),
                return_georeference=self.return_georeference,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                self.val_root,
                self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.val_label_data_root,
                split=self.val_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                embedding_input=self.embedding_input,
                pca_step=self.pca_step,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                tortilla_df=self.tortilla_df,
                tortilla_indices=self._get_tortilla_indices(stage),
                return_georeference=self.return_georeference,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                self.test_root,
                self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.test_label_data_root,
                split=self.test_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                embedding_input=self.embedding_input,
                pca_step=self.pca_step,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                tortilla_df=self.tortilla_df,
                tortilla_indices=self._get_tortilla_indices(stage),
                return_georeference=self.return_georeference,
            )
        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                self.predict_root,
                self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                dataset_bands=self.predict_dataset_bands,
                output_bands=self.predict_output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                embedding_input=self.embedding_input,
                pca_step=self.pca_step,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
                return_georeference=self.return_georeference,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        if self.check_stackability:
            logger.info(f"Checking stackability for {split} split.")
            batch_size = check_dataset_stackability(dataset, batch_size)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=self.pin_memory,
        )


class GenericNonGeoPixelwiseRegressionDataModule(NonGeoDataModule):
    """This is a generic datamodule class for instantiating data modules at runtime.
    Composes several
    [GenericNonGeoPixelwiseRegressionDataset][terratorch.datasets.GenericNonGeoPixelwiseRegressionDataset]
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_data_root: Path,
        val_data_root: Path,
        test_data_root: Path,
        means: list[float] | str | None = None,
        stds: list[float] | str | None = None,
        predict_data_root: Path | None = None,
        img_grep: str | None = "*",
        label_grep: str | None = "*",
        train_label_data_root: Path | None = None,
        val_label_data_root: Path | None = None,
        test_label_data_root: Path | None = None,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        predict_dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        predict_output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        rgb_indices: list[int] | None = None,
        train_transform: list[Any] | None = None,
        val_transform: list[Any] | None = None,
        test_transform: list[Any] | None = None,
        expand_temporal_dimension: bool = False,
        reduce_zero_label: bool = False,
        no_data_replace: float | None = None,
        no_label_replace: int | None = None,
        embedding_input: bool = False,
        pca_step: int = 4,
        drop_last: bool = True,
        pin_memory: bool = False,
        check_stackability: bool = True,
        **kwargs: Any,
    ) -> None:
        """Constructor

        Args:
            batch_size (int): _description_
            num_workers (int): _description_
            train_data_root (Path): _description_
            val_data_root (Path): _description_
            test_data_root (Path): _description_
            predict_data_root (Path): _description_
            img_grep (str): _description_
            label_grep (str): _description_
            means (list[float]): _description_
            stds (list[float]): _description_
            train_label_data_root (Path | None, optional): _description_. Defaults to None.
            val_label_data_root (Path | None, optional): _description_. Defaults to None.
            test_label_data_root (Path | None, optional): _description_. Defaults to None.
            train_split (Path | None, optional): _description_. Defaults to None.
            val_split (Path | None, optional): _description_. Defaults to None.
            test_split (Path | None, optional): _description_. Defaults to None.
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            dataset_bands (list[HLSBands | int] | None): Bands present in the dataset. Defaults to None.
            output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
                Naming must match that of dataset_bands. Defaults to None.
            predict_dataset_bands (list[HLSBands | int] | None): Overwrites dataset_bands
                with this value at predict time.
                Defaults to None, which does not overwrite.
            predict_output_bands (list[HLSBands | int] | None): Overwrites output_bands
                with this value at predict time. Defaults to None, which does not overwrite.
            constant_scale (float, optional): _description_. Defaults to 1.
            rgb_indices (list[int] | None, optional): _description_. Defaults to None.
            train_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            val_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            test_transform (Albumentations.Compose | None): Albumentations transform
                to be applied to the train dataset.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value. If none, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value. If none, does no replacement. Defaults to None.
            embedding_input (bool): Whether the input represents embeddings rather than an image, used for plotting. Defaults to False.
            pca_step (int): Spatial subsampling factor for PCA fitting in embedding visualizations.
                PCA components are estimated using only every pca_step-th spatial embedding
                (e.g. pca_step=4 uses 1/4 of embeddings), then applied to all embeddings. Defaults to 4.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to False.
            drop_last (bool): Drop the last batch if it is not complete. Defaults to True.
            pin_memory (bool): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them. Defaults to False.
            check_stackability (bool): Check if all the files in the dataset has the same size and can be stacked.
        """
        super().__init__(GenericNonGeoPixelwiseRegressionDataset, batch_size, num_workers, **kwargs)
        self.img_grep = img_grep
        self.label_grep = label_grep
        self.train_root = train_data_root
        self.val_root = val_data_root
        self.test_root = test_data_root
        self.predict_root = predict_data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.ignore_split_file_extensions = ignore_split_file_extensions
        self.allow_substring_split_file = allow_substring_split_file
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.embedding_input = embedding_input
        self.pca_step = pca_step
        self.expand_temporal_dimension = expand_temporal_dimension
        self.reduce_zero_label = reduce_zero_label

        self.train_label_data_root = train_label_data_root
        self.val_label_data_root = val_label_data_root
        self.test_label_data_root = test_label_data_root

        self.constant_scale = constant_scale

        self.dataset_bands = dataset_bands
        self.predict_dataset_bands = predict_dataset_bands if predict_dataset_bands else dataset_bands
        self.predict_output_bands = predict_output_bands if predict_output_bands else output_bands
        self.output_bands = output_bands
        self.rgb_indices = rgb_indices

        # self.aug = AugmentationSequential(
        #     K.Normalize(means, stds),
        #     data_keys=["image"],
        # )
        if means and stds:
            means = load_from_file_or_attribute(means)
            stds = load_from_file_or_attribute(stds)

            self.aug = Normalize(means, stds)
        else:
            self.aug = lambda x: x
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        self.check_stackability = check_stackability

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                self.train_root,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.train_label_data_root,
                split=self.train_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                embedding_input=self.embedding_input,
                pca_step=self.pca_step,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                self.val_root,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.val_label_data_root,
                split=self.val_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                embedding_input=self.embedding_input,
                pca_step=self.pca_step,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                self.test_root,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                label_data_root=self.test_label_data_root,
                split=self.test_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                embedding_input=self.embedding_input,
                pca_step=self.pca_step,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )

        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                self.predict_root,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                dataset_bands=self.predict_dataset_bands,
                output_bands=self.predict_output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
                embedding_input=self.embedding_input,
                pca_step=self.pca_step,
                expand_temporal_dimension=self.expand_temporal_dimension,
                reduce_zero_label=self.reduce_zero_label,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        if self.check_stackability:
            logger.info("Checking stackability.")
            batch_size = check_dataset_stackability(dataset, batch_size)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=self.pin_memory,
        )
