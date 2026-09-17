# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Module containing generic object detection datamodule."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import albumentations as A
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import MisconfigurationException, NonGeoDataModule

from terratorch.datamodules.utils import Normalize, wrap_in_compose_is_list
from terratorch.datasets import HLSBands
from terratorch.datasets.generic_nongeo_od_dataset import GenericNonGeoObjectDetectionDataset
from terratorch.io.file import load_from_file_or_attribute

logger = logging.getLogger("terratorch")


class GenericNonGeoObjectDetectionDataModule(NonGeoDataModule):
    """
    Generic object detection datamodule for arbitrary geotiff images with JSON bbox annotations.
    
    Mirrors GenericNonGeoSegmentationDataModule but for object detection.
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
        label_grep: str = "*.json",
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
        constant_scale: float = 1,
        rgb_indices: list[int] | None = None,
        train_transform: list[Any] | None = None,
        val_transform: list[Any] | None = None,
        test_transform: list[Any] | None = None,
        no_data_replace: float | None = None,
        drop_last: bool = True,
        pin_memory: bool = False,
        class_names: list[str] | None = None,
        label_offset: int = 0,
        **kwargs: Any,
    ) -> None:
        """Constructor for GenericNonGeoObjectDetectionDataModule.

        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            num_classes: Number of object classes
            train_data_root: Path to training images
            val_data_root: Path to validation images
            test_data_root: Path to test images
            img_grep: Glob pattern for finding images
            label_grep: Glob pattern for finding label files
            means: Per-channel means for normalization
            stds: Per-channel standard deviations for normalization
            predict_data_root: Path to prediction images
            train_label_data_root: Path to training labels
            val_label_data_root: Path to validation labels
            test_label_data_root: Path to test labels
            train_split: Path to train split file
            val_split: Path to validation split file
            test_split: Path to test split file
            ignore_split_file_extensions: Whether to ignore file extensions in split files
            allow_substring_split_file: Whether split files contain substrings or exact matches
            dataset_bands: Bands present in the dataset
            output_bands: Bands to output from the dataset
            constant_scale: Scale factor for image values
            rgb_indices: Indices of RGB channels for visualization
            train_transform: Albumentations transforms for training
            val_transform: Albumentations transforms for validation
            test_transform: Albumentations transforms for testing
            no_data_replace: Value to replace NaN pixels with
            drop_last: Whether to drop the last incomplete batch
            pin_memory: Whether to pin memory for faster transfer to GPU
            class_names: List of class names for visualization
            label_offset: Integer offset added to every label value after loading.
                Use -1 to convert 1-indexed labels (1, 2, ...) to 0-indexed (0, 1, ...).
                Defaults to 0 (no change).
        """
        super().__init__(GenericNonGeoObjectDetectionDataset, batch_size, num_workers, **kwargs)

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
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.class_names = class_names
        self.label_offset = label_offset

        self.train_label_data_root = train_label_data_root
        self.val_label_data_root = val_label_data_root
        self.test_label_data_root = test_label_data_root

        self.dataset_bands = dataset_bands
        self.output_bands = output_bands
        self.rgb_indices = rgb_indices

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        # Normalization
        if means and stds:
            means = load_from_file_or_attribute(means)
            stds = load_from_file_or_attribute(stds)
            self.aug = Normalize(means, stds)
        else:
            self.aug = lambda x: x

    def detection_collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function for object detection.
        
        Stacks images into [B, C, H, W] but keeps boxes and labels as lists
        since each image may have a different number of objects.
        """
        images = torch.stack([b["image"] for b in batch])
        boxes = [b["boxes"] for b in batch]
        labels = [b["labels"] for b in batch]
        filenames = [b["filename"] for b in batch]

        return {
            "image": images,
            "boxes": boxes,
            "labels": labels,
            "filename": filenames,
        }

    def setup(self, stage: str) -> None:
        """Setup datasets for each stage."""
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                self.train_root,
                num_classes=self.num_classes,
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
                class_names=self.class_names,
                label_offset=self.label_offset,
            )

        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                self.val_root,
                num_classes=self.num_classes,
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
                class_names=self.class_names,
                label_offset=self.label_offset,
            )

        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                self.test_root,
                num_classes=self.num_classes,
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
                class_names=self.class_names,
                label_offset=self.label_offset,
            )

        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                self.predict_root,
                num_classes=self.num_classes,
                image_grep=self.img_grep,
                label_grep=self.label_grep,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                class_names=self.class_names,
                label_offset=self.label_offset,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[Dict[str, Tensor]]:
        """Create a dataloader for the specified split."""
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.detection_collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=self.pin_memory,
        )
