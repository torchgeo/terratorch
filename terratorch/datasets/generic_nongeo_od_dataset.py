# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Module containing generic object detection dataset class."""

import json
import logging
import os
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Dict, List, Optional

import albumentations as A
import matplotlib as mpl
import numpy as np
import pandas as pd
import rioxarray
import torch
import xarray as xr
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

logger = logging.getLogger("terratorch")


class GenericNonGeoObjectDetectionDataset(NonGeoDataset):
    """
    Generic object detection dataset for arbitrary geotiff images with JSON annotations.
    
    Mirrors GenericNonGeoSegmentationDataset but for bounding box object detection.
    Annotations are stored as per-image JSON files with format:
    {
        "boxes": [[x1, y1, x2, y2], ...],
        "labels": [class_idx, ...]
    }
    """

    def __init__(
        self,
        data_root: Path | None = None,
        num_classes: int = 1,
        label_data_root: Path | None = None,
        image_grep: str | None = "*",
        label_grep: str | None = "*.json",
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[int] | None = None,
        dataset_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        output_bands: list[HLSBands | int | tuple[int, int] | str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float | None = None,
        class_names: list[str] | None = None,
        label_offset: int = 0,
    ) -> None:
        """Constructor for GenericNonGeoObjectDetectionDataset.

        Args:
            data_root: Path to data root directory containing images
            num_classes: Number of object classes (required)
            label_data_root: Path to data root directory with JSON label files.
                If not specified, will use the same as for images.
            image_grep: Regular expression appended to data_root to find input images.
                Defaults to "*".
            label_grep: Regular expression appended to label_data_root to find label files.
                Defaults to "*.json".
            split: Path to file containing files to be used for this split.
                File should be a new-line separated prefixes contained in the desired files.
            ignore_split_file_extensions: Whether to disregard extensions when using the split file.
                Defaults to True.
            allow_substring_split_file: Whether the split file contains substrings
                that must be present in file names (True) or exact matches (False).
                Defaults to True.
            rgb_indices: Indices of RGB channels for visualization. Defaults to [0, 1, 2].
            dataset_bands: Bands present in the dataset. Defaults to None.
            output_bands: Bands that should be output by the dataset.
                Must match dataset_bands. Defaults to None.
            constant_scale: Factor to multiply image values by. Defaults to 1.
            transform: Albumentations transform to be applied. Should include
                BboxParams for box augmentation. Defaults to None.
            no_data_replace: Replace nan values in input images with this value.
                If None, does no replacement. Defaults to None.
            class_names: List of class names for plotting. Defaults to None.
            label_offset: Integer offset added to every label value after loading.
                Use -1 to convert 1-indexed labels (1, 2, ...) to 0-indexed (0, 1, ...).
                Defaults to 0 (no change).
        """
        super().__init__()

        if data_root is None:
            msg = "Please provide data_root"
            raise Exception(msg)

        self.data_root = Path(data_root)
        self.label_data_root = Path(label_data_root) if label_data_root is not None else self.data_root
        self.split_file = Path(split) if split is not None else None
        self.ignore_split_file_extensions = ignore_split_file_extensions
        self.allow_substring_split_file = allow_substring_split_file
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.label_offset = label_offset

        self.rgb_indices = rgb_indices if rgb_indices is not None else [0, 1, 2]
        self.dataset_bands = dataset_bands
        self.output_bands = output_bands

        if transform is None:
            transform = default_transform()
        self.transform = transform

        # Find all images
        self.image_filenames = sorted(self.data_root.glob(image_grep))
        self.image_filenames = [f for f in self.image_filenames if f.is_file()]

        logger.info(f"Found {len(self.image_filenames)} images in {self.data_root}")

        # Apply split if provided
        if self.split_file is not None:
            with open(self.split_file) as f:
                split_lines = f.readlines()
            valid_files = {line.strip() for line in split_lines}
            self.image_filenames = filter_valid_files(
                self.image_filenames,
                valid_files=valid_files,
                ignore_extensions=self.ignore_split_file_extensions,
                allow_substring=self.allow_substring_split_file,
            )
            logger.info(f"After split filter: {len(self.image_filenames)} images")

        # Find corresponding label files
        self.label_filenames = []
        for img_path in self.image_filenames:
            # Construct label filename by replacing extension with .json
            label_name = img_path.stem + ".json"
            label_path = self.label_data_root / label_name
            self.label_filenames.append(label_path)

        if len(self.image_filenames) == 0:
            msg = f"No images found in {self.data_root}"
            raise Exception(msg)

        # Band info
        if self.dataset_bands is not None:
            self.band_intervals = generate_bands_intervals(self.dataset_bands)
        else:
            self.band_intervals = None

    def __len__(self) -> int:
        return len(self.image_filenames)

    def _load_file(self, path: Path) -> np.ndarray:
        """Load a geotiff file as numpy array."""
        array = rioxarray.open_rasterio(path)
        if len(array.shape) == 2:
            array = array.expand_dims(0)
        return array.values

    def _load_labels(self, label_path: Path) -> tuple[list[list[float]], list[int]]:
        """Load JSON label file with bounding box annotations.
        
        Expected format:
        {
            "boxes": [[x1, y1, x2, y2], ...],
            "labels": [class_idx, ...]
        }
        
        Returns:
            boxes: list of [x1, y1, x2, y2]
            labels: list of class indices
        """
        if not label_path.exists():
            return [], []

        try:
            with open(label_path, "r") as f:
                data = json.load(f)
            boxes = data.get("boxes", [])
            labels = data.get("labels", [])
            return boxes, labels
        except Exception as e:
            logger.warning(f"Failed to load labels from {label_path}: {e}")
            return [], []

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a sample.
        
        Returns:
            Dictionary with keys:
                - image: Tensor of shape [C, H, W]
                - boxes: Tensor of shape [N, 4] with bounding boxes in pascal_voc format
                - labels: Tensor of shape [N] with class indices
                - filename: str
                - metadata: dict (only if georeference info available)
        """
        image_path = self.image_filenames[index]
        label_path = self.label_filenames[index]

        # Load image
        image = self._load_file(image_path)

        # Select bands
        if self.band_intervals is not None:
            image = image[self.band_intervals]
        if self.output_bands is not None and self.dataset_bands is not None:
            indices = []
            for out_band in self.output_bands:
                indices.append(self.dataset_bands.index(out_band))
            image = image[indices]

        # Handle no data
        if self.no_data_replace is not None:
            image = np.where(np.isnan(image), self.no_data_replace, image)

        # Scale
        image = image * self.constant_scale

        # Load labels
        boxes, labels = self._load_labels(label_path)

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            if self.label_offset != 0:
                labels = labels + self.label_offset
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        # Convert image to float32
        image = image.astype(np.float32)

        # Transpose from [C, H, W] to [H, W, C] for albumentations
        image = np.transpose(image, (1, 2, 0))
        
        # Prepare for albumentations
        sample = {
            "image": image,
            "boxes": boxes.numpy() if len(boxes) > 0 else np.array([]),
            "labels": labels.numpy() if len(labels) > 0 else np.array([]),
        }

        # Apply transform with bbox handling
        if self.transform is not None:
            sample = self.transform(**sample)

        # Convert to tensors
        image = sample["image"]  # already torch tensor from transform
        boxes = sample.get("boxes", [])
        labels = sample.get("labels", [])

        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes) if boxes.size > 0 else torch.zeros((0, 4))
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels) if labels.size > 0 else torch.zeros((0,), dtype=torch.long)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        if labels.numel() == 0:
            labels = torch.zeros((0,), dtype=torch.long)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "filename": image_path.name,
        }

    def _plot_sample(self, sample: dict) -> Figure:
        """Plot a single sample for visualization."""
        image = sample["image"].numpy()
        boxes = sample["boxes"].numpy() if torch.is_tensor(sample["boxes"]) else sample["boxes"]
        labels = sample["labels"].numpy() if torch.is_tensor(sample["labels"]) else sample["labels"]

        # Convert image to RGB for display
        if image.shape[0] >= 3:
            rgb = to_rgb(
                image,
                self.rgb_indices,
            )
        else:
            rgb = image[0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(rgb, cmap="gray" if image.shape[0] == 1 else None)

        # Draw boxes
        colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, self.num_classes))
        for box, label in zip(boxes, labels):
            if len(box) == 4:
                x1, y1, x2, y2 = box
                rect = Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=colors[label],
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, self.class_names[label], color=colors[label], fontsize=8)

        ax.set_title(f"Filename: {sample['filename']}")
        return fig
