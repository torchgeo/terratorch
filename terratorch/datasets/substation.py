# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Substation segmentation dataset."""

import glob
import os
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools
import pycocotools.coco
import requests
import skimage
import torch
from matplotlib import patches
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from torchgeo.datasets.errors import DatasetNotFoundError
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import Path, download_url, extract_archive


def download_file_from_presigned(url, target_folder, filename):

    target_path = os.path.join(target_folder, filename)
    if os.path.isfile(target_path):
        print(f"File already exists at: {target_path}. Skipping download.")
    else:
        # Stream download for large files
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


def convert_coco_poly_to_mask(segmentations: list[int], height: int, width: int) -> Tensor:
    """Convert coco polygons to mask tensor.

    Args:
        segmentations (List[int]): polygon coordinates
        height (int): image height
        width (int): image width

    Returns:
        Tensor: Mask tensor

    Raises:
        DependencyNotFoundError: If pycocotools is not installed.
    """
    masks = []
    for polygons in segmentations:
        rles = pycocotools.mask.frPyObjects(polygons, height, width)
        mask = pycocotools.mask.decode(rles)
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    masks_tensor = torch.stack(masks, dim=0)
    return masks_tensor


class ConvertCocoAnnotations:
    """Callable for converting the boxes, masks and labels into tensors.

    This is a modified version of ConvertCocoPolysToMask() from torchvision found in
    https://github.com/pytorch/vision/blob/v0.14.0/references/detection/coco_utils.py
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Converts MS COCO fields (boxes, masks & labels) from list of ints to tensors.

        Args:
            sample: Sample

        Returns:
            Processed sample
        """
        image = sample["image"]
        h, w = image.size()[-2:]
        target = sample["label"]

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        bboxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        categories = [obj["category_id"] for obj in anno]
        classes = torch.tensor(categories, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]

        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {"boxes": boxes, "labels": classes, "image_id": image_id}
        if masks.nelement() > 0:
            masks = masks[keep]
            target["masks"] = masks

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd
        return {"image": image, "label": target}


class Substation(NonGeoDataset):
    """Substation dataset.

    The `Substation <https://github.com/Lindsay-Lab/substation-seg>`__
    dataset is curated by TransitionZero and sourced from publicly
    available data repositories, including OpenSreetMap (OSM) and
    Copernicus Sentinel data. The dataset consists of Sentinel-2
    images from 27k+ locations; the task is to segment power-substations,
    which appear in the majority of locations in the dataset.
    Most locations have 4-5 images taken at different timepoints
    (i.e., revisits).

    Dataset Format:

    * .npz file for each datapoint

    Dataset Features:

    * 26,522 image-mask pairs stored as numpy files.
    * Data from 5 revisits for most locations.
    * Multi-temporal, multi-spectral images (13 channels) paired with masks,
      with a spatial resolution of 228x228 pixels

    If you use this dataset in your research, please cite the following:

    * https://doi.org/10.48550/arXiv.2409.17363
    """

    directory = "Substation"
    filename_images = "image_stack.tar.gz"
    filename_masks = "mask.tar.gz"
    filename_detection_labels = "annotations.json"
    url_for_images = "https://storage.googleapis.com/tz-ml-public/substation-over-10km2-csv-main-444e360fd2b6444b9018d509d0e4f36e/image_stack.tar.gz"
    url_for_masks = "https://storage.googleapis.com/tz-ml-public/substation-over-10km2-csv-main-444e360fd2b6444b9018d509d0e4f36e/mask.tar.gz"
    url_for_detection_labels = "annotations.json"
    categories = ("background", "substation")

    def __init__(
        self,
        root: Path,
        bands: Sequence[int],
        mask_2d: bool,
        timepoint_aggregation: str = "concat",
        download: bool = False,
        checksum: bool = False,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        num_of_timepoints: int = 4,
        use_timepoints: bool = False,
        mode: str = "segmentation",
        split: str = "train",
        dataset_version: str = "full",
        plot_indexes: Sequence[int] = [2, 1, 0],
    ) -> None:
        """Initialize the Substation.

        Args:
            root: Path to the directory containing the dataset.
            bands: Channels to use from the image.
            mask_2d: Whether to use a 2D mask.
            timepoint_aggregation: How to aggregate multiple timepoints.
            download: Whether to download the dataset if it is not found.
            checksum: Whether to verify the dataset after downloading.
            transforms: A transform takes input sample and returns a transformed version.
            num_of_timepoints: Number of timepoints to use for each image.
            use_timepoints: Whether to use multiple timepoints for each image.
            mode: Either segmentation or object_detection.
            split: Split to load (either train/val/test).
            dataset_version: Dataset version to select (either'full' or 'geobench') for '' for full dataset or 'substation_meta_splits_geobench.csv' for GEO-Bench subsampled dataset.
            plot_indexes: list of indexes to use for plotting.
        """

        self.root = root
        self.bands = bands
        self.mask_2d = mask_2d
        self.timepoint_aggregation = timepoint_aggregation
        self.download = download
        self.use_timepoints = use_timepoints
        self.checksum = checksum
        self.transforms = transforms
        self.image_dir = os.path.join(root, "image_stack")
        self.mode = mode
        if self.mode == "segmentation":
            self.mask_dir = os.path.join(root, "mask")
        self.num_of_timepoints = num_of_timepoints
        self._verify()
        self.image_filenames = pd.Series(sorted(os.listdir(self.image_dir)))

        self.split = split
        self.dataset_version = dataset_version
        if self.dataset_version == "full":
            self.meta_df = pd.read_csv(os.path.join(self.root, "substation_meta_splits_full.csv"))

        elif self.dataset_version == "geobench":
            self.meta_df = pd.read_csv(os.path.join(self.root, "substation_meta_splits_geobench.csv"))

        self.meta_df = self.meta_df[self.meta_df["split"].values == self.split]
        self.meta_df["image"] = [x.split("/")[-1] for x in self.meta_df["image"].values]

        if self.mode == "object_detection":
            self.coco = pycocotools.coco.COCO(os.path.join(self.root, self.filename_detection_labels))
            self.coco_convert = ConvertCocoAnnotations()

        self.image_filenames = self.meta_df["image"].values

        assert self.timepoint_aggregation in {"first", "last", "random", "concat", "median", "identity"}
        if self.use_timepoints == False:
            assert self.timepoint_aggregation in {"first", "last", "random"}

        self.plot_indexes = plot_indexes

    def _load_image(self, image_path):

        image = np.load(image_path)["arr_0"]

        # selecting channels
        image = image[:, self.bands, :, :]
        # handling multiple images across timepoints

        if self.use_timepoints:
            if image.shape[0] < self.num_of_timepoints:
                # Padding: cycle through existing timepoints
                padded_images = []
                for i in range(self.num_of_timepoints):
                    padded_images.append(image[i % image.shape[0]])
                image = np.stack(padded_images)
            elif image.shape[0] > self.num_of_timepoints:
                # Removal: take the most recent timepoints
                image = image[-self.num_of_timepoints :]

            if self.timepoint_aggregation == "concat":
                image = np.reshape(image, (-1, image.shape[2], image.shape[3]))  # (num_of_timepoints*channels,h,w)
            elif self.timepoint_aggregation == "median":
                image = np.median(image, axis=0)
            elif self.timepoint_aggregation == "identity":
                ## No changes
                image = image
        elif self.timepoint_aggregation == "first":
            image = image[0]
        elif self.timepoint_aggregation == "last":
            image = image[-1]
        elif self.timepoint_aggregation == "random":
            image = image[np.random.randint(image.shape[0])]

        tensor_image = torch.from_numpy(image)
        tensor_image = tensor_image.float()

        return tensor_image

    def _load_od_target(self, id_):
        """Load the annotations for a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the annotations
        """
        # Images in the "negative" image set have no annotations
        annot = []
        ann_ids = self.coco.getAnnIds(imgIds=self.meta_df["id"].values[id_])
        annot = self.coco.loadAnns(ann_ids)

        target = dict(image_id=id_, annotations=annot)

        return target

    def _load_segmentation_mask(self, mask_path):

        mask = np.load(mask_path)["arr_0"]
        mask[mask != 3] = 0
        mask[mask == 3] = 1
        mask = torch.from_numpy(mask).long()
        mask = mask.unsqueeze(dim=0)

        if self.mask_2d:
            mask_0 = 1.0 - mask
            mask = torch.concat([mask_0, mask], dim=0)
        mask = mask.squeeze()

        return mask

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Get an item from the dataset by index.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A dictionary containing the image and corresponding mask.
        """
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        image = self._load_image(image_path)

        if self.mode == "segmentation":
            mask_path = os.path.join(self.mask_dir, image_filename)
            mask = self._load_segmentation_mask(mask_path)
            sample = {"image": image, "mask": mask}

        if self.mode == "object_detection":
            od_label = self._load_od_target(index)
            sample: dict[str, Any] = {"image": image, "label": od_label}

            if sample["label"]["annotations"]:
                sample = self.coco_convert(sample)
                sample["class"] = sample["label"]["labels"]
                sample["boxes"] = sample["label"]["boxes"]
                sample["masks"] = sample["label"]["masks"].float()
                sample["labels"] = sample.pop("class")

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.image_filenames)

    def _plot_segmentation(
        self, sample: dict[str, Tensor], show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            A matplotlib Figure containing the rendered sample.
        """
        ncols = 2

        image = sample["image"]
        image = image[-1] if len(image.shape) == 4 else image
        # get rgb indexes from
        image = image[self.plot_indexes, :, :].permute(1, 2, 0).numpy()

        if image.mean() > 1:
            image = image / 10000

        image = np.clip(image, 0, 1)

        if self.mask_2d:
            mask = sample["mask"][0].squeeze(0).cpu().numpy()
        else:
            mask = sample["mask"].cpu().numpy()
        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = sample["prediction"].cpu().numpy()
            if self.mask_2d:
                prediction = prediction[0]
            ncols = 3

        fig, axs = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask, cmap="gray", interpolation="none")
        axs[1].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(prediction, cmap="gray", interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle:
            fig.suptitle(suptitle)

        return fig

    def _plot_object_detection(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        show_feats: str | None = "both",
        box_alpha: float = 0.7,
        mask_alpha: float = 0.7,
        confidence_score=0.5,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            show_titles: flag indicating whether to show titles above each panel
            show_feats: optional string to pick features to be shown: boxes, masks, both
            box_alpha: alpha value of box
            mask_alpha: alpha value of mask
            confidence_score:

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            AssertionError: if ``show_feats`` argument is invalid
            DependencyNotFoundError: If plotting masks and scikit-image is not installed.

        .. versionadded:: 0.4
        """
        assert show_feats in {"boxes", "masks", "both"}

        image = sample["image"]
        image = image[-1] if len(image.shape) == 4 else image
        # get indexes to plot
        image = image[self.plot_indexes, :, :].permute(1, 2, 0).numpy()

        if image.mean() > 1:
            image = image / 10000

        image = np.clip(image, 0, 1)

        boxes = sample["boxes"].cpu().numpy()
        labels = sample["labels"].cpu().numpy()

        if "masks" in sample:
            masks = [mask.squeeze().cpu().numpy() for mask in sample["masks"]]

        n_gt = len(boxes)

        ncols = 1
        show_predictions = "prediction_labels" in sample

        if show_predictions:
            show_pred_boxes = False
            show_pred_masks = False
            prediction_labels = sample["prediction_labels"].numpy()
            prediction_scores = sample["prediction_scores"].numpy()
            if "prediction_boxes" in sample:
                prediction_boxes = sample["prediction_boxes"].numpy()
                show_pred_boxes = True
            if "prediction_masks" in sample:
                prediction_masks = sample["prediction_masks"].numpy()
                show_pred_masks = True

            n_pred = len(prediction_labels)
            ncols += 1

        # Display image
        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 10, 13))
        axs[0, 0].imshow(image)
        axs[0, 0].axis("off")

        cm = plt.get_cmap("gist_rainbow")
        for i in range(n_gt):
            class_num = labels[i]
            color = cm(class_num / len(self.categories))

            # Add bounding boxes
            x1, y1, x2, y2 = boxes[i]
            if show_feats in {"boxes", "both"}:
                r = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    alpha=box_alpha,
                    linestyle="dashed",
                    edgecolor=color,
                    facecolor="none",
                )
                axs[0, 0].add_patch(r)

            # Add labels
            label = self.categories[class_num]
            caption = label
            axs[0, 0].text(x1, y1 - 8, caption, color="white", size=11, backgroundcolor="none")

            # Add masks
            if show_feats in {"masks", "both"} and "masks" in sample:
                mask = masks[i]
                contours = skimage.measure.find_contours(mask, 0.5)
                for verts in contours:
                    verts = np.fliplr(verts)
                    p = patches.Polygon(verts, facecolor=color, alpha=mask_alpha, edgecolor="white")
                    axs[0, 0].add_patch(p)

            if show_titles:
                axs[0, 0].set_title("Ground Truth")
        if show_predictions:
            axs[0, 1].imshow(image)
            axs[0, 1].axis("off")
            for i in range(n_pred):
                score = prediction_scores[i]
                if score < confidence_score:
                    continue

                class_num = prediction_labels[i]
                color = cm(class_num / len(self.categories))

                if show_pred_boxes:
                    # Add bounding boxes
                    x1, y1, x2, y2 = prediction_boxes[i]
                    r = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle="dashed",
                        edgecolor=color,
                        facecolor="none",
                    )
                    axs[0, 1].add_patch(r)

                    # Add labels
                    label = self.categories[class_num]
                    caption = f"{label} {score:.3f}"
                    axs[0, 1].text(
                        x1,
                        y1 - 8,
                        caption,
                        color="white",
                        size=11,
                        backgroundcolor="none",
                    )

                # Add masks
                if show_pred_masks:
                    mask = prediction_masks[i][0]
                    contours = skimage.measure.find_contours(mask, 0.5)
                    for verts in contours:
                        verts = np.fliplr(verts)
                        p = patches.Polygon(verts, facecolor=color, alpha=mask_alpha, edgecolor="white")
                        axs[0, 1].add_patch(p)

            if show_titles:
                axs[0, 1].set_title("Prediction")
        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()

        return fig

    def plot(
        self, sample: dict[str, Tensor], show_titles: bool = True, suptitle: str | None = None, **kwargs
    ) -> Figure:

        if self.mode == "segmentation":
            figure = self._plot_segmentation(sample, show_titles, suptitle)
        else:
            figure = self._plot_object_detection(sample, show_titles, suptitle, **kwargs)

        return figure

    def _extract(self) -> None:
        """Extract the dataset."""
        img_pathname = os.path.join(self.root, self.filename_images)
        extract_archive(img_pathname)

        if self.mode == "segmentation":
            mask_pathname = os.path.join(self.root, self.filename_masks)
            extract_archive(mask_pathname)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        image_path = os.path.join(self.image_dir, "*.npz")

        if self.mode == "segmentation":
            mask_path = os.path.join(self.mask_dir, "*.npz")
            if glob.glob(image_path) and glob.glob(mask_path):
                return
        elif self.mode == "object_detection":
            detection_labels_exist = os.path.exists(os.path.join(self.root, self.filename_detection_labels))
            if glob.glob(image_path) and detection_labels_exist:
                return

        # Check if the tar.gz files for images and masks have already been downloaded
        image_exists = os.path.exists(os.path.join(self.root, self.filename_images))

        if self.mode == "segmentation":
            mask_exists = os.path.exists(os.path.join(self.root, self.filename_masks))
            if image_exists and mask_exists:
                self._extract()
                return
        elif self.mode == "object_detection":
            if image_exists:
                self._extract()
                return

        # If dataset files are missing and download is not allowed, raise an error
        if not getattr(self, "download", True):
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset and extract it."""
        # Download and verify images
        download_url(
            self.url_for_images,
            self.root,
            filename=self.filename_images,
            md5="bc0e3d2565f30b1c84a0d4cf37d44be6" if self.checksum else None,
        )
        extract_archive(os.path.join(self.root, self.filename_images), self.root)

        # Download splits
        download_file_from_presigned(
            "https://ibm.box.com/shared/static/1wv1fiva5w722aka3r8dd0vrmly4zat0.csv",
            self.root,
            "substation_meta_splits_geobench.csv",
        )
        download_file_from_presigned(
            "https://ibm.box.com/shared/static/sgfsyewhtjpwvigu9wrsc8ia26t1ug9d.csv",
            self.root,
            "substation_meta_splits_full.csv",
        )

        if self.mode == "segmentation":
            # Download and verify masks
            download_url(
                self.url_for_masks,
                self.root,
                filename=self.filename_masks,
                md5="919bb9599f47f44f17a1a4ecce56d81c" if self.checksum else None,
            )
            extract_archive(os.path.join(self.root, self.filename_masks), self.root)

        elif self.mode == "object_detection":
            download_file_from_presigned(
                "https://ibm.box.com/shared/static/sulkq5q5v3gthjf7o2jrpyn9kbw1uszd.json", self.root, "annotations.json"
            )
