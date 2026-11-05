from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as orig_transforms
from albumentations.pytorch import transforms as T
from matplotlib import patches
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets.utils import (
    Path,
    check_integrity,
    download_and_extract_archive,
    download_url,
    lazy_import,
)

from terratorch.datasets import mVHR10


def collate_fn_detection(batch, boxes_tag="boxes", labels_tag="labels", masks_tag="masks"):
    new_batch = {
        "image": [item["image"] for item in batch],
        boxes_tag: [item[boxes_tag] for item in batch],
        labels_tag: [item[labels_tag] for item in batch],
        masks_tag: [item[masks_tag] for item in batch],
    }
    return new_batch


def get_transform(train, image_size=896, pad=True, labels_tag="labels"):
    transforms = []
    if pad:
        transforms.append(A.PadIfNeeded(min_height=image_size, min_width=image_size, value=0, border_mode=0))
    else:
        transforms.append(A.Resize(height=image_size, width=image_size))
    if train:
        transforms.append(A.CenterCrop(width=image_size, height=image_size))
        transforms.append(A.HorizontalFlip(p=0.5))
    else:
        transforms.append(A.CenterCrop(width=image_size, height=image_size))
    transforms.append(T.ToTensorV2())
    print(labels_tag)
    return A.Compose(
        transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=[labels_tag]), is_check_shapes=False
    )


def apply_transforms(sample, transforms, boxes_tag="boxes", labels_tag="labels", masks_tag="masks"):

    sample["image"] = torch.stack(tuple(sample["image"]))
    sample["image"] = (
        sample["image"].permute(1, 2, 0) if len(sample["image"].shape) == 3 else sample["image"].permute(0, 2, 3, 1)
    )
    sample["image"] = np.array(sample["image"].cpu())
    sample[masks_tag] = [np.array(torch.stack(tuple(x)).cpu()) for x in sample[masks_tag]]
    sample[boxes_tag] = np.array(sample[boxes_tag].cpu())
    sample[labels_tag] = np.array(sample[labels_tag].cpu())
    transformed = transforms(
        image=sample["image"], masks=sample[masks_tag], bboxes=sample[boxes_tag], labels=sample[labels_tag]
    )
    transformed[boxes_tag] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
    transformed[labels_tag] = torch.tensor(transformed["labels"], dtype=torch.int64)
    transformed[masks_tag] = [x for x in transformed["masks"] if x.any()]

    del transformed["bboxes"]

    return transformed


class Normalize(Callable):
    def __init__(self, means, stds, max_pixel_value=None):
        super().__init__()
        self.means = means
        self.stds = stds
        self.max_pixel_value = max_pixel_value

    def __call__(self, batch):

        batch["image"] = torch.stack(tuple(batch["image"]))
        image = batch["image"] / self.max_pixel_value if self.max_pixel_value is not None else batch["image"]
        if len(image.shape) == 5:
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1, 1)
        elif len(image.shape) == 4:
            means = torch.tensor(self.means, device=image.device).view(1, -1, 1, 1)
            stds = torch.tensor(self.stds, device=image.device).view(1, -1, 1, 1)
        else:
            msg = f"Expected batch to have 5 or 4 dimensions, but got {len(image.shape)}"
            raise Exception(msg)
        batch["image"] = (image - means) / stds
        return batch


class IdentityTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class mVHR10DataModule(NonGeoDataModule):
    def __init__(
        self,
        root: Path = "data",
        split: str = "positive",
        download: bool = False,
        checksum: bool = False,
        second_level_split="train",
        second_level_split_proportions=(0.7, 0.15, 0.15),
        batch_size: int = 4,
        num_workers: int = 0,
        pad=True,
        image_size=896,
        collate_fn=None,
        boxes_output_tag="boxes",
        labels_output_tag="labels",
        masks_output_tag="masks",
        scores_output_tag="scores",
        apply_norm_in_datamodule=True,
        *args,
        **kwargs,
    ):

        super().__init__(
            mVHR10,
            batch_size=batch_size,
            num_workers=num_workers,
            root=root,
            split=split,
            download=download,
            checksum=checksum,
            second_level_split=second_level_split,
            second_level_split_proportions=second_level_split_proportions,
            boxes_output_tag=boxes_output_tag,
            labels_output_tag=labels_output_tag,
            masks_output_tag=masks_output_tag,
            scores_output_tag=scores_output_tag,
            **kwargs,
        )

        self.train_transform = partial(
            apply_transforms,
            transforms=get_transform(True, image_size, pad, "labels"),
            boxes_tag=boxes_output_tag,
            labels_tag=labels_output_tag,
            masks_tag=masks_output_tag,
        )
        self.val_transform = partial(
            apply_transforms,
            transforms=get_transform(False, image_size, pad, "labels"),
            boxes_tag=boxes_output_tag,
            labels_tag=labels_output_tag,
            masks_tag=masks_output_tag,
        )
        self.test_transform = partial(
            apply_transforms,
            transforms=get_transform(False, image_size, pad, "labels"),
            boxes_tag=boxes_output_tag,
            labels_tag=labels_output_tag,
            masks_tag=masks_output_tag,
        )

        if apply_norm_in_datamodule:
            self.aug = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=255)
        else:
            self.aug = Normalize((0, 0, 0), (1, 1, 1), max_pixel_value=255)

        self.root = root
        self.split = split
        self.second_level_split = second_level_split
        self.second_level_split_proportions = second_level_split_proportions
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = (
            partial(
                collate_fn_detection,
                boxes_tag=boxes_output_tag,
                labels_tag=labels_output_tag,
                masks_tag=masks_output_tag,
            )
            if collate_fn is None
            else collate_fn
        )
        self.download = download
        self.checksum = checksum
        self.boxes_output_tag = boxes_output_tag
        self.labels_output_tag = labels_output_tag
        self.masks_output_tag = masks_output_tag
        self.scores_output_tag = scores_output_tag

    def setup(self, stage: str) -> None:

        if stage in ["fit"]:
            self.train_dataset = mVHR10(
                root=self.root,
                split=self.split,
                transforms=self.train_transform,
                download=self.download,
                checksum=self.checksum,
                second_level_split="train",
                second_level_split_proportions=self.second_level_split_proportions,
                boxes_output_tag=self.boxes_output_tag,
                labels_output_tag=self.labels_output_tag,
                masks_output_tag=self.masks_output_tag,
                scores_output_tag=self.scores_output_tag,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = mVHR10(
                root=self.root,
                split=self.split,
                transforms=self.val_transform,
                download=self.download,
                checksum=self.checksum,
                second_level_split="val",
                second_level_split_proportions=self.second_level_split_proportions,
                boxes_output_tag=self.boxes_output_tag,
                labels_output_tag=self.labels_output_tag,
                masks_output_tag=self.masks_output_tag,
                scores_output_tag=self.scores_output_tag,
            )

        if stage in ["test"]:
            self.test_dataset = mVHR10(
                root=self.root,
                split=self.split,
                transforms=self.test_transform,
                download=self.download,
                checksum=self.checksum,
                second_level_split="test",
                second_level_split_proportions=self.second_level_split_proportions,
                boxes_output_tag=self.boxes_output_tag,
                labels_output_tag=self.labels_output_tag,
                masks_output_tag=self.masks_output_tag,
                scores_output_tag=self.scores_output_tag,
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
        batch_size = self.batch_size

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
