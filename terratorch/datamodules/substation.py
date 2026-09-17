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

import terratorch

# from terratorch.datasets import mVHR10
from terratorch.datasets.substation import Substation


def collate_fn_detection(batch):
    new_batch = {
        "image": [item["image"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "masks": [item["masks"] for item in batch],
    }
    return new_batch


def get_transform(train, image_size=228, n_timesteps=4):
    transforms = []
    if image_size != 228:
        transforms.append(A.Resize(height=image_size, width=image_size))
    if n_timesteps > 1:
        transforms.append(terratorch.datasets.transforms.FlattenTemporalIntoChannels())
    transforms.append(T.ToTensorV2())
    if n_timesteps > 1:
        transforms.append(terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=n_timesteps))
    return A.Compose(
        transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]), is_check_shapes=False
    )


def apply_transforms(sample, transforms):

    sample["image"] = torch.stack(tuple(sample["image"]))
    sample["image"] = (
        sample["image"].permute(1, 2, 0) if len(sample["image"].shape) == 3 else sample["image"].permute(0, 2, 3, 1)
    )
    sample["image"] = np.array(sample["image"].cpu())
    sample["masks"] = [np.array(torch.stack(tuple(x)).cpu()) for x in sample["masks"]]
    sample["boxes"] = np.array(sample["boxes"].cpu())
    sample["labels"] = np.array(sample["labels"].cpu())
    transformed = transforms(
        image=sample["image"], masks=sample["masks"], bboxes=sample["boxes"], labels=sample["labels"]
    )

    indexes = [i for i, x in enumerate(transformed["masks"]) if x.any()]
    transformed["masks"] = [x for i, x in enumerate(transformed["masks"]) if i in indexes]
    transformed["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
    transformed["boxes"] = transformed["boxes"][indexes]
    transformed["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64)[indexes]

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
            means = self.means.clone().detach().to(device=image.device).view(1, -1, 1, 1, 1)
            stds = self.stds.clone().detach().to(device=image.device).view(1, -1, 1, 1, 1)
        elif len(image.shape) == 4:
            means = self.means.clone().detach().to(device=image.device).view(1, -1, 1, 1)
            stds = self.stds.clone().detach().to(device=image.device).view(1, -1, 1, 1)
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


means = [
    1423.3745710003543,
    1219.2943391189588,
    1182.4678074598091,
    1158.0459084351446,
    1401.8725467408829,
    2198.8962678953585,
    2582.976482945787,
    2505.9577635634005,
    2812.9741141745803,
    808.0196180351925,
    15.818041869437652,
    2185.9854710710974,
    1469.6518299083612,
]


stds = [
    285.46691956407517,
    394.44991388670013,
    450.0629924473796,
    671.2598383733988,
    594.6671241798275,
    618.4261097309303,
    787.2213184558726,
    795.4738600694346,
    862.1076233925734,
    331.81864168987255,
    14.444442475684806,
    840.1079725056068,
    818.1891190853091,
]


class SubstationDataModule(NonGeoDataModule):
    def __init__(
        self,
        root: Path = "/opt/app-root/src/use-cases/substation-dataset-conversion/Substation",
        bands=[3, 2, 1],
        mask_2d: bool = False,
        batch_size: int = 8,
        num_workers: int = 4,
        means=means,
        stds=stds,
        timepoint_aggregation: str = "concat",
        download: bool = False,
        checksum: bool = False,
        num_of_timepoints: int = 4,
        use_timepoints: bool = True,
        mode: str = "segmentation",
        collate_fn=None,
        dataset_version: str = "full",
        image_size: int = 224,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        plot_indexes=[2, 1, 0],
        *args,
        **kwargs,
    ):

        super().__init__(
            SubstationDataModule,
            batch_size=batch_size,
            num_workers=num_workers,
            root=root,
            bands=bands,
            mask_2d=mask_2d,
            timepoint_aggregation=timepoint_aggregation,
            download=download,
            checksum=checksum,
            num_of_timepoints=num_of_timepoints,
            use_timepoints=use_timepoints,
            mode=mode,
            dataset_version=dataset_version,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            **kwargs,
        )

        if (use_timepoints == False) | (timepoint_aggregation == "median"):
            num_of_timepoints = 1
        if mode == "object_detection":
            self.train_transform = (
                partial(apply_transforms, transforms=get_transform(True, image_size, num_of_timepoints))
                if train_transform is None
                else train_transform
            )
            self.val_transform = (
                partial(apply_transforms, transforms=get_transform(False, image_size, num_of_timepoints))
                if val_transform is None
                else val_transform
            )
            self.test_transform = (
                partial(apply_transforms, transforms=get_transform(False, image_size, num_of_timepoints))
                if test_transform is None
                else test_transform
            )
        else:
            self.train_transform = train_transform
            self.val_transform = val_transform
            self.test_transform = test_transform

        self.bands = bands
        self.means = torch.tensor(means)[self.bands]
        self.stds = torch.tensor(stds)[self.bands]
        self.root = root
        self.aug = Normalize(self.means, self.stds)
        self.mask_2d = mask_2d
        self.timepoint_aggregation = timepoint_aggregation
        self.download = download
        self.checksum = checksum
        self.collate_fn = collate_fn_detection if collate_fn is None else collate_fn
        self.num_of_timepoints = num_of_timepoints
        self.use_timepoints = use_timepoints
        self.mode = mode
        self.dataset_version = dataset_version
        self.image_size = image_size
        self.plot_indexes = plot_indexes

    def setup(self, stage: str) -> None:

        if stage in ["fit"]:
            self.train_dataset = Substation(
                root=self.root,
                bands=self.bands,
                mask_2d=self.mask_2d,
                timepoint_aggregation=self.timepoint_aggregation,
                download=self.download,
                checksum=self.checksum,
                transforms=self.train_transform,
                num_of_timepoints=self.num_of_timepoints,
                use_timepoints=self.use_timepoints,
                mode=self.mode,
                split="train",
                dataset_version=self.dataset_version,
                plot_indexes=self.plot_indexes,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = Substation(
                root=self.root,
                bands=self.bands,
                mask_2d=self.mask_2d,
                timepoint_aggregation=self.timepoint_aggregation,
                download=self.download,
                checksum=self.checksum,
                transforms=self.train_transform,
                num_of_timepoints=self.num_of_timepoints,
                use_timepoints=self.use_timepoints,
                mode=self.mode,
                split="val",
                dataset_version=self.dataset_version,
                plot_indexes=self.plot_indexes,
            )
        if stage in ["test"]:
            self.test_dataset = Substation(
                root=self.root,
                bands=self.bands,
                mask_2d=self.mask_2d,
                timepoint_aggregation=self.timepoint_aggregation,
                download=self.download,
                checksum=self.checksum,
                transforms=self.train_transform,
                num_of_timepoints=self.num_of_timepoints,
                use_timepoints=self.use_timepoints,
                mode=self.mode,
                split="test",
                dataset_version=self.dataset_version,
                plot_indexes=self.plot_indexes,
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
