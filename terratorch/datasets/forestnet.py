import datetime
import glob
import json
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torchgeo.datasets import NonGeoDataset

from terratorch.datasets.utils import default_transform, validate_bands


class ForestNetNonGeo(NonGeoDataset):
    """NonGeo dataset implementation for [ForestNet](https://huggingface.co/datasets/ibm-nasa-geospatial/ForestNet)."""

    all_band_names = ("RED", "GREEN", "BLUE", "NIR", "SWIR_1", "SWIR_2")

    rgb_bands = (
        "RED",
        "GREEN",
        "BLUE",
    )

    splits = ("train", "test", "val")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    default_label_map = {  # noqa: RUF012
        "Plantation": 0,
        "Smallholder agriculture": 1,
        "Grassland shrubland": 2,
        "Other": 3,
    }

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        label_map: dict[str, int] = default_label_map,
        transform: A.Compose | None = None,
        fraction: float = 1.0,
        bands: Sequence[str] = BAND_SETS["all"],
        use_metadata: bool = False,
    ) -> None:
        """
        Initialize the ForestNetNonGeo dataset.

        Args:
            data_root (str): Path to the data root directory.
            split (str): One of 'train', 'val', or 'test'.
            label_map (Dict[str, int]): Mapping from label names to integer labels.
            transform: Transformations to be applied to the images.
            fraction (float): Fraction of the dataset to use. Defaults to 1.0 (use all data).
        """
        super().__init__()
        if split not in self.splits:
            msg = f"Incorrect split '{split}', please choose one of {list(self.splits)}."
            raise ValueError(msg)
        self.split = split

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        self.band_indices = [self.all_band_names.index(b) for b in bands]

        self.use_metadata = use_metadata

        self.data_root = Path(data_root)
        self.label_map = label_map

        # Load the CSV file corresponding to the split
        csv_file = self.data_root / f"{split}_filtered.csv"
        original_df = pd.read_csv(csv_file)

        # Apply stratified sampling if fraction < 1.0
        if fraction < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - fraction, random_state=47)
            stratified_indices, _ = next(sss.split(original_df, original_df["merged_label"]))
            self.dataset = original_df.iloc[stratified_indices].reset_index(drop=True)
        else:
            self.dataset = original_df

        self.transform = transform if transform else default_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_coords(self, event_path: Path) -> torch.Tensor:
        auxiliary_path = event_path / "auxiliary"
        osm_json_path = auxiliary_path / "osm.json"

        with open(osm_json_path) as f:
            osm_data = json.load(f)
            lat = float(osm_data["closest_city"]["lat"])
            lon = float(osm_data["closest_city"]["lon"])
            lat_lon = np.asarray([lat, lon])

        return torch.tensor(lat_lon, dtype=torch.float32)

    def _get_dates(self, image_files: list) -> list:
        dates = []
        pattern = re.compile(r"(\d{4})_(\d{2})_(\d{2})_cloud_\d+\.(png|npy)")
        for img_path in image_files:
            match = pattern.search(img_path)
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            date_obj = datetime.datetime(year, month, day)  # noqa: DTZ001
            julian_day = date_obj.timetuple().tm_yday
            date_tensor = torch.tensor([year, julian_day], dtype=torch.int32)
            dates.append(date_tensor)
        return torch.stack(dates, dim=0)

    def __getitem__(self, index: int):
        path = self.data_root / self.dataset["example_path"][index]
        label = self.map_label(index)

        visible_images, infrared_images, temporal_coords = self._load_images(path)

        visible_images = np.stack(visible_images, axis=0)
        infrared_images = np.stack(infrared_images, axis=0)
        merged_images = np.concatenate([visible_images, infrared_images], axis=-1)
        merged_images = merged_images[..., self.band_indices]  # (T, H, W, 2C)
        output = {"image": merged_images.astype(np.float32)}

        if self.transform:
            output = self.transform(**output)

        if self.use_metadata:
            location_coords = self._get_coords(path)
            output["location_coords"] = location_coords
            output["temporal_coords"] = temporal_coords

        output["label"] = label

        return output

    def _load_images(self, path: str):
        """Load visible and infrared images from the given event path"""
        visible_image_files = glob.glob(os.path.join(path, "images/visible/*_cloud_*.png"))
        infra_image_files = glob.glob(os.path.join(path, "images/infrared/*_cloud_*.npy"))

        selected_visible_images = self.select_images(visible_image_files)
        selected_infra_images = self.select_images(infra_image_files)

        dates = None
        if self.use_metadata:
            dates = self._get_dates(selected_visible_images)

        vis_images = [np.array(Image.open(img)) for img in selected_visible_images]  # (T, H, W, C)
        inf_images = [np.load(img, allow_pickle=False) for img in selected_infra_images]  # (T, H, W, C)
        return vis_images, inf_images, dates

    def least_cloudy_image(self, image_files):
        pattern = re.compile(r"(\d{4})_\d{2}_\d{2}_cloud_(\d+)\.(png|npy)")
        lowest_cloud_images = defaultdict(lambda: {"path": None, "cloud_value": float("inf")})

        for path in image_files:
            match = pattern.search(path)
            if match:
                year, cloud_value = match.group(1), int(match.group(2))
                if cloud_value < lowest_cloud_images[year]["cloud_value"]:
                    lowest_cloud_images[year] = {"path": path, "cloud_value": cloud_value}

        return [info["path"] for info in lowest_cloud_images.values()]

    def match_timesteps(self, image_files, selected_images):
        if len(selected_images) < 3:
            extra_imgs = [img for img in image_files if img not in selected_images]
            selected_images += extra_imgs[: 3 - len(selected_images)]

        while len(selected_images) < 3:
            selected_images.append(selected_images[-1])
        return selected_images[:3]

    def select_images(self, image_files):
        selected = self.least_cloudy_image(image_files)
        return self.match_timesteps(image_files, selected)

    def map_label(self, index: int) -> torch.Tensor:
        """Map the label name to an integer label."""
        label_name = self.dataset["merged_label"][index]
        label = self.label_map[label_name]
        return label

    def plot(self, sample: dict[str, torch.Tensor], suptitle: str | None = None):

        num_images = sample["image"].shape[1] + 1

        rgb_indices = [self.bands.index(band) for band in self.rgb_bands if band in self.bands]
        if len(rgb_indices) != 3:
            msg = "Dataset doesn't contain some of the RGB bands"
            raise ValueError(msg)

        fig, ax = plt.subplots(1, num_images, figsize=(15, 5))

        for i in range(sample["image"].shape[1]):
            image = sample["image"][:, i, :, :]
            if torch.is_tensor(image):
                image = image.permute(1, 2, 0).numpy()
            rgb_image = image[..., rgb_indices]
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
            rgb_image = np.clip(rgb_image, 0, 1)
            ax[i].imshow(rgb_image)
            ax[i].axis("off")
            ax[i].set_title(f"Timestep {i + 1}")

        legend_handles = [Rectangle((0, 0), 1, 1, color="blue")]
        legend_label = [self.label_map.get(sample["label"], "Unknown Label")]
        ax[-1].legend(legend_handles, legend_label, loc="center")
        ax[-1].axis("off")

        if suptitle:
            plt.suptitle(suptitle)

        plt.tight_layout()
        return fig
