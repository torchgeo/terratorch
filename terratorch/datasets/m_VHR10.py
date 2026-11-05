from collections.abc import Callable
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import VHR10
from torchgeo.datasets.utils import (
    Path,
    check_integrity,
    download_and_extract_archive,
    download_url,
    lazy_import,
    quantile_normalization,
)


class mVHR10(VHR10):
    def __init__(
        self,
        second_level_split="train",
        second_level_split_proportions=(0.90, 0.05, 0.05),
        boxes_output_tag="boxes",
        labels_output_tag="labels",
        masks_output_tag="masks",
        scores_output_tag="scores",
        *args,
        **kwargs,
    ) -> None:
        """Initialize a new m-VHR-10 dataset instance. On the original VHR10 splits (positive or negative) it creates train/val/test splits.

        Args:
            second_level_split: either "train", "val", "test"
            second_level_split_proportions: proportion for train, val and test as a tuple. It has to sum to 1.
        Raises:
            AssertionError: if ``split`` argument is invalid
            AssertionError: if ``second_level_split_proportions`` is not composed of 3 elements and sums up to 1.
            DatasetNotFoundError: If dataset is not found and *download* is False.
            DependencyNotFoundError: if ``split="positive"`` and pycocotools is
                not installed.
        """
        super().__init__(*args, **kwargs)

        assert second_level_split in ["train", "val", "test"]
        assert len(second_level_split_proportions) == 3
        assert np.sum(second_level_split_proportions) == 1
        self.second_level_split = second_level_split
        df = pd.DataFrame({"ids": self.ids})
        df = df.sort_values("ids")
        df = df.sample(frac=1, random_state=123)
        index_train_end = int(df.shape[0] * second_level_split_proportions[0])
        index_val_end = int(df.shape[0] * (second_level_split_proportions[0] + second_level_split_proportions[1]))

        self.boxes_output_tag = boxes_output_tag
        self.labels_output_tag = labels_output_tag
        self.masks_output_tag = masks_output_tag
        self.scores_output_tag = scores_output_tag

        if self.second_level_split == "train":
            self.ids = list(df["ids"].values[:index_train_end])
        elif self.second_level_split == "val":
            self.ids = list(df["ids"].values[index_train_end:index_val_end])
        else:
            self.ids = list(df["ids"].values[index_val_end:])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """

        id_ = self.ids[index] + 1

        sample: dict[str, Any] = {
            "image": self._load_image(id_),
            "label": self._load_target(id_),
        }

        if sample["label"]["annotations"]:
            sample = self.coco_convert(sample)
            sample[self.boxes_output_tag] = sample["label"]["boxes"]
            sample[self.masks_output_tag] = sample["label"]["masks"]
            sample[self.labels_output_tag] = sample["label"]["labels"]
            if self.labels_output_tag != "label":
                del sample["label"]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
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

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            AssertionError: if ``show_feats`` argument is invalid
            DependencyNotFoundError: If plotting masks and scikit-image is not installed.

        .. versionadded:: 0.4
        """
        assert show_feats in {"boxes", "masks", "both"}
        image = quantile_normalization(sample["image"].permute(1, 2, 0)).numpy()

        if self.split == "negative":
            fig, axs = plt.subplots(squeeze=False)
            axs[0, 0].imshow(image)
            axs[0, 0].axis("off")

            if suptitle is not None:
                plt.suptitle(suptitle)
            return fig

        if show_feats != "boxes":
            skimage = lazy_import("skimage")

        boxes = sample[self.boxes_output_tag].cpu().numpy()
        labels = sample[self.labels_output_tag].cpu().numpy()

        if self.masks_output_tag in sample:
            masks = [mask.squeeze().cpu().numpy() for mask in sample[self.masks_output_tag]]

        n_gt = len(boxes)

        ncols = 1
        show_predictions = "prediction_" + self.labels_output_tag in sample

        if show_predictions:
            show_pred_boxes = False
            show_pred_masks = False
            prediction_labels = sample["prediction_" + self.labels_output_tag].numpy()
            prediction_scores = sample["prediction_" + self.scores_output_tag].numpy()
            if "prediction_" + self.boxes_output_tag in sample:
                prediction_boxes = sample["prediction_" + self.boxes_output_tag].numpy()
                show_pred_boxes = True
            if "prediction_" + self.masks_output_tag in sample:
                prediction_masks = sample["prediction_" + self.masks_output_tag].numpy()
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
            if show_feats in {"masks", "both"} and self.masks_output_tag in sample:
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
