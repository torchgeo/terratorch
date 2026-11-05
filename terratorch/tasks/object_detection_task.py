# modified from https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/trainers/detection.html#ObjectDetectionTask
# and from https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/trainers/instance_segmentation.html#InstanceSegmentationTask

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for object detection."""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import RGBBandsMissingError
from torchgeo.tasks import BaseTask
from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms

from terratorch.datasets.utils import unbind_samples
from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory


def get_batch_size(x):
    if isinstance(x, torch.Tensor):
        batch_size = x.shape[0]
    elif isinstance(x, dict):
        batch_size = list(x.values())[0].shape[0]
    else:
        raise ValueError(f"Expect x to be torch.Tensor or dict, got {type(x)}")
    return batch_size


class ObjectDetectionTask(BaseTask):
    ignore = None
    monitor = "val_map"
    mode = "max"

    def __init__(
        self,
        model_factory: str,
        model_args: dict,
        lr: float = 0.001,
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        class_names: list[str] | None = None,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.5,
        boxes_field: str = "boxes",
        labels_field: str = "labels",
        masks_field: str = "masks",
        ignore_index: int = -1,
    ) -> None:
        """
        Initialize a new ObjectDetectionTask instance.

        Args:
            model_factory (str): Name of the model factory to use.
            model_args (dict): Arguments for the model factory.
            lr (float, optional): Learning rate for optimizer. Defaults to 0.001.
            optimizer (str | None, optional): Name of the optimizer to use. Defaults to None.
            optimizer_hparams (dict | None, optional): Hyperparameters for the optimizer. Defaults to None.
            scheduler (str | None, optional): Name of the scheduler to use. Defaults to None.
            scheduler_hparams (dict | None, optional): Hyperparameters for the scheduler. Defaults to None.
            freeze_backbone (bool, optional): Freeze the backbone network to fine-tune the detection head. Defaults to False.
            freeze_decoder (bool, optional): Freeze the decoder network to fine-tune the detection head. Defaults to False.
            class_names (list[str] | None, optional): List of class names. Defaults to None.
            iou_threshold (float, optional): Intersection over union threshold for evaluation. Defaults to 0.5.
            score_threshold (float, optional): Score threshold for evaluation. Defaults to 0.5.
            boxes_field (str): The field containing the bbox information in the sample. Deafult is boxes
            labels_field (str): The field containing the labels information in the sample. Deafult is labels
            masks_field (str): The field containing the masks information in the sample. Deafult is masks
            ignore_index (int): Index to be ignored when calculating the loss and metrics.

        Returns:
            None
        """
        warnings.warn(
            "The Object Detection Task has to be considered experimental. This is less mature than the other tasks and being further improved."
        )

        self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)
        self.model_args = model_args
        self.framework = model_args["framework"]
        _has_masks = self.framework == "mask-rcnn" or model_args.get("framework_masks", False)
        self.monitor = "val_segm_map" if _has_masks else self.monitor

        super().__init__()
        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler = LossHandler(self.test_metrics.prefix)
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.lr = lr
        if optimizer_hparams is not None:
            if "lr" in self.hparams["optimizer_hparams"].keys():
                self.lr = float(self.hparams["optimizer_hparams"]["lr"])
                del self.hparams["optimizer_hparams"]["lr"]

        self.boxes_field = boxes_field
        self.labels_field = labels_field
        self.masks_field = masks_field

        self.ignore_index = ignore_index

    def configure_models(self) -> None:
        """
        It instantiates the model and freezes/unfreezes the backbone and decoder networks.
        """

        self.model: Model = self.model_factory.build_model("object_detection", **self.hparams["model_args"])
        if self.hparams["freeze_backbone"]:
            self.model.freeze_encoder()
        if self.hparams["freeze_decoder"]:
            self.model.freeze_decoder()

    def configure_metrics(self) -> None:
        """
        Configure metrics for the task.
        """
        if self.framework == "mask-rcnn":
            metrics = MetricCollection({"mAP": MeanAveragePrecision(iou_type=("bbox", "segm"), average="macro")})
        _has_masks = self.framework == "mask-rcnn" or self.model_args.get("framework_masks", False)
        if _has_masks:
            metrics = MetricCollection({"mAP": MeanAveragePrecision(iou_type=("bbox", "segm"), average="macro")})
        else:
            metrics = MetricCollection({"mAP": MeanAveragePrecision(iou_type=("bbox"), average="macro")})

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        """
        Configure optimiser for the task.
        """
        optimizer = self.hparams["optimizer"]
        if optimizer is None:
            optimizer = "Adam"
        return optimizer_factory(
            optimizer,
            self.lr,
            self.parameters(),
            self.hparams["optimizer_hparams"],
            self.hparams["scheduler"],
            self.monitor,
            self.hparams["scheduler_hparams"],
        )

    def reformat_batch(self, batch: Any, batch_size: int):
        """
        Reformat batch to calculate loss and metrics.

        Args:
            batch: The output of your DataLoader.
            batch_size: Size of your batch
        Returns:
            Reformated batch
        """

        if ("masks" in batch.keys()) | ("mask" in batch.keys()) | (self.masks_field in batch.keys()):
            y = [
                {
                    "boxes": batch[self.boxes_field][i],
                    "labels": batch[self.labels_field][i],
                    "masks": torch.cat([x[None].to(torch.uint8) for x in batch[self.masks_field][i]]),
                }
                for i in range(batch_size)
            ]
        else:
            y = [
                {"boxes": batch[self.boxes_field][i], "labels": batch[self.labels_field][i]} for i in range(batch_size)
            ]

        return y

    def apply_ignore_index(self, batch, ignore_index):

        if ignore_index != -1:
            for i in range(len(batch[self.labels_field])):
                labels_unfiltered = batch[self.labels_field][i]
                batch[self.labels_field][i] = torch.cat(
                    [x[None] for x in batch[self.labels_field][i] if x != ignore_index]
                )
                batch[self.boxes_field][i] = torch.cat(
                    [x[None] for x, l in zip(batch[self.boxes_field][i], labels_unfiltered) if l != ignore_index]
                )
                if self.masks_field in batch.keys():
                    batch[self.masks_field][i] = torch.cat(
                        [x[None] for x, l in zip(batch[self.masks_field][i], labels_unfiltered) if l != ignore_index]
                    )

        return batch

    def apply_nms_sample(self, y_hat, iou_threshold=0.5, score_threshold=0.5):
        """
        It applies nms to a sample predictions of the model.

        Args:
            y_hat: Predictions dictionary.
            iou_threshold: IoU threshold for evaluation.
            score_threshold: Score threshold for evaluation.
        Returns:
            fintered predictions for a sample after applying nms batch
        """

        boxes, scores, labels = y_hat["boxes"], y_hat["scores"], y_hat["labels"]
        masks = y_hat["masks"] if "masks" in y_hat.keys() else None

        # Filter based on score threshold
        keep_score = scores > score_threshold
        boxes, scores, labels = boxes[keep_score], scores[keep_score], labels[keep_score]
        if masks is not None:
            masks = masks[keep_score]

        # Apply NMS
        keep_nms = nms(boxes, scores, iou_threshold)

        y_hat["boxes"], y_hat["scores"], y_hat["labels"] = boxes[keep_nms], scores[keep_nms], labels[keep_nms]

        if masks is not None:
            y_hat["masks"] = masks[keep_nms]

        return y_hat

    def apply_nms_batch(self, y_hat: Any, batch_size: int):
        """
        It applies nms to a batch predictions of the model.

        Args:
            y_hat: List of predictions dictionaries.
            iou_threshold: IoU threshold for evaluation.
            score_threshold: Score threshold for evaluation.
        Returns:
            fintered predictions for a batch after applying nms batch
        """

        for i in range(batch_size):
            y_hat[i] = self.apply_nms_sample(
                y_hat[i], iou_threshold=self.iou_threshold, score_threshold=self.score_threshold
            )

        return y_hat

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """
        Compute the training loss.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss dictionary.
        """

        x = batch["image"]
        batch_size = get_batch_size(x)
        batch = self.apply_ignore_index(batch, self.ignore_index)
        y = self.reformat_batch(batch, batch_size)
        loss_dict = self(x, y)
        if isinstance(loss_dict, dict) is False:
            loss_dict = loss_dict.output
        train_loss: Tensor = sum(loss_dict.values())
        self.log_dict(loss_dict, batch_size=batch_size)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Compute the validation metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """

        x = batch["image"]
        batch_size = get_batch_size(x)
        batch = self.apply_ignore_index(batch, self.ignore_index)
        y = self.reformat_batch(batch, batch_size)
        y_hat = self(x)
        if isinstance(y_hat, dict) is False:
            y_hat = y_hat.output

        y_hat = self.apply_nms_batch(y_hat, batch_size)

        if self.framework == "mask-rcnn":
            for i in range(len(y_hat)):
                if y_hat[i]["masks"].shape[0] > 0:
                    y_hat[i]["masks"] = (y_hat[i]["masks"] > 0.5).squeeze(1).to(torch.uint8)
        if self.framework == "mask-rcnn" or self.model_args.get("framework_masks", False):
            for i in range(len(y_hat)):
                if "masks" in y_hat[i] and y_hat[i]["masks"].shape[0] > 0:
                    y_hat[i]["masks"] = (y_hat[i]["masks"] > 0.5).squeeze(1).to(torch.uint8)

        metrics = self.val_metrics(y_hat, y)

        # https://github.com/Lightning-AI/torchmetrics/pull/1832#issuecomment-1623890714
        metrics.pop("val_classes", None)

        self.log_dict(metrics, batch_size=batch_size)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and (hasattr(self.logger.experiment, "add_figure") | hasattr(self.logger.experiment, "log_figure"))
        ):
            if "boxes" not in batch.keys():
                batch["boxes"] = batch.pop(self.boxes_field)
            if "labels" not in batch.keys():
                batch["labels"] = batch.pop(self.labels_field)
            if self.framework == "mask-rcnn":
                if "masks" not in batch.keys():
                    batch["masks"] = batch.pop(self.masks_field)
            if "boxes" not in batch.keys():
                batch["boxes"] = batch.pop(self.boxes_field)
            if "labels" not in batch.keys():
                batch["labels"] = batch.pop(self.labels_field)
            if self.framework == "mask-rcnn" or self.model_args.get("framework_masks", False):
                if "masks" not in batch.keys() and self.masks_field in batch.keys():
                    batch["masks"] = batch.pop(self.masks_field)

            # dataset = self.trainer.datamodule.val_dataset
            batch["prediction_boxes"] = [b["boxes"].cpu() for b in y_hat]
            batch["prediction_labels"] = [b["labels"].cpu() for b in y_hat]
            batch["prediction_scores"] = [b["scores"].cpu() for b in y_hat]

            if "masks" in y_hat[0].keys():
                batch["prediction_masks"] = [b["masks"].cpu() for b in y_hat]
                if self.framework == "mask-rcnn":
                    batch["prediction_masks"] = [b.unsqueeze(1) for b in batch["prediction_masks"]]

            batch["image"] = batch["image"].cpu()
            fig: Figure | None = None
            try:
                sample = unbind_samples(batch)[0]
                if hasattr(self.trainer.datamodule, "val_dataset"):
                    if hasattr(self.trainer.datamodule.val_dataset, "plot") and hasattr(
                        self.trainer.datamodule.val_dataset, "plot"
                    ):
                        fig = self.trainer.datamodule.val_dataset.plot(sample)
                    elif hasattr(self.trainer.datamodule, "plot") and callable(self.trainer.datamodule.plot):
                        fig = self.trainer.datamodule.plot(sample)
                elif hasattr(self.trainer.datamodule, "plot") and callable(self.trainer.datamodule.plot):
                    fig = self.trainer.datamodule.plot(sample)

            except (RGBBandsMissingError, ValueError, TypeError):
                pass

            if fig:
                summary_writer = self.logger.experiment
                if hasattr(self.logger.experiment, "add_figure"):
                    summary_writer.add_figure(f"image/{batch_idx}", fig, global_step=self.global_step)
                elif hasattr(self.logger.experiment, "log_figure"):
                    summary_writer.log_figure(self.logger.run_id, fig, f"epoch_{self.current_epoch}_{batch_idx}.png")
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """
        Compute the test metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """

        x = batch["image"]
        batch_size = get_batch_size(x)
        batch = self.apply_ignore_index(batch, self.ignore_index)
        y = self.reformat_batch(batch, batch_size)
        y_hat = self(x)
        if isinstance(y_hat, dict) is False:
            y_hat = y_hat.output

        y_hat = self.apply_nms_batch(y_hat, batch_size)

        if self.framework == "mask-rcnn":
            for i in range(len(y_hat)):
                if y_hat[i]["masks"].shape[0] > 0:
                    y_hat[i]["masks"] = (y_hat[i]["masks"] > 0.5).squeeze(1).to(torch.uint8)
        if self.framework == "mask-rcnn" or self.model_args.get("framework_masks", False):
            for i in range(len(y_hat)):
                if "masks" in y_hat[i] and y_hat[i]["masks"].shape[0] > 0:
                    y_hat[i]["masks"] = (y_hat[i]["masks"] > 0.5).squeeze(1).to(torch.uint8)

        metrics = self.test_metrics(y_hat, y)

        # https://github.com/Lightning-AI/torchmetrics/pull/1832#issuecomment-1623890714
        metrics.pop("test_classes", None)

        self.log_dict(metrics, batch_size=batch_size)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> list[dict[str, Tensor]]:
        """
        Output predicted bounding boxes, classes and masks.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted bounding boxes, classes and masks.
        """
        x = batch["image"]
        batch_size = get_batch_size(x)
        y_hat: list[dict[str, Tensor]] = self(x)
        if isinstance(y_hat, dict) is False:
            y_hat = y_hat.output

        y_hat = self.apply_nms_batch(y_hat, batch_size)

        return y_hat
