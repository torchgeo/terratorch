# Copyright contributors to the Terratorch project

import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import Tensor, nn
from torchgeo.tasks import BaseTask
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.wrappers import ClasswiseWrapper

from terratorch.models.decoders.sklearn_decoder import SklearnDecoder
from terratorch.models.model import ModelOutput
from terratorch.registry.registry import MODEL_FACTORY_REGISTRY

logger = logging.getLogger("terratorch")


class EmbeddingDecodingTask(BaseTask):
    """Task for sklearn decoders on frozen embeddings.
    Accumulates features during training, calls fit() at epoch end."""

    automatic_optimization: bool = False

    def __init__(
        self,
        model_args: dict,
        model_factory: str = "EncoderDecoderFactory",
        freeze_backbone: bool = True,
    ) -> None:
        self.model_args = model_args
        self.model_factory_name = model_factory
        self.freeze_backbone = freeze_backbone
        self._y_buf: list[np.ndarray] = []
        super().__init__()

    def configure_models(self) -> None:
        factory = MODEL_FACTORY_REGISTRY.build(self.model_factory_name)
        self.model = factory.build_model("classification", **self.model_args)

        if self.freeze_backbone:
            self.model.freeze_encoder()

        self._validate_decoder()

    def _validate_decoder(self) -> None:
        decoder = self.model.decoder
        if not isinstance(decoder, SklearnDecoder):
            raise TypeError(
                f"EmbeddingDecodingTask requires a SklearnDecoder, "
                f"got {type(decoder).__name__}"
            )

    def forward(self, x, **kwargs) -> ModelOutput:
        return self.model(x, **kwargs)

    def configure_optimizers(self):
        return []

    def training_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}

        with torch.no_grad():
            self(x, **rest)

        self._y_buf.append(y.detach().cpu().numpy())
        return torch.tensor(0.0)

    def on_train_epoch_end(self) -> None:
        if not self._y_buf:
            logger.warning("no labels accumulated, skipping fit")
            if hasattr(self, "train_metrics"):
                self.train_metrics.reset()
            return

        y_all = np.concatenate(self._y_buf, axis=0).astype(int)
        self._y_buf = []

        self.model.decoder.fit(y_all)
        if hasattr(self, "train_metrics"):
            self.train_metrics.reset()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        estimator_path = Path(self.trainer.log_dir) / "sklearn_model.joblib"
        estimator_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model.decoder.estimator, estimator_path, compress=3)
        checkpoint["sklearn_model_path"] = str(estimator_path)
        checkpoint["sklearn_fitted"] = self.model.decoder._fitted
        logger.info("saved sklearn model to %s", estimator_path)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "sklearn_model_path" in checkpoint:
            path = checkpoint["sklearn_model_path"]
            self.model.decoder.estimator = joblib.load(path)
            self.model.decoder._fitted = checkpoint.get("sklearn_fitted", True)
            logger.info("loaded sklearn model from %s", path)


class EmbeddingClassificationTask(EmbeddingDecodingTask):
    """Multilabel classification on frozen embeddings using sklearn decoders."""

    def __init__(
        self,
        model_args: dict,
        model_factory: str = "EncoderDecoderFactory",
        loss: str = "bce",
        class_weights: list[float] | None = None,
        ignore_index: int | None = -100,
        freeze_backbone: bool = True,
        class_names: list[str] | None = None,
        test_dataloaders_names: list[str] | None = None,
    ) -> None:
        super().__init__(
            model_args=model_args,
            model_factory=model_factory,
            freeze_backbone=freeze_backbone,
        )

    def configure_losses(self) -> None:
        loss = self.hparams["loss"]
        if loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == "ce":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss '{loss}' not supported. Use 'bce' or 'ce'.")

    def configure_metrics(self) -> None:
        num_classes = self.hparams["model_args"]["num_classes"]
        ignore_index = self.hparams["ignore_index"]
        class_names = self.hparams["class_names"]

        metrics = MetricCollection(
            {
                "Multilabel_Accuracy": MultilabelAccuracy(
                    num_labels=num_classes, ignore_index=ignore_index, average="macro"
                ),
                "Multilabel_Accuracy_Micro": MultilabelAccuracy(
                    num_labels=num_classes, ignore_index=ignore_index, average="micro"
                ),
                "Multilabel_F1_Score": MultilabelF1Score(
                    num_labels=num_classes, ignore_index=ignore_index, average="macro"
                ),
                "Multilabel_Precision": MultilabelPrecision(
                    num_labels=num_classes, ignore_index=ignore_index, average="macro",
                ),
                "Multilabel_Recall": MultilabelRecall(
                    num_labels=num_classes, ignore_index=ignore_index, average="macro",
                ),
                "Multilabel_AUROC": MultilabelAUROC(
                    num_labels=num_classes, ignore_index=ignore_index, average="macro",
                ),
                "Class_Accuracy": ClasswiseWrapper(
                    MultilabelAccuracy(
                        num_labels=num_classes, ignore_index=ignore_index, average=None,
                    ),
                    labels=class_names,
                    prefix="Class_Accuracy_",
                ),
                "Class_F1": ClasswiseWrapper(
                    MultilabelF1Score(
                        num_labels=num_classes, ignore_index=ignore_index, average=None,
                    ),
                    labels=class_names,
                    prefix="Class_F1_",
                ),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        if self.hparams["test_dataloaders_names"] is not None:
            self.test_metrics = nn.ModuleList(
                [metrics.clone(prefix=f"test/{name}/") for name in self.hparams["test_dataloaders_names"]]
            )
        else:
            self.test_metrics = nn.ModuleList([metrics.clone(prefix="test/")])

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch["label"].to(torch.float32)
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}

        model_output: ModelOutput = self(x, **rest)
        loss = self.criterion(model_output.output, y)
        self.log("val/loss", loss, batch_size=y.shape[0], prog_bar=True)

        y_hat = torch.sigmoid(model_output.output)
        self.val_metrics.update(y_hat, y.to(torch.int32))

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()

        # print to stdout so iterate can parse metrics
        for key, value in self.trainer.callback_metrics.items():
            if key.startswith("val/"):
                print(f"{key}: {value.item():.6f}", flush=True)

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch["label"].to(torch.float32)
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}

        model_output: ModelOutput = self(x, **rest)
        loss = self.criterion(model_output.output, y)

        prefix = self.test_metrics[dataloader_idx].prefix
        self.log(f"{prefix}loss", loss, batch_size=y.shape[0])

        y_hat = torch.sigmoid(model_output.output)
        self.test_metrics[dataloader_idx].update(y_hat, y.to(torch.int32))

    def on_test_epoch_end(self) -> None:
        for test_metric in self.test_metrics:
            metrics = test_metric.compute()
            self.log_dict(metrics)
            test_metric.reset()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> tuple:
        x = batch["image"]
        file_names = batch.get("filename", None)
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}

        model_output = self(x, **rest)
        y_hat = torch.sigmoid(model_output.output)
        return y_hat, file_names
