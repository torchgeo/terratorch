import logging
from functools import partial

import lightning
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss
from torch import Tensor, nn
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from terratorch.datasets.utils import unbind_samples
from terratorch.models.model import AuxiliaryHead, Model, ModelOutput
from terratorch.registry.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.base_task import TerraTorchTask
from terratorch.tasks.loss_handler import CombinedLoss, LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory
from terratorch.tasks.utils import _instantiate_from_path

logger = logging.getLogger("terratorch")


def to_class_prediction(y: ModelOutput) -> Tensor:
    y_hat = y.output
    return y_hat.argmax(dim=1)


def init_loss(
    loss: str,
    ignore_index: int = None,
    custom_loss: bool = False,
    custom_loss_kwargs: dict = None,
    class_weights: list = None,
) -> nn.Module:
    if custom_loss:
        assert custom_loss_kwargs, "If you are using a custom loss, the `custom_loss_kwargs` are required."
        return _instantiate_from_path(loss, **custom_loss_kwargs)
    if loss == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
    elif loss == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss == "jaccard":
        return JaccardLoss(mode="multiclass")
    elif loss == "focal":
        return FocalLoss(mode="multiclass", normalized=True)
    else:
        raise ValueError(f"Loss type '{loss}' is not valid. Only 'ce', 'bce', 'jaccard', or 'focal' supported.")


class ClassificationTask(TerraTorchTask):
    """Classification Task that accepts models from a range of sources.

    This class is analog in functionality to the class ClassificationTask defined by torchgeo.
    However, it has some important differences:
        - Accepts the specification of a model factory
        - Logs metrics per class
        - Does not have any callbacks by default (TorchGeo tasks do early stopping by default)
        - Allows the setting of optimizers in the constructor
        - It provides mIoU with both Micro and Macro averaging
        - Allows to evaluate on multiple test dataloaders

    .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging gives equal weight to each class, useful
             for balanced performance assessment across imbalanced classes.
    """

    def __init__(
        self,
        model_args: dict,
        model_factory: str | None = None,
        model: torch.nn.Module | None = None,
        loss: str | list[str] | dict[str, float] | torch.nn.Module = "ce",
        aux_heads: list[AuxiliaryHead] | None = None,
        aux_loss: dict[str, float] | None = None,
        class_weights: list[float] | None = None,
        ignore_index: int | None = -100,
        custom_loss: bool = False,
        custom_loss_kwargs: dict = None,
        lr: float = 0.001,
        # the following are optional so CLI doesnt need to pass them
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        freeze_backbone: bool = False,  # noqa: FBT001, FBT002
        freeze_decoder: bool = False,  # noqa: FBT002, FBT001
        freeze_head: bool = False,  # noqa: FBT002, FBT001
        plot_on_val: bool | int = False,  # Deactivate for classification to reduce overhead
        class_names: list[str] | None = None,
        test_dataloaders_names: list[str] | None = None,
        lr_overrides: dict[str, float] | None = None,
        path_to_record_metrics: str = None,
    ) -> None:
        """Constructor

        Args:
            model_args (Dict): Arguments passed to the model factory.
            model_factory (str, optional): ModelFactory class to be used to instantiate the model.
                Is ignored when model is provided.
            model (torch.nn.Module, optional): Custom model.
            loss (str | list[str] | dict[str, float], optional): Loss to be used. Single loss as string.
                Multiple losses can be provided as list of strings or as dict with float values defining loss weights.
                Currently, supports 'ce', 'bce', 'jaccard' or 'focal' loss. Defaults to "ce".
            aux_loss (dict[str, float] | None, optional): Auxiliary loss weights.
                Should be a dictionary where the key is the name given to the loss
                and the value is the weight to be applied to that loss.
                The name of the loss should match the key in the dictionary output by the model's forward
                method containing that output. Defaults to None.
            plot_on_val (bool | int, optional): Whether to plot visualizations on validation and in test.
                If true, log every epoch. Defaults to False. If int, will plot every plot_on_val epochs.
            class_weights (list[float] | None, optional): List of class weights to be applied to the loss.
                Defaults to None.
            ignore_index (int | None, optional): Label to ignore in the loss computation. Defaults to None.
            lr (float, optional): Learning rate to be used. Defaults to 0.001.
            optimizer (str | None, optional): Name of optimizer class from torch.optim to be used.
                If None, will use Adam. Defaults to None. Overriden by config / cli specification through LightningCLI.
            optimizer_hparams (dict | None): Parameters to be passed for instantiation of the optimizer.
                Overriden by config / cli specification through LightningCLI.
            scheduler (str, optional): Name of Torch scheduler class from torch.optim.lr_scheduler
                to be used (e.g. ReduceLROnPlateau). Defaults to None.
                Overriden by config / cli specification through LightningCLI.
            scheduler_hparams (dict | None): Parameters to be passed for instantiation of the scheduler.
                Overriden by config / cli specification through LightningCLI.
            freeze_backbone (bool, optional): Whether to freeze the backbone. Defaults to False.
            freeze_decoder (bool, optional): Whether to freeze the decoder. Defaults to False.
            freeze_head (bool, optional): Whether to freeze the segmentation_head. Defaults to False.
            class_names (list[str] | None, optional): List of class names passed to metrics for better naming.
                Defaults to numeric ordering.
            test_dataloaders_names (list[str] | None, optional): Names used to differentiate metrics when
                multiple dataloaders are returned by test_dataloader in the datamodule. Defaults to None,
                which assumes only one test dataloader is used.
            lr_overrides (dict[str, float] | None, optional): Dictionary to override the default lr in specific
                parameters. The key should be a substring of the parameter names (it will check the substring is
                contained in the parameter name)and the value should be the new lr. Defaults to None.
            path_to_record_metrics (str): A path to save the file containing the metrics log.
        """

        self.aux_loss = aux_loss
        self.aux_heads = aux_heads
        self.model_args = model_args

        if model is not None and model_factory is not None:
            logger.warning("A model_factory and a model was provided. The model_factory is ignored.")
        if model is None and model_factory is None:
            raise ValueError("A model_factory or a model (torch.nn.Module) must be provided.")

        if model_factory and model is None:
            self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)

        super().__init__(
            task="classification",
            path_to_record_metrics=path_to_record_metrics,
            plot_on_val=plot_on_val,
        )

        if model:
            # Custom model
            self.model = model

        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler: list[LossHandler] = []
        for metrics in self.test_metrics:
            self.test_loss_handler.append(LossHandler(metrics.prefix))
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]
        custom_loss = self.hparams["custom_loss"]
        custom_loss_kwargs = self.hparams["custom_loss_kwargs"]

        class_weights = (
            torch.Tensor(self.hparams["class_weights"]) if self.hparams["class_weights"] is not None else None
        )

        if isinstance(loss, str):
            # Single loss
            self.criterion = init_loss(
                loss,
                ignore_index=ignore_index,
                custom_loss=custom_loss,
                custom_loss_kwargs=custom_loss_kwargs,
                class_weights=class_weights,
            )
        elif isinstance(loss, nn.Module):
            # Custom loss
            self.criterion = loss
        elif isinstance(loss, list):
            # List of losses with equal weights
            losses = {
                loss: init_loss(
                    loss,
                    ignore_index=ignore_index,
                    custom_loss=custom_loss,
                    custom_loss_kwargs=custom_loss_kwargs,
                    class_weights=class_weights,
                )
                for loss in loss
            }
            self.criterion = CombinedLoss(losses=losses)
        elif isinstance(loss, dict):
            # Equal weighting of losses
            loss, weight = list(loss.keys()), list(loss.values())
            losses = {
                loss: init_loss(
                    loss,
                    ignore_index=ignore_index,
                    custom_loss=custom_loss,
                    custom_loss_kwargs=custom_loss_kwargs,
                    class_weights=class_weights,
                )
                for loss in loss
            }
            self.criterion = CombinedLoss(losses=losses, weight=weight)
        else:
            raise ValueError(
                f"The loss type {loss} isn't supported. Provide loss as string, list, or dict[name, weights]."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["model_args"]["num_classes"]
        ignore_index: int = self.hparams["ignore_index"]
        class_names = self.hparams["class_names"]
        metrics = MetricCollection(
            {
                "Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Accuracy_Micro": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "F1_Score": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Precision": MulticlassPrecision(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Recall": MulticlassRecall(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Class_Accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=class_names,
                    prefix="Class_Accuracy_",
                ),
                "Class_F1": ClasswiseWrapper(
                    MulticlassF1Score(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
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
                [metrics.clone(prefix=f"test/{dl_name}/") for dl_name in self.hparams["test_dataloaders_names"]]
            )
        else:
            self.test_metrics = nn.ModuleList([metrics.clone(prefix="test/")])

    def training_step(self, batch: object, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat_hard = to_class_prediction(model_output)
        self.train_metrics.update(y_hat_hard, y)

        return loss["loss"]

    def validation_step(self, batch: object, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat_hard = to_class_prediction(model_output)
        self.val_metrics.update(y_hat_hard, y)

        if self._do_plot_samples(batch_idx):
            batch["prediction"] = y_hat_hard
            self.plot_sample(batch, batch_idx)

    def test_step(self, batch: object, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["label"]
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        if dataloader_idx >= len(self.test_loss_handler):
            msg = "You are returning more than one test dataloader but not defining enough test_dataloaders_names."
            raise ValueError(msg)
        loss = self.test_loss_handler[dataloader_idx].compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.test_loss_handler[dataloader_idx].log_loss(
            partial(self.log, add_dataloader_idx=False),  # We don't need the dataloader idx as prefixes are different
            loss_dict=loss,
            batch_size=y.shape[0],
        )
        y_hat_hard = to_class_prediction(model_output)
        self.test_metrics[dataloader_idx].update(y_hat_hard, y)

        self.record_metrics(dataloader_idx, y_hat_hard, y)

        if self._do_plot_samples(batch_idx):
            batch["prediction"] = y_hat_hard
            self.plot_sample(batch, batch_idx)

    def predict_step(self, batch: object, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        file_names = batch["filename"] if "filename" in batch else None
        other_keys = batch.keys() - {"image", "label", "filename"}
        rest = {k: batch[k] for k in other_keys}
        y_hat = self(x, **rest).output
        y_hat = y_hat.argmax(dim=1)
        return y_hat, file_names
