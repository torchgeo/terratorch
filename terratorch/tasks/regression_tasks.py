"""This module contains the regression task and its auxiliary classes."""

import importlib
import logging
from collections.abc import Sequence
from functools import partial
from importlib import import_module
from typing import Any

import lightning
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch import Tensor, nn
from torchmetrics import ClasswiseWrapper, MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score
from torchmetrics.metric import Metric
from torchmetrics.wrappers.abstract import WrapperMetric

from terratorch.datasets.utils import unbind_samples
from terratorch.models.model import AuxiliaryHead, Model, ModelOutput
from terratorch.registry.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.base_task import TerraTorchTask
from terratorch.tasks.loss_handler import CombinedLoss, LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory
from terratorch.tasks.tiled_inference import TiledInferenceParameters, tiled_inference
from terratorch.tasks.utils import _instantiate_from_path

BATCH_IDX_FOR_VALIDATION_PLOTTING = 10

logger = logging.getLogger("terratorch")


class RootLossWrapper(nn.Module):
    def __init__(self, loss_function: nn.Module, reduction: None | str = "mean") -> None:
        super().__init__()
        self.loss_function = loss_function
        self.reduction = reduction

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss = torch.sqrt(self.loss_function.forward(output, target))
        if self.reduction is None:
            return loss

        if self.reduction == "mean":
            return loss.mean()

        msg = "Only 'mean' and None reduction supported"
        raise Exception(msg)


class WeightedMultivariateLossWrapper(nn.Module):
    """Wrapper to weight the individual losses of each variable predicted in the multivariate regression scenario."""

    def __init__(self, loss_function: nn.Module, weights: torch.Tensor, reduction: None | str = "mean") -> None:
        super().__init__()
        if weights.ndim != 1:
            raise ValueError("Expected 1D class weights of shape [num_outputs]")
        self.loss_function = loss_function
        self.register_buffer("weights", weights)
        self.reduction = reduction

    def forward(
        self,
        output: Tensor,
        target: Tensor,
    ) -> Tensor:
        loss = self.loss_function(output, target)
        weights = self.weights.view(1, -1, *([1] * (loss.ndim - 2)))  # [1, num_vars, 1, 1] for pixel-wise

        weighted_loss = weights * loss

        if self.reduction is None:
            return weighted_loss  # [B, num_vars, H, W] or [B, num_vars]

        if self.reduction == "mean":
            return weighted_loss.sum() / weights.expand_as(loss).sum()  # weighted scalar mean for backprop

        msg = "Only 'mean' and None reduction supported"
        raise Exception(msg)


class IgnoreIndexLossWrapper(nn.Module):
    """Wrapper over loss that will ignore certain values"""

    def __init__(self, loss_function: nn.Module, ignore_index: int, reduction: None | str = "mean") -> None:
        super().__init__()
        self.loss_function = loss_function
        self.ignore_index = ignore_index
        self.reduction = reduction

    def _mask(self, loss: Tensor, target: Tensor):
        mask = torch.ones_like(target)
        mask[target == self.ignore_index] = 0

        # avoid ZeroDivisionError
        eps = torch.finfo(torch.float32).eps
        return loss * mask.float(), mask.sum() + eps

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        loss: Tensor = self.loss_function(output, target.float())
        if self.ignore_index is not None:
            loss, norm_value = self._mask(loss, target)
        else:
            norm_value = loss.numel()

        if self.reduction is None:
            return loss

        if self.reduction == "mean":
            return loss.sum() / norm_value

        msg = "Only 'mean' and None reduction supported"
        raise Exception(msg)


class IgnoreIndexMetricWrapper(WrapperMetric):  # to be fixed for multiple variable outputs
    """Wrapper over other metric that will ignore certain values.

    This class implements ignore_index by removing values where the target matches the ignored value.
    This will only work for metrics where the shape can be flattened.

    WARNING: This is trickier than it seems. Implementation inspired by https://github.com/Lightning-AI/torchmetrics/blob/99d6d9d6ac4eb1b3398241df558604e70521e6b0/src/torchmetrics/wrappers/classwise.py#L27-L211

    The wrapper needs to change the inputs on the forward directly, so that it affects the update of the wrapped metric
    """

    def __init__(self, wrapped_metric: Metric, ignore_index: int, num_outputs: int = 1) -> None:
        super().__init__()
        self.metric = wrapped_metric
        self.ignore_index = ignore_index
        self.num_outputs = num_outputs  # now only for scalar regression - needs to be rethinked for pixel-wise

    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.num_outputs == 1:
            if self.ignore_index is not None:
                items_to_take = target != self.ignore_index
                items_to_take[torch.isnan(target)] = False  # Filter NaN values as well
                target = target[items_to_take]
                preds = preds[items_to_take]
            else:
                preds = preds.flatten()
                target = target.flatten()
            return self.metric.update(preds, target)

        elif self.num_outputs > 1:
            if self.ignore_index is not None:
                return NotImplementedError  # TODO: expand to multivariate case both for pixel and scalar regression
            return self.metric.update(preds, target)

    def forward(self, preds: Tensor, target: Tensor, *args, **kwargs) -> Any:
        if self.ignore_index is not None:
            items_to_take = target != self.ignore_index
            items_to_take[torch.isnan(target)] = False  # Filter NaN values as well
            target = target[items_to_take]
            preds = preds[items_to_take]
        return self.metric.forward(preds, target, *args, **kwargs)

    def compute(self) -> Tensor:
        return self.metric.compute()

    def plot(self, val: Tensor | Sequence[Tensor] | None = None, ax=None):
        return self.metric.plot(val, ax)

    def reset(self) -> None:
        self.metric.reset()


class WeightedMetricWrapper(WrapperMetric):
    def __init__(self, wrapped_metric: Metric, weights: list[float] | None, num_var: int) -> None:
        super().__init__()
        self.wrapped_metric = wrapped_metric
        if weights is not None:
            self.weights = torch.Tensor(weights)
        else:
            self.weights = torch.ones(num_var)

    def update(self, preds: Tensor, target: Tensor) -> None:
        return self.wrapped_metric.update(preds, target)

    def compute(self) -> Tensor:
        values = self.wrapped_metric.compute()
        weighted_values = values * self.weights
        return weighted_values.sum() / self.weights.sum()


def get_module_and_class(path):
    class_name = path.split(".")[-1]
    path_ = path.replace("." + class_name, "")
    return path_, class_name


def init_loss(
    loss: str,
    ignore_index: int = None,
    var_weights: list[float] = None,
    num_outputs: int = 1,
    custom_loss: bool = False,
    custom_loss_kwargs: dict = None,
):
    if custom_loss:
        assert custom_loss_kwargs, "If you are using a custom loss, the `custom_loss_kwargs` are required."
        base_criterion = _instantiate_from_path(loss, **custom_loss_kwargs)
    elif loss == "mse":
        base_criterion = IgnoreIndexLossWrapper(nn.MSELoss(reduction="none"), ignore_index)
    elif loss == "mae":
        base_criterion = IgnoreIndexLossWrapper(nn.L1Loss(reduction="none"), ignore_index)
    elif loss == "rmse":
        # IMPORTANT! Root is done only after ignore index! Otherwise, the mean taken is incorrect
        base_criterion = RootLossWrapper(
            IgnoreIndexLossWrapper(nn.MSELoss(reduction="none"), ignore_index), reduction=None
        )
    elif loss == "huber":
        base_criterion = IgnoreIndexLossWrapper(nn.HuberLoss(reduction="none"), ignore_index)
    else:
        raise ValueError(f"Loss type '{loss}' is not valid. Currently, supports 'mse', 'rmse', 'mae', or 'huber' loss.")

    if var_weights is not None:
        check_weights_classes(var_weights, num_outputs)
        base_criterion = WeightedMultivariateLossWrapper(base_criterion, var_weights, reduction="mean")

    base_criterion = IgnoreIndexLossWrapper(base_criterion, ignore_index)

    if loss == "rmse":
        # Root after IgnoreIndex
        base_criterion = RootLossWrapper(base_criterion, reduction=None)

    # Either weighted mean of the losses or a simple mean
    return base_criterion


def check_weights_classes(var_weights: Tensor, num_outputs: int):
    if len(var_weights) != num_outputs:
        exception_message = f"Number of weights must correspond to number of variables. Got {len(var_weights)} weights for {num_outputs} variables."
        raise ValueError(exception_message)


class PixelwiseRegressionTask(TerraTorchTask):
    """Pixelwise Regression Task that accepts models from a range of sources.

    This class is analog in functionality to PixelwiseRegressionTask defined by torchgeo.
    However, it has some important differences:
        - Accepts the specification of a model factory
        - Logs metrics per class
        - Does not have any callbacks by default (TorchGeo tasks do early stopping by default)
        - Allows the setting of optimizers in the constructor
        - Allows to evaluate on multiple test dataloaders"""

    def __init__(
        self,
        model_args: dict,
        model_factory: str | None = None,
        model: torch.nn.Module | None = None,
        loss: str | list[str] | dict[str, float] | torch.nn.Module = "mse",
        aux_heads: list[AuxiliaryHead] | None = None,
        aux_loss: dict[str, float] | None = None,
        var_weights: list[float] | None = None,
        ignore_index: int | None = None,
        custom_loss: bool = False,
        custom_loss_kwargs: dict = None,
        lr: float = 0.001,
        # TODO: customize for multivariate regression as well
        # the following are optional so CLI doesnt need to pass them
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        freeze_backbone: bool = False,  # noqa: FBT001, FBT002
        freeze_decoder: bool = False,  # noqa: FBT001, FBT002
        freeze_head: bool = False,  # noqa: FBT001, FBT002
        plot_on_val: bool | int = 10,
        tiled_inference_parameters: dict | None = None,
        test_dataloaders_names: list[str] | None = None,
        lr_overrides: dict[str, float] | None = None,
        tiled_inference_on_testing: bool = False,
        tiled_inference_on_validation: bool = False,
        path_to_record_metrics: str = None,
    ) -> None:
        """Constructor

        Args:
            model_args (Dict): Arguments passed to the model factory.
            model_factory (str, optional): Name of ModelFactory class to be used to instantiate the model.
                Is ignored when model is provided.
            model (torch.nn.Module, optional): Custom model.
            loss (str | list[str] | dict[str, float], optional): Loss to be used. Single loss as string.
                Multiple losses can be provided as list of strings or as dict with float values defining loss weights.
                Currently, supports 'mse', 'rmse', 'mae' or 'huber' loss. Defaults to "mse".
            aux_loss (dict[str, float] | None, optional): Auxiliary loss weights.
                Should be a dictionary where the key is the name given to the loss
                and the value is the weight to be applied to that loss.
                The name of the loss should match the key in the dictionary output by the model's forward
                method containing that output. Defaults to None.
            var_weights (list[float] | None, optional): List of class weights to be applied to the loss.
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
            freeze_head (bool, optional): Whether to freeze the segmentation head. Defaults to False.
            plot_on_val (bool | int, optional): Whether to plot visualizations on validation and in test.
                If true, log every epoch. Defaults to 10. If int, will plot every plot_on_val epochs.
            tiled_inference_parameters (dict | None, optional): Inference parameters
                used to determine if inference is done on the whole image or through tiling.
            test_dataloaders_names (list[str] | None, optional): Names used to differentiate metrics when
                multiple dataloaders are returned by test_dataloader in the datamodule. Defaults to None,
                which assumes only one test dataloader is used.
            lr_overrides (dict[str, float] | None, optional): Dictionary to override the default lr in specific
                parameters. The key should be a substring of the parameter names (it will check the substring is
                contained in the parameter name)and the value should be the new lr. Defaults to None.
            tiled_inference_on_testing (bool): A boolean to define if tiled inference will be used during the test step.
            tiled_inference_on_validation (bool): A boolean to define if tiled inference will be used during the val step.
            path_to_record_metrics (str): A path to save the file containing the metrics log.
        """

        self.tiled_inference_parameters = tiled_inference_parameters
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
            task="regression",
            tiled_inference_on_testing=tiled_inference_on_testing,
            tiled_inference_on_validation=tiled_inference_on_validation,
            path_to_record_metrics=path_to_record_metrics,
            plot_on_val=plot_on_val,
        )

        if model:
            # Custom_model
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

        if isinstance(loss, str):
            # Single loss
            self.criterion = init_loss(
                loss, ignore_index=ignore_index, custom_loss=custom_loss, custom_loss_kwargs=custom_loss_kwargs
            )
        elif isinstance(loss, nn.Module):
            # Custom loss
            self.criterion = loss
        elif isinstance(loss, list):
            # List of losses with equal weights
            losses = {loss: init_loss(loss, ignore_index=ignore_index, custom_loss=custom_loss) for loss in loss}
            self.criterion = CombinedLoss(losses=losses)
        elif isinstance(loss, dict):
            # Equal weighting of losses
            loss, weight = list(loss.keys()), list(loss.values())
            losses = {loss: init_loss(loss, ignore_index=ignore_index) for loss in loss}
            self.criterion = CombinedLoss(losses=losses, weight=weight)
        else:
            raise ValueError(
                f"The loss type {loss} isn't supported. Provide loss as string, list, or dict[name, weights]."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""

        def instantiate_metrics():
            return {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
                "R2_Score": R2Score(),
            }

        def wrap_metrics_with_ignore_index(metrics):
            return {
                name: IgnoreIndexMetricWrapper(metric, ignore_index=self.hparams["ignore_index"])
                for name, metric in metrics.items()
            }

        self.train_metrics = MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix="train/")
        self.val_metrics = MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix="val/")
        if self.hparams["test_dataloaders_names"] is not None:
            self.test_metrics = nn.ModuleList(
                [
                    MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix=f"test/{dl_name}/")
                    for dl_name in self.hparams["test_dataloaders_names"]
                ]
            )
        else:
            self.test_metrics = nn.ModuleList(
                [MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix="test/")]
            )

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output: ModelOutput = self(x, **rest)
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = model_output.output
        self.train_metrics.update(y_hat, y)

        return loss["loss"]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output = self.handle_full_or_tiled_inference(x, self.tiled_inference_on_validation, **rest)
        if self.tiled_inference_on_validation:
            model_output.output = torch.squeeze(model_output.output, 1)
        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = model_output.output
        self.val_metrics.update(y_hat, y)

        if self._do_plot_samples(batch_idx):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat
                batch["image"] = x
                if isinstance(batch["image"], dict):
                    rgb_modality = getattr(datamodule, "rgb_modality", None) or list(batch["image"].keys())[0]
                    batch["image"] = batch["image"][rgb_modality]
                for key in ["image", "mask", "prediction"]:
                    val = batch[key]
                    if isinstance(val, torch.Tensor):
                        batch[key] = val.cpu()
                    elif isinstance(val, (list, tuple)):
                        # Move all tensors inside the collection to cpu
                        batch[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in val]
                sample = unbind_samples(batch)[0]
                fig = datamodule.val_dataset.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    if hasattr(summary_writer, "add_figure"):
                        summary_writer.add_figure(f"image/{batch_idx}", fig, global_step=self.global_step)
                    elif hasattr(summary_writer, "log_figure"):
                        summary_writer.log_figure(
                            self.logger.run_id, fig, f"epoch_{self.current_epoch}_{batch_idx}.png"
                        )
            except (ValueError, TypeError):
                pass
            finally:
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        model_output = self.handle_full_or_tiled_inference(x, self.tiled_inference_on_testing, **rest)
        if self.tiled_inference_on_testing:
            model_output.output = torch.squeeze(model_output.output, 1)

        if dataloader_idx >= len(self.test_loss_handler):
            msg = "You are returning more than one test dataloader but not defining enough test_dataloaders_names."
            raise ValueError(msg)
        loss = self.test_loss_handler[dataloader_idx].compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.test_loss_handler[dataloader_idx].log_loss(
            partial(self.log, add_dataloader_idx=False),  # We don't need the dataloader idx as prefixes are different
            loss_dict=loss,
            batch_size=y.shape[0],
        )
        y_hat = model_output.output
        self.test_metrics[dataloader_idx].update(y_hat, y)

        self.record_metrics(dataloader_idx, y_hat, y)

        if self._do_plot_samples(batch_idx):
            batch["prediction"] = y_hat
            self.plot_sample(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
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
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        def model_forward(x, **kwargs):
            return self(x, **kwargs).output

        if self.tiled_inference_parameters:
            y_hat: Tensor = tiled_inference(model_forward, x, **self.tiled_inference_parameters, **rest)
        else:
            y_hat: Tensor = self(x, **rest).output
        return y_hat, file_names


class ScalarRegressionTask(TerraTorchTask):
    """Scalar Regression Task that accepts models from a range of sources.

    This class is analog in functionality to RegressionTask defined by torchgeo.
    However, it has some important differences:
        - Accepts the specification of a model factory
        - Logs metrics per class
        - Does not have any callbacks by default (TorchGeo tasks do early stopping by default)
        - Allows the setting of optimizers in the constructor
        - Allows to evaluate on multiple test dataloaders"""

    def __init__(
        self,
        model_args: dict,
        model_factory: str | None = None,
        model: torch.nn.Module | None = None,
        loss: str | list[str] | dict[str, float] = "mse",
        aux_heads: list[AuxiliaryHead] | None = None,
        aux_loss: dict[str, float] | None = None,
        var_weights: list[float] | None = None,
        ignore_index: int | None = None,
        lr: float = 0.001,
        num_outputs: int = 1,
        # the following are optional so CLI doesnt need to pass them
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        freeze_backbone: bool = False,  # noqa: FBT001, FBT002
        freeze_decoder: bool = False,  # noqa: FBT001, FBT002
        plot_on_val: bool | int = False,
        freeze_head: bool = False,  # noqa: FBT001, FBT002
        var_names: list[str] | None = None,
        test_dataloaders_names: list[str] | None = None,
        lr_overrides: dict[str, float] | None = None,
        path_to_record_metrics: str = None,
    ) -> None:
        """Constructor

        Args:
            model_args (Dict): Arguments passed to the model factory.
            model_factory (str, optional): Name of ModelFactory class to be used to instantiate the model.
                Is ignored when model is provided.
            model (torch.nn.Module, optional): Custom model.
            loss (str, optional): Loss to be used. Currently, supports 'mse', 'rmse', 'mae' or 'huber' loss.
                Defaults to "mse".
            aux_loss (dict[str, float] | None, optional): Auxiliary loss weights.
                Should be a dictionary where the key is the name given to the loss
                and the value is the weight to be applied to that loss.
                The name of the loss should match the key in the dictionary output by the model's forward
                method containing that output. Defaults to None.
            var_weights (list[float] | None, optional): List of variable weights to be applied to the loss.
                Defaults to None.
            ignore_index (int | None, optional): Label to ignore in the loss computation. Defaults to None.
            lr (float, optional): Learning rate to be used. Defaults to 0.001.
            num_outputs (int): Number of predicted regression variables. Defaults to single regression.
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
            freeze_head (bool, optional): Whether to freeze the segmentation head. Defaults to False.
            plot_on_val (bool | int, optional): Whether to plot visualizations on validation.
                If true, log every epoch. Defaults to 10. If int, will plot every plot_on_val epochs.
            var_names (list[str] | None, optional): List of variable names passed to metrics for better naming.
            test_dataloaders_names (list[str] | None, optional): Names used to differentiate metrics when
                multiple dataloaders are returned by test_dataloader in the datamodule. Defaults to None,
                which assumes only one test dataloader is used.
            lr_overrides (dict[str, float] | None, optional): Dictionary to override the default lr in specific
                parameters. The key should be a substring of the parameter names (it will check the substring is
                contained in the parameter name)and the value should be the new lr. Defaults to None.
            path_to_record_metrics (str): A path to save the file containing the metrics log.
        """
        self.model_args = model_args
        self.aux_loss = aux_loss
        self.aux_heads = aux_heads
        if num_outputs < 1:
            raise ValueError("num_outputs can't be less than 1.")
        self.num_outputs = num_outputs
        self.var_weights = torch.Tensor(var_weights) if var_weights is not None else None

        if model is not None and model_factory is not None:
            logger.warning("A model_factory and a model was provided. The model_factory is ignored.")
        if model is None and model_factory is None:
            raise ValueError("A model_factory or a model (torch.nn.Module) must be provided.")

        if model_factory and model is None:
            self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)

        super().__init__(task="scalar_regression", path_to_record_metrics=path_to_record_metrics)

        if model:
            # Custom_model
            self.model = model

        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler: list[LossHandler] = []
        for metrics in self.test_metrics:
            self.test_loss_handler.append(LossHandler(metrics.prefix))
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"
        self.plot_on_val = int(plot_on_val)

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]

        if isinstance(loss, str):
            # custom nn.Module type loss
            if "." in loss:
                module_name, class_name = loss.rsplit(".", 1)
                module = import_module(module_name)
                loss_cls = getattr(module, class_name)
                self.criterion = loss_cls()
            # Single loss
            else:
                self.criterion = init_loss(loss, ignore_index=ignore_index)

        elif isinstance(loss, nn.Module):
            # Custom loss
            self.criterion = loss

        elif isinstance(loss, list):
            # List of losses with equal weights
            losses = {loss: init_loss(loss, ignore_index=ignore_index) for loss in loss}
            self.criterion = CombinedLoss(losses=losses)
        elif isinstance(loss, dict):
            # Equal weighting of losses
            loss, weight = list(loss.keys()), list(loss.values())
            losses = {loss: init_loss(loss, ignore_index=ignore_index) for loss in loss}
            self.criterion = CombinedLoss(losses=losses, weight=weight)
        else:
            raise ValueError(
                f"The loss type {loss} isn't supported. Provide loss as string, nn.Module, list, or "
                f"dict[name, weights]."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        var_names = self.hparams["var_names"]

        def instantiate_metrics():
            metrics = {
                "RMSE": MeanSquaredError(num_outputs=self.num_outputs, squared=False),
                "MSE": MeanSquaredError(num_outputs=self.num_outputs, squared=True),
                "MAE": MeanAbsoluteError(num_outputs=self.num_outputs),
                "R2_Score": R2Score(multioutput="raw_values"),
            }

            if self.num_outputs > 1:
                per_class_metrics = {
                    name: ClasswiseWrapper(metric, labels=var_names, postfix=" ") for name, metric in metrics.items()
                }

                if self.var_weights is not None:
                    check_weights_classes(self.var_weights, self.num_outputs)
                    postfix = "weighted"
                else:
                    postfix = "mean"

                for name, metric in metrics.items():
                    per_class_metrics[f"{name}_{postfix}"] = WeightedMetricWrapper(
                        metric, self.var_weights, self.num_outputs
                    )

                return per_class_metrics

            return metrics

        def wrap_metrics_with_ignore_index(metrics):
            return {
                name: IgnoreIndexMetricWrapper(
                    metric, ignore_index=self.hparams["ignore_index"], num_outputs=self.num_outputs
                )
                for name, metric in metrics.items()
            }

        self.train_metrics = MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix="train/")
        self.val_metrics = MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix="val/")
        if self.hparams["test_dataloaders_names"] is not None:
            self.test_metrics = nn.ModuleList(
                [
                    MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix=f"test/{dl_name}/")
                    for dl_name in self.hparams["test_dataloaders_names"]
                ]
            )
        else:
            self.test_metrics = nn.ModuleList(
                [MetricCollection(wrap_metrics_with_ignore_index(instantiate_metrics()), prefix="test/")]
            )

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
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

        # loss = {"loss": tensor as weighted mean of the outputs}
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = model_output.output
        self.train_metrics.update(y_hat, y)

        return loss["loss"]  # scalar

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
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
        # model_output: ModelOutput = self(x, **rest)
        model_output = self.handle_full_or_tiled_inference(x, self.tiled_inference_on_validation, **rest)

        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = model_output.output
        self.val_metrics.update(y_hat, y)

        if self._do_plot_samples(batch_idx):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat
                if isinstance(batch["image"], dict):
                    rgb_modality = getattr(datamodule, "rgb_modality", None) or list(batch["image"].keys())[0]
                    batch["image"] = batch["image"][rgb_modality]
                for key in ["image", "label", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.val_dataset.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    if hasattr(summary_writer, "add_figure"):
                        summary_writer.add_figure(f"image/{batch_idx}", fig, global_step=self.global_step)
                    elif hasattr(summary_writer, "log_figure"):
                        summary_writer.log_figure(
                            self.logger.run_id, fig, f"epoch_{self.current_epoch}_{batch_idx}.png"
                        )
            except (ValueError, TypeError):
                pass
            finally:
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
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

        model_output = self.handle_full_or_tiled_inference(x, self.tiled_inference_on_testing, **rest)

        if dataloader_idx >= len(self.test_loss_handler):
            msg = "You are returning more than one test dataloader but not defining enough test_dataloaders_names."
            raise ValueError(msg)
        loss = self.test_loss_handler[dataloader_idx].compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.test_loss_handler[dataloader_idx].log_loss(
            partial(self.log, add_dataloader_idx=False),  # We don't need the dataloader idx as prefixes are different
            loss_dict=loss,
            batch_size=y.shape[0],
        )
        y_hat = model_output.output
        self.test_metrics[dataloader_idx].update(y_hat, y)

        self.record_metrics(dataloader_idx, y_hat, y)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the predicted variables probabilities.

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
        y_hat: Tensor = self(x, **rest).output
        return y_hat, file_names
