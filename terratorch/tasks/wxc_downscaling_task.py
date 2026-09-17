from collections.abc import Mapping
from typing import Any

import torch
from granitewxc.utils.config import ExperimentConfig
from torch import Tensor, nn
from torchgeo.tasks import BaseTask
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from terratorch.models.model import Model  # , get_factory
from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory
from terratorch.tasks.regression_tasks import RootLossWrapper


class WxCDownscalingTask(BaseTask):
    def __init__(
        self,
        model_args: dict,
        model_factory: str,
        extra_kwargs: dict,
        model_config: ExperimentConfig,
        loss: str = "mse",
        lr: float = 0.001,
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        freeze_backbone: bool = False,  # noqa: FBT001, FBT002
        freeze_decoder: bool = False,  # noqa: FBT001, FBT002
        plot_on_val: bool | int = 10,
    ) -> None:

        # Special cases for some parameters that could not be read in
        # their own fields.
        mask_unit_size = tuple(model_args.pop("mask_unit_size"))
        encoder_decoder_kernel_size_per_stage = extra_kwargs.pop("encoder_decoder_kernel_size_per_stage")
        output_vars = extra_kwargs.pop("output_vars")
        type_dataset = extra_kwargs.pop("type")
        input_levels = extra_kwargs.pop("input_levels")
        downscaling_patch_size = extra_kwargs.pop("downscaling_patch_size")
        n_input_timestamps = extra_kwargs.pop("n_input_timestamps")
        downscaling_embed_dim = extra_kwargs.pop("downscaling_embed_dim")
        encoder_decoder_conv_channels = extra_kwargs.pop("encoder_decoder_conv_channels")
        encoder_decoder_scale_per_stage = extra_kwargs.pop("encoder_decoder_scale_per_stage")
        encoder_decoder_upsampling_mode = extra_kwargs.pop("encoder_decoder_upsampling_mode")
        encoder_shift = extra_kwargs.pop("encoder_shift")
        drop_path = extra_kwargs.pop("drop_path")
        encoder_decoder_type = extra_kwargs.pop("encoder_decoder_type")
        input_size_lat = extra_kwargs.pop("input_size_lat")
        input_size_lon = extra_kwargs.pop("input_size_lon")
        freeze_backbone = extra_kwargs.pop("freeze_backbone")
        freeze_decoder = extra_kwargs.pop("freeze_decoder")
        data_path_surface = extra_kwargs.pop("data_path_surface")
        data_path_vertical = extra_kwargs.pop("data_path_vertical")
        climatology_path_surface = extra_kwargs.pop("climatology_path_surface")
        climatology_path_vertical = extra_kwargs.pop("climatology_path_vertical")
        residual_connection = model_args.pop("residual_connection")
        residual = extra_kwargs.pop("residual", True)

        # Special cases for required variables
        input_scalers_surface_path = extra_kwargs.pop("input_scalers_surface_path", None)
        if not input_scalers_surface_path:
            raise Exception("The parameter `input_scalers_surface_path` must be defined in `extra_kwargs`.")

        input_scalers_vertical_path = extra_kwargs.pop("input_scalers_vertical_path", None)
        if not input_scalers_vertical_path:
            raise Exception("The parameter `input_scalers_vertical_path` must be defined in `extra_kwargs`.")

        output_scalers_surface_path = extra_kwargs.pop("output_scalers_surface_path")
        output_scalers_vertical_path = extra_kwargs.pop("output_scalers_vertical_path")

        model_config.freeze_backbone = freeze_backbone
        model_config.freeze_decoder = freeze_decoder
        model_config.mask_unit_size = mask_unit_size
        model_config.model.mask_unit_size = mask_unit_size
        model_config.model.encoder_decoder_kernel_size_per_stage = encoder_decoder_kernel_size_per_stage
        model_config.model.input_scalers_surface_path = input_scalers_surface_path
        model_config.model.input_scalers_vertical_path = input_scalers_vertical_path
        model_config.data.output_vars = output_vars
        model_config.data.type = type_dataset
        model_config.data.input_surface_vars = model_config.data.surface_vars
        model_config.data.input_vertical_vars = model_config.data.vertical_vars
        model_config.data.input_static_surface_vars = model_config.data.static_surface_vars
        model_config.data.input_levels = input_levels
        model_config.model.downscaling_patch_size = downscaling_patch_size
        model_config.data.n_input_timestamps = n_input_timestamps
        model_config.model.downscaling_embed_dim = downscaling_embed_dim
        model_config.model.encoder_decoder_conv_channels = encoder_decoder_conv_channels
        model_config.model.encoder_decoder_scale_per_stage = encoder_decoder_scale_per_stage
        model_config.model.encoder_decoder_upsampling_mode = encoder_decoder_upsampling_mode
        model_config.model.encoder_shift = encoder_shift
        model_config.model.drop_path = drop_path
        model_config.model.encoder_decoder_type = encoder_decoder_type
        model_config.data.input_size_lat = input_size_lat
        model_config.data.input_size_lon = input_size_lon
        model_config.data.data_path_surface = data_path_surface
        model_config.data.data_path_vertical = data_path_vertical
        model_config.data.climatology_path_surface = climatology_path_surface
        model_config.data.climatology_path_vertical = climatology_path_vertical
        model_config.model.output_scalers_surface_path = output_scalers_surface_path
        model_config.model.output_scalers_vertical_path = output_scalers_vertical_path
        model_config.model.residual_connection = residual_connection
        model_config.model.residual = residual

        self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)
        self.model_config = model_config
        self.model_args = model_args
        # TODO Unify it with self.hparams
        self.extended_hparams = self.model_config.to_dict()
        super().__init__()

        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler = LossHandler(self.test_metrics.prefix)
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"
        self.plot_on_val = int(plot_on_val)

    def configure_models(self) -> None:
        self.model: Model = self.model_factory.build_model(
            "regression", aux_decoders=None, model_config=self.model_config, **self.hparams["model_args"]
        )
        if self.extended_hparams["freeze_backbone"]:
            self.model.freeze_encoder()
        if self.extended_hparams["freeze_decoder"]:
            self.model.freeze_decoder()

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        optimizer = self.hparams["optimizer"]
        if optimizer is None:
            optimizer = "Adam"
        return optimizer_factory(
            self.hparams["optimizer"],
            self.hparams["lr"],
            self.parameters(),
            self.hparams["optimizer_hparams"],
            self.hparams["scheduler"],
            self.monitor,
            self.hparams["scheduler_hparams"],
        )

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        # TODO 'reduction' should be chosen using the config and
        # a similar class as IgnoreIndex should be defined for this class

        loss: str = self.hparams["loss"].lower()
        if loss == "mse":
            self.criterion: nn.Module = nn.MSELoss(reduction="mean")
        elif loss == "mae":
            self.criterion = nn.L1Loss(reduction="mean")
        elif loss == "rmse":
            # IMPORTANT! Root is done only after ignore index! Otherwise the mean taken is incorrect
            self.criterion = RootLossWrapper(nn.MSELoss(reduction="none"), reduction="none")
        elif loss == "huber":
            self.criterion = nn.HuberLoss(reduction="none")
        else:
            exception_message = f"Loss type '{loss}' is not valid. Currently, supports 'mse', 'rmse' or 'mae' loss."
            raise ValueError(exception_message)

    def configure_metrics(self) -> None:

        def instantiate_metrics():
            return {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            }

        self.train_metrics = MetricCollection(instantiate_metrics(), prefix="train/")
        self.val_metrics = MetricCollection(instantiate_metrics(), prefix="val/")
        self.test_metrics = MetricCollection(instantiate_metrics(), prefix="test/")

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:

        x = batch["image"]
        mask = batch["mask"]
        model_output: ModelOutput = self(x)
        y = mask
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, None)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = model_output.output
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, on_epoch=True)

        return loss["loss"]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        model_output: ModelOutput = self(x)

        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, None)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat = model_output.output
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, on_epoch=True)

        return loss["loss"]

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        raise NotImplementedError("This function is not yet implemented.")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch["image"]
        file_names = batch["filename"]

        y_hat: Tensor = self(x).output
        return y_hat, file_names
