"""This module contains pre-training tasks."""

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from torch import nn
from torchgeo.tasks import BaseTask
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from terratorch.datasets.utils import unbind_samples
from terratorch.models.model import Model, ReconstructionOutput
from terratorch.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.regression_tasks import IgnoreIndexLossWrapper, IgnoreIndexMetricWrapper, RootLossWrapper
from terratorch.tasks.tiled_inference import TiledInferenceParameters, tiled_inference

BATCH_IDX_FOR_VALIDATION_PLOTTING = 10


class ReconstructionTask(BaseTask):
    """
    Task with image reconstruction.
    """

    def __init__(
        self,
        model_factory: str,
        model_args: dict,
        loss: str = "mse",
        ignore_index: int | float = torch.nan,
        masked_metric: bool = True,
        lr: float = 0.001,
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        freeze_encoder: bool = False,  # noqa: FBT001, FBT002
        freeze_decoder: bool = False,  # noqa: FBT001, FBT002
        plot_on_val: bool | int = 10,
        tiled_inference_parameters: dict | None = None,
        modalities: list[str] | None = None,
    ) -> None:
        """Constructor

        Args:
            model_factory (str): Name of ModelFactory class to be used to instantiate the model.
            model_args (Dict): Arguments passed to the model factory.
            loss (str, optional): Loss to be used. Currently, supports 'mse', 'rmse', 'mae' or 'huber' loss.
                Defaults to "mse".
            aux_loss (dict[str, float] | None, optional): Auxiliary loss weights.
                Should be a dictionary where the key is the name given to the loss
                and the value is the weight to be applied to that loss.
                The name of the loss should match the key in the dictionary output by the model's forward
                method containing that output. Defaults to None.
            ignore_index (int | float, optional): Values to ignore in the metric computation. Defaults to torch.nan.
            masked_metric (bool, optional): Whether to mask the pred for the metric computation, so that metrics are
                only computed on masked pixels. Expects masks with 0 == not masked and 1 = masked. Defaults to True.
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
            freeze_encoder (bool, optional): Whether to freeze the encoder. Defaults to False.
            freeze_decoder (bool, optional): Whether to freeze the decoder and segmentation head. Defaults to False.
            plot_on_val (bool | int, optional): Whether to plot visualizations on validation.
                If true, log every epoch. Defaults to 10. If int, will plot every plot_on_val epochs.
            tiled_inference_parameters (dict | None, optional): Inference parameters
                used to determine if inference is done on the whole image or through tiling.
            modalities list(str), optional: List of modality names that are reconstructed. Expect reconstructions as
                dict with modelity names as keys and tensors as values. Computes metrics for every modality.
                Currently, only supports image modalities (regression), no segmentation maps or similar.
        """
        self.tiled_inference_parameters = tiled_inference_parameters
        self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)
        self.model_args = model_args
        self.modalities = modalities
        self.ignore_index = ignore_index
        self.masked_metric = masked_metric
        super().__init__()
        self.train_loss_handler = LossHandler("train/")
        self.val_loss_handler = LossHandler("val/")
        self.test_loss_handler = LossHandler("test/")
        self.monitor = "val/loss"
        self.plot_on_val = int(plot_on_val)

    # overwrite early stopping
    def configure_callbacks(self) -> list[Callback]:
        return []

    def configure_models(self) -> None:
        self.model: Model = self.model_factory.build_model(**self.hparams["model_args"])

        if self.hparams["freeze_encoder"]:
            if self.hparams.get("peft_config", None) is not None:
                msg = "PEFT should be run with freeze_encoder = False and freeze_decoder = False."
                raise ValueError(msg)
            self.model.freeze_encoder()
        if self.hparams["freeze_decoder"]:
            self.model.freeze_decoder()

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""

        def instantiate_metrics():
            # TODO: Handle segmentation outputs (e.g. for LULC maps with multimodal models)
            metrics = {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            }

            return {
                name: IgnoreIndexMetricWrapper(metric, ignore_index=self.hparams["ignore_index"])
                for name, metric in metrics.items()
            }

        if self.modalities:
            self.train_metrics = nn.ModuleDict(
                {
                    modality: MetricCollection(instantiate_metrics(), prefix=f"train/{modality}_")
                    for modality in self.modalities
                }
            )
            self.val_metrics = nn.ModuleDict(
                {
                    modality: MetricCollection(instantiate_metrics(), prefix=f"val/{modality}_")
                    for modality in self.modalities
                }
            )
            self.test_metrics = nn.ModuleDict(
                {
                    modality: MetricCollection(instantiate_metrics(), prefix=f"test/{modality}_")
                    for modality in self.modalities
                }
            )
        else:
            self.train_metrics = MetricCollection(instantiate_metrics(), prefix="train/")
            self.val_metrics = MetricCollection(instantiate_metrics(), prefix="val/")
            self.test_metrics = MetricCollection(instantiate_metrics(), prefix="test/")

    def mask_input(self, x: torch.Tensor | dict, mask: torch.Tensor | dict) -> torch.Tensor:
        """
        Args:
            x: input tensor or dict of input tensors, [B, C, H, W] or [B, C, T, H, W].
            mask: mask tensor or dict of mask tensors, [B, H, W] or [B, T, H, W].
        """
        if isinstance(x, dict):
            # Multimodal data
            for key in x.keys():
                x[key] = self.mask_input(x[key], mask[key])
        else:
            x = x.clone()
            if len(x.shape) == 4:
                # Adding channel dim to mask
                mask = mask.unsqueeze(1).repeat(1, x.shape[1], 1, 1)
            elif len(x.shape) == 5:
                # Temporal data
                mask = mask.unsqueeze(1).repeat(1, x.shape[1], 1, 1, 1)
            if x.dtype == torch.uint8:
                # Convert for NaN ignore_index
                x = x.to(torch.float32)
            # Mask input
            x[mask == True] = self.ignore_index

        return x

    def training_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        if isinstance(x, dict):
            batch_size = list(x.values())[0].shape[0]
        elif isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
        else:
            raise ValueError("Could not infer batch size from input data.")

        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        output: ReconstructionOutput | tuple = self(x, **rest)
        if not isinstance(output, ReconstructionOutput):
            output = ReconstructionOutput(*output)

        self.train_loss_handler.log_loss(self.log, loss_dict=output.loss, batch_size=batch_size)

        if self.masked_metric and output.mask is not None:
            x = self.mask_input(x, output.mask)

        if isinstance(x, dict):
            # Multimodal data
            for modality in x.keys():
                if modality in self.train_metrics:
                    self.train_metrics[modality].update(output.pred[modality], x[modality])
        else:
            # Single modality
            self.train_metrics.update(output.pred, x)

        return output.loss["loss"]

    def on_train_epoch_end(self) -> None:
        if isinstance(self.test_metrics, MetricCollection):
            self.log_dict(self.train_metrics.compute(), sync_dist=True)
            self.train_metrics.reset()
        else:
            for metric in self.train_metrics.values():
                self.log_dict(metric.compute(), sync_dist=True)
                metric.reset()
        return super().on_train_epoch_end()

    def _do_plot_samples(self, batch_index):
        if not self.plot_on_val:  # dont plot if self.plot_on_val is 0
            return False

        return (
            batch_index < BATCH_IDX_FOR_VALIDATION_PLOTTING
            and hasattr(self.trainer, "datamodule")
            and self.logger
            and not self.current_epoch % self.plot_on_val  # will be True every self.plot_on_val epochs
            and hasattr(self.logger, "experiment")
            and (hasattr(self.logger.experiment, "add_figure") or hasattr(self.logger.experiment, "log_figure"))
        )

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        if isinstance(x, dict):
            batch_size = list(x.values())[0].shape[0]
        elif isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
        else:
            raise ValueError("Could not infer batch size from input data.")

        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        output: ReconstructionOutput | tuple = self(x, **rest)
        if not isinstance(output, ReconstructionOutput):
            output = ReconstructionOutput(*output)

        self.val_loss_handler.log_loss(self.log, loss_dict=output.loss, batch_size=batch_size)

        if self.masked_metric and output.mask is not None:
            x = self.mask_input(x, output.mask)

        if isinstance(x, dict):
            if not isinstance(self.val_metrics, nn.ModuleDict):
                raise ValueError(
                    "Multimodal data provided but no image modalities. "
                    "Please provide modalities (list of str) to the ReconstructionTask."
                )
            # Multimodal data
            for modality in x.keys():
                if modality in self.val_metrics:
                    self.val_metrics[modality].update(output.pred[modality], x[modality])
        else:
            # Single modality
            self.val_metrics.update(output.pred, x)

        if self._do_plot_samples(batch_idx):
            try:
                datamodule = self.trainer.datamodule
                if isinstance(batch["image"], dict):
                    # Multimodal input
                    rgb_modality = getattr(datamodule, "rgb_modality", None) or list(batch["image"].keys())[0]
                    batch["image"] = batch["image"][rgb_modality]
                    batch["prediction"] = output.pred[rgb_modality].cpu()
                    if output.mask is not None:
                        batch["mask"] = output.mask[rgb_modality].cpu()
                else:
                    batch["prediction"] = output.pred.cpu()
                    batch["mask"] = output.mask.cpu()
                batch["image"] = batch["image"].cpu()
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

    def on_validation_epoch_end(self) -> None:
        if isinstance(self.test_metrics, MetricCollection):
            self.log_dict(self.val_metrics.compute(), sync_dist=True)
            self.val_metrics.reset()
        else:
            for metric in self.val_metrics.values():
                self.log_dict(metric.compute(), sync_dist=True)
                metric.reset()
        return super().on_validation_epoch_end()

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        if isinstance(x, dict):
            batch_size = list(x.values())[0].shape[0]
        elif isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
        else:
            raise ValueError("Could not infer batch size from input data.")

        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        output: ReconstructionOutput | tuple = self(x, **rest)
        if not isinstance(output, ReconstructionOutput):
            output = ReconstructionOutput(*output)

        self.test_loss_handler.log_loss(self.log, loss_dict=output.loss, batch_size=batch_size)

        if self.masked_metric and output.mask is not None:
            x = self.mask_input(x, output.mask)

        if isinstance(x, dict):
            # Multimodal data
            for modality in x.keys():
                if modality in self.test_metrics:
                    self.test_metrics[modality].update(output.pred[modality], x[modality])
        else:
            # Single modality
            self.test_metrics.update(output.pred, x)

    def on_test_epoch_end(self) -> None:
        if isinstance(self.test_metrics, MetricCollection):
            self.log_dict(self.test_metrics.compute(), sync_dist=True)
            self.test_metrics.reset()
        else:
            # multiple metric dicts
            for metric in self.test_metrics.values():
                self.log_dict(metric.compute(), sync_dist=True)
                metric.reset()
        return super().on_test_epoch_end()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        image_size = x.shape[-2:]
        file_names = batch["filename"]
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}

        def model_forward(x):
            out = self(x)[1]
            if isinstance(out, ReconstructionOutput):
                pred = out.pred
            else:
                pred = out[1]
            return pred

        # Tiled inference for autoencoder ? Is it making sense ?
        if self.tiled_inference_parameters:
            pred: torch.Tensor = tiled_inference(model_forward, x, **self.tiled_inference_parameters, **rest)
        else:
            pred: torch.Tensor = model_forward(x)
        # TODO: Return mask and file_names?
        return pred, file_names
