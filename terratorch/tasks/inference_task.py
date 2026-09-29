import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import lightning
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch import Tensor, nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score
from torchmetrics.metric import Metric
from torchmetrics.wrappers.abstract import WrapperMetric

from terratorch.datasets.utils import unbind_samples
from terratorch.models.model import AuxiliaryHead, Model, ModelOutput
from terratorch.registry.registry import MODEL_FACTORY_REGISTRY
from terratorch.tasks.base_task import TerraTorchTask
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory
from terratorch.tasks.tiled_inference import TiledInferenceParameters, tiled_inference


class InferenceTask(TerraTorchTask):
    """Pixelwise Inference task for the the heliophysics model Surya.

    This class is analog in functionality to PixelwiseRegressionTask defined by torchgeo.
    However, it has some important differences:
        - Accepts the specification of a model factory
        - Logs metrics per class
        - Does not have any callbacks by default (TorchGeo tasks do early stopping by default)
        - Allows the setting of optimizers in the constructor
        - Allows to evaluate on multiple test dataloaders"""

    def __init__(
        self,
        model_args: dict | None = None,
        model_factory: str | None = None,
        model: torch.nn.Module | None = None,
        freeze_backbone: bool = True,
        freeze_decoder: bool = True,
        tiled_inference_parameters: dict | None = None,
    ) -> None:
        """Constructor

        Args:
            model_args (Dict): Arguments passed to the model factory.
            model_factory (str, optional): Name of ModelFactory class to be used to instantiate the model.
                Is ignored when model is provided.
            model (torch.nn.Module, optional): Custom model.
            tiled_inference_parameters (dict | None, optional): Inference parameters
        """

        self.model_args = model_args
        self.tiled_inference_parameters = tiled_inference_parameters

        if model is not None and model_factory is not None:
            logger.warning("A model_factory and a model was provided. The model_factory is ignored.")
        if model is None and model_factory is None:
            raise ValueError("A model_factory or a model (torch.nn.Module) must be provided.")

        if model_factory and model is None:
            self.model_factory = MODEL_FACTORY_REGISTRY.build(model_factory)

        super().__init__(
            task="inference",
        )

        if model:
            # Custom_model
            self.model = model

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """

        def model_forward(x, **kwargs):
            return self(x)

        if self.tiled_inference_parameters:
            y_hat: Tensor = tiled_inference(model_forward, batch, **self.tiled_inference_parameters, **rest)
        else:
            y_hat: Tensor = self(batch)

        return y_hat
