# Copyright contributors to the Terratorch project

"""
This is just an example of a possible structure to include timm models
"""

import os

from timm import create_model
from torch import nn
from torchgeo.models import get_weight
from torchvision.models._api import WeightsEnum

try:
    from torchgeo.trainers import utils as _torchgeo_utils
except ImportError:
    _torchgeo_utils = None

from terratorch.models.model import Model, ModelFactory, ModelOutput
from terratorch.models.utils import extract_prefix_keys
from terratorch.registry import MODEL_FACTORY_REGISTRY


@MODEL_FACTORY_REGISTRY.register
class TimmModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str,
        in_channels: int,
        num_classes: int,
        pretrained: str | bool = True,
        **kwargs,
    ) -> Model:
        """Build a classifier from timm

        Args:
            task (str): Must be "classification".
            backbone (str): Name of the backbone in timm.
            in_channels (int): Number of input channels.
            num_classes (int): Number of classes.

        Returns:
            Model: Timm model wrapped in TimmModelWrapper.
        """
        if task != "classification":
            msg = f"timm models can only perform classification, but got task {task}"
            raise Exception(msg)
        backbone_kwargs, kwargs = extract_prefix_keys(kwargs, "backbone_")
        if isinstance(pretrained, bool):
            model = create_model(
                backbone, pretrained=pretrained, num_classes=num_classes, in_chans=in_channels, **backbone_kwargs
            )
        else:
            model = create_model(backbone, num_classes=num_classes, in_chans=in_channels, **backbone_kwargs)

        # Load weights
        # Code adapted from geobench
        if pretrained and pretrained is not True:
            try:
                weights = WeightsEnum(pretrained)
                state_dict = weights.get_state_dict(progress=True)
            except ValueError:
                if os.path.exists(pretrained):
                    if _torchgeo_utils is None:
                        msg = (
                            "Loading weights from checkpoint requires torchgeo<0.6. "
                            "Please install a compatible version or provide weights as a WeightsEnum."
                        )
                        raise ImportError(msg)
                    _, state_dict = _torchgeo_utils.extract_backbone(pretrained)
                else:
                    state_dict = get_weight(pretrained).get_state_dict(progress=True)
            if _torchgeo_utils is not None:
                model = _torchgeo_utils.load_state_dict(model, state_dict)
            else:
                model.load_state_dict(state_dict, strict=False)

        return TimmModelWrapper(model)


class TimmModelWrapper(Model, nn.Module):
    def __init__(self, timm_model) -> None:
        super().__init__()
        self.timm_model = timm_model

    def forward(self, *args, **kwargs):
        return ModelOutput(self.timm_model(*args, **kwargs))

    def freeze_encoder(self):
        for param in self.timm_model.parameters():
            param.requires_grad = False
        for param in self.timm_model.get_classifier().parameters():
            param.requires_grad = True

    def freeze_decoder(self):
        for param in self.timm_model.get_classifier().parameters():
            param.requires_grad = False
