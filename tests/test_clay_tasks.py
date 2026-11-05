# Copyright contributors to the Terratorch project
import gc
import pytest
import torch

from terratorch.models import ClayModelFactory
from terratorch.models.backbones.clay_v1 import WAVELENGTHS
from terratorch.tasks import ClassificationTask, PixelwiseRegressionTask, SemanticSegmentationTask

NUM_CHANNELS = 6
NUM_CLASSES = 2
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)


@pytest.fixture(scope="session")
def model_factory() -> str:
    return "ClayModelFactory"


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))

@pytest.mark.parametrize("backbone", ["clay_v1_base"])
@pytest.mark.parametrize("decoder", ["UperNetDecoder"])
@pytest.mark.parametrize("loss", ["dice"])
def test_create_segmentation_task(backbone, decoder, loss, model_factory: ClayModelFactory):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "in_channels": NUM_CHANNELS,
        "pretrained": False,
        "num_classes": NUM_CLASSES,
        "bands": list(WAVELENGTHS.keys()),
    }

    if decoder == "UperNetDecoder":
        model_args["out_indices"] = [1, 2, 3, 4]
    SemanticSegmentationTask(
        model_args,
        model_factory,
        loss=loss,
    )

    gc.collect()

@pytest.mark.parametrize("backbone", ["clay_v1_base"])
@pytest.mark.parametrize("decoder", ["FCNDecoder"])
@pytest.mark.parametrize("loss", ["mae"])
def test_create_regression_task(backbone, decoder, loss, model_factory: ClayModelFactory):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "in_channels": NUM_CHANNELS,
        "pretrained": False,
        "bands": list(WAVELENGTHS.keys()),
    }

    if decoder == "UperNetDecoder":
        model_args["out_indices"] = [1, 2, 3, 4]

    PixelwiseRegressionTask(
        model_args,
        model_factory,
        loss=loss,
    )

    gc.collect()

@pytest.mark.parametrize("backbone", ["clay_v1_base"])
@pytest.mark.parametrize("decoder", ["IdentityDecoder"])
@pytest.mark.parametrize("loss", ["bce"])
def test_create_classification_task(backbone, decoder, loss, model_factory: ClayModelFactory):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "in_channels": NUM_CHANNELS,
        "pretrained": False,
        "num_classes": NUM_CLASSES,
        "bands": list(WAVELENGTHS.keys()),
    }

    if decoder == "UperNetDecoder":
        model_args["out_indices"] = [1, 2, 3, 4]

    ClassificationTask(
        model_args,
        model_factory,
        loss=loss,
    )

    gc.collect()
