# Copyright contributors to the Terratorch project
"""
This module should be removed when PrithviModelFactory is removed. For now, this tests backwards compatibility.
"""
import gc
import pytest
import torch

from terratorch.models import PrithviModelFactory
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.models.model import AuxiliaryHead
import gc 

NUM_CHANNELS = 6
NUM_CLASSES = 2
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)
EXPECTED_REGRESSION_OUTPUT_SHAPE = (1, 224, 224)
EXPECTED_CLASSIFICATION_OUTPUT_SHAPE = (1, NUM_CLASSES)

PIXELWISE_TASK_EXPECTED_OUTPUT = [
    ("regression", EXPECTED_REGRESSION_OUTPUT_SHAPE),
    ("segmentation", EXPECTED_SEGMENTATION_OUTPUT_SHAPE),
]

@pytest.fixture(scope="session")
def model_factory() -> PrithviModelFactory:
    return PrithviModelFactory()


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
def test_create_classification_model(backbone, model_factory: PrithviModelFactory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        in_channels=NUM_CHANNELS,
        bands=PRETRAINED_BANDS,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

    gc.collect()

@pytest.mark.parametrize("backbone", ["prithvi_eo_v2_300"])
def test_create_classification_model_no_in_channels(backbone, model_factory: PrithviModelFactory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        bands=PRETRAINED_BANDS,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

    gc.collect()

@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["FCNDecoder", "IdentityDecoder"])
def test_create_pixelwise_model(backbone, task, expected, decoder, model_factory: PrithviModelFactory, model_input):
    # UperNetDecoder is intentionally excluded: it requires a pyramidal feature map, which on a
    # plain ViT backbone used to be produced by the now-removed `scale_modules`. The deprecated
    # PrithviModelFactory forces necks=None, so the pyramidal neck migration path is unavailable
    # here; UperNet is covered via EncoderDecoderFactory in test_encoder_decoder_model_factory.py.
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "in_channels": NUM_CHANNELS,
        "bands": PRETRAINED_BANDS,
        "pretrained": False,
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()

@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["FCNDecoder"])
def test_create_pixelwise_model_no_in_channels(
    backbone, task, expected, decoder, model_factory: PrithviModelFactory, model_input
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "bands": PRETRAINED_BANDS,
        "pretrained": False,
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()

@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["FCNDecoder"])
def test_create_pixelwise_model_with_aux_heads(
    backbone, task, expected, decoder, model_factory: PrithviModelFactory, model_input
):
    # Uses FCNDecoder rather than UperNetDecoder: the deprecated PrithviModelFactory forces
    # necks=None, so UperNet has no pyramidal-input path now that scale_modules is removed.
    # UperNet + auxiliary heads is covered via EncoderDecoderFactory in
    # test_encoder_decoder_model_factory.py.
    aux_heads_name = ["first_aux", "second_aux"]
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "in_channels": NUM_CHANNELS,
        "bands": PRETRAINED_BANDS,
        "pretrained": False,
        "aux_decoders": [AuxiliaryHead(name, "FCNDecoder", None) for name in aux_heads_name],
    }
    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        model_output = model(model_input)
        assert model_output.output.shape == expected

        assert len(model_output.auxiliary_heads.keys() & aux_heads_name) == len(aux_heads_name)
        for _, output in model_output.auxiliary_heads.items():
            assert output.shape == expected

    gc.collect()

@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["IdentityDecoder"])
def test_create_pixelwise_model_with_extra_bands(backbone, task, expected, decoder, model_factory: PrithviModelFactory):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "in_channels": NUM_CHANNELS + 1,
        "bands": [*PRETRAINED_BANDS, 7],
        "pretrained": False,
    }
    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()
    model_input = torch.ones((1, NUM_CHANNELS + 1, 224, 224))
    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()
