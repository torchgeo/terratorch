# Copyright contributors to the Terratorch project
import gc
import importlib

import pytest
import torch

from terratorch.models import EncoderDecoderFactory
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.models.model import AuxiliaryHead
from terratorch.models.backbones import torchgeo_resnet as torchgeo_resnet

NUM_CHANNELS = 6
NUM_CLASSES = 2  #10
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)
EXPECTED_REGRESSION_OUTPUT_SHAPE = (1, 224, 224)
EXPECTED_CLASSIFICATION_OUTPUT_SHAPE = (1, NUM_CLASSES)

PIXELWISE_TASK_EXPECTED_OUTPUT = [
    ("regression", EXPECTED_REGRESSION_OUTPUT_SHAPE),
    ("segmentation", EXPECTED_SEGMENTATION_OUTPUT_SHAPE),
]

VIT_UPERNET_NECK = [
    {"name": "SelectIndices", "indices": [0, 1, 2, 3]},
    {"name": "ReshapeTokensToImage"},
    {"name": "LearnedInterpolateToPyramidal"},
]

PRETRAINED_BANDS = ["RED", "GREEN", "BLUE", "NIR_NARROW", "SWIR_1", "SWIR_2"]

@pytest.fixture(scope="session")
def model_factory() -> EncoderDecoderFactory:
    return EncoderDecoderFactory()


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))

backbones = ["fmow_resnet50_fmow_rgb_gassl",
             "satlas_resnet152_sentinel2_mi_ms",
             "ssl4eos12_resnet18_sentinel2_all_moco"]
@pytest.mark.parametrize("backbone", backbones)
def test_create_classification_model_resnet(backbone, model_factory: EncoderDecoderFactory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_model_bands=PRETRAINED_BANDS,
        backbone_pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

    gc.collect()

backbones = ["ssl4eos12_resnet50_sentinel2_all_decur"]
@pytest.mark.parametrize("backbone", backbones)
def test_create_classification_model_resnet_pretrained(backbone, model_factory: EncoderDecoderFactory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_model_bands=PRETRAINED_BANDS,
        backbone_pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

    gc.collect()

backbones = ["dofa_base_patch16_224"]
@pytest.mark.parametrize("backbone", backbones)
def test_create_classification_model_dofa(backbone, model_factory: EncoderDecoderFactory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_model_bands=PRETRAINED_BANDS,
        backbone_pretrained=False,
        num_classes=NUM_CLASSES,
        necks = [{"name": "PermuteDims", "new_order": [0, 2, 1]}]
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

    gc.collect()

backbones = ["satlas_swin_b_sentinel2_si_ms"]
@pytest.mark.parametrize("backbone", backbones)
def test_create_classification_model_swin(backbone, model_factory: EncoderDecoderFactory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_model_bands=PRETRAINED_BANDS,
        backbone_pretrained=False,
        num_classes=NUM_CLASSES,
        necks = [{"name": "PermuteDims", "new_order": [0, 3, 1, 2]}]
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

    gc.collect()

@pytest.mark.parametrize("backbone", ["ssl4eos12_resnet50_sentinel2_all_decur"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["smp_Unet"])
def test_create_pixelwise_model_resnet(backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_model_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "backbone_out_indices": [0, 1, 2, 3, 4], 
        
    }
        
    if decoder == "smp_Unet":
        model_args["decoder_decoder_channels"] = [512, 256, 128, 64]
    
    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()

@pytest.mark.skip("Skip these tests for now.")
#@pytest.mark.parametrize("backbone", ["dofa_large_patch16_224"])
@pytest.mark.parametrize("backbone", ["dofa_base_patch16_224"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["IdentityDecoder"])
def test_create_pixelwise_model_dofa(backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_model_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "backbone_out_indices":  [1, 2, 3, 4] #[5, 11, 17, 23]
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
        
    model_args["necks"] = [{"name": "ReshapeTokensToImage"}]

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()

@pytest.mark.parametrize("backbone", ["satlas_swin_b_sentinel2_si_ms"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["UperNetDecoder"])
def test_create_pixelwise_model_swin(backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_model_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "backbone_out_indices": [1, 3, 5, 7]
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
        
    model_args["necks"] = [{"name": "PermuteDims", "new_order": [0, 3, 1, 2]}]

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["UperNetDecoder", "UNetDecoder"])
def test_create_pixelwise_model_gfm_swin(task, expected, decoder, model_factory: EncoderDecoderFactory, model_input):
    """GFM emits NCHW directly, so unlike the other Swin backbones it needs no PermuteDims neck."""
    model_args = {
        "task": task,
        "backbone": "gfm_swin_base",
        "decoder": decoder,
        "backbone_model_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [512, 256, 128, 64]

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()
