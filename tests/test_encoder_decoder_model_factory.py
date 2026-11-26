# Copyright contributors to the Terratorch project

import gc
import importlib

import pytest
import torch

from terratorch.models import EncoderDecoderFactory
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.models.model import AuxiliaryHead

NUM_CHANNELS = 6
NUM_CLASSES = 2
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

LORA_CONFIG = {
    "prithvi": {
        "method": "LORA",
        "replace_qkv": "qkv",  # As we want to apply LoRA separately and only to Q and V, we need to separate the matrix.
        "peft_config_kwargs": {
            "target_modules": ["qkv.q_linear", "qkv.v_linear", "mlp.fc1", "mlp.fc2"],
            "lora_alpha": 16,
            "r": 16,
        },
    },
    "clay": {
        "method": "LORA",
        "replace_qkv": "to_qkv",
        "peft_config_kwargs": {
            "target_modules": ["to_qkv.q_linear", "to_qkv.v_linear", "1.net.1", "1.net.3"],
            "lora_alpha": 16,
            "r": 16,
        },
    },
}


@pytest.fixture(scope="session")
def model_factory() -> EncoderDecoderFactory:
    return EncoderDecoderFactory()


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))


def test_unused_args_raise_exception(model_factory: EncoderDecoderFactory):
    with pytest.raises(ValueError) as excinfo:
        model_factory.build_model(
            "classification",
            backbone="prithvi_eo_v1_100",
            decoder="IdentityDecoder",
            backbone_bands=PRETRAINED_BANDS,
            backbone_pretrained=False,
            num_classes=NUM_CLASSES,
            unused_argument="unused_argument",
        )
    assert "unused_argument" in str(excinfo.value)

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v2_300"])
def test_create_classification_model(backbone, model_factory: EncoderDecoderFactory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_bands=PRETRAINED_BANDS,
        backbone_pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v2_tiny_tl"])
def test_create_classification_model_no_in_channels(backbone, model_factory: EncoderDecoderFactory, model_input):
    model = model_factory.build_model(
        "classification",
        backbone=backbone,
        decoder="IdentityDecoder",
        backbone_bands=PRETRAINED_BANDS,
        backbone_pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_CLASSIFICATION_OUTPUT_SHAPE

    gc.collect()


def test_linear_decoder_segmentation():
    class TinyBackbone(torch.nn.Module):
        out_channels = [8]

        def forward(self, x):
            return [x]

    model = EncoderDecoderFactory().build_model(
        task="segmentation",
        backbone=TinyBackbone(),
        decoder="LinearDecoder",
        decoder_upsampling_size=1,
        num_classes=2,
    )

    assert isinstance(model.head, torch.nn.Identity)
    with torch.no_grad():
        assert model(torch.ones(1, 8, 4, 4)).output.shape == (1, 2, 4, 4)


@pytest.mark.parametrize("backbone", ["prithvi_eo_v2_100_tl"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["UNetDecoder"])
def test_create_pixelwise_model(backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input):
    if decoder == "LinearDecoder" and task == "regression":
        pytest.skip("LinearDecoder is not supported for regression tasks")
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]
    if decoder == "LinearDecoder":
        model_args["decoder_upsampling_size"] = 16
        model_args["rescale"] = False

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
def test_create_model_with_smp_fpn_decoder(backbone, task, expected, model_factory: EncoderDecoderFactory, model_input):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": "smp_FPN",
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "necks": VIT_UPERNET_NECK,
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
def test_create_model_with_smp_unet_decoder(
    backbone, task, expected, model_factory: EncoderDecoderFactory, model_input
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": "smp_Unet",
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "decoder_decoder_channels": [256, 128, 64],
        "necks": VIT_UPERNET_NECK,
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.skip(reason="Failing without clear reason.")
@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
def test_create_model_with_smp_deeplabv3plus_decoder(
    backbone, task, expected, model_factory: EncoderDecoderFactory, model_input
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": "smp_DeepLabV3Plus",
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "necks": VIT_UPERNET_NECK + [{"name": "AddBottleneckLayer"}],
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == tuple(expected)

    gc.collect()


@pytest.mark.skipif(not importlib.util.find_spec("mmseg"), reason="mmsegmentation not installed")
@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
def test_create_model_with_mmseg_fcn_decoder(
    backbone, task, expected, model_factory: EncoderDecoderFactory, model_input
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": "mmseg_FCNHead",
        "decoder_channels": 128,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "necks": [
            {"name": "SelectIndices", "indices": [-1]},
            {"name": "ReshapeTokensToImage"},
        ],
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
    else:
        model_args["num_classes"] = 1

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.skipif(not importlib.util.find_spec("mmseg"), reason="mmsegmentation not installed")
@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
def test_create_model_with_mmseg_uperhead_decoder(
    backbone, task, expected, model_factory: EncoderDecoderFactory, model_input
):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": "mmseg_UPerHead",
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "decoder_channels": 256,
        "decoder_in_index": [0, 1, 2, 3],
        "necks": [
            {"name": "SelectIndices", "indices": [0, 1, 2, 3]},
            {"name": "ReshapeTokensToImage"},
        ],
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
    else:
        model_args["num_classes"] = 1

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v2_300_tl"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["FCNDecoder"])
def test_create_pixelwise_model_no_in_channels(
    backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input
):
    if decoder == "LinearDecoder" and task == "regression":
        pytest.skip("LinearDecoder is not supported for regression tasks")
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
    }

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]
    if decoder == "LinearDecoder":
        model_args["decoder_upsampling_size"] = 16
        model_args["rescale"] = False

    model = model_factory.build_model(**model_args)
    model.eval()

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["UperNetDecoder"])
def test_create_pixelwise_model_with_aux_heads(
    backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input
):
    aux_heads_name = ["first_aux", "second_aux"]
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_in_chans": NUM_CHANNELS,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "aux_decoders": [AuxiliaryHead(name, "FCNDecoder", None) for name in aux_heads_name],
    }
    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]

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
@pytest.mark.parametrize("decoder", ["FCNDecoder"])
def test_create_pixelwise_model_with_extra_bands(
    backbone, task, expected, decoder, model_factory: EncoderDecoderFactory
):
    if decoder == "LinearDecoder" and task == "regression":
        pytest.skip("LinearDecoder is not supported for regression tasks")
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_in_chans": NUM_CHANNELS + 1,
        "backbone_bands": [*PRETRAINED_BANDS, 7],
        "backbone_pretrained": False,
    }
    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES

    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]
    if decoder == "LinearDecoder":
        model_args["decoder_upsampling_size"] = 16
        model_args["rescale"] = False
    model = model_factory.build_model(**model_args)
    model.eval()
    model_input = torch.ones((1, NUM_CHANNELS + 1, 224, 224))
    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300", "clay_v1_base"])
@pytest.mark.parametrize("task,expected", PIXELWISE_TASK_EXPECTED_OUTPUT)
@pytest.mark.parametrize("decoder", ["UNetDecoder"])
def test_create_model_with_lora(backbone, task, expected, decoder, model_factory: EncoderDecoderFactory, model_input):
    model_args = {
        "task": task,
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "peft_config": LORA_CONFIG[backbone.split("_")[0]],
    }

    if backbone == "clay_v1_base":
        model_args["backbone_img_size"] = 224

    if task == "segmentation":
        model_args["num_classes"] = NUM_CLASSES
    if decoder in ["UperNetDecoder", "UNetDecoder"]:
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]

    model = model_factory.build_model(**model_args)
    model.eval()

    encoder_trainable_params = 0
    for param in model.encoder.parameters():
        if param.requires_grad:
            encoder_trainable_params += param.numel()

    if backbone == "clay_v1_base":
        num_layers = 12
        embed_dim = 768
        linear_in_out = 768 * 4
    else:
        num_layers = len(model.encoder.blocks)
        embed_dim = model.encoder.embed_dim
        linear_in_out = model.encoder.blocks[0].mlp.fc1.out_features
    r = LORA_CONFIG[backbone.split("_")[0]]["peft_config_kwargs"]["r"]
    assert encoder_trainable_params == num_layers * (2 * embed_dim * r * 2 + 2 * linear_in_out * r + 2 * embed_dim * r)

    with torch.no_grad():
        assert model(model_input).output.shape == expected

    gc.collect()
