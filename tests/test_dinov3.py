import gc

import pytest
import torch

from terratorch.models.backbones.dinov3.dinov3_wrapper import DinoV3Wrapper


# Fixture for dummy input
@pytest.fixture
def dummy_input():
    return torch.randn(1, 3, 224, 224)  # Batch size 1, RGB, 224x224


# Fixture for model
@pytest.fixture
def model():
    return DinoV3Wrapper(model="dinov3_vits16", img_size=224)


# Test model initialization
def test_model_initialization(model):
    assert model.dinov3 is not None
    assert hasattr(model.dinov3, "get_intermediate_layers")


# Test forward pass
def test_forward_features(model, dummy_input):
    output = model.forward(dummy_input)
    assert isinstance(output, list)
    assert isinstance(output[0], torch.Tensor)
    # batch size match
    gc.collect()


# Test output shape consistency
def test_output_shape(model, dummy_input):
    output = model.forward(dummy_input)
    # Depending on the model, output shape may vary. Adjust as needed.
    assert len(output) == 12
    assert output[0].shape[0] == 1
    assert output[0].shape[1] == 197
    assert output[0].shape[2] == 384

    model.return_cls_token = False

    output = model.forward(dummy_input)
    assert output[0].shape[1] == 196
    gc.collect()


# Test model with different image size
def test_model_different_img_size():
    model = DinoV3Wrapper(model="dinov3_vits16", img_size=256)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model.forward(dummy_input)
    assert output[0].shape[0] == 1
    gc.collect()


S2_S1_DEM = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "VV", "VH", "DEM"]


@pytest.mark.parametrize(("name", "conv"), [("dinov3_vits16", "patch_embed.proj"), ("dinov3_convnext_tiny", "stem")])
def test_multispectral_bands_graft_rgb(name, conv):
    torch.manual_seed(0)
    rgb = DinoV3Wrapper(model=name)
    torch.manual_seed(0)  # same init, so the RGB kernels match the 3-channel model's
    model = DinoV3Wrapper(model=name, bands=S2_S1_DEM)
    get = (
        (lambda m: m.dinov3.patch_embed.proj)
        if conv == "patch_embed.proj"
        else (lambda m: m.dinov3.downsample_layers[0][0])
    )
    old, new = get(rgb).weight, get(model).weight
    assert new.shape[1] == len(S2_S1_DEM)
    for src, band in enumerate(["B4", "B3", "B2"]):  # pretrained channel order is RGB
        assert torch.equal(new[:, S2_S1_DEM.index(band)], old[:, src])
    with torch.no_grad():
        out = model(torch.randn(1, len(S2_S1_DEM), 64, 64))
    assert out[0].shape[0] == 1
    gc.collect()


def test_hls_band_names_and_norm():
    model = DinoV3Wrapper(
        model="dinov3_convnext_tiny", bands=["BLUE", "GREEN", "RED", "NIR_NARROW"], norm=False, return_cls_token=False
    )
    with torch.no_grad():
        out = model(torch.randn(1, 4, 64, 64))
    assert [o.shape for o in out] == [(1, 256, 96), (1, 64, 192), (1, 16, 384), (1, 4, 768)]
    gc.collect()
