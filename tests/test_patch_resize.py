# Copyright contributors to the Terratorch project

import gc

import pytest
import torch

from terratorch.models.utils import PatchResize, TemporalWrapper, _get_backbone


@pytest.mark.parametrize(("patch_size", "grid"), [(16, 16), (8, 32), (4, 64)])
def test_prithvi_token_grid(patch_size, grid):
    model = _get_backbone(
        "prithvi_eo_v2_tiny_tl",
        patch_resize=patch_size,
        pretrained=False,
        num_frames=2,
        img_size=256,
        bands=["BLUE", "GREEN", "RED"],
    )
    assert isinstance(model, PatchResize) and model.patch_size == patch_size
    with torch.no_grad():
        out = model(torch.randn(1, 3, 2, 256, 256))
    assert (out[-1].shape[1] - 1) // 2 == grid**2  # cls token, then grid tokens per time step
    gc.collect()


def test_terramind_dict_input_inside_temporal_wrapper():
    model = _get_backbone(
        "terramind_v1_tiny",
        patch_resize=8,
        use_temporal=True,
        temporal_pooling="mean",
        pretrained=False,
        modalities=["S2L2A", "S1RTC"],
    )
    assert isinstance(model, TemporalWrapper) and isinstance(model.encoder, PatchResize)
    x = {"S2L2A": torch.randn(1, 12, 2, 64, 64), "S1RTC": torch.randn(1, 2, 2, 64, 64)}
    with torch.no_grad():
        out = model(x)
    assert out[-1].shape[1] == 8**2  # 64 px at patch size 8
    gc.collect()


def test_encoder_decoder_factory():
    from terratorch.models import EncoderDecoderFactory

    model = EncoderDecoderFactory().build_model(
        task="segmentation",
        backbone="terramind_v1_tiny",
        backbone_pretrained=False,
        backbone_modalities=["S2L2A"],
        backbone_patch_resize=8,
        necks=[{"name": "ReshapeTokensToImage"}],
        decoder="FCNDecoder",
        num_classes=2,
    )
    model.eval()
    with torch.no_grad():
        out = model({"S2L2A": torch.randn(1, 12, 64, 64)}).output
    assert out.shape == (1, 2, 64, 64)
    gc.collect()


def test_native_patch_size_resolution(monkeypatch):
    seen = {}

    class Dummy(torch.nn.Module):
        out_channels = [1]

    class Embed(torch.nn.Module):
        patch_size = [1, 8, 8]  # Prithvi-style

    def build(name, **kwargs):
        seen.update(kwargs)
        model = Dummy()
        if kwargs.get("with_embed"):
            model.embed = Embed()
        return model

    monkeypatch.setattr("terratorch.models.utils.BACKBONE_REGISTRY.build", build)
    with pytest.warns(UserWarning, match="native patch size 16"):
        assert _get_backbone("x", patch_resize=8).scale == 2
    with pytest.warns(UserWarning, match="native patch size 8"):
        assert _get_backbone("x", patch_resize=4, with_embed=True).scale == 2  # inferred native 8
    assert _get_backbone("x", patch_resize=4, with_embed=True, patch_size=16).scale == 4  # kwarg wins
    assert seen.pop("patch_size") == 16
    model = _get_backbone("x", patch_resize=4, patch_resize_native=32)
    assert model.scale == 8 and "patch_size" not in seen and "patch_resize_native" not in seen


def test_freeze_uses_wrapped_freeze():
    class Encoder(torch.nn.Module):
        out_channels = [1]

        def __init__(self):
            super().__init__()
            self.frozen, self.prompt = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)

        def freeze(self):
            self.frozen.requires_grad_(False)

    model = PatchResize(Encoder(), patch_size=8)
    model.freeze()
    assert not model.encoder.frozen.weight.requires_grad and model.encoder.prompt.weight.requires_grad
