import pytest
import torch

from terratorch.models.backbones import satmae
from terratorch.models.backbones.satmae import SATMAE_BANDS, WEIGHTS, SatMAEEncoder
from terratorch.registry import BACKBONE_REGISTRY


def _tiny(**kwargs):
    return SatMAEEncoder(variant="base", embed_dim=320, depth=3, num_heads=4, pretrained=False, **kwargs)


def test_tiny_encoder_tokens_maps_and_taps():
    model = _tiny(bands=SATMAE_BANDS, out_indices=[0, -1])
    seen = []
    handle = model.feature_taps["2"].register_forward_hook(lambda _module, _args, output: seen.append(output))
    features = model(torch.rand(2, 10, 96, 96))
    handle.remove()
    assert [feature.shape for feature in features] == [(2, 433, 320), (2, 433, 320)]
    assert model.prepare_features_for_image_model(features)[0].shape == (2, 320, 12, 12)
    assert seen[0].shape == (2, 320, 12, 12)


def test_bands_and_geometry_are_strict():
    with pytest.raises(ValueError, match="exact band order"):
        _tiny(bands=list(reversed(SATMAE_BANDS)))
    model = _tiny()
    with pytest.raises(ValueError, match="B, 10, 96, 96"):
        model(torch.rand(1, 10, 95, 96))


def test_checkpoint_drops_decoder_and_rejects_bad_encoder(tmp_path):
    model = _tiny()
    state = model.state_dict()
    assert "channel_cls_embed" in state
    state["decoder_pred.0.weight"] = torch.ones(1)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": state}, checkpoint)
    loaded = _tiny()
    loaded.load_checkpoint(checkpoint)
    assert torch.equal(loaded.cls_token, model.cls_token)
    state.pop("cls_token")
    torch.save({"model": state}, checkpoint)
    with pytest.raises(ValueError, match="Incompatible"):
        loaded.load_checkpoint(checkpoint)


def test_weight_metadata_and_registry():
    assert "terratorch_satmae_base" in BACKBONE_REGISTRY
    assert WEIGHTS == {
        "base": ("pretrain-vit-base-e199.pth", "1269e6a8255cee0affdeb6bb75776c86"),
        "large": ("pretrain-vit-large-e199.pth", "436bcd815194afdae38f3e023c74e9cd"),
    }


def test_registered_variants_use_fixed_architectures(monkeypatch):
    captured = []

    class CaptureEncoder:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(satmae, "SatMAEEncoder", CaptureEncoder)
    satmae.satmae_base(pretrained=False)
    satmae.satmae_large(pretrained=False)
    assert captured == [
        {"variant": "base", "embed_dim": 768, "depth": 12, "num_heads": 12, "pretrained": False},
        {"variant": "large", "embed_dim": 1024, "depth": 24, "num_heads": 16, "pretrained": False},
    ]
