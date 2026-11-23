# Copyright contributors to the Terratorch project

import gc

import pytest
import torch

from terratorch.models.backbones.galileo import (
    GALILEO_STATS,
    GalileoBackbone,
    _create_galileo_backbone,
    galileo_normalization,
)
from terratorch.models.backbones.galileo import SPACE_BANDS, SPACE_TIME_BANDS
from terratorch.registry import BACKBONE_REGISTRY

S2_S1_DEM = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "VV", "VH", "elevation"]


def nano(**kwargs) -> GalileoBackbone:
    return _create_galileo_backbone("nano", pretrained=False, patch_size=8, **kwargs).eval()


@pytest.mark.parametrize("name", ["galileo_nano", "galileo_tiny", "galileo_base"])
def test_registered(name):
    assert name in BACKBONE_REGISTRY


def test_space_time_input_and_temporal_pooling():
    x = torch.randn(1, len(SPACE_TIME_BANDS), 3, 32, 32)
    with torch.no_grad():
        assert nano()(x)[0].shape == (1, 128, 4, 4)
        model = nano(temporal_pooling="concat", n_timestamps=3)
        assert model(x)[0].shape == (1, 3 * 128, 4, 4)
        assert nano()(x[:, :, 0])[0].shape == (1, 128, 4, 4)  # single image
    assert model.out_channels == [3 * 128]
    gc.collect()


def test_bands_split():
    with pytest.warns(UserWarning, match="B1"):
        model = nano(bands=S2_S1_DEM, compute_ndvi=False)
    # NDVI is neither given nor computed, so its group is masked; of the space family only SRTM is given
    assert model.masked_groups == {"space_time": [6], "space": [1, 2]}
    x = torch.zeros(1, len(S2_S1_DEM), 2, 32, 32)
    x[:, -1] = 5.0
    seen = {}
    forward = model.encoder.forward
    model.encoder.forward = lambda s_t_x, sp_x, *a, **k: seen.update(st=s_t_x, sp=sp_x) or forward(s_t_x, sp_x, *a, **k)
    with torch.no_grad():
        model(x)
    assert seen["st"].abs().max() == 0  # the DEM did not leak into the space-time family
    assert (seen["sp"][..., SPACE_BANDS.index("elevation")] == 5.0).all()
    gc.collect()


def test_out_indices_and_dict_input():
    model = nano(out_indices=[0, 1, 2, 3])
    with torch.no_grad():
        out = model({"space_time": torch.randn(1, len(SPACE_TIME_BANDS), 32, 32), "months": torch.tensor([[5]])})
    assert [o.shape for o in out] == [(1, 128, 4, 4)] * 4
    assert model.out_channels == [128] * 4
    assert not torch.allclose(out[0], out[3])
    gc.collect()


def test_out_indices_with_masked_tokens():
    with pytest.warns(UserWarning):
        model = nano(bands=S2_S1_DEM, out_indices=[1, 3])
    with torch.no_grad():
        out = model(torch.randn(2, len(S2_S1_DEM), 2, 32, 32))
    assert [o.shape for o in out] == [(2, 128, 4, 4)] * 2
    gc.collect()


def test_errors():
    with pytest.raises(ValueError, match="space-time bands"):
        nano()(torch.randn(1, 12, 32, 32))
    with pytest.raises(ValueError, match="time steps"):
        nano(temporal_pooling="concat", n_timestamps=4)(torch.randn(1, len(SPACE_TIME_BANDS), 2, 32, 32))


def test_ndvi_from_b4_b8():
    with pytest.warns(UserWarning):
        model = nano(bands=S2_S1_DEM)
    assert model.masked_groups["space_time"] == []  # NDVI is computed, so not masked
    means, stds = galileo_normalization(S2_S1_DEM)
    raw = torch.rand(1, len(S2_S1_DEM), 1, 8, 8) * 3000
    raw[:, S2_S1_DEM.index("B4"), 0, 0, 0] = -2007  # negative reflectance, as in PASTIS
    x = (raw - torch.tensor(means)[:, None, None, None]) / torch.tensor(stds)[:, None, None, None]
    red, nir = raw[:, S2_S1_DEM.index("B4")].clamp(min=0), raw[:, S2_S1_DEM.index("B8")]
    ndvi = model._ndvi(x[:, S2_S1_DEM.index("B4")], x[:, S2_S1_DEM.index("B8")])
    assert torch.isfinite(ndvi).all() and ndvi.abs().max() <= 1
    assert torch.allclose(ndvi, (nir - red) / (nir + red), atol=1e-4)


def test_galileo_normalization():
    means, stds = galileo_normalization(["VV", "B4", "NDVI", "DW_water", "WC_maize", "B1"])
    mean, std = GALILEO_STATS["VV"]
    assert means[:2] == [mean - 2 * std, GALILEO_STATS["B4"][0] - 2 * GALILEO_STATS["B4"][1]]
    assert stds[0] == 4 * std
    assert (means[2:], stds[2:]) == ([0.0] * 4, [1.0, 1.0, 100.0, 1.0])


def test_masked_band_groups_not_averaged():
    """Only observed band groups enter the feature map; masked ones come back as identical filler tokens."""
    model = BACKBONE_REGISTRY.build(
        "galileo_tiny", pretrained=False, patch_size=8, bands=["VV", "VH"], temporal_pooling="concat", n_timestamps=2
    ).eval()
    captured = {}
    model.encoder.register_forward_hook(lambda module, inputs, output: captured.update(out=output))
    with torch.no_grad():
        features = model(torch.randn(1, 2, 2, 32, 32))[-1]
    tokens, mask = captured["out"][0], captured["out"][4]
    observed = mask[0, 0, 0, 0] == 0
    assert 0 < observed.sum() < observed.numel()
    expected = tokens[..., observed, :].mean(dim=4).flatten(start_dim=3).permute(0, 3, 1, 2)
    torch.testing.assert_close(features, expected)
