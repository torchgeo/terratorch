# Copyright contributors to the Terratorch project

import pytest
import torch

from terratorch.models.backbones.galileo import (
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    GalileoBackbone,
    _create_galileo_backbone,
    galileo_normalization,
)
from terratorch.registry import BACKBONE_REGISTRY

S2_S1_DEM = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "VV", "VH", "elevation"]


def nano(**kwargs) -> GalileoBackbone:
    return _create_galileo_backbone("nano", pretrained=False, patch_size=8, **kwargs).eval()


def test_space_time_input_and_temporal_pooling():
    """One output per block (Nano: 4), the last the final one; mean / concat over time; dict input."""
    x = torch.randn(1, len(SPACE_TIME_BANDS), 3, 32, 32)
    model, concat = nano(), nano(temporal_pooling="concat", n_timestamps=3)
    assert model.out_channels == [128] * 4 and concat.out_channels == [3 * 128] * 4
    with torch.no_grad():
        out = model(x)
        assert [o.shape for o in out] == [(1, 128, 4, 4)] * 4 and not torch.allclose(out[0], out[-1])
        assert concat(x)[-1].shape == (1, 3 * 128, 4, 4)
        assert model({"space_time": x[:, :, 0], "months": torch.tensor([[5]])})[-1].shape == (1, 128, 4, 4)
        with pytest.raises(ValueError, match="time steps"):
            concat(x[:, :, :2])


def test_bands_split():
    """Bands are routed to their families; with masked groups, block outputs are put back on the grid."""
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
        out = model(x)
    assert [o.shape for o in out] == [(1, 128, 4, 4)] * 4
    assert seen["st"].abs().max() == 0  # the DEM did not leak into the space-time family
    assert (seen["sp"][..., SPACE_BANDS.index("elevation")] == 5.0).all()


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


def test_masked_band_groups_not_averaged():
    """Only observed band groups enter the feature map; masked ones come back as identical filler tokens."""
    model = BACKBONE_REGISTRY.build(
        "galileo_nano", pretrained=False, patch_size=8, bands=["VV", "VH"], temporal_pooling="concat", n_timestamps=2
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
