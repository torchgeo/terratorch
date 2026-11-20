# Copyright contributors to the Terratorch project

import pytest
import torch

from terratorch.models.necks import TemporalPooling, build_neck_list

T, C, B, H = 4, 8, 2, 6


def test_mean_averages_time_steps():
    neck = TemporalPooling([C * T, C * T], n_timestamps=T)
    x = torch.cat([torch.full((B, C, H, H), float(t)) for t in range(T)], dim=1)  # time step t holds t
    out = neck([x, x])
    assert [o.shape for o in out] == [(B, C, H, H)] * 2
    assert torch.allclose(out[0], torch.full((B, C, H, H), 1.5))


def test_linear_is_weighted_sum():
    neck = TemporalPooling([C * T], n_timestamps=T, pooling="linear")
    assert sum(p.numel() for p in neck.parameters()) == T + 1
    x = torch.cat([torch.full((B, C, H, H), float(t)) for t in range(T)], dim=1)
    w, b = neck.tmap[0].weight[0].detach(), neck.tmap[0].bias.detach()
    expected = sum(w[t] * t for t in range(T)) + b
    assert torch.allclose(neck([x])[0], expected.expand(B, C, H, H), atol=1e-6)


def test_build_neck_list_channels():
    necks, channels = build_neck_list([{"name": "TemporalPooling", "n_timestamps": T}], [C * T, 2 * C * T])
    assert channels == [C, 2 * C]


def test_rejects_indivisible_channels():
    with pytest.raises(ValueError, match="divisible"):
        TemporalPooling([C * T + 1], n_timestamps=T)


def test_conv_merges_time_and_channels():
    neck = TemporalPooling([C * T, 2 * C * T], n_timestamps=T, pooling="conv")
    assert sum(p.numel() for p in neck.parameters()) == (C * T * C + C) + (2 * C * T * 2 * C + 2 * C)
    out = neck([torch.randn(B, C * T, H, H), torch.randn(B, 2 * C * T, H, H)])
    assert [o.shape for o in out] == [(B, C, H, H), (B, 2 * C, H, H)]
    # equals a per-pixel linear map over the full T*C concatenation
    x = torch.randn(B, C * T, H, H)
    conv = neck.tmap[0]
    expected = torch.einsum("oi,bihw->bohw", conv.weight[:, :, 0, 0], x) + conv.bias[None, :, None, None]
    assert torch.allclose(neck([x, torch.randn(B, 2 * C * T, H, H)])[0], expected, atol=1e-5)
