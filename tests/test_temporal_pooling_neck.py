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


def test_max_diff_subsets_and_passthrough():
    x = torch.cat([torch.full((B, C, H, H), float(t)) for t in range(T)], dim=1)
    assert torch.all(TemporalPooling([C * T], T, "max")([x])[0] == T - 1)
    # subsets [1, 3]: means 0 and 2, diff = first - second = -2
    out = TemporalPooling([C * T], T, "diff", subset_lengths=[1, 3])([x])[0]
    assert out.shape == (B, C, H, H) and torch.all(out == -2)
    for pooling in ("keep", "concat"):
        neck = TemporalPooling([C * T], T, pooling)
        assert neck([x])[0] is x and neck.process_channel_list([C * T]) == [C * T]
    with pytest.raises(ValueError, match="exactly two subsets"):
        TemporalPooling([C * T], T, "diff", subset_lengths=[1, 1, 2])


def test_token_dim():
    """(B, N, T*D) tokens pooled over T with dim=2, e.g. before ReshapeTokensToImage."""
    tokens = torch.randn(B, 16, T * C)
    out = TemporalPooling([C * T], T, "mean", dim=2)([tokens])[0]
    assert torch.allclose(out, tokens.unflatten(2, (T, C)).mean(dim=2))
    assert TemporalPooling([C * T], T, "linear", dim=-1)([tokens])[0].shape == (B, 16, C)
    with pytest.raises(ValueError, match="dim=1"):
        TemporalPooling([C * T], T, "conv", dim=2)


@pytest.mark.parametrize("pooling", ["linear", "conv"])
def test_temporal_wrapper_learned_pooling_matches_neck(pooling):
    """TemporalWrapper(pooling) == TemporalWrapper(concat) + TemporalPooling neck, same weights."""
    from terratorch.models.utils import TemporalWrapper

    encoder = torch.nn.Conv2d(3, C, 1)
    encoder.out_channels = [C]
    wrapped = TemporalWrapper(encoder, pooling=pooling, n_timestamps=T)
    concat = TemporalWrapper(encoder, pooling="concat", n_timestamps=T)
    neck = TemporalPooling(concat.out_channels, T, pooling)
    neck.load_state_dict(wrapped.temporal_pooling.state_dict())
    assert wrapped.out_channels == [C]
    x = torch.randn(B, 3, T, H, H)
    assert torch.allclose(wrapped(x)[0], neck(concat(x))[0], atol=1e-6)


def test_freeze_backbone_keeps_temporal_pooling_trainable():
    from terratorch.models import EncoderDecoderFactory

    model = EncoderDecoderFactory().build_model(
        task="segmentation",
        backbone="terramind_v1_tiny",
        backbone_pretrained=False,
        backbone_modalities=["S2L2A"],
        backbone_use_temporal=True,
        backbone_temporal_pooling="linear",
        backbone_temporal_n_timestamps=T,
        necks=[{"name": "ReshapeTokensToImage"}],
        decoder="FCNDecoder",
        num_classes=2,
    )
    model.freeze_encoder()
    wrapper = model.encoder
    assert not any(p.requires_grad for p in wrapper.encoder.parameters())
    assert all(p.requires_grad for p in wrapper.temporal_pooling.parameters())
