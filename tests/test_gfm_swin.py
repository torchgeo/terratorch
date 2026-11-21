# Copyright contributors to the Terratorch project
import gc
from collections import OrderedDict

import pytest
import torch

from terratorch.datasets.utils import HLSBands, generate_bands_intervals
from terratorch.models.backbones.gfm_swin import (
    GFM_OUT_CHANNELS,
    GFM_PRETRAINED_BANDS,
    GFM_SWIN_BASE_ARGS,
    _load_gfm_checkpoint,
    gfm_checkpoint_filter_fn,
)
from terratorch.models.backbones.prithvi_swin import convert_weights_swin2mmseg
from terratorch.models.backbones.swin_encoder_decoder import MMSegSwinTransformer
from terratorch.registry import BACKBONE_REGISTRY

EXPECTED_PYRAMID = {
    192: [(1, 128, 48, 48), (1, 256, 24, 24), (1, 512, 12, 12), (1, 1024, 6, 6)],
    224: [(1, 128, 56, 56), (1, 256, 28, 28), (1, 512, 14, 14), (1, 1024, 7, 7)],
}

SIX_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

# The GFM checkpoint carries none of these, and they initialize to the identity.
ALLOWED_MISSING = {f"stages.{i}.norm.{s}" for i in range(4) for s in ("weight", "bias")} | {
    "head.fc.weight",
    "head.fc.bias",
}


def _build_swin(in_chans: int = 3) -> MMSegSwinTransformer:
    return MMSegSwinTransformer(**GFM_SWIN_BASE_ARGS, in_chans=in_chans)


def _synthetic_upstream_checkpoint(model: MMSegSwinTransformer) -> dict:
    """Build a fake upstream GFM checkpoint from a model's own state dict.

    Applies the inverse of the mmseg renames so the result looks like the released
    checkpoint: an `encoder.*` student, a `teacher.*` branch, pretraining heads, a
    SimMIM `mask_token`, and non-persistent `attn_mask` buffers.
    """
    upstream = OrderedDict()
    for key, value in model.state_dict().items():
        if key.startswith("head."):
            continue
        inverse = key.replace("stages", "layers", 1) if key.startswith("stages") else key
        inverse = inverse.replace("attn.w_msa.", "attn.")
        inverse = inverse.replace("ffn.layers.0.0.", "mlp.fc1.").replace("ffn.layers.1.", "mlp.fc2.")
        inverse = inverse.replace("patch_embed.projection", "patch_embed.proj")
        upstream[f"encoder.{inverse}"] = value
        upstream[f"teacher.{inverse}"] = torch.zeros_like(value)

    upstream["encoder.mask_token"] = torch.zeros(1, 1, GFM_SWIN_BASE_ARGS["embed_dim"])
    upstream["encoder.layers.0.blocks.1.attn_mask"] = torch.zeros(4, 36, 36)
    upstream["encoder.norm.weight"] = torch.ones(GFM_OUT_CHANNELS[-1])
    upstream["encoder.norm.bias"] = torch.zeros(GFM_OUT_CHANNELS[-1])
    upstream["decoder.0.weight"] = torch.zeros(1)
    upstream["projector.0.weight"] = torch.zeros(1)

    return {"model": upstream}


def test_can_create_gfm_swin_from_registry():
    backbone = BACKBONE_REGISTRY.build("gfm_swin_base", pretrained=False)

    assert backbone.out_channels == GFM_OUT_CHANNELS
    assert backbone.patch_size == GFM_SWIN_BASE_ARGS["patch_size"]

    gc.collect()


@pytest.mark.parametrize("input_size", [192, 224])
def test_gfm_swin_feature_pyramid(input_size):
    backbone = BACKBONE_REGISTRY.build("gfm_swin_base", pretrained=False)
    backbone.eval()  # MMSegSwinTransformer.eval() returns None, so never chain it

    with torch.no_grad():
        output = backbone(torch.randn(1, 3, input_size, input_size))

    assert isinstance(output, list)
    assert [tuple(t.shape) for t in output] == EXPECTED_PYRAMID[input_size]

    gc.collect()


def test_gfm_swin_out_indices_filters_channels_and_output():
    backbone = BACKBONE_REGISTRY.build("gfm_swin_base", pretrained=False, out_indices=(1, 3))
    backbone.eval()

    assert backbone.out_channels == [256, 1024]

    with torch.no_grad():
        output = backbone(torch.randn(1, 3, 224, 224))

    assert [tuple(t.shape) for t in output] == [(1, 256, 28, 28), (1, 1024, 7, 7)]

    gc.collect()


def test_gfm_swin_accepts_non_rgb_bands():
    backbone = BACKBONE_REGISTRY.build("gfm_swin_base", pretrained=False, model_bands=SIX_BANDS)
    backbone.eval()

    with torch.no_grad():
        output = backbone(torch.randn(1, len(SIX_BANDS), 224, 224))

    assert [tuple(t.shape) for t in output] == EXPECTED_PYRAMID[224]

    gc.collect()


def test_filter_fn_keeps_only_encoder_and_loads_cleanly():
    model = _build_swin()
    checkpoint = _synthetic_upstream_checkpoint(model)

    filtered = gfm_checkpoint_filter_fn(checkpoint, model, GFM_PRETRAINED_BANDS, GFM_PRETRAINED_BANDS)

    assert not any(k.startswith(("teacher", "decoder", "projector")) for k in filtered)
    assert "mask_token" not in filtered
    assert not any(k.endswith("attn_mask") for k in filtered)
    # GFM's single final norm has no mmseg counterpart and must be dropped.
    assert "norm.weight" not in filtered
    assert "norm.bias" not in filtered

    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)

    assert unexpected_keys == []
    assert set(missing_keys) <= ALLOWED_MISSING

    gc.collect()


def test_missing_stage_norms_are_seed_independent():
    """The absent per-stage norms init to the identity, so features must not depend on the seed."""
    checkpoint = _synthetic_upstream_checkpoint(_build_swin())

    outputs = []
    for seed in (0, 12345):
        torch.manual_seed(seed)
        model = _build_swin()
        filtered = gfm_checkpoint_filter_fn(checkpoint, model, GFM_PRETRAINED_BANDS, GFM_PRETRAINED_BANDS)
        model.load_state_dict(filtered, strict=False)
        model.eval()

        with torch.no_grad():
            stage_input = model.patch_embed(torch.ones(1, 3, 192, 192))
            features = []
            for stage in model.stages:
                normed, _, downsampled, down_hw_shape = stage(stage_input)
                features.append(normed)
                stage_input = (downsampled, down_hw_shape)
        outputs.append(features)

    for first, second in zip(outputs[0], outputs[1], strict=True):
        assert torch.equal(first, second)

    gc.collect()


def test_convert_weights_permutes_downsample():
    """Regression guard: mmseg's PatchMerging needs a [0, 2, 1, 3] reordering, not a copy.

    Skipping the permutation loads without error but silently produces wrong features, so
    assert the converter actually reorders. Tested through the public outer function because
    the correct_unfold_* helpers are nested closures.
    """
    reduction = torch.randn(8, 16)
    norm = torch.randn(16)
    converted = convert_weights_swin2mmseg(
        OrderedDict(
            {
                "layers.0.downsample.reduction.weight": reduction,
                "layers.0.downsample.norm.weight": norm,
            }
        )
    )

    new_reduction = converted["stages.0.downsample.reduction.weight"]
    new_norm = converted["stages.0.downsample.norm.weight"]

    assert not torch.equal(new_reduction, reduction)
    assert not torch.equal(new_norm, norm)
    # A permutation preserves the multiset of values.
    assert torch.equal(new_reduction.flatten().sort().values, reduction.flatten().sort().values)
    assert torch.equal(new_norm.sort().values, norm.sort().values)

    expected_norm = norm.reshape(4, 4)[[0, 2, 1, 3], :].transpose(0, 1).reshape(16)
    assert torch.equal(new_norm, expected_norm)

    gc.collect()


def test_pretrained_bands_land_in_correct_positions():
    """RGB weights must map onto the requested band positions, with extra bands initialized fresh."""
    source = _build_swin()
    checkpoint = _synthetic_upstream_checkpoint(source)
    original_proj = source.state_dict()["patch_embed.projection.weight"]

    model = _build_swin(in_chans=len(SIX_BANDS))
    model_bands = generate_bands_intervals([HLSBands.try_convert_to_hls_bands_enum(b) for b in SIX_BANDS])
    filtered = gfm_checkpoint_filter_fn(checkpoint, model, GFM_PRETRAINED_BANDS, model_bands)

    proj = filtered["patch_embed.projection.weight"]
    assert proj.shape == (GFM_SWIN_BASE_ARGS["embed_dim"], len(SIX_BANDS), 4, 4)

    # GFM_PRETRAINED_BANDS is [RED, GREEN, BLUE]; SIX_BANDS starts [BLUE, GREEN, RED].
    for model_position, pretrained_index in ((0, 2), (1, 1), (2, 0)):
        assert torch.equal(proj[:, model_position], original_proj[:, pretrained_index])

    # Bands absent from pretraining get fresh weights rather than copied RGB.
    for extra_position in (3, 4, 5):
        assert not any(torch.equal(proj[:, extra_position], original_proj[:, i]) for i in range(3))

    gc.collect()


def test_malformed_checkpoint_raises(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"not_model": {}}, path)

    with pytest.raises(ValueError, match="model"):
        _load_gfm_checkpoint(str(path))

    gc.collect()


@pytest.mark.slow
def test_gfm_swin_pretrained_downloads_from_hf():
    backbone = BACKBONE_REGISTRY.build("gfm_swin_base", pretrained=True)
    backbone.eval()

    with torch.no_grad():
        output = backbone(torch.randn(1, 3, 192, 192))

    assert [tuple(t.shape) for t in output] == EXPECTED_PYRAMID[192]

    gc.collect()
