# Copyright contributors to the Terratorch project

"""Backwards compatibility for the `scale_modules` option removed from UperNetDecoder.

Configs and checkpoints published before the removal set `decoder_scale_modules: true` and store
the pyramidal projections as `decoder.fpn1` ... `decoder.fpn4`. Both must keep working: the flag
is migrated to a `LearnedInterpolateToPyramidal` neck and the weights are renamed while loading.
"""

import gc
import warnings
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from terratorch.models import EncoderDecoderFactory
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.models.model import AuxiliaryHead
from terratorch.models.necks import LearnedInterpolateToPyramidal

NUM_CHANNELS = 6
NUM_CLASSES = 2
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)

# the necks published models used together with `decoder_scale_modules`
LEGACY_NECKS = [
    {"name": "SelectIndices", "indices": [0, 1, 2, 3]},
    {"name": "ReshapeTokensToImage"},
]
MIGRATED_NECKS = [*LEGACY_NECKS, {"name": "LearnedInterpolateToPyramidal"}]


@pytest.fixture(scope="module")
def model_factory() -> EncoderDecoderFactory:
    return EncoderDecoderFactory()


@pytest.fixture(scope="module")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))


def _model_args(**overrides) -> dict:
    args = {
        "task": "segmentation",
        "backbone": "prithvi_eo_v2_tiny_tl",
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "decoder": "UperNetDecoder",
        "decoder_channels": 256,
        "num_classes": NUM_CLASSES,
    }
    args.update(overrides)
    return args


def _build(model_factory: EncoderDecoderFactory, seed: int, **overrides) -> nn.Module:
    torch.manual_seed(seed)
    model = model_factory.build_model(**_model_args(**overrides))
    model.eval()
    return model


def _to_legacy_checkpoint(state_dict: dict, neck_index: int = 2) -> dict:
    """Rewrite a current state dict the way it was stored before scale_modules was removed."""
    neck_prefix = f"neck.{neck_index}.fpn"
    return {
        (f"decoder.fpn{k[len(neck_prefix) :]}" if k.startswith(neck_prefix) else k): v
        for k, v in state_dict.items()
    }


def test_legacy_flag_is_migrated_to_neck(model_factory: EncoderDecoderFactory, model_input):
    with pytest.warns(DeprecationWarning, match="decoder_scale_modules"):
        model = _build(model_factory, 0, necks=LEGACY_NECKS, decoder_scale_modules=True)

    assert isinstance(model.neck[-1], LearnedInterpolateToPyramidal)
    # the decoder no longer owns the projections
    assert not any(name.startswith(("fpn1", "fpn2", "fpn3", "fpn4")) for name, _ in model.decoder.named_modules())

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE

    gc.collect()


def test_legacy_flag_matches_explicit_neck(model_factory: EncoderDecoderFactory, model_input):
    """The migrated model must be the very same architecture as the explicit-neck one."""
    with pytest.warns(DeprecationWarning):
        legacy = _build(model_factory, 42, necks=LEGACY_NECKS, decoder_scale_modules=True)
    explicit = _build(model_factory, 43, necks=MIGRATED_NECKS)

    assert set(legacy.state_dict()) == set(explicit.state_dict())

    explicit.load_state_dict(legacy.state_dict())
    with torch.no_grad():
        assert torch.equal(legacy(model_input).output, explicit(model_input).output)

    gc.collect()


@contextmanager
def no_deprecation_warning():
    """Assert the scale_modules deprecation warning is not raised."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        yield
        assert not [r for r in records if "decoder_scale_modules" in str(r.message)]


def test_legacy_flag_false_adds_no_neck(model_factory: EncoderDecoderFactory):
    with no_deprecation_warning():
        model = _build(model_factory, 0, necks=MIGRATED_NECKS, decoder_scale_modules=False)

    assert len(model.neck) == len(MIGRATED_NECKS)

    gc.collect()


@pytest.mark.parametrize("necks", [LEGACY_NECKS, MIGRATED_NECKS])
def test_legacy_checkpoint_is_remapped(necks, model_factory: EncoderDecoderFactory, model_input):
    """A checkpoint holding `decoder.fpn*` loads into the neck, from either config style."""
    scale_modules = necks is LEGACY_NECKS
    with pytest.warns(DeprecationWarning) if scale_modules else no_deprecation_warning():
        trained = _build(model_factory, 7, necks=necks, decoder_scale_modules=scale_modules)

    legacy_checkpoint = _to_legacy_checkpoint(trained.state_dict())
    assert "decoder.fpn1.0.weight" in legacy_checkpoint

    with pytest.warns(DeprecationWarning) if scale_modules else no_deprecation_warning():
        restored = _build(model_factory, 8, necks=necks, decoder_scale_modules=scale_modules)

    incompatible = restored.load_state_dict(legacy_checkpoint)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    with torch.no_grad():
        assert torch.equal(trained(model_input).output, restored(model_input).output)

    gc.collect()


def test_legacy_checkpoint_is_remapped_under_a_prefix(model_factory: EncoderDecoderFactory, model_input):
    """Lightning checkpoints store the keys under a `model.` prefix."""

    class Wrapper(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model

    with pytest.warns(DeprecationWarning):
        trained = _build(model_factory, 11, necks=LEGACY_NECKS, decoder_scale_modules=True)
    legacy_checkpoint = {f"model.{k}": v for k, v in _to_legacy_checkpoint(trained.state_dict()).items()}

    with pytest.warns(DeprecationWarning):
        restored = Wrapper(_build(model_factory, 12, necks=LEGACY_NECKS, decoder_scale_modules=True))

    incompatible = restored.load_state_dict(legacy_checkpoint)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    with torch.no_grad():
        assert torch.equal(trained(model_input).output, restored.model(model_input).output)

    gc.collect()


def test_decoder_fpn_convs_are_not_remapped(model_factory: EncoderDecoderFactory):
    """`fpn_convs` and `fpn_bottleneck` belong to the decoder and must stay there."""
    with pytest.warns(DeprecationWarning):
        model = _build(model_factory, 13, necks=LEGACY_NECKS, decoder_scale_modules=True)

    state_dict = model.state_dict()
    assert any(k.startswith("decoder.fpn_convs") for k in state_dict)
    incompatible = model.load_state_dict(_to_legacy_checkpoint(state_dict))
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    gc.collect()


def test_legacy_flag_on_auxiliary_decoder(model_factory: EncoderDecoderFactory, model_input):
    """Auxiliary decoders read the neck output, so their own flag is dropped as well."""
    aux_decoders = [
        AuxiliaryHead(
            "aux",
            "UperNetDecoder",
            {"decoder_channels": 256, "decoder_scale_modules": True},
        )
    ]
    with pytest.warns(DeprecationWarning, match="decoder_scale_modules"):
        model = _build(
            model_factory,
            17,
            necks=LEGACY_NECKS,
            decoder_scale_modules=True,
            aux_decoders=aux_decoders,
        )

    assert isinstance(model.neck[-1], LearnedInterpolateToPyramidal)
    with torch.no_grad():
        output = model(model_input)
    assert output.output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE
    assert output.auxiliary_heads["aux"].shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE

    gc.collect()


def test_legacy_flag_is_ignored_when_neck_is_already_declared(model_factory: EncoderDecoderFactory, model_input):
    """A half-migrated config must not scale the features twice."""
    with pytest.warns(DeprecationWarning, match="already declares"):
        model = _build(model_factory, 19, necks=MIGRATED_NECKS, decoder_scale_modules=True)

    assert len(model.neck) == len(MIGRATED_NECKS)
    assert sum(isinstance(neck, LearnedInterpolateToPyramidal) for neck in model.neck) == 1

    with torch.no_grad():
        assert model(model_input).output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE

    gc.collect()


def test_legacy_flag_leaves_auxiliary_args_untouched(model_factory: EncoderDecoderFactory):
    """Building twice from the same arguments must give the same model both times."""
    aux_args = {"decoder_channels": 256, "decoder_scale_modules": True}
    aux_decoders = [AuxiliaryHead("aux", "UperNetDecoder", aux_args)]

    for _ in range(2):
        with pytest.warns(DeprecationWarning, match="decoder_scale_modules"):
            model = _build(model_factory, 23, necks=LEGACY_NECKS, aux_decoders=aux_decoders)
        assert isinstance(model.neck[-1], LearnedInterpolateToPyramidal)

    assert aux_args == {"decoder_channels": 256, "decoder_scale_modules": True}

    gc.collect()


def test_legacy_flag_leaves_decoder_kwargs_untouched(model_factory: EncoderDecoderFactory):
    """`decoder_kwargs` belongs to the caller and must survive the migration."""
    decoder_kwargs = {"channels": 256, "scale_modules": True}

    for _ in range(2):
        with pytest.warns(DeprecationWarning, match="decoder_scale_modules"):
            model = model_factory.build_model(
                task="segmentation",
                backbone="prithvi_eo_v2_tiny_tl",
                backbone_bands=PRETRAINED_BANDS,
                backbone_pretrained=False,
                decoder="UperNetDecoder",
                decoder_kwargs=decoder_kwargs,
                num_classes=NUM_CLASSES,
                necks=LEGACY_NECKS,
            )
        assert isinstance(model.neck[-1], LearnedInterpolateToPyramidal)

    assert decoder_kwargs == {"channels": 256, "scale_modules": True}

    gc.collect()


def test_decoder_owning_fpn_names_keeps_its_weights(model_factory: EncoderDecoderFactory):
    """A decoder with its own `fpn1` must keep those weights instead of losing them to the neck."""
    model = _build(model_factory, 29, necks=MIGRATED_NECKS)
    # a third-party decoder could legitimately name a submodule `fpn1`
    model.decoder.fpn1 = nn.Conv2d(4, 4, 1)

    state_dict = model.state_dict()
    assert "decoder.fpn1.weight" in state_dict

    incompatible = model.load_state_dict(state_dict)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    gc.collect()
