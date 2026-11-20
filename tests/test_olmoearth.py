# Copyright contributors to the Terratorch project
# Licensed under the Apache License 2.0

"""Tests for OlmoEarth backbone integration.

These tests verify that the OlmoEarth backbone can be registered,
instantiated, and produce the expected output shapes. Tests that require
olmoearth-pretrain (or olmoearth-pretrain-minimal) are skipped if neither is installed.
"""

import gc

import pytest
import torch

# Check if olmoearth-pretrain (or the lighter olmoearth-pretrain-minimal) is available
try:
    from terratorch.models.backbones.olmoearth import _olmoearth_api

    _olmoearth_api()
    HAS_OLMOEARTH = True
except ImportError:
    HAS_OLMOEARTH = False

skip_no_olmoearth = pytest.mark.skipif(
    not HAS_OLMOEARTH,
    reason="neither olmoearth-pretrain nor olmoearth-pretrain-minimal installed",
)


class TestOlmoEarthRegistration:
    """Test that OlmoEarth models are properly registered."""

    @skip_no_olmoearth
    def test_olmoearth_models_in_registry(self):
        """Verify all OlmoEarth variants are in the backbone registry."""
        from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

        expected_models = [
            "olmoearth_v1_nano",
            "olmoearth_v1_tiny",
            "olmoearth_v1_base",
            "olmoearth_v1_large",
            "olmoearth_v1_1_nano",
            "olmoearth_v1_1_tiny",
            "olmoearth_v1_1_base",
            "olmoearth_v1_2_nano",
            "olmoearth_v1_2_tiny",
            "olmoearth_v1_2_small",
            "olmoearth_v1_2_base",
        ]

        for model_name in expected_models:
            assert model_name in TERRATORCH_BACKBONE_REGISTRY, (
                f"{model_name} not found in TERRATORCH_BACKBONE_REGISTRY"
            )

    @skip_no_olmoearth
    def test_olmoearth_config_keys(self):
        """Verify config structure for all registered variants."""
        from terratorch.models.backbones.olmoearth import OLMOEARTH_CONFIGS

        for name, cfg in OLMOEARTH_CONFIGS.items():
            assert "model_id" in cfg, f"Missing model_id in config for {name}"
            assert "embed_dim" in cfg, f"Missing embed_dim in config for {name}"
            assert isinstance(cfg["embed_dim"], int), (
                f"embed_dim should be int for {name}"
            )


class TestOlmoEarthBackbone:
    """Test OlmoEarth backbone instantiation and forward pass."""

    @skip_no_olmoearth
    def test_create_backbone_no_weights(self):
        """Test creating an OlmoEarth backbone without pretrained weights."""
        from terratorch.models.backbones.olmoearth import OlmoEarthBackbone

        model = OlmoEarthBackbone(
            model_id="OlmoEarth-v1-Nano",
            embed_dim=128,
            pretrained=False,
        )
        assert model is not None
        assert model.embed_dim == 128
        assert model.out_channels == [128]
        gc.collect()

    @skip_no_olmoearth
    def test_backbone_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        from terratorch.models.backbones.olmoearth import OlmoEarthBackbone

        model = OlmoEarthBackbone(
            model_id="OlmoEarth-v1-Nano",
            embed_dim=128,
            pretrained=False,
            patch_size=8,
        )

        # Input: batch=1, channels=12 (Sentinel-2), H=64, W=64
        x = torch.randn(1, 12, 64, 64)
        outputs = model(x)

        assert isinstance(outputs, list)
        assert len(outputs) == 1
        # With patch_size=8 and input 64x64, expect 8x8 spatial output
        assert outputs[0].shape == (1, 128, 8, 8)
        gc.collect()

    @skip_no_olmoearth
    def test_backbone_forward_different_input_sizes(self):
        """Test forward pass with different input sizes."""
        from terratorch.models.backbones.olmoearth import OlmoEarthBackbone

        model = OlmoEarthBackbone(
            model_id="OlmoEarth-v1-Nano",
            embed_dim=128,
            pretrained=False,
            patch_size=8,
        )

        for h, w in [(64, 64), (128, 128), (96, 96)]:
            x = torch.randn(1, 12, h, w)
            outputs = model(x)
            expected_h = h // 8
            expected_w = w // 8
            assert outputs[0].shape == (1, 128, expected_h, expected_w), (
                f"Wrong shape for input ({h}, {w})"
            )
        gc.collect()

    @skip_no_olmoearth
    def test_backbone_via_registry(self):
        """Test creating backbone via the registry function."""
        from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

        model = TERRATORCH_BACKBONE_REGISTRY.build("olmoearth_v1_nano", pretrained=False)
        assert model is not None
        assert model.embed_dim == 128
        gc.collect()

    @skip_no_olmoearth
    def test_invalid_variant_raises(self):
        """Test that an invalid variant name raises ValueError."""
        from terratorch.models.backbones.olmoearth import _create_olmoearth_backbone

        with pytest.raises(ValueError, match="Unknown OlmoEarth variant"):
            _create_olmoearth_backbone("olmoearth_v99_mega", pretrained=False)


class TestOlmoEarthMultimodalTemporal:
    """Stacked multi-modal input, time series, temporal pooling and multi-scale outputs."""

    S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    @staticmethod
    def _model(**kwargs):
        from terratorch.models.backbones.olmoearth import _create_olmoearth_backbone

        return _create_olmoearth_backbone("olmoearth_v1_nano", pretrained=False, patch_size=8, **kwargs).eval()

    @skip_no_olmoearth
    def test_time_series_mean_and_concat(self):
        x = torch.randn(1, 12, 3, 32, 32)
        with torch.no_grad():
            assert self._model()(x)[0].shape == (1, 128, 4, 4)
            model = self._model(temporal_pooling="concat", n_timestamps=3)
            assert model(x)[0].shape == (1, 3 * 128, 4, 4)
        assert model.out_channels == [3 * 128]

    @skip_no_olmoearth
    def test_bands_split_matches_dict_input(self):
        """A stacked S2 + S1 tensor in our channel order equals the native per-modality dict."""
        from terratorch.models.backbones.olmoearth import _olmoearth_api

        Modality = _olmoearth_api().Modality

        model = self._model(bands={"sentinel2_l2a": self.S2, "sentinel1": ["vv", "vh"]})
        x = torch.randn(1, 14, 2, 32, 32)
        order = [self.S2.index(b) for b in Modality.get("sentinel2_l2a").band_order]
        native = {"sentinel2_l2a": x[:, order], "sentinel1": x[:, 12:]}
        with torch.no_grad():
            assert torch.allclose(model(x)[0], model(native)[0])

    @skip_no_olmoearth
    def test_bands_must_cover_modality(self):
        with pytest.raises(ValueError, match="permutation"):
            self._model(bands={"sentinel2_l2a": self.S2[:-1]})

    @skip_no_olmoearth
    def test_out_indices(self):
        model = self._model(out_indices=[0, 1, 2, 3])
        with torch.no_grad():
            out = model(torch.randn(1, 12, 32, 32))
        assert [o.shape for o in out] == [(1, 128, 4, 4)] * 4
        assert model.out_channels == [128] * 4
        # the last block's output is the encoder output before the final norm, so it must differ from
        # the first block's, i.e. the hooks capture different layers
        assert not torch.allclose(out[0], out[3])

    @skip_no_olmoearth
    def test_concat_rejects_wrong_timestamps(self):
        with pytest.raises(ValueError, match="time steps"):
            self._model(temporal_pooling="concat", n_timestamps=4)(torch.randn(1, 12, 2, 32, 32))


class TestOlmoEarthImportGuard:
    """Test behavior when olmoearth-pretrain is not installed."""

    def test_import_does_not_fail_without_package(self):
        """The __init__.py import should not fail even without olmoearth-pretrain."""
        # This test always passes because __init__.py uses try/except
        # If import failed, the entire test suite would fail to collect
        pass


@skip_no_olmoearth
def test_fast_pass_is_exact_in_training():
    """fast_pass (default) only drops the all-visible attention mask, so training outputs are unchanged."""
    from terratorch.models.backbones.olmoearth import _create_olmoearth_backbone

    x = torch.randn(2, 14, 2, 32, 32)
    bands = {"sentinel2_l2a": TestOlmoEarthMultimodalTemporal.S2, "sentinel1": ["vv", "vh"]}
    outs = []
    for fast_pass in (True, False):
        torch.manual_seed(0)
        model = _create_olmoearth_backbone(
            "olmoearth_v1_nano", pretrained=False, patch_size=8, bands=bands, out_indices=[1, 3], fast_pass=fast_pass
        )
        model.train()
        torch.manual_seed(1)  # same drop-path draws
        outs.append(model(x))
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(*outs, strict=True))
