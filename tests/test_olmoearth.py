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
            "olmoearth_v1_3_base",
        ]

        for model_name in expected_models:
            assert model_name in TERRATORCH_BACKBONE_REGISTRY, f"{model_name} not found in TERRATORCH_BACKBONE_REGISTRY"

    @skip_no_olmoearth
    def test_olmoearth_config_keys(self):
        """Verify config structure for all registered variants."""
        from terratorch.models.backbones.olmoearth import OLMOEARTH_CONFIGS

        for name, cfg in OLMOEARTH_CONFIGS.items():
            assert "model_id" in cfg, f"Missing model_id in config for {name}"
            assert "embed_dim" in cfg, f"Missing embed_dim in config for {name}"
            assert isinstance(cfg["embed_dim"], int), f"embed_dim should be int for {name}"


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
            assert outputs[0].shape == (1, 128, expected_h, expected_w), f"Wrong shape for input ({h}, {w})"
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
        """concat is built through the factory: backbone_temporal_pooling must reach the backbone."""
        from terratorch.models import EncoderDecoderFactory

        x = torch.randn(1, 12, 3, 32, 32)
        concat = EncoderDecoderFactory().build_model(
            task="segmentation", backbone="olmoearth_v1_nano", backbone_pretrained=False, backbone_patch_size=8,
            backbone_temporal_pooling="concat", backbone_n_timestamps=3, decoder="FCNDecoder", num_classes=2,
        ).encoder.eval()
        assert concat.out_channels == [3 * 128]
        with torch.no_grad():
            assert self._model()(x)[0].shape == (1, 128, 4, 4)
            assert concat(x)[0].shape == (1, 3 * 128, 4, 4)
            for t in (2, 1):  # T=1 imagery is not repeated over time, only e.g. srtm
                with pytest.raises(ValueError, match="time steps"):
                    concat(x[:, :, :t])

    @skip_no_olmoearth
    def test_multimodal_bands(self):
        """A stacked S2 + S1 tensor equals the native per-modality dict; modality concat keeps both."""
        from terratorch.models.backbones.olmoearth import _olmoearth_api

        bands = {"sentinel2_l2a": self.S2, "sentinel1": ["vv", "vh"]}
        mean, concat = self._model(bands=bands), self._model(bands=bands, modality_pooling="concat")
        concat.load_state_dict(mean.state_dict())
        x = torch.randn(1, 14, 32, 32)
        order = [self.S2.index(b) for b in _olmoearth_api().Modality.get("sentinel2_l2a").band_order]
        with torch.no_grad():
            m, c = mean(x)[0], concat(x)[0]
            assert torch.allclose(m, mean({"sentinel2_l2a": x[:, order], "sentinel1": x[:, 12:]})[0])
        assert concat.out_channels == [2 * 128] and c.shape == (1, 256, 4, 4)
        assert torch.allclose(m, (c[:, :128] + c[:, 128:]) / 2, atol=1e-6)

    @skip_no_olmoearth
    def test_band_subset_masks_missing_band_sets(self):
        """RGBN fills Sentinel-2's first band set; the other two are masked as missing and removed from attention."""
        rgbn = ["B04", "B03", "B02", "B08"]
        model = self._model(bands={"sentinel2_l2a": rgbn}).train()
        assert model.missing_band_sets == {"sentinel2_l2a": [1, 2]}
        x = torch.randn(2, 4, 32, 32)
        torch.manual_seed(0)
        out = model(x)[0]
        assert out.shape == (2, 128, 4, 4)
        # skipping the all-True attention mask is exact: same as the encoder's own masked path
        model.encoder._maybe_get_attn_mask = type(model.encoder)._maybe_get_attn_mask.__get__(model.encoder)
        torch.manual_seed(0)
        assert torch.allclose(out, model(x)[0], atol=1e-5)

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
        with pytest.raises(ValueError, match="out_indices"):
            self._model(out_indices=[2, 5, 8, 11])  # nano has 4 blocks


class TestOlmoEarthImportGuard:
    """Test behavior when olmoearth-pretrain is not installed."""

    def test_import_does_not_fail_without_package(self):
        """The __init__.py import should not fail even without olmoearth-pretrain."""
        # This test always passes because __init__.py uses try/except
        # If import failed, the entire test suite would fail to collect
        pass


@skip_no_olmoearth
def test_olmoearth_normalization_matches_normalizer():
    """Datamodule-style (x - mean) / std with the helper's values == OlmoEarth's Normalizer."""
    import numpy as np

    from terratorch.models.backbones.olmoearth import _olmoearth_api, olmoearth_normalization

    try:
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import Normalizer
    except ImportError:
        from functools import partial

        from olmoearth_pretrain.data.normalize import Normalizer, Strategy

        Normalizer = partial(Normalizer, Strategy.COMPUTED)
    spec = _olmoearth_api().Modality.get("sentinel1")
    means, stds = olmoearth_normalization({"sentinel1": ["vh", "vv"]})  # reversed band order
    x = np.random.uniform(-30, 0, (1, 4, 4, 1, 2)).astype(np.float32)  # [..., vv, vh]
    expected = Normalizer().normalize(spec, x)
    ours = (x[..., ::-1] - np.array(means)) / np.array(stds)
    assert np.allclose(ours[..., ::-1], expected, atol=1e-5)
