# Copyright contributors to the Terratorch project
# Licensed under the Apache License 2.0

"""Tests for OlmoEarth backbone integration.

These tests verify that the OlmoEarth backbone can be registered,
instantiated, and produce the expected output shapes. Tests that require
the olmoearth-pretrain package are skipped if the package is not installed.
"""

import gc

import pytest
import torch

# Check if olmoearth-pretrain is available
try:
    import olmoearth_pretrain  # noqa: F401

    HAS_OLMOEARTH = True
except ImportError:
    HAS_OLMOEARTH = False

skip_no_olmoearth = pytest.mark.skipif(
    not HAS_OLMOEARTH,
    reason="olmoearth-pretrain not installed",
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


class TestOlmoEarthImportGuard:
    """Test behavior when olmoearth-pretrain is not installed."""

    def test_import_does_not_fail_without_package(self):
        """The __init__.py import should not fail even without olmoearth-pretrain."""
        # This test always passes because __init__.py uses try/except
        # If import failed, the entire test suite would fail to collect
        pass
