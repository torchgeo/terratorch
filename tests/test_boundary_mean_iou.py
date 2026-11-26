"""Comprehensive tests for BoundaryMeanIoU metric.

Tests cover initialization, update, compute methods with various configurations
to ensure maximum code coverage.
"""

import gc

import pytest
import torch

from terratorch.tasks.metrics import BoundaryMeanIoU


class TestBoundaryMeanIoUInitialization:
    """Test suite for BoundaryMeanIoU initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        metric = BoundaryMeanIoU(num_classes=5)
        
        assert metric.num_classes == 5
        assert metric.thickness == 4
        assert metric.ignore_index is None
        assert metric.average == "macro"
        assert metric.include_background is True
        assert metric.intersections.shape == torch.Size([5])
        assert metric.unions.shape == torch.Size([5])

    def test_initialization_with_custom_thickness(self):
        """Test initialization with custom thickness."""
        metric = BoundaryMeanIoU(num_classes=3, thickness=5)
        
        assert metric.thickness == 5

    def test_initialization_with_ignore_index(self):
        """Test initialization with ignore_index."""
        metric = BoundaryMeanIoU(num_classes=4, ignore_index=0)
        
        assert metric.ignore_index == 0

    def test_initialization_with_micro_average(self):
        """Test initialization with micro average."""
        metric = BoundaryMeanIoU(num_classes=3, average="micro")
        
        assert metric.average == "micro"

    def test_initialization_without_background(self):
        """Test initialization excluding background class."""
        metric = BoundaryMeanIoU(num_classes=5, include_background=False)
        
        assert metric.include_background is False

    def test_invalid_average_raises_error(self):
        """Test that invalid average parameter raises ValueError."""
        with pytest.raises(ValueError, match="average must be 'macro' or 'micro'"):
            BoundaryMeanIoU(num_classes=3, average="invalid")

    def test_initialization_with_all_parameters(self):
        """Test initialization with all custom parameters."""
        metric = BoundaryMeanIoU(
            num_classes=10,
            thickness=3,
            ignore_index=255,
            average="micro",
            include_background=False
        )
        
        assert metric.num_classes == 10
        assert metric.thickness == 3
        assert metric.ignore_index == 255
        assert metric.average == "micro"
        assert metric.include_background is False


class TestBoundaryMeanIoUUpdate:
    """Test suite for BoundaryMeanIoU update method."""

    def test_update_with_4d_predictions(self):
        """Test update with 4D predictions (logits/probs)."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        preds = torch.randn(2, 3, 32, 32)  # (N, C, H, W)
        target = torch.randint(0, 3, (2, 32, 32))  # (N, H, W)
        
        metric.update(preds, target)
        
        # Check that state was updated
        assert metric.intersections.sum() >= 0
        assert metric.unions.sum() >= 0

    def test_update_with_3d_predictions(self):
        """Test update with 3D predictions (class indices)."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        preds = torch.randint(0, 3, (2, 32, 32))  # (N, H, W)
        target = torch.randint(0, 3, (2, 32, 32))  # (N, H, W)
        
        metric.update(preds, target)
        
        assert metric.intersections.sum() >= 0
        assert metric.unions.sum() >= 0

    def test_update_with_perfect_prediction(self):
        """Test update when predictions match target perfectly."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, 10:20, 10:20] = 1
        target[:, 20:30, 20:30] = 2
        
        preds = target.clone()
        
        metric.update(preds, target)
        
        # With perfect predictions, should have high IoU
        iou = metric.compute()
        assert iou > 0.5

    def test_update_with_ignore_index(self):
        """Test update with ignore_index."""
        metric = BoundaryMeanIoU(num_classes=4, ignore_index=0)
        
        target = torch.randint(0, 4, (2, 32, 32))
        preds = torch.randint(0, 4, (2, 32, 32))
        
        # Set some pixels to ignore_index
        target[:, 0:5, 0:5] = 0
        
        metric.update(preds, target)
        
        assert metric.intersections.sum() >= 0

    def test_update_without_background(self):
        """Test update excluding background class."""
        metric = BoundaryMeanIoU(num_classes=4, include_background=False)
        
        target = torch.randint(0, 4, (2, 32, 32))
        preds = torch.randint(0, 4, (2, 32, 32))
        
        metric.update(preds, target)
        
        # Background class (0) should not be counted
        assert metric.intersections[0] == 0
        assert metric.unions[0] == 0

    def test_update_with_different_thicknesses(self):
        """Test update with various boundary thicknesses."""
        thicknesses = [1, 2, 3, 5]
        
        target = torch.zeros(1, 32, 32, dtype=torch.long)
        target[0, 10:20, 10:20] = 1
        preds = target.clone()
        
        for thickness in thicknesses:
            metric = BoundaryMeanIoU(num_classes=2, thickness=thickness)
            metric.update(preds, target)
            iou = metric.compute()
            assert iou >= 0.0

    def test_update_invalid_prediction_shape(self):
        """Test that invalid prediction shape raises ValueError."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        preds = torch.randn(2, 3)  # Invalid: 2D
        target = torch.randint(0, 3, (2, 32, 32))
        
        with pytest.raises(ValueError, match="preds must be"):
            metric.update(preds, target)

    def test_update_invalid_target_shape(self):
        """Test that invalid target shape raises ValueError."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        preds = torch.randint(0, 3, (2, 32, 32))
        target = torch.randint(0, 3, (2, 3, 32, 32))  # Invalid: 4D
        
        with pytest.raises(ValueError, match="target must be"):
            metric.update(preds, target)

    def test_update_multiple_batches(self):
        """Test updating metric with multiple batches."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        for _ in range(5):
            preds = torch.randint(0, 3, (2, 32, 32))
            target = torch.randint(0, 3, (2, 32, 32))
            metric.update(preds, target)
        
        # State should accumulate
        assert metric.unions.sum() > 0

    def test_update_with_binary_segmentation(self):
        """Test with binary segmentation (2 classes)."""
        metric = BoundaryMeanIoU(num_classes=2)
        
        target = torch.zeros(2, 64, 64, dtype=torch.long)
        target[:, 20:40, 20:40] = 1
        preds = target.clone()
        
        metric.update(preds, target)
        iou = metric.compute()
        
        assert iou > 0.5

    def test_update_with_no_boundary_pixels(self):
        """Test when there are no boundary pixels for some classes."""
        metric = BoundaryMeanIoU(num_classes=5)
        
        # All pixels belong to class 0
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        preds = torch.zeros(2, 32, 32, dtype=torch.long)
        
        metric.update(preds, target)
        
        # Classes 1-4 should have zero union
        assert metric.unions[1:].sum() == 0


class TestBoundaryMeanIoUCompute:
    """Test suite for BoundaryMeanIoU compute method."""

    def test_compute_macro_average(self):
        """Test compute with macro averaging."""
        metric = BoundaryMeanIoU(num_classes=3, average="macro")
        
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, 10:20, 10:20] = 1
        target[:, 20:30, 20:30] = 2
        preds = target.clone()
        
        metric.update(preds, target)
        result = metric.compute()
        
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # scalar
        assert 0.0 <= result <= 1.0

    def test_compute_micro_average(self):
        """Test compute with micro averaging."""
        metric = BoundaryMeanIoU(num_classes=3, average="micro")
        
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, 10:20, 10:20] = 1
        target[:, 20:30, 20:30] = 2
        preds = target.clone()
        
        metric.update(preds, target)
        result = metric.compute()
        
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0
        assert 0.0 <= result <= 1.0

    def test_compute_with_no_valid_classes(self):
        """Test compute when no classes have valid boundaries."""
        metric = BoundaryMeanIoU(num_classes=3, average="macro")
        
        # No update called, all unions are zero
        result = metric.compute()
        
        assert result == 0.0

    def test_compute_after_reset(self):
        """Test that compute works after reset."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        target = torch.randint(0, 3, (2, 32, 32))
        preds = torch.randint(0, 3, (2, 32, 32))
        
        metric.update(preds, target)
        result1 = metric.compute()
        
        metric.reset()
        
        result2 = metric.compute()
        assert result2 == 0.0
        assert torch.all(metric.intersections == 0)
        assert torch.all(metric.unions == 0)

    def test_compute_perfect_prediction(self):
        """Test compute with perfect predictions."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        target[0, 20:40, 20:40] = 1
        target[0, 40:60, 40:60] = 2
        preds = target.clone()
        
        metric.update(preds, target)
        result = metric.compute()
        
        # Perfect prediction should have IoU close to 1
        assert result > 0.9

    def test_compute_worst_prediction(self):
        """Test compute with worst case predictions."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        target[0, 20:40, 20:40] = 1
        
        # Predictions are completely different
        preds = torch.ones(1, 64, 64, dtype=torch.long) * 2
        preds[0, 40:60, 40:60] = 1
        
        metric.update(preds, target)
        result = metric.compute()
        
        # Should have low IoU
        assert 0.0 <= result < 0.5

    def test_compute_excludes_classes_with_zero_union(self):
        """Test that classes with zero union are excluded from macro average."""
        metric = BoundaryMeanIoU(num_classes=5, average="macro")
        
        # Only use classes 0 and 1
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, 10:20, 10:20] = 1
        preds = target.clone()
        
        metric.update(preds, target)
        result = metric.compute()
        
        # Should only average over classes with boundaries (0 and 1)
        assert result > 0.0


class TestBoundaryMeanIoUEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_pixel_objects(self):
        """Test with very small objects (single pixels)."""
        metric = BoundaryMeanIoU(num_classes=2, thickness=1)
        
        target = torch.zeros(1, 10, 10, dtype=torch.long)
        target[0, 5, 5] = 1
        preds = target.clone()
        
        metric.update(preds, target)
        result = metric.compute()
        
        assert result >= 0.0

    def test_large_image(self):
        """Test with large image."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        target = torch.randint(0, 3, (1, 512, 512))
        preds = torch.randint(0, 3, (1, 512, 512))
        
        metric.update(preds, target)
        result = metric.compute()
        
        assert 0.0 <= result <= 1.0
        gc.collect()

    def test_all_same_class(self):
        """Test when all pixels belong to the same class."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        target = torch.ones(2, 32, 32, dtype=torch.long)
        preds = torch.ones(2, 32, 32, dtype=torch.long)
        
        metric.update(preds, target)
        result = metric.compute()
        
        # Only class 1 has pixels, and it's perfect
        assert result >= 0.0

    def test_checkerboard_pattern(self):
        """Test with checkerboard pattern."""
        metric = BoundaryMeanIoU(num_classes=2, thickness=1)
        
        target = torch.zeros(1, 32, 32, dtype=torch.long)
        target[0, ::2, ::2] = 1
        target[0, 1::2, 1::2] = 1
        preds = target.clone()
        
        metric.update(preds, target)
        result = metric.compute()
        
        assert result > 0.5

    def test_ignore_index_equals_class(self):
        """Test when ignore_index equals a class label."""
        metric = BoundaryMeanIoU(num_classes=4, ignore_index=2)
        
        target = torch.zeros(2, 32, 32, dtype=torch.long)
        target[:, 10:20, 10:20] = 1
        target[:, 20:30, 20:30] = 2  # This class will be ignored
        target[:, 5:15, 5:15] = 3
        
        preds = target.clone()
        
        metric.update(preds, target)
        result = metric.compute()
        
        # Class 2 should not contribute to IoU
        assert metric.unions[2] == 0

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        target = torch.randint(0, 3, (1, 32, 32))
        preds = torch.randint(0, 3, (1, 32, 32))
        
        metric.update(preds, target)
        result = metric.compute()
        
        assert 0.0 <= result <= 1.0

    def test_many_classes(self):
        """Test with many classes."""
        metric = BoundaryMeanIoU(num_classes=50)
        
        target = torch.randint(0, 50, (2, 64, 64))
        preds = torch.randint(0, 50, (2, 64, 64))
        
        metric.update(preds, target)
        result = metric.compute()
        
        assert 0.0 <= result <= 1.0


class TestBoundaryMeanIoUIntegration:
    """Integration tests for BoundaryMeanIoU."""

    def test_multiple_updates_and_compute(self):
        """Test multiple update calls followed by compute."""
        metric = BoundaryMeanIoU(num_classes=4, average="macro")
        
        for i in range(3):
            target = torch.randint(0, 4, (2, 32, 32))
            preds = torch.randint(0, 4, (2, 32, 32))
            metric.update(preds, target)
        
        result = metric.compute()
        assert 0.0 <= result <= 1.0

    def test_reset_and_reuse(self):
        """Test that metric can be reset and reused."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        # First epoch
        target1 = torch.randint(0, 3, (2, 32, 32))
        preds1 = torch.randint(0, 3, (2, 32, 32))
        metric.update(preds1, target1)
        result1 = metric.compute()
        
        # Reset
        metric.reset()
        
        # Second epoch
        target2 = torch.randint(0, 3, (2, 32, 32))
        preds2 = torch.randint(0, 3, (2, 32, 32))
        metric.update(preds2, target2)
        result2 = metric.compute()
        
        # Results should be independent
        assert 0.0 <= result1 <= 1.0
        assert 0.0 <= result2 <= 1.0

    def test_macro_vs_micro_difference(self):
        """Test that macro and micro averages can produce different results."""
        target = torch.zeros(2, 64, 64, dtype=torch.long)
        target[:, 10:20, 10:20] = 1
        target[:, 40:60, 40:60] = 2
        
        # Good prediction for class 1, poor for class 2
        preds = target.clone()
        preds[:, 40:60, 40:60] = 0
        
        metric_macro = BoundaryMeanIoU(num_classes=3, average="macro")
        metric_micro = BoundaryMeanIoU(num_classes=3, average="micro")
        
        metric_macro.update(preds, target)
        metric_micro.update(preds, target)
        
        result_macro = metric_macro.compute()
        result_micro = metric_micro.compute()
        
        # Both should be valid, but may differ
        assert 0.0 <= result_macro <= 1.0
        assert 0.0 <= result_micro <= 1.0

    def test_background_inclusion_effect(self):
        """Test effect of including/excluding background."""
        target = torch.zeros(2, 64, 64, dtype=torch.long)
        target[:, 20:40, 20:40] = 1
        preds = target.clone()
        
        metric_with_bg = BoundaryMeanIoU(num_classes=2, include_background=True)
        metric_without_bg = BoundaryMeanIoU(num_classes=2, include_background=False)
        
        metric_with_bg.update(preds, target)
        metric_without_bg.update(preds, target)
        
        result_with_bg = metric_with_bg.compute()
        result_without_bg = metric_without_bg.compute()
        
        # Both should be valid
        assert 0.0 <= result_with_bg <= 1.0
        assert 0.0 <= result_without_bg <= 1.0
        
        # Without background, only class 1 is counted
        assert metric_without_bg.intersections[0] == 0
        assert metric_without_bg.unions[0] == 0

    def test_gradients_disabled(self):
        """Test that gradients are disabled during metric computation."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        preds = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randint(0, 3, (2, 32, 32))
        
        # Should not raise error despite requires_grad=True
        metric.update(preds, target)
        result = metric.compute()
        
        assert not result.requires_grad

    def test_consistency_across_devices(self):
        """Test that metric produces consistent results on CPU."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        target = torch.randint(0, 3, (2, 32, 32))
        preds = torch.randint(0, 3, (2, 32, 32))
        
        metric.update(preds, target)
        result = metric.compute()
        
        assert result.device == torch.device('cpu')
        assert 0.0 <= result <= 1.0

    def test_state_persistence(self):
        """Test that state persists correctly across multiple updates."""
        metric = BoundaryMeanIoU(num_classes=3)
        
        target = torch.zeros(1, 32, 32, dtype=torch.long)
        target[0, 10:20, 10:20] = 1
        preds = target.clone()
        
        # First update
        metric.update(preds, target)
        intersections1 = metric.intersections.clone()
        unions1 = metric.unions.clone()
        
        # Second update
        metric.update(preds, target)
        intersections2 = metric.intersections.clone()
        unions2 = metric.unions.clone()
        
        # State should accumulate
        assert torch.all(intersections2 >= intersections1)
        assert torch.all(unions2 >= unions1)


def test_boundary_iou_value():
    """Hand-computed Boundary IoU: 16 x 16 square touching the left image edge, prediction shifted right by d = 4.

    Bands are 4 px deep inside each square, with no band along the image edge:
    target band 16 * 16 - 8 * 12 = 160 px, prediction band 16 * 16 - 8 * 8 = 192 px. They overlap only in the top
    and bottom 4 rows, over columns 4-15: 2 * 4 * 12 = 96 px. IoU = 96 / (160 + 192 - 96) = 0.375.
    """
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 8:24, 0:16] = 1
    preds = torch.zeros_like(target)
    preds[:, 8:24, 4:20] = 1
    metric = BoundaryMeanIoU(num_classes=2, include_background=False)
    metric.update(preds, target)
    assert metric.intersections[1] == 96 and metric.unions[1] == 256
    assert metric.compute() == pytest.approx(0.375)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
