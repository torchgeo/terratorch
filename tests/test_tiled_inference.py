# Copyright contributors to the Terratorch project
"""Unit tests for terratorch/tasks/tiled_inference.py

Tests cover:
  * The unchanged (default) in-memory path – behavioural parity with the old API.
  * Custom TileSource – chips are read on demand; the result equals the default path.
  * Custom TileSink   – predictions are streamed to the sink; finalize() is called once.
  * Sink that returns None – tiled_inference propagates None to the caller.
  * Protocol isinstance checks for TileSource / TileSink.
"""

import torch
import pytest

from terratorch.tasks.tiled_inference import (
    InferenceInput,
    InMemoryTileSource,
    InMemoryTileSink,
    TileSource,
    TileSink,
    tiled_inference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_identity_model(out_channels: int = 2):
    """Return a callable that produces a fixed output tensor of shape (B, C, H, W).

    The output values are the mean of all channels in the input, broadcast to
    ``out_channels`` – simple enough to reason about while still exercising the
    blend/average logic.
    """

    def model(x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (B, C, H, W)
        mean = x.float().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        return mean.expand(-1, out_channels, -1, -1)

    return model


def _image(batch: int = 1, channels: int = 4, h: int = 256, w: int = 256) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.rand(batch, channels, h, w)


# ---------------------------------------------------------------------------
# Default (in-memory) path – backward compatibility
# ---------------------------------------------------------------------------


class TestDefaultPath:
    """The new default path should produce the same result as the old implementation."""

    def test_basic_returns_tensor(self):
        img = _image()
        model = _make_identity_model(out_channels=3)
        result = tiled_inference(model, img, h_crop=64, w_crop=64, h_stride=48, w_stride=48, delta=4)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 256, 256)

    def test_output_shape_matches_input(self):
        batch, c = 2, 6
        img = _image(batch=batch, channels=c, h=128, w=192)
        model = _make_identity_model(out_channels=4)
        result = tiled_inference(model, img, h_crop=64, w_crop=64, h_stride=48, w_stride=48, delta=4)
        assert result.shape == (batch, 4, 128, 192)

    def test_no_padding_mode(self):
        img = _image(h=128, w=128)
        model = _make_identity_model(out_channels=2)
        result = tiled_inference(
            model, img, h_crop=64, w_crop=64, h_stride=48, w_stride=48, delta=0, padding=False
        )
        assert result.shape == (1, 2, 128, 128)

    def test_small_image_single_forward(self):
        """Images that fit in one crop must trigger a single forward without tiling."""
        calls = []

        def model(x, **kwargs):
            calls.append(x.shape)
            return x[:, :2]

        img = _image(h=32, w=32)
        result = tiled_inference(model, img, h_crop=64, w_crop=64)
        assert len(calls) == 1  # exactly one pass
        assert result is not None


# ---------------------------------------------------------------------------
# InMemoryTileSource
# ---------------------------------------------------------------------------


class TestInMemoryTileSource:
    def test_returns_input_data(self):
        chip_data = torch.rand(4, 64, 64)
        tile = InferenceInput(
            batch=0,
            input_coords=(slice(0, 56), slice(0, 56)),
            input_data=chip_data,
            blend_mask=torch.ones(56, 56),
            output_crop=(slice(4, 60), slice(4, 60)),
        )
        source = InMemoryTileSource()
        out = source(tile)
        assert out is chip_data  # same object, no copy

    def test_isinstance_protocol(self):
        assert isinstance(InMemoryTileSource(), TileSource)


# ---------------------------------------------------------------------------
# Custom TileSource – on-demand reading
# ---------------------------------------------------------------------------


class _TensorBackedSource:
    """Simulates an out-of-core source: reads chips from a big in-memory tensor
    but only when asked (like a rasterio dataset would).
    """

    def __init__(self, full_image: torch.Tensor):
        # full_image: (B, C, H, W)  – already padded if needed
        self._img = full_image
        self.calls: list[tuple[int, tuple]] = []

    def __call__(self, tile: InferenceInput) -> torch.Tensor:
        r, c = tile.input_coords
        # We need the chip window, not the output window.
        # For the test we'll just return the pre-sliced input_data so that
        # results are identical to the default path.
        self.calls.append((tile.batch, (r, c)))
        return tile.input_data  # reuse pre-sliced data for comparison


class TestCustomSource:
    def test_custom_source_called_for_every_chip(self):
        img = _image(h=128, w=128)
        src = _TensorBackedSource(img)
        model = _make_identity_model(out_channels=2)

        result = tiled_inference(
            model, img, h_crop=64, w_crop=64, h_stride=48, w_stride=48, delta=4, source=src
        )
        assert len(src.calls) > 0
        assert result.shape == (1, 2, 128, 128)

    def test_custom_source_result_matches_default(self):
        """When custom source returns the same data as the default, outputs must be identical."""
        img = _image(h=128, w=128)

        # Source that returns the same chip data stored on the descriptor
        class PassthroughSource:
            def __call__(self, tile: InferenceInput) -> torch.Tensor:
                return tile.input_data

        model = _make_identity_model(out_channels=2)
        params = dict(h_crop=64, w_crop=64, h_stride=48, w_stride=48, delta=4, padding="reflect")

        result_default = tiled_inference(model, img, **params)
        result_custom = tiled_inference(model, img, **params, source=PassthroughSource())

        assert torch.allclose(result_default, result_custom, atol=1e-5)

    def test_isinstance_protocol(self):
        img = _image()
        src = _TensorBackedSource(img)
        assert isinstance(src, TileSource)


# ---------------------------------------------------------------------------
# InMemoryTileSink
# ---------------------------------------------------------------------------


class TestInMemoryTileSink:
    def test_single_chip_no_padding(self):
        h_img, w_img = 56, 56
        sink = InMemoryTileSink(
            input_batch_size=1,
            h_img=h_img,
            w_img=w_img,
            delta=0,
            device="cpu",
            padding=False,
            average_patches=True,
        )
        pred = torch.ones(3, 56, 56)
        blend = torch.ones(56, 56)
        tile = InferenceInput(
            batch=0,
            input_coords=(slice(0, 56), slice(0, 56)),
            input_data=pred,
            blend_mask=blend,
            output_crop=None,
        )
        sink.write(tile, pred)
        result = sink.finalize()
        assert result.shape == (1, 3, 56, 56)
        assert torch.allclose(result, torch.ones(1, 3, 56, 56))

    def test_finalize_raises_if_empty(self):
        sink = InMemoryTileSink(1, 64, 64, 0, "cpu", False, True)
        with pytest.raises(RuntimeError, match="No chips were written"):
            sink.finalize()

    def test_isinstance_protocol(self):
        sink = InMemoryTileSink(1, 64, 64, 0, "cpu", False, True)
        assert isinstance(sink, TileSink)


# ---------------------------------------------------------------------------
# Custom TileSink – streaming output
# ---------------------------------------------------------------------------


class _CollectingSink:
    """Sink that stores every (tile, prediction) pair for later inspection."""

    def __init__(self):
        self.received: list[tuple[InferenceInput, torch.Tensor]] = []
        self._finalized = False

    def write(self, tile: InferenceInput, prediction: torch.Tensor) -> None:
        self.received.append((tile, prediction.clone()))

    def finalize(self):
        self._finalized = True
        return None  # signals "I handle my own output"


class TestCustomSink:
    def test_write_called_per_chip(self):
        img = _image(h=128, w=128)
        custom_sink = _CollectingSink()
        model = _make_identity_model(out_channels=2)

        result = tiled_inference(
            model, img, h_crop=64, w_crop=64, h_stride=48, w_stride=48, delta=4, sink=custom_sink
        )
        assert result is None  # sink returned None
        assert custom_sink._finalized
        assert len(custom_sink.received) > 0  # at least one chip

    def test_custom_sink_predictions_on_cpu(self):
        """Predictions dispatched to the sink must already be on CPU."""
        img = _image(h=128, w=128)

        class DeviceCheckSink:
            def __init__(self):
                self.devices = []

            def write(self, tile, pred):
                self.devices.append(pred.device)

            def finalize(self):
                return None

        checker = DeviceCheckSink()
        model = _make_identity_model(out_channels=2)
        tiled_inference(model, img, h_crop=64, w_crop=64, h_stride=48, w_stride=48, delta=4, sink=checker)

        assert all(str(d) == "cpu" for d in checker.devices)

    def test_isinstance_protocol(self):
        assert isinstance(_CollectingSink(), TileSink)

    def test_sink_result_matches_default(self):
        """A sink that does the same averaging as InMemoryTileSink must produce identical output."""
        img = _image(h=128, w=128)
        model = _make_identity_model(out_channels=2)
        params = dict(h_crop=64, w_crop=64, h_stride=48, w_stride=48, delta=4, padding="reflect")

        default_result = tiled_inference(model, img, **params)

        # Build a manual sink that mirrors InMemoryTileSink
        manual_sink = InMemoryTileSink(
            input_batch_size=1,
            h_img=128,
            w_img=128,
            delta=4,
            device="cpu",
            padding="reflect",
            average_patches=True,
        )
        custom_result = tiled_inference(model, img, **params, sink=manual_sink)

        assert torch.allclose(default_result, custom_result, atol=1e-5)


# ---------------------------------------------------------------------------
# Source + Sink combined
# ---------------------------------------------------------------------------


class TestSourceAndSinkCombined:
    def test_source_and_sink_together(self):
        """Using both a custom source and a custom sink must not raise."""
        img = _image(h=128, w=128)
        src = _TensorBackedSource(img)
        snk = _CollectingSink()
        model = _make_identity_model(out_channels=2)

        result = tiled_inference(
            model, img, h_crop=64, w_crop=64, h_stride=48, w_stride=48, delta=4, source=src, sink=snk
        )
        assert result is None
        assert len(snk.received) > 0
        assert len(src.calls) == len(snk.received)
