"""This module contains logic for tiled inference.
This does some additional things over the obvious fold -> predict -> unfold logic,
e.g. cropping out areas around model prediction to reduce artifacts

It additionally rebatches after the fold operation to gain speed up.

Source / Sink API
-----------------
The :func:`tiled_inference` function accepts optional *source* and *sink* objects that
decouple **reading** and **writing** from the in-memory tensor path.

* A **source** is any callable that matches :class:`TileSource`.  Given an
  :class:`InferenceInput` descriptor it should return the chip tensor
  ``(C, H, W)`` – or ``(C, T, H, W)`` for multi-temporal inputs – on the CPU.
  This lets callers read from disk (e.g. a rasterio window) instead of holding
  the whole image in memory.

* A **sink** is an object that matches :class:`TileSink`.  Its
  :meth:`~TileSink.write` method is called for every predicted chip, and its
  :meth:`~TileSink.finalize` method is called once at the end to produce (or
  flush) the final result.  This lets callers write directly to a file and
  avoid building the full output tensor.

Both are optional.  When omitted the function behaves exactly as before:
``input_batch`` is used as the in-memory source and all predicted chips are
accumulated in RAM before being averaged and returned.
"""

import math
import tempfile
import warnings
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
import tqdm

from terratorch.models.utils import pad_images


# TODO: Remove TiledInferenceParameters in version 1.3.
@dataclass
class TiledInferenceParameters:
    """
    Parameters to be used for inference. Deprecated, please us directly pass the parameters to tiled_inference.
    """

    h_crop: int = (224,)
    h_stride: int = (200,)
    w_crop: int = (224,)
    w_stride: int = (200,)
    delta: int = (4,)
    average_patches: bool = (True,)
    blend_overlaps: bool = (True,)
    batch_size: int = (16,)
    verbose: bool = (False,)


# ---------------------------------------------------------------------------
# Source / Sink protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class TileSource(Protocol):
    """Protocol for objects that supply chip tensors on demand.

    Implement this to read tiles lazily (e.g. from a file or a remote store)
    instead of keeping the whole image in memory.

    Example::

        class RasterioSource:
            def __init__(self, path: str):
                import rasterio
                self._ds = rasterio.open(path)

            def __call__(self, tile: InferenceInput) -> torch.Tensor:
                row_off = tile.input_coords[0].start
                col_off = tile.input_coords[1].start
                h = tile.input_coords[0].stop - row_off
                w = tile.input_coords[1].stop - col_off
                window = rasterio.windows.Window(col_off, row_off, w, h)
                data = self._ds.read(window=window)        # (C, H, W)
                return torch.from_numpy(data).float()
    """

    @abstractmethod
    def __call__(self, tile: "InferenceInput") -> torch.Tensor:
        """Return the chip tensor (CPU) for *tile*.

        Args:
            tile: Descriptor produced by :func:`get_input_chips` that carries
                the destination coordinates and blend mask for this chip.
                The caller has already sliced ``input_data`` on ``tile`` when
                using the default in-memory path; a custom source may ignore
                ``tile.input_data`` and read the raw window from wherever it
                likes.

        Returns:
            torch.Tensor: A CPU tensor of shape ``(C, H, W)`` or
            ``(C, T, H, W)``.  It will be stacked with other chips and moved
            to the inference device by the caller.
        """
        ...


@runtime_checkable
class TileSink(Protocol):
    """Protocol for objects that consume predicted chip tensors one at a time.

    Implement this to stream predictions directly to disk or another consumer
    instead of accumulating them in a large in-memory tensor.

    The lifecycle is::

        for chip, prediction in ...:
            sink.write(chip, prediction)
        result = sink.finalize()

    Example – write to a file using :class:`BlendingSink`::

        class RasterioSink(BlendingSink):
            def __init__(self, path, profile, h_img, w_img, delta=0, padding=False):
                super().__init__(input_batch_size=1, h_img=h_img, w_img=w_img, delta=delta, padding=padding)
                self._path = path
                self._profile = profile

            def finalize(self):
                result = super().finalize()
                import rasterio
                with rasterio.open(self._path, "w", **self._profile) as dst:
                    dst.write(result[0].numpy())
                return result
    """

    @abstractmethod
    def write(self, tile: "InferenceInput", prediction: torch.Tensor) -> None:
        """Accept one predicted chip.

        ``prediction`` is a CPU tensor already transferred from the inference
        device.  ``tile`` provides the destination coordinates and blend mask.
        The sink is responsible for any blending / averaging it wants to apply
        (or can inherit from :class:`BlendingSink` to let the library handle it).

        Args:
            tile: The :class:`InferenceInput` descriptor for this chip.
            prediction: CPU tensor of shape ``(C, H, W)`` (one sample from the
                batch dimension has already been indexed by the caller).
        """
        ...

    @abstractmethod
    def finalize(self) -> "torch.Tensor | None":
        """Called once after all chips have been written.

        Returns:
            The final output tensor, or ``None`` if the sink writes to an
            external store (file, database, …) and the caller does not need an
            in-memory result.
        """
        ...


# ---------------------------------------------------------------------------
# Default (in-memory) source / sink implementations
# ---------------------------------------------------------------------------


class InMemoryTileSource:
    """Default :class:`TileSource` backed by a pre-loaded tensor.

    This is the source used internally by :func:`tiled_inference` when no
    custom source is supplied.  It simply returns the ``input_data`` that was
    already sliced and stored on the :class:`InferenceInput` descriptor.
    """

    def __call__(self, tile: "InferenceInput") -> torch.Tensor:
        if tile.input_data is None:
            raise ValueError("tile.input_data is None in InMemoryTileSource.")
        return tile.input_data


class BlendingSink:
    """Base class for sinks that handles patch cropping, blend weighting, and accumulation.

    Custom sinks can inherit from this class to avoid re-implementing edge cropping,
    blend mask multiplication, and normalization math.

    Args:
        input_batch_size: Number of images in the batch.
        h_img: Height of the original unpadded image canvas.
        w_img: Width of the original unpadded image canvas.
        delta: Border width used for padding; set to 0 when ``padding`` is ``False``.
        device: Device to which the final tensor is moved (default: "cpu").
        padding: The padding mode that was used (or ``False``).
        average_patches: If ``True``, overlapping areas are averaged using
            the blend mask. If ``False``, later chips overwrite earlier ones.
    """

    def __init__(
        self,
        input_batch_size: int,
        h_img: int,
        w_img: int,
        delta: int = 0,
        device: "str | torch.device" = "cpu",
        padding: "str | bool" = "reflect",
        average_patches: bool = True,
    ) -> None:
        self._input_batch_size = input_batch_size
        self._h_img = h_img
        self._w_img = w_img
        self._delta = delta
        self._device = device
        self._padding = padding
        self._average_patches = average_patches

        # Lazily allocated once we see the first prediction.
        self._preds: torch.Tensor | None = None
        self._preds_count: torch.Tensor | None = None

    def _init_accumulators(self, out_channels: int) -> None:
        if self._padding:
            h = self._h_img + 2 * self._delta
            w = self._w_img + 2 * self._delta
        else:
            h, w = self._h_img, self._w_img
        self._preds = torch.zeros((self._input_batch_size, out_channels, h, w))
        self._preds_count = torch.zeros(self._input_batch_size, h, w)

    def write(self, tile: "InferenceInput", prediction: torch.Tensor) -> None:
        """Process and accumulate a single predicted chip.

        Crops the prediction with `tile.output_crop` and weights it with `tile.blend_mask`.
        """
        if self._preds is None:
            out_channels = 1 if len(prediction.shape) == 2 else prediction.shape[0]
            self._init_accumulators(out_channels)

        if tile.output_crop is not None:
            prediction = prediction[..., tile.output_crop[0], tile.output_crop[1]]

        if self._preds is None or self._preds_count is None:
            raise RuntimeError("Accumulators not initialized.")

        if self._average_patches:
            self._preds[
                tile.batch,
                :,
                tile.input_coords[0],
                tile.input_coords[1],
            ] += prediction * tile.blend_mask
        else:
            self._preds[
                tile.batch,
                :,
                tile.input_coords[0],
                tile.input_coords[1],
            ] = prediction

        self._preds_count[
            tile.batch,
            tile.input_coords[0],
            tile.input_coords[1],
        ] += tile.blend_mask

    def finalize(self) -> torch.Tensor:
        """Finalize accumulation, remove padding, normalize by blend weights, and return."""
        if self._preds is None or self._preds_count is None:
            raise RuntimeError("No chips were written to the sink.")

        preds = self._preds
        preds_count = self._preds_count

        if self._padding:
            d = self._delta
            preds = preds[..., d:-d, d:-d]
            preds_count = preds_count[..., d:-d, d:-d]

        if (preds_count == 0).sum() != 0:
            msg = "Some pixels did not receive a classification!"
            raise RuntimeError(msg)

        output = preds
        if self._average_patches:
            output = output / preds_count.unsqueeze(1)

        return output.to(self._device)


class InMemoryTileSink(BlendingSink):
    """Default :class:`TileSink` that accumulates all chips into a single tensor in memory."""
    pass


class MemmapTileSink(BlendingSink):
    """Tile sink backed by memory-mapped files on disk.

    Useful when accumulating predictions for very large images that exceed RAM.
    Predictions and blend counts are memory-mapped to temporary files.

    Args:
        input_batch_size: Number of images in the batch.
        h_img: Height of the original unpadded image canvas.
        w_img: Width of the original unpadded image canvas.
        delta: Border width used for padding; set to 0 when ``padding`` is ``False``.
        device: Device to which the final tensor is moved (default: "cpu").
        padding: The padding mode that was used (or ``False``).
        average_patches: If ``True``, overlapping areas are averaged using
            the blend mask. If ``False``, later chips overwrite earlier ones.
        tmp_dir: Directory where temporary files are stored (default: system temp dir).
    """

    def __init__(
        self,
        input_batch_size: int,
        h_img: int,
        w_img: int,
        delta: int = 0,
        device: "str | torch.device" = "cpu",
        padding: "str | bool" = "reflect",
        average_patches: bool = True,
        tmp_dir: str | None = None,
    ) -> None:
        super().__init__(
            input_batch_size=input_batch_size,
            h_img=h_img,
            w_img=w_img,
            delta=delta,
            device=device,
            padding=padding,
            average_patches=average_patches,
        )
        self._tmp_dir = tmp_dir
        self._preds_file = None
        self._preds_count_file = None
        self._preds_np: np.memmap | None = None
        self._preds_count_np: np.memmap | None = None

    def _init_accumulators(self, out_channels: int) -> None:
        if self._padding:
            h = self._h_img + 2 * self._delta
            w = self._w_img + 2 * self._delta
        else:
            h, w = self._h_img, self._w_img

        self._preds_file = tempfile.NamedTemporaryFile(dir=self._tmp_dir)
        self._preds_count_file = tempfile.NamedTemporaryFile(dir=self._tmp_dir)

        preds_shape = (self._input_batch_size, out_channels, h, w)
        counts_shape = (self._input_batch_size, h, w)

        self._preds_np = np.memmap(
            self._preds_file, dtype=np.float32, mode="w+", shape=preds_shape
        )
        self._preds_count_np = np.memmap(
            self._preds_count_file, dtype=np.float32, mode="w+", shape=counts_shape
        )
        self._preds = torch.from_numpy(self._preds_np)
        self._preds_count = torch.from_numpy(self._preds_count_np)

    def close(self) -> None:
        """Clean up temporary memory-mapped files."""
        self._preds = None
        self._preds_count = None
        self._preds_np = None
        self._preds_count_np = None
        if self._preds_file is not None:
            try:
                self._preds_file.close()
            except Exception:
                pass
            self._preds_file = None
        if self._preds_count_file is not None:
            try:
                self._preds_count_file.close()
            except Exception:
                pass
            self._preds_count_file = None

    def __del__(self) -> None:
        self.close()


def get_blend_mask(
    h_crop: int = 224,
    h_stride: int = 200,
    w_crop: int = 224,
    w_stride: int = 200,
    delta: int = 0,
) -> torch.Tensor:
    overlap_w = min(w_crop // 2, w_crop - w_stride) - delta
    overlap_h = min(h_crop // 2, h_crop - h_stride) - delta

    # Vertical window
    y_pos = torch.arange(h_crop - 2 * delta, device="cpu")
    y = torch.ones_like(y_pos, dtype=torch.float)
    if overlap_h:
        # ramp = (torch.cos(math.pi * (y_pos[:overlap_w] + 1) / (overlap_w + 1) / 2))
        ramp = torch.cos(math.pi * (y_pos[:overlap_h] + 1) / (overlap_h + 1)) / 2 + 0.5
        y[:overlap_h] = ramp.flip(0)  # top edge
        y[-overlap_h:] = ramp  # bottom edge

    # Horizontal window
    x_pos = torch.arange(w_crop - 2 * delta, device="cpu")
    x = torch.ones_like(x_pos, dtype=torch.float)
    if overlap_w:
        # ramp = (torch.cos(math.pi * (x_pos[:overlap_w] + 1) / (overlap_w + 1) / 2))
        ramp = torch.cos(math.pi * (x_pos[:overlap_w] + 1) / (overlap_w + 1)) / 2 + 0.5
        x[:overlap_w] = ramp.flip(0)  # left edge
        x[-overlap_w:] = ramp  # right edge

    # Get outer product (2D mask)
    mask = y[:, None] * x[None, :]

    # Add buffer to ensure every pixel gets a generation
    mask += 1e-6

    return mask


@dataclass
class InferenceInput:
    batch: int
    input_coords: tuple[slice, slice]
    input_data: torch.Tensor | None
    blend_mask: torch.Tensor
    output_crop: None | tuple[slice, slice]


def get_input_chips(
    input_batch: torch.Tensor | None,
    h_crop: int,
    h_stride: int,
    w_crop: int,
    w_stride: int,
    delta: int,
    blend_overlaps: bool,
    padding: str | bool,
    h_img: int | None = None,
    w_img: int | None = None,
    input_batch_size: int = 1,
) -> list[InferenceInput]:
    """
    Create input chips of type InferenceInput for tiled inference. These contain:
      0. batch
      1. Coordinates where this should end up in the preds
      2. output/input data (optional when using a custom TileSource)
      3. Blend mask for weighting the edges of the chips
      4. Optionally, for inputs, how to crop the output
    """
    if input_batch is not None:
        input_batch_size = input_batch.shape[0]

        if padding:
            w_pad, h_pad = delta, delta

            if len(input_batch.shape) > 4:
                # Ignore additional during padding (e.g. with multi-temporal input)
                add_dim = [0, 0] * (len(input_batch.shape) - 4)
            else:
                add_dim = []

            input_batch = torch.nn.functional.pad(input_batch, (w_pad, w_pad, h_pad, h_pad, *add_dim), mode=padding)
        h_img, w_img = input_batch.shape[-2:]
    else:
        if h_img is None or w_img is None:
            raise ValueError("h_img and w_img must be provided when input_batch is None.")
        if padding:
            # Adjust dimensions for virtual padded image
            h_img = h_img + 2 * delta
            w_img = w_img + 2 * delta

    if padding:
        border_output_crop = (slice(delta, h_crop - delta), slice(delta, w_crop - delta))
    else:
        border_output_crop = None
    inner_output_crop = (slice(delta, h_crop - delta), slice(delta, w_crop - delta))

    # Blend overlapping areas using weighted masks
    if blend_overlaps:
        inner_blend_mask = get_blend_mask(h_crop, h_stride, w_crop, w_stride, delta)
        border_blend_mask = inner_blend_mask if padding else get_blend_mask(h_crop, h_stride, w_crop, w_stride)
    else:
        inner_blend_mask = torch.ones((h_crop - 2 * delta, w_crop - 2 * delta), device="cpu", dtype=torch.float)
        border_blend_mask = (
            inner_blend_mask if padding else torch.ones((h_crop, w_crop), device="cpu", dtype=torch.float)
        )

    # Stage 1: deal with border patches (using border settings and subtract delta from coords only if padding is used)
    # Deal with patches near the right border
    coordinates_and_inputs: list[InferenceInput] = []
    for i in range(0, h_img - h_crop - 1, h_stride):
        patch = input_batch[..., i : i + h_crop, w_img - w_crop : w_img] if input_batch is not None else None
        coordinates_and_inputs += [
            InferenceInput(
                b,
                (slice(i + delta, i + h_crop - delta), slice(w_img - w_crop + delta, w_img - delta))
                if padding
                else (slice(i, i + h_crop), slice(w_img - w_crop, w_img)),
                patch[b] if patch is not None else None,
                border_blend_mask,
                border_output_crop,
            )
            for b in range(input_batch_size)
        ]

    # Deal with patches near the bottom of the image
    for i in range(0, w_img - w_crop - 1, w_stride):
        patch = input_batch[..., h_img - h_crop : h_img, i : i + w_crop] if input_batch is not None else None
        coordinates_and_inputs += [
            InferenceInput(
                b,
                (slice(h_img - h_crop + delta, h_img - delta), slice(i + delta, i + w_crop - delta))
                if padding
                else (slice(h_img - h_crop, h_img), slice(i, i + w_crop)),
                patch[b] if patch is not None else None,
                border_blend_mask,
                border_output_crop,
            )
            for b in range(input_batch_size)
        ]

    # Deal with last patches at the right bottom of the image
    patch = input_batch[..., h_img - h_crop : h_img, w_img - w_crop : w_img] if input_batch is not None else None
    coordinates_and_inputs += [
        InferenceInput(
            b,
            (slice(h_img - h_crop + delta, h_img - delta), slice(w_img - w_crop + delta, w_img - delta))
            if padding
            else (slice(h_img - h_crop, h_img), slice(w_img - w_crop, w_img)),
            patch[b] if patch is not None else None,
            border_blend_mask,
            border_output_crop,
        )
        for b in range(input_batch_size)
    ]

    for row in range(0, h_img - h_crop - 1, h_stride):
        for col in range(0, w_img - w_crop - 1, w_stride):
            patch = input_batch[..., row : row + h_crop, col : col + w_crop] if input_batch is not None else None
            if row == 0 or col == 0:
                # Add patches along the left and top of the image
                coordinates_and_inputs += [
                    InferenceInput(
                        b,
                        (slice(row + delta, row + h_crop - delta), slice(col + delta, col + w_crop - delta))
                        if padding
                        else (slice(row, row + h_crop), slice(col, col + w_crop)),
                        patch[b] if patch is not None else None,
                        border_blend_mask,
                        border_output_crop,
                    )
                    for b in range(input_batch_size)
                ]
            else:
                # Stage 2: process internally with patch overlap
                coordinates_and_inputs += [
                    InferenceInput(
                        b,
                        (slice(row + delta, row + h_crop - delta), slice(col + delta, col + w_crop - delta)),
                        patch[b] if patch is not None else None,
                        inner_blend_mask,
                        inner_output_crop,
                    )
                    for b in range(input_batch_size)
                ]

    return coordinates_and_inputs


def prepare_tiled_inference_input(
    input_batch: torch.Tensor | dict | None = None,
    out_channels: int | None = None,
    crop: int = 224,
    stride: int = 192,
    delta: int = 8,
    blend_overlaps: bool = True,
    h_crop: int | None = None,
    w_crop: int | None = None,
    h_stride: int | None = None,
    w_stride: int | None = None,
    padding: str | bool = "reflect",
    device: str | torch.device | None = None,
    h_img: int | None = None,
    w_img: int | None = None,
    input_batch_size: int = 1,
) -> tuple[list[InferenceInput], Callable[[torch.Tensor], dict | torch.Tensor], int, int, int, str | torch.device, int]:

    if isinstance(input_batch, dict):
        # Handle dict inputs for tiled inference
        modalities, tensors = list(input_batch.keys()), list(input_batch.values())

        # Check that all values in dict are tensors and have a same image shape
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise ValueError("input for tiled inference must be either a torch.Tensor or a dict of torch.Tensors")
        img_shapes = [t.shape[-2:] for t in tensors]
        if len(set(img_shapes)) != 1:
            raise ValueError(
                f"Tensors in input dict must have the same height and width for tiled inference, "
                f"found {dict(zip(modalities, img_shapes))}"
            )
        t_dims = [len(t.shape) for t in tensors]
        if len(set(t_dims)) != 1:
            raise ValueError(
                f"Tensors in input dict must have the same number of dimensions for tiled inference, "
                f"found {dict(zip(modalities, t_dims, strict=False))}"
            )

        # Tiled inference is implemented for single tensors.
        # We concatenate all tensors and reshape them before the model forward
        t_dims = t_dims[0]
        if t_dims == 4:  # B, C, H, W
            channel_length = [t.shape[-3] for t in tensors]
            channel_start = torch.tensor([0] + channel_length).cumsum(0)
            input_batch = torch.concat(tensors, dim=-3)
        elif t_dims == 5:  # B, C, T, H, W
            channel_length = [t.shape[-4] for t in tensors]
            channel_start = torch.tensor([0] + channel_length).cumsum(0)
            input_batch = torch.concat(tensors, dim=-4)
        else:
            raise ValueError("Tensors must have 4 or 5 dimensions")

        def tensor_reshape(t):
            # Convert tensor back to dict of tensors
            if t_dims == 4:  # B, C, H, W
                out = {m: t[..., s : s + l, :, :] for m, s, l in zip(modalities, channel_start, channel_length)}
            elif t_dims == 5:  # B, C, T, H, W
                out = {m: t[..., s : s + l, :, :, :] for m, s, l in zip(modalities, channel_start, channel_length)}
            return out

        input_batch_size = input_batch.shape[0]
        h_img, w_img = input_batch.shape[-2:]
        ret_device = device or input_batch.device
        input_batch = input_batch.cpu()

    elif isinstance(input_batch, torch.Tensor):
        # Dummy function if input is a tensor
        tensor_reshape = lambda x: x
        input_batch_size = input_batch.shape[0]
        h_img, w_img = input_batch.shape[-2:]
        ret_device = device or input_batch.device
        input_batch = input_batch.cpu()

    elif input_batch is None:
        tensor_reshape = lambda x: x
        if h_img is None or w_img is None:
            raise ValueError("When input_batch is None, h_img and w_img must be provided.")
        ret_device = device or "cpu"
    else:
        raise ValueError("input for tiled inference must be either a torch.Tensor, a dict of torch.Tensors, or None")

    h_crop = h_crop or crop
    w_crop = w_crop or crop
    h_stride = h_stride or stride
    w_stride = w_stride or stride

    if (h_crop - h_stride) // 2 < delta or (w_crop - w_stride) // 2 < delta:
        # Ensure that every pixel is covered
        delta = min((h_crop - h_stride) // 2, (w_crop - w_stride) // 2)
        warnings.warn(f"Tiled inference: delta is higher than overlap, reducing delta to {delta}.")

    if out_channels is not None:
        warnings.warn(
            "out_channels is deprecated and automatically selected after first forward pass.", DeprecationWarning
        )

    # Get smaller inputs
    coordinates_and_inputs = get_input_chips(
        input_batch=input_batch,
        h_crop=h_crop,
        h_stride=h_stride,
        w_crop=w_crop,
        w_stride=w_stride,
        delta=delta,
        blend_overlaps=blend_overlaps,
        padding=padding,
        h_img=h_img,
        w_img=w_img,
        input_batch_size=input_batch_size,
    )

    return coordinates_and_inputs, tensor_reshape, input_batch_size, h_img, w_img, ret_device, delta


def generate_tiled_inference_output(
    outputs,
    input_batch_size: int,
    h_img: int,
    w_img: int,
    delta: int,
    padding: str | bool = "reflect",
    average_patches: bool = True,
) -> torch.Tensor:
    preds: torch.Tensor | None = None
    preds_count: torch.Tensor | None = None

    for batch_input, predicted in outputs:
        if preds is None:
            # Initialize preds based on first output
            out_channels = 1 if len(predicted.shape) == 2 else predicted.shape[0]
            if padding:
                # Add padding areas to align with input indexes
                preds = torch.zeros((input_batch_size, out_channels, h_img + (2 * delta), w_img + (2 * delta)))
                preds_count = torch.zeros(input_batch_size, h_img + (2 * delta), w_img + (2 * delta))
            else:
                preds = torch.zeros((input_batch_size, out_channels, h_img, w_img))
                preds_count = torch.zeros(input_batch_size, h_img, w_img)
        if batch_input.output_crop is not None:
            predicted = predicted[..., batch_input.output_crop[0], batch_input.output_crop[1]]
        if average_patches:
            preds[
                batch_input.batch,
                :,
                batch_input.input_coords[0],
                batch_input.input_coords[1],
            ] += predicted * batch_input.blend_mask
        else:
            preds[
                batch_input.batch,
                :,
                batch_input.input_coords[0],
                batch_input.input_coords[1],
            ] = predicted

        preds_count[
            batch_input.batch,
            batch_input.input_coords[0],
            batch_input.input_coords[1],
        ] += batch_input.blend_mask

    if padding:
        # Remove padded areas
        preds = preds[..., delta:-delta, delta:-delta]
        preds_count = preds_count[..., delta:-delta, delta:-delta]
    if (preds_count == 0).sum() != 0:
        msg = "Some pixels did not receive a classification!"
        raise RuntimeError(msg)

    output = preds

    if average_patches:
        output = output / preds_count.unsqueeze(1)

    return output


def tiled_inference(
    model_forward: Callable,
    input_batch: "torch.Tensor | dict | None" = None,
    out_channels: int | None = None,
    inference_parameters: TiledInferenceParameters | None = None,
    crop: int = 224,
    stride: int = 192,
    delta: int = 8,
    h_crop: int | None = None,
    w_crop: int | None = None,
    h_stride: int | None = None,
    w_stride: int | None = None,
    average_patches: bool = True,
    blend_overlaps: bool = True,
    batch_size: int = 16,
    verbose: bool = False,
    padding: "str | bool" = "reflect",
    device: str | None = None,
    source: "TileSource | None" = None,
    sink: "TileSink | None" = None,
    h_img: int | None = None,
    w_img: int | None = None,
    input_batch_size: int = 1,
    accumulator: str = "memory",
    **kwargs,
) -> "torch.Tensor | None":
    """
    Divide an image into (potentially) overlapping chips and perform inference on them.
    Additionally, re-batch for variable GPU utilization defined by crop size and batch_size.
    The overlap between chips is defined with: crop - stride - 2 * delta.

    Args:
        model_forward (Callable): Callable that returns the output of the model.
        input_batch (torch.Tensor | dict | None): Input batch to be processed.
            May be ``None`` when a custom *source* is provided that reads
            on demand. When ``None``, *h_img* and *w_img* should be specified.
        out_channels (int): Number of output channels. Deprecated.
        inference_parameters (TiledInferenceParameters): Parameters to be used for inference.
            Deprecated, please pass the parameters directly to tiled_inference.
        crop (int): height and width of the smaller chips. Ignored if h_crop or w_crop is provided. Defaults to 224.
        stride (int): size of the stride. Ignored if h_stride or w_stride is provided. Defaults to 192.
        delta (int): size of the border cropped from each chip. Defaults to 8.
        h_crop (int, optional): height of the smaller chips.
        w_crop (int, optional): width of the smaller chips.
        h_stride (int, optional): size of the stride on the y-axis.
        w_stride (int, optional): size of the stride on the x-axis.
        average_patches (bool): Whether to average the overlapping regions. Defaults to True.
        blend_overlaps (bool): Whether to use blend masks on overlapping edges. Defaults to True.
        batch_size (int): Number of chips per forward pass. Defaults to 16.
        verbose (bool): Show a tqdm progress bar. Defaults to False.
        padding (str | bool): Padding mode for input image to reduce artefacts on edges.
            Deactivate padding with False. Defaults to reflect.
        device (str | None): Device for inference. Inferred from *input_batch* when not given.
        source (TileSource | None): Optional callable that returns the chip tensor for a
            given :class:`InferenceInput` descriptor. When provided, chips are read
            on-demand, enabling out-of-core reading without loading full tensors into RAM.

            The source receives the full :class:`InferenceInput` (including
            ``input_coords``) so it can map coordinates to a file window.
            It must return a CPU tensor of shape ``(C, H, W)`` or
            ``(C, T, H, W)``.
        sink (TileSink | None): Optional object that receives each predicted
            chip via :meth:`~TileSink.write` and is asked to produce the final
            result via :meth:`~TileSink.finalize`. Use this (or subclass
            :class:`BlendingSink`) to write predictions incrementally (e.g. to a GeoTIFF)
            instead of building a large in-memory accumulator.

            When *sink* is ``None`` (the default) an :class:`InMemoryTileSink`
            is used, which replicates the original behaviour.
        h_img (int | None): Height of the input image, required when *input_batch* is ``None``.
        w_img (int | None): Width of the input image, required when *input_batch* is ``None``.
        input_batch_size (int): Batch size, used when *input_batch* is ``None`` (default: 1).
        accumulator (str): Accumulator backend to use when *sink* is not provided.
            ``"memory"`` (default) accumulates in RAM via :class:`InMemoryTileSink`.
            ``"disk"`` or ``"memmap"`` accumulates on disk using memory-mapped temporary files
            via :class:`MemmapTileSink`.

    Returns:
        torch.Tensor | None: The result of the inference, or ``None`` if the
        *sink* returns ``None`` from :meth:`~TileSink.finalize` (e.g. because
        it writes directly to disk).
    """

    if inference_parameters is not None:
        # TODO: Remove inference_parameters in version 1.3.
        warnings.warn(
            "Using inference_parameters and ignoring other parameters."
            "The parameter `inference_parameters` is deprecated and is removed in version 1.3, "
            "please pass the parameters directly to `tiled_inference`. ",
            DeprecationWarning,
        )
        h_crop = inference_parameters.h_crop
        h_stride = inference_parameters.h_stride
        w_crop = inference_parameters.w_crop
        w_stride = inference_parameters.w_stride
        delta = inference_parameters.delta
        blend_overlaps = inference_parameters.blend_overlaps
        average_patches = inference_parameters.average_patches
        batch_size = inference_parameters.batch_size
        verbose = inference_parameters.verbose

    h_crop = h_crop or crop
    w_crop = w_crop or crop

    # If input_batch is given and already fits within a single chip, skip tiling and run a single forward pass.
    if input_batch is not None:
        sample_tensor = next(iter(input_batch.values())) if isinstance(input_batch, dict) else input_batch
        h_img_curr, w_img_curr = sample_tensor.shape[-2:]
        if h_img_curr <= h_crop and w_img_curr <= w_crop:
            with torch.no_grad():
                return model_forward(input_batch, **kwargs)

    coordinates_and_inputs, tensor_reshape, input_batch_size, h_img, w_img, device, delta = (
        prepare_tiled_inference_input(
            input_batch=input_batch,
            out_channels=out_channels,
            crop=crop,
            stride=stride,
            delta=delta,
            h_crop=h_crop,
            w_crop=w_crop,
            h_stride=h_stride,
            w_stride=w_stride,
            blend_overlaps=blend_overlaps,
            padding=padding,
            device=device,
            h_img=h_img,
            w_img=w_img,
            input_batch_size=input_batch_size,
        )
    )

    # Resolve source: default to reading pre-sliced chip from the descriptor.
    effective_source: TileSource = source if source is not None else InMemoryTileSource()

    # Resolve sink: default to the classic in-memory accumulator or disk memmap.
    effective_sink: TileSink
    if sink is not None:
        effective_sink = sink
    elif accumulator in ("disk", "memmap"):
        effective_sink = MemmapTileSink(
            input_batch_size=input_batch_size,
            h_img=h_img,
            w_img=w_img,
            delta=delta,
            device=device,
            padding=padding,
            average_patches=average_patches,
        )
    else:
        effective_sink = InMemoryTileSink(
            input_batch_size=input_batch_size,
            h_img=h_img,
            w_img=w_img,
            delta=delta,
            device=device,
            padding=padding,
            average_patches=average_patches,
        )

    # NOTE: the output may be SLIGHTLY different using batched inputs because of layers such as nn.LayerNorm
    # During inference, these layers compute batch statistics that affect the output.
    # However, this should still be correct.
    with torch.no_grad():
        for start in tqdm.tqdm(
            range(0, len(coordinates_and_inputs), batch_size), desc="Tiled inference", disable=not verbose
        ):
            end = min(len(coordinates_and_inputs), start + batch_size)
            batch = coordinates_and_inputs[start:end]

            # Read chip data via the source (custom or default in-memory).
            chip_tensors = [effective_source(chip) for chip in batch]
            tensor_input = torch.stack(chip_tensors, dim=0)
            tensor_input = tensor_input.to(device)
            tensor_input = tensor_reshape(tensor_input)  # Optional reshaping for inputs other than plain tensors
            output = model_forward(tensor_input, **kwargs).cpu()

            # Dispatch each chip's prediction to the sink.
            for i, chip in enumerate(batch):
                effective_sink.write(chip, output[i])

    return effective_sink.finalize()
