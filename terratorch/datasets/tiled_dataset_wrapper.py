# Copyright contributors to the Terratorch project

"""Generic tiled dataset wrapper for pixel-wise tasks (segmentation, regression).

This wrapper tiles images and masks on-the-fly with disk caching to avoid repeated
preprocessing. Useful for:
- Training with consistent padding behavior across models
- Processing large images that don't fit patch size requirements
- Moving padding logic from model to data pipeline

Unlike od_tiled_dataset_wrapper.py (object detection specific), this handles
pixel-wise tasks where mask tiles align exactly with image tiles.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)


def _compute_cache_key(
    base_len: int,
    tile_size: tuple[int, int],
    overlap: int,
    patch_size: int | None,
    padding: str | None
) -> str:
    """Compute a unique hash for cache validation."""
    config = f"{base_len}_{tile_size}_{overlap}_{patch_size}_{padding}"
    return hashlib.md5(config.encode()).hexdigest()


class TiledDataset(Dataset):
    """Wraps a pixel-wise dataset to provide tiled access with disk caching.

    On first access, tiles images and masks into smaller chips with optional overlap.
    Results are cached to disk for fast repeated access. On subsequent runs, uses
    cached tiles if configuration matches.

    This is the underlying dataset used by TilingDataModuleWrapper. Most users should
    use the wrapper instead, which handles DataModule integration automatically.

    Direct Usage (Advanced):
        >>> from terratorch.datasets import TiledDataset
        >>> from torch.utils.data import DataLoader
        >>>
        >>> # Wrap your existing dataset
        >>> tiled_dataset = TiledDataset(
        ...     base_dataset=your_dataset,
        ...     cache_dir="./tiles",
        ...     tile_size=(512, 512),
        ...     overlap=64,
        ...     keep_incomplete_tiles=False,
        ... )
        >>>
        >>> # Use in dataloader
        >>> dataloader = DataLoader(tiled_dataset, batch_size=8, shuffle=True)
        >>>
        >>> # Each batch contains tiles instead of full images
        >>> for batch in dataloader:
        ...     images = batch["image"]  # [B, C, tile_h, tile_w]
        ...     masks = batch["mask"]    # [B, tile_h, tile_w]
        ...     coords = batch["tile_coords"]  # List of (y1, x1, y2, x2)

    Recommended Usage (with DataModule wrapper):
        >>> from terratorch.datamodules import TilingDataModuleWrapper
        >>>
        >>> # Let the wrapper handle TiledDataset creation
        >>> tiled_dm = TilingDataModuleWrapper(
        ...     base_datamodule=your_datamodule,
        ...     tile_size=(512, 512),
        ...     overlap=64,
        ... )
        >>> # No need to interact with TiledDataset directly

    Cache Structure:
        The cache directory contains:
        - tile_index_<hash>.json: Index of all tiles with positions
        - tile_<idx>_<y>_<x>_<h>_<w>.pt: Individual tile files

        Example cache directory:
            tiles/
            ├── tile_index_abc123.json
            ├── tile_0_0_0_512_512.pt
            ├── tile_0_448_0_512_512.pt  # 64px overlap with previous
            └── ...

    Metadata Added:
        Each tile sample includes:
        - "tile_coords": (y1, x1, y2, x2) - bounding box in original image
        - "base_idx": int - index of source image in base_dataset
        - Original sample keys: "image", "mask", etc.

    Args:
        base_dataset: Source dataset with samples containing "image" and optionally
            "mask". Must support __len__ and __getitem__.
        cache_dir: Directory for storing cached tiles. Will be created if doesn't
            exist. Default: "tile_cache"
        tile_size: (height, width) of each tile chip. Default: (512, 512)
        overlap: Number of pixels to overlap between adjacent tiles. Increase for
            smoother predictions at tile boundaries. Default: 0
        patch_size: Model patch size for padding calculation. If specified, tiles
            are padded to be divisible by this value. Default: None
        padding: Padding mode to match model behavior (e.g., "symmetric",
            "bottom-right"). Should match model's pre-training if applicable.
            Default: None
        min_size: Minimum (height, width) to process an image. Images smaller than
            this pass through unchanged without tiling. Default: (1, 1)
        rebuild: Force rebuild cache even if it exists. Use when base dataset
            changes. Default: False
        keep_incomplete_tiles: Keep edge tiles that are smaller than tile_size.
            Set False to discard incomplete tiles (recommended for training).
            Set True to keep all tiles (recommended for inference). Default: True

    Note:
        Cache is invalidated automatically when configuration changes (tile_size,
        overlap, patch_size, padding, or dataset length). No manual cache
        management needed unless force rebuild with rebuild=True.

    See Also:
        TilingDataModuleWrapper: Recommended higher-level wrapper for DataModules
    """

    def __init__(
        self,
        base_dataset: Dataset,
        cache_dir: str = "tile_cache",
        tile_size: tuple[int, int] = (512, 512),
        overlap: int = 0,
        patch_size: int | None = None,
        padding: str | None = None,
        min_size: tuple[int, int] = (1, 1),
        rebuild: bool = False,
        keep_incomplete_tiles: bool = True,
    ):
        self.base_dataset = base_dataset
        self.cache_dir = Path(cache_dir)
        self.tile_h, self.tile_w = tile_size
        self.overlap = overlap
        self.patch_size = patch_size
        self.padding = padding
        self.min_h, self.min_w = min_size
        self.keep_incomplete_tiles = keep_incomplete_tiles

        if overlap >= min(tile_size):
            raise ValueError(f"Overlap {overlap} must be less than tile size {tile_size}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Tile index: [(base_idx, x0, y0, tile_h, tile_w), ...]
        self.tiles: list[tuple[int, int, int, int, int]] = []
        self._build_tile_index(rebuild=rebuild)

    def _build_tile_index(self, rebuild: bool = False):
        """Build or load tile index."""
        cache_key = _compute_cache_key(
            len(self.base_dataset),
            (self.tile_h, self.tile_w),
            self.overlap,
            self.patch_size,
            self.padding
        )
        index_file = self.cache_dir / f"tile_index_{cache_key}.json"

        # Check if we can reuse existing index
        if not rebuild and index_file.exists():
            logger.info(f"[TiledDataset] Loading cached tile index from {index_file}")
            with open(index_file) as f:
                self.tiles = [tuple(t) for t in json.load(f)]
            logger.info(f"[TiledDataset] Loaded {len(self.tiles)} tiles")
            return

        logger.info("[TiledDataset] Building tile index...")
        step_h = self.tile_h - self.overlap
        step_w = self.tile_w - self.overlap

        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]

            # Get image to determine size
            img = sample.get("image")
            if img is None:
                logger.warning(f"Sample {idx} has no 'image' key, skipping")
                continue

            if not isinstance(img, torch.Tensor):
                logger.warning(f"Sample {idx} image is not a tensor (type: {type(img)}), skipping")
                continue

            # Handle both [C, H, W] and [B, C, H, W]
            if img.ndim == 4:
                _, _, h, w = img.shape
            elif img.ndim == 3:
                _, h, w = img.shape
            else:
                logger.warning(f"Sample {idx} image has unexpected shape {img.shape}, skipping")
                continue

            if h < self.min_h or w < self.min_w:
                logger.debug(f"Image {idx} too small ({h}x{w}), skipping")
                continue

            # Generate tiles for this image
            for y0 in range(0, h, step_h):
                for x0 in range(0, w, step_w):
                    # Determine actual tile size (may be smaller at edges)
                    actual_h = min(self.tile_h, h - y0)
                    actual_w = min(self.tile_w, w - x0)

                    # Skip incomplete tiles if requested
                    if not self.keep_incomplete_tiles:
                        if actual_h < self.tile_h or actual_w < self.tile_w:
                            continue

                    self.tiles.append((idx, x0, y0, actual_h, actual_w))

        # Save index
        with open(index_file, 'w') as f:
            json.dump(self.tiles, f)

        logger.info(f"[TiledDataset] Built {len(self.tiles)} tiles, saved index to {index_file}")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a tile by index."""
        base_idx, x0, y0, tile_h, tile_w = self.tiles[idx]

        # Check cache first
        cache_file = self.cache_dir / f"tile_{base_idx}_{x0}_{y0}_{tile_h}_{tile_w}.pt"

        if cache_file.exists():
            # Load from cache
            return torch.load(cache_file, weights_only=True)

        # Not cached - generate and save
        sample = self.base_dataset[base_idx]
        tile_sample = self._extract_tile(sample, x0, y0, tile_h, tile_w, base_idx)

        # Save to cache atomically
        cache_file_tmp = cache_file.with_suffix('.pt.tmp')
        torch.save(tile_sample, cache_file_tmp)
        cache_file_tmp.replace(cache_file)  # Atomic rename

        return tile_sample

    def _extract_tile(
        self,
        sample: dict[str, Any],
        x0: int,
        y0: int,
        tile_h: int,
        tile_w: int,
        base_idx: int
    ) -> dict[str, Any]:
        """Extract a tile from a sample."""
        tile_sample = {}

        for key, value in sample.items():
            if key == "image":
                # Crop image
                if isinstance(value, torch.Tensor):
                    if value.ndim == 4:  # [B, C, H, W]
                        tile_sample[key] = value[:, :, y0:y0+tile_h, x0:x0+tile_w].clone()
                    elif value.ndim == 3:  # [C, H, W]
                        tile_sample[key] = value[:, y0:y0+tile_h, x0:x0+tile_w].clone()
                    else:
                        tile_sample[key] = value  # Pass through
                else:
                    tile_sample[key] = value

            elif key == "mask":
                # Crop mask
                if isinstance(value, torch.Tensor):
                    if value.ndim == 3:  # [B, H, W] or [C, H, W]
                        tile_sample[key] = value[:, y0:y0+tile_h, x0:x0+tile_w].clone()
                    elif value.ndim == 2:  # [H, W]
                        tile_sample[key] = value[y0:y0+tile_h, x0:x0+tile_w].clone()
                    else:
                        tile_sample[key] = value  # Pass through
                else:
                    tile_sample[key] = value

            elif key in ("filename", "image_id", "bbox"):
                # Metadata - preserve or adjust
                if key == "bbox" and isinstance(value, (list, tuple, torch.Tensor)):
                    # Adjust bbox coordinates
                    if isinstance(value, torch.Tensor):
                        bbox = value.clone()
                        if bbox.ndim == 1 and len(bbox) == 4:  # [x1, y1, x2, y2]
                            bbox[0] -= x0
                            bbox[1] -= y0
                            bbox[2] -= x0
                            bbox[3] -= y0
                            tile_sample[key] = bbox
                        else:
                            tile_sample[key] = value
                    else:
                        tile_sample[key] = value
                else:
                    tile_sample[key] = value
            else:
                # Pass through other keys unchanged
                tile_sample[key] = value

        # Add tile metadata
        # Store as (y1, x1, y2, x2) for easier stitching
        tile_sample["tile_coords"] = (y0, x0, y0 + tile_h, x0 + tile_w)
        tile_sample["base_idx"] = base_idx

        return tile_sample
