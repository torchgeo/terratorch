# Copyright contributors to the Terratorch project
# SPDX-License-Identifier: Apache-2.0
"""CPU smoke tests for the fine-tuning configs.

The full base set in test_base_set.py needs cluster storage under /dccstor,
GPU hardware and roughly 16 GB of checkpoints, so it cannot run on a
GitHub-hosted runner. These tests keep the same shape - generate a config,
shell out to "terratorch fit", assert it succeeds - but use a synthetic
dataset and a single epoch so they finish in minutes on CPU.

They cover the CLI wiring, the datamodule, the model factories and the
segmentation task. They do not verify accuracy or pretrained weights.

Run with:

    pytest integrationtests/test_smoke_fit.py -v
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
yaml = pytest.importorskip("yaml")

N_BANDS = 6
TILE_SIZE = 64
N_CLASSES = 2
TRAIN_TILES = 4
VAL_TILES = 2


def _write_tile(path: Path, bands: int, size: int, seed: int, kind: str) -> None:
    rng = np.random.default_rng(seed)
    if kind == "image":
        data = rng.integers(0, 2000, size=(bands, size, size), dtype=np.uint16)
    else:
        data = rng.integers(0, N_CLASSES, size=(1, size, size), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=data.shape[0],
        dtype=data.dtype.name,
    ) as dst:
        dst.write(data)


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory) -> dict:
    """A tiny image/mask dataset laid out the way the datamodule expects."""
    root = tmp_path_factory.mktemp("smoke_data")
    for split, count in (("train", TRAIN_TILES), ("val", VAL_TILES)):
        images = root / split
        images.mkdir(parents=True)
        labels = root / (split + "_labels")
        labels.mkdir(parents=True)
        for i in range(count):
            _write_tile(images / f"tile_{i:03d}_merged.tif", N_BANDS, TILE_SIZE, i, "image")
            _write_tile(labels / f"tile_{i:03d}_merged.mask.tif", 1, TILE_SIZE, i, "mask")
    return {
        "train": root / "train",
        "train_labels": root / "train_labels",
        "val": root / "val",
        "val_labels": root / "val_labels",
    }


def _base_config(work_dir: Path, n_bands: int) -> dict:
    return {
        "seed_everything": 0,
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32",
            "logger": False,
            "enable_checkpointing": True,
            "max_epochs": 1,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 1,
            "default_root_dir": str(work_dir),
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "dirpath": str(work_dir / "checkpoints"),
                        "mode": "min",
                        "monitor": "val/loss",
                        "filename": "best-{epoch:02d}",
                        "save_weights_only": True,
                    },
                }
            ],
        },
        "data": {
            "class_path": "terratorch.datamodules.GenericNonGeoSegmentationDataModule",
            "init_args": {
                "batch_size": 1,
                "num_workers": 0,
                "no_label_replace": -1,
                "no_data_replace": 0,
                "constant_scale": 1.0,
                "dataset_bands": [str(b) for b in range(n_bands)],
                "output_bands": [str(b) for b in range(n_bands)],
                "rgb_indices": [0, 1, 2],
                "means": [0.5] * n_bands,
                "stds": [0.25] * n_bands,
                "num_classes": N_CLASSES,
                "img_grep": "*_merged.tif",
                "label_grep": "*.mask.tif",
            },
        },
        "optimizer": {
            "class_path": "torch.optim.Adam",
            "init_args": {"lr": 6e-5, "weight_decay": 0.05},
        },
    }


VARIANTS = {
    "timm_resnet18": {
        "backbone": "timm_resnet18",
        "backbone_pretrained": False,
        "backbone_in_chans": N_BANDS,
        "necks": [{"name": "SelectIndices", "indices": [0, 1, 2, 3]}],
    },
    "prithvi_eo_v1_100": {
        "backbone": "prithvi_eo_v1_100",
        "backbone_pretrained": False,
        "backbone_bands": [str(b) for b in range(N_BANDS)],
        "backbone_drop_path": 0.0,
        "necks": [
            {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
            {"name": "ReshapeTokensToImage"},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
    },
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_fit_smoke(variant, synthetic_dataset, tmp_path):
    config = _base_config(tmp_path, N_BANDS)
    for key, value in (
        ("train_data_root", synthetic_dataset["train"]),
        ("train_label_data_root", synthetic_dataset["train_labels"]),
        ("val_data_root", synthetic_dataset["val"]),
        ("val_label_data_root", synthetic_dataset["val_labels"]),
        ("test_data_root", synthetic_dataset["val"]),
        ("test_label_data_root", synthetic_dataset["val_labels"]),
    ):
        config["data"]["init_args"][key] = str(value)

    model_args = dict(VARIANTS[variant])
    model_args.update(
        {
            "decoder": "UNetDecoder",
            "decoder_channels": [256, 128, 64, 32],
            "num_classes": N_CLASSES,
            "head_dropout": 0.1,
        }
    )
    config["model"] = {
        "class_path": "terratorch.tasks.SemanticSegmentationTask",
        "init_args": {
            "model_factory": "EncoderDecoderFactory",
            "model_args": model_args,
            "loss": "ce",
            "ignore_index": -1,
            "plot_on_val": 0,
            "freeze_backbone": False,
            "freeze_decoder": False,
        },
    }

    config_path = tmp_path / (variant + ".yaml")
    config_path.write_text(yaml.safe_dump(config))

    result = subprocess.run(
        [sys.executable, "-m", "terratorch", "fit", "-c", str(config_path)],
        capture_output=True,
        text=True,
        timeout=1800,
    )

    assert result.returncode == 0, (
        "terratorch fit failed for " + variant + " (exit " + str(result.returncode) + ")\n"
        "--- stdout ---\n" + result.stdout[-4000:] + "\n--- stderr ---\n" + result.stderr[-4000:]
    )

    checkpoints = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert checkpoints, "no checkpoint written for " + variant
