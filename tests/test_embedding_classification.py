# Copyright contributors to the Terratorch project

"""Unit tests for the sklearn-on-frozen-embeddings feature (PR #1160):
SklearnDecoder, EmbeddingDataset and EmbeddingClassificationDataModule."""

import numpy as np
import pytest
import torch

from terratorch.datamodules.embedding_classification_data_module import (
    EmbeddingClassificationDataModule,
)
from terratorch.datasets.embedding_dataset import EmbeddingDataset
from terratorch.models.decoders.sklearn_decoder import SklearnDecoder
from terratorch.registry import TERRATORCH_DECODER_REGISTRY

EMBED_DIM = 16
NUM_CLASSES = 3
N_SAMPLES = 20


def _make_decoder(**kwargs) -> SklearnDecoder:
    kwargs.setdefault("n_estimators", 5)
    kwargs.setdefault("random_state", 0)
    return SklearnDecoder(
        embed_dim=[EMBED_DIM],
        num_classes=NUM_CLASSES,
        estimator_class="sklearn.ensemble.RandomForestClassifier",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# SklearnDecoder
# --------------------------------------------------------------------------- #
def test_decoder_is_registered():
    assert "SklearnDecoder" in TERRATORCH_DECODER_REGISTRY


def test_decoder_basic_attributes():
    dec = _make_decoder()
    assert dec.out_channels == NUM_CLASSES
    assert dec.includes_head is True
    assert dec._embed_dim == EMBED_DIM


def test_string_kwargs_are_cast():
    # CLI passes everything as strings; ints/floats should be recovered,
    # non-numeric strings left untouched.
    dec = _make_decoder(n_estimators="7", max_features="sqrt")
    assert dec.estimator.n_estimators == 7
    assert dec.estimator.max_features == "sqrt"


def test_training_forward_accumulates_and_returns_zeros():
    dec = _make_decoder()
    dec.train()
    feats = torch.randn(N_SAMPLES, EMBED_DIM)
    out = dec([feats])
    assert out.shape == (N_SAMPLES, NUM_CLASSES)
    assert torch.count_nonzero(out) == 0
    assert len(dec._X_buf) == 1


def test_eval_forward_before_fit_returns_zeros():
    dec = _make_decoder()
    dec.eval()
    feats = torch.randn(N_SAMPLES, EMBED_DIM)
    out = dec([feats])
    assert out.shape == (N_SAMPLES, NUM_CLASSES)
    assert torch.count_nonzero(out) == 0


def test_forward_pools_4d_features():
    dec = _make_decoder()
    dec.train()
    feats = torch.randn(N_SAMPLES, EMBED_DIM, 4, 4)  # B, C, H, W
    out = dec([feats])
    assert out.shape == (N_SAMPLES, NUM_CLASSES)
    # buffered features must be pooled to (B, C)
    assert dec._X_buf[0].shape == (N_SAMPLES, EMBED_DIM)


def test_forward_rejects_3d_features():
    dec = _make_decoder()
    dec.train()
    with pytest.raises(ValueError, match="2-D or 4-D"):
        dec([torch.randn(N_SAMPLES, EMBED_DIM, 4)])


def test_fit_without_features_raises():
    dec = _make_decoder()
    with pytest.raises(RuntimeError, match="no features accumulated"):
        dec.fit(np.zeros((N_SAMPLES, NUM_CLASSES), dtype=int))


def test_fit_sample_count_mismatch_raises():
    dec = _make_decoder()
    dec.train()
    dec([torch.randn(N_SAMPLES, EMBED_DIM)])
    with pytest.raises(ValueError, match="samples but y has"):
        dec.fit(np.zeros((N_SAMPLES - 1, NUM_CLASSES), dtype=int))


def test_reset_clears_state():
    dec = _make_decoder()
    dec.train()
    dec([torch.randn(N_SAMPLES, EMBED_DIM)])
    dec._fitted = True
    dec.reset()
    assert dec._X_buf == []
    assert dec._fitted is False


def test_full_multilabel_fit_predict_cycle():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dec = _make_decoder()

    feats = torch.randn(N_SAMPLES, EMBED_DIM)
    y = rng.integers(0, 2, size=(N_SAMPLES, NUM_CLASSES)).astype(int)
    # guarantee every label column has both classes so RF trains per-label
    y[0, :] = 1
    y[1, :] = 0

    dec.train()
    dec([feats])
    dec.fit(y)
    assert dec._fitted is True
    assert dec._X_buf == []  # buffer flushed after fit

    dec.eval()
    logits = dec([feats])
    assert logits.shape == (N_SAMPLES, NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_predict_proba_pads_missing_single_label_classes():
    # num_classes larger than the classes actually observed in training ->
    # missing columns must be zero-padded, output still (B, num_classes).
    dec = SklearnDecoder(
        embed_dim=[EMBED_DIM],
        num_classes=4,
        n_estimators=5,
        random_state=0,
    )
    feats = torch.randn(N_SAMPLES, EMBED_DIM)
    y = np.array([0, 1, 2] * (N_SAMPLES // 3) + [0] * (N_SAMPLES % 3))  # class 3 never seen
    dec.train()
    dec([feats])
    dec.fit(y)
    dec.eval()
    logits = dec([feats])
    assert logits.shape == (N_SAMPLES, 4)
    assert torch.isfinite(logits).all()


def test_proba_to_logits_is_monotonic_and_bounded():
    proba = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    logits = SklearnDecoder._proba_to_logits(proba)
    assert np.isfinite(logits).all()             # clipping keeps 0 and 1 finite
    assert logits[0, 0] < logits[0, 1] < logits[0, 2]  # monotonic in probability
    assert abs(logits[0, 1]) < 1e-6              # p=0.5 -> logit 0


# --------------------------------------------------------------------------- #
# EmbeddingDataset
# --------------------------------------------------------------------------- #
def _write_embedding_files(dir_path, n=N_SAMPLES, d=EMBED_DIM, c=NUM_CLASSES, n_labels=None):
    n_labels = n if n_labels is None else n_labels
    torch.save(torch.randn(n, d), dir_path / "embeddings.pt")
    torch.save(torch.randint(0, 2, (n_labels, c)), dir_path / "labels.pt")
    torch.save([f"patch_{i}.tif" for i in range(n)], dir_path / "patch_ids.pt")


def test_embedding_dataset_getitem(tmp_path):
    _write_embedding_files(tmp_path)
    ds = EmbeddingDataset(tmp_path)
    assert len(ds) == N_SAMPLES
    sample = ds[0]
    assert set(sample) == {"image", "label", "filename"}
    assert sample["image"].shape == (EMBED_DIM,)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].shape == (NUM_CLASSES,)
    assert sample["filename"] == "patch_0.tif"


def test_embedding_dataset_loads_patch_ids_safely(tmp_path):
    # Regression guard: patch_ids.pt must load under weights_only=True
    # (PR #1160 originally used weights_only=False, undoing the #1208 hardening).
    _write_embedding_files(tmp_path)
    ds = EmbeddingDataset(tmp_path)
    assert ds.patch_ids[-1] == f"patch_{N_SAMPLES - 1}.tif"


def test_embedding_dataset_size_mismatch_raises(tmp_path):
    _write_embedding_files(tmp_path, n_labels=N_SAMPLES - 2)
    with pytest.raises(ValueError, match="size mismatch"):
        EmbeddingDataset(tmp_path)


# --------------------------------------------------------------------------- #
# EmbeddingClassificationDataModule
# --------------------------------------------------------------------------- #
def test_datamodule_setup_and_dataloaders(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    _write_embedding_files(train_dir)
    _write_embedding_files(val_dir, n=8)

    dm = EmbeddingClassificationDataModule(
        batch_size=4,
        num_workers=0,
        train_data_root=str(train_dir),
        val_data_root=str(val_dir),
        drop_last=False,
    )
    dm.setup("fit")
    assert len(dm.train_dataset) == N_SAMPLES
    assert len(dm.val_dataset) == 8

    batch = next(iter(dm.train_dataloader()))
    assert batch["image"].shape == (4, EMBED_DIM)
    assert batch["label"].shape == (4, NUM_CLASSES)
    assert len(batch["filename"]) == 4


def test_datamodule_setup_without_roots_is_noop():
    dm = EmbeddingClassificationDataModule(batch_size=2)
    dm.setup("fit")
    assert dm.train_dataset is None
    assert dm.val_dataset is None
