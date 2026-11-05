# Copyright contributors to the Terratorch project

import importlib
import logging

import numpy as np
import torch
from torch import Tensor, nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY

logger = logging.getLogger("terratorch")


@TERRATORCH_DECODER_REGISTRY.register
class SklearnDecoder(nn.Module):
    """sklearn estimator wrapped as a decoder. Not differentiable,
    features are accumulated during training and fit() is called at epoch end."""

    includes_head: bool = True

    def __init__(
        self,
        embed_dim: list[int],
        num_classes: int,
        in_index: int = -1,
        estimator_class: str = "sklearn.ensemble.RandomForestClassifier",
        **estimator_kwargs,
    ) -> None:
        super().__init__()
        self.in_index = in_index
        self.num_classes = num_classes
        self._embed_dim = embed_dim[in_index]

        # CLI overrides come in as strings, cast to int/float where possible
        for k, v in list(estimator_kwargs.items()):
            if isinstance(v, str):
                try:
                    estimator_kwargs[k] = int(v)
                except ValueError:
                    try:
                        estimator_kwargs[k] = float(v)
                    except ValueError:
                        pass

        # resolve estimator class from dotted path
        module_path, class_name = estimator_class.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        self.estimator = cls(**estimator_kwargs)
        self._estimator_name = class_name

        self._fitted: bool = False
        self._X_buf: list[np.ndarray] = []

    @property
    def out_channels(self) -> int:
        return self.num_classes

    def forward(self, x: list[Tensor]) -> Tensor:
        features = x[self.in_index]

        if features.dim() == 4:
            features = features.mean(dim=[2, 3])

        if features.dim() != 2:
            raise ValueError(
                f"expected 2-D or 4-D features, got shape {tuple(features.shape)}"
            )

        features_np = features.detach().cpu().numpy().astype(np.float32)

        if self.training:
            self._X_buf.append(features_np)
            return torch.zeros(features.shape[0], self.num_classes, device=features.device)

        if not self._fitted:
            return torch.zeros(features.shape[0], self.num_classes, device=features.device)

        proba = self._predict_proba(features_np)
        logits = self._proba_to_logits(proba)
        return torch.from_numpy(logits).float().to(features.device)

    def fit(self, y: np.ndarray) -> None:
        if not self._X_buf:
            raise RuntimeError("no features accumulated, run a training epoch first")

        X = np.concatenate(self._X_buf, axis=0)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"feature buffer has {X.shape[0]} samples but y has {y.shape[0]}"
            )

        logger.info("fitting %s on %d samples, %d features", self._estimator_name, X.shape[0], X.shape[1])
        self.estimator.fit(X, y)
        self._fitted = True
        self._X_buf = []

    def reset(self) -> None:
        self._X_buf = []
        self._fitted = False

    @staticmethod
    def _proba_to_logits(proba: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        proba = np.clip(proba, eps, 1.0 - eps)
        return np.log(proba / (1.0 - proba)).astype(np.float32)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        # sklearn may return fewer columns if not all classes appeared in training
        proba_raw = self.estimator.predict_proba(X)
        B = X.shape[0]

        if isinstance(proba_raw, list):
            # multi-label: one array per label
            cols = []
            for i, p in enumerate(proba_raw):
                if p.shape[1] == 1:
                    observed = self.estimator.classes_[i][0]
                    cols.append(p[:, 0] if observed == 1 else np.zeros(B, dtype=np.float32))
                else:
                    cols.append(p[:, 1])
            return np.stack(cols, axis=1).astype(np.float32)

        observed_classes = self.estimator.classes_
        if len(observed_classes) == self.num_classes:
            return proba_raw.astype(np.float32)

        # pad missing class columns
        full = np.zeros((B, self.num_classes), dtype=np.float32)
        for col_idx, cls in enumerate(observed_classes):
            full[:, int(cls)] = proba_raw[:, col_idx]
        return full
