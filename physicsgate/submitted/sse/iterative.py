"""SSE shared helpers.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/arrhenius/run_sse_iterative_target_adaptive.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = WORKDIR

from physicsgate.submitted.common.equations import ARRHENIUS_KB_EV_PER_K  # noqa: E402
from physicsgate.submitted.common.oof import generate_oof_predictions  # noqa: E402

TARGET_ORDER = ("L", "P", "E")

TARGET_COLUMNS = {
    "L": "ln_sigma_T",
    "P": "ln_sigma0",
    "E": "Ea",
}


def _as_1d(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _as_2d(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    return array


def _direct_features(X_base: ArrayLike, T: ArrayLike) -> NDArray[np.float64]:
    X = _as_2d(X_base, name="X_base")
    temperature = _as_1d(T, name="T")
    if X.shape[0] != temperature.shape[0]:
        raise ValueError("X_base and T must have equal lengths")
    return np.column_stack([X, temperature])


def _group_oof_predictions(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    *,
    n_splits: int,
) -> NDArray[np.float64]:
    X_array = _as_2d(X, name="X")
    y_array = _as_1d(y, name="y")
    group_array = np.asarray(groups)
    unique_groups = np.unique(group_array)
    split_count = min(int(n_splits), len(unique_groups))
    if split_count < 2:
        raise ValueError("At least two training groups are required for group OOF")
    splitter = GroupKFold(n_splits=split_count)
    return np.asarray(
        generate_oof_predictions(
            estimator,
            X_array,
            y_array,
            folds=splitter.split(X_array, y_array, group_array),
        ),
        dtype=float,
    ).reshape(-1)


def _fit_and_predict_test(
    estimator: Any,
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
) -> NDArray[np.float64]:
    X = _as_2d(X_train, name="X_train")
    y = _as_1d(y_train, name="y_train")
    fitted = clone(estimator).fit(X, y)
    return np.asarray(fitted.predict(_as_2d(X_test, name="X_test")), dtype=float).reshape(-1)


def physics_prediction_for_target(
    target: str,
    predictions: Mapping[str, ArrayLike],
    T: ArrayLike,
    *,
    kB: float = ARRHENIUS_KB_EV_PER_K,
) -> NDArray[np.float64]:
    """Build target-specific Arrhenius physics predictions from predictions."""
    temperature = _as_1d(T, name="T")
    if target == "L":
        P = _as_1d(predictions["P"], name="P_pred")
        E = _as_1d(predictions["E"], name="E_pred")
        return P - E / (kB * temperature)
    if target == "P":
        L = _as_1d(predictions["L"], name="L_pred")
        E = _as_1d(predictions["E"], name="E_pred")
        return L + E / (kB * temperature)
    if target == "E":
        P = _as_1d(predictions["P"], name="P_pred")
        L = _as_1d(predictions["L"], name="L_pred")
        return kB * temperature * (P - L)
    raise ValueError(f"Unknown target: {target}")


def _split_group_indices(
    groups: ArrayLike,
    *,
    test_size: float,
    random_state: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    dummy_X = np.zeros((len(groups), 1), dtype=float)
    train_idx, test_idx = next(splitter.split(dummy_X, groups=groups))
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)
