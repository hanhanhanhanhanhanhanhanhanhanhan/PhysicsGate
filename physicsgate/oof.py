"""Out-of-fold prediction API contracts."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.model_selection import KFold


def _as_2d_array(X: ArrayLike) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("X must be a 1D or 2D array")
    return array


def _as_1d_array(y: ArrayLike) -> np.ndarray:
    array = np.asarray(y, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise ValueError("y must be one-dimensional")
    return array


def generate_oof_predictions(
    estimator: Any,
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    folds: Iterable[tuple[ArrayLike, ArrayLike]],
) -> ArrayLike:
    """Generate predictions for training rows from models that did not fit them."""
    X = _as_2d_array(X_train)
    y = _as_1d_array(y_train)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X_train and y_train must have equal lengths")

    predictions = np.full(y.shape[0], np.nan, dtype=float)
    fold_counts = np.zeros(y.shape[0], dtype=int)
    for train_idx, valid_idx in folds:
        train_idx = np.asarray(train_idx, dtype=int)
        valid_idx = np.asarray(valid_idx, dtype=int)
        fold_model = clone(estimator)
        fold_model.fit(X[train_idx], y[train_idx])
        fold_pred = np.asarray(fold_model.predict(X[valid_idx]), dtype=float).reshape(-1)
        if fold_pred.shape[0] != valid_idx.shape[0]:
            raise ValueError("Fold estimator returned unexpected prediction length")
        predictions[valid_idx] = fold_pred
        fold_counts[valid_idx] += 1
    if not np.all(fold_counts == 1):
        raise ValueError("Each training row must receive exactly one OOF prediction")
    return predictions


def fit_predict_oof(
    estimator: Any,
    X_train: ArrayLike,
    y_train: ArrayLike,
    *,
    cv: int,
    random_state: int | None = None,
) -> ArrayLike:
    """Fit fold models and return OOF predictions for training data."""
    X = _as_2d_array(X_train)
    y = _as_1d_array(y_train)
    if not 2 <= cv <= len(y):
        raise ValueError("cv must be between 2 and the number of training rows")
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    return generate_oof_predictions(estimator, X, y, folds=splitter.split(X, y))


def refit_and_predict_test(
    estimator: Any,
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
) -> ArrayLike:
    """Refit on all training data and predict the untouched test set."""
    X = _as_2d_array(X_train)
    y = _as_1d_array(y_train)
    X_test_array = _as_2d_array(X_test)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X_train and y_train must have equal lengths")
    fitted = clone(estimator).fit(X, y)
    return np.asarray(fitted.predict(X_test_array), dtype=float).reshape(-1)


__all__ = [
    "fit_predict_oof",
    "generate_oof_predictions",
    "refit_and_predict_test",
]
