"""Regression metrics.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/physstack/metrics.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _paired_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("metric inputs must have matching shapes")
    return truth, prediction


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    truth, prediction = _paired_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    truth, prediction = _paired_arrays(y_true, y_pred)
    return float(np.mean(np.abs(truth - prediction)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination."""
    truth, prediction = _paired_arrays(y_true, y_pred)
    denominator = np.sum(np.square(truth - np.mean(truth)))
    if denominator == 0:
        raise ValueError("R2 is undefined when y_true is constant")
    numerator = np.sum(np.square(truth - prediction))
    return float(1.0 - numerator / denominator)
