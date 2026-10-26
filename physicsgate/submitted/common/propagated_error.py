"""Propagated auxiliary-error proxy (R3).

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/physicsgate_global_uncertainty_utils.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from physicsgate.submitted.common.gate_features import as_float_array, safe_output_path  # noqa: E402

PhiFunction = Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray]


def rmse_value(y_true: object, y_pred: object) -> float:
    """Return RMSE."""
    truth = as_float_array(y_true)
    pred = as_float_array(y_pred)
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def global_auxiliary_sigmas(
    *,
    aux_pred_train: np.ndarray,
    aux_true_train: np.ndarray,
    aux_names: list[str],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Estimate global auxiliary OOF RMSE values from train data only."""
    pred = np.asarray(aux_pred_train, dtype=float)
    true = np.asarray(aux_true_train, dtype=float)
    sigmas = []
    rows = []
    for idx, name in enumerate(aux_names):
        mask = np.isfinite(pred[:, idx]) & np.isfinite(true[:, idx])
        sigma = rmse_value(true[mask, idx], pred[mask, idx]) if mask.sum() else np.nan
        aux_std = float(np.std(true[mask, idx], ddof=0)) if mask.sum() else np.nan
        sigmas.append(float(sigma))
        rows.append(
            {
                "auxiliary_variable": name,
                "sigma_z_global": float(sigma),
                "sigma_z_global_definition": "RMSE(auxiliary OOF prediction, true train auxiliary)",
                "auxiliary_train_std": aux_std,
                "normalized_auxiliary_rmse": float(sigma / aux_std) if aux_std > 0 else np.nan,
                "n_train_aux": int(mask.sum()),
                "derivative_method": "finite_difference_or_analytic",
                "notes": "Computed from train-side OOF auxiliary predictions only.",
            }
        )
    return np.asarray(sigmas, dtype=float), rows


def propagated_sigma_finite_difference(
    *,
    aux_pred: np.ndarray,
    known: dict[str, np.ndarray],
    aux_sigmas: np.ndarray,
    phi: PhiFunction,
) -> tuple[np.ndarray, dict[str, int]]:
    """Propagate global auxiliary RMSE through Phi using finite differences."""
    aux = np.asarray(aux_pred, dtype=float)
    sigmas = np.asarray(aux_sigmas, dtype=float)
    n_rows, n_aux = aux.shape
    base = phi(aux, known)
    variance = np.zeros(n_rows, dtype=float)
    audit = {
        "central_difference_failures": 0,
        "one_sided_fallback_cases": 0,
        "zero_derivative_fallback_cases": 0,
    }
    for col in range(n_aux):
        step = np.maximum(1e-6, 1e-4 * np.abs(aux[:, col]))
        plus = aux.copy()
        minus = aux.copy()
        plus[:, col] += step
        minus[:, col] -= step
        with np.errstate(all="ignore"):
            plus_phi = phi(plus, known)
            minus_phi = phi(minus, known)
            central = (plus_phi - minus_phi) / (2.0 * step)
            one_sided = (plus_phi - base) / step
        derivative = central.copy()
        bad = ~np.isfinite(derivative)
        if np.any(bad):
            audit["central_difference_failures"] += int(np.sum(bad))
            derivative[bad] = one_sided[bad]
            still_bad = ~np.isfinite(derivative)
            audit["one_sided_fallback_cases"] += int(np.sum(bad & ~still_bad))
            if np.any(still_bad):
                audit["zero_derivative_fallback_cases"] += int(np.sum(still_bad))
                derivative[still_bad] = 0.0
        sigma = sigmas[col] if np.isfinite(sigmas[col]) else 0.0
        variance += (derivative * sigma) ** 2
    return np.sqrt(np.maximum(variance, 0.0)), audit
