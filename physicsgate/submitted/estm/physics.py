"""ESTM physics routes and stabilization.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/run_estm_all_targets_wide_meta_a2.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from numpy.typing import NDArray

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

TARGET_COLUMNS = (
    "seebeck_uV_per_K",
    "log10_electrical_conductivity",
    "log10_thermal_conductivity",
    "ZT",
)

TARGET_DISPLAY = {
    "seebeck_uV_per_K": "Seebeck coefficient",
    "log10_electrical_conductivity": "log10 electrical conductivity",
    "log10_thermal_conductivity": "log10 thermal conductivity",
    "ZT": "ZT",
}


def _as_array(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def _positive(values: Any, floor: float = 1e-300) -> NDArray[np.float64]:
    return np.maximum(_as_array(values), floor)


def compute_estm_physics_prediction(
    target: str,
    predictions: dict[str, NDArray[np.float64]],
    temperature_K: NDArray[np.float64],
    *,
    direct_target_prediction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute target-specific physics prediction from predicted branches only."""
    temperature = _positive(temperature_K, floor=1e-12)
    seebeck = predictions.get("seebeck_uV_per_K")
    log_sigma = predictions.get("log10_electrical_conductivity")
    log_kappa = predictions.get("log10_thermal_conductivity")
    zt = predictions.get("ZT")

    if target == "ZT":
        if seebeck is None or log_sigma is None or log_kappa is None:
            raise ValueError("ZT physics requires predicted Seebeck, log_sigma, and log_kappa")
        seebeck_v = _as_array(seebeck) * 1e-6
        sigma = np.power(10.0, np.clip(_as_array(log_sigma), -30.0, 30.0))
        kappa = _positive(np.power(10.0, np.clip(_as_array(log_kappa), -30.0, 30.0)))
        return np.square(seebeck_v) * sigma * temperature / kappa

    if target == "seebeck_uV_per_K":
        if zt is None or log_sigma is None or log_kappa is None:
            raise ValueError("Seebeck physics requires predicted ZT, log_sigma, and log_kappa")
        sigma = _positive(np.power(10.0, np.clip(_as_array(log_sigma), -30.0, 30.0)))
        kappa = _positive(np.power(10.0, np.clip(_as_array(log_kappa), -30.0, 30.0)))
        magnitude_v = np.sqrt(_positive(_as_array(zt)) * kappa / (sigma * temperature))
        sign = np.sign(_as_array(direct_target_prediction))
        sign[sign == 0.0] = 1.0
        return sign * magnitude_v * 1e6

    if target == "log10_electrical_conductivity":
        if seebeck is None or log_kappa is None or zt is None:
            raise ValueError("log_sigma physics requires predicted Seebeck, log_kappa, and ZT")
        seebeck_v = _as_array(seebeck) * 1e-6
        kappa = _positive(np.power(10.0, np.clip(_as_array(log_kappa), -30.0, 30.0)))
        sigma = _positive(_as_array(zt)) * kappa / (_positive(np.square(seebeck_v)) * temperature)
        return np.log10(_positive(sigma))

    if target == "log10_thermal_conductivity":
        if seebeck is None or log_sigma is None or zt is None:
            raise ValueError("log_kappa physics requires predicted Seebeck, log_sigma, and ZT")
        seebeck_v = _as_array(seebeck) * 1e-6
        sigma = np.power(10.0, np.clip(_as_array(log_sigma), -30.0, 30.0))
        kappa = np.square(seebeck_v) * sigma * temperature / _positive(_as_array(zt))
        return np.log10(_positive(kappa))

    raise ValueError(f"Unknown ESTM target: {target}")


def stabilize_inverse_physics_branch(
    target: str,
    physics_prediction: NDArray[np.float64],
    y_train: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Clip ill-conditioned inverse ESTM branches using train-only target range.

    The ZT equation is well behaved when predicting ZT from auxiliary branches.
    The inverse tasks are not: predicted ZT near zero, negative ZT, or very small
    predicted Seebeck magnitude can send log_sigma/log_kappa/Seebeck physics
    branches to unphysical +/- hundreds.  This train-only range guard prevents a
    few numerical singularities from dominating physics-only and residual-only
    scores without using validation/test auxiliary targets.
    """
    values = _as_array(physics_prediction)
    if target == "ZT":
        return values
    finite_train = _as_array(y_train)
    finite_train = finite_train[np.isfinite(finite_train)]
    if finite_train.size == 0:
        return values
    lower = float(np.min(finite_train))
    upper = float(np.max(finite_train))
    if lower >= upper:
        return values
    return np.clip(values, lower, upper)


def complete_estm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["temperature_K", "formula_id", *TARGET_COLUMNS]
    data = frame.dropna(subset=required).copy()
    if data.empty:
        raise ValueError("No complete ESTM rows for all-target physics comparison")
    data["core_v2_row_index"] = np.arange(len(data), dtype=int)
    return data
