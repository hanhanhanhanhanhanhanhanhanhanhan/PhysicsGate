"""Gate model fit and grouped inner folds.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/run_sse_all_targets_physicsgate_r4_r6_tuned.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

from physicsgate.submitted.common.gate_features import safe_logit, safe_sigmoid  # noqa: E402

KB = 8.617333262e-5

EPS = 1e-12


def _as_1d(values: object) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def phi_l(aux: np.ndarray, known: dict[str, np.ndarray]) -> np.ndarray:
    """L = P - Ea/(kB*T)."""
    return aux[:, 0] - aux[:, 1] / (KB * np.maximum(known["T_K"], EPS))


def phi_p(aux: np.ndarray, known: dict[str, np.ndarray]) -> np.ndarray:
    """P = L + Ea/(kB*T)."""
    return aux[:, 0] + aux[:, 1] / (KB * np.maximum(known["T_K"], EPS))


def phi_e(aux: np.ndarray, known: dict[str, np.ndarray]) -> np.ndarray:
    """Ea = kB*T*(P - L)."""
    return KB * np.maximum(known["T_K"], EPS) * (aux[:, 1] - aux[:, 0])


def metric_dict(y_true: object, y_pred: object) -> dict[str, float]:
    truth = _as_1d(y_true)
    pred = _as_1d(y_pred)
    rmse = float(np.sqrt(mean_squared_error(truth, pred)))
    std = float(np.std(truth, ddof=0))
    return {
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(truth, pred)),
        "nRMSE": float(rmse / std) if std > 0 else np.nan,
        "R2": float(r2_score(truth, pred)),
    }


def series_arrays(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tmp = pd.DataFrame(
        {
            "arrhenius_series_id": frame["arrhenius_series_id"].astype(str).to_numpy(),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    grouped = tmp.groupby("arrhenius_series_id", as_index=False)[["y_true", "y_pred"]].mean()
    return grouped["y_true"].to_numpy(dtype=float), grouped["y_pred"].to_numpy(dtype=float)


def reliability_features(
    *,
    direct: np.ndarray,
    physics: np.ndarray,
    residual: np.ndarray,
    y_train: np.ndarray,
    r4: np.ndarray | None,
    sigma_phys: np.ndarray,
) -> pd.DataFrame:
    direct = _as_1d(direct)
    physics = _as_1d(physics)
    residual = _as_1d(residual)
    y_train = _as_1d(y_train)
    y_scale = max(float(np.std(y_train, ddof=0)), EPS)
    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    y_range = max(y_max - y_min, EPS)
    lower = np.maximum(y_min - physics, 0.0)
    upper = np.maximum(physics - y_max, 0.0)
    features = {
        "R1_direct_physics_disagreement": np.abs(direct - physics) / y_scale,
        "R2_direct_residual_disagreement": np.abs(direct - residual) / y_scale,
        "R3_residual_physics_correction_size": np.abs(residual - physics) / y_scale,
        "R5_physics_out_of_range_amount": np.maximum(lower, upper) / y_range,
        "R6_global_propagated_uncertainty_proxy": _as_1d(sigma_phys) / y_scale,
    }
    if r4 is not None:
        features["R4_local_auxiliary_oof_error_proxy"] = _as_1d(r4)
    return pd.DataFrame(features)


def split_positions(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups, dtype=object)
    unique_groups = np.unique(groups)
    n_use = min(max(2, n_splits), len(unique_groups))
    if len(unique_groups) >= 2:
        return list(GroupKFold(n_splits=n_use).split(np.arange(len(groups)), groups=groups))
    return list(KFold(n_splits=min(n_use, len(groups)), shuffle=True, random_state=seed).split(np.arange(len(groups))))


def fit_gate_model(
    *,
    y_train: np.ndarray,
    direct_train: np.ndarray,
    direct_test: np.ndarray,
    residual_train: np.ndarray,
    residual_test: np.ndarray,
    reliability_train: pd.DataFrame,
    reliability_test: pd.DataFrame,
    features: list[str],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _as_1d(y_train)
    direct = _as_1d(direct_train)
    residual = _as_1d(residual_train)
    delta = residual - direct
    usable = np.abs(delta) > EPS
    w_star = np.zeros_like(y)
    w_star[usable] = np.clip((y[usable] - direct[usable]) / delta[usable], 0.0, 1.0)
    target = safe_logit(w_star)
    sample_weight = delta**2
    if float(np.sum(sample_weight)) <= EPS:
        train_w = np.zeros_like(y)
        test_w = np.zeros_like(_as_1d(direct_test))
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(
            reliability_train[features].to_numpy(dtype=float),
            target,
            ridge__sample_weight=sample_weight,
        )
        train_w = safe_sigmoid(model.predict(reliability_train[features].to_numpy(dtype=float)))
        test_w = safe_sigmoid(model.predict(reliability_test[features].to_numpy(dtype=float)))
    train_pred = direct + train_w * (residual - direct)
    test_pred = _as_1d(direct_test) + test_w * (_as_1d(residual_test) - _as_1d(direct_test))
    return train_pred, test_pred, test_w
