"""Reliability features and the PhysicsGate fit.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/physicsgate_utils.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-12


def as_float_array(values: object) -> np.ndarray:
    """Return a flat float array."""
    return np.asarray(values, dtype=float).reshape(-1)


def safe_output_path(path: Path) -> Path:
    """Avoid overwriting existing CSV outputs by appending a timestamp."""
    if not path.exists():
        return path
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def safe_sigmoid(values: object) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.clip(as_float_array(values), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(values: object, eps: float = 1e-4) -> np.ndarray:
    """Clipped logit transform for target gate weights."""
    p = np.clip(as_float_array(values), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def reliability_features(
    *,
    direct: object,
    physics: object,
    residual: object,
    y_train: object,
    aux_uncertainty: object | None = None,
    propagated_uncertainty: object | None = None,
) -> pd.DataFrame:
    """Build the six reliability features requested for PhysicsGate.

    R4 and R6 default to zero because the current per-sample exports do not
    include uncertainty estimates.  Their availability is recorded separately
    in the audit table.
    """
    direct_arr = as_float_array(direct)
    physics_arr = as_float_array(physics)
    residual_arr = as_float_array(residual)
    train_y = as_float_array(y_train)
    y_std = max(float(np.std(train_y, ddof=0)), EPS)
    y_min = float(np.min(train_y))
    y_max = float(np.max(train_y))
    y_range = max(y_max - y_min, EPS)

    if aux_uncertainty is None:
        r4 = np.zeros_like(direct_arr)
    else:
        r4 = as_float_array(aux_uncertainty) / y_std
    if propagated_uncertainty is None:
        r6 = np.zeros_like(direct_arr)
    else:
        r6 = as_float_array(propagated_uncertainty) / y_std

    outside_low = np.maximum(y_min - physics_arr, 0.0)
    outside_high = np.maximum(physics_arr - y_max, 0.0)

    return pd.DataFrame(
        {
            "R1_direct_physics_disagreement": np.abs(direct_arr - physics_arr) / y_std,
            "R2_direct_residual_disagreement": np.abs(direct_arr - residual_arr) / y_std,
            "R3_residual_physics_disagreement": np.abs(residual_arr - physics_arr) / y_std,
            "R4_auxiliary_uncertainty": r4,
            "R5_physics_out_of_range_amount": (outside_low + outside_high) / y_range,
            "R6_propagated_equation_uncertainty": r6,
        }
    )
