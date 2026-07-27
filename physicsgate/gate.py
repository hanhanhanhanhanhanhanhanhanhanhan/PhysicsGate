"""Leakage-safe two-branch PhysicsGate used in the manuscript.

The gate combines an out-of-fold direct prediction and an out-of-fold
physics-anchored residual prediction:

    A_gate = A_direct + w (A_residual - A_direct)

where ``w = sigmoid(g(R1, R2, R3))`` and ``g`` is a standardized ridge model.
All arrays supplied to ``fit`` must be training-side out-of-fold predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
FEATURES = ("R1", "R2", "R3")


def _one_dimensional(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(_one_dimensional(values), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(_one_dimensional(values), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def reliability_features(
    *,
    direct: np.ndarray,
    physics: np.ndarray,
    residual: np.ndarray,
    training_targets: np.ndarray,
    propagated_uncertainty: np.ndarray,
) -> pd.DataFrame:
    """Return the three manuscript reliability features.

    R1 is the normalized disagreement between the two branches being blended.
    R2 is the normalized amount by which the physics prediction lies outside
    the outer-training target range.
    R3 is the normalized equation-propagation uncertainty estimate.
    """

    direct = _one_dimensional(direct)
    physics = _one_dimensional(physics)
    residual = _one_dimensional(residual)
    targets = _one_dimensional(training_targets)
    uncertainty = _one_dimensional(propagated_uncertainty)
    if not (len(direct) == len(physics) == len(residual) == len(uncertainty)):
        raise ValueError("Prediction and uncertainty arrays must have equal length")

    scale = max(float(np.std(targets, ddof=0)), EPS)
    target_min = float(np.min(targets))
    target_max = float(np.max(targets))
    target_range = max(target_max - target_min, EPS)
    below = np.maximum(target_min - physics, 0.0)
    above = np.maximum(physics - target_max, 0.0)
    return pd.DataFrame(
        {
            "R1": np.abs(direct - residual) / scale,
            "R2": np.maximum(below, above) / target_range,
            "R3": uncertainty / scale,
        }
    )


@dataclass
class PhysicsGate:
    """Fitted standardized-ridge gate with weights constrained to [0, 1]."""

    alpha: float = 1.0
    model: object | None = None

    def fit(
        self,
        *,
        y_oof: np.ndarray,
        direct_oof: np.ndarray,
        residual_oof: np.ndarray,
        reliability_oof: pd.DataFrame,
    ) -> "PhysicsGate":
        y = _one_dimensional(y_oof)
        direct = _one_dimensional(direct_oof)
        residual = _one_dimensional(residual_oof)
        delta = residual - direct
        usable = np.abs(delta) > EPS
        optimal_weight = np.zeros_like(y)
        optimal_weight[usable] = np.clip(
            (y[usable] - direct[usable]) / delta[usable],
            0.0,
            1.0,
        )
        sample_weight = delta**2
        if float(np.sum(sample_weight)) <= EPS:
            self.model = None
            return self
        model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        model.fit(
            reliability_oof.loc[:, FEATURES].to_numpy(dtype=float),
            logit(optimal_weight),
            ridge__sample_weight=sample_weight,
        )
        self.model = model
        return self

    def weights(self, reliability: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(reliability), dtype=float)
        return sigmoid(
            self.model.predict(reliability.loc[:, FEATURES].to_numpy(dtype=float))
        )

    def predict(
        self,
        *,
        direct: np.ndarray,
        residual: np.ndarray,
        reliability: pd.DataFrame,
    ) -> np.ndarray:
        direct_array = _one_dimensional(direct)
        residual_array = _one_dimensional(residual)
        weight = self.weights(reliability)
        return direct_array + weight * (residual_array - direct_array)


__all__ = ["FEATURES", "PhysicsGate", "reliability_features", "sigmoid"]
