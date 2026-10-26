"""The gate fitted to the error of the blended prediction, with non-positive risk coefficients (Table S12).

Not part of the revised protocol. Drop-in replacement of
``physicsgate.submitted.common.gate_fit.fit_gate_model``, through which every
PhysicsGate gate of the benchmark is fitted. The reported gate regresses the logit of each
training sample's clipped optimal weight, clip((y - D) / (R - D), 0, 1), with
weights (R - D)^2; most of those weights sit at 0 or 1, so the regression
target is close to two large constants and the fit is not the error of the
blend. Here the weight w = sigmoid(b0 + b'z) of the standardized reliability
descriptors z is fitted to the blend itself,

    minimize  mean[(y - D - w (R - D))^2] / mean[(R - D)^2] + alpha |b|^2,

with an unpenalized intercept, starting from the best constant weight on the
training rows. The coefficients of the physics range risk R2 and the propagated
uncertainty R3 are non-positive, so that the weight of the physics-anchored
branch never increases with either risk; the branch disagreement R1 is free.
As alpha grows the gate tends to the best fixed blend. The descriptors, the
out-of-fold inputs, the alpha grids and their selection are unchanged.

    python -m physicsgate.revised.gate <module> [module arguments]

installs the gate and runs the module; ``scripts/reproduce_submitted.py
--protocol blend_gate`` runs the stages of the revised benchmark this way.
"""

from __future__ import annotations

import runpy
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

EPS = 1e-12
# code names of R2 (physics range risk) and R3 (propagated uncertainty) in the paper's notation
RISK_FEATURES = ("R5_physics_out_of_range_amount", "R6_global_propagated_uncertainty_proxy",
                 "R6_propagated_equation_uncertainty")


def _as_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


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
    y, direct, residual = _as_1d(y_train), _as_1d(direct_train), _as_1d(residual_train)
    direct_t, residual_t = _as_1d(direct_test), _as_1d(residual_test)
    delta, gap = residual - direct, y - direct
    scale = float(np.mean(delta ** 2))
    if scale <= EPS:
        return direct.copy(), direct_t.copy(), np.zeros_like(direct_t)
    Z = reliability_train[features].to_numpy(dtype=float)
    mean, std = Z.mean(axis=0), Z.std(axis=0)
    std[std <= EPS] = 1.0
    Z = (Z - mean) / std
    Z_test = (reliability_test[features].to_numpy(dtype=float) - mean) / std
    n = len(y)

    def loss(theta):
        w = expit(theta[0] + Z @ theta[1:])
        error = gap - w * delta
        dw = -2.0 * error * delta / (n * scale) * w * (1.0 - w)
        value = float(np.mean(error ** 2) / scale + alpha * theta[1:] @ theta[1:])
        return value, np.concatenate([[dw.sum()], Z.T @ dw + 2.0 * alpha * theta[1:]])

    w0 = np.clip(np.sum(delta * gap) / np.sum(delta ** 2), 1e-4, 1.0 - 1e-4)  # best constant weight
    bounds = [(None, None)] + [(None, 0.0) if f in RISK_FEATURES else (None, None) for f in features]
    theta = minimize(loss, np.concatenate([[logit(w0)], np.zeros(Z.shape[1])]), jac=True, method="L-BFGS-B",
                     bounds=bounds).x
    train_w = expit(theta[0] + Z @ theta[1:])
    test_w = expit(theta[0] + Z_test @ theta[1:])
    return direct + train_w * delta, direct_t + test_w * (residual_t - direct_t), test_w


def install() -> None:
    """Replace the gate before the pipelines import it (they bind it by name on import)."""
    from physicsgate.submitted.common import gate_fit
    gate_fit.fit_gate_model = fit_gate_model


def main() -> None:
    install()
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
