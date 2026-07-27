from __future__ import annotations

import numpy as np

from physicsgate.gate import PhysicsGate, reliability_features


def test_physicsgate_weights_are_bounded_and_predictions_are_convex() -> None:
    y = np.array([0.2, 1.1, 1.8, 3.2, 3.9, 5.1])
    direct = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    physics = np.array([0.4, 1.2, 1.7, 3.4, 3.7, 5.2])
    residual = np.array([0.3, 1.1, 1.9, 3.3, 3.8, 5.1])
    uncertainty = np.full_like(y, 0.1)
    reliability = reliability_features(
        direct=direct,
        physics=physics,
        residual=residual,
        training_targets=y,
        propagated_uncertainty=uncertainty,
    )
    gate = PhysicsGate(alpha=1.0).fit(
        y_oof=y,
        direct_oof=direct,
        residual_oof=residual,
        reliability_oof=reliability,
    )
    weight = gate.weights(reliability)
    prediction = gate.predict(
        direct=direct,
        residual=residual,
        reliability=reliability,
    )
    assert np.all((0.0 <= weight) & (weight <= 1.0))
    assert np.all(prediction >= np.minimum(direct, residual))
    assert np.all(prediction <= np.maximum(direct, residual))


def test_reliability_features_do_not_require_test_labels() -> None:
    reliability = reliability_features(
        direct=np.array([1.0, 2.0]),
        physics=np.array([0.5, 3.5]),
        residual=np.array([1.2, 1.8]),
        training_targets=np.array([0.0, 1.0, 2.0]),
        propagated_uncertainty=np.array([0.1, 0.2]),
    )
    assert list(reliability.columns) == ["R1", "R2", "R3"]
    assert np.all(reliability.to_numpy() >= 0.0)
