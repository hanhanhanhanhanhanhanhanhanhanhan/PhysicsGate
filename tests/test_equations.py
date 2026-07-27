import inspect

import numpy as np
import pytest

from physicsgate.equations import (
    LinearEquationConfig,
    compute_physics_prediction,
    compute_physics_residual,
)


def test_linear_physics_prediction_uses_predicted_auxiliaries():
    config = LinearEquationConfig(
        name="toy_linear",
        target="A",
        auxiliary_targets=("B", "C"),
        coefficients={"B": 1.7, "C": -0.8},
    )

    prediction = compute_physics_prediction(
        config,
        auxiliary_predictions={
            "B": np.array([1.0, 2.0]),
            "C": np.array([3.0, -1.0]),
        },
    )

    np.testing.assert_allclose(prediction, [-0.7, 4.2])


def test_linear_physics_prediction_rejects_missing_auxiliary_prediction():
    config = LinearEquationConfig(
        name="toy_linear",
        target="A",
        auxiliary_targets=("B", "C"),
        coefficients={"B": 1.7, "C": -0.8},
    )

    with pytest.raises(ValueError, match="Missing auxiliary predictions"):
        compute_physics_prediction(config, auxiliary_predictions={"B": [1.0]})


def test_physics_api_does_not_accept_true_test_auxiliary_targets():
    signature = inspect.signature(compute_physics_prediction)

    assert "auxiliary_predictions" in signature.parameters
    assert "test_auxiliary_targets" not in signature.parameters
    assert "auxiliary_targets" not in signature.parameters


def test_compute_physics_residual_returns_signed_difference():
    residual = compute_physics_residual([2.0, 3.0], [1.5, 3.5])

    np.testing.assert_allclose(residual, [0.5, -0.5])
