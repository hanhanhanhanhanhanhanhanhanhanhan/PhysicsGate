import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from physicsgate.oof import fit_predict_oof, generate_oof_predictions, refit_and_predict_test


def test_fit_predict_oof_returns_one_prediction_per_training_row():
    X = np.arange(30, dtype=float).reshape(-1, 1)
    y = np.square(X[:, 0])

    oof = fit_predict_oof(
        DecisionTreeRegressor(random_state=0),
        X,
        y,
        cv=5,
        random_state=0,
    )

    assert oof.shape == y.shape
    assert np.isfinite(oof).all()
    assert not np.allclose(oof, y)


def test_generate_oof_predictions_rejects_non_oof_fold_coverage():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.arange(6, dtype=float)
    folds = [(np.array([0, 1, 2]), np.array([3, 4]))]

    with pytest.raises(ValueError, match="exactly one OOF prediction"):
        generate_oof_predictions(Ridge(), X, y, folds=folds)


def test_refit_and_predict_test_uses_training_data_for_final_model():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = 3.0 * X[:, 0] - 2.0

    prediction = refit_and_predict_test(Ridge(alpha=0.0), X, y, [[10.0]])

    assert prediction.shape == (1,)
    assert prediction[0] == pytest.approx(28.0)
