"""Candidate model families and inner-CV model selection.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/physstack/model_selection.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DEFAULT_MODEL_POOL = (
    "ridge",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "knn",
    "svr",
)

SCALED_MODELS = {"ridge", "elasticnet", "svr", "knn"}

DEFAULT_SCORING = "neg_root_mean_squared_error"


@dataclass(frozen=True)
class ModelSelectionResult:
    """Fitted estimator and auditable model-selection metadata."""

    estimator: BaseEstimator
    selected_model: str
    selected_params_json: str
    best_score: float | None = None


def _as_2d(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    return array


def _as_1d(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _metadata_json(**metadata: Any) -> str:
    return json.dumps(_safe_json(metadata), sort_keys=True)


def _split_count(groups: ArrayLike, inner_cv: int) -> int:
    group_array = np.asarray(groups)
    unique_groups = np.unique(group_array)
    split_count = min(int(inner_cv), len(unique_groups))
    if split_count < 2:
        raise ValueError("At least two groups are required for GroupKFold tuning")
    return split_count


def _pool_iter_budget(n_iter: int, model_count: int) -> int:
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    return max(1, int(n_iter) // max(1, model_count))


def _feature_selection_values(values: Sequence[int], n_features: int | None) -> list[int]:
    if n_features is None:
        return list(values)
    return [value for value in values if int(value) <= int(n_features)]


def get_feature_selector(strategy: str, random_state: int) -> BaseEstimator | None:
    """Return an unfitted feature selector for a pipeline."""
    if strategy == "none":
        return None
    if strategy == "select_k_best":
        return SelectKBest(score_func=f_regression, k="all")
    if strategy == "model_based":
        return SelectFromModel(
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                max_iter=100000,
                tol=1e-3,
                random_state=random_state,
                selection="cyclic",
            ),
            threshold="median",
        )
    if strategy == "rfe":
        return RFE(estimator=Ridge(alpha=1.0), n_features_to_select=0.5)
    raise ValueError(
        "feature_selection must be one of none, select_k_best, model_based, or rfe"
    )


def build_pipeline(
    model_name: str,
    model: BaseEstimator,
    feature_selection: str = "none",
    random_state: int = 42,
) -> BaseEstimator:
    """Build a leakage-safe preprocessing/selection/model pipeline."""
    selector = get_feature_selector(feature_selection, random_state)
    needs_scaling = model_name in SCALED_MODELS or feature_selection == "model_based"
    if selector is None and not needs_scaling:
        return model

    steps: list[tuple[str, BaseEstimator]] = []
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("model", model))
    return Pipeline(steps)


def get_feature_selection_search_space(
    feature_selection: str,
    n_features: int | None,
) -> Mapping[str, Any]:
    """Return selector hyperparameters, with invalid feature counts removed."""
    if feature_selection == "none":
        return {}
    if feature_selection == "select_k_best":
        values = _feature_selection_values([5, 8, 12], n_features)
        return {"selector__k": [*values, "all"]}
    if feature_selection == "model_based":
        return {"selector__threshold": ["median", "mean"]}
    if feature_selection == "rfe":
        values = _feature_selection_values([5, 8, 12], n_features)
        return {"selector__n_features_to_select": [*values, 0.5]}
    raise ValueError(
        "feature_selection must be one of none, select_k_best, model_based, or rfe"
    )


def get_fixed_model(
    model_name: str,
    random_state: int,
    feature_selection: str = "none",
) -> BaseEstimator:
    """Return a deterministic sklearn estimator for fixed baselines."""
    if model_name == "ridge":
        return build_pipeline(model_name, Ridge(alpha=1.0), feature_selection, random_state)
    if model_name == "elasticnet":
        return build_pipeline(
            model_name,
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                max_iter=100000,
                tol=1e-3,
                random_state=random_state,
                selection="cyclic",
            ),
            feature_selection,
            random_state,
        )
    if model_name == "random_forest":
        return build_pipeline(
            model_name,
            RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            ),
            feature_selection,
            random_state,
        )
    if model_name == "extra_trees":
        return build_pipeline(
            model_name,
            ExtraTreesRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            ),
            feature_selection,
            random_state,
        )
    if model_name == "gradient_boosting":
        return build_pipeline(
            model_name,
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=2,
                random_state=random_state,
            ),
            feature_selection,
            random_state,
        )
    if model_name == "knn":
        return build_pipeline(
            model_name,
            KNeighborsRegressor(n_neighbors=5),
            feature_selection,
            random_state,
        )
    if model_name == "svr":
        return build_pipeline(
            model_name,
            SVR(C=1.0, epsilon=0.1, kernel="rbf"),
            feature_selection,
            random_state,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def get_tunable_model(
    model_name: str,
    random_state: int,
    feature_selection: str = "none",
) -> BaseEstimator:
    """Return the base estimator used by RandomizedSearchCV."""
    if model_name == "ridge":
        return build_pipeline(model_name, Ridge(), feature_selection, random_state)
    if model_name == "elasticnet":
        return build_pipeline(
            model_name,
            ElasticNet(
                max_iter=100000,
                tol=1e-3,
                random_state=random_state,
                selection="cyclic",
            ),
            feature_selection,
            random_state,
        )
    if model_name == "random_forest":
        return build_pipeline(
            model_name,
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            feature_selection,
            random_state,
        )
    if model_name == "extra_trees":
        return build_pipeline(
            model_name,
            ExtraTreesRegressor(random_state=random_state, n_jobs=1),
            feature_selection,
            random_state,
        )
    if model_name == "gradient_boosting":
        return build_pipeline(
            model_name,
            GradientBoostingRegressor(random_state=random_state),
            feature_selection,
            random_state,
        )
    if model_name == "knn":
        return build_pipeline(model_name, KNeighborsRegressor(), feature_selection, random_state)
    if model_name == "svr":
        return build_pipeline(model_name, SVR(), feature_selection, random_state)
    raise ValueError(f"Unknown model_name: {model_name}")


def get_search_space(
    model_name: str,
    feature_selection: str = "none",
    n_features: int | None = None,
) -> Mapping[str, Any]:
    """Return transparent sklearn-only hyperparameter distributions."""
    prefix = "model__" if model_name in SCALED_MODELS or feature_selection != "none" else ""
    if model_name == "ridge":
        space: dict[str, Any] = {f"{prefix}alpha": loguniform(1e-4, 1e3)}
    elif model_name == "elasticnet":
        space = {
            f"{prefix}alpha": loguniform(1e-3, 1e2),
            f"{prefix}l1_ratio": uniform(0.05, 0.90),
        }
    elif model_name in {"random_forest", "extra_trees"}:
        space = {
            f"{prefix}n_estimators": [200, 400, 800],
            f"{prefix}max_depth": [None, 3, 5, 8, 12],
            f"{prefix}min_samples_leaf": [1, 2, 4, 8],
            f"{prefix}max_features": ["sqrt", "log2", 0.5, 1.0],
        }
    elif model_name == "gradient_boosting":
        space = {
            f"{prefix}n_estimators": [100, 200, 400],
            f"{prefix}learning_rate": [0.01, 0.03, 0.05, 0.1],
            f"{prefix}max_depth": [2, 3, 4],
            f"{prefix}min_samples_leaf": [1, 2, 4],
        }
    elif model_name == "knn":
        space = {
            f"{prefix}n_neighbors": [3, 5, 7, 11, 15],
            f"{prefix}weights": ["uniform", "distance"],
            f"{prefix}p": [1, 2],
        }
    elif model_name == "svr":
        space = {
            f"{prefix}C": loguniform(1e-2, 1e3),
            f"{prefix}epsilon": loguniform(1e-3, 1.0),
            f"{prefix}gamma": ["scale", "auto"],
            f"{prefix}kernel": ["rbf"],
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return {**space, **get_feature_selection_search_space(feature_selection, n_features)}


def tune_model(
    X: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    model_name: str,
    n_iter: int = 10,
    inner_cv: int = 5,
    scoring: str = DEFAULT_SCORING,
    random_state: int = 42,
    n_jobs: int = 1,
    feature_selection: str = "none",
) -> ModelSelectionResult:
    """Tune one sklearn model with GroupKFold on outer-training data only."""
    X_array = _as_2d(X, name="X")
    y_array = _as_1d(y, name="y")
    group_array = np.asarray(groups)
    if not (len(X_array) == len(y_array) == len(group_array)):
        raise ValueError("X, y, and groups must have the same length")
    split_count = _split_count(group_array, inner_cv)
    search = RandomizedSearchCV(
        estimator=get_tunable_model(model_name, random_state, feature_selection),
        param_distributions=get_search_space(
            model_name,
            feature_selection,
            n_features=X_array.shape[1],
        ),
        n_iter=int(n_iter),
        scoring=scoring,
        cv=GroupKFold(n_splits=split_count),
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_array, y_array, groups=group_array)
    metadata = {
        "strategy": "tuned_sklearn",
        "model_name": model_name,
        "n_iter": int(n_iter),
        "inner_cv": split_count,
        "scoring": scoring,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "safe_two_stage_oof": True,
        "feature_selection": feature_selection,
    }
    return ModelSelectionResult(
        estimator=search.best_estimator_,
        selected_model=f"{model_name}_tuned_sklearn",
        selected_params_json=_metadata_json(**metadata),
        best_score=float(search.best_score_),
    )


def tune_model_pool(
    X: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    model_names: Sequence[str] | None = None,
    n_iter: int = 10,
    inner_cv: int = 5,
    scoring: str = DEFAULT_SCORING,
    random_state: int = 42,
    n_jobs: int = 1,
    feature_selection: str = "none",
) -> ModelSelectionResult:
    """Tune a pool of sklearn models and return the highest-CV-score model.

    ``n_iter`` is treated as a total pool budget and is split across models to
    keep quick benchmark runs practical.
    """
    names = tuple(model_names or DEFAULT_MODEL_POOL)
    per_model_iter = _pool_iter_budget(int(n_iter), len(names))
    results = [
        tune_model(
            X,
            y,
            groups,
            model_name=name,
            n_iter=per_model_iter,
            inner_cv=inner_cv,
            scoring=scoring,
            random_state=random_state + index * 101,
            n_jobs=n_jobs,
            feature_selection=feature_selection,
        )
        for index, name in enumerate(names)
    ]
    best = max(results, key=lambda result: float(result.best_score))
    pool_metadata = {
        "strategy": "tuned_sklearn",
        "pool": list(names),
        "pool_n_iter": int(n_iter),
        "per_model_n_iter": per_model_iter,
        "selected_model": best.selected_model,
        "selected_params": json.loads(best.selected_params_json),
        "safe_two_stage_oof": True,
        "feature_selection": feature_selection,
    }
    return ModelSelectionResult(
        estimator=best.estimator,
        selected_model=best.selected_model,
        selected_params_json=_metadata_json(**pool_metadata),
        best_score=best.best_score,
    )


def fit_model_by_strategy(
    X: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    model_strategy: str,
    model_names: Sequence[str] | None = None,
    n_iter: int = 10,
    inner_cv: int = 5,
    scoring: str = DEFAULT_SCORING,
    random_state: int = 42,
    n_jobs: int = 1,
    feature_selection: str = "none",
) -> ModelSelectionResult:
    """Fit a fixed model or leakage-safe tuned sklearn model on train data."""
    X_array = _as_2d(X, name="X")
    y_array = _as_1d(y, name="y")
    group_array = np.asarray(groups)
    if not (len(X_array) == len(y_array) == len(group_array)):
        raise ValueError("X, y, and groups must have the same length")
    names = tuple(model_names or ("ridge",))
    if model_strategy == "fixed":
        model_name = names[0]
        estimator = clone(
            get_fixed_model(model_name, random_state, feature_selection)
        ).fit(X_array, y_array)
        return ModelSelectionResult(
            estimator=estimator,
            selected_model=f"{model_name}_fixed",
            selected_params_json=_metadata_json(
                strategy="fixed",
                model_name=model_name,
                random_state=random_state,
                feature_selection=feature_selection,
            ),
        )
    if model_strategy == "tuned_sklearn":
        return tune_model_pool(
            X_array,
            y_array,
            group_array,
            model_names=names if model_names is not None else DEFAULT_MODEL_POOL,
            n_iter=n_iter,
            inner_cv=inner_cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            feature_selection=feature_selection,
        )
    raise ValueError("model_strategy must be 'fixed' or 'tuned_sklearn'")


def extract_selected_feature_names(
    estimator: BaseEstimator,
    feature_names: Sequence[str],
) -> list[str]:
    """Extract selected feature names from a fitted selector pipeline if present."""
    if not isinstance(estimator, Pipeline) or "selector" not in estimator.named_steps:
        return []
    selector = estimator.named_steps["selector"]
    if not hasattr(selector, "get_support"):
        return []
    try:
        support = np.asarray(selector.get_support(), dtype=bool)
    except Exception:
        return []
    names = list(feature_names)
    if support.shape[0] != len(names):
        return []
    return [name for name, keep in zip(names, support) if keep]
