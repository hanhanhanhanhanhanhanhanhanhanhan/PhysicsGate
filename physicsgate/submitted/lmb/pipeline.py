"""LMB base pipeline (entry point).

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/tmp_lmb_fixed_group_no_rfe_random_search_multiseed.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = WORKDIR

from physicsgate.submitted.lmb import gate_vc as gate_base  # noqa: E402
from physicsgate.submitted.lmb import gate_tuned as runner  # noqa: E402
from physicsgate.submitted.lmb.common import CAPACITY_TARGET, DISCHARGE_ENERGY_TARGET, ENERGY_DENSITY_TARGET, FEATURE_COLUMNS, MASS_KG_COLUMN, NAME_COLUMN, TABLE_DIR, TARGET_COLUMNS, VOLTAGE_TARGET, build_energy_density_regressor, build_rf_regressor, selected_model_name  # noqa: E402
from physicsgate.submitted.lmb.data import DEFAULT_OUTPUT as DEFAULT_INPUT  # noqa: E402

DEFAULT_SEEDS = (
    4, 16, 42, 57, 99, 127, 178, 183, 188, 215,
    230, 244, 296, 338, 374, 403, 413, 451, 476, 483,
    488, 566, 603, 639, 717, 752, 769, 798, 809, 828,
    845, 909, 914, 978, 991, 1012, 1243, 1244, 1521, 1554,
    1676, 1693, 1703, 1809, 1847, 1928, 1969, 2024, 2255, 2772,
)

DEFAULT_ALPHAS = ""

DEFAULT_TAG = "tmp_lmb_fixed_group_name_no_rfe_random_search_50seed_20260719"

BASE_SEARCH_SCORING = "neg_root_mean_squared_error"

MODEL_SEED_OFFSETS = {
    ENERGY_DENSITY_TARGET: 0,
    VOLTAGE_TARGET: 31,
    CAPACITY_TARGET: 62,
    DISCHARGE_ENERGY_TARGET: 31,
}

VC_TARGETS = (ENERGY_DENSITY_TARGET, VOLTAGE_TARGET, CAPACITY_TARGET)

VC_AUXILIARIES = {
    ENERGY_DENSITY_TARGET: (VOLTAGE_TARGET, CAPACITY_TARGET),
    VOLTAGE_TARGET: (ENERGY_DENSITY_TARGET, CAPACITY_TARGET),
    CAPACITY_TARGET: (ENERGY_DENSITY_TARGET, VOLTAGE_TARGET),
}

DE_TARGETS = (ENERGY_DENSITY_TARGET, DISCHARGE_ENERGY_TARGET)

DE_AUXILIARIES = {
    ENERGY_DENSITY_TARGET: (DISCHARGE_ENERGY_TARGET,),
    DISCHARGE_ENERGY_TARGET: (ENERGY_DENSITY_TARGET,),
}

CURRENT_GROUPS: NDArray[np.object_] | None = None

ORIGINAL_GATE_TUNER = runner.tune_r2_r5_r6_gate


def _array(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def _group_splits(groups: NDArray[np.object_], n_splits: int) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    unique = np.unique(groups)
    n_use = min(n_splits, len(unique))
    if n_use < 2:
        raise ValueError("At least two training groups are required")
    splitter = GroupKFold(n_splits=n_use)
    return list(splitter.split(np.arange(len(groups)), groups=groups))


def group_oof_and_test(
    model: Any,
    X_train: pd.DataFrame,
    y_train: NDArray[np.float64],
    X_test: pd.DataFrame,
    groups: NDArray[np.object_],
    n_splits: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    oof = np.full(len(y_train), np.nan, dtype=float)
    for fit_idx, valid_idx in _group_splits(groups, n_splits):
        fold_model = clone(model)
        fold_model.fit(X_train.iloc[fit_idx], y_train[fit_idx])
        oof[valid_idx] = _array(fold_model.predict(X_train.iloc[valid_idx]))
    final = clone(model)
    final.fit(X_train, y_train)
    return oof, _array(final.predict(X_test))


def _serializable_params(params: dict[str, Any]) -> str:
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, BaseEstimator):
            cleaned[key] = value.__class__.__name__
        elif isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return json.dumps(cleaned, sort_keys=True)


def _base_search(
    *,
    target: str,
    X_train: pd.DataFrame,
    y_train: NDArray[np.float64],
    groups: NDArray[np.object_],
    args: argparse.Namespace,
    random_state: int,
    selection_rows: list[dict[str, Any]],
) -> BaseEstimator:
    if target == ENERGY_DENSITY_TARGET:
        estimator = build_energy_density_regressor(random_state=random_state, n_jobs=1)
        step = estimator.steps[-1][0]
        distributions: dict[str, list[Any]] = {
            f"{step}__n_estimators": [160, 240, 320, 450, 600, 800],
            f"{step}__max_depth": [2, 3, 4, 5, 6, 8],
            f"{step}__learning_rate": [0.015, 0.025, 0.04, 0.06, 0.08, 0.12],
            f"{step}__subsample": [0.65, 0.75, 0.85, 0.95, 1.0],
            f"{step}__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            f"{step}__min_child_weight": [1, 2, 4, 7, 12],
            f"{step}__reg_alpha": [0.0, 1e-4, 0.001, 0.01, 0.1, 1.0],
            f"{step}__reg_lambda": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        }
    else:
        estimator = build_rf_regressor(random_state=random_state, n_jobs=1)
        step = estimator.steps[-1][0]
        distributions = {
            f"{step}__n_estimators": [160, 240, 320, 450, 600, 800],
            f"{step}__max_depth": [None, 5, 8, 12, 18, 26],
            f"{step}__max_features": [1.0, 0.8, 0.6, 0.4, "sqrt", "log2"],
            f"{step}__min_samples_leaf": [1, 2, 3, 5, 8],
            f"{step}__min_samples_split": [2, 4, 7, 12, 20],
            f"{step}__bootstrap": [True, False],
        }
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=distributions,
        n_iter=args.base_search_iter,
        scoring=BASE_SEARCH_SCORING,
        cv=_group_splits(groups, args.tune_cv),
        refit=True,
        random_state=random_state,
        n_jobs=args.n_jobs,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups)
    for index, params in enumerate(search.cv_results_["params"]):
        selection_rows.append(
            {
                "role": f"{target}::base_model",
                "seed": int(args.random_state),
                "selection_stage": "base_random_search",
                "candidate_model": selected_model_name(estimator),
                "candidate_params_json": _serializable_params(params),
                "inner_group_cv_RMSE": float(-search.cv_results_["mean_test_score"][index]),
                "rank": int(search.cv_results_["rank_test_score"][index]),
                "selected": index == int(search.best_index_),
                "cv_mode": "GroupKFold",
                "group_key": NAME_COLUMN,
                "target": target,
                "output_tag": args.output_tag,
            }
        )
    best = clone(search.best_estimator_)
    n_jobs_key = f"{step}__n_jobs"
    if n_jobs_key in best.get_params(deep=True):
        best.set_params(**{n_jobs_key: args.n_jobs})
    return best


def _residual_search_space(random_state: int) -> list[dict[str, list[Any]]]:
    return [
        {
            "model": [Ridge()],
            "model__alpha": list(np.logspace(-8, 6, 29)),
        },
        {
            "model": [ElasticNet(max_iter=30000, random_state=random_state)],
            "model__alpha": list(np.logspace(-7, 3, 23)),
            "model__l1_ratio": [0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
            "model__selection": ["cyclic", "random"],
        },
        {
            "model": [RandomForestRegressor(random_state=random_state, n_jobs=1)],
            "model__n_estimators": [120, 200, 320, 500, 750],
            "model__max_depth": [None, 3, 5, 8, 12, 18],
            "model__max_features": [1.0, 0.75, 0.5, "sqrt"],
            "model__min_samples_leaf": [1, 2, 4, 7, 12],
        },
        {
            "model": [ExtraTreesRegressor(random_state=random_state, n_jobs=1)],
            "model__n_estimators": [120, 200, 320, 500, 750],
            "model__max_depth": [None, 3, 5, 8, 12, 18],
            "model__max_features": [1.0, 0.75, 0.5, "sqrt"],
            "model__min_samples_leaf": [1, 2, 4, 7, 12],
        },
        {
            "model": [HistGradientBoostingRegressor(random_state=random_state)],
            "model__max_iter": [100, 180, 280, 420],
            "model__learning_rate": [0.015, 0.03, 0.05, 0.08, 0.12],
            "model__max_leaf_nodes": [5, 9, 15, 25, 40],
            "model__min_samples_leaf": [5, 10, 20, 35, 60],
            "model__l2_regularization": [0.0, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0],
        },
    ]


def group_select_prediction(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    *,
    role: str,
    args: argparse.Namespace,
    offset: int,
    selection_rows: list[dict[str, Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, float]:
    if CURRENT_GROUPS is None:
        raise RuntimeError("CURRENT_GROUPS is not initialized")
    random_state = int(args.random_state + offset)
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=_residual_search_space(random_state),
        n_iter=args.residual_search_iter,
        scoring=BASE_SEARCH_SCORING,
        cv=_group_splits(CURRENT_GROUPS, args.tune_cv),
        refit=True,
        random_state=random_state,
        n_jobs=args.n_jobs,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=CURRENT_GROUPS)
    for index, params in enumerate(search.cv_results_["params"]):
        selection_rows.append(
            {
                "role": role,
                "candidate_model": params["model"].__class__.__name__,
                "candidate_params_json": _serializable_params(params),
                "search_level": "randomized_broad",
                "inner_oof_RMSE": float(-search.cv_results_["mean_test_score"][index]),
                "rank": int(search.cv_results_["rank_test_score"][index]),
                "selected": index == int(search.best_index_),
                "cv_mode": "GroupKFold",
                "group_key": NAME_COLUMN,
                "target": role.split("::", 1)[0],
                "output_tag": args.output_tag,
            }
        )
    best = clone(search.best_estimator_)
    model = best.named_steps["model"]
    if hasattr(model, "n_jobs"):
        best.set_params(model__n_jobs=args.n_jobs)
    train_frame = pd.DataFrame(np.asarray(X_train, dtype=float))
    test_frame = pd.DataFrame(np.asarray(X_test, dtype=float))
    best_oof, best_test = group_oof_and_test(
        best, train_frame, y_train, test_frame, CURRENT_GROUPS, args.inner_cv
    )
    best_name = best.named_steps["model"].__class__.__name__
    best_score = float(-search.best_score_)
    return best_oof, best_test, best_name, best_score


def group_gate_tuner(**kwargs: Any):
    if CURRENT_GROUPS is None:
        raise RuntimeError("CURRENT_GROUPS is not initialized")
    kwargs["groups"] = CURRENT_GROUPS
    return ORIGINAL_GATE_TUNER(**kwargs)


def de_physics_prediction(
    target: str,
    predictions: dict[str, NDArray[np.float64]],
    mass_kg: NDArray[np.float64],
) -> NDArray[np.float64]:
    mass = np.maximum(_array(mass_kg), 1e-12)
    if target == ENERGY_DENSITY_TARGET:
        return _array(predictions[DISCHARGE_ENERGY_TARGET]) / mass
    if target == DISCHARGE_ENERGY_TARGET:
        return _array(predictions[ENERGY_DENSITY_TARGET]) * mass
    raise ValueError(target)


def de_propagated_sigma(
    target: str,
    predictions: dict[str, NDArray[np.float64]],
    mass_kg: NDArray[np.float64],
    sigmas: dict[str, float],
) -> NDArray[np.float64]:
    del predictions
    mass = np.maximum(_array(mass_kg), 1e-12)
    if target == ENERGY_DENSITY_TARGET:
        return np.full_like(mass, sigmas[DISCHARGE_ENERGY_TARGET]) / mass
    if target == DISCHARGE_ENERGY_TARGET:
        return np.full_like(mass, sigmas[ENERGY_DENSITY_TARGET]) * mass
    raise ValueError(target)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def summarize(detailed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["formula_family", "target", "method"]
    for key, group in detailed.groupby(keys, sort=False):
        rows.append(
            {
                "formula_family": key[0],
                "target": key[1],
                "method": key[2],
                "n_seeds": int(group["seed"].nunique()),
                "RMSE_mean": float(group["RMSE"].mean()),
                "RMSE_std": float(group["RMSE"].std(ddof=1)),
                "MAE_mean": float(group["MAE"].mean()),
                "MAE_std": float(group["MAE"].std(ddof=1)),
                "R2_mean": float(group["R2"].mean()),
                "R2_std": float(group["R2"].std(ddof=1)),
                "percent_RMSE_change_vs_direct_mean": float(group["percent_RMSE_change_vs_direct"].mean()),
                "percent_RMSE_change_vs_direct_std": float(group["percent_RMSE_change_vs_direct"].std(ddof=1)),
                "win_count": int((group["percent_RMSE_change_vs_direct"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def paths(tag: str) -> dict[str, Path]:
    return {
        name: TABLE_DIR / f"{tag}_{name}.csv"
        for name in ("detailed", "summary", "selection", "audit", "validity")
    }


def checkpoint(
    outputs: dict[str, Path],
    detailed: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    validity: list[dict[str, Any]],
) -> None:
    detail_frame = pd.DataFrame(detailed)
    atomic_csv(detail_frame, outputs["detailed"])
    atomic_csv(summarize(detail_frame), outputs["summary"])
    atomic_csv(pd.DataFrame(selection), outputs["selection"])
    atomic_csv(pd.DataFrame(audit), outputs["audit"])
    atomic_csv(pd.DataFrame(validity), outputs["validity"])


def load_checkpoint(outputs: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    def records(name: str) -> list[dict[str, Any]]:
        path = outputs[name]
        return pd.read_csv(path).to_dict("records") if path.exists() else []
    return records("detailed"), records("selection"), records("audit"), records("validity")


def configure_route(
    targets: tuple[str, ...],
    auxiliaries: dict[str, tuple[str, ...]],
    family: str,
    description: str,
    physics: Callable[..., NDArray[np.float64]],
    sigma: Callable[..., NDArray[np.float64]],
) -> None:
    runner.FORMULA_TARGETS = targets
    runner.FORMULA_AUXILIARIES = auxiliaries
    runner.FORMULA_FAMILY = family
    runner.FORMULA_DESCRIPTION = description
    runner.physics_prediction = physics
    runner.propagated_sigma = sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--gate-inner-cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--base-search-iter", type=int, default=16)
    parser.add_argument("--residual-search-iter", type=int, default=28)
    parser.add_argument("--tune-cv", type=int, default=3)
    parser.add_argument("--gate-alpha-random-iter", type=int, default=40)
    parser.add_argument("--gate-alpha-seed", type=int, default=20260719)
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS)
    parser.add_argument("--output-tag", default=DEFAULT_TAG)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    global CURRENT_GROUPS
    args = parse_args()
    if args.alphas.strip():
        args.alpha_grid = [float(value) for value in args.alphas.split(",") if value.strip()]
    else:
        alpha_rng = np.random.default_rng(args.gate_alpha_seed)
        random_alphas = np.power(
            10.0, alpha_rng.uniform(-8.0, 8.0, size=args.gate_alpha_random_iter)
        )
        anchor_alphas = np.logspace(-8.0, 8.0, 33)
        args.alpha_grid = sorted(np.unique(np.concatenate([random_alphas, anchor_alphas])).tolist())
    args.feature_mode = "no_feature_selection"
    args.gate_mode = "fixed"
    args.output_tag = str(args.output_tag)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = paths(args.output_tag)
    if args.no_resume:
        detailed_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        validity_rows: list[dict[str, Any]] = []
    else:
        detailed_rows, selection_rows, audit_rows, validity_rows = load_checkpoint(outputs)
    completed = {int(row["seed"]) for row in validity_rows if bool(row.get("complete", False))}

    data = pd.read_csv(args.input).dropna(
        subset=[*TARGET_COLUMNS, MASS_KG_COLUMN, NAME_COLUMN]
    ).reset_index(drop=True)
    all_groups = data[NAME_COLUMN].astype(str).to_numpy(dtype=object)

    gate_base.select_prediction = group_select_prediction
    runner.tune_r2_r5_r6_gate = group_gate_tuner

    for seed in args.seeds:
        if int(seed) in completed:
            print(f"[seed {seed}] already complete; skipping", flush=True)
            continue
        started = time.time()
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(data, groups=all_groups))
        train = data.iloc[train_idx].reset_index(drop=True)
        test = data.iloc[test_idx].reset_index(drop=True)
        train_groups = train[NAME_COLUMN].astype(str).to_numpy(dtype=object)
        test_groups = test[NAME_COLUMN].astype(str).to_numpy(dtype=object)
        overlap = sorted(set(train_groups).intersection(test_groups))
        if overlap:
            raise RuntimeError(f"Group overlap for seed {seed}: {overlap}")
        CURRENT_GROUPS = train_groups
        args.random_state = int(seed)

        seed_selection: list[dict[str, Any]] = []
        base_oof: dict[str, NDArray[np.float64]] = {}
        base_test: dict[str, NDArray[np.float64]] = {}
        model_names: dict[str, str] = {}
        feature_counts: dict[str, int] = {}
        for target in TARGET_COLUMNS:
            model_seed = int(seed + MODEL_SEED_OFFSETS[target])
            model = _base_search(
                target=target,
                X_train=train.loc[:, FEATURE_COLUMNS],
                y_train=train[target].to_numpy(dtype=float),
                groups=train_groups,
                args=args,
                random_state=model_seed,
                selection_rows=seed_selection,
            )
            oof, holdout = group_oof_and_test(
                model,
                train.loc[:, FEATURE_COLUMNS],
                train[target].to_numpy(dtype=float),
                test.loc[:, FEATURE_COLUMNS],
                train_groups,
                args.inner_cv,
            )
            base_oof[target] = oof
            base_test[target] = holdout
            model_names[target] = f"{selected_model_name(model)}_RandomizedSearchCV"
            feature_counts[target] = len(FEATURE_COLUMNS)

        seed_detail: list[dict[str, Any]] = []
        seed_audit: list[dict[str, Any]] = []
        route_configs = (
            (
                VC_TARGETS,
                VC_AUXILIARIES,
                "VC_mass_only_group_name_no_rfe",
                "ED=capacity*voltage/mass",
                runner.base.vc_mass_physics_prediction,
                runner.base.propagated_sigma_vc_mass,
            ),
            (
                DE_TARGETS,
                DE_AUXILIARIES,
                "DE_ED_mass_only_group_name_no_rfe",
                "DE=ED*mass; ED=DE/mass",
                de_physics_prediction,
                de_propagated_sigma,
            ),
        )
        for targets, auxiliaries, family, description, physics, sigma in route_configs:
            configure_route(targets, auxiliaries, family, description, physics, sigma)
            for target in targets:
                rows, selection, audit = runner.evaluate_target(
                    seed=int(seed),
                    target=target,
                    train=train,
                    test=test,
                    base_oof=base_oof,
                    base_test=base_test,
                    model_names=model_names,
                    feature_counts=feature_counts,
                    args=args,
                    alpha_grid=args.alpha_grid,
                )
                for selection_row in selection:
                    selection_row["formula_family"] = family
                for row in rows:
                    row.update(
                        {
                            "outer_split": "GroupShuffleSplit",
                            "inner_oof": "GroupKFold",
                            "group_key": NAME_COLUMN,
                            "feature_mode": "all_41_features_no_rfe",
                            "train_test_group_overlap_count": 0,
                            "base_tuning": "RandomizedSearchCV_GroupKFold_train_only",
                            "residual_tuning": "RandomizedSearchCV_GroupKFold_train_only_broad",
                            "gate_tuning": "broad_randomized_alpha_GroupKFold_train_only",
                        }
                    )
                audit.update(
                    {
                        "outer_split": "GroupShuffleSplit",
                        "inner_oof": "GroupKFold",
                        "group_key": NAME_COLUMN,
                        "feature_mode": "all_41_features_no_rfe",
                        "n_train_groups": int(len(np.unique(train_groups))),
                        "n_test_groups": int(len(np.unique(test_groups))),
                        "train_test_group_overlap_count": 0,
                        "base_search_iter": int(args.base_search_iter),
                        "residual_search_iter": int(args.residual_search_iter),
                        "tune_cv": int(args.tune_cv),
                        "gate_alpha_candidate_count": int(len(args.alpha_grid)),
                        "hyperparameter_selection_scope": "outer_train_only_group_cv",
                    }
                )
                seed_detail.extend(rows)
                seed_selection.extend(selection)
                seed_audit.append(audit)

        detailed_rows.extend(seed_detail)
        selection_rows.extend(seed_selection)
        audit_rows.extend(seed_audit)
        validity_rows.append(
            {
                "seed": int(seed),
                "complete": True,
                "group_key": NAME_COLUMN,
                "outer_split": "GroupShuffleSplit",
                "test_size": float(args.test_size),
                "n_total_rows": int(len(data)),
                "n_train_rows": int(len(train)),
                "n_test_rows": int(len(test)),
                "n_total_groups": int(len(np.unique(all_groups))),
                "n_train_groups": int(len(np.unique(train_groups))),
                "n_test_groups": int(len(np.unique(test_groups))),
                "train_test_group_overlap_count": 0,
                "train_groups_json": json.dumps(sorted(np.unique(train_groups).tolist())),
                "test_groups_json": json.dumps(sorted(np.unique(test_groups).tolist())),
                "feature_mode": "all_41_features_no_rfe",
                "feature_count": int(len(FEATURE_COLUMNS)),
                "inner_oof": "GroupKFold",
                "base_tuning": "RandomizedSearchCV_GroupKFold_train_only",
                "base_search_iter": int(args.base_search_iter),
                "residual_tuning": "RandomizedSearchCV_GroupKFold_train_only_broad",
                "residual_search_iter": int(args.residual_search_iter),
                "tune_cv": int(args.tune_cv),
                "gate_tuning": "broad_randomized_alpha_GroupKFold_train_only",
                "gate_alpha_candidate_count": int(len(args.alpha_grid)),
                "hyperparameter_selection_scope": "outer_train_only_group_cv",
                "elapsed_seconds": float(time.time() - started),
            }
        )
        checkpoint(outputs, detailed_rows, selection_rows, audit_rows, validity_rows)
        print(f"[seed {seed}] complete in {time.time() - started:.1f}s", flush=True)

    print(summarize(pd.DataFrame(detailed_rows)).to_string(index=False))
    print(f"Wrote {outputs['summary']}")


if __name__ == "__main__":
    main()
