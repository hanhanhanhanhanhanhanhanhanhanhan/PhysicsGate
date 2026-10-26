"""LMB base-model selection and branch helpers.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/tmp_lmb_seed1693_fixed_ridge_five_method_check.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = WORKDIR

from physicsgate.submitted.lmb import pipeline as fixed  # noqa: E402
from physicsgate.submitted.lmb.common import CAPACITY_TARGET, DISCHARGE_ENERGY_TARGET, ENERGY_DENSITY_TARGET, FEATURE_COLUMNS, VOLTAGE_TARGET, build_energy_density_regressor, build_rf_regressor, metrics_dict  # noqa: E402
from physicsgate.submitted.lmb.clean_methods import _candidate_factories  # noqa: E402

DISPLAY = {
    ENERGY_DENSITY_TARGET: "Energy Density(Wh/kg)",
    VOLTAGE_TARGET: "Normal Discharge Voltage(V)",
    CAPACITY_TARGET: "Discharge capacity(Ah)",
    DISCHARGE_ENERGY_TARGET: "Discharge Energy(Wh)",
}

TARGET_ORDER = (
    ENERGY_DENSITY_TARGET,
    VOLTAGE_TARGET,
    CAPACITY_TARGET,
    DISCHARGE_ENERGY_TARGET,
)

EVALUATIONS = (
    ("VC", ENERGY_DENSITY_TARGET),
    ("VC", VOLTAGE_TARGET),
    ("VC", CAPACITY_TARGET),
    ("DE", ENERGY_DENSITY_TARGET),
    ("DE", DISCHARGE_ENERGY_TARGET),
)

SHORT_TARGET = {
    ENERGY_DENSITY_TARGET: "ED",
    VOLTAGE_TARGET: "V",
    CAPACITY_TARGET: "C",
    DISCHARGE_ENERGY_TARGET: "DE",
}

TARGET_OFFSETS = {
    ENERGY_DENSITY_TARGET: 0,
    VOLTAGE_TARGET: 31,
    CAPACITY_TARGET: 62,
    DISCHARGE_ENERGY_TARGET: 31,
}


def arr(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def route(route_name: str, target: str) -> tuple[str, str, tuple[str, ...], Any]:
    if route_name == "VC":
        auxiliaries = {
            ENERGY_DENSITY_TARGET: (VOLTAGE_TARGET, CAPACITY_TARGET),
            VOLTAGE_TARGET: (ENERGY_DENSITY_TARGET, CAPACITY_TARGET),
            CAPACITY_TARGET: (ENERGY_DENSITY_TARGET, VOLTAGE_TARGET),
        }
        return (
            "VC_mass_only_group_name_no_rfe",
            "VC_mass_only_group_name_no_rfe:",
            auxiliaries[target],
            fixed.runner.base.vc_mass_physics_prediction,
        )
    if route_name != "DE" or target not in (ENERGY_DENSITY_TARGET, DISCHARGE_ENERGY_TARGET):
        raise ValueError(f"Unsupported route-target: {route_name}/{target}")
    auxiliaries = {
        ENERGY_DENSITY_TARGET: (DISCHARGE_ENERGY_TARGET,),
        DISCHARGE_ENERGY_TARGET: (ENERGY_DENSITY_TARGET,),
    }
    return (
        "DE_ED_mass_only_group_name_no_rfe",
        "DE_ED_mass_only_group_name_no_rfe:",
        auxiliaries[target],
        fixed.de_physics_prediction,
    )


def selected_rows(selection: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    rows = selection[mask & selection["selected"].astype(str).str.lower().eq("true")]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one selected {label} row, found {len(rows)}")
    return rows


def selected_base_model(
    target: str,
    selection: pd.DataFrame,
    args: argparse.Namespace,
) -> Any:
    row = selected_rows(
        selection,
        selection["seed"].eq(args.random_state)
        & selection["selection_stage"].eq("base_random_search")
        & selection["target"].eq(target),
        f"base model for {target}",
    ).iloc[0]
    model_seed = args.random_state + TARGET_OFFSETS[target]
    model = (
        build_energy_density_regressor(random_state=model_seed, n_jobs=args.n_jobs)
        if target == ENERGY_DENSITY_TARGET
        else build_rf_regressor(random_state=model_seed, n_jobs=args.n_jobs)
    )
    model.set_params(**json.loads(row["candidate_params_json"]))
    return model


def selected_residual_model(
    target: str,
    family: str,
    target_index: int,
    selection: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[Pipeline, str]:
    role = f"{target}::FixedGate_Ares_old_h"
    row = selected_rows(
        selection,
        selection["seed"].eq(args.random_state)
        & selection["selection_stage"].eq("Ares_old_h")
        & selection["role"].eq(role)
        & selection["formula_family"].eq(family),
        f"residual model for {target}/{family}",
    ).iloc[0]
    params = json.loads(row["candidate_params_json"])
    model_name = params.pop("model")
    random_state = args.random_state + 1001 + target_index * 37
    if model_name == "Ridge":
        estimator: Any = Ridge()
    elif model_name == "ElasticNet":
        estimator = ElasticNet(max_iter=30000, random_state=random_state)
    elif model_name == "RandomForestRegressor":
        estimator = RandomForestRegressor(random_state=random_state, n_jobs=args.n_jobs)
    elif model_name == "ExtraTreesRegressor":
        estimator = ExtraTreesRegressor(random_state=random_state, n_jobs=args.n_jobs)
    elif model_name == "HistGradientBoostingRegressor":
        estimator = HistGradientBoostingRegressor(random_state=random_state)
    else:
        raise ValueError(f"Unsupported retained residual model: {model_name}")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    pipeline.set_params(**params)
    return pipeline, model_name


def ordinary_prediction(
    *,
    target: str,
    X_train: pd.DataFrame,
    y_train: NDArray[np.float64],
    X_test: pd.DataFrame,
    groups: NDArray[np.object_],
    args: argparse.Namespace,
    selection_rows: list[dict[str, Any]],
) -> tuple[NDArray[np.float64], str]:
    factories = _candidate_factories(
        args.random_state + 400 + TARGET_ORDER.index(target) * 29,
        args.n_jobs,
        search_level="standard",
    )
    train_columns: list[NDArray[np.float64]] = []
    test_columns: list[NDArray[np.float64]] = []
    names: list[str] = []
    for name, factory in factories.items():
        oof, test = fixed.group_oof_and_test(
            factory(), X_train, y_train, X_test, groups, args.inner_cv
        )
        train_columns.append(oof)
        test_columns.append(test)
        names.append(name)
    _, test, meta_name, _ = fixed.group_select_prediction(
        np.column_stack(train_columns),
        y_train,
        np.column_stack(test_columns),
        role=f"{target}::Ordinary_stacking_meta",
        args=args,
        offset=481,
        selection_rows=selection_rows,
    )
    return test, f"bases={';'.join(names)};g={meta_name}"


def metric_row(
    *,
    target: str,
    method: str,
    prediction: NDArray[np.float64],
    y_test: NDArray[np.float64],
    selected_model: str,
    family: str,
    route_target: str,
    args: argparse.Namespace,
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": "LMB",
        "target": DISPLAY[target],
        "raw_target_column": target,
        "seed": args.random_state,
        "method": method,
        "formula_family": family,
        "route_target": route_target,
        "split_mode": "name_group_disjoint_90_10",
        "test_size": args.test_size,
        "feature_mode": "all_41_features_no_rfe",
        "feature_count": len(FEATURE_COLUMNS),
        "selected_model": selected_model,
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "uses_true_test_auxiliary_values": False,
        "uses_oof_training_predictions": True,
        "output_tag": args.output_tag,
        "source": "seed1693_retained_train_selected_configs_recomputed",
    }
    row.update(metrics_dict(y_test, prediction))
    return row
