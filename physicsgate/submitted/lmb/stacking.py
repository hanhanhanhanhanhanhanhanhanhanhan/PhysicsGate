"""LMB Ordinary stacking and PhysResStack.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/tmp_lmb_fixed_ridge_top10_six_methods.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupShuffleSplit

REPO = WORKDIR

from physicsgate.submitted.lmb import pipeline as fixed  # noqa: E402
from physicsgate.submitted.lmb import five_methods as check  # noqa: E402
from physicsgate.submitted.lmb.common import FEATURE_COLUMNS, MASS_KG_COLUMN, NAME_COLUMN, TABLE_DIR, TARGET_COLUMNS, selected_model_name  # noqa: E402

REFERENCE_TAG = "tmp_lmb_fixed_group_name_no_rfe_random_search_50seed_20260719"

MISSING5 = (798, 1243, 1244, 1012, 178)

METHODS = (
    "Direct",
    "Physics-only",
    "Ares_old",
    "Ordinary stacking",
    "PhysResStack",
    "FixedGate_R2_R5_R6",
)


def route_target(frame: pd.DataFrame) -> pd.Series:
    route = np.where(
        frame["formula_family"].astype(str).str.startswith("VC_mass_only"),
        "VC",
        "DE",
    )
    code = frame["target"].map(
        {
            "Energy Density(Wh/kg)": "ED",
            "Normal Discharge Voltage(V)": "V",
            "Discharge capacity(Ah)": "C",
            "Discharge Energy(Wh)": "DE",
        }
    )
    return pd.Series(route, index=frame.index) + "-" + code.astype(str)


def compute_missing(args: argparse.Namespace) -> pd.DataFrame:
    retained_selection = pd.read_csv(
        TABLE_DIR / f"{REFERENCE_TAG}_selection.csv"
    )
    retained_detailed = pd.read_csv(
        TABLE_DIR / f"{REFERENCE_TAG}_detailed.csv"
    )
    data = pd.read_csv(args.input).dropna(
        subset=[*TARGET_COLUMNS, MASS_KG_COLUMN, NAME_COLUMN]
    ).reset_index(drop=True)
    groups_all = data[NAME_COLUMN].astype(str).to_numpy(dtype=object)

    args.feature_mode = "no_feature_selection"
    args.gate_mode = "fixed"
    new_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for seed in MISSING5:
        started = time.time()
        args.random_state = int(seed)
        train_idx, test_idx = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=args.test_size,
                random_state=seed,
            ).split(data, groups=groups_all)
        )
        train = data.iloc[train_idx].reset_index(drop=True)
        test = data.iloc[test_idx].reset_index(drop=True)
        train_groups = train[NAME_COLUMN].astype(str).to_numpy(dtype=object)
        test_groups = test[NAME_COLUMN].astype(str).to_numpy(dtype=object)
        if set(train_groups).intersection(test_groups):
            raise RuntimeError(f"Outer group overlap for seed {seed}")
        fixed.CURRENT_GROUPS = train_groups

        base_oof: dict[str, NDArray[np.float64]] = {}
        base_test: dict[str, NDArray[np.float64]] = {}
        base_names: dict[str, str] = {}
        for target in TARGET_COLUMNS:
            model = check.selected_base_model(
                target, retained_selection, args
            )
            oof, holdout = fixed.group_oof_and_test(
                model,
                train.loc[:, FEATURE_COLUMNS],
                train[target].to_numpy(dtype=float),
                test.loc[:, FEATURE_COLUMNS],
                train_groups,
                args.inner_cv,
            )
            base_oof[target] = oof
            base_test[target] = holdout
            base_names[target] = (
                f"{selected_model_name(model)}_retained_RandomizedSearchCV"
            )

        ordinary: dict[str, tuple[NDArray[np.float64], str]] = {}
        for target in TARGET_COLUMNS:
            ordinary[target] = check.ordinary_prediction(
                target=target,
                X_train=train.loc[:, FEATURE_COLUMNS],
                y_train=train[target].to_numpy(dtype=float),
                X_test=test.loc[:, FEATURE_COLUMNS],
                groups=train_groups,
                args=args,
                selection_rows=selection_rows,
            )

        for route_name, target in check.EVALUATIONS:
            route_code = f"{route_name}-{check.SHORT_TARGET[target]}"
            family, _prefix, auxiliaries, physics_function = check.route(
                route_name, target
            )
            energy_density, voltage, capacity, discharge_energy = (
                check.TARGET_OFFSETS.keys()
            )
            target_index = (
                (energy_density, voltage, capacity).index(target)
                if route_name == "VC"
                else (energy_density, discharge_energy).index(target)
            )
            y_train = train[target].to_numpy(dtype=float)
            y_test = test[target].to_numpy(dtype=float)
            mass_train = train[MASS_KG_COLUMN].to_numpy(dtype=float)
            mass_test = test[MASS_KG_COLUMN].to_numpy(dtype=float)
            direct_oof = base_oof[target]
            direct_test = base_test[target]
            physics_oof = check.arr(
                physics_function(
                    target,
                    {name: base_oof[name] for name in auxiliaries},
                    mass_train,
                )
            )
            physics_test = check.arr(
                physics_function(
                    target,
                    {name: base_test[name] for name in auxiliaries},
                    mass_test,
                )
            )
            residual_train = np.column_stack(
                [
                    direct_oof,
                    physics_oof,
                    *[base_oof[name] for name in auxiliaries],
                    mass_train,
                ]
            )
            residual_test = np.column_stack(
                [
                    direct_test,
                    physics_test,
                    *[base_test[name] for name in auxiliaries],
                    mass_test,
                ]
            )
            residual_model, _ = check.selected_residual_model(
                target,
                family,
                target_index,
                retained_selection,
                args,
            )
            delta_oof, delta_test = fixed.group_oof_and_test(
                residual_model,
                pd.DataFrame(residual_train),
                y_train - physics_oof,
                pd.DataFrame(residual_test),
                train_groups,
                args.inner_cv,
            )
            ares_oof = physics_oof + delta_oof
            ares_test = physics_test + delta_test
            _, physres_test, physres_name, _ = fixed.group_select_prediction(
                np.column_stack([direct_oof, physics_oof, ares_oof]),
                y_train,
                np.column_stack([direct_test, physics_test, ares_test]),
                role=f"{target}::PhysResStack_meta",
                args=args,
                offset=181,
                selection_rows=selection_rows,
            )
            ordinary_test, ordinary_name = ordinary[target]
            for method, prediction, model_name in (
                ("PhysResStack", physres_test, physres_name),
                ("Ordinary stacking", ordinary_test, ordinary_name),
            ):
                row = check.metric_row(
                    target=target,
                    method=method,
                    prediction=prediction,
                    y_test=y_test,
                    selected_model=model_name,
                    family=family,
                    route_target=route_code,
                    args=args,
                    n_train=len(train),
                    n_test=len(test),
                )
                row["source"] = (
                    "retained_train_selected_configs_recomputed_for_top10"
                )
                new_rows.append(row)
        print(
            f"Completed LMB missing seed={seed} in "
            f"{time.time() - started:.1f}s",
            flush=True,
        )

    result = pd.DataFrame(new_rows)
    fixed.atomic_csv(
        result,
        TABLE_DIR / f"{args.output_tag}_missing5_new_methods_detailed.csv",
    )
    fixed.atomic_csv(
        pd.DataFrame(selection_rows),
        TABLE_DIR / f"{args.output_tag}_missing5_new_methods_selection.csv",
    )

    core = retained_detailed.loc[
        retained_detailed["seed"].astype(int).isin(MISSING5)
        & retained_detailed["method"].isin(
            ["Direct", "Physics-only", "Ares_old", "FixedGate_R2_R5_R6"]
        )
    ].copy()
    core["route_target"] = route_target(core)
    combined = pd.concat([core, result], ignore_index=True, sort=False)
    return combined


def add_direct_comparison(detail: pd.DataFrame) -> pd.DataFrame:
    result = detail.copy()
    direct = (
        result.loc[
            result["method"].eq("Direct"),
            ["seed", "route_target", "RMSE"],
        ]
        .rename(columns={"RMSE": "direct_RMSE"})
    )
    result = result.drop(
        columns=["percent_RMSE_change_vs_direct"], errors="ignore"
    ).merge(direct, on=["seed", "route_target"], how="left")
    result["percent_RMSE_change_vs_direct"] = (
        100.0 * (result["RMSE"] - result["direct_RMSE"])
        / result["direct_RMSE"]
    )
    result["method_rank"] = result.groupby(
        ["seed", "route_target"]
    )["RMSE"].rank(method="min", ascending=True)
    return result
