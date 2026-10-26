"""LMB gate tuning.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/tmp_lmb_fixed_gate_tuned_multiseed_vc.py`` in the original analysis code; only imports and paths were adapted.
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
from sklearn.model_selection import train_test_split

ROOT = WORKDIR

from physicsgate.submitted.lmb import gate_vc as base  # noqa: E402
from physicsgate.submitted.lmb.gate_features import tune_r2_r5_r6_gate  # noqa: E402

METHOD = "FixedGate_R2_R5_R6"

ADAPTIVE_METHOD = "AdaptiveGate_Achoose_Acorrect_R2_R5_R6"

FORMULA_TARGETS = (
    base.ENERGY_DENSITY_TARGET,
    base.VOLTAGE_TARGET,
    base.CAPACITY_TARGET,
)

FORMULA_AUXILIARIES = {
    base.ENERGY_DENSITY_TARGET: (base.VOLTAGE_TARGET, base.CAPACITY_TARGET),
    base.VOLTAGE_TARGET: (base.ENERGY_DENSITY_TARGET, base.CAPACITY_TARGET),
    base.CAPACITY_TARGET: (base.ENERGY_DENSITY_TARGET, base.VOLTAGE_TARGET),
}

FORMULA_FAMILY = "VC_mass_only"

FORMULA_DESCRIPTION = "ED=capacity*voltage/mass"


def physics_prediction(
    target: str,
    predictions: dict[str, NDArray[np.float64]],
    mass_kg: NDArray[np.float64],
) -> NDArray[np.float64]:
    return base.vc_mass_physics_prediction(target, predictions, mass_kg)


def propagated_sigma(
    target: str,
    predictions: dict[str, NDArray[np.float64]],
    mass_kg: NDArray[np.float64],
    sigmas: dict[str, float],
) -> NDArray[np.float64]:
    return base.propagated_sigma_vc_mass(target, predictions, mass_kg, sigmas)


def _array(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def metric_row(
    *,
    seed: int,
    target: str,
    method: str,
    y_true: NDArray[np.float64],
    prediction: NDArray[np.float64],
    direct_rmse: float,
    selected_model: str,
    selected_alpha: float | None,
    selection_rmse: float | None,
    train_rows: int,
    test_rows: int,
    feature_count: int,
    output_tag: str,
    selected_anchor: str = "",
    selected_correction: str = "",
) -> dict[str, Any]:
    is_gate = method in {METHOD, ADAPTIVE_METHOD}
    row = {
        "dataset": base.DATASET,
        "target": base.TARGET_DISPLAY[target],
        "raw_target_column": target,
        "seed": seed,
        "method": method,
        "split_mode": "single_90_10_holdout",
        "test_size": 0.10,
        "formula_family": f"{FORMULA_FAMILY}: {FORMULA_DESCRIPTION}",
        "gate_structure": (
            "Achoose+w*(Acorrect-Achoose)"
            if method == ADAPTIVE_METHOD
            else "Direct+w*(Ares_old-Direct)"
        ),
        "selected_anchor": selected_anchor,
        "selected_correction": selected_correction,
        "gate_features": "R2_R5_R6",
        "gate_weighting": "delta2",
        "gate_alpha_selection": "train_oof_RMSE" if is_gate else "",
        "selected_gate_alpha": selected_alpha,
        "gate_selection_RMSE": selection_rmse,
        "selected_model": selected_model,
        "feature_mode": "paper_matched_rfe",
        "feature_count": feature_count,
        "n_train_rows": train_rows,
        "n_test_rows": test_rows,
        "uses_true_test_auxiliary_values": False,
        "uses_oof_training_predictions": True,
        "uses_discharge_energy_auxiliary": base.DISCHARGE_ENERGY_TARGET
        in FORMULA_AUXILIARIES[target],
        "output_tag": output_tag,
    }
    row.update(base.metrics_dict(y_true, prediction))
    row["percent_RMSE_change_vs_direct"] = (row["RMSE"] - direct_rmse) / direct_rmse * 100.0
    return row


def evaluate_target(
    *,
    seed: int,
    target: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_oof: dict[str, NDArray[np.float64]],
    base_test: dict[str, NDArray[np.float64]],
    model_names: dict[str, str],
    feature_counts: dict[str, int],
    args: argparse.Namespace,
    alpha_grid: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    y_train = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    mass_train = train[base.MASS_KG_COLUMN].to_numpy(dtype=float)
    mass_test = test[base.MASS_KG_COLUMN].to_numpy(dtype=float)
    aux_columns = FORMULA_AUXILIARIES[target]
    direct_oof = base_oof[target]
    direct_test = base_test[target]
    aux_oof = {column: base_oof[column] for column in aux_columns}
    aux_test = {column: base_test[column] for column in aux_columns}
    physics_oof = physics_prediction(target, aux_oof, mass_train)
    physics_test = physics_prediction(target, aux_test, mass_test)

    residual_train = np.column_stack(
        [direct_oof, physics_oof, *[aux_oof[column] for column in aux_columns], mass_train]
    )
    residual_test = np.column_stack(
        [direct_test, physics_test, *[aux_test[column] for column in aux_columns], mass_test]
    )
    selection_rows: list[dict[str, Any]] = []
    target_index = FORMULA_TARGETS.index(target)
    delta_oof, delta_test, residual_model, _ = base.select_prediction(
        residual_train,
        y_train - physics_oof,
        residual_test,
        role=f"{target}::FixedGate_Ares_old_h",
        args=args,
        offset=1001 + target_index * 37,
        selection_rows=selection_rows,
    )
    ares_oof = physics_oof + _array(delta_oof)
    ares_test = physics_test + _array(delta_test)

    ares_d_oof: NDArray[np.float64] | None = None
    ares_d_test: NDArray[np.float64] | None = None
    ares_d_model = ""
    if args.gate_mode == "adaptive":
        delta_d_oof, delta_d_test, ares_d_model, _ = base.select_prediction(
            residual_train,
            y_train - direct_oof,
            residual_test,
            role=f"{target}::AdaptiveGate_Ares_d_h",
            args=args,
            offset=2001 + target_index * 37,
            selection_rows=selection_rows,
        )
        ares_d_oof = direct_oof + _array(delta_d_oof)
        ares_d_test = direct_test + _array(delta_d_test)

    train_mae_direct = float(np.mean(np.abs(y_train - direct_oof)))
    train_mae_physics = float(np.mean(np.abs(y_train - physics_oof)))
    train_mae_ares_old = float(np.mean(np.abs(y_train - ares_oof)))
    train_mae_ares_d = (
        float(np.mean(np.abs(y_train - ares_d_oof)))
        if ares_d_oof is not None
        else np.nan
    )
    if args.gate_mode == "adaptive":
        selected_anchor = (
            "Direct" if train_mae_direct <= train_mae_physics else "Physics-only"
        )
        selected_correction = (
            "Ares_old" if train_mae_ares_old <= train_mae_ares_d else "Ares_d"
        )
        anchor_oof = direct_oof if selected_anchor == "Direct" else physics_oof
        anchor_test = direct_test if selected_anchor == "Direct" else physics_test
        correction_oof = ares_oof if selected_correction == "Ares_old" else ares_d_oof
        correction_test = ares_test if selected_correction == "Ares_old" else ares_d_test
        gate_method = ADAPTIVE_METHOD
    else:
        selected_anchor = "Direct"
        selected_correction = "Ares_old"
        anchor_oof = direct_oof
        anchor_test = direct_test
        correction_oof = ares_oof
        correction_test = ares_test
        gate_method = METHOD
    assert correction_oof is not None and correction_test is not None

    sigmas: dict[str, float] = {}
    for column in aux_columns:
        error = base_oof[column] - train[column].to_numpy(dtype=float)
        sigmas[column] = float(np.sqrt(np.mean(error**2)))
    sigma_train = propagated_sigma(target, aux_oof, mass_train, sigmas)
    sigma_test = propagated_sigma(target, aux_test, mass_test, sigmas)
    rel_train = base.reliability_features(
        direct=anchor_oof,
        physics=physics_oof,
        residual=correction_oof,
        y_train=y_train,
        propagated_uncertainty=sigma_train,
    )
    rel_test = base.reliability_features(
        direct=anchor_test,
        physics=physics_test,
        residual=correction_test,
        y_train=y_train,
        propagated_uncertainty=sigma_test,
    )
    # The retained LMB and shared tuned-gate modules use different names for
    # the same propagated-equation uncertainty feature.
    rel_train["R6_global_propagated_uncertainty_proxy"] = rel_train[
        "R6_propagated_equation_uncertainty"
    ]
    rel_test["R6_global_propagated_uncertainty_proxy"] = rel_test[
        "R6_propagated_equation_uncertainty"
    ]
    gate_test, selected, gate_records = tune_r2_r5_r6_gate(
        y_train=y_train,
        direct_train=anchor_oof,
        direct_test=anchor_test,
        residual_train=correction_oof,
        residual_test=correction_test,
        reliability_train=rel_train,
        reliability_test=rel_test,
        groups=np.arange(len(train), dtype=object),
        seed=seed + 7001 + target_index * 43,
        inner_cv=args.gate_inner_cv,
        alpha_grid=alpha_grid,
    )
    for row in selection_rows:
        role = str(row.get("role", ""))
        row.update({
            "seed": seed,
            "target": base.TARGET_DISPLAY[target],
            "selection_stage": "Ares_d_h" if "Ares_d" in role else "Ares_old_h",
        })
    for row in gate_records:
        row.update({
            "seed": seed,
            "target": base.TARGET_DISPLAY[target],
            "selection_stage": "gate_alpha",
            "selected": float(row["alpha"]) == float(selected["alpha"]),
            "gate_mode": args.gate_mode,
        })
    selection_rows.extend(gate_records)

    direct_rmse = base.metrics_dict(y_test, direct_test)["RMSE"]
    common = {
        "seed": seed,
        "target": target,
        "y_true": y_test,
        "direct_rmse": direct_rmse,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": feature_counts[target],
        "output_tag": args.output_tag,
        "selected_anchor": selected_anchor,
        "selected_correction": selected_correction,
    }
    rows = [
        metric_row(method="Direct", prediction=direct_test, selected_model=model_names[target], selected_alpha=None, selection_rmse=None, **common),
        metric_row(method="Physics-only", prediction=physics_test, selected_model="equation", selected_alpha=None, selection_rmse=None, **common),
        metric_row(method="Ares_old", prediction=ares_test, selected_model=residual_model, selected_alpha=None, selection_rmse=None, **common),
    ]
    if ares_d_test is not None:
        rows.append(
            metric_row(
                method="Ares_d",
                prediction=ares_d_test,
                selected_model=ares_d_model,
                selected_alpha=None,
                selection_rmse=None,
                **common,
            )
        )
    rows.append(
        metric_row(
            method=gate_method,
            prediction=gate_test,
            selected_model=f"StandardScaler+Ridge(alpha={selected['alpha']})",
            selected_alpha=float(selected["alpha"]),
            selection_rmse=float(selected["selection_RMSE"]),
            **common,
        )
    )
    audit = {
        "seed": seed,
        "target": base.TARGET_DISPLAY[target],
        "formula_family": FORMULA_FAMILY,
        "formula_auxiliaries": json.dumps(list(aux_columns)),
        "selected_gate_alpha": float(selected["alpha"]),
        "gate_selection_RMSE": float(selected["selection_RMSE"]),
        "gate_features": "R2_R5_R6",
        "gate_weighting": "delta2",
        "uses_true_test_auxiliary_values": False,
        "uses_oof_training_predictions": True,
        "uses_discharge_energy_auxiliary": base.DISCHARGE_ENERGY_TARGET
        in aux_columns,
        "gate_mode": args.gate_mode,
        "selected_anchor": selected_anchor,
        "selected_correction": selected_correction,
        "train_MAE_Direct": train_mae_direct,
        "train_MAE_Physics": train_mae_physics,
        "train_MAE_Ares_old": train_mae_ares_old,
        "train_MAE_Ares_d": train_mae_ares_d,
        "target_feature_count": feature_counts[target],
    }
    return rows, selection_rows, audit
