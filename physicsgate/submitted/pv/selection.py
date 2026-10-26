"""PV Direct-model selection on training OOF (entry point).

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/tmp_pv_paper_matched_fixed_gate_multiseed_broad_ridge.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO = WORKDIR

from physicsgate.submitted.pv import physics as pvmain  # noqa: E402
from physicsgate.submitted.pv import meta as paper  # noqa: E402
from physicsgate.submitted.pv import base_models as pvgated  # noqa: E402
from physicsgate.submitted.common.propagated_error import global_auxiliary_sigmas, propagated_sigma_finite_difference  # noqa: E402
from physicsgate.submitted.common.gate_features import reliability_features  # noqa: E402
from physicsgate.submitted.common.gate_fit import fit_gate_model  # noqa: E402
from physicsgate.submitted.pv.common import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402
from physicsgate.submitted.pv.data import DEFAULT_OUTPUT  # noqa: E402

TABLE = REPO / "results" / "submitted"

PREFIX = "tmp_pv_paper_matched_fixed_gate_multiseed_broad_ridge"

SEEDS = (4, 42, 99, 178, 2024)

METHOD = "FixedGate_R2_R5_R6"

FEATURES = [
    "R2_direct_residual_disagreement",
    "R5_physics_out_of_range_amount",
    "R6_propagated_equation_uncertainty",
]


def arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def phi_for_target(target: str):
    aux = [c for c in TARGET_COLUMNS if c != target]
    def phi(values: np.ndarray, _known: dict[str, np.ndarray]) -> np.ndarray:
        return pvmain.compute_pv_physics_prediction(
            target, {column: values[:, i] for i, column in enumerate(aux)}
        )
    return phi


def tune_gate(
    *, y: np.ndarray, direct_oof: np.ndarray, direct_test: np.ndarray,
    residual_oof: np.ndarray, residual_test: np.ndarray,
    rel_oof: pd.DataFrame, rel_test: pd.DataFrame, seed: int, alphas: np.ndarray,
) -> tuple[np.ndarray, float, float, list[dict[str, Any]]]:
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(y))
    records: list[dict[str, Any]] = []
    best_alpha, best_score = float(alphas[0]), float("inf")
    for alpha in alphas:
        pred = np.full_like(y, np.nan, dtype=float)
        for fit, valid in folds:
            _, pred[valid], _ = fit_gate_model(
                y_train=y[fit], direct_train=direct_oof[fit], direct_test=direct_oof[valid],
                residual_train=residual_oof[fit], residual_test=residual_oof[valid],
                reliability_train=rel_oof.iloc[fit].reset_index(drop=True),
                reliability_test=rel_oof.iloc[valid].reset_index(drop=True),
                features=FEATURES, alpha=float(alpha),
            )
        score = paper.metric_dict(y, pred)["RMSE"]
        records.append({"alpha": float(alpha), "train_inner_oof_RMSE": float(score)})
        if score < best_score:
            best_alpha, best_score = float(alpha), float(score)
    _, test_pred, weight = fit_gate_model(
        y_train=y, direct_train=direct_oof, direct_test=direct_test,
        residual_train=residual_oof, residual_test=residual_test,
        reliability_train=rel_oof, reliability_test=rel_test,
        features=FEATURES, alpha=best_alpha,
    )
    for row in records:
        row["selected"] = row["alpha"] == best_alpha
    return test_pred, best_alpha, float(np.mean(weight)), records


def run_seed(seed: int, frame: pd.DataFrame, args: argparse.Namespace, alphas: np.ndarray):
    data = frame.dropna(subset=[*TARGET_COLUMNS, *FEATURE_COLUMNS]).copy().reset_index(drop=True)
    train_idx, test_idx = pvgated.split_indices(len(data), test_size=args.test_size, random_state=seed)
    base_oof: dict[str, np.ndarray] = {}
    base_test: dict[str, np.ndarray] = {}
    base_models: dict[str, str] = {}
    base_rows: list[dict[str, Any]] = []
    for i, target in enumerate(TARGET_COLUMNS):
        oof, test, model, params, rows = pvgated.tune_base_model(
            data, target=target, train_idx=train_idx, test_idx=test_idx,
            random_state=seed + i * 31, n_jobs=args.n_jobs,
            inner_cv=args.inner_cv, rf_n_estimators=args.rf_n_estimators,
        )
        base_oof[target], base_test[target], base_models[target] = arr(oof), arr(test), model
        for row in rows:
            row.update({"seed": seed, "target": target, "selected_params_json": json.dumps(params, sort_keys=True)})
        base_rows.extend(rows)
    metrics: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    for ti, target in enumerate(TARGET_COLUMNS):
        y_all = data[target].to_numpy(dtype=float)
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        aux = [c for c in TARGET_COLUMNS if c != target]
        aux_oof = {c: base_oof[c] for c in aux}
        aux_test = {c: base_test[c] for c in aux}
        direct_oof, direct_test = base_oof[target], base_test[target]
        phys_oof = arr(pvmain.compute_pv_physics_prediction(target, aux_oof))
        phys_test = arr(pvmain.compute_pv_physics_prediction(target, aux_test))
        feature_oof = np.column_stack([direct_oof, phys_oof, *[aux_oof[c] for c in aux]])
        feature_test = np.column_stack([direct_test, phys_test, *[aux_test[c] for c in aux]])
        meta_rows: list[dict[str, Any]] = []
        delta_oof, delta_test, h_name, h_score = paper.select_meta(
            feature_oof, y_train - phys_oof, feature_test,
            y_test=y_test - phys_test, target=target, role=f"{target}_Ares_old_h",
            random_state=seed + 1001 + ti * 37, n_jobs=args.n_jobs,
            inner_cv=args.inner_cv, meta_rows=meta_rows,
        )
        ares_oof, ares_test = phys_oof + delta_oof, phys_test + delta_test
        aux_matrix = np.column_stack([aux_oof[c] for c in aux])
        aux_test_matrix = np.column_stack([aux_test[c] for c in aux])
        aux_true = np.column_stack([data[c].to_numpy(dtype=float)[train_idx] for c in aux])
        sigmas, _ = global_auxiliary_sigmas(aux_pred_train=aux_matrix, aux_true_train=aux_true, aux_names=aux)
        sigma_oof, _ = propagated_sigma_finite_difference(aux_pred=aux_matrix, known={}, aux_sigmas=sigmas, phi=phi_for_target(target))
        sigma_test, _ = propagated_sigma_finite_difference(aux_pred=aux_test_matrix, known={}, aux_sigmas=sigmas, phi=phi_for_target(target))
        rel_oof = reliability_features(direct=direct_oof, physics=phys_oof, residual=ares_oof, y_train=y_train, propagated_uncertainty=sigma_oof)[FEATURES]
        rel_test = reliability_features(direct=direct_test, physics=phys_test, residual=ares_test, y_train=y_train, propagated_uncertainty=sigma_test)[FEATURES]
        gate_test, alpha, mean_w, alpha_rows = tune_gate(
            y=y_train, direct_oof=direct_oof, direct_test=direct_test,
            residual_oof=ares_oof, residual_test=ares_test,
            rel_oof=rel_oof, rel_test=rel_test, seed=seed + 9001 + ti * 41, alphas=alphas,
        )
        direct_rmse = paper.metric_dict(y_test, direct_test)["RMSE"]
        for method, pred, model in (
            ("Direct", direct_test, base_models[target]),
            ("Physics-only", phys_test, "equation_from_predicted_auxiliaries"),
            ("Ares_old", ares_test, h_name),
            (METHOD, gate_test, f"Ridge_alpha={alpha}"),
        ):
            row = {
                "dataset": "PV", "target": pvmain.TARGET_DISPLAY[target], "target_code": target,
                "seed": seed, "method": method, "split_type": "paper_matched_row_holdout",
                "paper_matched": True, "test_size": args.test_size,
                "n_train_rows": len(train_idx), "n_test_rows": len(test_idx),
                "selected_model": model, "selected_gate_alpha": alpha if method == METHOD else np.nan,
                "mean_test_weight": mean_w if method == METHOD else np.nan,
                "uses_true_test_auxiliary_values": False, "uses_test_target_for_selection": False,
            }
            row.update(paper.metric_dict(y_test, pred))
            row["percent_RMSE_change_vs_direct"] = (row["RMSE"] / direct_rmse - 1.0) * 100.0
            metrics.append(row)
        for row in meta_rows:
            row.update({"seed": seed, "target_code": target, "selection_stage": "Ares_old_h"})
        selection.extend(meta_rows)
        for row in alpha_rows:
            row.update({"seed": seed, "target_code": target, "selection_stage": "gate_alpha", "selection_uses_test": False})
        selection.extend(alpha_rows)
        print(f"PV paper-matched fixed gate seed={seed} target={target}", flush=True)
    return metrics, base_rows, selection, {"seed": seed, "split_type": "paper_matched_row_holdout", "paper_matched": True, "test_size": args.test_size, "n_train_rows": len(train_idx), "n_test_rows": len(test_idx), "preprocessing_scope": "paper-matched loader then row split; model fitting train-only", "uses_true_test_auxiliary_values": False, "uses_test_target_for_selection": False, "gate_features": "R2+R5+R6", "gate_alpha_candidates": len(alphas)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--rf-n-estimators", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output-prefix", default=PREFIX)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    alphas = np.logspace(-8.0, 8.0, 73)
    detail_path = TABLE / f"{args.output_prefix}_detailed.csv"
    base_path = TABLE / f"{args.output_prefix}_base_selection.csv"
    selection_path = TABLE / f"{args.output_prefix}_selection.csv"
    audit_path = TABLE / f"{args.output_prefix}_audit.csv"
    if args.resume and detail_path.exists():
        all_metrics = pd.read_csv(detail_path).to_dict("records")
        all_base = pd.read_csv(base_path).to_dict("records") if base_path.exists() else []
        all_selection = pd.read_csv(selection_path).to_dict("records") if selection_path.exists() else []
        audits = pd.read_csv(audit_path).to_dict("records") if audit_path.exists() else []
    else:
        all_metrics: list[dict[str, Any]] = []
        all_base: list[dict[str, Any]] = []
        all_selection: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
    completed = {int(row["seed"]) for row in audits}
    TABLE.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        if seed in completed:
            print(f"PV seed={seed} already complete; skipping", flush=True)
            continue
        m, b, s, a = run_seed(seed, frame, args, alphas)
        all_metrics.extend(m); all_base.extend(b); all_selection.extend(s); audits.append(a)
        pd.DataFrame(all_metrics).to_csv(detail_path, index=False)
        pd.DataFrame(all_base).to_csv(base_path, index=False)
        pd.DataFrame(all_selection).to_csv(selection_path, index=False)
        pd.DataFrame(audits).to_csv(audit_path, index=False)
        print(f"Checkpointed completed seed={seed}", flush=True)
    detailed = pd.DataFrame(all_metrics)
    rows: list[dict[str, Any]] = []
    for (target, code, method), group in detailed.groupby(["target", "target_code", "method"], sort=False):
        rows.append({"dataset": "PV", "target": target, "target_code": code, "method": method, "n_seeds": group.seed.nunique(), "seeds": ";".join(map(str, args.seeds)), "RMSE_mean": group.RMSE.mean(), "RMSE_std": group.RMSE.std(ddof=1), "MAE_mean": group.MAE.mean(), "MAE_std": group.MAE.std(ddof=1), "R2_mean": group.R2.mean(), "R2_std": group.R2.std(ddof=1), "percent_RMSE_change_vs_direct_mean": group.percent_RMSE_change_vs_direct.mean(), "percent_RMSE_change_vs_direct_std": group.percent_RMSE_change_vs_direct.std(ddof=1), "win_count_vs_direct": int((group.percent_RMSE_change_vs_direct < 0).sum())})
    detailed.to_csv(detail_path, index=False)
    pd.DataFrame(rows).to_csv(TABLE / f"{args.output_prefix}_summary.csv", index=False)
    pd.DataFrame(all_base).to_csv(base_path, index=False)
    pd.DataFrame(all_selection).to_csv(selection_path, index=False)
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
