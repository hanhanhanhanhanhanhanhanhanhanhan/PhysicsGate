"""PV six-method pipeline (entry point).

Copy of ``physicsgate.submitted.pv.pipeline`` with change C1
(``physicsgate/revised/CHANGES.md``), marked ``REVISION``. Submitted-protocol
definitions are copied verbatim from
``figma5/scripts/run_pv_unified_train_range_50seed.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR, PV_WORKDIR  # noqa: E402
import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = PV_WORKDIR

REPO = WORKDIR

OUTPUT = ROOT / "outputs" / "pv_unified_train_range_50seed"

BY_SEED = OUTPUT / "by_seed"

SEED_FILE = (
    ROOT
    / "outputs"
    / "figure4_gate_mechanism"
    / "seed_plans"
    / "Figure4_PV_50Seed_List.txt"
)

OLD_BASE_SELECTION = (
    ROOT
    / "outputs"
    / "figure4_gate_mechanism"
    / "50seed"
    / "pv"
    / "Figure4_PV_50Seed_Rerun_BaseSelection.csv"
)

FROZEN_FIVE = {243, 938, 1025, 1790, 2024}

METHOD_GATE = "FixedGate_R2_R5_R6"

PAPER_FEATURES = ("R1", "R2", "R3")

SOURCE_FEATURES = {
    "R1": "R2_direct_residual_disagreement",
    "R2": "R5_physics_out_of_range_amount",
    "R3": "R6_propagated_equation_uncertainty",
}

FEATURE_SETS = [
    ("R1",),
    ("R2",),
    ("R3",),
    ("R1", "R2"),
    ("R1", "R3"),
    ("R2", "R3"),
    ("R1", "R2", "R3"),
]

from physicsgate.submitted.pv import selection as pv  # noqa: E402
from physicsgate.submitted.common.propagated_error import global_auxiliary_sigmas, propagated_sigma_finite_difference  # noqa: E402
from physicsgate.submitted.common.gate_features import reliability_features  # noqa: E402
from physicsgate.submitted.common.gate_fit import fit_gate_model  # noqa: E402
from physicsgate.submitted.pv.common import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402
from physicsgate.submitted.pv.stabilization import clipping_audit, fit_pv_base_once, train_range_clip  # noqa: E402
from physicsgate.revised import ordinary  # noqa: E402


def arr(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def rmse(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(np.mean(np.square(arr(y_true) - arr(y_pred)))))


def read_seeds() -> list[int]:
    text = SEED_FILE.read_text(encoding="utf-8-sig")
    return [
        int(value.strip())
        for value in text.replace("\n", ",").split(",")
        if value.strip()
    ]


def selected_base_models() -> dict[tuple[int, str], str]:
    frame = pd.read_csv(OLD_BASE_SELECTION)
    frame = frame.loc[frame["selected"].astype(str).str.lower().eq("true")]
    mapping = {
        (int(row.seed), str(row.target)): str(row.candidate_model)
        for row in frame.itertuples(index=False)
    }
    expected = {(seed, target) for seed in read_seeds() for target in TARGET_COLUMNS}
    missing = expected.difference(mapping)
    if missing:
        raise RuntimeError(f"Missing retained train-OOF Direct selections: {sorted(missing)[:5]}")
    return mapping


def clipped_phi(target: str, aux_columns: list[str], y_train: np.ndarray):
    def phi(values: np.ndarray, _known: dict[str, np.ndarray]) -> np.ndarray:
        raw = pv.pvmain.compute_pv_physics_prediction(
            target,
            {column: values[:, index] for index, column in enumerate(aux_columns)},
        )
        return train_range_clip(
            target,
            arr(raw),
            y_train,
            forward_target="PCE_percent",
        )

    return phi


def tune_subset(
    *,
    y: np.ndarray,
    direct_oof: np.ndarray,
    direct_test: np.ndarray,
    residual_oof: np.ndarray,
    residual_test: np.ndarray,
    rel_oof: pd.DataFrame,
    rel_test: pd.DataFrame,
    seed: int,
    alphas: np.ndarray,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(y))
    best_alpha = float(alphas[0])
    best_score = np.inf
    for alpha in alphas:
        validation = np.full_like(y, np.nan, dtype=float)
        for fit_idx, valid_idx in folds:
            _, validation[valid_idx], _ = fit_gate_model(
                y_train=y[fit_idx],
                direct_train=direct_oof[fit_idx],
                direct_test=direct_oof[valid_idx],
                residual_train=residual_oof[fit_idx],
                residual_test=residual_oof[valid_idx],
                reliability_train=rel_oof.iloc[fit_idx].reset_index(drop=True),
                reliability_test=rel_oof.iloc[valid_idx].reset_index(drop=True),
                alpha=float(alpha),
                features=features,
            )
        score = rmse(y, validation)
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    _, prediction, weight = fit_gate_model(
        y_train=y,
        direct_train=direct_oof,
        direct_test=direct_test,
        residual_train=residual_oof,
        residual_test=residual_test,
        reliability_train=rel_oof,
        reliability_test=rel_test,
        alpha=best_alpha,
        features=features,
    )
    return arr(prediction), arr(weight), best_alpha, float(best_score)


def metric_row(
    *,
    target: str,
    seed: int,
    method: str,
    prediction: np.ndarray,
    y_test: np.ndarray,
    direct_rmse: float,
    n_train: int,
    selected_model: str,
    selected_alpha: float | None = None,
    mean_weight: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": "PV",
        "target": pv.pvmain.TARGET_DISPLAY[target],
        "target_code": target,
        "seed": int(seed),
        "method": method,
        "split_type": "paper_matched_row_holdout",
        "paper_matched": True,
        "stabilization": (
            "outer_train_target_minmax_for_inverse_targets; "
            "forward_PCE_unclipped"
        ),
        "test_size": 0.30,
        "n_train_rows": int(n_train),
        "n_test_rows": int(len(y_test)),
        "selected_model": selected_model,
        "selected_gate_alpha": selected_alpha,
        "mean_test_weight": mean_weight,
        "uses_true_test_auxiliary_values": False,
        "uses_test_target_for_selection": False,
    }
    row.update(pv.paper.metric_dict(y_test, prediction))
    row["percent_RMSE_change_vs_direct"] = (
        float(row["RMSE"]) / direct_rmse - 1.0
    ) * 100.0
    return row


def run_seed(
    seed: int,
    frame: pd.DataFrame,
    *,
    inner_cv: int,
    n_jobs: int,
    base_map: dict[tuple[int, str], str],
) -> None:
    seed_dir = BY_SEED / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    complete_path = seed_dir / "COMPLETE.txt"
    if complete_path.exists():
        print(f"[PV unified] seed={seed} complete; skipping", flush=True)
        return

    data = (
        frame.dropna(subset=[*TARGET_COLUMNS, *FEATURE_COLUMNS])
        .copy()
        .reset_index(drop=False)
        .rename(columns={"index": "_source_row_index"})
    )
    train_idx, test_idx = pv.pvgated.split_indices(
        len(data), test_size=0.30, random_state=seed
    )
    base_oof: dict[str, np.ndarray] = {}
    base_test: dict[str, np.ndarray] = {}
    base_rows: list[dict[str, Any]] = []
    for index, target in enumerate(TARGET_COLUMNS):
        selected_name = base_map[(seed, target)]
        oof, test = fit_pv_base_once(
            data=data,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            random_state=seed + index * 31,
            n_jobs=n_jobs,
            inner_cv=inner_cv,
            selected_name=selected_name,
        )
        base_oof[target] = arr(oof)
        base_test[target] = arr(test)
        base_rows.append(
            {
                "seed": seed,
                "target_code": target,
                "selected_base_model": selected_name,
                "selection_source": OLD_BASE_SELECTION.name,
                "selection_is_independent_of_physics_stabilization": True,
            }
        )

    alpha_grid = np.logspace(-8.0, 8.0, 73)
    metrics: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    blend_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for target_index, target in enumerate(TARGET_COLUMNS):
        y_all = data[target].to_numpy(dtype=float)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        direct_oof = base_oof[target]
        direct_test = base_test[target]
        aux_columns = [column for column in TARGET_COLUMNS if column != target]
        aux_oof = {column: base_oof[column] for column in aux_columns}
        aux_test = {column: base_test[column] for column in aux_columns}

        raw_phys_oof = arr(
            pv.pvmain.compute_pv_physics_prediction(target, aux_oof)
        )
        raw_phys_test = arr(
            pv.pvmain.compute_pv_physics_prediction(target, aux_test)
        )
        phys_oof = train_range_clip(
            target,
            raw_phys_oof,
            y_train,
            forward_target="PCE_percent",
        )
        phys_test = train_range_clip(
            target,
            raw_phys_test,
            y_train,
            forward_target="PCE_percent",
        )
        residual_feature_oof = np.column_stack(
            [direct_oof, phys_oof, *[aux_oof[column] for column in aux_columns]]
        )
        residual_feature_test = np.column_stack(
            [direct_test, phys_test, *[aux_test[column] for column in aux_columns]]
        )

        residual_selection: list[dict[str, Any]] = []
        delta_oof, delta_test, residual_name, _ = pv.paper.select_meta(
            residual_feature_oof,
            y_train - phys_oof,
            residual_feature_test,
            y_test=y_test - phys_test,
            target=target,
            role=f"{target}_Ares_old_h_unified_train_range",
            random_state=seed + 1001 + target_index * 37,
            n_jobs=n_jobs,
            inner_cv=inner_cv,
            meta_rows=residual_selection,
        )
        residual_oof = phys_oof + arr(delta_oof)
        residual_test = phys_test + arr(delta_test)
        for row in residual_selection:
            row.update(
                {
                    "seed": seed,
                    "target_code": target,
                    "selection_stage": "Ares_old_h",
                    "stabilization": "outer_train_target_minmax",
                }
            )
        selections.extend(residual_selection)

        aux_matrix = np.column_stack([aux_oof[column] for column in aux_columns])
        aux_test_matrix = np.column_stack(
            [aux_test[column] for column in aux_columns]
        )
        aux_true = np.column_stack(
            [
                data[column].to_numpy(dtype=float)[train_idx]
                for column in aux_columns
            ]
        )
        aux_sigmas, _ = global_auxiliary_sigmas(
            aux_pred_train=aux_matrix,
            aux_true_train=aux_true,
            aux_names=aux_columns,
        )
        phi = clipped_phi(target, aux_columns, y_train)
        sigma_oof, _ = propagated_sigma_finite_difference(
            aux_pred=aux_matrix,
            known={},
            aux_sigmas=aux_sigmas,
            phi=phi,
        )
        sigma_test, _ = propagated_sigma_finite_difference(
            aux_pred=aux_test_matrix,
            known={},
            aux_sigmas=aux_sigmas,
            phi=phi,
        )
        rel_oof_all = reliability_features(
            direct=direct_oof,
            physics=raw_phys_oof,  # REVISION C1
            residual=residual_oof,
            y_train=y_train,
            propagated_uncertainty=sigma_oof,
        )
        rel_test_all = reliability_features(
            direct=direct_test,
            physics=raw_phys_test,  # REVISION C1
            residual=residual_test,
            y_train=y_train,
            propagated_uncertainty=sigma_test,
        )
        rel_oof = rel_oof_all[pv.FEATURES]
        rel_test = rel_test_all[pv.FEATURES]

        gate_test, alpha, mean_weight, alpha_selection = pv.tune_gate(
            y=y_train,
            direct_oof=direct_oof,
            direct_test=direct_test,
            residual_oof=residual_oof,
            residual_test=residual_test,
            rel_oof=rel_oof,
            rel_test=rel_test,
            seed=seed + 9001 + target_index * 41,
            alphas=alpha_grid,
        )
        _, gate_check, gate_weight = fit_gate_model(
            y_train=y_train,
            direct_train=direct_oof,
            direct_test=direct_test,
            residual_train=residual_oof,
            residual_test=residual_test,
            reliability_train=rel_oof,
            reliability_test=rel_test,
            alpha=float(alpha),
            features=pv.FEATURES,
        )
        if not np.allclose(arr(gate_test), arr(gate_check), rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"Gate prediction replay failed for {(seed, target)}")
        if not np.isclose(float(np.mean(gate_weight)), mean_weight):
            raise RuntimeError(f"Gate weight replay failed for {(seed, target)}")
        for row in alpha_selection:
            row.update(
                {
                    "seed": seed,
                    "target_code": target,
                    "selection_stage": "gate_alpha",
                    "stabilization": "outer_train_target_minmax",
                }
            )
        selections.extend(alpha_selection)

        direct_value = rmse(y_test, direct_test)
        method_predictions: list[tuple[str, np.ndarray, str, float | None, float | None]] = [
            ("Direct", direct_test, base_map[(seed, target)], None, None),
            (
                "Physics-only",
                phys_test,
                "equation_from_predicted_auxiliaries_train_range_stabilized",
                None,
                None,
            ),
            ("Ares_old", residual_test, residual_name, None, None),
            (
                METHOD_GATE,
                arr(gate_test),
                f"Ridge_alpha={alpha}",
                float(alpha),
                float(mean_weight),
            ),
        ]

        if seed in FROZEN_FIVE:
            ordinary_rows: list[dict[str, Any]] = []
            # REVISION C2: the meta-learner combines base learners fitted to the target from the Direct inputs.
            extra_oof, extra_test = ordinary.pv_base_predictions(
                data, target, train_idx, test_idx, feature_columns=FEATURE_COLUMNS,
                encode=pv.pvgated.encode_features_for_target, random_state=seed + target_index * 31 + 211,
                n_jobs=n_jobs, inner_cv=inner_cv,
            )
            _, ordinary_test, ordinary_name, _ = pv.paper.select_meta(
                np.column_stack([direct_oof, *extra_oof]),
                y_train,
                np.column_stack([direct_test, *extra_test]),
                y_test=y_test,
                target=target,
                role=f"{target}_Ordinary_stacking_unified_train_range",
                random_state=seed + 149,
                n_jobs=n_jobs,
                inner_cv=inner_cv,
                meta_rows=ordinary_rows,
            )
            physres_rows: list[dict[str, Any]] = []
            _, physres_test, physres_name, _ = pv.paper.select_meta(
                np.column_stack([direct_oof, phys_oof, residual_oof]),
                y_train,
                np.column_stack([direct_test, phys_test, residual_test]),
                y_test=y_test,
                target=target,
                role=f"{target}_PhysResStack_unified_train_range",
                random_state=seed + 181,
                n_jobs=n_jobs,
                inner_cv=inner_cv,
                meta_rows=physres_rows,
            )
            for stage, records in (
                ("ordinary_stacking_meta", ordinary_rows),
                ("physresstack_meta", physres_rows),
            ):
                for row in records:
                    row.update(
                        {
                            "seed": seed,
                            "target_code": target,
                            "selection_stage": stage,
                            "stabilization": "outer_train_target_minmax",
                        }
                    )
                selections.extend(records)
            method_predictions.extend(
                [
                    (
                        "Ordinary stacking",
                        arr(ordinary_test),
                        ordinary_name,
                        None,
                        None,
                    ),
                    (
                        "PhysResStack",
                        arr(physres_test),
                        physres_name,
                        None,
                        None,
                    ),
                ]
            )

        for method, prediction, model, selected_alpha, selected_weight in method_predictions:
            metrics.append(
                metric_row(
                    target=target,
                    seed=seed,
                    method=method,
                    prediction=prediction,
                    y_test=y_test,
                    direct_rmse=direct_value,
                    n_train=len(train_idx),
                    selected_model=model,
                    selected_alpha=selected_alpha,
                    mean_weight=selected_weight,
                )
            )

        source_indices = data["_source_row_index"].to_numpy(dtype=int)[test_idx]
        gate_weight = arr(gate_weight)
        gate_test = arr(gate_test)
        for row_index in range(len(y_test)):
            sample_rows.append(
                {
                    "dataset": "PV",
                    "target": pv.pvmain.TARGET_DISPLAY[target],
                    "target_code": target,
                    "seed": seed,
                    "source_row_index": int(source_indices[row_index]),
                    "y_true": float(y_test[row_index]),
                    "A_direct": float(direct_test[row_index]),
                    "A_physics": float(phys_test[row_index]),
                    "A_residual": float(residual_test[row_index]),
                    "A_final": float(gate_test[row_index]),
                    "w": float(gate_weight[row_index]),
                    "R1_branch_disagreement": float(
                        rel_test_all.iloc[row_index][SOURCE_FEATURES["R1"]]
                    ),
                    "R2_physics_extrapolation": float(
                        rel_test_all.iloc[row_index][SOURCE_FEATURES["R2"]]
                    ),
                    "R3_propagated_uncertainty": float(
                        rel_test_all.iloc[row_index][SOURCE_FEATURES["R3"]]
                    ),
                    "selected_gate_alpha": float(alpha),
                    "uses_test_target_for_selection": False,
                }
            )

        for paper_features in FEATURE_SETS:
            source_subset = [SOURCE_FEATURES[name] for name in paper_features]
            if paper_features == PAPER_FEATURES:
                prediction = gate_test
                weight = gate_weight
                subset_alpha = float(alpha)
                subset_score = min(
                    float(row["train_inner_oof_RMSE"]) for row in alpha_selection
                )
            else:
                prediction, weight, subset_alpha, subset_score = tune_subset(
                    y=y_train,
                    direct_oof=direct_oof,
                    direct_test=direct_test,
                    residual_oof=residual_oof,
                    residual_test=residual_test,
                    rel_oof=rel_oof_all,
                    rel_test=rel_test_all,
                    seed=seed + 9001 + target_index * 41,
                    alphas=alpha_grid,
                    features=source_subset,
                )
            value = rmse(y_test, prediction)
            ablation_rows.append(
                {
                    "dataset": "PV",
                    "target": pv.pvmain.TARGET_DISPLAY[target],
                    "target_code": target,
                    "seed": seed,
                    "feature_set": "+".join(paper_features),
                    "selected_gate_alpha": subset_alpha,
                    "train_oof_selection_RMSE": subset_score,
                    "test_RMSE": value,
                    "direct_test_RMSE": direct_value,
                    "paired_log2_gain_vs_direct": np.log2(direct_value / value),
                    "mean_test_weight": float(np.mean(weight)),
                }
            )

        delta_train = residual_oof - direct_oof
        denominator = float(np.dot(delta_train, delta_train))
        fixed_alpha = (
            float(
                np.clip(
                    np.dot(delta_train, y_train - direct_oof) / denominator,
                    0.0,
                    1.0,
                )
            )
            if denominator > 1e-15
            else 0.0
        )
        fixed_prediction = direct_test + fixed_alpha * (
            residual_test - direct_test
        )
        for method, prediction in (
            ("Direct", direct_test),
            ("Residual-only", residual_test),
            ("Fixed scalar blend", fixed_prediction),
            ("PhysicsGate", gate_test),
        ):
            value = rmse(y_test, prediction)
            blend_rows.append(
                {
                    "dataset": "PV",
                    "target": pv.pvmain.TARGET_DISPLAY[target],
                    "target_code": target,
                    "seed": seed,
                    "method": method,
                    "fixed_alpha_train_oof": (
                        fixed_alpha if method == "Fixed scalar blend" else np.nan
                    ),
                    "test_RMSE": value,
                    "direct_test_RMSE": direct_value,
                    "paired_log2_gain_vs_direct": np.log2(direct_value / value),
                }
            )

        audit = clipping_audit(raw_phys_test, phys_test, y_train)
        audit.update(
            {
                "dataset": "PV",
                "seed": seed,
                "target": pv.pvmain.TARGET_DISPLAY[target],
                "target_code": target,
                "forward_target": target == "PCE_percent",
                "uses_outer_train_target_range": target != "PCE_percent",
                "selected_base_model": base_map[(seed, target)],
                "selected_residual_model": residual_name,
                "uses_true_test_auxiliary_values": False,
                "uses_test_target_for_selection": False,
            }
        )
        audit_rows.append(audit)
        print(f"[PV unified] seed={seed} target={target}", flush=True)

    tables = {
        "Detailed.csv": pd.DataFrame(metrics),
        "BaseSelection.csv": pd.DataFrame(base_rows),
        "Selection.csv": pd.DataFrame(selections),
        "PerSample_GateDiagnostics.csv": pd.DataFrame(sample_rows),
        "Reliability_Ablation.csv": pd.DataFrame(ablation_rows),
        "FixedBlend_Comparison.csv": pd.DataFrame(blend_rows),
        "Audit.csv": pd.DataFrame(audit_rows),
    }
    for suffix, table in tables.items():
        table.to_csv(seed_dir / f"PV_Unified_{suffix}", index=False)
    complete_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "n_targets": len(TARGET_COLUMNS),
                "inverse_stabilization": "outer_train_target_minmax",
                "forward_target": "PCE_percent",
                "uses_true_test_auxiliary_values": False,
                "uses_test_target_for_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[PV unified] seed={seed} checkpointed", flush=True)


def aggregate(seeds: list[int]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stems = [
        "Detailed",
        "BaseSelection",
        "Selection",
        "PerSample_GateDiagnostics",
        "Reliability_Ablation",
        "FixedBlend_Comparison",
        "Audit",
    ]
    for stem in stems:
        frames = [
            pd.read_csv(
                BY_SEED / f"seed_{seed}" / f"PV_Unified_{stem}.csv"
            )
            for seed in seeds
        ]
        pd.concat(frames, ignore_index=True, sort=False).to_csv(
            OUTPUT / f"PV_Unified_50Seed_{stem}.csv", index=False
        )

    detail = pd.read_csv(OUTPUT / "PV_Unified_50Seed_Detailed.csv")
    summary_rows: list[dict[str, Any]] = []
    for (target, target_code, method), group in detail.groupby(
        ["target", "target_code", "method"], sort=False
    ):
        summary_rows.append(
            {
                "dataset": "PV",
                "target": target,
                "target_code": target_code,
                "method": method,
                "n_seeds": int(group["seed"].nunique()),
                "seeds": ";".join(map(str, sorted(group["seed"].unique()))),
                "RMSE_mean": float(group["RMSE"].mean()),
                "RMSE_std": float(group["RMSE"].std(ddof=1)),
                "MAE_mean": float(group["MAE"].mean()),
                "MAE_std": float(group["MAE"].std(ddof=1)),
                "R2_mean": float(group["R2"].mean()),
                "R2_std": float(group["R2"].std(ddof=1)),
                "percent_RMSE_change_vs_direct_mean": float(
                    group["percent_RMSE_change_vs_direct"].mean()
                ),
                "percent_RMSE_change_vs_direct_std": float(
                    group["percent_RMSE_change_vs_direct"].std(ddof=1)
                ),
                "win_count_vs_direct": int(
                    (group["percent_RMSE_change_vs_direct"] < 0).sum()
                ),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT / "PV_Unified_50Seed_Summary.csv", index=False
    )

    if not FROZEN_FIVE.issubset(set(detail["seed"].astype(int).unique())):
        print(
            "Frozen-five export deferred until all five publication seeds are present",
            flush=True,
        )
        return

    frozen = detail.loc[detail["seed"].isin(FROZEN_FIVE)].copy()
    expected = len(FROZEN_FIVE) * len(TARGET_COLUMNS) * 6
    if len(frozen) != expected:
        raise RuntimeError(
            f"Frozen-five six-method table has {len(frozen)} rows; expected {expected}"
        )
    method_order = {
        "Direct": 0,
        "Physics-only": 1,
        "Ares_old": 2,
        "Ordinary stacking": 3,
        "PhysResStack": 4,
        METHOD_GATE: 5,
    }
    frozen["_method_order"] = frozen["method"].map(method_order)
    frozen = frozen.sort_values(["target_code", "seed", "_method_order"])
    frozen["rank_by_RMSE_within_seed_target"] = (
        frozen.groupby(["seed", "target_code"])["RMSE"]
        .rank(method="min")
        .astype(int)
    )
    frozen["selection_scope"] = (
        "frozen_publication_5_outer_seeds; unified_train_range_stabilization"
    )
    frozen = frozen.drop(columns="_method_order")
    frozen.to_csv(OUTPUT / "PV_Unified_Frozen5_Detailed.csv", index=False)

    frozen_summary: list[dict[str, Any]] = []
    for (target, target_code, method), group in frozen.groupby(
        ["target", "target_code", "method"], sort=False
    ):
        frozen_summary.append(
            {
                "dataset": "PV",
                "target": target,
                "target_code": target_code,
                "method": method,
                "n_seeds": 5,
                "seeds": ";".join(map(str, sorted(FROZEN_FIVE))),
                "RMSE_mean": float(group["RMSE"].mean()),
                "RMSE_std": float(group["RMSE"].std(ddof=1)),
                "MAE_mean": float(group["MAE"].mean()),
                "MAE_std": float(group["MAE"].std(ddof=1)),
                "R2_mean": float(group["R2"].mean()),
                "R2_std": float(group["R2"].std(ddof=1)),
                "percent_RMSE_change_vs_direct_mean": float(
                    group["percent_RMSE_change_vs_direct"].mean()
                ),
                "percent_RMSE_change_vs_direct_std": float(
                    group["percent_RMSE_change_vs_direct"].std(ddof=1)
                ),
                "win_count_vs_direct": int(
                    (group["percent_RMSE_change_vs_direct"] < 0).sum()
                ),
                "selection_scope": (
                    "frozen_publication_5_outer_seeds; "
                    "unified_train_range_stabilization"
                ),
            }
        )
    summary = pd.DataFrame(frozen_summary)
    summary["rank_RMSE_1_best"] = (
        summary.groupby("target_code")["RMSE_mean"]
        .rank(method="min")
        .astype(int)
    )
    summary.to_csv(OUTPUT / "PV_Unified_Frozen5_Summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else read_seeds()
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    base_map = selected_base_models()
    frame = pd.read_csv(pv.DEFAULT_OUTPUT)
    if not args.aggregate_only:
        for position, seed in enumerate(seeds, start=1):
            print(f"[PV unified {position}/{len(seeds)}] seed={seed}", flush=True)
            run_seed(
                seed,
                frame,
                inner_cv=args.inner_cv,
                n_jobs=args.n_jobs,
                base_map=base_map,
            )
    aggregate(seeds)
    print(f"Wrote unified PV outputs to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
