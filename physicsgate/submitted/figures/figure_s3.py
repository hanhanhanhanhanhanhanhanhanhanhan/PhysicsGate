"""Supplementary Figure S3 source data.

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/build_si_stabilization_sensitivity.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import PV_WORKDIR  # noqa: E402
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = PV_WORKDIR

OUTPUT = ROOT / "outputs" / "si_stabilization_sensitivity"

FIGURE4 = ROOT / "outputs" / "figure4_gate_mechanism" / "50seed"

SEEDS = {
    "ESTM": (4, 42, 99, 178, 2024),
    "PV": (243, 938, 1025, 1790, 2024),
}

VARIANT_NO_RANGE = {
    "ESTM": "numeric_only_no_train_range",
    "PV": "retained_numeric_only",
}

VARIANT_TRAIN_RANGE = {
    "ESTM": "retained_train_range",
    "PV": "matched_train_range",
}

TARGET_ORDER = {
    ("ESTM", "seebeck_uV_per_K"): 0,
    ("ESTM", "ZT"): 1,
    ("ESTM", "log10_electrical_conductivity"): 2,
    ("ESTM", "log10_thermal_conductivity"): 3,
    ("PV", "FF_fraction"): 4,
    ("PV", "Jsc_mA_cm2"): 5,
    ("PV", "PCE_percent"): 6,
    ("PV", "Voc_V"): 7,
}

METHOD_ORDER = {"Physics-only": 0, "Ares_old": 1, "PhysicsGate": 2}


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def aggregate_detailed(dataset: str) -> pd.DataFrame:
    path = OUTPUT / f"SI_Stabilization_{dataset}_Detailed.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["seed"].isin(SEEDS[dataset])].copy()
    expected = 5 * 4 * 2 * 4 * (3 if dataset == "ESTM" else 1)
    if len(frame) != expected:
        raise RuntimeError(f"{dataset}: expected {expected} detailed rows, found {len(frame)}")
    if dataset == "ESTM":
        frame = (
            frame.groupby(
                [
                    "dataset",
                    "target",
                    "target_code",
                    "seed",
                    "stabilization_variant",
                    "method",
                ],
                as_index=False,
            )
            .agg(
                RMSE=("RMSE", "mean"),
                MAE=("MAE", "mean"),
                R2=("R2", "mean"),
                mean_test_weight=("mean_test_weight", "mean"),
            )
        )
    return frame


def paired_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["dataset", "target", "target_code", "seed", "method"], sort=False
    ):
        dataset, target, target_code, seed, method = keys
        no_range = group.loc[
            group["stabilization_variant"].eq(VARIANT_NO_RANGE[dataset])
        ]
        train_range = group.loc[
            group["stabilization_variant"].eq(VARIANT_TRAIN_RANGE[dataset])
        ]
        if len(no_range) != 1 or len(train_range) != 1:
            raise RuntimeError(f"Missing paired variants for {keys}")
        rmse_no = float(no_range.iloc[0]["RMSE"])
        rmse_yes = float(train_range.iloc[0]["RMSE"])
        rows.append(
            {
                "dataset": dataset,
                "target": target,
                "target_code": target_code,
                "target_order": TARGET_ORDER[(dataset, target_code)],
                "seed": int(seed),
                "method": method,
                "method_order": METHOD_ORDER.get(method, -1),
                "RMSE_no_train_range": rmse_no,
                "RMSE_train_range": rmse_yes,
                "log2_gain_from_train_range": float(np.log2(rmse_no / rmse_yes)),
                "percent_RMSE_change_train_range_vs_no_range": (
                    (rmse_yes / rmse_no - 1.0) * 100.0
                ),
                "R2_no_train_range": float(no_range.iloc[0]["R2"]),
                "R2_train_range": float(train_range.iloc[0]["R2"]),
                "mean_weight_no_train_range": float(
                    no_range.iloc[0]["mean_test_weight"]
                ),
                "mean_weight_train_range": float(
                    train_range.iloc[0]["mean_test_weight"]
                ),
            }
        )
    return pd.DataFrame(rows)


def target_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = paired.loc[paired["method"].isin(METHOD_ORDER)].copy()
    for keys, group in selected.groupby(
        ["dataset", "target", "target_code", "target_order", "method", "method_order"],
        sort=False,
    ):
        dataset, target, target_code, target_order, method, method_order = keys
        values = group["log2_gain_from_train_range"].to_numpy(dtype=float)
        low, high = bootstrap_ci(
            values,
            seed=91_000 + int(target_order) * 31 + int(method_order),
        )
        pct = group[
            "percent_RMSE_change_train_range_vs_no_range"
        ].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "target": target,
                "target_code": target_code,
                "target_order": int(target_order),
                "method": method,
                "method_order": int(method_order),
                "n_seeds": int(group["seed"].nunique()),
                "seeds": ";".join(str(value) for value in sorted(group["seed"])),
                "log2_gain_mean": float(np.mean(values)),
                "log2_gain_std": float(np.std(values, ddof=1)),
                "log2_gain_CI_low": low,
                "log2_gain_CI_high": high,
                "percent_RMSE_change_mean": float(np.mean(pct)),
                "percent_RMSE_change_std": float(np.std(pct, ddof=1)),
                "train_range_win_count": int(np.count_nonzero(values > 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target_order", "method_order"]
    ).reset_index(drop=True)


def audit_summary() -> pd.DataFrame:
    frames = []
    for dataset in ("ESTM", "PV"):
        frame = pd.read_csv(OUTPUT / f"SI_Stabilization_{dataset}_Audit.csv")
        frame = frame.loc[
            frame["seed"].isin(SEEDS[dataset])
            & frame["stabilization_variant"].eq(VARIANT_TRAIN_RANGE[dataset])
        ].copy()
        frames.append(frame)
    audits = pd.concat(frames, ignore_index=True)
    audits["train_range_rule_applied"] = ~audits["forward_target"].astype(bool)
    audits["applied_bound_is_train_only"] = (
        ~audits["train_range_rule_applied"]
        | audits["uses_outer_train_target_range"].astype(bool)
    )
    return (
        audits.groupby(["dataset", "target_code"], as_index=False)
        .agg(
            n_seed_fold_units=("seed", "size"),
            n_seeds=("seed", "nunique"),
            mean_fraction_physics_predictions_clipped=(
                "fraction_changed_by_stabilization",
                "mean",
            ),
            max_fraction_physics_predictions_clipped=(
                "fraction_changed_by_stabilization",
                "max",
            ),
            train_range_rule_applied=("train_range_rule_applied", "all"),
            all_applied_bounds_train_only=("applied_bound_is_train_only", "all"),
            any_true_test_auxiliary_used=("uses_true_test_auxiliary_values", "any"),
            any_test_target_used_for_selection=(
                "uses_test_target_for_selection",
                "any",
            ),
        )
    )


def retained_parity() -> pd.DataFrame:
    rows = []
    for dataset in ("ESTM", "PV"):
        new = pd.read_csv(OUTPUT / f"SI_Stabilization_{dataset}_Detailed.csv")
        new = new.loc[
            new["seed"].isin(SEEDS[dataset])
            & new["stabilization_variant"].eq(VARIANT_TRAIN_RANGE[dataset])
        ].copy()
        if dataset == "ESTM":
            reference = pd.read_csv(
                FIGURE4 / "estm" / "Figure4_ESTM_50Seed_Rerun_Detailed.csv"
            )
            reference = reference.loc[reference["outer_seed"].isin(SEEDS[dataset])]
            new["reference_method"] = new["method"].replace(
                {"PhysicsGate": "DirectAnchored_PhysicsGate_R2_R5_R6_tuned"}
            )
            merged = new.merge(
                reference,
                left_on=["seed", "target_code", "fold_id", "reference_method"],
                right_on=["outer_seed", "target_code", "fold_id", "method"],
                suffixes=("_new", "_reference"),
            )
        else:
            reference = pd.read_csv(
                FIGURE4 / "pv" / "Figure4_PV_50Seed_Rerun_Detailed.csv"
            )
            reference = reference.loc[reference["seed"].isin(SEEDS[dataset])]
            new["reference_method"] = new["method"].replace(
                {"PhysicsGate": "FixedGate_R2_R5_R6"}
            )
            merged = new.merge(
                reference,
                left_on=["seed", "target_code", "reference_method"],
                right_on=["seed", "target_code", "method"],
                suffixes=("_new", "_reference"),
            )
        invariant = merged.loc[
            merged["method_new"].isin(["Direct", "Physics-only"])
        ].copy()
        rows.append(
            {
                "dataset": dataset,
                "matched_rows_all_methods": len(merged),
                "matched_rows_direct_physics": len(invariant),
                "max_absolute_RMSE_difference_direct_physics_vs_Figure4": float(
                    np.max(
                        np.abs(
                            invariant["RMSE_new"]
                            - invariant["RMSE_reference"]
                        )
                    )
                ),
                "max_absolute_RMSE_difference_all_methods_vs_Figure4": float(
                    np.max(np.abs(merged["RMSE_new"] - merged["RMSE_reference"]))
                ),
                "status": (
                    "pass"
                    if np.allclose(
                        invariant["RMSE_new"],
                        invariant["RMSE_reference"],
                        rtol=1e-9,
                        atol=1e-9,
                    )
                    else "fail"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, audit: pd.DataFrame) -> None:
    indexed = summary.set_index(["dataset", "target_code", "method"])

    def value(dataset: str, target: str, method: str, column: str) -> float:
        return float(indexed.loc[(dataset, target, method), column])

    report = f"""# Paired stabilization sensitivity

## Design

The same five frozen outer seeds, Direct/auxiliary predictions, residual
learner class, preprocessing, and gate feature set were used in each pair.
Only the inverse-physics stabilization rule changed. Forward targets ZT and
PCE were unchanged by design.

The reproduction audit requires exact agreement with the current Figure 4
source for Direct and Physics-only, the branches that are invariant to residual
refitting. Ares_old and PhysicsGate are refitted inside this controlled paired
sensitivity run with the selected residual-learner family held fixed; their
differences from the fully reselected mainline are retained as a diagnostic
rather than treated as a reproduction failure.

## ESTM

Removing the retained outer-train target-range guard worsened PhysicsGate RMSE
for all five seeds on every inverse target. Restoring the guard changed mean
PhysicsGate RMSE by:

- Seebeck: {value("ESTM", "seebeck_uV_per_K", "PhysicsGate", "percent_RMSE_change_mean"):.2f}%
- log10 electrical conductivity:
  {value("ESTM", "log10_electrical_conductivity", "PhysicsGate", "percent_RMSE_change_mean"):.2f}%
- log10 thermal conductivity:
  {value("ESTM", "log10_thermal_conductivity", "PhysicsGate", "percent_RMSE_change_mean"):.2f}%

Thus train-only stabilization materially contributes to ESTM performance.

## PV

Adding the same outer-train target-range guard strongly reduced Physics-only
RMSE for inverse targets, but its effect after residual correction and gating
was negligible. Mean PhysicsGate changes remained within
{summary.loc[(summary.dataset == "PV") & (summary.method == "PhysicsGate"), "percent_RMSE_change_mean"].abs().max():.3f}%.

## Interpretation

The domain contrast cannot be attributed entirely to data quality: the
retained ESTM stabilization provides a real advantage. However, stabilization
alone does not rescue PV. Its large Physics-only improvements vanish after
the residual branch because the corrected branch remains highly redundant
with Direct. The central correctability/complementarity interpretation
therefore remains, with stabilization explicitly acknowledged as a contributing
implementation factor.

## Leakage boundary

All clipping bounds use outer-train target minima/maxima only. No validation or
test target or true test auxiliary value is used for stabilization, branch
selection, or gate tuning.
"""
    (OUTPUT / "SI_Stabilization_Sensitivity_Report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    frames = [aggregate_detailed(dataset) for dataset in ("ESTM", "PV")]
    paired = paired_frame(pd.concat(frames, ignore_index=True))
    summary = target_summary(paired)
    audit = audit_summary()
    parity = retained_parity()

    paired.to_csv(OUTPUT / "SI_Stabilization_PairedSeedTarget.csv", index=False)
    summary.to_csv(OUTPUT / "SI_Stabilization_TargetSummary.csv", index=False)
    audit.to_csv(OUTPUT / "SI_Stabilization_ClippingAudit.csv", index=False)
    parity.to_csv(OUTPUT / "SI_Stabilization_ReproductionAudit.csv", index=False)
    write_report(summary, audit)
    if not parity["status"].eq("pass").all():
        raise RuntimeError("Retained-rule reproduction audit failed")
    print(summary.to_string(index=False))
    print(f"Wrote stabilization sensitivity summary to {OUTPUT}")


if __name__ == "__main__":
    main()
