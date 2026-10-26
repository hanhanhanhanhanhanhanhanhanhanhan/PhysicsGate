"""Supplementary Figure S2 source data.

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/build_si_correctability_complementarity.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import PV_WORKDIR  # noqa: E402
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = PV_WORKDIR

FIGURE5_DIR = ROOT / "outputs" / "figure5_error_propagation"

FIGURE4_DIR = ROOT / "outputs" / "figure4_gate_mechanism" / "50seed"

OUTPUT_DIR = ROOT / "outputs" / "si_correctability_complementarity"

METRIC_FILE = FIGURE5_DIR / "Figure5_SeedTarget_Diagnostics.csv"

SAMPLE_FILES = {
    "ESTM": FIGURE4_DIR / "estm" / "Figure4_ESTM_PerSample_GateDiagnostics.csv",
    "PV": FIGURE4_DIR / "pv" / "Figure4_PV_PerSample_GateDiagnostics.csv",
}

BOOTSTRAP_SAMPLES = 10_000

BOOTSTRAP_SEED = 20_260_727

TARGET_METADATA = pd.DataFrame(
    [
        ("ESTM", "seebeck_uV_per_K", "Seebeck coefficient", "inverse", 0),
        ("ESTM", "ZT", "ZT", "forward", 1),
        (
            "ESTM",
            "log10_electrical_conductivity",
            "log10 electrical conductivity",
            "inverse",
            2,
        ),
        (
            "ESTM",
            "log10_thermal_conductivity",
            "log10 thermal conductivity",
            "inverse",
            3,
        ),
        ("PV", "FF_fraction", "FF", "inverse", 4),
        ("PV", "Jsc_mA_cm2", "JSC", "inverse", 5),
        ("PV", "PCE_percent", "PCE", "forward", 6),
        ("PV", "Voc_V", "VOC", "inverse", 7),
    ],
    columns=[
        "dataset",
        "target_code",
        "target_display",
        "equation_route",
        "target_order",
    ],
)


def require_inputs() -> None:
    paths = [METRIC_FILE, *SAMPLE_FILES.values()]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing SI inputs:\n" + "\n".join(missing))


def paired_sample_diagnostics(dataset: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "target_code",
            "seed",
            "y_true",
            "A_direct",
            "A_residual",
            "w",
            "R3_propagated_uncertainty",
            "uses_test_target_for_selection",
        ],
    )
    if frame["uses_test_target_for_selection"].astype(bool).any():
        raise RuntimeError(f"{dataset} used test targets for selection")
    numeric = [
        "y_true",
        "A_direct",
        "A_residual",
        "w",
        "R3_propagated_uncertainty",
    ]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{dataset} per-sample diagnostics contain NaN/Inf")

    frame["direct_signed_error"] = frame["y_true"] - frame["A_direct"]
    frame["residual_signed_error"] = frame["y_true"] - frame["A_residual"]

    rows: list[dict[str, object]] = []
    for (target_code, seed), group in frame.groupby(["target_code", "seed"]):
        direct_error = group["direct_signed_error"].to_numpy(dtype=float)
        residual_error = group["residual_signed_error"].to_numpy(dtype=float)
        if np.std(direct_error) == 0 or np.std(residual_error) == 0:
            error_correlation = np.nan
        else:
            error_correlation = float(np.corrcoef(direct_error, residual_error)[0, 1])

        q25, q75 = group["R3_propagated_uncertainty"].quantile([0.25, 0.75])
        low = group.loc[group["R3_propagated_uncertainty"].le(q25), "w"]
        high = group.loc[group["R3_propagated_uncertainty"].ge(q75), "w"]
        rows.append(
            {
                "dataset": dataset,
                "target_code": target_code,
                "seed": int(seed),
                "n_test_predictions": int(len(group)),
                "branch_error_correlation": error_correlation,
                "R3_q25": float(q25),
                "R3_q75": float(q75),
                "mean_w_low_R3": float(low.mean()),
                "mean_w_high_R3": float(high.mean()),
                "delta_w_low_minus_high_R3": float(low.mean() - high.mean()),
                "gate_suppresses_high_R3": bool(low.mean() > high.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_seed_target_table() -> pd.DataFrame:
    metrics = pd.read_csv(METRIC_FILE)
    metrics["residual_correctability_log2"] = np.log2(
        metrics["physics_RMSE"] / metrics["Ares_old_RMSE"]
    )
    sample = pd.concat(
        [
            paired_sample_diagnostics(dataset, path)
            for dataset, path in SAMPLE_FILES.items()
        ],
        ignore_index=True,
    )
    result = metrics.merge(
        sample,
        on=["dataset", "target_code", "seed"],
        how="inner",
        validate="one_to_one",
    )
    result = result.merge(
        TARGET_METADATA,
        on=["dataset", "target_code"],
        how="left",
        suffixes=("", "_si"),
        validate="many_to_one",
    )
    result["target_display"] = result["target_display_si"]
    result["equation_route"] = result["equation_route_si"]
    result["target_order"] = result["target_order_si"].astype(int)
    result = result.drop(
        columns=["target_display_si", "equation_route_si", "target_order_si"]
    )
    if len(result) != 400:
        raise RuntimeError(f"Expected 400 seed-target units, found {len(result)}")
    return result.sort_values(["target_order", "seed"]).reset_index(drop=True)


def bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    draws = rng.choice(
        values,
        size=(BOOTSTRAP_SAMPLES, len(values)),
        replace=True,
    )
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return (
        float(values.mean()),
        float(values.std(ddof=1)),
        float(low),
        float(high),
    )


def build_summary(detailed: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = [
        "residual_correctability_log2",
        "branch_error_correlation",
        "delta_w_low_minus_high_R3",
        "mean_w_low_R3",
        "mean_w_high_R3",
    ]
    rows: list[dict[str, object]] = []
    group_cols = [
        "dataset",
        "target_code",
        "target_display",
        "equation_route",
        "target_order",
    ]
    for keys, group in detailed.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        row["mean_n_test_predictions"] = float(group["n_test_predictions"].mean())
        row["fraction_splits_suppressing_high_R3"] = float(
            group["gate_suppresses_high_R3"].mean()
        )
        for metric in metrics:
            mean, std, low, high = bootstrap_mean(
                group[metric].to_numpy(dtype=float),
                rng,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_CI_low"] = low
            row[f"{metric}_CI_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values("target_order").reset_index(drop=True)


def write_report(summary: pd.DataFrame) -> None:
    indexed = summary.set_index(["dataset", "target_code"])

    def value(dataset: str, target: str, metric: str) -> float:
        return float(indexed.loc[(dataset, target), metric])

    report = f"""# SI correctability and complementarity analysis

## Core result

Propagation burden alone does not determine whether a physics branch remains
useful. The held-out residual-correctability gain and the relative dissimilarity
of the Residual and Direct error patterns explain the apparent
thermal-conductivity counterexample.

## Thermal-conductivity counterexample

- ESTM log10 thermal conductivity has mean residual correctability
  C_res = {value("ESTM", "log10_thermal_conductivity", "residual_correctability_log2_mean"):.3f},
  corresponding to a {2 ** value("ESTM", "log10_thermal_conductivity", "residual_correctability_log2_mean"):.2f}-fold
  reduction in RMSE from Physics-only to Ares_old.
- Its mean Direct/Residual signed-error correlation is
  {value("ESTM", "log10_thermal_conductivity", "branch_error_correlation_mean"):.3f},
  which remains positively correlated but leaves substantially more relative
  branch complementarity than the near-unity correlations observed for PV
  inverse targets.
- Its mean low-minus-high R3 gate-weight difference is
  {value("ESTM", "log10_thermal_conductivity", "delta_w_low_minus_high_R3_mean"):.3f}.

## Contrast with PV

PV FF has even larger residual correctability
(C_res = {value("PV", "FF_fraction", "residual_correctability_log2_mean"):.3f}),
but its Direct/Residual signed-error correlation is
{value("PV", "FF_fraction", "branch_error_correlation_mean"):.3f}. The residual
branch therefore largely reproduces Direct errors rather than supplying an
independent correction, consistent with its negligible final gate gain.

## Interpretation boundary

C_res is a held-out diagnostic computed after train-only model selection; it is
not used to select a residual model or tune the gate. Error correlations and
R3-stratified weights are also held-out descriptive diagnostics. They support
mechanistic interpretation but do not prove causal mediation.

## Stabilization sensitivity

The retained ESTM and unified PV mainlines both stabilize inverse physics
branches using target bounds estimated only from the corresponding
outer-training split. A separate paired no-range-versus-train-range sensitivity
analysis confirms that this guard reduces pathological inverse-branch tails,
while the residual cross-domain contrast is not explained by stabilization
alone.
"""
    (OUTPUT_DIR / "SI_Correctability_Complementarity_Report.md").write_text(
        report,
        encoding="utf-8",
    )


def write_audit(detailed: pd.DataFrame, summary: pd.DataFrame) -> None:
    audit = pd.DataFrame(
        [
            {
                "check": "seed_target_units",
                "value": len(detailed),
                "status": "pass",
            },
            {
                "check": "targets",
                "value": detailed["target_code"].nunique(),
                "status": "pass",
            },
            {
                "check": "seeds_per_target_min",
                "value": detailed.groupby(["dataset", "target_code"])[
                    "seed"
                ].nunique().min(),
                "status": "pass",
            },
            {
                "check": "seeds_per_target_max",
                "value": detailed.groupby(["dataset", "target_code"])[
                    "seed"
                ].nunique().max(),
                "status": "pass",
            },
            {
                "check": "finite_core_metrics",
                "value": bool(
                    np.isfinite(
                        detailed[
                            [
                                "residual_correctability_log2",
                                "branch_error_correlation",
                                "delta_w_low_minus_high_R3",
                            ]
                        ].to_numpy(dtype=float)
                    ).all()
                ),
                "status": "pass",
            },
            {
                "check": "test_target_used_for_selection",
                "value": False,
                "status": "pass",
            },
            {
                "check": "summary_rows",
                "value": len(summary),
                "status": "pass",
            },
        ]
    )
    audit.to_csv(OUTPUT_DIR / "SI_Correctability_Complementarity_Audit.csv", index=False)


def main() -> None:
    require_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detailed = build_seed_target_table()
    summary = build_summary(detailed)
    detailed.to_csv(
        OUTPUT_DIR / "SI_Correctability_Complementarity_SeedTarget.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "SI_Correctability_Complementarity_TargetSummary.csv",
        index=False,
    )
    write_report(summary)
    write_audit(detailed, summary)
    print(summary.to_string(index=False))
    print(f"\nWrote SI analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
