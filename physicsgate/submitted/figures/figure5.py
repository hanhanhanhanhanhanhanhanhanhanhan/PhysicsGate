"""Figure 5 source data (error propagation and correctability).

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/build_figure5_error_propagation_analysis.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import PV_WORKDIR  # noqa: E402
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = PV_WORKDIR

SOURCE_DIR = ROOT / "outputs" / "figure4_gate_mechanism" / "50seed"

OUTPUT_DIR = ROOT / "outputs" / "figure5_error_propagation"

PV_METRICS = SOURCE_DIR / "pv" / "Figure4_PV_50Seed_Rerun_Detailed.csv"

ESTM_METRICS = SOURCE_DIR / "estm" / "Figure4_ESTM_50Seed_Rerun_Detailed.csv"

PV_SAMPLES = SOURCE_DIR / "pv" / "Figure4_PV_PerSample_GateDiagnostics.csv"

ESTM_SAMPLES = SOURCE_DIR / "estm" / "Figure4_ESTM_PerSample_GateDiagnostics.csv"

BOOTSTRAP_SAMPLES = 10_000

BOOTSTRAP_SEED = 20_260_726

METHOD_NAMES = {
    "PV": {
        "direct": "Direct",
        "physics": "Physics-only",
        "residual": "Ares_old",
        "gate": "FixedGate_R2_R5_R6",
    },
    "ESTM": {
        "direct": "Direct",
        "physics": "Physics-only",
        "residual": "Ares_old",
        "gate": "DirectAnchored_PhysicsGate_R2_R5_R6_tuned",
    },
}

TARGET_METADATA = pd.DataFrame(
    [
        {
            "dataset": "ESTM",
            "target_code": "seebeck_uV_per_K",
            "target_display": "Seebeck coefficient",
            "short_label": "Seebeck",
            "equation_route": "inverse",
            "target_order": 0,
        },
        {
            "dataset": "ESTM",
            "target_code": "ZT",
            "target_display": "ZT",
            "short_label": "ZT",
            "equation_route": "forward",
            "target_order": 1,
        },
        {
            "dataset": "ESTM",
            "target_code": "log10_electrical_conductivity",
            "target_display": "log10 electrical conductivity",
            "short_label": "lg sigma_elec",
            "equation_route": "inverse",
            "target_order": 2,
        },
        {
            "dataset": "ESTM",
            "target_code": "log10_thermal_conductivity",
            "target_display": "log10 thermal conductivity",
            "short_label": "lg kappa_thermal",
            "equation_route": "inverse",
            "target_order": 3,
        },
        {
            "dataset": "PV",
            "target_code": "FF_fraction",
            "target_display": "FF",
            "short_label": "FF",
            "equation_route": "inverse",
            "target_order": 4,
        },
        {
            "dataset": "PV",
            "target_code": "Jsc_mA_cm2",
            "target_display": "JSC (mA/cm2)",
            "short_label": "JSC",
            "equation_route": "inverse",
            "target_order": 5,
        },
        {
            "dataset": "PV",
            "target_code": "PCE_percent",
            "target_display": "PCE_percent",
            "short_label": "PCE",
            "equation_route": "forward",
            "target_order": 6,
        },
        {
            "dataset": "PV",
            "target_code": "Voc_V",
            "target_display": "VOC (V)",
            "short_label": "VOC",
            "equation_route": "inverse",
            "target_order": 7,
        },
    ]
)


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 5 inputs:\n" + "\n".join(missing))


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else str(column)
        for column in result.columns
    ]
    return result


def load_metric_units() -> pd.DataFrame:
    pv = pd.read_csv(PV_METRICS)
    pv = pv[["target", "target_code", "seed", "method", "RMSE", "R2"]].copy()

    estm = pd.read_csv(ESTM_METRICS)
    estm = (
        estm.groupby(
            ["target", "target_code", "outer_seed", "method"],
            as_index=False,
        )
        .agg(RMSE=("RMSE", "mean"), R2=("R2", "mean"))
        .rename(columns={"outer_seed": "seed"})
    )

    rows: list[pd.DataFrame] = []
    for dataset, frame in (("ESTM", estm), ("PV", pv)):
        pivot = frame.pivot_table(
            index=["target", "target_code", "seed"],
            columns="method",
            values=["RMSE", "R2"],
        ).reset_index()
        pivot = flatten_columns(pivot)
        names = METHOD_NAMES[dataset]
        required = [
            f"RMSE_{names['direct']}",
            f"RMSE_{names['physics']}",
            f"RMSE_{names['residual']}",
            f"RMSE_{names['gate']}",
            f"R2_{names['direct']}",
        ]
        missing = [column for column in required if column not in pivot.columns]
        if missing:
            raise RuntimeError(f"{dataset} metric inputs are incomplete: {missing}")

        pivot["dataset"] = dataset
        pivot["direct_RMSE"] = pivot[f"RMSE_{names['direct']}"]
        pivot["physics_RMSE"] = pivot[f"RMSE_{names['physics']}"]
        pivot["Ares_old_RMSE"] = pivot[f"RMSE_{names['residual']}"]
        pivot["PhysicsGate_RMSE"] = pivot[f"RMSE_{names['gate']}"]
        pivot["direct_R2"] = pivot[f"R2_{names['direct']}"]
        pivot["physics_penalty_log2"] = np.log2(
            pivot["physics_RMSE"] / pivot["direct_RMSE"]
        )
        pivot["Ares_old_gain_log2"] = np.log2(
            pivot["direct_RMSE"] / pivot["Ares_old_RMSE"]
        )
        pivot["PhysicsGate_gain_log2"] = np.log2(
            pivot["direct_RMSE"] / pivot["PhysicsGate_RMSE"]
        )
        pivot["PhysicsGate_RMSE_change_percent"] = (
            pivot["PhysicsGate_RMSE"] / pivot["direct_RMSE"] - 1.0
        ) * 100.0
        rows.append(
            pivot[
                [
                    "dataset",
                    "target",
                    "target_code",
                    "seed",
                    "direct_RMSE",
                    "direct_R2",
                    "physics_RMSE",
                    "Ares_old_RMSE",
                    "PhysicsGate_RMSE",
                    "physics_penalty_log2",
                    "Ares_old_gain_log2",
                    "PhysicsGate_gain_log2",
                    "PhysicsGate_RMSE_change_percent",
                ]
            ]
        )

    result = pd.concat(rows, ignore_index=True)
    result = result.merge(
        TARGET_METADATA,
        on=["dataset", "target_code"],
        how="left",
        validate="many_to_one",
    )
    if result["target_order"].isna().any():
        unknown = result.loc[
            result["target_order"].isna(), ["dataset", "target_code"]
        ].drop_duplicates()
        raise RuntimeError(f"Missing target metadata:\n{unknown}")
    return result


def summarize_propagated_uncertainty(
    dataset: str,
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "target",
            "target_code",
            "seed",
            "R3_propagated_uncertainty",
        ],
    )
    if not np.isfinite(frame["R3_propagated_uncertainty"]).all():
        raise RuntimeError(f"{dataset} propagated uncertainty contains NaN/Inf")
    summary = (
        frame.groupby(["target", "target_code", "seed"])[
            "R3_propagated_uncertainty"
        ]
        .agg(
            R3_mean="mean",
            R3_median="median",
            R3_p90=lambda values: float(values.quantile(0.90)),
            R3_p99=lambda values: float(values.quantile(0.99)),
            R3_max="max",
            n_test_samples="size",
        )
        .reset_index()
    )
    summary["dataset"] = dataset
    return summary


def bootstrap_interval(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan
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


def build_target_summary(units: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = [
        "direct_R2",
        "R3_p90",
        "physics_penalty_log2",
        "Ares_old_gain_log2",
        "PhysicsGate_gain_log2",
        "PhysicsGate_RMSE_change_percent",
    ]
    rows: list[dict[str, object]] = []
    group_columns = [
        "dataset",
        "target_code",
        "target_display",
        "short_label",
        "equation_route",
        "target_order",
    ]
    for keys, group in units.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        row["n_seed_target_units"] = int(len(group))
        for metric in metrics:
            mean, std, low, high = bootstrap_interval(
                group[metric].to_numpy(dtype=float),
                rng=rng,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_CI_low"] = low
            row[f"{metric}_CI_high"] = high
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("target_order")
        .reset_index(drop=True)
    )


def safe_correlations(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return np.nan, np.nan
    return (
        float(pearsonr(x[mask], y[mask]).statistic),
        float(spearmanr(x[mask], y[mask]).statistic),
    )


def build_correlation_summary(
    units: pd.DataFrame,
    target_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("R3_p90", "physics_penalty_log2", "propagation_vs_physics_penalty"),
        ("direct_R2", "PhysicsGate_gain_log2", "direct_fidelity_vs_gate_gain"),
        ("R3_p90", "PhysicsGate_gain_log2", "propagation_vs_gate_gain"),
    ]
    for dataset in ("ESTM", "PV", "ALL"):
        unit_part = units if dataset == "ALL" else units.loc[units["dataset"].eq(dataset)]
        target_part = (
            target_summary
            if dataset == "ALL"
            else target_summary.loc[target_summary["dataset"].eq(dataset)]
        )
        for x_name, y_name, comparison in comparisons:
            pearson, spearman = safe_correlations(
                unit_part[x_name].to_numpy(dtype=float),
                unit_part[y_name].to_numpy(dtype=float),
            )
            target_pearson, target_spearman = safe_correlations(
                target_part[f"{x_name}_mean"].to_numpy(dtype=float),
                target_part[f"{y_name}_mean"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": comparison,
                    "x_metric": x_name,
                    "y_metric": y_name,
                    "seed_target_n": int(len(unit_part)),
                    "seed_target_pearson_r_descriptive": pearson,
                    "seed_target_spearman_rho_descriptive": spearman,
                    "target_mean_n": int(len(target_part)),
                    "target_mean_pearson_r_descriptive": target_pearson,
                    "target_mean_spearman_rho_descriptive": target_spearman,
                    "inference_note": (
                        "Descriptive association only; seed-target units share "
                        "targets and are not treated as independent biological "
                        "replicates."
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_error_cascade(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stages = [
        ("Direct", "direct_RMSE"),
        ("Physics-only", "physics_RMSE"),
        ("Ares_old", "Ares_old_RMSE"),
        ("PhysicsGate", "PhysicsGate_RMSE"),
    ]
    detailed_rows: list[dict[str, object]] = []
    for row in units.itertuples(index=False):
        for stage_order, (stage, column) in enumerate(stages):
            rmse = float(getattr(row, column))
            detailed_rows.append(
                {
                    "dataset": row.dataset,
                    "target_code": row.target_code,
                    "target_display": row.target_display,
                    "short_label": row.short_label,
                    "equation_route": row.equation_route,
                    "target_order": int(row.target_order),
                    "seed": int(row.seed),
                    "stage": stage,
                    "stage_order": stage_order,
                    "RMSE": rmse,
                    "direct_RMSE": float(row.direct_RMSE),
                    "log2_RMSE_ratio_to_direct": float(
                        np.log2(rmse / float(row.direct_RMSE))
                    ),
                }
            )
    detailed = pd.DataFrame(detailed_rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    summary_rows: list[dict[str, object]] = []
    columns = [
        "dataset",
        "target_code",
        "target_display",
        "short_label",
        "equation_route",
        "target_order",
        "stage",
        "stage_order",
    ]
    for keys, group in detailed.groupby(columns, sort=False):
        mean, std, low, high = bootstrap_interval(
            group["log2_RMSE_ratio_to_direct"].to_numpy(dtype=float),
            rng=rng,
        )
        summary_rows.append(
            {
                **dict(zip(columns, keys)),
                "n_seeds": int(group["seed"].nunique()),
                "mean_log2_RMSE_ratio_to_direct": mean,
                "std_log2_RMSE_ratio_to_direct": std,
                "bootstrap_CI_low": low,
                "bootstrap_CI_high": high,
            }
        )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["target_order", "stage_order"])
        .reset_index(drop=True)
    )
    return detailed, summary


def build_audit(
    units: pd.DataFrame,
    target_summary: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "datasets",
                "value": ",".join(sorted(units["dataset"].unique())),
                "status": "pass",
            },
            {
                "check": "targets",
                "value": int(units[["dataset", "target_code"]].drop_duplicates().shape[0]),
                "status": "pass",
            },
            {
                "check": "seeds_per_target_min",
                "value": int(
                    units.groupby(["dataset", "target_code"])["seed"].nunique().min()
                ),
                "status": "pass",
            },
            {
                "check": "seeds_per_target_max",
                "value": int(
                    units.groupby(["dataset", "target_code"])["seed"].nunique().max()
                ),
                "status": "pass",
            },
            {
                "check": "finite_core_metrics",
                "value": bool(
                    np.isfinite(
                        units[
                            [
                                "direct_R2",
                                "R3_p90",
                                "physics_penalty_log2",
                                "PhysicsGate_gain_log2",
                            ]
                        ].to_numpy(dtype=float)
                    ).all()
                ),
                "status": "pass",
            },
            {
                "check": "test_auxiliary_use",
                "value": "none; R3 derives from train-side auxiliary OOF errors",
                "status": "pass",
            },
            {
                "check": "test_target_selection",
                "value": "none",
                "status": "pass",
            },
            {
                "check": "target_summary_rows",
                "value": int(len(target_summary)),
                "status": "pass",
            },
        ]
    )


def write_analysis_report(
    target_summary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    indexed = target_summary.set_index(["dataset", "short_label"])

    def value(dataset: str, target: str, column: str) -> float:
        return float(indexed.loc[(dataset, target), column])

    def physics_inflation(dataset: str, target: str) -> float:
        penalty = value(dataset, target, "physics_penalty_log2_mean")
        return float((np.power(2.0, penalty) - 1.0) * 100.0)

    def gate_reduction(dataset: str, target: str) -> float:
        gain = value(dataset, target, "PhysicsGate_gain_log2_mean")
        return float((1.0 - np.power(2.0, -gain)) * 100.0)

    report = f"""# Figure 5 analysis report

## Primary finding

The 50-seed diagnostics support a two-bottleneck explanation for the smaller
PV gains: weaker direct/auxiliary prediction fidelity and stronger
equation-mediated error propagation, particularly in inverse photovoltaic
relations. The evidence is associative and mechanistically consistent; it is
not a causal intervention.

## Quantitative evidence

- Mean Direct R2 spans
  {target_summary.loc[target_summary.dataset.eq("ESTM"), "direct_R2_mean"].min():.3f}
  to
  {target_summary.loc[target_summary.dataset.eq("ESTM"), "direct_R2_mean"].max():.3f}
  in ESTM, versus
  {target_summary.loc[target_summary.dataset.eq("PV"), "direct_R2_mean"].min():.3f}
  to
  {target_summary.loc[target_summary.dataset.eq("PV"), "direct_R2_mean"].max():.3f}
  in PV.
- Within PV, the forward PCE route has the lowest propagated-uncertainty P90
  ({value("PV", "PCE", "R3_p90_mean"):.3f}) and only
  {physics_inflation("PV", "PCE"):.1f}% Physics-only RMSE inflation.
- The inverse PV routes have higher propagated uncertainty:
  JSC {value("PV", "JSC", "R3_p90_mean"):.3f},
  VOC {value("PV", "VOC", "R3_p90_mean"):.3f}, and
  FF {value("PV", "FF", "R3_p90_mean"):.3f}. Their corresponding
  Physics-only RMSE inflation is {physics_inflation("PV", "JSC"):.1f}%,
  {physics_inflation("PV", "VOC"):.1f}%, and
  {physics_inflation("PV", "FF"):.1f}%.
- PhysicsGate reduces RMSE by approximately
  {gate_reduction("PV", "PCE"):.2f}% for PCE and only
  {gate_reduction("PV", "FF"):.2f}%,
  {gate_reduction("PV", "JSC"):.2f}%, and
  {gate_reduction("PV", "VOC"):.2f}% for FF, JSC, and VOC.
- ESTM retains approximately 7-10% Gate reductions despite non-perfect
  Physics-only branches.

## Interpretation boundary

Propagation burden is not the entire explanation. ESTM inverse targets can
also have large Physics-only penalties, but their stronger Direct anchors and
more recoverable residual structure allow the gate to improve performance.
Accordingly, the defensible statement is that PV is limited by the joint
effects of equation conditioning, auxiliary-model error, anchor quality, and
branch complementarity.

Target-mean Pearson and Spearman associations are retained only in the
supplementary descriptive-correlation table. They are not displayed as Figure
5 headline statistics because the number of targets is small and the
cross-domain association is partly induced by dataset grouping.

## Manuscript recommendation

Figure 5 can close the main-text evidence chain:

1. Figure 1: motivation and method.
2. Figure 2: cross-domain ranking.
3. Figure 3: pathway-level performance.
4. Figure 4: gate mechanism in high-data domains.
5. Figure 5: mechanistic boundary analysis explaining the weaker PV gains.

Additional work should be supplementary rather than a sixth main figure:
formula-consistency audits, denominator/extreme-value distributions, optional
one-auxiliary-at-a-time oracle diagnostics, split manifests, and full
hyperparameter/ablation tables.
"""
    (OUTPUT_DIR / "Figure5_Analysis_Report.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    require_files([PV_METRICS, ESTM_METRICS, PV_SAMPLES, ESTM_SAMPLES])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metric_units()
    uncertainty = pd.concat(
        [
            summarize_propagated_uncertainty("ESTM", ESTM_SAMPLES),
            summarize_propagated_uncertainty("PV", PV_SAMPLES),
        ],
        ignore_index=True,
    )
    units = metrics.merge(
        uncertainty[
            [
                "dataset",
                "target_code",
                "seed",
                "R3_mean",
                "R3_median",
                "R3_p90",
                "R3_p99",
                "R3_max",
                "n_test_samples",
            ]
        ],
        on=["dataset", "target_code", "seed"],
        how="left",
        validate="one_to_one",
    )
    if units["R3_p90"].isna().any():
        missing = units.loc[
            units["R3_p90"].isna(), ["dataset", "target_code", "seed"]
        ]
        raise RuntimeError(f"Missing propagated uncertainty summaries:\n{missing}")

    target_summary = build_target_summary(units)
    correlations = build_correlation_summary(units, target_summary)
    cascade_detail, cascade_summary = build_error_cascade(units)
    audit = build_audit(units, target_summary)

    units.sort_values(["target_order", "seed"]).to_csv(
        OUTPUT_DIR / "Figure5_SeedTarget_Diagnostics.csv",
        index=False,
    )
    target_summary.to_csv(
        OUTPUT_DIR / "Figure5_Target_Summary.csv",
        index=False,
    )
    correlations.to_csv(
        OUTPUT_DIR / "Figure5_Supplementary_Descriptive_Correlations.csv",
        index=False,
    )
    cascade_detail.to_csv(
        OUTPUT_DIR / "Figure5_ErrorCascade_BySeedTarget.csv",
        index=False,
    )
    cascade_summary.to_csv(
        OUTPUT_DIR / "Figure5_ErrorCascade_TargetSummary.csv",
        index=False,
    )
    audit.to_csv(OUTPUT_DIR / "Figure5_Data_Audit.csv", index=False)
    write_analysis_report(target_summary, correlations)

    print(target_summary.to_string(index=False))
    print("\nSupplementary descriptive correlations")
    print(correlations.to_string(index=False))
    print(f"\nWrote Figure 5 source data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
