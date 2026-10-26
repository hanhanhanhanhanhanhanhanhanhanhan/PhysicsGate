"""Gate-weight, calibration, ablation and fixed-blend tables.

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/build_figure4_mechanism_analysis.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import PV_WORKDIR  # noqa: E402
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = PV_WORKDIR

BASE = ROOT / "outputs" / "figure4_gate_mechanism"

SAMPLE_FILES = {
    "SSE": BASE / "sse_rerun" / "Figure4_SSE_PerSample_GateDiagnostics.csv",
    "ESTM": BASE / "estm_rerun" / "Figure4_ESTM_PerSample_GateDiagnostics.csv",
    "PV": BASE / "pv_rerun" / "Figure4_PV_PerSample_GateDiagnostics.csv",
    "LMB": BASE / "lmb_rerun" / "Figure4_LMB_PerSample_GateDiagnostics.csv",
}

ABLATION_FILES = {
    "SSE": BASE / "sse_rerun" / "Figure4_SSE_Reliability_Ablation.csv",
    "ESTM": BASE / "estm_rerun" / "Figure4_ESTM_Reliability_Ablation.csv",
    "PV": BASE / "pv_rerun" / "Figure4_PV_Reliability_Ablation.csv",
    "LMB": BASE / "lmb_rerun" / "Figure4_LMB_Reliability_Ablation.csv",
}

BLEND_FILES = {
    "SSE": BASE / "sse_rerun" / "Figure4_SSE_FixedBlend_Comparison.csv",
    "ESTM": BASE / "estm_rerun" / "Figure4_ESTM_FixedBlend_Comparison.csv",
    "PV": BASE / "pv_rerun" / "Figure4_PV_FixedBlend_Comparison.csv",
    "LMB": BASE / "lmb_rerun" / "Figure4_LMB_FixedBlend_Comparison.csv",
}

TARGET_DISPLAY = {
    ("SSE", "Ea"): r"$E_a$",
    ("SSE", "ln_sigma0"): r"$\ln \sigma_0$",
    ("SSE", "ln_sigma_T"): r"$\ln \sigma_T$",
    ("ESTM", "Seebeck coefficient (uV/K)"): "Seebeck coefficient",
    ("ESTM", "ZT"): r"$ZT$",
    ("ESTM", "log10 electrical conductivity"): r"$\lg \sigma_{\rm elec}$",
    ("ESTM", "log10 thermal conductivity"): r"$\lg \kappa_{\rm thermal}$",
    ("PV", "VOC (V)"): r"$V_{\rm OC}$",
    ("PV", "JSC (mA/cm2)"): r"$J_{\rm SC}$",
    ("PV", "FF"): "FF",
    ("PV", "PCE_percent"): "PCE",
    ("PV", "PCE (%)"): "PCE",
    ("LMB", "VC-C"): "Discharge capacity",
    ("LMB", "VC-ED"): "Energy density (VC route)",
    ("LMB", "VC-V"): "Discharge voltage",
    ("LMB", "DE-DE"): "Discharge energy",
    ("LMB", "DE-ED"): "Energy density (DE route)",
}

TARGET_ORDER = [
    ("SSE", r"$E_a$"),
    ("SSE", r"$\ln \sigma_0$"),
    ("SSE", r"$\ln \sigma_T$"),
    ("ESTM", "Seebeck coefficient"),
    ("ESTM", r"$ZT$"),
    ("ESTM", r"$\lg \sigma_{\rm elec}$"),
    ("ESTM", r"$\lg \kappa_{\rm thermal}$"),
    ("PV", "FF"),
    ("PV", r"$J_{\rm SC}$"),
    ("PV", "PCE"),
    ("PV", r"$V_{\rm OC}$"),
    ("LMB", "Discharge energy"),
    ("LMB", "Discharge capacity"),
    ("LMB", "Energy density (VC route)"),
    ("LMB", "Energy density (DE route)"),
    ("LMB", "Discharge voltage"),
]

ORDER_MAP = {pair: index for index, pair in enumerate(TARGET_ORDER)}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def add_target_identity(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    result = frame.copy()
    result["dataset"] = dataset
    if dataset == "LMB":
        result["target_key"] = result["route_target"].astype(str)
    else:
        result["target_key"] = result["target"].astype(str)
    result["target_display"] = [
        TARGET_DISPLAY.get((dataset, key), key) for key in result["target_key"]
    ]
    result["target_id"] = dataset + " | " + result["target_display"]
    result["target_order"] = [
        ORDER_MAP[(dataset, display)] for display in result["target_display"]
    ]
    return result


def pool_estm_fold_rmse(
    frame: pd.DataFrame, group_columns: list[str], value_column: str
) -> pd.DataFrame:
    if "fold_id" not in frame.columns:
        return frame
    result_rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=False, dropna=False):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys_tuple))
        n_test = group["n_test"].to_numpy(dtype=float)
        values = group[value_column].to_numpy(dtype=float)
        row[value_column] = float(np.sqrt(np.sum(n_test * values**2) / np.sum(n_test)))
        row["n_test"] = int(np.sum(n_test))
        if "direct_test_RMSE" in group:
            direct = group["direct_test_RMSE"].to_numpy(dtype=float)
            row["direct_test_RMSE"] = float(
                np.sqrt(np.sum(n_test * direct**2) / np.sum(n_test))
            )
        if "selected_gate_alpha" in group:
            row["selected_gate_alpha"] = float(
                np.average(group["selected_gate_alpha"], weights=n_test)
            )
        if "mean_test_weight" in group:
            row["mean_test_weight"] = float(
                np.average(group["mean_test_weight"], weights=n_test)
            )
        result_rows.append(row)
    return pd.DataFrame(result_rows)


def load_samples() -> pd.DataFrame:
    frames = [
        add_target_identity(pd.read_csv(path), dataset)
        for dataset, path in SAMPLE_FILES.items()
    ]
    samples = pd.concat(frames, ignore_index=True, sort=False)
    samples["direct_abs_error"] = np.abs(samples["y_true"] - samples["A_direct"])
    samples["residual_abs_error"] = np.abs(samples["y_true"] - samples["A_residual"])
    samples["gate_abs_error"] = np.abs(samples["y_true"] - samples["A_final"])
    samples["residual_better"] = (
        samples["residual_abs_error"] < samples["direct_abs_error"]
    )
    samples["branch_advantage"] = (
        samples["direct_abs_error"] - samples["residual_abs_error"]
    )
    delta = samples["A_residual"] - samples["A_direct"]
    usable = np.abs(delta) > np.finfo(float).eps
    samples["oracle_weight"] = 0.0
    samples.loc[usable, "oracle_weight"] = np.clip(
        (samples.loc[usable, "y_true"] - samples.loc[usable, "A_direct"])
        / delta.loc[usable],
        0.0,
        1.0,
    )
    samples["oracle_sample_weight"] = delta**2
    return samples


def weight_target_summary(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_rows: list[dict[str, object]] = []
    group_cols = [
        "dataset",
        "target_id",
        "target_display",
        "target_order",
        "seed",
    ]
    for keys, group in samples.groupby(group_cols, sort=False):
        values = group["w"].to_numpy(dtype=float)
        advantage = group["branch_advantage"].to_numpy(dtype=float)
        oracle_values = group["oracle_weight"].to_numpy(dtype=float)
        oracle_sample_weight = group["oracle_sample_weight"].to_numpy(dtype=float)
        if float(np.sum(oracle_sample_weight)) > np.finfo(float).eps:
            weighted_predicted_weight = float(
                np.average(values, weights=oracle_sample_weight)
            )
            weighted_oracle_weight = float(
                np.average(oracle_values, weights=oracle_sample_weight)
            )
        else:
            weighted_predicted_weight = float(np.mean(values))
            weighted_oracle_weight = float(np.mean(oracle_values))
        rho = spearmanr(values, advantage, nan_policy="omit").statistic
        unit_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_test": len(group),
                "mean_weight": float(np.mean(values)),
                "median_weight": float(np.median(values)),
                "weight_q25": float(np.quantile(values, 0.25)),
                "weight_q75": float(np.quantile(values, 0.75)),
                "delta2_weighted_predicted_weight": weighted_predicted_weight,
                "delta2_weighted_oracle_weight": weighted_oracle_weight,
                "residual_win_fraction": float(np.mean(group["residual_better"])),
                "spearman_weight_vs_residual_advantage": float(rho),
            }
        )
    units = pd.DataFrame(unit_rows)
    summary = (
        units.groupby(
            ["dataset", "target_id", "target_display", "target_order"], as_index=False
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mean_weight=("mean_weight", "mean"),
            std_weight=("mean_weight", "std"),
            min_weight=("mean_weight", "min"),
            max_weight=("mean_weight", "max"),
            delta2_weighted_predicted_weight=(
                "delta2_weighted_predicted_weight",
                "mean",
            ),
            delta2_weighted_oracle_weight=(
                "delta2_weighted_oracle_weight",
                "mean",
            ),
            residual_win_fraction=("residual_win_fraction", "mean"),
            mean_spearman=("spearman_weight_vs_residual_advantage", "mean"),
            std_spearman=("spearman_weight_vs_residual_advantage", "std"),
        )
        .sort_values("target_order")
        .reset_index(drop=True)
    )
    return units, summary


def calibration_table(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    unit_cols = [
        "dataset",
        "target_id",
        "target_display",
        "target_order",
        "seed",
    ]
    for keys, group in samples.groupby(unit_cols, sort=False):
        group = group.copy()
        rank = group["w"].rank(method="first", pct=True)
        group["weight_quintile"] = np.minimum(np.ceil(rank * 5).astype(int), 5)
        direct_rmse = rmse(group["y_true"].to_numpy(), group["A_direct"].to_numpy())
        for quintile, part in group.groupby("weight_quintile"):
            oracle_sample_weight = part["oracle_sample_weight"].to_numpy(dtype=float)
            if float(np.sum(oracle_sample_weight)) > np.finfo(float).eps:
                predicted_weight = float(
                    np.average(part["w"], weights=oracle_sample_weight)
                )
                oracle_weight = float(
                    np.average(part["oracle_weight"], weights=oracle_sample_weight)
                )
            else:
                predicted_weight = float(part["w"].mean())
                oracle_weight = float(part["oracle_weight"].mean())
            residual_rmse = rmse(
                part["y_true"].to_numpy(), part["A_residual"].to_numpy()
            )
            gate_rmse = rmse(part["y_true"].to_numpy(), part["A_final"].to_numpy())
            direct_part_rmse = rmse(
                part["y_true"].to_numpy(), part["A_direct"].to_numpy()
            )
            rows.append(
                {
                    **dict(zip(unit_cols, keys)),
                    "weight_quintile": int(quintile),
                    "n_samples": len(part),
                    "mean_weight": predicted_weight,
                    "mean_oracle_weight": oracle_weight,
                    "oracle_weight_calibration_error": predicted_weight - oracle_weight,
                    "residual_win_fraction": float(part["residual_better"].mean()),
                    "residual_log2_gain_vs_direct": float(
                        np.log2(direct_part_rmse / residual_rmse)
                    ),
                    "gate_log2_gain_vs_direct": float(
                        np.log2(direct_part_rmse / gate_rmse)
                    ),
                    "whole_unit_direct_RMSE": direct_rmse,
                }
            )
    units = pd.DataFrame(rows)
    summary = (
        units.groupby("weight_quintile", as_index=False)
        .agg(
            n_target_seed_units=("target_id", "size"),
            mean_weight=("mean_weight", "mean"),
            sem_weight=("mean_weight", "sem"),
            mean_oracle_weight=("mean_oracle_weight", "mean"),
            oracle_weight_sem=("mean_oracle_weight", "sem"),
            mean_absolute_calibration_error=(
                "oracle_weight_calibration_error",
                lambda x: float(np.mean(np.abs(x))),
            ),
            residual_win_fraction=("residual_win_fraction", "mean"),
            residual_win_fraction_sem=("residual_win_fraction", "sem"),
            residual_log2_gain_vs_direct=("residual_log2_gain_vs_direct", "mean"),
            residual_log2_gain_sem=("residual_log2_gain_vs_direct", "sem"),
            gate_log2_gain_vs_direct=("gate_log2_gain_vs_direct", "mean"),
            gate_log2_gain_sem=("gate_log2_gain_vs_direct", "sem"),
        )
        .reset_index(drop=True)
    )
    return units, summary


def ablation_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_frames: list[pd.DataFrame] = []
    for dataset, path in ABLATION_FILES.items():
        frame = add_target_identity(pd.read_csv(path), dataset)
        if dataset == "ESTM":
            frame = pool_estm_fold_rmse(
                frame,
                [
                    "dataset",
                    "target_id",
                    "target_display",
                    "target_order",
                    "seed",
                    "feature_set",
                ],
                "test_RMSE",
            )
        frame["paired_log2_gain_vs_direct"] = np.log2(
            frame["direct_test_RMSE"] / frame["test_RMSE"]
        )
        unit_frames.append(frame)
    units = pd.concat(unit_frames, ignore_index=True, sort=False)
    target_seed_mean = (
        units.groupby(
            [
                "dataset",
                "target_id",
                "target_display",
                "target_order",
                "feature_set",
            ],
            as_index=False,
        )
        .agg(
            mean_log2_gain=("paired_log2_gain_vs_direct", "mean"),
            std_log2_gain=("paired_log2_gain_vs_direct", "std"),
            win_count=("paired_log2_gain_vs_direct", lambda x: int((x > 0).sum())),
            n_seeds=("seed", "nunique"),
        )
        .reset_index(drop=True)
    )
    macro = (
        target_seed_mean.groupby("feature_set", as_index=False)
        .agg(
            macro_mean_log2_gain=("mean_log2_gain", "mean"),
            target_sem=("mean_log2_gain", "sem"),
            target_win_count=("mean_log2_gain", lambda x: int((x > 0).sum())),
            n_targets=("target_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    return units, target_seed_mean, macro


def blend_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_frames: list[pd.DataFrame] = []
    for dataset, path in BLEND_FILES.items():
        frame = add_target_identity(pd.read_csv(path), dataset)
        if dataset == "ESTM":
            frame = pool_estm_fold_rmse(
                frame,
                [
                    "dataset",
                    "target_id",
                    "target_display",
                    "target_order",
                    "seed",
                    "method",
                ],
                "test_RMSE",
            )
        frame["paired_log2_gain_vs_direct"] = np.log2(
            frame["direct_test_RMSE"] / frame["test_RMSE"]
        )
        unit_frames.append(frame)
    units = pd.concat(unit_frames, ignore_index=True, sort=False)
    target = (
        units.groupby(
            [
                "dataset",
                "target_id",
                "target_display",
                "target_order",
                "method",
            ],
            as_index=False,
        )
        .agg(
            mean_log2_gain=("paired_log2_gain_vs_direct", "mean"),
            std_log2_gain=("paired_log2_gain_vs_direct", "std"),
            win_count=("paired_log2_gain_vs_direct", lambda x: int((x > 0).sum())),
            n_seeds=("seed", "nunique"),
        )
        .reset_index(drop=True)
    )
    macro = (
        target.groupby("method", as_index=False)
        .agg(
            macro_mean_log2_gain=("mean_log2_gain", "mean"),
            target_sem=("mean_log2_gain", "sem"),
            target_win_count=("mean_log2_gain", lambda x: int((x > 0).sum())),
            n_targets=("target_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    return units, target, macro


def main() -> None:
    samples = load_samples()
    weight_units, weight_summary = weight_target_summary(samples)
    calibration_units, calibration_summary = calibration_table(samples)
    ablation_units, ablation_target, ablation_macro = ablation_tables()
    blend_units, blend_target, blend_macro = blend_tables()

    outputs = {
        "Figure4_SourceData_PerSample.csv": samples,
        "Figure4a_Weight_BySeedTarget.csv": weight_units,
        "Figure4a_Weight_TargetSummary.csv": weight_summary,
        "Figure4b_Calibration_BySeedTargetQuintile.csv": calibration_units,
        "Figure4b_Calibration_Summary.csv": calibration_summary,
        "Figure4c_Ablation_BySeedTarget.csv": ablation_units,
        "Figure4c_Ablation_TargetSummary.csv": ablation_target,
        "Figure4c_Ablation_MacroSummary.csv": ablation_macro,
        "Figure4d_FixedBlend_BySeedTarget.csv": blend_units,
        "Figure4d_FixedBlend_TargetSummary.csv": blend_target,
        "Figure4d_FixedBlend_MacroSummary.csv": blend_macro,
    }
    for filename, table in outputs.items():
        table.to_csv(BASE / filename, index=False)
        print(f"Wrote {len(table):>7} rows: {BASE / filename}")

    print("\nWeight-target summary:")
    print(
        weight_summary[
            [
                "dataset",
                "target_display",
                "mean_weight",
                "std_weight",
                "residual_win_fraction",
                "mean_spearman",
            ]
        ].to_string(index=False)
    )
    print("\nCalibration:")
    print(calibration_summary.to_string(index=False))
    print("\nAblation macro:")
    print(ablation_macro.sort_values("macro_mean_log2_gain", ascending=False).to_string(index=False))
    print("\nFixed blend macro:")
    print(blend_macro.sort_values("macro_mean_log2_gain", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
