"""Figure 4 source data (ESTM and PV, repeated splits).

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/build_figure4_estm_pv_50seed_analysis.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import PV_WORKDIR  # noqa: E402
from pathlib import Path
import numpy as np
import pandas as pd
from physicsgate.submitted.figures import mechanism as analysis  # noqa: E402

ROOT = PV_WORKDIR

RUN_BASE = ROOT / "outputs" / "figure4_gate_mechanism" / "50seed"

OUTPUT_DIR = RUN_BASE / "final_estm_pv" / "mechanism_tables"


def write_gate_vs_fixed_tables() -> None:
    units = pd.read_csv(OUTPUT_DIR / "Figure4d_FixedBlend_BySeedTarget.csv")
    paired = (
        units.loc[
            units["method"].isin(["PhysicsGate", "Fixed scalar blend"]),
            [
                "dataset",
                "target_id",
                "target_display",
                "target_order",
                "seed",
                "method",
                "paired_log2_gain_vs_direct",
            ],
        ]
        .pivot(
            index=[
                "dataset",
                "target_id",
                "target_display",
                "target_order",
                "seed",
            ],
            columns="method",
            values="paired_log2_gain_vs_direct",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    paired["gate_minus_fixed_log2_gain"] = (
        paired["PhysicsGate"] - paired["Fixed scalar blend"]
    )
    paired["gate_beats_fixed"] = paired["gate_minus_fixed_log2_gain"] > 0
    paired.to_csv(
        OUTPUT_DIR / "Figure4d_PhysicsGate_vs_FixedBlend_BySeedTarget.csv",
        index=False,
    )

    target = (
        paired.groupby(
            ["dataset", "target_id", "target_display", "target_order"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mean_gate_minus_fixed_log2_gain=(
                "gate_minus_fixed_log2_gain",
                "mean",
            ),
            std_gate_minus_fixed_log2_gain=(
                "gate_minus_fixed_log2_gain",
                "std",
            ),
            gate_win_count=("gate_beats_fixed", "sum"),
            gate_win_fraction=("gate_beats_fixed", "mean"),
        )
        .sort_values("target_order")
        .reset_index(drop=True)
    )
    target["equivalent_RMSE_reduction_vs_fixed_percent"] = (
        1.0 - np.power(2.0, -target["mean_gate_minus_fixed_log2_gain"])
    ) * 100.0
    target.to_csv(
        OUTPUT_DIR / "Figure4d_PhysicsGate_vs_FixedBlend_TargetSummary.csv",
        index=False,
    )

    dataset = (
        target.groupby("dataset", as_index=False)
        .agg(
            n_targets=("target_id", "nunique"),
            macro_mean_gate_minus_fixed_log2_gain=(
                "mean_gate_minus_fixed_log2_gain",
                "mean",
            ),
            target_sem=("mean_gate_minus_fixed_log2_gain", "sem"),
            total_gate_wins=("gate_win_count", "sum"),
            total_seed_target_units=("n_seeds", "sum"),
        )
        .reset_index(drop=True)
    )
    dataset["gate_win_fraction"] = (
        dataset["total_gate_wins"] / dataset["total_seed_target_units"]
    )
    dataset["equivalent_RMSE_reduction_vs_fixed_percent"] = (
        1.0
        - np.power(2.0, -dataset["macro_mean_gate_minus_fixed_log2_gain"])
    ) * 100.0
    dataset.to_csv(
        OUTPUT_DIR / "Figure4d_PhysicsGate_vs_FixedBlend_DatasetSummary.csv",
        index=False,
    )


def write_dataset_calibration_table() -> None:
    units = pd.read_csv(OUTPUT_DIR / "Figure4b_Calibration_BySeedTargetQuintile.csv")
    summary = (
        units.groupby(["dataset", "weight_quintile"], as_index=False)
        .agg(
            n_target_seed_units=("target_id", "size"),
            mean_weight=("mean_weight", "mean"),
            sem_weight=("mean_weight", "sem"),
            mean_oracle_weight=("mean_oracle_weight", "mean"),
            oracle_weight_sem=("mean_oracle_weight", "sem"),
            mean_absolute_calibration_error=(
                "oracle_weight_calibration_error",
                lambda values: float(np.mean(np.abs(values))),
            ),
            residual_win_fraction=("residual_win_fraction", "mean"),
            residual_win_fraction_sem=("residual_win_fraction", "sem"),
            residual_log2_gain_vs_direct=(
                "residual_log2_gain_vs_direct",
                "mean",
            ),
            residual_log2_gain_sem=("residual_log2_gain_vs_direct", "sem"),
            gate_log2_gain_vs_direct=("gate_log2_gain_vs_direct", "mean"),
            gate_log2_gain_sem=("gate_log2_gain_vs_direct", "sem"),
        )
        .reset_index(drop=True)
    )
    summary.to_csv(
        OUTPUT_DIR / "Figure4b_Calibration_ByDatasetSummary.csv",
        index=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis.BASE = OUTPUT_DIR
    analysis.SAMPLE_FILES = {
        "ESTM": RUN_BASE / "estm" / "Figure4_ESTM_PerSample_GateDiagnostics.csv",
        "PV": RUN_BASE / "pv" / "Figure4_PV_PerSample_GateDiagnostics.csv",
    }
    analysis.ABLATION_FILES = {
        "ESTM": RUN_BASE / "estm" / "Figure4_ESTM_Reliability_Ablation.csv",
        "PV": RUN_BASE / "pv" / "Figure4_PV_Reliability_Ablation.csv",
    }
    analysis.BLEND_FILES = {
        "ESTM": RUN_BASE / "estm" / "Figure4_ESTM_FixedBlend_Comparison.csv",
        "PV": RUN_BASE / "pv" / "Figure4_PV_FixedBlend_Comparison.csv",
    }
    analysis.TARGET_ORDER = [
        pair for pair in analysis.TARGET_ORDER if pair[0] in analysis.SAMPLE_FILES
    ]
    analysis.ORDER_MAP = {
        pair: index for index, pair in enumerate(analysis.TARGET_ORDER)
    }
    analysis.main()
    write_dataset_calibration_table()
    write_gate_vs_fixed_tables()


if __name__ == "__main__":
    main()
