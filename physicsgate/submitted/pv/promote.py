"""Copy the PV repeated-split outputs into the mechanism-analysis layout.

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/promote_pv_unified_train_range.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import PV_WORKDIR  # noqa: E402
import shutil
from pathlib import Path
import pandas as pd

ROOT = PV_WORKDIR

SOURCE = ROOT / "outputs" / "pv_unified_train_range_50seed"

FIGURE4_PV = ROOT / "outputs" / "figure4_gate_mechanism" / "50seed" / "pv"

STAGING = ROOT / "outputs" / "publication_export_staging"

FOUR_METHODS = {
    "Direct",
    "Physics-only",
    "Ares_old",
    "FixedGate_R2_R5_R6",
}


def require_rows(frame: pd.DataFrame, expected: int, label: str) -> None:
    if len(frame) != expected:
        raise RuntimeError(f"{label}: expected {expected} rows, found {len(frame)}")


def main() -> None:
    FIGURE4_PV.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)

    detailed = pd.read_csv(SOURCE / "PV_Unified_50Seed_Detailed.csv")
    core = detailed.loc[detailed["method"].isin(FOUR_METHODS)].copy()
    require_rows(core, 50 * 4 * 4, "Figure 4 core detailed")
    if core["seed"].nunique() != 50 or core["target_code"].nunique() != 4:
        raise RuntimeError("Figure 4 detailed table has incomplete seed/target coverage")
    core.to_csv(FIGURE4_PV / "Figure4_PV_50Seed_Rerun_Detailed.csv", index=False)

    file_map = {
        "PV_Unified_50Seed_BaseSelection.csv": (
            "Figure4_PV_50Seed_Rerun_BaseSelection.csv"
        ),
        "PV_Unified_50Seed_Selection.csv": (
            "Figure4_PV_50Seed_Rerun_Selection.csv"
        ),
        "PV_Unified_50Seed_Audit.csv": "Figure4_PV_50Seed_Rerun_Audit.csv",
        "PV_Unified_50Seed_PerSample_GateDiagnostics.csv": (
            "Figure4_PV_PerSample_GateDiagnostics.csv"
        ),
        "PV_Unified_50Seed_Reliability_Ablation.csv": (
            "Figure4_PV_Reliability_Ablation.csv"
        ),
        "PV_Unified_50Seed_FixedBlend_Comparison.csv": (
            "Figure4_PV_FixedBlend_Comparison.csv"
        ),
    }
    for source_name, destination_name in file_map.items():
        shutil.copy2(SOURCE / source_name, FIGURE4_PV / destination_name)

    status = (
        core.groupby("seed", as_index=False)
        .agg(
            n_targets=("target_code", "nunique"),
            n_methods=("method", "nunique"),
            n_rows=("method", "size"),
        )
        .sort_values("seed")
    )
    status["complete"] = (
        status["n_targets"].eq(4)
        & status["n_methods"].eq(4)
        & status["n_rows"].eq(16)
    )
    if not status["complete"].all():
        raise RuntimeError("At least one Figure 4 seed is incomplete")
    status["stabilization"] = (
        "outer_train_target_minmax_for_inverse_targets; forward_PCE_unclipped"
    )
    status.to_csv(FIGURE4_PV / "Figure4_PV_50Seed_RunStatus.csv", index=False)

    frozen = pd.read_csv(SOURCE / "PV_Unified_Frozen5_Detailed.csv")
    frozen_summary = pd.read_csv(SOURCE / "PV_Unified_Frozen5_Summary.csv")
    require_rows(frozen, 5 * 4 * 6, "Frozen-five detailed")
    require_rows(frozen_summary, 4 * 6, "Frozen-five summary")
    frozen.to_csv(STAGING / "PV_FixedGate_5Seeds_Detailed.csv", index=False)
    frozen_summary.to_csv(STAGING / "PV_FixedGate_5Seeds_Summary.csv", index=False)

    audit = pd.DataFrame(
        [
            {
                "check": "figure4_seed_count",
                "value": core["seed"].nunique(),
                "expected": 50,
                "pass": core["seed"].nunique() == 50,
            },
            {
                "check": "figure4_target_count",
                "value": core["target_code"].nunique(),
                "expected": 4,
                "pass": core["target_code"].nunique() == 4,
            },
            {
                "check": "figure4_core_rows",
                "value": len(core),
                "expected": 800,
                "pass": len(core) == 800,
            },
            {
                "check": "frozen5_six_method_rows",
                "value": len(frozen),
                "expected": 120,
                "pass": len(frozen) == 120,
            },
            {
                "check": "true_test_auxiliary_use",
                "value": int(core["uses_true_test_auxiliary_values"].astype(bool).sum()),
                "expected": 0,
                "pass": not core["uses_true_test_auxiliary_values"].astype(bool).any(),
            },
            {
                "check": "test_target_selection_use",
                "value": int(core["uses_test_target_for_selection"].astype(bool).sum()),
                "expected": 0,
                "pass": not core["uses_test_target_for_selection"].astype(bool).any(),
            },
        ]
    )
    audit.to_csv(SOURCE / "PV_Unified_Promotion_Audit.csv", index=False)
    if not audit["pass"].all():
        raise RuntimeError("Promotion audit failed")
    print(audit.to_string(index=False))
    print(f"Promoted Figure 4 sources to {FIGURE4_PV}")
    print(f"Staged publication PV tables in {STAGING}")


if __name__ == "__main__":
    main()
