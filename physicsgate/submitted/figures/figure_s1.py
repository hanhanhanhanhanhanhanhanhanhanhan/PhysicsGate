"""Supplementary Figure S1 source data (SSE and LMB, repeated splits).

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/build_figure4_sse_lmb_50seed_analysis.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import PV_WORKDIR  # noqa: E402
from pathlib import Path
from physicsgate.submitted.figures import figure4 as helpers  # noqa: E402
from physicsgate.submitted.figures import mechanism as analysis  # noqa: E402

ROOT = PV_WORKDIR

RUN_BASE = ROOT / "outputs" / "figure4_gate_mechanism" / "50seed"

OUTPUT_DIR = RUN_BASE / "final_sse_lmb" / "mechanism_tables"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis.BASE = OUTPUT_DIR
    analysis.SAMPLE_FILES = {
        "SSE": RUN_BASE / "sse" / "Figure4_SSE_PerSample_GateDiagnostics.csv",
        "LMB": RUN_BASE / "lmb" / "Figure4_LMB_PerSample_GateDiagnostics.csv",
    }
    analysis.ABLATION_FILES = {
        "SSE": RUN_BASE / "sse" / "Figure4_SSE_Reliability_Ablation.csv",
        "LMB": RUN_BASE / "lmb" / "Figure4_LMB_Reliability_Ablation.csv",
    }
    analysis.BLEND_FILES = {
        "SSE": RUN_BASE / "sse" / "Figure4_SSE_FixedBlend_Comparison.csv",
        "LMB": RUN_BASE / "lmb" / "Figure4_LMB_FixedBlend_Comparison.csv",
    }
    analysis.TARGET_ORDER = [
        pair for pair in analysis.TARGET_ORDER if pair[0] in analysis.SAMPLE_FILES
    ]
    analysis.ORDER_MAP = {
        pair: index for index, pair in enumerate(analysis.TARGET_ORDER)
    }
    analysis.main()

    helpers.OUTPUT_DIR = OUTPUT_DIR
    helpers.write_dataset_calibration_table()
    helpers.write_gate_vs_fixed_tables()


if __name__ == "__main__":
    main()
