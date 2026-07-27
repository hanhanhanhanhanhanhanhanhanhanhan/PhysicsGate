"""Validate retained source data and rebuild Python-generated figures.

Figures 1-3 were authored in Origin. Their final images and source CSV files
are retained, so they are not redrawn by this command.
"""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

PIPELINE = [
    "plot_figure4_estm_pv_50seed_nature.py",
    "plot_figure4_sse_lmb_50seed_si_nature.py",
    "plot_figure5_error_propagation_nature.py",
    "plot_si_correctability_complementarity_nature.py",
    "plot_si_stabilization_sensitivity_nature.py",
]


def validate_origin_source() -> None:
    figure2 = ROOT / "source_data" / "figure2"
    with (figure2 / "Figure2_RankMatrix.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rank_rows = list(csv.DictReader(handle))
    rank_columns = [
        "Direct",
        "Physics-only",
        "Residual-only",
        "Ordinary stacking",
        "PhysResStack",
        "PhysicsGate",
    ]
    if len(rank_rows) != 16:
        raise RuntimeError("Figure 2 rank matrix must contain 16 rows")
    for row in rank_rows:
        ranks = sorted(int(row[column]) for column in rank_columns)
        if ranks != [1, 2, 3, 4, 5, 6]:
            raise RuntimeError(f"Invalid Figure 2 rank row: {row}")

    with (figure2 / "Figure2_MethodSummary.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        method_rows = list(csv.DictReader(handle))
    if len(method_rows) != 6 or any(
        int(row["target_count"]) != 16 for row in method_rows
    ):
        raise RuntimeError("Figure 2 method summary must contain six 16-target rows")

    figure3 = ROOT / "source_data" / "figure3"
    for filename in ("Figure3_PhysicsInformed.csv", "Figure3_Ensemble.csv"):
        with (figure3 / filename).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 16:
            raise RuntimeError(f"{filename} must contain 16 aligned target rows")
    print("Validated minimal Origin source data for Figures 2-3.")


def run_pipeline() -> None:
    validate_origin_source()
    for name in PIPELINE:
        command = [sys.executable, str(SCRIPTS / name)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
