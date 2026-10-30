"""Grouped against row-level outer splits for SSE and LMB.

Reads the unit-by-unit comparison written by ``scripts/compare_with_data_s2.py``
for a run of ``scripts/reproduce_submitted.py --protocol row_split`` (the same
pipelines with row-level instead of material-grouped outer splits)
and reports, per endpoint, the Direct R2 and RMSE under both splits and the
paired gain G = log2(RMSE_Direct / RMSE_method) of every method, averaged over
the five evaluated splits. Row-level splits put rows of the same material or
chemical system in training and test; they measure interpolation among seen
materials and do not replace the grouped protocol. The domain-level gain of
PhysicsGate averages the properties of a dataset with equal weight within each
split, the two LMB energy-density routes counted once, as in Table S9.

    python scripts/analyze_row_split.py [--comparison CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "results" / "row_split" / "workdir" / "results" / "submitted" / "data_s2_comparison.csv"
METHODS = ["Physics-only", "Residual-only", "Ordinary stacking", "PhysResStack", "PhysicsGate"]


def gains(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    wide = frame.pivot_table(index=["Target order", "Dataset", "Target", "Route", "Outer seed"], columns="Method",
                             values=column)
    return pd.DataFrame({m: np.log2(wide["Direct"] / wide[m]) for m in METHODS})


def domain_gain(frame: pd.DataFrame, column: str) -> pd.Series:
    """Mean over splits of the equal-weight property mean of the PhysicsGate gain, per dataset."""
    wide = frame.pivot_table(index=["Dataset", "Target", "Route", "Outer seed"], columns="Method", values=column)
    gain = np.log2(wide["Direct"] / wide["PhysicsGate"]).rename("G").reset_index()
    per_property = gain.groupby(["Dataset", "Target", "Outer seed"])["G"].mean()  # LMB energy-density routes once
    return per_property.groupby(["Dataset", "Outer seed"]).mean().groupby("Dataset").mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", type=Path, default=COMPARISON)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revised" / "analyses")
    args = parser.parse_args()
    frame = pd.read_csv(args.comparison)
    frame = frame[frame["Dataset"].isin(["SSE", "LMB"])].assign(Route=lambda f: f["Route"].fillna(""))
    keys = ["Target order", "Dataset", "Target", "Route"]
    direct = frame[frame["Method"].eq("Direct")].groupby(keys)[["R2_S2", "R2_recomputed", "RMSE_S2", "RMSE_recomputed"]].mean()
    direct.columns = ["Direct R2 grouped", "Direct R2 row", "Direct RMSE grouped", "Direct RMSE row"]
    grouped = gains(frame, "RMSE_S2").groupby(level=keys).mean().add_suffix(" grouped")
    row = gains(frame, "RMSE_recomputed")
    positive = (row["PhysicsGate"] > 0).groupby(level=keys).sum().rename("PhysicsGate positive splits row")
    row = row.groupby(level=keys).mean().add_suffix(" row")
    summary = direct.join(grouped).join(row).join(positive).reset_index().sort_values("Target order")
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "row_split_summary.csv", index=False)
    domains = pd.DataFrame({"PhysicsGate domain G grouped": domain_gain(frame, "RMSE_S2"),
                            "PhysicsGate domain G row": domain_gain(frame, "RMSE_recomputed")})
    domains.to_csv(args.out / "row_split_domains.csv")
    print(domains.round(3).to_string())
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary[["Dataset", "Target", "Route", "Direct R2 grouped", "Direct R2 row", "PhysicsGate grouped",
                       "PhysicsGate row", "PhysicsGate positive splits row", "Residual-only row",
                       "PhysResStack row"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
