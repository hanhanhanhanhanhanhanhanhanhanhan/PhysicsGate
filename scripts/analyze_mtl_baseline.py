"""Multitask baselines against Direct and PhysicsGate.

Reads the test predictions written by ``physicsgate.revised.mtl_baseline`` and
reports, per endpoint and domain, the paired gain G = log2(RMSE_Direct /
RMSE_model) of the multitask and single-task MLPs and forests, next to that of
PhysicsGate from the revised-protocol Data S2. For the MLPs, G is averaged over
the three initializations. ESTM metrics are averaged over the three folds of a
split first, as in Data S2. The effect of sharing alone is the gain of each
multitask model over its single-task counterpart. Domain values average the
properties of a split with equal weight (the two LMB energy-density routes
count once) and then the five splits.

    python scripts/analyze_mtl_baseline.py [--records DIR] [--reference CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "mtl_baseline" / "workdir" / "results" / "submitted" / "mtl_baseline"
REFERENCE = ROOT / "source_data" / "supplementary" / "Supplementary_Data_S2_All_Evaluation_Unit_Metrics.csv"
MODELS = ["mtl_mlp", "stl_mlp", "mtl_forest", "stl_forest"]


def unit_rmse(records: pd.DataFrame) -> pd.DataFrame:
    """RMSE of every model column per dataset, split, fold and endpoint."""
    columns = [c for c in records.columns if c.startswith(("mtl_", "stl_"))]
    rows = []
    for (dataset, seed, fold, target), unit in records.groupby(["dataset", "seed", "fold", "target"]):
        row = {"Dataset": dataset, "Outer seed": seed, "fold": fold, "Target code": target}
        for column in columns:
            row[column] = float(np.sqrt(np.mean((unit["y_true"] - unit[column]) ** 2)))
        rows.append(row)
    rmse = pd.DataFrame(rows)
    return rmse.groupby(["Dataset", "Outer seed", "Target code"], as_index=False).mean(numeric_only=True).drop(columns="fold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revised" / "analyses")
    args = parser.parse_args()
    records = pd.concat([pd.read_csv(p) for p in sorted(args.records.glob("*.csv.gz"))], ignore_index=True)
    rmse = unit_rmse(records)
    reference = pd.read_csv(args.reference).assign(Route=lambda f: f["Route"].fillna(""))
    wide = reference.pivot_table(index=["Target order", "Dataset", "Target", "Target code", "Route", "Outer seed"],
                                 columns="Method", values="RMSE").reset_index()
    merged = wide.merge(rmse, on=["Dataset", "Target code", "Outer seed"], how="inner")
    merged["G PhysicsGate"] = np.log2(merged["Direct"] / merged["PhysicsGate"])
    for model in MODELS:
        inits = [c for c in merged.columns if c.startswith(f"{model}_")]
        gains = np.column_stack([np.log2(merged["Direct"] / merged[c]) for c in inits])
        merged[f"G {model}"] = gains.mean(axis=1)
        if len(inits) > 1:
            merged[f"G {model} min"], merged[f"G {model} max"] = gains.min(axis=1), gains.max(axis=1)
    for family in ("mlp", "forest"):
        merged[f"sharing {family}"] = merged[f"G mtl_{family}"] - merged[f"G stl_{family}"]
    gain_columns = ["G PhysicsGate", *[f"G {m}" for m in MODELS], "sharing mlp", "sharing forest"]

    keys = ["Target order", "Dataset", "Target", "Route"]
    endpoint = merged.groupby(keys)[gain_columns].mean().reset_index().sort_values("Target order")
    endpoint["splits"] = merged.groupby(keys).size().values
    property_split = merged.groupby(["Dataset", "Target code", "Outer seed"])[gain_columns].mean().reset_index()
    domain_split = property_split.groupby(["Dataset", "Outer seed"])[gain_columns].mean().reset_index()
    domain = domain_split.groupby("Dataset", sort=False)[gain_columns].mean()
    domain.loc["Mean of domains"] = domain.mean()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out / "mtl_baseline_by_split.csv", index=False)
    endpoint.to_csv(args.out / "mtl_baseline_endpoints.csv", index=False)
    domain.to_csv(args.out / "mtl_baseline_domains.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(endpoint[["Dataset", "Target", "Route", "splits", *gain_columns]].round(3).to_string(index=False))
        print()
        print(domain.round(3).to_string())


if __name__ == "__main__":
    main()
