"""PhysicsGate against a Direct ensemble with the same training time.

Reads the records of ``physicsgate.revised.bagged_direct`` (test predictions of
the Direct model and of the running average of m bootstrap copies of it) and
compares them, endpoint by endpoint, with the revised-protocol Data S2:

- G_bagged = log2(RMSE_Direct / RMSE_bagged): what the extra training time
  buys Direct;
- G_PhysicsGate = log2(RMSE_Direct / RMSE_PhysicsGate), as reported;
- G_PhysicsGate - G_bagged = log2(RMSE_bagged / RMSE_PhysicsGate): the gain of
  PhysicsGate over the time-matched Direct ensemble.

Values are averaged over the five evaluated outer splits; ESTM metrics are
averaged over the three folds of a split first, as in Data S2. The Direct
predictions of the records are checked against Data S2.

    python scripts/analyze_bagged_direct.py [--records DIR] [--reference CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "bagged_direct" / "workdir" / "results" / "submitted" / "bagged_direct"
REFERENCE = ROOT / "source_data" / "supplementary" / "Supplementary_Data_S2_All_Evaluation_Unit_Metrics.csv"


def rmse(group: pd.DataFrame, column: str) -> float:
    return float(np.sqrt(np.mean((group["y_true"] - group[column]) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revised" / "analyses")
    args = parser.parse_args()
    records = pd.concat([pd.read_csv(p) for p in sorted(args.records.glob("*.csv.gz"))], ignore_index=True)
    bagged = sorted((c for c in records.columns if c.startswith("bagged_")), key=lambda c: int(c.split("_")[1]))

    rows = []
    for (dataset, seed, fold, target), unit in records.groupby(["dataset", "seed", "fold", "target"]):
        m = int(unit["m"].iloc[0])
        row = {"Dataset": dataset, "Outer seed": seed, "fold": fold, "Target code": target, "m": m,
               "bagging_seconds": unit["bagging_seconds"].iloc[0], "RMSE_direct": rmse(unit, "direct")}
        for column in bagged:
            if unit[column].notna().all():
                row[f"RMSE_{column}"] = rmse(unit, column)
        row["RMSE_bagged"] = rmse(unit, f"bagged_{m}")
        rows.append(row)
    folds = pd.DataFrame(rows)
    split = folds.groupby(["Dataset", "Outer seed", "Target code"], as_index=False).mean(numeric_only=True)

    reference = pd.read_csv(args.reference)
    reference["Route"] = reference["Route"].fillna("")
    wide = reference.pivot_table(index=["Target order", "Dataset", "Target", "Target code", "Route", "Outer seed"],
                                 columns="Method", values="RMSE").reset_index()
    merged = wide.merge(split, on=["Dataset", "Target code", "Outer seed"], how="inner")
    differs = (merged["RMSE_direct"] / merged["Direct"] - 1).abs() > 1e-9
    print(f"Direct check: {int((~differs).sum())} of {len(merged)} endpoint-split pairs identical to Data S2")

    merged["G_bagged"] = np.log2(merged["Direct"] / merged["RMSE_bagged"])
    merged["G_PhysicsGate"] = np.log2(merged["Direct"] / merged["PhysicsGate"])
    merged["G_PhysicsGate_vs_bagged"] = np.log2(merged["RMSE_bagged"] / merged["PhysicsGate"])
    for column in bagged:
        if f"RMSE_{column}" in merged:
            merged[f"G_{column}"] = np.log2(merged["Direct"] / merged[f"RMSE_{column}"])
    keys = ["Target order", "Dataset", "Target", "Route"]
    summary = merged.groupby(keys, as_index=False).agg(
        splits=("Outer seed", "size"), m=("m", "first"),
        G_bagged=("G_bagged", "mean"), G_PhysicsGate=("G_PhysicsGate", "mean"),
        G_PhysicsGate_vs_bagged=("G_PhysicsGate_vs_bagged", "mean"),
        splits_PhysicsGate_better=("G_PhysicsGate_vs_bagged", lambda s: int((s > 0).sum())),
        **{f"G_{c}": (f"G_{c}", "mean") for c in bagged if f"G_{c}" in merged},
    ).sort_values("Target order")
    macro = {"Target": f"Macro mean ({len(summary)} endpoints)"}
    for column in [c for c in summary.columns if c.startswith("G_")]:
        if summary[column].notna().all():  # intermediate m beyond the m of some datasets are left out
            macro[column] = summary[column].mean()
    summary = pd.concat([summary, pd.DataFrame([macro])], ignore_index=True)
    for column in ("G_bagged", "G_PhysicsGate", "G_PhysicsGate_vs_bagged"):
        summary[column.replace("G_", "reduction % ")] = 100 * (1 - 2.0 ** -summary[column])
    args.out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out / "bagged_direct_by_split.csv", index=False)
    summary.to_csv(args.out / "bagged_direct_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary[["Dataset", "Target", "Route", "m", "G_bagged", "G_PhysicsGate", "G_PhysicsGate_vs_bagged",
                       "splits_PhysicsGate_better"]].round(4).to_string(index=False))
        print("\nG_bagged at intermediate m (saturation):")
        print(summary[["Target", "Route"] + [f"G_{c}" for c in bagged if f"G_{c}" in summary]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
