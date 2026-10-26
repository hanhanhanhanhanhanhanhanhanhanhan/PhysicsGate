"""Domain-level paired gains, with the outer split as the unit of resampling.

For each method, the paired gain over Direct of an endpoint and outer split is
G = log2(RMSE_Direct / RMSE_method). Within a domain and split, the gains of
its properties are averaged with equal weight; for LMB the two energy-density
routes are averaged first, so that energy density, discharge energy, capacity
and voltage count once each. The domain gain is the mean over the five splits.
Its 95% interval is the exact bootstrap over splits (all 5^5 resamples), which
keeps the properties of a split together. The equivalent RMSE reduction is
100 (1 - 2^-G).

    python scripts/analyze_domain_summary.py [--data-s2 CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_S2 = ROOT / "source_data" / "supplementary" / "Supplementary_Data_S2_All_Evaluation_Unit_Metrics.csv"
METHODS = ["Physics-only", "Residual-only", "Ordinary stacking", "PhysResStack", "PhysicsGate"]
DOMAINS = ["SSE", "ESTM", "LMB", "PV"]


def split_gains(s2: pd.DataFrame) -> pd.DataFrame:
    """Domain gain of every method on every outer split."""
    s2 = s2.assign(Route=s2["Route"].fillna(""))
    wide = s2.pivot_table(index=["Dataset", "Target code", "Route", "Outer seed"], columns="Method", values="RMSE")
    gains = pd.DataFrame({m: np.log2(wide["Direct"] / wide[m]) for m in METHODS}).reset_index()
    gains["property"] = gains["Target code"]  # the two LMB energy-density routes share a target code
    per_property = gains.groupby(["Dataset", "property", "Outer seed"])[METHODS].mean().reset_index()
    return per_property.groupby(["Dataset", "Outer seed"])[METHODS].mean().reset_index()


def exact_bootstrap(values: np.ndarray) -> tuple[float, float]:
    means = [np.mean(values[list(index)]) for index in itertools.product(range(len(values)), repeat=len(values))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-s2", type=Path, default=DATA_S2)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revised" / "analyses")
    args = parser.parse_args()
    by_split = split_gains(pd.read_csv(args.data_s2))
    rows = []
    for dataset in DOMAINS:
        unit = by_split[by_split["Dataset"].eq(dataset)]
        for method in METHODS:
            values = unit[method].to_numpy()
            low, high = exact_bootstrap(values)
            rows.append({"Dataset": dataset, "Method": method, "G": values.mean(), "CI_low": low, "CI_high": high,
                         "splits_positive": int((values > 0).sum()), "splits": len(values)})
    summary = pd.DataFrame(rows)
    macro = summary.groupby("Method", sort=False)["G"].mean().rename("G").reset_index().assign(Dataset="Mean of 4 domains")
    summary = pd.concat([summary, macro], ignore_index=True)
    for column in ("G", "CI_low", "CI_high"):
        summary[column.replace("G", "reduction %").replace("CI_", "reduction % CI_")] = 100 * (1 - 2.0 ** -summary[column])
    summary["rank"] = summary.groupby("Dataset")["G"].rank(ascending=False).astype(int)
    args.out.mkdir(parents=True, exist_ok=True)
    by_split.to_csv(args.out / "domain_gains_by_split.csv", index=False)
    summary.to_csv(args.out / "domain_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
