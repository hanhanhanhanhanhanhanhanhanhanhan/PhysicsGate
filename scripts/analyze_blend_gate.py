"""PhysicsGate with the reported gate and with the gate fitted to the blend error (Supplementary Table S12).

Compares the revised Data S2 with the metrics of a run of
``scripts/reproduce_submitted.py --protocol blend_gate``: per endpoint, the
equivalent RMSE reduction 100 (1 - 2^-G) of PhysicsGate relative to Direct,
from the mean paired gain G = log2(RMSE_Direct / RMSE_PhysicsGate) over the five
evaluated splits, and the rank of PhysicsGate among the six architectures by
mean RMSE, each within its own run.

    python scripts/analyze_blend_gate.py [--reference CSV] [--blend CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "source_data" / "supplementary" / "Supplementary_Data_S2_All_Evaluation_Unit_Metrics.csv"
BLEND = ROOT / "results" / "blend_gate" / "workdir" / "results" / "submitted" / "data_s2_recomputed.csv"
KEYS = ["Target order", "Dataset", "Target", "Route"]


def endpoint_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.assign(Route=frame["Route"].fillna(""))
    wide = frame.pivot_table(index=[*KEYS, "Outer seed"], columns="Method", values="RMSE")
    gain = np.log2(wide["Direct"] / wide["PhysicsGate"]).groupby(level=KEYS).mean()
    rank = wide.groupby(level=KEYS).mean().rank(axis=1)["PhysicsGate"]
    positive = (np.log2(wide["Direct"] / wide["PhysicsGate"]) > 0).groupby(level=KEYS).sum()
    return pd.DataFrame({"G": gain, "RMSE reduction %": 100 * (1 - 2.0 ** -gain), "rank": rank.astype(int),
                         "positive splits": positive})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--blend", type=Path, default=BLEND)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revised" / "analyses")
    args = parser.parse_args()
    reported = endpoint_summary(pd.read_csv(args.reference, encoding="utf-8-sig"))
    blend = endpoint_summary(pd.read_csv(args.blend, encoding="utf-8-sig"))
    table = reported.add_suffix(", reported gate").join(blend.add_suffix(", blend-error gate")).reset_index()
    macro = {g: table[f"G, {g}"].mean() for g in ("reported gate", "blend-error gate")}
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "blend_gate_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))
    for gate, g in macro.items():
        print(f"{gate}: endpoint-macro G {g:.4f} ({100 * (1 - 2 ** -g):.2f}%), "
              f"ranked first {int((table[f'rank, {gate}'] == 1).sum())}/16, "
              f"positive {int((table[f'G, {gate}'] > 0).sum())}/16")


if __name__ == "__main__":
    main()
