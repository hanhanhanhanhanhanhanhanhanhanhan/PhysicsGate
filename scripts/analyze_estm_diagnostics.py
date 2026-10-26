"""ESTM runs of ``physicsgate.revised.estm_diagnostics`` (Supplementary Tables S18 and S19).

Compares the outputs of ``physicsgate.revised.estm_diagnostics`` with the
revised-protocol Data S2 (row-level interpolation, element fractions): per
endpoint, the mean Direct R2 and the paired gain G = log2(RMSE_Direct /
RMSE_method) of every method over the five evaluated splits, each method
measured against the Direct of its own run. With ``--same-splits`` (runs that
keep the outer splits: another representation or tuned base models), it also
reports the gain of the reported PhysicsGate over the Direct of the run. The
MAE (log conductivities in natural-log units, as in the source study) and R2 of
Direct and PhysicsGate are written as well: averaged over the three folds of
each outer seed, as in Data S2, then mean and standard deviation over the five
seeds.

    python scripts/analyze_estm_diagnostics.py --results-dir DIR --name NAME [--same-splits] [--reference CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_with_data_s2 import recomputed  # noqa: E402

REFERENCE = ROOT / "source_data" / "supplementary" / "Supplementary_Data_S2_All_Evaluation_Unit_Metrics.csv"
METHODS = ["Physics-only", "Residual-only", "Ordinary stacking", "PhysResStack", "PhysicsGate"]


def summary(frame: pd.DataFrame, rmse: str, r2: str) -> pd.DataFrame:
    wide = frame.pivot_table(index=["Target", "Outer seed"], columns="Method", values=rmse)
    gains = pd.DataFrame({m: np.log2(wide["Direct"] / wide[m]) for m in METHODS})
    out = gains.groupby(level="Target").mean()
    out["PhysicsGate positive splits"] = (gains["PhysicsGate"] > 0).groupby(level="Target").sum()
    out["Direct R2"] = frame[frame["Method"].eq("Direct")].groupby("Target")[r2].mean()
    return out


FOLD_TABLE = "tmp_estm_fixed_gate_multiseed_with_physresstack_and_ordinary_detailed.csv"
METHOD_NAMES = {"Direct": "Direct", "DirectAnchored_PhysicsGate_R2_R5_R6_tuned": "PhysicsGate"}


def seed_errors(results_dir: Path) -> pd.DataFrame:
    folds = pd.read_csv(results_dir / FOLD_TABLE)
    folds = folds[folds["method"].isin(METHOD_NAMES)].assign(method=lambda f: f["method"].map(METHOD_NAMES))
    folds["MAE"] = folds["MAE"] * np.where(folds["target"].str.startswith("log10"), np.log(10), 1.0)
    seeds = folds.groupby(["target", "method", "outer_seed"])[["MAE", "R2"]].mean().reset_index()
    return seeds.groupby(["target", "method"]).agg(MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), R2=("R2", "mean"),
                                                   R2_sd=("R2", "std"), seeds=("MAE", "size")).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--same-splits", action="store_true")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revised" / "analyses")
    args = parser.parse_args()
    reference = pd.read_csv(args.reference)
    reference = reference[reference["Dataset"].eq("ESTM")]
    names = reference.drop_duplicates("Target code").set_index("Target code")["Target"]
    variant = recomputed(args.results_dir)
    variant = variant[variant["Dataset"].eq("ESTM")].rename(columns={"seed": "Outer seed"})
    variant["Target"] = variant["code"].map(names)
    table = summary(reference, "RMSE", "R2").add_suffix(" reference").join(summary(variant, "RMSE", "R2").add_suffix(f" {args.name}"))
    if args.same_splits:
        ref = reference.pivot_table(index=["Target", "Outer seed"], columns="Method", values="RMSE")
        new = variant.pivot_table(index=["Target", "Outer seed"], columns="Method", values="RMSE")
        gain = np.log2(new["Direct"] / ref["PhysicsGate"])
        table["reported PhysicsGate vs this Direct"] = gain.groupby(level="Target").mean()
    table.loc["Mean"] = table.mean()
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / f"estm_{args.name}_summary.csv")
    errors = seed_errors(args.results_dir)
    errors.to_csv(args.out / f"estm_{args.name}_errors.csv", index=False)
    print(errors.round(4).to_string(index=False))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(3).T.to_string())


if __name__ == "__main__":
    main()
