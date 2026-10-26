"""Recompute the source data of the repeated-split figures from the raw datasets.

Covers Figures 4 and 5 and Supplementary Figures S1-S3. ``--protocol
submitted`` reproduces ``source_data/``; ``--protocol revised`` applies the
correction listed in ``physicsgate/revised/CHANGES.md`` (ESTM and PV). Outer
splits are listed in ``configs/repeated_splits.json``; Supplementary Figure S3
uses ``configs/evaluated_splits.json``.

The publication tables are written to ``results/<protocol>/source_data`` and
compared with ``source_data/``.

Usage:
    python scripts/reproduce_figures.py --protocol revised
    python scripts/reproduce_figures.py --protocol revised --steps builders export
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce_submitted as rs  # noqa: E402

STEPS = ("data", "splits", "pv_selection", "pv", "stabilization", "promote", "builders", "export")
PV_PREFIX = "repeated_pv_selection"
# PV processes run in parallel, each with all cores (n_jobs=-1) as in the original runs:
# XGBoost results, and hence some model choices, depend on its thread count.
SELECTION_PROCESSES = 3
PV_BATCH = 3
FIGURES = {  # published folder -> output folder of the builders, relative to the analysis directory
    "figure4": Path("figure4_gate_mechanism", "50seed", "final_estm_pv", "mechanism_tables"),
    "figureS1": Path("figure4_gate_mechanism", "50seed", "final_sse_lmb", "mechanism_tables"),
    "figure5": Path("figure5_error_propagation"),
    "figureS2": Path("si_correctability_complementarity"),
    "figureS3": Path("si_stabilization_sensitivity"),
}
RENAMES = {"Ares_old": "Residual"}            # column names in the published tables
LABELS = {"Ares_old": "Residual-only"}        # method labels in the published tables


def split_ids(config: str, dataset: str) -> list[int]:
    data = json.loads((ROOT / "configs" / config).read_text(encoding="utf-8"))
    return [int(s) for s in data["datasets"][dataset]["split_ids"]]


class Runner:
    def __init__(self, protocol: str):
        self.protocol = protocol
        self.workdir = rs.workdir(protocol)
        self.analysis = self.workdir / "results" / "submitted" / "pv" / "outputs"
        self.logs = self.workdir / "results" / "submitted" / "logs" / "figures"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                        PHYSICSGATE_WORKDIR=str(self.workdir))
        self.prefix = [sys.executable, "-m"]

    def module(self, name: str) -> str:
        revised = {"physicsgate.submitted.pv.pipeline": "physicsgate.revised.pv_pipeline",
                   "physicsgate.submitted.pv.stabilization": "physicsgate.revised.stabilization",
                   "physicsgate.submitted.mechanism.estm": "physicsgate.revised.mechanism_estm"}
        return revised.get(name, name) if self.protocol == "revised" else name

    def run(self, jobs: list[tuple[str, list[str]]], check: bool = True) -> list[str]:
        """Run (log name, module arguments) jobs in parallel; return the failed names (or exit if check)."""
        procs = []
        for name, args in jobs:
            print("+", " ".join(args), flush=True)
            log = open(self.logs / f"{name}.log", "w", encoding="utf-8")
            procs.append((name, subprocess.Popen([*self.prefix, *args], cwd=ROOT, env=self.env,
                                                 stdout=log, stderr=subprocess.STDOUT)))
        failed = [name for name, p in procs if p.wait()]
        if failed and check:
            raise SystemExit(f"failed: {failed} (logs in {self.logs})")
        return failed


def step_data(r: Runner) -> None:
    rs.prepare_workdir(r.protocol)
    r.run([(f"data_{d}", [f"physicsgate.submitted.{d}.data"]) for d in ("estm", "lmb", "pv")])


def step_splits(r: Runner) -> None:
    """Per-split gate diagnostics for SSE, ESTM and LMB, one dataset and one split at a time.

    As in the original runs, the datasets run one after another with one split
    at a time: the runners use all cores, and running several of them together
    made XGBoost fail inside the parallel model search."""
    plans = r.analysis / "figure4_gate_mechanism" / "seed_plans"
    plans.mkdir(parents=True, exist_ok=True)
    for d in ("SSE", "ESTM", "LMB", "PV"):
        (plans / f"Figure4_{d}_50Seed_List.txt").write_text(
            ",".join(map(str, split_ids("repeated_splits.json", d))) + "\n", encoding="utf-8")
    jobs = []
    for d in ("SSE", "ESTM", "LMB"):
        jobs.append((f"splits_{d}", [
            "physicsgate.submitted.mechanism.run_splits", "--dataset", d,
            "--module", r.module(f"physicsgate.submitted.mechanism.{d.lower()}"),
            "--seed-list", str(plans / f"Figure4_{d}_50Seed_List.txt"),
            "--output-dir", str(r.analysis / "figure4_gate_mechanism" / "50seed" / d.lower())]))
    for job in jobs:
        r.run([job])


def step_pv_selection(r: Runner) -> None:
    """PV Direct-model selection on the training rows of each split, in parallel chunks of splits."""
    tables = r.workdir / "results" / "submitted"
    if (tables / f"{PV_PREFIX}_base_selection.csv").exists():
        print(f"{PV_PREFIX}_base_selection.csv exists; selection skipped")
        return
    seeds = split_ids("repeated_splits.json", "PV")
    chunks = [seeds[j::SELECTION_PROCESSES] for j in range(SELECTION_PROCESSES)]
    r.run([(f"pv_selection_{j}", ["physicsgate.submitted.pv.selection", "--seeds", *map(str, chunk),
                                  "--n-jobs", "-1", "--output-prefix", f"{PV_PREFIX}_chunk{j}", "--resume"])
           for j, chunk in enumerate(chunks)])
    for suffix in ("base_selection", "selection", "detailed", "audit"):
        parts = [pd.read_csv(tables / f"{PV_PREFIX}_chunk{j}_{suffix}.csv") for j in range(len(chunks))]
        pd.concat(parts, ignore_index=True).to_csv(tables / f"{PV_PREFIX}_{suffix}.csv", index=False)


def staged(r: Runner, seeds: list[str]) -> bool:
    """True if the PV selections and split list are already in place for the pipeline."""
    plans = r.analysis / "figure4_gate_mechanism"
    base = plans / "50seed" / "pv" / "Figure4_PV_50Seed_Rerun_BaseSelection.csv"
    seed_file = plans / "seed_plans" / "Figure4_PV_50Seed_List.txt"
    source = r.workdir / "results" / "submitted" / f"{PV_PREFIX}_base_selection.csv"
    if not (base.exists() and seed_file.exists() and source.exists()):
        return False
    same_seeds = seed_file.read_text(encoding="utf-8").strip() == ",".join(seeds)
    return same_seeds and pd.read_csv(base).equals(pd.read_csv(source))


def step_pv(r: Runner) -> None:
    seeds = [str(s) for s in split_ids("repeated_splits.json", "PV")]
    if staged(r, seeds):  # resume: keep the splits already computed with these selections
        print("PV selections already staged; completed splits are kept", flush=True)
    else:  # stage_selection also removes per-split results computed with other selections
        r.run([("pv_stage_selection", ["physicsgate.submitted.pv.stage_selection", "--seeds", *seeds,
                                       "--selection-prefix", PV_PREFIX])])
    pipeline = r.module("physicsgate.submitted.pv.pipeline")
    by_seed = r.analysis / "pv_unified_train_range_50seed" / "by_seed"
    # Each split runs in its own process. The pipeline also aggregates all splits after each
    # run, and concurrent aggregations can collide; success is therefore judged by the
    # per-split completion marker, and the aggregation is repeated once at the end.
    for i in range(0, len(seeds), PV_BATCH):
        r.run([(f"pv_seed{s}", [pipeline, "--seeds", s, "--n-jobs", "-1"]) for s in seeds[i:i + PV_BATCH]],
              check=False)
    missing = [s for s in seeds if not (by_seed / f"seed_{s}" / "COMPLETE.txt").exists()]
    if missing:
        raise SystemExit(f"PV splits without results: {missing} (logs in {r.logs})")
    r.run([("pv_aggregate", [pipeline, "--seeds", *seeds, "--aggregate-only"])])


def step_stabilization(r: Runner) -> None:
    module = r.module("physicsgate.submitted.pv.stabilization")
    for d in ("ESTM", "PV"):  # one at a time: each run uses all cores
        r.run([(f"stabilization_{d}", [module, "--dataset", d, "--seeds",
                                       *map(str, split_ids("evaluated_splits.json", d))])])


def step_promote(r: Runner) -> None:
    """Copy the PV outputs into the mechanism-analysis layout (after the stabilization run,
    which reads the selection tables that this step replaces)."""
    r.run([("pv_promote", ["physicsgate.submitted.pv.promote"])])


def step_builders(r: Runner) -> None:
    for name in ("figure4", "figure_s1", "figure5", "figure_s2", "figure_s3"):
        r.run([(f"build_{name}", [f"physicsgate.submitted.figures.{name}"])])


def format_sig(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    return f"{value:.{max(0, digits - math.floor(math.log10(abs(value))) - 1)}f}"


def table_s6(source: Path) -> pd.DataFrame:
    """Supplementary Table S6: the Figure 5 and Figure S2 mechanism chains, joined by target."""
    f5 = pd.read_csv(source / "figure5" / "Figure5_Target_Summary.csv", encoding="utf-8-sig")
    s2 = pd.read_csv(source / "figureS2" / "SI_Correctability_Complementarity_TargetSummary.csv", encoding="utf-8-sig")
    merged = f5.merge(s2, on=["dataset", "target_code"], suffixes=("", "_correctability"), validate="one_to_one")
    ci = lambda row, p: (f"{format_sig(row[p + '_mean'])} ({format_sig(row[p + '_CI_low'])}, "  # noqa: E731
                         f"{format_sig(row[p + '_CI_high'])})")
    rows = []
    for _, row in merged.sort_values(["dataset", "target_order"]).iterrows():
        rows.append({
            "Dataset": row["dataset"], "Target": row["target_display"], "Equation route": row["equation_route"],
            "Splits": int(row["n_seeds"]),
            "Direct R2, mean (95% CI)": ci(row, "direct_R2"),
            "R3 (P90), mean (95% CI)": ci(row, "R3_p90"),
            "Physics distortion, mean G (95% CI)": ci(row, "physics_penalty_log2"),
            "Residual correctability, mean G (95% CI)": ci(row, "residual_correctability_log2"),
            "Branch-error correlation, mean (95% CI)": ci(row, "branch_error_correlation"),
            "High-R3 suppression fraction": row["fraction_splits_suppressing_high_R3"],
            "Delta w, mean (95% CI)": ci(row, "delta_w_low_minus_high_R3"),
            "PhysicsGate gain, mean G (95% CI)": ci(row, "PhysicsGate_gain_log2"),
        })
    return pd.DataFrame(rows)


def step_export(r: Runner) -> None:
    out = ROOT / "results" / r.protocol / "source_data"
    for folder, relative in FIGURES.items():
        (out / folder).mkdir(parents=True, exist_ok=True)
        for published in sorted((ROOT / "source_data" / folder).glob("*.csv")):
            if published.name == "Figure5_Literature_Direct_R2.csv":
                (out / folder / published.name).write_bytes(published.read_bytes())  # literature values
                continue
            table = pd.read_csv(r.analysis / relative / published.name)
            for old, new in RENAMES.items():
                table.columns = [c.replace(f"{old}_", f"{new}_") for c in table.columns]
            for column in table.columns:
                if table[column].dtype == object:
                    table[column] = table[column].replace(LABELS)
            columns = list(pd.read_csv(published, nrows=0, encoding="utf-8-sig").columns)
            table[columns].to_csv(out / folder / published.name, index=False)
    tables = ROOT / "results" / r.protocol / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table_s6(out).to_csv(tables / "TableS6.csv", index=False, encoding="utf-8-sig")
    pd.read_csv(out / "figureS3" / "SI_Stabilization_TargetSummary.csv").to_csv(
        tables / "TableS8.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote publication tables to {out} and Tables S6 and S8 to {tables}")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "compare_source_data.py"), str(out)], cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=["submitted", "revised"], default="submitted")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    args = parser.parse_args()
    r = Runner(args.protocol)
    for name in STEPS:
        if name in args.steps:
            print(f"== {name}", flush=True)
            globals()[f"step_{name}"](r)


if __name__ == "__main__":
    main()
