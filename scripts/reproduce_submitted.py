"""Recompute the benchmark from the raw datasets and compare with Data S2.

``--protocol submitted`` (default) runs the pipelines in ``physicsgate.submitted``
and reproduces Supplementary Data S2. ``--protocol revised`` runs the same
pipelines with the correction listed in ``physicsgate/revised/CHANGES.md``
in a separate directory, ``results/revised/workdir``. ``--protocol row_split``
is a diagnostic for SSE and LMB: the revised protocol with row-level instead of
grouped splits (``physicsgate/revised/row_split.py``), in
``results/row_split/workdir``. ``--protocol blend_gate`` is the revised protocol
with the gate fitted to the error of the blended prediction
(``physicsgate/revised/gate.py``; Supplementary Table S12), in
``results/blend_gate/workdir``. All use the five evaluated outer splits of each
dataset (``configs/evaluated_splits.json``).

The four raw files listed in ``data/sources.csv`` must be in ``data/raw/``.

Usage:
    python scripts/reproduce_submitted.py                 # all datasets, one after another
    python scripts/reproduce_submitted.py --parallel      # datasets in parallel
    python scripts/reproduce_submitted.py --datasets SSE  # a subset
    python scripts/reproduce_submitted.py --protocol revised --parallel
    python scripts/reproduce_submitted.py --protocol row_split --parallel --datasets SSE LMB
    python scripts/reproduce_submitted.py --protocol blend_gate --parallel
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = ("SSE.xlsx", "estm.csv", "Energy density.xlsx", "Perovskite_database_content_all_data.csv")
ESTM_TABLES = "tmp_estm_fixed_gate_multiseed_with_physresstack_and_ordinary"


def workdir(protocol: str) -> Path:
    return ROOT if protocol == "submitted" else ROOT / "results" / protocol / "workdir"


def prepare_workdir(protocol: str) -> None:
    """Give a revised or diagnostic run its own working directory sharing the raw files."""
    if protocol == "submitted":
        return
    raw = workdir(protocol) / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        target = raw / name
        if not target.exists():
            try:
                os.link(ROOT / "data" / "raw" / name, target)
            except OSError:
                shutil.copy2(ROOT / "data" / "raw" / name, target)


def seeds(dataset: str) -> list[str]:
    config = json.loads((ROOT / "configs" / "evaluated_splits.json").read_text(encoding="utf-8"))
    return [str(s) for s in config["datasets"][dataset]["split_ids"]]


def submitted_stages(dataset: str) -> list[list[list[str]]]:
    m = "physicsgate.submitted"
    s = seeds(dataset)
    if dataset == "ESTM":
        steps = [[f"{m}.estm.data"], [f"{m}.estm.pipeline", "--seeds", ",".join(s)]]
    elif dataset == "SSE":
        steps = [[f"{m}.sse.pipeline", "--seeds", ",".join(s), "--include-physresstack", "--output-tag", "submitted"],
                 [f"{m}.sse.ordinary", "--seeds", *s]]
    elif dataset == "LMB":
        steps = [[f"{m}.lmb.data"],
                 [f"{m}.lmb.pipeline", "--seeds", *s, "--output-tag", "submitted_lmb", "--no-resume"],
                 [f"{m}.lmb.complete", "--seeds", *s, "--reference-tag", "submitted_lmb"]]
    elif dataset == "PV":
        steps = [[f"{m}.pv.data"],
                 [f"{m}.pv.selection", "--seeds", *s, "--output-prefix", "submitted_pv_selection"],
                 [f"{m}.pv.stage_selection", "--seeds", *s, "--selection-prefix", "submitted_pv_selection"],
                 [f"{m}.pv.pipeline", "--seeds", *s]]
    else:
        raise ValueError(dataset)
    return [[step] for step in steps]  # one command per stage


def revised_stages(dataset: str) -> list[list[list[str]]]:
    """The submitted stages, with the ESTM and PV pipelines replaced by their C1 versions."""
    swap = {"physicsgate.submitted.estm.pipeline": "physicsgate.revised.estm_pipeline",
            "physicsgate.submitted.pv.pipeline": "physicsgate.revised.pv_pipeline"}
    return [[[swap.get(cmd[0], cmd[0]), *cmd[1:]] for cmd in stage] for stage in submitted_stages(dataset)]


def run_dataset(dataset: str, protocol: str) -> subprocess.Popen:
    """Start one dataset as a shell-free chain of stages."""
    logs = workdir(protocol) / "results" / "submitted" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PHYSICSGATE_WORKDIR=str(workdir(protocol)))
    if protocol == "submitted":
        stages = [[[sys.executable, "-m", *cmd] for cmd in stage] for stage in submitted_stages(dataset)]
    elif protocol == "revised":
        stages = [[[sys.executable, "-m", *cmd] for cmd in stage] for stage in revised_stages(dataset)]
    elif protocol == "blend_gate":
        stages = [[[sys.executable, "-m", "physicsgate.revised.gate", *cmd] for cmd in stage]
                  for stage in revised_stages(dataset)]
    else:  # row_split: row-level splits; LMB gets one name per row after preprocessing
        stages = [[[sys.executable, "-m", "physicsgate.revised.row_split", *cmd] for cmd in stage]
                  for stage in revised_stages(dataset)]
        if dataset == "LMB":
            stages.insert(1, [[sys.executable, "-m", "physicsgate.revised.row_split", "--ungroup-lmb"]])
    code = (
        "import subprocess, sys\n"
        f"stages = {stages!r}\n"
        f"logs = {str(logs)!r}\n"
        f"name = {dataset.lower()!r}\n"
        "for i, stage in enumerate(stages):\n"
        "    procs = []\n"
        "    for j, cmd in enumerate(stage):\n"
        "        print('+', ' '.join(cmd[2:]), flush=True)\n"
        "        log = sys.stdout if len(stage) == 1 else open(f'{logs}/{name}_stage{i}_{j}.log', 'w')\n"
        "        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))\n"
        "    codes = [p.wait() for p in procs]\n"
        "    if any(codes):\n"
        "        print('failed:', codes, flush=True)\n"
        "        sys.exit(1)\n"
    )
    log = open(logs / f"{dataset.lower()}.log", "w", encoding="utf-8")
    return subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["SSE", "ESTM", "LMB", "PV"],
                        choices=["SSE", "ESTM", "LMB", "PV"])
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--protocol", choices=["submitted", "revised", "row_split", "blend_gate"], default="submitted")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    if args.protocol == "row_split" and not set(args.datasets) <= {"SSE", "LMB"}:
        raise SystemExit("--protocol row_split applies to the grouped datasets only: --datasets SSE LMB")
    missing = [f for f in RAW_FILES if not (ROOT / "data" / "raw" / f).exists()]
    if missing and not args.compare_only:
        raise SystemExit(f"Missing raw data in data/raw/: {missing} (see data/sources.csv)")
    prepare_workdir(args.protocol)
    log_dir = (workdir(args.protocol) / "results" / "submitted" / "logs").relative_to(ROOT)

    if not args.compare_only:
        if args.protocol != "submitted" and "ESTM" in args.datasets:
            # the ESTM pipeline writes a timestamped copy instead of overwriting its tables, so earlier ones go first
            for path in (workdir(args.protocol) / "results" / "submitted").glob(f"{ESTM_TABLES}*.csv"):
                path.unlink()
        failed = []
        if args.parallel:
            procs = {d: run_dataset(d, args.protocol) for d in args.datasets}
            for d, p in procs.items():
                if p.wait():
                    failed.append(d)
                print(f"{d}: {'failed' if d in failed else 'done'} (log: {log_dir / (d.lower() + '.log')})")
        else:
            for d in args.datasets:
                start = time.time()
                rc = run_dataset(d, args.protocol).wait()
                if rc:
                    failed.append(d)
                print(f"{d}: {'failed' if rc else 'done'} in {time.time() - start:.0f} s "
                      f"(log: {log_dir / (d.lower() + '.log')})")
        if failed:
            raise SystemExit(f"Failed: {failed}")

    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "compare_with_data_s2.py"),
                           "--results-dir", str(workdir(args.protocol) / "results" / "submitted")], cwd=ROOT)


if __name__ == "__main__":
    main()
