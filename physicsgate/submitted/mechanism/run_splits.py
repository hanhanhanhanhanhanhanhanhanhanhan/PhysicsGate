"""Run the per-split gate diagnostics one outer split at a time.

Submitted-protocol driver, adapted from
``figma5/scripts/run_figure4_mechanism_seed_manifest.py`` in the original
analysis code: the dataset runners are imported as package modules instead of
by file path. Each split runs in a fresh subprocess and writes to its own
directory, so long runs are resumable. Subprocesses are started through the
launcher named in ``PHYSICSGATE_LAUNCHER`` when it is set.

    python -m physicsgate.submitted.mechanism.run_splits --dataset ESTM \
        --seed-list <file> --output-dir <dir> [--workers N] [--module <runner>]
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODULES = {
    "SSE": "physicsgate.submitted.mechanism.sse",
    "ESTM": "physicsgate.submitted.mechanism.estm",
    "LMB": "physicsgate.submitted.mechanism.lmb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(MODULES), required=True)
    parser.add_argument("--module", help="runner module (default: the submitted runner of the dataset)")
    parser.add_argument("--seed-list")
    parser.add_argument("--output-dir")
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def read_seeds(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8-sig").strip()
    seeds = [int(value.strip()) for value in text.replace("\n", ",").split(",") if value.strip()]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds in {path}")
    return seeds


def worker(module_name: str, dataset: str, seed: int, output_dir: Path) -> None:
    module = importlib.import_module(module_name)
    module.FROZEN_SEEDS = (seed,)
    module.OUT_DIR = output_dir
    module.SAMPLE_ROWS.clear()
    module.ABLATION_ROWS.clear()
    module.FIXED_BLEND_ROWS.clear()
    module.main()
    (output_dir / "_COMPLETE.txt").write_text(
        f"dataset={dataset}\nseed={seed}\n", encoding="ascii"
    )


def canonical_output_name(name: str) -> str:
    return name.replace("Frozen5_Rerun", "50Seed_Rerun")


def merge_outputs(output_dir: Path) -> None:
    grouped: dict[str, list[Path]] = {}
    for seed_dir in sorted((output_dir / "by_seed").glob("seed_*")):
        if not (seed_dir / "_COMPLETE.txt").exists():
            continue
        for path in seed_dir.glob("*.csv"):
            grouped.setdefault(canonical_output_name(path.name), []).append(path)
    for name, paths in grouped.items():
        frames = [pd.read_csv(path) for path in paths if path.stat().st_size > 0]
        if not frames:
            continue
        pd.concat(frames, ignore_index=True, sort=False).to_csv(
            output_dir / name, index=False
        )


def write_run_status(
    dataset: str, seeds: list[int], output_dir: Path
) -> tuple[int, int]:
    rows = []
    complete_count = 0
    for seed in seeds:
        seed_dir = output_dir / "by_seed" / f"seed_{seed}"
        complete = (seed_dir / "_COMPLETE.txt").exists()
        complete_count += int(complete)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "complete": complete,
                "seed_output_dir": str(seed_dir),
            }
        )
    with (output_dir / f"Figure4_{dataset}_50Seed_RunStatus.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return complete_count, len(seeds)


def orchestrate(args: argparse.Namespace, module_name: str) -> None:
    if not args.seed_list or not args.output_dir:
        raise ValueError("--seed-list and --output-dir are required")
    seed_list = Path(args.seed_list).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = read_seeds(seed_list)
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]

    pending: list[tuple[int, int, Path]] = []
    for position, seed in enumerate(seeds, start=1):
        seed_dir = output_dir / "by_seed" / f"seed_{seed}"
        complete = seed_dir / "_COMPLETE.txt"
        if complete.exists():
            print(f"[{args.dataset} {position}/{len(seeds)}] seed={seed} already complete")
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        pending.append((position, seed, seed_dir))

    launcher = os.environ.get("PHYSICSGATE_LAUNCHER")
    prefix = [sys.executable, "-m", *([launcher] if launcher else []), "physicsgate.submitted.mechanism.run_splits"]
    worker_count = max(1, int(args.workers))
    for start in range(0, len(pending), worker_count):
        batch = pending[start : start + worker_count]
        processes: list[tuple[int, int, subprocess.Popen[bytes]]] = []
        for position, seed, seed_dir in batch:
            command = [*prefix, "--dataset", args.dataset, "--module", module_name,
                       "--worker-seed", str(seed), "--output-dir", str(seed_dir)]
            print(f"[{args.dataset} {position}/{len(seeds)}] seed={seed} starting", flush=True)
            processes.append((position, seed, subprocess.Popen(command)))
        failed = []
        for _, seed, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failed.append((seed, return_code))
        if failed:
            raise RuntimeError(f"{args.dataset} worker failures: {failed}")
        complete_count, total = write_run_status(args.dataset, seeds, output_dir)
        print(f"[{args.dataset}] checkpoint {complete_count}/{total} seeds", flush=True)

    merge_outputs(output_dir)
    complete_count, total = write_run_status(args.dataset, seeds, output_dir)
    if complete_count != total:
        raise RuntimeError(f"{args.dataset}: only {complete_count}/{total} seeds complete")
    print(f"[{args.dataset}] all {total} seeds complete: {output_dir}")


def main() -> None:
    args = parse_args()
    module_name = args.module or MODULES[args.dataset]
    if args.worker_seed is not None:
        if not args.output_dir:
            raise ValueError("--output-dir is required in worker mode")
        worker(module_name, args.dataset, args.worker_seed, Path(args.output_dir).resolve())
        return
    orchestrate(args, module_name)


if __name__ == "__main__":
    main()
