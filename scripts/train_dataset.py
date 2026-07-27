"""Shared command-line frontend used by the four dataset entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physicsgate.benchmark import run_benchmark


ROOT = Path(__file__).resolve().parents[1]


def run_cli(dataset: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / f"{dataset.lower()}.csv",
    )
    parser.add_argument("--split-id", type=int, action="append")
    parser.add_argument("--all-evaluated-splits", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "refit" / f"{dataset.lower()}_metrics.csv",
    )
    args = parser.parse_args()
    if args.all_evaluated_splits:
        config = json.loads(
            (ROOT / "configs" / "evaluated_splits.json").read_text(encoding="utf-8")
        )
        split_ids = config["datasets"][dataset]["split_ids"]
    elif args.split_id:
        split_ids = args.split_id
    else:
        raise SystemExit("Pass --split-id at least once or --all-evaluated-splits")
    result = run_benchmark(
        dataset,
        args.input,
        split_ids=split_ids,
        output_path=args.output,
        n_jobs=args.n_jobs,
    )
    print(f"Wrote {len(result)} evaluation rows to {args.output}")
