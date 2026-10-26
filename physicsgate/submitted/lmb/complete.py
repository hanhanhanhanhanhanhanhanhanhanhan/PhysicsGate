"""LMB Ordinary stacking and PhysResStack for the evaluated outer splits.

Submitted protocol: reuses the training-OOF base-model selections and branch
predictions recorded by ``physicsgate.submitted.lmb.pipeline`` (run with the
same ``--reference-tag``) and fits the two stacking baselines for each split.
"""

from __future__ import annotations

import argparse
from argparse import Namespace

from physicsgate.submitted.lmb import stacking as six
from physicsgate.submitted.lmb.data import DEFAULT_OUTPUT
from physicsgate.submitted.paths import WORKDIR

OUTPUT = WORKDIR / "results" / "submitted" / "lmb_six_methods_detailed.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--reference-tag", required=True)
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    six.REFERENCE_TAG = args.reference_tag
    six.MISSING5 = seeds
    six.TOP10 = seeds
    detail = six.compute_missing(Namespace(
        input=str(DEFAULT_OUTPUT),
        test_size=0.10,
        inner_cv=5,
        tune_cv=3,
        n_jobs=-1,
        residual_search_iter=28,
        output_tag=f"{args.reference_tag}_complete",
    ))
    detail = six.add_direct_comparison(detail)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(detail)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
