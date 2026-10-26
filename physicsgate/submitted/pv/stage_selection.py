"""Pass the PV Direct-model selections to the PV pipeline.

``physicsgate.submitted.pv.selection`` chooses each Direct base model and the
residual meta-learner by training-side OOF error. ``physicsgate.submitted.pv.pipeline``
reads the base-model choices and the list of evaluated splits from fixed
locations, and the stabilization-sensitivity run (``pv.stabilization``) reads
both choices from the same folder; this step writes all three.
"""

from __future__ import annotations

import argparse
import shutil

import pandas as pd

from physicsgate.submitted.paths import WORKDIR
from physicsgate.submitted.pv.pipeline import BY_SEED, OLD_BASE_SELECTION, SEED_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--selection-prefix", required=True)
    args = parser.parse_args()
    selection = pd.read_csv(WORKDIR / "results" / "submitted" / f"{args.selection_prefix}_base_selection.csv")
    OLD_BASE_SELECTION.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(OLD_BASE_SELECTION, index=False)
    meta = WORKDIR / "results" / "submitted" / f"{args.selection_prefix}_selection.csv"
    if meta.exists():
        pd.read_csv(meta).to_csv(OLD_BASE_SELECTION.parent / "Figure4_PV_50Seed_Rerun_Selection.csv", index=False)
    SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEED_FILE.write_text(",".join(map(str, args.seeds)) + "\n", encoding="utf-8")
    if BY_SEED.exists():
        shutil.rmtree(BY_SEED)  # the pipeline skips splits that already have results
    print(f"Wrote {OLD_BASE_SELECTION} and {SEED_FILE}")


if __name__ == "__main__":
    main()
