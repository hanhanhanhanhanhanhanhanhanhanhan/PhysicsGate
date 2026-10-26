"""Row-level splits in place of the grouped splits of SSE and LMB (diagnostic).

Split-sensitivity diagnostic for the comparison with the source papers, which
did not split by material or chemical system. Every ``GroupShuffleSplit``
becomes a ``ShuffleSplit`` over rows with the same test fraction and seed, and
every ``GroupKFold`` becomes a shuffled ``KFold`` over rows. Everything else is
the revised protocol.

    python -m physicsgate.revised.row_split <module> [args]
    python -m physicsgate.revised.row_split --ungroup-lmb

``--ungroup-lmb`` gives each row of the processed LMB table its own name, so
that the pipeline's check for chemical systems shared by the training and test
sets does not apply.
"""

from __future__ import annotations

import os
import runpy
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit, _split

INNER_SEED = 0


def _row_shuffle_split(self, X, y=None, groups=None):
    splitter = ShuffleSplit(n_splits=self.n_splits, test_size=self.test_size,
                            train_size=self.train_size, random_state=self.random_state)
    return splitter.split(X)


def _row_kfold(self, X, y=None, groups=None):
    return KFold(n_splits=self.n_splits, shuffle=True, random_state=INNER_SEED).split(X)


def enable() -> None:
    _split.GroupShuffleSplit.split = _row_shuffle_split
    _split.GroupKFold.split = _row_kfold


def ungroup_lmb() -> None:
    from physicsgate.submitted.lmb.common import DATA_PROCESSED_DIR, NAME_COLUMN

    path = DATA_PROCESSED_DIR / "lmb_clean.csv"
    data = pd.read_csv(path)
    data[NAME_COLUMN] = [f"row{i}" for i in np.arange(len(data))]
    data.to_csv(path, index=False)
    print(f"Ungrouped {path}: {len(data)} rows, one name per row")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--ungroup-lmb":
        ungroup_lmb()
        return
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    os.environ["PHYSICSGATE_LAUNCHER"] = "physicsgate.revised.row_split"  # nested runs use the same launcher
    enable()
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
