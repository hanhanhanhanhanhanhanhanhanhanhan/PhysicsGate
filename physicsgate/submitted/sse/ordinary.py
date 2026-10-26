"""SSE Ordinary stacking for the evaluated outer splits.

Submitted protocol: grouped 75:25 outer split by material, the ordinary
base-model pool with a Ridge meta-learner fitted on grouped OOF predictions,
and random state ``seed + 303 + 31 * target_index``.
"""

from __future__ import annotations

import argparse

import pandas as pd

from physicsgate.submitted.paths import WORKDIR
from physicsgate.submitted.sse import pipeline as base
from physicsgate.submitted.sse.target_adaptive import (
    TARGET_COLUMNS,
    TARGET_ORDER,
    _ordinary_stacking_oof_and_test,
    _split_group_indices,
)

OUTPUT = WORKDIR / "results" / "submitted" / "sse_ordinary_stacking.csv"


def run(seeds: list[int]) -> pd.DataFrame:
    data, feature_columns = base.load_and_prepare_data()
    rows = []
    for seed in seeds:
        for target in TARGET_ORDER:
            target_col = TARGET_COLUMNS[target]
            valid = data.loc[data[target_col].notna()].reset_index(drop=True)
            groups = valid["Materials"].astype(str).to_numpy(dtype=object)
            train_idx, test_idx = _split_group_indices(groups, test_size=0.25, random_state=seed)
            train = valid.iloc[train_idx].reset_index(drop=True)
            test = valid.iloc[test_idx].reset_index(drop=True)
            y_test = test[target_col].to_numpy(dtype=float)
            _, prediction = _ordinary_stacking_oof_and_test(
                X_train=train[feature_columns].to_numpy(dtype=float),
                y_train=train[target_col].to_numpy(dtype=float),
                groups_train=train["Materials"].astype(str).to_numpy(dtype=object),
                X_test=test[feature_columns].to_numpy(dtype=float),
                random_state=seed + 303 + TARGET_ORDER.index(target) * 31,
                inner_cv=base.INNER_CV,
            )
            rows.append({"seed": seed, "target_code": target, "method": "Ordinary stacking",
                         "metric_level": "row", **base.metric_dict(y_test, prediction)})
            print(f"SSE Ordinary stacking seed={seed} target={target} done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    run(args.seeds).to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
