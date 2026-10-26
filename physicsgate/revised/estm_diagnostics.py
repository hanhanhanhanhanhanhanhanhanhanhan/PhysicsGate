"""ESTM analyses of Supplementary Tables S18 and S19.

Runs the revised ESTM pipeline (``physicsgate.revised.estm_pipeline``) with one
or more of the following changes and everything else unchanged:

- ``--split formula_group`` (Table S19): each outer split holds out one third
  of the chemical formulae (GroupKFold over ``formula_id``, shuffled with the
  outer seed), so that test compositions are absent from training, instead of
  the row-level 3-fold interpolation of the source paper. The inner folds of the
  base, residual and stacking models are grouped by formula as well; the gate
  already uses formula-grouped folds.
- ``--representation element_stats`` (Table S19): the composition is described
  by statistics of elemental properties instead of the element fractions: the
  fraction-weighted mean and standard deviation and the minimum, maximum and
  range over the elements present, of the atomic number, atomic mass, Pauling
  electronegativity, atomic radius, period, group, Mendeleev number and melting
  point (pymatgen), together with the number of elements and the temperature.
- ``--base github_all`` (Table S18): every XGBoost model of the branches uses
  the setting of the ESTM authors' public code (github.com/KRICT-DATA/SIMD):
  700 trees of depth 7, all other parameters at their defaults (squared error).
- ``--aux-params CSV`` (Table S18): the auxiliary models are tuned on the
  training rows of each outer fold by 3-fold cross-validation, starting from the
  setting of the property, split and fold in CSV and comparing it with 20
  settings of the grid ``LOCAL_STEPS`` around it; as Direct the property keeps
  the setting of ``--base``. ``configs/estm_auxiliary_settings.csv`` holds the
  settings selected in the first stage, ``--base tuned``.
- ``--base tuned``: first stage of the auxiliary tuning. The Direct and
  auxiliary models are tuned on the training rows of each outer fold: the
  submitted setting and 20 random settings of ``SPACE`` are compared by 3-fold
  cross-validation. The selected settings are written to
  ``<WORKDIR>/results/submitted/estm_tuned_params.csv``.
- ``--tune-residual`` (Table S18): the residual model is tuned in the same way,
  on its inputs and targets, over ``RESIDUAL_SPACE``.
- ``--base-cache``: the out-of-fold and test predictions of the Direct and
  auxiliary models are stored per outer split, fold and property under
  ``results/estm_base_cache/`` and read back in later runs, so that a second
  run (for example with the gate of ``physicsgate.revised.gate``) refits only
  the residual model and the gate. ``--tune-workers N`` scores the candidate
  settings in N processes.

    python -m physicsgate.revised.estm_diagnostics [options] -- [pipeline arguments]
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, ParameterSampler, cross_val_score

from physicsgate.submitted.common import meta_selection
from physicsgate.submitted.estm import base_models, features
from physicsgate.submitted.paths import WORKDIR

PROPERTIES = ("Z", "atomic_mass", "X", "atomic_radius", "row", "group", "mendeleev_no", "melting_point")
TRAIN_GROUPS: dict[str, np.ndarray] = {}
CACHE = Path(__file__).resolve().parents[2] / "results" / "estm_base_cache"


# -------- formula-held-out outer splits, with formula-grouped inner folds


def formula_group_splits(frame, *, split_mode, n_splits, random_state):
    splitter = GroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    return [(np.asarray(train, dtype=int), np.asarray(test, dtype=int))
            for train, test in splitter.split(frame, groups=frame["formula_id"].to_numpy())]


class GroupedKFold:
    """KFold replacement that groups the training rows of the current outer fold by formula."""

    def __init__(self, n_splits, shuffle=False, random_state=None):
        self.n_splits, self.random_state = n_splits, random_state

    def split(self, X, y=None, groups=None):
        groups = TRAIN_GROUPS["current"]
        if len(groups) != len(X):
            raise RuntimeError("inner folds are only expected on the training rows of an outer fold")
        return GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state).split(X, groups=groups)


def track_groups(evaluate_fold):
    def wrapper(**kwargs):
        TRAIN_GROUPS["current"] = kwargs["data"]["formula_id"].to_numpy()[kwargs["train_idx"]]
        return evaluate_fold(**kwargs)
    return wrapper


# -------- element-property representation


def element_table(symbols: list[str]) -> pd.DataFrame:
    from pymatgen.core import Element

    rows = []
    for symbol in symbols:
        element = Element(symbol)
        rows.append({p: getattr(element, p, None) for p in PROPERTIES})
    table = pd.DataFrame(rows, index=symbols).apply(pd.to_numeric, errors="coerce")
    return table.fillna(table.mean())  # a few heavy elements lack a melting point or radius


def element_stats(frame: pd.DataFrame):
    columns = features.infer_element_columns(frame)
    fractions = frame[columns].to_numpy(dtype=float)
    fractions = fractions / fractions.sum(axis=1, keepdims=True)
    table = element_table([c.split("_")[2] for c in columns]).to_numpy(dtype=float)  # elements x properties
    present = fractions > 0
    mean = fractions @ table
    std = np.sqrt(np.maximum(fractions @ table ** 2 - mean ** 2, 0.0))
    masked = np.where(present[:, :, None], table[None, :, :], np.nan)
    low, high = np.nanmin(masked, axis=1), np.nanmax(masked, axis=1)
    X = np.column_stack([mean, std, low, high, high - low, present.sum(axis=1), frame["temperature_K"].to_numpy(dtype=float)])
    names = [f"{s}_{p}" for s in ("mean", "std", "min", "max", "range") for p in PROPERTIES] + ["n_elements", "temperature_K"]
    return X, names


def _fold_locals() -> dict:
    """Local variables of the pipeline's evaluate_fold, however many wrappers lie in between."""
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_name != "evaluate_fold":
        frame = frame.f_back
    return frame.f_locals if frame is not None else {}


# -------- tuned models

SPACE = {"n_estimators": [500, 1000, 2000], "max_depth": [4, 6, 8, 10, 12], "learning_rate": [0.02, 0.045, 0.08],
         "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.5, 0.7, 0.85, 1.0], "min_child_weight": [1, 2, 5],
         "reg_lambda": [0.3, 1.0, 3.0]}
RESIDUAL_SPACE = {
    "n_estimators": [300, 500, 700, 1000, 1500, 2000],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.5, 0.7, 0.85, 1.0],
    "colsample_bynode": [0.5, 0.7, 0.85, 1.0],
    "min_child_weight": [1, 2, 3, 5, 7, 10, 20],
    "gamma": [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
    "reg_alpha": [0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0],
    "reg_lambda": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    "grow_policy": ["depthwise", "lossguide"],
    "max_leaves": [0, 15, 31, 63, 127],
    "max_bin": [64, 128, 256, 512],
    "max_delta_step": [0, 1, 5],
}
# steps around a setting: factors for trees, learning rate and L2 penalty; offsets for depth, row and column sampling
# and minimum child weight
LOCAL_STEPS = dict(trees=(0.5, 1.0, 1.5), lr=(0.5, 1.0, 1.5), l2=(0.25, 1.0, 4.0), depth=(-2, 0, 2),
                   rows=(-0.1, 0.0, 0.1), cols=(-0.2, 0.0, 0.2), child=(-1, 0, 2))
TUNE_WORKERS = 1
TUNED: dict[tuple, dict] = {}
TUNE_BASE = False
TUNE_RESIDUAL = False
AUX_PARAMS: dict[tuple, dict] = {}


def inner_partition(X_train, n_splits, random_state):
    return list(base_models.KFold(n_splits=min(n_splits, len(X_train)), shuffle=True,
                                  random_state=random_state).split(X_train))


def fold_oof_and_test(estimator, X_train, y_train, X_test, *, n_splits, random_state):
    """``oof_and_test_predictions`` of the pipeline: out-of-fold predictions and a refit on all training rows."""
    folds = inner_partition(X_train, n_splits, random_state)
    oof = np.full(y_train.shape, np.nan, dtype=float)
    for fold_index, (fit_idx, valid_idx) in enumerate(folds):
        model = clone(estimator)
        try:
            model.set_params(random_state=random_state + fold_index * 17)
        except ValueError:
            pass
        model.fit(X_train[fit_idx], y_train[fit_idx])
        oof[valid_idx] = base_models._as_array(model.predict(X_train[valid_idx]))
    test = base_models._as_array(clone(estimator).fit(X_train, y_train).predict(X_test))
    return oof, test


def _cv_score(estimator, X, y, folds) -> float:
    return float(cross_val_score(estimator, X, y, cv=folds, scoring="neg_mean_squared_error").mean())


def local_space(start: dict) -> dict:
    """A grid around a setting: one step either side in each of its hyperparameters."""
    def grid(values):
        return sorted(set(values))
    s = LOCAL_STEPS
    return {"n_estimators": grid(int(round(start["n_estimators"] * f)) for f in s["trees"]),
            "max_depth": grid(max(2, int(start["max_depth"]) + d) for d in s["depth"]),
            "learning_rate": grid(round(start["learning_rate"] * f, 5) for f in s["lr"]),
            "subsample": grid(min(1.0, max(0.3, round(start["subsample"] + d, 3))) for d in s["rows"]),
            "colsample_bytree": grid(min(1.0, max(0.1, round(start["colsample_bytree"] + d, 3))) for d in s["cols"]),
            "min_child_weight": grid(max(1, int(start["min_child_weight"]) + d) for d in s["child"]),
            "reg_lambda": grid(round(start["reg_lambda"] * f, 5) for f in s["l2"])}


def sample_settings(space: dict, n: int, seed: int) -> list[dict]:
    """n random settings; each value drawn independently when the grid is too large for ParameterSampler."""
    if np.prod([float(len(v)) for v in space.values()]) < 2 ** 31 - 1:
        return list(ParameterSampler(space, n_iter=n, random_state=seed))
    rng = np.random.default_rng(seed)
    return [{k: v[int(rng.integers(len(v)))] for k, v in space.items()} for _ in range(n)]


def tune(estimator, X, y, key, space) -> dict:
    """The current setting of the model or one of 20 random settings, whichever has the lowest 3-fold CV error."""
    seed = zlib.crc32("|".join(map(str, key)).encode())
    current = {k: v for k, v in estimator.get_params().items() if k in space}
    candidates = [current, *sample_settings(space, 20, seed)]
    folds = KFold(n_splits=3, shuffle=True, random_state=seed)
    if TUNE_WORKERS > 1:
        from joblib import Parallel, delayed
        threads = max(1, (os.cpu_count() or 32) // TUNE_WORKERS)
        scores = Parallel(n_jobs=TUNE_WORKERS)(
            delayed(_cv_score)(clone(estimator).set_params(**c, n_jobs=threads), X, y, folds) for c in candidates)
    else:
        scores = [_cv_score(clone(estimator).set_params(**c), X, y, folds) for c in candidates]
    return candidates[int(np.argmax(scores))]


def tuned_oof_and_test(estimator, X_train, y_train, X_test, **kwargs):
    caller = _fold_locals()
    key = None
    if TUNE_BASE and X_train is caller.get("X_train"):  # a Direct or auxiliary model
        key = (caller["outer_seed"], caller["fold_id"], caller["column"])  # the same for every endpoint of a fold
    elif TUNE_RESIDUAL and X_train is caller.get("residual_features_train"):  # the residual model of an endpoint
        key = (caller["outer_seed"], caller["fold_id"], f"residual:{caller['target']}")
    if key is not None:
        if key not in TUNED:
            residual = key[2].startswith("residual:")
            TUNED[key] = tune(estimator, X_train, y_train, key, RESIDUAL_SPACE if residual else SPACE)
        estimator = clone(estimator).set_params(**TUNED[key])
    return fold_oof_and_test(estimator, X_train, y_train, X_test, **kwargs)


def cached(inner, directory: Path):
    """Read the Direct and auxiliary predictions of a fold from disk, computing and storing them once."""
    def wrapper(estimator, X_train, y_train, X_test, **kwargs):
        caller = _fold_locals()
        if X_train is not caller.get("X_train"):  # the residual model
            return inner(estimator, X_train, y_train, X_test, **kwargs)
        path = directory / f"{caller['outer_seed']}_{caller['fold_id']}_{caller['column']}.npz"
        if path.exists():
            stored = np.load(path)
            return stored["oof"], stored["test"]
        oof, test = inner(estimator, X_train, y_train, X_test, **kwargs)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(path, oof=oof, test=test)
        return oof, test
    return wrapper


def load_aux_params(path) -> None:
    """Settings of ``--base tuned`` (``estm_tuned_params.csv``, column ``model``) or of the configs file
    (column ``target``)."""
    for row in pd.read_csv(path).to_dict("records"):
        name = row.pop("model") if "model" in row else row.pop("target")
        key = (int(row.pop("outer_seed")), int(row.pop("fold_id")), name)
        AUX_PARAMS[key] = {k: (int(v) if k in ("n_estimators", "max_depth", "min_child_weight") else float(v))
                           for k, v in row.items()}


def tuned_auxiliaries(inner, directory: Path):
    """Auxiliary models tuned around their first-stage settings; the Direct model of the endpoint goes to ``inner``."""
    def compute(estimator, X_train, y_train, X_test, **kwargs):
        caller = _fold_locals()
        key = (caller["outer_seed"], caller["fold_id"], caller["column"])
        tuned = (*key[:2], f"aux:{key[2]}")
        if tuned not in TUNED:
            start = AUX_PARAMS[key]
            TUNED[tuned] = {**start, **tune(clone(estimator).set_params(**start), X_train, y_train, tuned,
                                            local_space(start))}
        return fold_oof_and_test(clone(estimator).set_params(**TUNED[tuned]), X_train, y_train, X_test, **kwargs)

    auxiliary = cached(compute, directory)

    def wrapper(estimator, X_train, y_train, X_test, **kwargs):
        caller = _fold_locals()
        if X_train is caller.get("X_train") and caller.get("column") != caller.get("target"):
            return auxiliary(estimator, X_train, y_train, X_test, **kwargs)
        return inner(estimator, X_train, y_train, X_test, **kwargs)
    return wrapper


def write_tuned() -> None:
    rows = [{"outer_seed": k[0], "fold_id": k[1], "model": k[2], **v} for k, v in TUNED.items()]
    pd.DataFrame(rows).to_csv(WORKDIR / "results" / "submitted" / "estm_tuned_params.csv", index=False)


# -------- the XGBoost setting of the ESTM authors' public code

GITHUB = {"max_depth": 7, "n_estimators": 700}


def github_all_builder(*, random_state, n_jobs, base_model):
    estimator = features.build_xgb_estimator(random_state=random_state, n_jobs=n_jobs, **GITHUB)
    return estimator, "xgb_github_700_depth7", str(GITHUB)


def main() -> None:
    global TUNE_BASE, TUNE_RESIDUAL, TUNE_WORKERS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["paper_interpolation", "formula_group"], default="paper_interpolation")
    parser.add_argument("--representation", choices=["element_fractions", "element_stats"], default="element_fractions")
    parser.add_argument("--base", choices=["fixed", "tuned", "github_all"], default="fixed")
    parser.add_argument("--aux-params", help="first-stage settings of the auxiliary models, tuned around them")
    parser.add_argument("--tune-residual", action="store_true")
    parser.add_argument("--base-cache", action="store_true")
    parser.add_argument("--tune-workers", type=int, default=1)
    args, rest = parser.parse_known_args()
    rest = [a for a in rest if a != "--"]
    if args.split == "formula_group":
        features.split_indices = formula_group_splits
        base_models.KFold = GroupedKFold
        meta_selection.KFold = GroupedKFold
    if args.representation == "element_stats":
        features.build_feature_matrix = element_stats
    if args.base == "github_all":
        base_models.build_base_tree_estimator = github_all_builder
    TUNE_BASE, TUNE_RESIDUAL, TUNE_WORKERS = args.base == "tuned", args.tune_residual, args.tune_workers
    if TUNE_BASE or TUNE_RESIDUAL:
        base_models.oof_and_test_predictions = tuned_oof_and_test
        atexit.register(write_tuned)
    if args.base_cache:
        directory = CACHE / f"{args.base}_{args.representation}_{args.split}"
        base_models.oof_and_test_predictions = cached(base_models.oof_and_test_predictions, directory)
    if args.aux_params:
        load_aux_params(args.aux_params)
        directory = CACHE / f"aux_tuned_local_big_{args.representation}_{args.split}"
        base_models.oof_and_test_predictions = tuned_auxiliaries(base_models.oof_and_test_predictions, directory)
    name = "physicsgate.revised.estm_pipeline"
    sys.argv = [name, *rest]
    module = importlib.import_module(name)  # imported after the patches, which it binds by name
    if args.split == "formula_group":
        module.evaluate_fold = track_groups(module.evaluate_fold)
    module.main()


if __name__ == "__main__":
    main()
