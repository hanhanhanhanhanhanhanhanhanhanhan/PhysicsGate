"""Multitask baselines that predict all properties of a domain jointly.

For every evaluated outer split, three models are trained on the training
rows and inputs of Direct and evaluated on the same test rows:

- ``mtl_mlp``: one MLP with shared hidden layers and one output per property;
- ``stl_mlp``: the same architecture and optimizer with a single output, one
  model per property;
- ``stl_forest``: a random forest per property, a single-task check of the
  strength of Direct.

Each property is standardized on the training rows, so that no property
dominates the joint loss; MLP inputs are quantile-normalized. Hyperparameters are selected on training data only
(grouped 3-fold CV for SSE and LMB, a 20% holdout of the training rows for ESTM
and PV): by the mean standardized MSE over properties for the multitask models
and per property for the single-task models. The MLPs are refitted with three
initializations, whose predictions are written separately. Independent fits run
in parallel worker processes (``--workers``), one core each.

Inputs are those of Direct, shared by all properties: SSE descriptors and
temperature; ESTM composition fractions and temperature; the 41 LMB features;
the PV features with each categorical feature target-encoded for every
property, the encoding fitted on the rows a model is trained on (within each
validation split and then on all training rows). For SSE, each endpoint has its own split; rows
of the training set without all three properties are left out of the
multitask fits.

Test predictions are written per split to ``<WORKDIR>/results/submitted/mtl_baseline/``.

    python -m physicsgate.revised.mtl_baseline --dataset {SSE,ESTM,LMB,PV} --seeds S [S ...]
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from physicsgate.submitted.paths import WORKDIR

OUT = WORKDIR / "results" / "submitted" / "mtl_baseline"
MLP_GRID = [{"hidden_layer_sizes": h, "alpha": a} for h, a in itertools.product([(64,), (128, 64), (256, 128)], [1e-4, 1e-2])]
FOREST_GRID = [{"max_features": f, "min_samples_leaf": leaf} for f, leaf in itertools.product(["sqrt", 0.5, 1.0], [1, 5])]
INITS = 3


# -------- units: (seed, fold, endpoints, X_train, Y_train, X_test, Y_test, groups_train)


def sse_units(seeds):
    from physicsgate.submitted.sse.data import load_and_prepare_data
    from physicsgate.submitted.sse.iterative import _direct_features, _split_group_indices
    from physicsgate.submitted.sse.target_adaptive import TARGET_COLUMNS, TARGET_ORDER

    data, features = load_and_prepare_data()
    columns = [TARGET_COLUMNS[t] for t in TARGET_ORDER]
    for seed in seeds:
        for target in TARGET_ORDER:
            valid = data.loc[data[TARGET_COLUMNS[target]].notna()].reset_index(drop=True)
            train_idx, test_idx = _split_group_indices(valid["Materials"].to_numpy(), test_size=0.25, random_state=seed)
            X = _direct_features(valid[features].to_numpy(dtype=float), valid["T_K"].to_numpy(dtype=float))
            Y = valid[columns].to_numpy(dtype=float)
            yield (seed, 0, [target], X[train_idx], Y[train_idx], X[test_idx], Y[test_idx],
                   valid["Materials"].astype(str).to_numpy()[train_idx], list(TARGET_ORDER))


def estm_units(seeds):
    from physicsgate.submitted.estm import physics as estm
    from physicsgate.submitted.estm.features import DEFAULT_INPUT_PATH, build_feature_matrix, split_indices

    frame = estm.complete_estm_frame(pd.read_csv(DEFAULT_INPUT_PATH))
    X, _ = build_feature_matrix(frame.drop(columns=["core_v2_row_index"]))
    targets = list(estm.TARGET_COLUMNS)
    Y = frame[targets].to_numpy(dtype=float)
    for seed in seeds:
        for fold, (train_idx, test_idx) in enumerate(split_indices(frame, split_mode="paper_interpolation",
                                                                   n_splits=3, random_state=seed)):
            yield seed, fold, targets, X[train_idx], Y[train_idx], X[test_idx], Y[test_idx], None, targets


def lmb_units(seeds):
    from physicsgate.submitted.lmb.common import FEATURE_COLUMNS, MASS_KG_COLUMN, NAME_COLUMN, TARGET_COLUMNS
    from physicsgate.submitted.lmb.data import DEFAULT_OUTPUT

    data = pd.read_csv(DEFAULT_OUTPUT).dropna(subset=[*TARGET_COLUMNS, MASS_KG_COLUMN, NAME_COLUMN]).reset_index(drop=True)
    groups = data[NAME_COLUMN].astype(str).to_numpy(dtype=object)
    X = data.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    targets = list(TARGET_COLUMNS)
    Y = data[targets].to_numpy(dtype=float)
    for seed in seeds:
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed).split(data, groups=groups))
        yield seed, 0, targets, X[train_idx], Y[train_idx], X[test_idx], Y[test_idx], groups[train_idx], targets


def pv_units(seeds):
    from physicsgate.submitted.pv.base_models import split_indices
    from physicsgate.submitted.pv.common import FEATURE_COLUMNS, TARGET_COLUMNS
    from physicsgate.submitted.pv.data import DEFAULT_OUTPUT

    data = pd.read_csv(DEFAULT_OUTPUT).dropna(subset=[*TARGET_COLUMNS, *FEATURE_COLUMNS]).reset_index(drop=True)
    targets = list(TARGET_COLUMNS)
    Y = data[targets].to_numpy(dtype=float)
    for seed in seeds:
        train_idx, test_idx = split_indices(len(data), test_size=0.30, random_state=seed)
        X_train_raw = data[FEATURE_COLUMNS].iloc[train_idx].reset_index(drop=True)
        X_test_raw = data[FEATURE_COLUMNS].iloc[test_idx].reset_index(drop=True)
        yield seed, 0, targets, X_train_raw, Y[train_idx], X_test_raw, Y[test_idx], None, targets


def pv_encode(X_fit_raw, Y_fit, X_apply_raw):
    """PV inputs: every categorical feature target-encoded against every property on the fitting rows."""
    from physicsgate.submitted.pv.common import CATEGORICAL_FEATURES, FEATURE_COLUMNS, encode_features_for_target

    fit_parts, apply_parts = [], []
    for k in range(Y_fit.shape[1]):
        fit_enc, apply_enc, _, _ = encode_features_for_target(
            X_fit_raw, X_apply_raw, pd.Series(Y_fit[:, k], index=X_fit_raw.index), protocol="leakage_safe")
        columns = FEATURE_COLUMNS if k == 0 else CATEGORICAL_FEATURES
        fit_parts.append(fit_enc[columns].to_numpy(dtype=float))
        apply_parts.append(apply_enc[columns].to_numpy(dtype=float))
    return np.hstack(fit_parts), np.hstack(apply_parts)


UNITS = {"SSE": sse_units, "ESTM": estm_units, "LMB": lmb_units, "PV": pv_units}
GROUPED = {"SSE", "LMB"}
ENCODERS = {"PV": pv_encode}


# -------- models


def mlp(params, seed):
    return MLPRegressor(**params, learning_rate_init=1e-3, batch_size=256, max_iter=1000, early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=20, random_state=seed)


def forest(params, seed):
    return RandomForestRegressor(n_estimators=300, **params, n_jobs=1, random_state=seed)


GRIDS = {"mlp": MLP_GRID, "forest": FOREST_GRID}


def fit_predict(kind, params, X_train, Y_train, X_test, seed):
    """Fit on standardized targets; predictions on the original scale.

    MLP inputs are mapped to normal scores by their training quantiles, which bounds test inputs outside the
    training range (grouped splits of LMB and SSE) instead of letting the network extrapolate them linearly.
    """
    y_scaler = StandardScaler().fit(Y_train)
    Z = y_scaler.transform(Y_train)
    if kind == "mlp":
        x_scaler = QuantileTransformer(n_quantiles=min(1000, len(X_train)), output_distribution="normal",
                                       random_state=0).fit(X_train)
        X_train, X_test = x_scaler.transform(X_train), x_scaler.transform(X_test)
    model = (mlp if kind == "mlp" else forest)(params, seed)
    model.fit(X_train, Z[:, 0] if Z.shape[1] == 1 else Z)
    return y_scaler.inverse_transform(np.asarray(model.predict(X_test), dtype=float).reshape(len(X_test), -1))


def validation_splits(dataset, n, groups, seed):
    if dataset in GROUPED:
        return list(GroupKFold(n_splits=3).split(np.zeros(n), groups=groups))
    train, valid = next(KFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(n)))
    return [(train, valid)]


def inputs(encode, X, Y_encode, fit, X_apply):
    """Model inputs of the fitting rows and of X_apply, with any target encoding fitted on the fitting rows."""
    if encode is None:
        return X[fit], X_apply
    return encode(X.iloc[fit].reset_index(drop=True), Y_encode[fit], X_apply.reset_index(drop=True))


def validation_error(kind, params, X, Y, Y_encode, train, valid, seed, encode):
    """Validation MSE in units of the training standard deviation of each property, averaged over properties."""
    X_fit, X_valid = inputs(encode, X, Y_encode, train, X.iloc[valid] if encode else X[valid])
    prediction = fit_predict(kind, params, X_fit, Y[train], X_valid, seed)
    return float(np.mean(((prediction - Y[valid]) / Y[train].std(axis=0)) ** 2))


def final_prediction(kind, params, X, Y, Y_encode, X_test, seed, encode):
    X_fit, X_apply = inputs(encode, X, Y_encode, np.arange(len(Y)), X_test)
    return fit_predict(kind, params, X_fit, Y, X_apply, seed)


def run_unit(dataset, unit, workers):
    seed, fold, endpoints, X_train, Y_train, X_test, Y_test, groups, targets = unit
    complete = np.all(np.isfinite(Y_train), axis=1)
    encode = ENCODERS.get(dataset)

    def rows_of(rows):
        return X_train.iloc[np.flatnonzero(rows)].reset_index(drop=True) if encode else X_train[rows]

    group_rows = lambda rows: None if groups is None else groups[rows]  # noqa: E731
    models = {}  # name -> (kind, X, Y, groups, Y for the target encoding)
    # multitask MLP: one model for all properties, trained on rows with every property
    models["mtl_mlp"] = ("mlp", rows_of(complete), Y_train[complete], group_rows(complete), Y_train[complete])
    for kind in GRIDS:  # single task: one model per endpoint, on the rows where that property is known
        for endpoint in endpoints:
            j = targets.index(endpoint)
            rows = np.isfinite(Y_train[:, j])
            models[f"stl_{kind}_{endpoint}"] = (kind, rows_of(rows), Y_train[rows][:, [j]], group_rows(rows),
                                                Y_train[rows])

    # independent fits run in parallel: first the hyperparameter selection, then the final fits
    jobs = [(name, index, train, valid)
            for name, (kind, X, Y, g, _) in models.items()
            for index in range(len(GRIDS[kind]))
            for train, valid in validation_splits(dataset, len(X), g, seed)]
    errors = Parallel(n_jobs=workers)(
        delayed(validation_error)(models[name][0], GRIDS[models[name][0]][index], models[name][1], models[name][2],
                                  models[name][4], train, valid, seed, encode)
        for name, index, train, valid in jobs)
    scores = pd.DataFrame([(name, index, e) for (name, index, _, _), e in zip(jobs, errors)],
                          columns=["name", "index", "error"]).groupby(["name", "index"])["error"].mean()
    best = {name: int(scores[name].idxmin()) for name in models}
    finals = [(name, i) for name, (kind, *_) in models.items() for i in range(INITS if kind == "mlp" else 1)]
    predictions = Parallel(n_jobs=workers)(
        delayed(final_prediction)(models[name][0], GRIDS[models[name][0]][best[name]], models[name][1],
                                  models[name][2], models[name][4], X_test, seed + 1000 * i, encode)
        for name, i in finals)

    frames = []
    for endpoint in endpoints:
        frame = pd.DataFrame({"dataset": dataset, "seed": seed, "fold": fold, "target": endpoint,
                              "y_true": Y_test[:, targets.index(endpoint)],
                              "n_train": len(X_train), "n_train_multitask": int(complete.sum())})
        for (name, i), prediction in zip(finals, predictions):
            kind = models[name][0]
            if name == f"mtl_{kind}":
                frame[f"mtl_{kind}_{i}"] = prediction[:, targets.index(endpoint)]
            elif name == f"stl_{kind}_{endpoint}":
                frame[f"stl_{kind}_{i}"] = prediction[:, 0]
        frame["params"] = str({name: GRIDS[models[name][0]][best[name]] for name in models})
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(UNITS))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=16, help="parallel fits")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for unit in UNITS[args.dataset](args.seeds):
        seed, fold, endpoints = unit[0], unit[1], unit[2]
        path = OUT / f"{args.dataset}_{seed}_{fold}_{'_'.join(endpoints) if len(endpoints) == 1 else 'all'}.csv.gz"
        if path.exists():
            continue
        run_unit(args.dataset, unit, args.workers).to_csv(path, index=False)
        print(f"{args.dataset} seed={seed} fold={fold} endpoints={endpoints} done", flush=True)


if __name__ == "__main__":
    main()
