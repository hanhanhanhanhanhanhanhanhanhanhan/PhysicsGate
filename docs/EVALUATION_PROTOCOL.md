# Evaluation protocol

## Compared methods

All 16 endpoint-route combinations compare the same six methods:

1. Direct;
2. Physics-only;
3. Residual-only;
4. Ordinary stacking;
5. PhysResStack;
6. PhysicsGate.

The principal metrics are RMSE, MAE and R<sup>2</sup>. Physics residuals and gate
diagnostics are retained where defined.

## Split designs

Exact evaluated identifiers are machine-readable in
`configs/evaluated_splits.json`.

- SSE: 75:25 material-group holdout.
- ESTM: three-fold paper-interpolation evaluation.
- LMB: 90:10 name-group holdout.
- PV: 70:30 paper-matched row holdout.

The five splits in Figures 2-3 were retained as representative visualization
splits established during method development. Figures 4-5 and the mechanism
supplements summarize the retained 50-split sets; their evaluation scope is
recorded in `source_data/provenance/Evaluation_Scope.csv`.

The compact public package omits unused detailed metric exports. Split
identifiers and the reporting scope inherited from the development workflow
are retained in `configs/evaluated_splits.json` and
`source_data/provenance/Evaluation_Scope.csv`.

## Leakage controls

- Auxiliary targets used by equations are independently predicted.
- Training predictions used by residual and second-stage models are OOF.
- Group-aware inner folds are used where the outer protocol is group-aware.
- Hyperparameters are selected with training-side data only.
- The held-out test labels are used only for reporting final metrics.
