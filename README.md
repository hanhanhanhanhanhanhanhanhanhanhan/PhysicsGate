# PhysicsGate

PhysicsGate is a reliability-guided machine-learning framework for combining
direct predictions with predictions constrained by known physical equations.
It is designed for related material or device properties whose physical
relationship is informative but whose equation-derived prediction may become
unreliable because auxiliary-model errors propagate through the equation.

## Method

For a target property, PhysicsGate combines a direct prediction and a
physics-anchored residual prediction:

$$
\hat A_{PG}
= (1-w)\hat A_D+w\hat A_R
= \hat A_D+w(\hat A_R-\hat A_D),
\quad 0 \leq w \leq 1.
$$

The residual branch first corrects the equation-derived prediction:

$$
\hat A_R=\hat A_P+\hat r.
$$

The physics-only prediction is calculated from independently predicted
auxiliary properties, and the residual correction is learned from
training-side out-of-fold (OOF) physics residuals. The sample-specific weight
is

$$
w=\frac{1}{1+e^{-g(R_1,R_2,R_3)}}.
$$

This is a sigmoid transformation, where *g* is a standardized ridge model and:

- R<sub>1</sub>: disagreement between the direct and residual branches;
- R<sub>2</sub>: risk that the physics prediction lies outside the training range;
- R<sub>3</sub>: uncertainty propagated through the physical equation.

The gate therefore learns when a physics-anchored correction is useful and
when the direct model should be retained.

![PhysicsGate framework](results/figures/Figure1.png)

## Installation

Python 3.10 or newer is recommended.

```bash
# Run these commands after cloning the repository.
cd PhysicsGate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick Start

The public API is provided by the `physicsgate` package:

```python
from physicsgate import PhysicsGate, reliability_features

features = reliability_features(
    direct=direct_oof,
    physics=physics_oof,
    residual=residual_oof,
    training_targets=y_train,
    propagated_uncertainty=uncertainty_oof,
)

gate = PhysicsGate(alpha=1.0).fit(
    y_oof=y_train,
    direct_oof=direct_oof,
    residual_oof=residual_oof,
    reliability_oof=features,
)

prediction = gate.predict(
    direct=direct_test,
    residual=residual_test,
    reliability=test_features,
)
```

All inputs used to fit the gate must be generated from the training data.
Direct, auxiliary and residual predictions supplied during training must be
OOF predictions.

## Reproduce the Paper Figures

Figures 1-3 were assembled in Origin; their plotted numerical values are
provided as compact CSV files. Figure 4, Figure 5 and Figures S1-S3 can be
regenerated directly:

```bash
python reproduce.py
```

The regenerated figures are written to `results/figures/`. To validate the
method contract:

```bash
python -m pytest
```

## Train from Source Data

The real datasets are not redistributed. Stable article identifiers,
download locations and expected filenames are listed in
[`data/sources.csv`](data/sources.csv). After downloading and preprocessing a
dataset, use its unified entrypoint:

```bash
python scripts/train_sse.py  --all-evaluated-splits
python scripts/train_estm.py --all-evaluated-splits
python scripts/train_lmb.py  --all-evaluated-splits
python scripts/train_pv.py   --all-evaluated-splits
```

Each command calls the same training implementation in
`physicsgate/benchmark.py`; only the physical equation, target definitions and
outer-split design differ. Preprocessing commands are documented in
[`data/README.md`](data/README.md).

## Benchmarks

The study evaluates PhysicsGate for four domains:

| Dataset | Domain | Physical relationship |
|---|---|---|
| SSE | Solid-state electrolytes | Arrhenius conductivity relation |
| ESTM | Thermoelectric materials | ZT = S<sup>2</sup>&sigma;T/&kappa; |
| LMB | Liquid-metal batteries | E<sub>d</sub> = QV/m = DE/m |
| PV | Photovoltaic devices | PCE = V<sub>OC</sub>J<sub>SC</sub>FF |

The benchmark reports Direct, Physics-only, Residual-only, Ordinary stacking,
PhysResStack and PhysicsGate under matched data splits and preprocessing.
PhysResStack is a comparison method; PhysicsGate is the method implemented by
this repository.

## Leakage Controls

- True validation or test auxiliary targets are never used in equations.
- Residual correction, stacking and gate fitting use training-side OOF
  predictions.
- Group-aware inner folds are used when the outer split is group-aware.
- Hyperparameters are selected using training data only.
- Held-out labels are reserved for final evaluation.

## Repository Structure

```text
physicsgate/     PhysicsGate, equations, preprocessing and training core
scripts/         Dataset preprocessing, training and figure scripts
source_data/     Numerical source data used in the paper figures
results/figures/ Reference and regenerated figures
configs/         Evaluation split identifiers and protocol configuration
data/            Dataset access instructions; no real datasets are committed
docs/            Method and reproducibility documentation
tests/           Leakage-control and method-contract tests
```

Further details are available in
[`docs/METHOD.md`](docs/METHOD.md),
[`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Citation

Citation information will be added upon publication.

## License

This project is released under the MIT License.
