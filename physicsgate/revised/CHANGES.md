# Revised protocol: changes relative to the submitted protocol

Run with `python scripts/reproduce_submitted.py --protocol revised --parallel` (benchmark,
Supplementary Data S2) and `python scripts/reproduce_figures.py --protocol revised`
(repeated-split figures, Figures 4 and 5 and Supplementary Figures S1-S3).
The pipelines, models, splits and seeds are those of the submitted protocol;
only the corrections below are applied. They are marked `REVISION` in the code.

| ID | Change | Reason | Scope |
|---|---|---|---|
| C1 | The range-risk feature R2 is computed from the unclipped physics prediction; the Physics-only and residual branches use the stabilized prediction as before. | Methods define R2 on the unclipped prediction; the submitted code computed it after clipping, so it was zero on the clipped inverse routes. | ESTM and PV (the datasets with clipped routes): `estm_pipeline.py`, `pv_pipeline.py`; repeated-split analyses `estm_stability.py` and `mechanism_estm.py`; Supplementary Figure S3 `stabilization.py` |
| C2 | Ordinary stacking combines out-of-fold predictions of base learners fitted to the target from the inputs of Direct: Direct and ridge, random forest, extra trees and gradient boosting models (`ordinary.py`). | Methods define Ordinary stacking as a meta-learner over Direct-type base learners, as implemented for SSE and LMB; for ESTM and PV the submitted code gave the meta-learner the Direct, auxiliary and Physics-only predictions. | ESTM and PV Ordinary stacking: `estm_pipeline.py`, `pv_pipeline.py` |

C1 changes 27 of the 480 evaluation units of Data S2, all of them PhysicsGate
units of clipped ESTM and PV routes; C2 changes the 40 Ordinary stacking units
of ESTM and PV. The rank of PhysicsGate is unchanged for every endpoint.

The Methods also listed the Direct prediction among the inputs of the residual
model for every dataset; the SSE residual model receives the descriptors and
the temperature instead, and the code is unchanged. Table S16 reports SSE with
the Direct prediction added (`sse_residual.py`, `sse_direct_residual.py`).
