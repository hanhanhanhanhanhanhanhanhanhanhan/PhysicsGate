# Revised protocol: change relative to the submitted protocol

Run with `python scripts/reproduce_submitted.py --protocol revised --parallel` (benchmark,
Supplementary Data S2) and `python scripts/reproduce_figures.py --protocol revised`
(repeated-split figures, Figures 4 and 5 and Supplementary Figures S1-S3).
The pipelines, models, splits and seeds are those of the submitted protocol;
only the correction below is applied. It is marked `REVISION` in the code.

| ID | Change | Reason | Scope |
|---|---|---|---|
| C1 | The range-risk feature R2 is computed from the unclipped physics prediction; the Physics-only and residual branches use the stabilized prediction as before. | Methods define R2 on the unclipped prediction; the submitted code computed it after clipping, so it was zero on the clipped inverse routes. | ESTM and PV (the datasets with clipped routes): `estm_pipeline.py`, `pv_pipeline.py`; repeated-split analyses `estm_stability.py` and `mechanism_estm.py`; Supplementary Figure S3 `stabilization.py` |

C1 changes 27 of the 480 evaluation units of Data S2, all of them PhysicsGate
units of clipped ESTM and PV routes; the endpoint rankings are unchanged.
