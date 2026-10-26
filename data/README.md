# Data

Raw datasets are not redistributed. Download them from the sources in
`sources.csv` and place them in `data/raw/` under these names:

| Dataset | File |
|---|---|
| SSE | `SSE.xlsx` |
| ESTM | `estm.csv` |
| LMB | `Energy density.xlsx` |
| PV | `Perovskite_database_content_all_data.csv` |

`python scripts/reproduce_submitted.py` processes the raw files itself and
writes the processed tables (`estm_clean.csv`, `lmb_clean.csv`, `pv_clean.csv`)
to `data/processed/`. SSE is read directly from `SSE.xlsx`.
