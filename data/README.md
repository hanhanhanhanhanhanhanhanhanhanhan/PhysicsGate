# Data

Raw datasets are not redistributed. Download them from the links in
`sources.csv`, place them in `data/raw/`, and run:

```powershell
python scripts/preprocess_sse.py  --input data/raw/SSE.xlsx
python scripts/preprocess_estm.py --input data/raw/estm.csv
python scripts/preprocess_lmb.py  --input "data/raw/Energy density.xlsx"
python scripts/preprocess_pv.py   --input data/raw/Perovskite_database_content_all_data.csv
```

Each command writes a processed CSV and schema JSON to `data/processed/`.
