# Data access

Raw scientific datasets are intentionally not committed. Article identifiers,
download locations and expected local names are listed in `sources.csv`.

A DOI is a citation, not a complete data-version record. For every downloaded
file also record:

1. retrieval date and article/repository version or commit;
2. exact local filename and workbook sheet, where applicable;
3. SHA-256 checksum;
4. any author-side corrections made after download.

Place permitted files in `data/raw/`, then run the matching preprocessor:

```powershell
python scripts/preprocess_sse.py  --input data/raw/SSE.xlsx
python scripts/preprocess_estm.py --input data/raw/estm.csv
python scripts/preprocess_lmb.py  --input "data/raw/Energy density.xlsx"
python scripts/preprocess_pv.py   --input data/raw/Perovskite_database_content_all_data.csv
```

Each command writes a processed CSV and a JSON schema/provenance sidecar under
`data/processed/`. Raw and processed scientific datasets remain git-ignored.

The underlying SSE study is identified by the versioned ChemRxiv DOI
`10.26434/chemrxiv.15005550/v1`. The analysed `SSE.xlsx` is an author-curated
revised workbook rather than an unmodified DOI deposit. Its checksum and all
author-side transformations relative to the deposited source therefore remain
part of the required provenance record.
