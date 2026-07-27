from __future__ import annotations

import json

import numpy as np
import pandas as pd

from physicsgate.preprocessing import preprocess_sse


def test_revised_sse_preprocessor_writes_schema(tmp_path):
    rows = 12
    frame = pd.DataFrame({"name": [f"material-{i // 2}" for i in range(rows)]})
    for index in range(25):
        frame[f"descriptor_{index:02d}"] = np.linspace(0, 1, rows) + index
    frame["1000*1/T"] = np.linspace(2.0, 4.0, rows)
    frame["lnsigmaT"] = np.linspace(-8, -3, rows)
    frame["MNR-Ea"] = np.linspace(0.1, 0.5, rows)
    frame["MNR-lnsigma0"] = np.linspace(1, 3, rows)
    input_path = tmp_path / "SSE.xlsx"
    output_path = tmp_path / "sse.csv"
    frame.to_excel(input_path, index=False)

    _, schema_path = preprocess_sse(input_path, output_path)

    cleaned = pd.read_csv(output_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["dataset"] == "SSE"
    assert len(schema["feature_columns"]) == 26
    assert schema["target_columns"] == ["ln_sigma_T", "Ea", "ln_sigma0"]
    assert schema["group_column"] == "group_id"
    assert np.isfinite(cleaned["T_K"]).all()
