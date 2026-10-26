"""Paths used by the submitted-protocol code.

``WORKDIR`` plays the role of the original analysis repository root: raw data
are read from ``WORKDIR/data/raw``, processed data are written to
``WORKDIR/data/processed`` and intermediate outputs to ``WORKDIR/results/submitted``.
Set ``PHYSICSGATE_WORKDIR`` to use another location.
"""

from __future__ import annotations

import os
from pathlib import Path

WORKDIR = Path(os.environ.get("PHYSICSGATE_WORKDIR", Path(__file__).resolve().parents[2])).resolve()
PV_WORKDIR = WORKDIR / "results" / "submitted" / "pv"
