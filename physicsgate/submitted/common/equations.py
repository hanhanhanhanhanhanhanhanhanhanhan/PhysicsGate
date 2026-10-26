"""Physical equations shared across datasets.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/physstack/equations.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np
from numpy.typing import ArrayLike, NDArray

ARRHENIUS_KB_EV_PER_K = 8.617333262e-5
