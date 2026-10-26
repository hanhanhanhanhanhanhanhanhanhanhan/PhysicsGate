"""ESTM ZT helpers.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/thermoelectric/run_estm_physresstack_zt.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import os
from pathlib import Path
import re
import sys
from typing import Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from scipy.stats import loguniform, uniform
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold, KFold, ParameterSampler, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = WORKDIR


def xgb_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False
