"""Portable paths, acoustic noise default and timestamp conversion.

Only helpers imported by the application are retained from the study config.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')

SIGMA_AC = 0.3

TEMPLATE_SSP = os.path.join(DATA_DIR, 'template_ssp.csv')

def to_unix_seconds(values) -> np.ndarray:
    """Convert timestamps to Unix seconds through datetime64[ns] so microsecond CSV timestamps are not divided by the wrong scale."""
    t = pd.to_datetime(values, errors='coerce')
    arr = np.asarray(t, dtype='datetime64[ns]')
    ns = arr.astype('datetime64[ns]').astype(np.int64)
    out = ns.astype(np.float64) / 1000000000.0
    nat = np.isnat(arr)
    if np.any(nat):
        out = np.array(out, copy=True)
        out[nat] = np.nan
    return out
