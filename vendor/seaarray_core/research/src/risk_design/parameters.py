"""Physical parameters for the minimum identifiable model.

Only a uniform sound-speed offset and one effective-delay gauge are estimated
until a development block says otherwise. Effective delay is not a proven pair
of transmit and receive clocks.
"""
from __future__ import annotations

import numpy as np

COORD_NAMES = ("E2", "N2", "U2", "E5", "N5", "U5", "E6", "N6", "U6")
ETA_NAMES = ("a0_mps", "B_s", "D_s_per_hour", "dB2_s", "dB6_s")
PRIOR_SIGMA = {
    "a0_mps": 2.0,
    "B_s": 0.05,
    "D_s_per_hour": 1.0e-3,
    "dB2_s": 0.05,
    "dB6_s": 0.05,
    "sound_rate_mps_per_hour": 2.0,
    "clock_quadratic_s_per_hour2": 0.01,
}


def prior_precision() -> np.ndarray:
    return np.array([1.0 / PRIOR_SIGMA[name] ** 2 for name in ETA_NAMES], dtype=np.float64)


def unpack_beacons(x: np.ndarray) -> dict[int, np.ndarray]:
    values = np.asarray(x, dtype=np.float64).reshape(9)
    return {2: values[0:3].copy(), 5: values[3:6].copy(), 6: values[6:9].copy()}


def pack_beacons(positions: dict[int, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(positions[b], dtype=np.float64).reshape(3) for b in (2, 5, 6)])
