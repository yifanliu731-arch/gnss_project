"""Travel times that keep Snell failures visible.

A straight-line fallback is recorded and must not be treated as an eigenray.
Inter-beacon times use a monotone Snell ray or one deep turning point.
"""
from __future__ import annotations

import numpy as np

from risk_design.ray_helpers import load_tracer
from risk_design.geometry import BEACONS
from risk_design.eigenray import interbeacon_time


class TravelTimeModel:
    def __init__(self, tracer=None, water_column_floor_m=None):
        self.tracer = tracer or load_tracer()
        # Fixed domain of the assumed water column, never an estimated coordinate.
        # The SSP extent supplies a computational domain, not verified bathymetry.
        self.water_column_floor_m = float(self.tracer.Z[-1] if water_column_floor_m is None
                                         else water_column_floor_m)

    def ship(self, ship_enu: np.ndarray, beacon_enu: np.ndarray, a0_mps: float) -> dict:
        self.tracer.delta_c = float(a0_mps)
        delay, method = self.tracer.calculate_travel_time_with_method_batch(ship_enu, beacon_enu)
        method = np.asarray(method)
        return {
            "delay_s": np.asarray(delay, dtype=np.float64),
            "method": method,
            "snell_ok": method == "Snell",
            "n_straight_fallback": int(np.sum(method == "StraightLine")),
            "n_fail": int(np.sum(method == "FAIL")),
        }

    def inter_beacon(self, positions: dict[int, np.ndarray], a0_mps: float) -> list[dict]:
        bottom = self.water_column_floor_m
        rows = []
        for left, right in ((5, 2), (5, 6), (2, 6)):
            ray = interbeacon_time(self.tracer, positions[left], positions[right], a0_mps, bottom)
            ray["pair"] = f"{left}-{right}"
            ray["water_column_assumption_m"] = bottom
            ray["bathymetry_verified"] = False
            ray["snell_ok"] = ray["method"] in ("snell_monotone", "snell_turning")
            rows.append(ray)
        return rows


def design_beacon_array() -> dict[int, np.ndarray]:
    """Design ranges, turned to the field azimuth.

    Beacon 5 stays at the origin. Beacon 2 lies northwest of it and beacon 6
    northeast, as in the catalog offsets. Depths and the three horizontal
    distances stay with the same beacon numbers.
    """
    from risk_design.geometry import design_positions

    raw = {beacon: np.asarray(value, dtype=np.float64).copy() for beacon, value in design_positions().items()}
    # Catalog horizontal offsets relative to beacon 5: 2 is northwest, 6 northeast.
    target = {2: np.array([-2050.0, 2200.0]), 6: np.array([2210.0, 2350.0])}
    origin = raw[5][:2]
    vector = {beacon: raw[beacon][:2] - origin for beacon in (2, 6)}
    vector[6][1] *= -1.0
    turn = np.arctan2(target[2][1], target[2][0]) - np.arctan2(vector[2][1], vector[2][0])
    rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    for beacon in (2, 6):
        raw[beacon][:2] = origin + rotation @ vector[beacon]
    return raw
