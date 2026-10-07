"""Inter-beacon eigenrays that may turn below the deeper beacon.

The ship solver only integrates between the two endpoint depths. A deep
baseline can lie past that monotone range. The connecting ray then goes down
to a horizontal vertex and back up, which is the standard split at a turning
point. A supplied water-column floor is a modeling assumption unless supported
by bathymetry. A rejected ray carries a diagnostic chord time, never a valid
eigenray for joint estimation.
"""
from __future__ import annotations

import math

import numpy as np

from risk_design.ray_helpers import chord_time


def _speed(tracer, depth_m: float, a0_mps: float) -> float:
    return float(np.interp(depth_m, tracer.Z, tracer.V)) + float(a0_mps)


def _profile(tracer, z0: float, z1: float, a0_mps: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = (z0, z1) if z0 <= z1 else (z1, z0)
    grid = np.asarray(tracer.Z)
    depths = np.r_[lo, grid[(grid > lo) & (grid < hi)], hi]
    speeds = np.interp(depths, grid, tracer.V) + float(a0_mps)
    return depths, speeds


def _leg_to_vertex(tracer, z_end: float, z_vertex: float, a0_mps: float) -> tuple[float, float]:
    """Horizontal range and travel time from an endpoint down to a horizontal vertex."""
    if z_vertex < z_end - 1e-6:
        raise ValueError("vertex is shallower than the endpoint")
    speed_turn = _speed(tracer, z_vertex, a0_mps)
    ray = 1.0 / speed_turn
    depths, speeds = _profile(tracer, z_end, z_vertex, a0_mps)
    speeds[-1] = speed_turn
    dz = np.diff(depths)
    c1, c2 = speeds[:-1], speeds[1:]
    valid = dz > 0
    dz, c1, c2 = dz[valid], c1[valid], c2[valid]
    if not len(dz): return 0., 0.
    if np.any(c1 >= speed_turn) or np.any(c2 > speed_turn):
        return float('nan'), float('nan')
    cosine_1 = np.sqrt(np.maximum(0., 1-(ray*c1)**2))
    cosine_2 = np.sqrt(np.maximum(0., 1-(ray*c2)**2))
    # Analytically zero; sqrt(roundoff) otherwise creates derivative jumps.
    cosine_2[-1] = 0.
    gradient = (c2-c1)/dz
    flat = np.abs(gradient) < 1e-8
    horizontal = np.empty(len(dz)); travel = np.empty(len(dz))
    horizontal[flat] = ray*c1[flat]/cosine_1[flat]*dz[flat]
    travel[flat] = dz[flat]/(c1[flat]*cosine_1[flat])
    curved = ~flat
    denominator = cosine_1[curved]+cosine_2[curved]
    difference_cos = ray**2*(c2[curved]-c1[curved])*(c2[curved]+c1[curved])/denominator
    horizontal[curved] = ray*(c1[curved]+c2[curved])*dz[curved]/denominator
    travel[curved] = (np.log1p((c2[curved]-c1[curved])/c1[curved])+
                     np.log1p(difference_cos/(1+cosine_2[curved])))/gradient[curved]
    return float(np.sum(horizontal)), float(np.sum(travel))


def _turning(tracer, z_left: float, z_right: float, range_m: float, a0_mps: float) -> dict | None:
    deeper = max(z_left, z_right)
    ceiling = float(tracer.Z[-1])
    if ceiling <= deeper + 0.05:
        return None

    def spanned(vertex: float) -> tuple[float, float]:
        x1, t1 = _leg_to_vertex(tracer, z_left, vertex, a0_mps)
        x2, t2 = _leg_to_vertex(tracer, z_right, vertex, a0_mps)
        return x1 + x2, t1 + t2

    lo = deeper + 1e-3
    hi = ceiling
    reach_lo, _ = spanned(lo)
    reach_hi, _ = spanned(hi)
    if reach_hi < range_m:
        return {"reachable": False, "range_at_floor_m": reach_hi, "floor_m": hi}
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        reach, _ = spanned(mid)
        if reach < range_m:
            lo = mid
        else:
            hi = mid
    vertex = 0.5 * (lo + hi)
    reach, travel = spanned(vertex)
    return {
        "reachable": True,
        "vertex_m": float(vertex),
        "range_m": float(reach),
        "delay_s": float(travel),
        "range_error_m": float(reach - range_m),
    }


def interbeacon_time(tracer, left: np.ndarray, right: np.ndarray, a0_mps: float,
                     bottom_m: float | None = None) -> dict:
    """Travel time between two beacons.

    A monotone Snell ray is used when it reaches the horizontal range. Otherwise
    a ray with one deep vertex is used. The vertex may lie below the deeper of
    the two endpoints. It is adopted only while it stays above `bottom_m`.
    The floor is an assumed water-column limit, not measured bathymetry. A deeper
    vertex is rejected; a layered chord is returned only as a diagnostic.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    horizontal = float(np.hypot(left[0] - right[0], left[1] - right[1]))
    z_left, z_right = abs(float(left[2])), abs(float(right[2]))
    deeper = max(z_left, z_right)
    chord = float(chord_time(left, right, tracer, a0_mps))
    saved = float(tracer.delta_c)
    tracer.delta_c = float(a0_mps)
    try:
        monotone_s, _ = tracer._solve_snell_eigenray(horizontal, min(z_left, z_right), deeper)
    finally:
        tracer.delta_c = saved
    monotone_s = float(monotone_s)
    if math.isfinite(monotone_s):
        return {
            "delay_s": monotone_s,
            "method": "snell_monotone",
            "adopted": True,
            "vertex_m": None,
            "below_deeper_beacon_m": 0.0,
            "chord_s": chord,
            "monotone_s": monotone_s,
            "horizontal_m": horizontal,
        }
    turned = _turning(tracer, z_left, z_right, horizontal, a0_mps)
    if turned is None or not turned["reachable"]:
        return {
            "delay_s": chord,
            "method": "chord_unreachable",
            "adopted": False,
            "vertex_m": None,
            "below_deeper_beacon_m": None,
            "chord_s": chord,
            "monotone_s": None,
            "horizontal_m": horizontal,
            "note": "No vertex inside the sound-speed profile reaches this range.",
        }
    below = float(turned["vertex_m"] - deeper)
    floor = deeper if bottom_m is None else float(bottom_m)
    if turned["vertex_m"] > floor + 0.5:
        return {
            "delay_s": chord,
            "method": "chord_bottom_blocked",
            "adopted": False,
            "turning_delay_s": turned["delay_s"],
            "vertex_m": turned["vertex_m"],
            "below_deeper_beacon_m": below,
            "bottom_m": floor,
            "range_error_m": turned["range_error_m"],
            "chord_s": chord,
            "monotone_s": None,
            "horizontal_m": horizontal,
            "note": "Vertex exceeds the assumed water-column floor; bathymetry is not verified.",
        }
    return {
        "delay_s": turned["delay_s"],
        "method": "snell_turning",
        "adopted": True,
        "vertex_m": turned["vertex_m"],
        "below_deeper_beacon_m": below,
        "bottom_m": floor,
        "range_error_m": turned["range_error_m"],
        "chord_s": chord,
        "monotone_s": None,
        "horizontal_m": horizontal,
        "note": "One deep vertex within the assumed water column; bathymetry is not verified.",
    }
