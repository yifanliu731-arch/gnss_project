"""Local ENU + full-attitude lever arm (web copy of SeaArrayCalib kinematics math).

Origin is taken from the uploaded GNSS cloud and rounded to 0.1 deg.
ENU uses WGS-84 meridional / prime-vertical radii at that origin.
Lever arm is right-forward-up, rotated by roll/pitch/yaw into ENU.
"""
from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 0.00669437999014


def origin_from_latlon(lats, lons) -> tuple[float, float]:
    """Mean lat/lon of the track, rounded to one decimal degree."""
    lat0 = round(float(np.mean(np.asarray(lats, dtype=np.float64))), 1)
    lon0 = round(float(np.mean(np.asarray(lons, dtype=np.float64))), 1)
    return lat0, lon0


def wgs84_radii(lat_deg: float) -> tuple[float, float]:
    lat = np.radians(float(lat_deg))
    s2 = np.sin(lat) ** 2
    rm = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s2) ** 1.5
    rn = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s2)
    return float(rm), float(rn)


def latlonalt_to_enu(lat, lon, alt, ref_lat: float, ref_lon: float):
    """Vectorized WGS-84 local ENU relative to (ref_lat, ref_lon)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)
    rm, rn = wgs84_radii(ref_lat)
    east = np.radians(lon - ref_lon) * rn * np.cos(np.radians(ref_lat))
    north = np.radians(lat - ref_lat) * rm
    return east, north, alt


def enu_matrix(lats, lons, alts, ref_lat: float, ref_lon: float) -> np.ndarray:
    e, n, u = latlonalt_to_enu(lats, lons, alts, ref_lat, ref_lon)
    return np.column_stack([e, n, u])


def find_column(rows: list, names: tuple[str, ...]):
    """Return the original key matching any of ``names`` (case-insensitive)."""
    if not rows:
        return None
    keys = {str(k).strip().lower(): k for k in rows[0].keys() if k is not None}
    for name in names:
        hit = keys.get(name.strip().lower())
        if hit is not None:
            return hit
    return None


def column_values(rows: list, col, default: float = 0.0) -> np.ndarray:
    n = len(rows)
    out = np.full(n, float(default), dtype=np.float64)
    if col is None:
        return out
    for i, row in enumerate(rows):
        try:
            out[i] = float(row.get(col, default) or default)
        except (TypeError, ValueError):
            out[i] = float(default)
    return out


def build_rotation_matrix(roll: float, pitch: float, yaw: float, deg: bool = True) -> np.ndarray:
    if deg:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def apply_lever_arm(
    antenna_xyz: np.ndarray,
    lever_arm: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
    heave: float = 0.0,
) -> np.ndarray:
    """GNSS antenna ENU + RFU lever arm with full attitude → transducer ENU.

    lever_arm is [right, forward, up]; down is negative up (e.g. -29.05).
    """
    r = build_rotation_matrix(roll, pitch, yaw, deg=True)
    lx, ly, lz = float(lever_arm[0]), float(lever_arm[1]), float(lever_arm[2])
    lever_ned = np.array([ly, lx, -lz])
    rotated = r @ lever_ned
    return np.array([
        float(antenna_xyz[0]) + rotated[1],
        float(antenna_xyz[1]) + rotated[0],
        float(antenna_xyz[2]) - rotated[2] + heave,
    ])


def apply_lever_arm_batch(
    antenna_enu: np.ndarray,
    lever_arm,
    roll: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
) -> np.ndarray:
    lever = np.asarray(lever_arm, dtype=np.float64).reshape(3)
    n = antenna_enu.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        out[i] = apply_lever_arm(antenna_enu[i], lever, float(roll[i]), float(pitch[i]), float(yaw[i]))
    return out


def heading_from_track(east, north) -> np.ndarray:
    """Course over ground (deg, clockwise from north) from an ENU track."""
    e = np.asarray(east, dtype=np.float64).reshape(-1)
    n = np.asarray(north, dtype=np.float64).reshape(-1)
    return np.rad2deg(np.arctan2(np.gradient(e), np.gradient(n))) % 360.0


def interp_heading(t_query, t, heading_deg) -> np.ndarray:
    """Interpolate heading with unwrap so 359°→1° does not jump through 180°."""
    t_query = np.asarray(t_query, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    h = np.asarray(heading_deg, dtype=np.float64).reshape(-1)
    if t_query.size == 0:
        return t_query
    if t.size == 0:
        return np.zeros_like(t_query)
    if t.size == 1:
        return np.full_like(t_query, float(h[0]) % 360.0)
    unw = np.unwrap(np.deg2rad(h))
    return np.rad2deg(np.interp(t_query, t, unw)) % 360.0
