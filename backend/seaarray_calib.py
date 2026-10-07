"""Thin wrapper around SeaArrayCalib SingleWaySolver (TOA + per-session clock).

Pins this package's RayTracer (6000 m SSP extension) so solver_single_TOA
uses the web engine, then imports the current TOA solver from SeaArrayCalib.
C_REF is taken from the uploaded / simulated SSP harmonic mean, not 1500.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

_BACKEND = os.path.dirname(os.path.abspath(__file__))
from research_bridge import RESEARCH_ROOT
_SEA_SRC = str(RESEARCH_ROOT / "src")

import ray_tracing as _web_rt  # noqa: E402

sys.modules["ray_tracing"] = _web_rt
if _SEA_SRC not in sys.path:
    sys.path.insert(0, _SEA_SRC)

_IMPORT_ERROR: Optional[str] = None
try:
    import solver_single_TOA as _toa
    from solver_single_TOA import SingleWaySolver
    from ray_tracing import RayTracer, normalize_ssp_columns, ssp_equivalent_speed
except Exception as exc:
    _IMPORT_ERROR = (
        f"Cannot import SeaArrayCalib TOA solver from '{_SEA_SRC}': {exc}\n"
        f"{traceback.format_exc()}"
    )
    RayTracer = None  # type: ignore[assignment]
    SingleWaySolver = None  # type: ignore[assignment]


def _prepare_dataframe(
    obs_ship_pos: np.ndarray,
    obs_delays: np.ndarray,
    times_s: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    n = len(obs_delays)
    df = pd.DataFrame(
        {
            "Rx_E": obs_ship_pos[:n, 0],
            "Rx_N": obs_ship_pos[:n, 1],
            "Rx_U": obs_ship_pos[:n, 2],
            "delay": obs_delays[:n],
        }
    )
    if times_s is not None and len(times_s) >= n:
        t0 = pd.Timestamp("2021-01-01", tz="UTC")
        df.index = t0 + pd.to_timedelta(np.asarray(times_s[:n], dtype=float), unit="s")
    else:
        df.index = pd.date_range("2021-01-01", periods=n, freq="1s", tz="UTC")
    return df


def _load_ssp(ssp_df: pd.DataFrame) -> pd.DataFrame:
    norm = normalize_ssp_columns(ssp_df)
    return norm[["SV (m/s)", "Depth (m)"]].dropna().copy()


def run_toa_calibration(
    obs_ship_pos: np.ndarray,
    obs_delays: np.ndarray,
    ssp_df: pd.DataFrame,
    initial_guess: Optional[np.ndarray] = None,
    progress_callback=None,
    lock_eof: bool = True,
    lock_k: bool = True,
    times_s: Optional[np.ndarray] = None,
    solver_type: str = "toa",
) -> Dict[str, Any]:
    if _IMPORT_ERROR:
        return {"status": "error", "message": _IMPORT_ERROR}

    df = _prepare_dataframe(obs_ship_pos, obs_delays, times_s=times_s)
    ssp_clean = _load_ssp(ssp_df)
    tracer = RayTracer(ssp_clean, use_cache=False)
    c_eq = float(getattr(tracer, "c_eq_measured", None) or ssp_equivalent_speed(ssp_clean))
    _toa.C_REF = c_eq
    solver = SingleWaySolver(tracer)

    if progress_callback:
        progress_callback(10, f"Starting TOA fit; C_EQ={c_eq:.2f} m/s")

    unpacked = solver.solve_robust(
        df,
        initial_guess=initial_guess,
        lock_eof=lock_eof,
        lock_k=lock_k,
        progress_callback=progress_callback,
    )
    if len(unpacked) == 5:
        final_params, residuals_m, method_stats, final_methods, precision = unpacked
    else:
        final_params, residuals_m, method_stats, final_methods = unpacked[:4]
        precision = {}

    if progress_callback:
        progress_callback(100, "TOA fit complete")

    inlier = np.asarray(final_methods) != "FAIL"
    r = np.asarray(residuals_m, dtype=np.float64)
    if np.any(inlier):
        mad = np.median(np.abs(r[inlier] - np.median(r[inlier])))
        robust = mad * 1.4826
        outlier_mask = (np.abs(r) > max(3.0 * robust, 1.5)) | (~inlier)
    else:
        outlier_mask = ~inlier
    n_out = int(np.sum(outlier_mask))

    residuals_inlier = np.asarray(residuals_m, dtype=np.float64)[~outlier_mask]
    rms_m = float(np.sqrt(np.mean(residuals_inlier ** 2))) if residuals_inlier.size else float("nan")
    residuals_ms = np.asarray(residuals_m, dtype=np.float64) / c_eq * 1000.0

    p = np.asarray(final_params, dtype=np.float64)
    k_zenith = float(p[9]) if len(p) > 9 else 0.0
    wrms = float(precision.get("wrms", rms_m)) if isinstance(precision, dict) else rms_m

    return {
        "status": "success",
        "result_x": round(float(p[0]), 2),
        "result_y": round(float(p[1]), 2),
        "result_z": round(float(p[2]), 2),
        "sigma_e_m": _safe_round(precision.get("m_E") if isinstance(precision, dict) else None, 4),
        "sigma_n_m": _safe_round(precision.get("m_N") if isinstance(precision, dict) else None, 4),
        "sigma_u_m": _safe_round(precision.get("m_U") if isinstance(precision, dict) else None, 4),
        "system_clock_bias_ms": round(float(p[3]) * 1000.0, 3),
        "clock_drift_ss": float(p[4]) if len(p) > 4 else 0.0,
        "clock_drift2_ss2": float(p[5]) if len(p) > 5 else 0.0,
        "eof1": float(p[6]) if len(p) > 6 else 0.0,
        "eof2": float(p[7]) if len(p) > 7 else 0.0,
        "zenith_k_ms_km2": round(k_zenith * 1000.0, 3),
        "c_eq_mps": round(c_eq, 3),
        "rmse_time_ms": round(rms_m / c_eq * 1000.0, 4) if np.isfinite(rms_m) else None,
        "rmse_distance_m": round(wrms, 4) if np.isfinite(wrms) else round(rms_m, 4),
        "residuals_ms": residuals_ms.tolist(),
        "residuals_m": np.asarray(residuals_m, dtype=np.float64).tolist(),
        "outlier_mask": outlier_mask.tolist(),
        "outlier_rate": f"{outlier_mask.mean() * 100:.1f}%",
        "method_stats": method_stats,
        "n_total": len(df),
        "n_clean": int(len(df) - n_out),
        "n_sessions": int(precision.get("n_sessions", 1)) if isinstance(precision, dict) else 1,
        "lock_k": bool(lock_k),
        "lock_eof": bool(lock_eof),
        "solver_type": solver_type,
    }


def _safe_round(val, ndigits: int):
    try:
        v = float(val)
        return round(v, ndigits) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None
