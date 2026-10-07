"""TOA solver with a constant horizontal sound-speed gradient and ABIC.

The estimated delay is

    t = t_ray(C(z)) + B + D1 τ + D2 (τ² − ⟨τ²⟩) + k d_h² / 1e6
        − (g_E mid_E + g_N mid_N) / C_EQ² · L

g_E and g_N are shared by every beacon. ABIC chooses the weight λ of the
Gaussian prior g ~ N(0, σ²/λ) (after scaling by G_REF). Track length is not
an ABIC input: it is an operational cost and stays in the survey-plan score.

ABIC, up to an additive constant that does not depend on λ
(Laplace approximation, prior and data sharing one variance σ²):

    Φ = ||r||² + λ ||g / G_REF||²
    ν = N + M − P
    ABIC(λ) = ν log Φ − M log λ + log det(JᵀJ + λ G_REF⁻² P_g)

M = 2 is the number of gradient components.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Prior is applied to g / G_REF so λ is O(1)–O(1e5) for g ~ 1e-5 (m/s)/m.
G_REF = 1.0e-4
_LAMBDA_GRID = np.logspace(-12.0, 0.0, 9)
_FD_STEP = 1.0
_MAX_POS_STEP = 300.0


def solve_with_abic(
    ship: np.ndarray,
    delays: np.ndarray,
    ssp_df: pd.DataFrame,
    times_s: Optional[np.ndarray] = None,
    origin_xy: Optional[Tuple[float, float]] = None,
    progress_callback: Optional[Callable] = None,
    n_outer: int = 4,
) -> Dict[str, Any]:
    """Joint TOA adjustment. ``delays`` is (N,) or (N, M)."""
    from ray_tracing import RayTracer

    ship = np.asarray(ship, dtype=np.float64)
    delays = np.asarray(delays, dtype=np.float64)
    if delays.ndim == 1:
        delays = delays[:, None]
    n_obs, n_beacons = delays.shape
    if ship.shape[0] < n_obs:
        return {"status": "error", "message": "船位数量少于时延数量"}
    ship = ship[:n_obs]
    if times_s is None or len(times_s) < n_obs:
        times = np.arange(n_obs, dtype=np.float64)
    else:
        times = np.asarray(times_s[:n_obs], dtype=np.float64)
    tau = times - float(times[0])
    tau2 = tau * tau - float(np.mean(tau * tau))

    if origin_xy is None:
        origin = (float(np.mean(ship[:, 0])), float(np.mean(ship[:, 1])))
    else:
        origin = (float(origin_xy[0]), float(origin_xy[1]))

    tracer = RayTracer(ssp_df, use_cache=True)
    c_eq = float(getattr(tracer, "c_eq_measured", None) or tracer.equivalent_speed_global())
    if not np.isfinite(c_eq) or c_eq <= 0:
        c_eq = 1500.0
    c2 = c_eq * c_eq

    # Per beacon: x, y, z, B, D1, D2, k. Then shared g_E, g_N.
    p_beacon = 7
    p_dim = p_beacon * n_beacons + 2
    theta = np.zeros(p_dim, dtype=np.float64)
    for j in range(n_beacons):
        theta[p_beacon * j: p_beacon * j + 3] = _geometric_guess(ship, delays[:, j], c_eq)

    n_data = n_obs * n_beacons
    if n_data + 2 <= p_dim:
        return {"status": "error", "message": "观测数量不足以估计位置、钟差和水平梯度"}

    g_index = np.array([p_dim - 2, p_dim - 1], dtype=int)
    best_lam = float(_LAMBDA_GRID[len(_LAMBDA_GRID) // 2])
    best_abic = None

    for outer in range(n_outer):
        if progress_callback:
            progress_callback(int(100 * outer / max(n_outer, 1)), f"ABIC outer iteration {outer + 1}/{n_outer}")
        packed = _linearize(tracer, ship, delays, theta, tau, tau2, origin, c2, n_beacons, p_beacon)
        if packed is None:
            return {"status": "error", "message": "声线追踪失败，无法建立 ABIC 法方程"}
        jac, residual, _ = packed
        chosen = None
        for lam in _LAMBDA_GRID:
            delta, phi, abic, ok = _gn_step(jac, residual, theta, g_index, float(lam), n_data, p_dim)
            if not ok:
                continue
            if chosen is None or abic < chosen[0]:
                chosen = (abic, float(lam), delta, phi)
        if chosen is None:
            return {"status": "error", "message": "ABIC 网格上没有可解的梯度先验"}
        best_abic, best_lam, delta, _phi = chosen
        theta = _apply_step(theta, delta, n_beacons, p_beacon)

    final = _evaluate(tracer, ship, delays, theta, tau, tau2, origin, c2, n_beacons, p_beacon)
    if final is None:
        return {"status": "error", "message": "ABIC 解的声线追踪失败"}
    resid_s, _ = final
    rms_s = float(np.sqrt(np.mean(resid_s ** 2)))
    g_e = float(theta[g_index[0]])
    g_n = float(theta[g_index[1]])

    beacons = []
    for j in range(n_beacons):
        base = p_beacon * j
        block = resid_s[j * n_obs:(j + 1) * n_obs]
        beacons.append({
            "id": j,
            "x": round(float(theta[base]), 3),
            "y": round(float(theta[base + 1]), 3),
            "z": round(float(theta[base + 2]), 3),
            "clock_bias_s": float(theta[base + 3]),
            "clock_drift_ss": float(theta[base + 4]),
            "clock_drift2_ss2": float(theta[base + 5]),
            "ntd_k": float(theta[base + 6]),
            "rmse_time_ms": round(float(np.sqrt(np.mean(block ** 2)) * 1000.0), 4),
            "rmse_equiv_m": round(float(np.sqrt(np.mean(block ** 2)) * c_eq), 4),
            "zenith_k_ms_km2": round(float(theta[base + 6]) * 1000.0, 6),
        })

    if progress_callback:
        progress_callback(100, "ABIC fit complete")

    return {
        "status": "success",
        "solver_type": "abic_grad",
        "result_x": beacons[0]["x"],
        "result_y": beacons[0]["y"],
        "result_z": beacons[0]["z"],
        "beacons": beacons,
        "system_clock_bias_ms": round(beacons[0]["clock_bias_s"] * 1000.0, 4),
        "clock_drift_ss": beacons[0]["clock_drift_ss"],
        "clock_drift2_ss2": beacons[0]["clock_drift2_ss2"],
        "zenith_k_ms_km2": beacons[0]["zenith_k_ms_km2"],
        "sv_grad_e": g_e,
        "sv_grad_n": g_n,
        "sv_grad_e_mps_per_km": round(g_e * 1000.0, 6),
        "sv_grad_n_mps_per_km": round(g_n * 1000.0, 6),
        "abic": None if best_abic is None else round(float(best_abic), 4),
        "abic_lambda": best_lam,
        "c_eq_mps": round(c_eq, 3),
        "rmse_time_ms": round(rms_s * 1000.0, 4),
        "rmse_distance_m": round(rms_s * c_eq, 4),
        "residuals_ms": (resid_s * 1000.0).tolist(),
        "n_total": int(n_data),
        "n_clean": int(n_data),
        "outlier_rate": "0.0%",
        "sigma_e_m": None,
        "sigma_n_m": None,
        "sigma_u_m": None,
    }


def _geometric_guess(ship: np.ndarray, delay: np.ndarray, c_eq: float) -> np.ndarray:
    x0 = float(np.median(ship[:, 0]))
    y0 = float(np.median(ship[:, 1]))
    horiz = np.hypot(ship[:, 0] - x0, ship[:, 1] - y0)
    slant = float(np.mean(delay) * c_eq)
    z0 = -float(np.sqrt(max(slant * slant - float(np.median(horiz)) ** 2, 1.0)))
    return np.array([x0, y0, z0], dtype=np.float64)


def _apply_step(theta: np.ndarray, delta: np.ndarray, n_beacons: int, p_beacon: int) -> np.ndarray:
    updated = theta + delta
    for j in range(n_beacons):
        base = p_beacon * j
        step = updated[base:base + 3] - theta[base:base + 3]
        norm = float(np.linalg.norm(step))
        if norm > _MAX_POS_STEP:
            updated[base:base + 3] = theta[base:base + 3] + step * (_MAX_POS_STEP / norm)
        updated[base + 2] = float(np.clip(updated[base + 2], -15000.0, -1.0))
    return updated


def _slant_and_mid(
    ship: np.ndarray, beacon: np.ndarray, origin: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diff = ship - beacon.reshape(1, 3)
    slant = np.sqrt(np.sum(diff * diff, axis=1))
    mid_e = 0.5 * (ship[:, 0] + beacon[0]) - origin[0]
    mid_n = 0.5 * (ship[:, 1] + beacon[1]) - origin[1]
    dh2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
    return slant, mid_e, mid_n, dh2


def _evaluate(
    tracer, ship, delays, theta, tau, tau2, origin, c2, n_beacons, p_beacon,
):
    n_obs = delays.shape[0]
    resid = np.empty(n_obs * n_beacons, dtype=np.float64)
    g_e = float(theta[-2])
    g_n = float(theta[-1])
    for j in range(n_beacons):
        base = p_beacon * j
        beacon = theta[base:base + 3]
        t_ray = np.asarray(tracer.calculate_travel_time_batch(ship, beacon), dtype=np.float64)
        if not np.any(np.isfinite(t_ray)):
            return None
        slant, mid_e, mid_n, dh2 = _slant_and_mid(ship, beacon, origin)
        bad = ~np.isfinite(t_ray) | (t_ray <= 0)
        t_ray = t_ray.copy()
        t_ray[bad] = slant[bad] / np.sqrt(c2)
        model = (
            t_ray
            + theta[base + 3]
            + theta[base + 4] * tau
            + theta[base + 5] * tau2
            + theta[base + 6] * dh2 / 1.0e6
            - (g_e * mid_e + g_n * mid_n) / c2 * slant
        )
        resid[j * n_obs:(j + 1) * n_obs] = delays[:, j] - model
    return resid, None


def _linearize(
    tracer, ship, delays, theta, tau, tau2, origin, c2, n_beacons, p_beacon,
):
    n_obs = delays.shape[0]
    p_dim = p_beacon * n_beacons + 2
    jac = np.zeros((n_obs * n_beacons, p_dim), dtype=np.float64)
    residual = np.empty(n_obs * n_beacons, dtype=np.float64)
    g_e = float(theta[-2])
    g_n = float(theta[-1])
    c2_sqrt = np.sqrt(c2)

    for j in range(n_beacons):
        base = p_beacon * j
        beacon = theta[base:base + 3].copy()
        rows = slice(j * n_obs, (j + 1) * n_obs)
        t_ray = np.asarray(tracer.calculate_travel_time_batch(ship, beacon), dtype=np.float64)
        slant, mid_e, mid_n, dh2 = _slant_and_mid(ship, beacon, origin)
        bad = ~np.isfinite(t_ray) | (t_ray <= 0)
        t_ray = np.array(t_ray, dtype=np.float64, copy=True)
        t_ray[bad] = slant[bad] / c2_sqrt

        k = float(theta[base + 6])
        grad_mid = g_e * mid_e + g_n * mid_n
        model = (
            t_ray
            + theta[base + 3]
            + theta[base + 4] * tau
            + theta[base + 5] * tau2
            + k * dh2 / 1.0e6
            - grad_mid / c2 * slant
        )
        residual[rows] = delays[:, j] - model

        jac[rows, base + 3] = 1.0
        jac[rows, base + 4] = tau
        jac[rows, base + 5] = tau2
        jac[rows, base + 6] = dh2 / 1.0e6
        jac[rows, -2] = -mid_e / c2 * slant
        jac[rows, -1] = -mid_n / c2 * slant

        for axis in range(3):
            bumped = beacon.copy()
            bumped[axis] += _FD_STEP
            t_bump = np.asarray(tracer.calculate_travel_time_batch(ship, bumped), dtype=np.float64)
            slant_b, mid_e_b, mid_n_b, dh2_b = _slant_and_mid(ship, bumped, origin)
            bad_b = ~np.isfinite(t_bump) | (t_bump <= 0)
            t_bump = np.array(t_bump, dtype=np.float64, copy=True)
            t_bump[bad_b] = slant_b[bad_b] / c2_sqrt
            model_b = (
                t_bump
                + theta[base + 3]
                + theta[base + 4] * tau
                + theta[base + 5] * tau2
                + k * dh2_b / 1.0e6
                - (g_e * mid_e_b + g_n * mid_n_b) / c2 * slant_b
            )
            jac[rows, base + axis] = (model - model_b) / _FD_STEP
            # residual = obs - model, Jacobian of model is what GN uses as J in r - J δ.
            # (model - model_b) / step = - ∂model/∂θ if step>0 and model_b = model(θ+step).
            # We want J = ∂model/∂θ so that δ solves JᵀJ δ = Jᵀ (obs-model).
            # ∂model/∂θ ≈ (model_b - model) / step = - (model - model_b) / step.
            # So flip the sign.
        jac[rows, base:base + 3] *= -1.0

    if not np.all(np.isfinite(jac)) or not np.all(np.isfinite(residual)):
        return None
    return jac, residual, None


def _gn_step(
    jac: np.ndarray,
    residual: np.ndarray,
    theta: np.ndarray,
    g_index: np.ndarray,
    lam: float,
    n_data: int,
    p_dim: int,
) -> Tuple[np.ndarray, float, float, bool]:
    m_prior = 2
    nu = n_data + m_prior - p_dim
    if nu <= 0 or lam <= 0:
        return theta * 0.0, np.inf, np.inf, False
    hessian = jac.T @ jac
    weight = lam / (G_REF ** 2)
    for idx in g_index:
        hessian[idx, idx] += weight
    sign, logdet = np.linalg.slogdet(hessian)
    if sign <= 0 or not np.isfinite(logdet):
        return theta * 0.0, np.inf, np.inf, False
    scale = np.maximum(np.diag(hessian), 1e-18)
    solvable = hessian + np.diag(1e-4 * scale)
    rhs = jac.T @ residual
    rhs[g_index] -= weight * theta[g_index]
    try:
        delta = np.linalg.solve(solvable, rhs)
    except np.linalg.LinAlgError:
        return theta * 0.0, np.inf, np.inf, False
    if not np.all(np.isfinite(delta)):
        return theta * 0.0, np.inf, np.inf, False
    g_new = theta[g_index] + delta[g_index]
    pred = residual - jac @ delta
    phi = float(np.dot(pred, pred) + lam * np.sum((g_new / G_REF) ** 2))
    if not np.isfinite(phi) or phi <= 0:
        return theta * 0.0, np.inf, np.inf, False
    abic = nu * np.log(phi) - m_prior * np.log(lam) + logdet
    return delta, phi, float(abic), True
