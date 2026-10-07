"""Gauss-Newton estimation that recomputes ship rays at each accepted step.

Straight-line fallback rows are excluded from the normal matrix and counted.
This is the minimum coordinate-only solver used for numerical checks.
"""
from __future__ import annotations

import numpy as np

from risk_design.forward import TravelTimeModel
from risk_design.parameters import unpack_beacons


def predict_ship(model: TravelTimeModel, x: np.ndarray, ship: np.ndarray, beacon_ids: np.ndarray, a0: float):
    positions = unpack_beacons(x)
    delays = np.empty(len(beacon_ids), dtype=np.float64)
    ok = np.zeros(len(beacon_ids), dtype=bool)
    fallback = 0
    fail = 0
    for beacon in (2, 5, 6):
        mask = beacon_ids == beacon
        if not np.any(mask):
            continue
        result = model.ship(ship[mask], positions[beacon], a0)
        delays[mask] = result["delay_s"]
        ok[mask] = result["snell_ok"]
        fallback += result["n_straight_fallback"]
        fail += result["n_fail"]
    return delays, ok, {"n_straight_fallback": fallback, "n_fail": fail}


def _clock(eta: np.ndarray, beacon_ids: np.ndarray, t_s: np.ndarray, t0_s: float, time_term: str) -> np.ndarray:
    hour = (t_s - t0_s) / 3600.0
    clock = float(eta[1]) + float(eta[2]) * hour
    clock = clock + np.where(beacon_ids == 2, float(eta[3]), 0.0)
    clock = clock + np.where(beacon_ids == 6, float(eta[4]), 0.0)
    if time_term == "clock_quadratic":
        clock = clock + float(eta[5]) * hour ** 2
    return clock


def predict_mean(model: TravelTimeModel, x: np.ndarray, eta: np.ndarray, ship: np.ndarray,
                  beacon_ids: np.ndarray, t_s: np.ndarray, t0_s: float, time_term: str = "none"):
    """Travel time plus effective delay. Beacon 5 carries the delay gauge.

    A sound-speed rate uses the local derivative dT/da at the current offset.
    Reported scores recompute rays in time bins instead of this linearization.
    """
    delays, ok, flags = predict_ship(model, x, ship, beacon_ids, float(eta[0]))
    if time_term == "sound_rate" and abs(float(eta[5])) > 0.0:
        plus, ok_plus, _ = predict_ship(model, x, ship, beacon_ids, float(eta[0]) + 0.02)
        both = ok & ok_plus
        slope = np.zeros(len(beacon_ids))
        slope[both] = (plus[both] - delays[both]) / 0.02
        delays = delays + slope * float(eta[5]) * (t_s - t0_s) / 3600.0
    return delays + _clock(eta, beacon_ids, t_s, t0_s, time_term), ok, flags


def predict_binned(model: TravelTimeModel, x: np.ndarray, eta: np.ndarray, ship: np.ndarray,
                   beacon_ids: np.ndarray, t_s: np.ndarray, t0_s: float, time_term: str = "none",
                   bin_s: float = 300.0):
    """Retrace each 5-minute bin at its own sound-speed offset."""
    hour = (t_s - t0_s) / 3600.0
    rate = float(eta[5]) if time_term == "sound_rate" else 0.0
    delays = np.empty(len(t_s))
    ok = np.zeros(len(t_s), dtype=bool)
    fallback = 0
    bins = np.floor(t_s / bin_s).astype(np.int64)
    for bucket in np.unique(bins):
        mask = bins == bucket
        offset = float(eta[0]) + rate * float(np.mean(hour[mask]))
        part, part_ok, flags = predict_ship(model, x, ship[mask], beacon_ids[mask], offset)
        delays[mask] = part
        ok[mask] = part_ok
        fallback += flags["n_straight_fallback"]
    return delays + _clock(eta, beacon_ids, t_s, t0_s, time_term), ok, {"n_straight_fallback": fallback}


def _jacobian(model: TravelTimeModel, x: np.ndarray, eta: np.ndarray, ship: np.ndarray,
              beacon_ids: np.ndarray, t_s: np.ndarray, t0_s: float, ok: np.ndarray,
              h_pos: float, h_a0: float, time_term: str = "none") -> np.ndarray:
    usable = np.flatnonzero(ok)
    n_eta = 6 if time_term in ("sound_rate", "clock_quadratic") else 5
    jac = np.zeros((usable.size, 9 + n_eta))
    for column in range(9):
        step = np.zeros(9)
        step[column] = h_pos
        plus, ok_p, _ = predict_ship(model, x + step, ship, beacon_ids, float(eta[0]))
        minus, ok_m, _ = predict_ship(model, x - step, ship, beacon_ids, float(eta[0]))
        both = ok_p[usable] & ok_m[usable]
        jac[both, column] = (plus[usable][both] - minus[usable][both]) / (2.0 * h_pos)
    for sign, scale in ((1.0, h_a0), (-1.0, h_a0)):
        trial = eta.copy()
        trial[0] = float(eta[0]) + sign * scale
        pred, ok_a, _ = predict_ship(model, x, ship, beacon_ids, float(trial[0]))
        if sign > 0:
            plus_a, ok_plus = pred, ok_a
        else:
            minus_a, ok_minus = pred, ok_a
    both = ok_plus[usable] & ok_minus[usable]
    jac[both, 9] = (plus_a[usable][both] - minus_a[usable][both]) / (2.0 * h_a0)
    jac[:, 10] = 1.0
    jac[:, 11] = (t_s[usable] - t0_s) / 3600.0
    jac[:, 12] = (beacon_ids[usable] == 2).astype(np.float64)
    jac[:, 13] = (beacon_ids[usable] == 6).astype(np.float64)
    hour = (t_s[usable] - t0_s) / 3600.0
    if time_term == "sound_rate":
        jac[:, 14] = jac[:, 9] * hour
    elif time_term == "clock_quadratic":
        jac[:, 14] = hour ** 2
    return jac


def estimate_delay_model(model: TravelTimeModel, x0: np.ndarray, ship: np.ndarray,
                         beacon_ids: np.ndarray, t_s: np.ndarray, observed_s: np.ndarray,
                         sigma_s: float, max_iter: int = 6, time_term: str = "none",
                         eta0: np.ndarray | None = None, t0_s: float | None = None,
                         prior_center: np.ndarray | None = None) -> dict:
    """Joint coordinates and effective delay. Rays are recomputed every step."""
    from risk_design.parameters import PRIOR_SIGMA, prior_precision

    if time_term not in ("none", "sound_rate", "clock_quadratic"):
        raise ValueError(time_term)
    x = np.asarray(x0, dtype=np.float64).copy()
    eta = np.zeros(6 if time_term != "none" else 5)
    if eta0 is not None:
        eta[: len(eta0)] = np.asarray(eta0, dtype=np.float64)
    t0_s = float(np.median(t_s) if t0_s is None else t0_s)
    weight = 1.0 / sigma_s ** 2
    precision = prior_precision()
    center = np.zeros(len(eta))
    if prior_center is not None:
        center[: len(prior_center)] = np.asarray(prior_center, dtype=np.float64)
    if time_term == "sound_rate":
        precision = np.append(precision, 1.0 / PRIOR_SIGMA["sound_rate_mps_per_hour"] ** 2)
    elif time_term == "clock_quadratic":
        precision = np.append(precision, 1.0 / PRIOR_SIGMA["clock_quadratic_s_per_hour2"] ** 2)
    limits = np.array([100.0] * 9 + [0.5, 0.05, 0.01, 0.05, 0.05])
    if time_term == "sound_rate":
        limits = np.append(limits, 0.5)
    elif time_term == "clock_quadratic":
        limits = np.append(limits, 0.002)
    history = []
    jac = None
    ok = None
    for _ in range(max_iter):
        pred, ok, flags = predict_mean(model, x, eta, ship, beacon_ids, t_s, t0_s, time_term)
        if int(np.sum(ok)) < 20:
            return {"success": False, "x": x, "eta": eta, "t0_s": t0_s, "reason": "too_few_snell_rays",
                    "flags": flags, "history": history, "time_term": time_term}
        usable = np.flatnonzero(ok)
        residual = observed_s[usable] - pred[usable]
        jac = _jacobian(model, x, eta, ship, beacon_ids, t_s, t0_s, ok, 0.1, 0.02, time_term)
        normal = weight * (jac.T @ jac)
        normal[9:, 9:] += np.diag(precision)
        rhs = weight * (jac.T @ residual)
        rhs[9:] -= precision * (eta - center)
        try:
            update = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            return {"success": False, "x": x, "eta": eta, "t0_s": t0_s, "reason": "singular_normal",
                    "flags": flags, "history": history, "jac": jac, "ok": ok, "time_term": time_term}
        # Clip each parameter on its own. One sound-speed step must not shrink the coordinates.
        update = np.clip(update, -limits, limits)
        rms = float(np.sqrt(np.mean(residual ** 2)))
        penalty = float(np.sum(precision * (eta - center) ** 2))
        accepted = np.zeros(9 + len(eta))
        improved = False
        for shrink in (1.0, 0.5, 0.25, 0.1):
            x_try = x + shrink * update[:9]
            eta_try = eta + shrink * update[9:]
            pred_try, ok_try, flags_try = predict_mean(
                model, x_try, eta_try, ship, beacon_ids, t_s, t0_s, time_term,
            )
            if int(np.sum(ok_try)) < 20:
                continue
            both = ok & ok_try
            if int(np.sum(both)) < 20:
                continue
            residual_same = observed_s[both] - pred[both]
            residual_try = observed_s[both] - pred_try[both]
            rms_try = float(np.sqrt(np.mean(residual_try ** 2)))
            penalty_try = float(np.sum(precision * (eta_try - center) ** 2))
            if weight * np.sum(residual_try ** 2) + penalty_try < weight * np.sum(residual_same ** 2) + penalty:
                x, eta, ok, flags = x_try, eta_try, ok_try, flags_try
                accepted = shrink * update
                rms = rms_try
                improved = True
                break
        history.append({
            "rms_s": rms,
            "step_coord_m": float(np.linalg.norm(accepted[:9])),
            "step_a0_mps": float(accepted[9]),
            "n_snell": int(np.sum(ok)),
            "accepted": improved,
        })
        if not improved:
            break
        extra_step = 0.0 if accepted.size < 15 else abs(float(accepted[14]))
        if np.linalg.norm(accepted[:9]) < 0.05 and abs(float(accepted[9])) < 0.01 and extra_step < 1e-3:
            break
    pred, ok, flags = predict_mean(model, x, eta, ship, beacon_ids, t_s, t0_s, time_term)
    jac = None
    if int(np.sum(ok)) >= 20:
        jac = _jacobian(model, x, eta, ship, beacon_ids, t_s, t0_s, ok, 0.1, 0.02, time_term)
    cov = None
    if jac is not None:
        normal = weight * (jac.T @ jac)
        normal[9:, 9:] += np.diag(precision)
        try:
            cov = np.linalg.inv(normal)
        except np.linalg.LinAlgError:
            cov = None
    final_rms = history[-1]["rms_s"] if history else None
    extra_ok = True
    if time_term == "sound_rate":
        extra_ok = abs(float(eta[5])) <= 5.0
    elif time_term == "clock_quadratic":
        extra_ok = abs(float(eta[5])) <= 0.05
    physical = cov is not None and abs(float(eta[0])) <= 10.0 and extra_ok and final_rms is not None and final_rms < 0.05
    return {
        "success": physical,
        "time_term": time_term,
        "x": x,
        "eta": eta,
        "t0_s": t0_s,
        "sigma_s": sigma_s,
        "weight": weight,
        "cov": cov,
        "reason": "within_physical_bounds" if physical else "outside_physical_bounds",
        "flags": flags,
        "history": history,
        "jac": jac,
        "ok": ok,
    }


def estimate_coordinates(model: TravelTimeModel, x0: np.ndarray, ship: np.ndarray,
                         beacon_ids: np.ndarray, observed_s: np.ndarray, a0: float = 0.0,
                         h_m: float = 0.1, max_iter: int = 8) -> dict:
    x = np.asarray(x0, dtype=np.float64).copy()
    history = []
    for _ in range(max_iter):
        pred, ok, flags = predict_ship(model, x, ship, beacon_ids, a0)
        if int(np.sum(ok)) < 9:
            return {"success": False, "x": x, "reason": "too_few_snell_rays", "flags": flags, "history": history}
        residual = observed_s[ok] - pred[ok]
        jac = np.zeros((int(np.sum(ok)), 9))
        usable = np.flatnonzero(ok)
        for column in range(9):
            step = np.zeros(9)
            step[column] = h_m
            plus, ok_p, _ = predict_ship(model, x + step, ship, beacon_ids, a0)
            minus, ok_m, _ = predict_ship(model, x - step, ship, beacon_ids, a0)
            both = ok_p[usable] & ok_m[usable]
            jac[both, column] = (plus[usable][both] - minus[usable][both]) / (2.0 * h_m)
        normal = jac.T @ jac
        try:
            update = np.linalg.solve(normal, jac.T @ residual)
        except np.linalg.LinAlgError:
            return {"success": False, "x": x, "reason": "singular_normal", "flags": flags, "history": history}
        x = x + update
        rms = float(np.sqrt(np.mean(residual ** 2)))
        history.append({"rms_s": rms, "step_m": float(np.linalg.norm(update)), "n_snell": int(np.sum(ok))})
        if np.linalg.norm(update) < 0.01 and rms < 1.0e-6:
            break
    return {"success": True, "x": x, "reason": "converged_or_iteration_cap", "flags": flags, "history": history}
