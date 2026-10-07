"""Fallback 3-parameter LS when the TOA solver is unavailable or SSP is missing."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


class RealDataSolver:
    def __init__(self, ray_tracer=None, c_ref: float | None = None):
        self.ray_tracer = ray_tracer
        if c_ref is not None:
            self.c_ref = float(c_ref)
        elif ray_tracer is not None and getattr(ray_tracer, "c_eq_measured", None):
            self.c_ref = float(ray_tracer.c_eq_measured)
        else:
            self.c_ref = 1496.32

    def solve(
        self,
        obs_ship_pos: np.ndarray,
        obs_delays: np.ndarray,
        depth_prior: float = None,
        depth_weight: float = 1e3,
        sigma_threshold: float = 3.0,
        max_iterations: int = 3,
        progress_callback=None,
    ):
        n = len(obs_delays)
        if depth_prior is None:
            depth_prior = abs(float(np.mean(obs_delays) * self.c_ref))
        self._depth_prior = depth_prior
        self._depth_weight = depth_weight

        best_B = None
        best_residuals_ms = None
        best_rmse_ms = np.inf
        best_mask = np.ones(n, dtype=bool)
        active_mask = np.ones(n, dtype=bool)

        for iteration in range(max_iterations):
            n_active = int(np.sum(active_mask))
            init = np.array([
                float(np.median(obs_ship_pos[active_mask, 0])),
                float(np.median(obs_ship_pos[active_mask, 1])),
                -abs(depth_prior),
            ])
            result = least_squares(
                lambda B: self._objective(B, obs_ship_pos[active_mask], obs_delays[active_mask]),
                init, method="trf", loss="soft_l1", f_scale=5.0, max_nfev=200,
            )
            B_est = result.x
            residuals_full = self._objective(B_est, obs_ship_pos, obs_delays)
            delay_residuals = residuals_full[:-1] * 1000.0
            rmse_ms = float(np.sqrt(np.mean(delay_residuals[active_mask] ** 2)))
            if rmse_ms < best_rmse_ms:
                best_B = B_est.copy()
                best_residuals_ms = delay_residuals.copy()
                best_rmse_ms = rmse_ms
                best_mask = active_mask.copy()
            sigma = float(np.std(delay_residuals[active_mask]))
            if progress_callback:
                progress_callback(min((iteration + 1) / max_iterations * 100.0, 100.0))
            if sigma < 1e-9:
                break
            new_mask = active_mask & (np.abs(delay_residuals) < sigma_threshold * sigma)
            if int(np.sum(new_mask)) == n_active:
                break
            active_mask = new_mask

        xy_m, z_m = self._estimate_precision(best_B, obs_ship_pos, obs_delays, best_mask)
        return best_B, best_rmse_ms, xy_m, z_m, best_residuals_ms, best_mask

    def _model_delay(self, B: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if self.ray_tracer is not None:
            batch = getattr(self.ray_tracer, "calculate_travel_time_batch", None)
            if callable(batch):
                return np.asarray(batch(positions, B), dtype=np.float64)
            t_model = np.empty(len(positions))
            for i in range(len(positions)):
                t_model[i] = self.ray_tracer.calculate_travel_time(positions[i], B)
            return t_model
        d = positions - B[np.newaxis, :]
        return np.sqrt(np.sum(d ** 2, axis=1)) / self.c_ref

    def _objective(self, B: np.ndarray, positions: np.ndarray, obs_delays: np.ndarray) -> np.ndarray:
        raw = obs_delays - self._model_delay(B, positions)
        z_residual = (B[2] - (-abs(self._depth_prior))) * self._depth_weight
        return np.append(raw, z_residual)

    def _estimate_precision(self, B_est, obs_ship, obs_delays, mask):
        if B_est is None:
            return np.nan, np.nan
        n = len(obs_delays)
        J = np.zeros((n + 1, 3))
        for i in range(n):
            dx = B_est[0] - obs_ship[i, 0]
            dy = B_est[1] - obs_ship[i, 1]
            dz = B_est[2] - obs_ship[i, 2]
            r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + 1e-9
            if self.ray_tracer:
                tau = float(self.ray_tracer.calculate_travel_time(obs_ship[i], B_est))
                c_eff = r / max(tau or 1e-9, 1e-9)
            else:
                c_eff = self.c_ref
            J[i, 0] = -dx / (r * max(c_eff, 1e-9))
            J[i, 1] = -dy / (r * max(c_eff, 1e-9))
            J[i, 2] = -dz / (r * max(c_eff, 1e-9))
        J[-1, 2] = self._depth_weight
        nmat = J.T @ J
        try:
            q = np.linalg.pinv(nmat, rcond=1e-12)
            xy = float(np.sqrt(max(q[0, 0] + q[1, 1], 0.0)))
            z = float(np.sqrt(max(q[2, 2], 0.0)))
            return xy, z
        except np.linalg.LinAlgError:
            return np.nan, np.nan
