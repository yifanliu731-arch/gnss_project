'Web acoustic tracer using layered Snell shooting and Brent root finding. A straight-line / equivalent-speed fallback is exposed explicitly when shooting fails. Public methods return travel time in seconds, acoustic arc length in metres, and the method label. No external ray executable or static lookup database is required.'

import numpy as np
from scipy.optimize import root_scalar
import pandas as pd
from typing import Optional, Tuple


def normalize_ssp_columns(df: pd.DataFrame) -> pd.DataFrame:
    'Normalize common depth and sound-speed column aliases, including supported Chinese headers, to Depth (m) and SV (m/s).'
    df = df.copy()
    depth_col = None
    sv_col = None
    for col in df.columns:
        low = str(col).strip().lower()
        if depth_col is None and ('depth' in low or '深度' in low or low in ('z', 'dep')):
            depth_col = col
        elif sv_col is None and any(k in low for k in ('sv', 'vel', 'speed', 'sound', '声速')):
            sv_col = col
    if depth_col is None or sv_col is None:
        raise KeyError(
            f"无法识别 SSP 列名 (需要深度列与声速列): {list(df.columns)}"
        )
    return df.rename(columns={depth_col: 'Depth (m)', sv_col: 'SV (m/s)'})


def resample_profile_to_z(z_src: np.ndarray, values: np.ndarray, z_dst: np.ndarray) -> np.ndarray:
    z_src = np.asarray(z_src, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    z_dst = np.asarray(z_dst, dtype=np.float64).reshape(-1)
    order = np.argsort(z_src)
    z_src, values = z_src[order], values[order]
    uniq, idx = np.unique(z_src, return_index=True)
    return np.interp(z_dst, uniq, values[idx])


def resample_eof_modes(eof_modes: np.ndarray, z_src: np.ndarray, z_dst: np.ndarray) -> np.ndarray:
    raw = np.atleast_2d(np.asarray(eof_modes, dtype=np.float64))
    return np.vstack([resample_profile_to_z(z_src, raw[i], z_dst) for i in range(raw.shape[0])])


def equivalent_sound_speed(delay_s: np.ndarray, arc_len_m: np.ndarray, fallback: float = 1496.32) -> np.ndarray:
    delay_s = np.asarray(delay_s, dtype=np.float64)
    arc_len_m = np.asarray(arc_len_m, dtype=np.float64)
    c_eq = np.full(delay_s.shape, float(fallback), dtype=np.float64)
    ok = np.isfinite(delay_s) & np.isfinite(arc_len_m) & (delay_s > 1e-12) & (arc_len_m > 0.0)
    c_eq[ok] = arc_len_m[ok] / delay_s[ok]
    return c_eq


def ssp_equivalent_speed(df: pd.DataFrame) -> float:
    """Harmonic-mean sound speed of the measured SSP (no deep-water extension)."""
    norm = normalize_ssp_columns(df).sort_values(by="Depth (m)")
    z = norm["Depth (m)"].values.astype(np.float64)
    v = np.clip(norm["SV (m/s)"].values.astype(np.float64), 1e-6, None)
    if len(z) < 2:
        return float(np.mean(v)) if len(v) else 1496.32
    dz = np.diff(z)
    vmid = 0.5 * (v[:-1] + v[1:])
    return float(np.sum(dz) / np.sum(dz / np.clip(vmid, 1e-6, None)))


class RayTracer:
    'Layered Snell shooting with an explicitly labelled straight-line fallback.'

    def __init__(self, df_ssp_simplified: pd.DataFrame, use_cache: bool = True):
        # --- SSP Setup ---
        df = normalize_ssp_columns(df_ssp_simplified).sort_values(by="Depth (m)")
        self.Z = df["Depth (m)"].values.astype(np.float64)
        self.V = df["SV (m/s)"].values.astype(np.float64)

        # Keep the original SSP depth as the measured support boundary, not the seabed.
        self.Z_orig = self.Z.copy()
        self.V_orig = self.V.copy()

        # The web tracer extends the nominal SSP to 6000 m to support deep beacons.
        self._extend_ssp_to_deep()

        self.c_eq_measured = self.equivalent_speed_global(
            z_top=float(self.Z_orig[0]), z_bot=float(self.Z_orig[-1])
        )
        self.delta_c = 0.0

        # Cache (source depth, receiver depth, horizontal range) to (time, arc length, method).
        # Disable rounded caching during least squares: rounding to 0.1 m can hide
        # finite-difference position changes and make horizontal derivatives vanish.
        self.use_cache = use_cache
        self._cache: dict = {}

    # ==================================================================
    # Deep nominal SSP extension.
    # ==================================================================
    def _extend_ssp_to_deep(self):
        z_max = self.Z[-1]
        if z_max >= 5500:
            return
        z_ext = np.linspace(z_max + 100, max(z_max * 1.5, 6000), 10)
        v_deep = self.V[-1] + 0.016 * (z_ext - z_max)
        self.Z = np.concatenate([self.Z, z_ext])
        self.V = np.concatenate([self.V, v_deep])
        print(f"[RayTracer] SSP extended: {len(self.Z_orig)} -> {len(self.Z)} pts, "
              f"max depth {self.Z[-1]:.0f} m")

    # ==================================================================
    #  Public API
    # ==================================================================
    def calculate_travel_time(self, trans_pos: np.ndarray, beacon_pos: np.ndarray) -> float:
        delay, _, _ = self._resolve(trans_pos, beacon_pos)
        return delay

    def calculate_acoustic_range(self, trans_pos: np.ndarray, beacon_pos: np.ndarray) -> float:
        _, arc_len, _ = self._resolve(trans_pos, beacon_pos)
        return arc_len

    def calculate_acoustic_range_with_method(
        self, trans_pos: np.ndarray, beacon_pos: np.ndarray
    ) -> Tuple[float, str]:
        _, arc_len, method = self._resolve(trans_pos, beacon_pos)
        if np.isnan(arc_len):
            return np.nan, "FAIL"
        return arc_len, method

    def calculate_travel_time_with_method(
        self, trans_pos: np.ndarray, beacon_pos: np.ndarray
    ) -> Tuple[float, str]:
        delay, _, method = self._resolve(trans_pos, beacon_pos)
        if np.isnan(delay):
            return np.nan, "FAIL"
        return delay, method

    # ==================================================================
    # Vectorized API for many pings to one beacon in the least-squares loop.
    # ==================================================================
    def calculate_travel_time_batch(
        self, trans_pos: np.ndarray, beacon_pos: np.ndarray
    ) -> np.ndarray:
        delay, _, _ = self._resolve_batch(trans_pos, beacon_pos)
        return delay

    def calculate_acoustic_range_batch(
        self, trans_pos: np.ndarray, beacon_pos: np.ndarray
    ) -> np.ndarray:
        _, arc_len, _ = self._resolve_batch(trans_pos, beacon_pos)
        return arc_len

    def calculate_acoustic_range_with_method_batch(
        self, trans_pos: np.ndarray, beacon_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, arc_len, snell_ok = self._resolve_batch(trans_pos, beacon_pos)
        trans = np.asarray(trans_pos, dtype=np.float64)
        b = np.asarray(beacon_pos, dtype=np.float64).reshape(-1)
        z_src = np.abs(trans[:, 2])
        z_rcv = abs(float(b[2]))
        method = np.empty(arc_len.shape, dtype=object)
        method[:] = "StraightLine"
        method[snell_ok] = "Snell"
        method[z_rcv < z_src] = "FAIL"
        return arc_len, method

    def calculate_travel_time_with_method_batch(
        self, trans_pos: np.ndarray, beacon_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        delay, _, snell_ok = self._resolve_batch(trans_pos, beacon_pos)
        trans = np.asarray(trans_pos, dtype=np.float64)
        b = np.asarray(beacon_pos, dtype=np.float64).reshape(-1)
        z_src = np.abs(trans[:, 2])
        z_rcv = abs(float(b[2]))
        method = np.empty(delay.shape, dtype=object)
        method[:] = "StraightLine"
        method[snell_ok] = "Snell"
        method[z_rcv < z_src] = "FAIL"
        return delay, method

    def calculate_travel_time_and_range_batch(
        self, trans_pos: np.ndarray, beacon_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        delay, arc_len, _ = self._resolve_batch(trans_pos, beacon_pos)
        return delay, arc_len

    def calculate_travel_time_and_range_with_method_batch(
        self, trans_pos: np.ndarray, beacon_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        delay, arc_len, snell_ok = self._resolve_batch(trans_pos, beacon_pos)
        trans = np.asarray(trans_pos, dtype=np.float64)
        b = np.asarray(beacon_pos, dtype=np.float64).reshape(-1)
        z_src = np.abs(trans[:, 2])
        z_rcv = abs(float(b[2]))
        method = np.empty(delay.shape, dtype=object)
        method[:] = "StraightLine"
        method[snell_ok] = "Snell"
        method[z_rcv < z_src] = "FAIL"
        return delay, arc_len, method

    def equivalent_speed_global(self, z_top: Optional[float] = None, z_bot: Optional[float] = None) -> float:
        'Harmonic-mean SSP speed; by default integrate measured support only, excluding deep nominal extrapolation.'
        z = np.asarray(self.Z, dtype=np.float64)
        v = np.clip(np.asarray(self.V, dtype=np.float64), 1e-6, None)
        z_orig = getattr(self, "Z_orig", None)
        if z_top is None:
            z_top = float(z_orig[0]) if z_orig is not None else float(z[0])
        if z_bot is None:
            z_bot = float(z_orig[-1]) if z_orig is not None else float(z[-1])
        z_lo, z_hi = min(z_top, z_bot), max(z_top, z_bot)
        sel = (z >= z_lo) & (z <= z_hi)
        if np.count_nonzero(sel) < 2:
            return float(np.mean(v))
        zz, vv = z[sel], v[sel]
        dz = np.diff(zz)
        vmid = 0.5 * (vv[:-1] + vv[1:])
        return float(np.sum(dz) / np.sum(dz / np.clip(vmid, 1e-6, None)))

    def _resolve_batch(
        self, trans_pos: np.ndarray, beacon_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        'Return (delay_s, arc_len_m, snell_ok) for pings to one beacon. snell_ok marks converged Snell rays; other supported rays use the straight-line fallback.'
        trans = np.asarray(trans_pos, dtype=np.float64)
        b = np.asarray(beacon_pos, dtype=np.float64).reshape(-1)
        N = trans.shape[0]
        z_src = np.abs(trans[:, 2])
        z_rcv = abs(float(b[2]))
        dx = trans[:, 0] - b[0]
        dy = trans[:, 1] - b[1]
        range_m = np.sqrt(dx * dx + dy * dy)

        delay = np.full(N, np.nan, dtype=np.float64)
        arc_len = np.full(N, np.nan, dtype=np.float64)
        snell_ok = np.zeros(N, dtype=bool)

        geom_ok = z_rcv >= z_src

        # Group transducer depths at 1 cm resolution and use each group median.
        # For a 3000 m ray the resulting arc-length approximation is millimetric,
        # below the stated acoustic noise; coarse 0.5 m depth bins were unsuitable.
        q = np.round(z_src, 2)
        for qv in np.unique(q):
            sel = (q == qv) & geom_ok
            if not np.any(sel):
                continue
            zs = float(np.median(z_src[sel]))
            d, a = self._solve_snell_eigenray_batch(range_m[sel], zs, z_rcv)
            ok = np.isfinite(d)
            delay[sel] = d
            arc_len[sel] = a
            snell_ok[sel] = ok

        # Straight-line / equivalent-speed fallback for geometrically valid failed rays.
        fallback = geom_ok & ~snell_ok
        if np.any(fallback):
            v_surface = float(self._velocity(0.0))
            bottom_depth = max(self.Z[-1], float(z_src.max()), z_rcv)
            v_bottom = float(self._velocity(bottom_depth))
            c_avg = 0.5 * (v_surface + v_bottom)
            dz = np.abs(z_rcv - z_src[fallback])
            slant = np.sqrt(range_m[fallback] ** 2 + dz ** 2)
            delay[fallback] = slant / c_avg
            arc_len[fallback] = slant

        return delay, arc_len, snell_ok

    # ==================================================================
    #  Core Resolution Engine
    # ==================================================================
    def _resolve(
        self, trans_pos: np.ndarray, beacon_pos: np.ndarray
    ) -> Tuple[float, float, str]:
        'Return (delay_s, arc_len_m, method).'
        # Normalize depths positive downward; reject invalid source/receiver geometry.
        z_src = abs(float(trans_pos[2]))
        z_rcv = abs(float(beacon_pos[2]))
        if z_rcv < z_src:
            return np.nan, np.nan, "FAIL"

        dx = float(trans_pos[0]) - float(beacon_pos[0])
        dy = float(trans_pos[1]) - float(beacon_pos[1])
        range_m = np.sqrt(dx * dx + dy * dy)

        cache_key = (round(z_src, 2), round(z_rcv, 2), round(range_m, 1))
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        delay = np.nan
        arc_len = np.nan
        method = "FAIL"

        # First attempt Snell shooting.
        delay, arc_len = self._solve_snell_eigenray(range_m, z_src, z_rcv)
        if not np.isnan(delay):
            method = "Snell"

        # Then use the straight-line / equivalent-speed fallback.
        if np.isnan(delay):
            method = "StraightLine"
            v_surface = float(self._velocity(0.0))
            bottom_depth = max(self.Z[-1], z_src, z_rcv)
            v_bottom = float(self._velocity(bottom_depth))
            c_avg = 0.5 * (v_surface + v_bottom)
            dz = abs(z_rcv - z_src)
            slant_m = np.sqrt(range_m * range_m + dz * dz)
            delay = slant_m / c_avg
            arc_len = slant_m

        result = (delay, arc_len, method)
        if self.use_cache:
            self._cache[cache_key] = result
        return result

    # ==================================================================
    # Evaluate the SSP.
    # ==================================================================
    def _velocity(self, z: float) -> float:
        return float(np.interp(z, self.Z, self.V)) + self.delta_c

    # ==================================================================
    # Snell shooting with Brent root finding.
    # ==================================================================
    def _get_effective_ssp(self, z_start: float, z_end: float) -> Tuple[np.ndarray, np.ndarray]:
        if z_start > z_end:
            z_start, z_end = z_end, z_start

        z_eff, v_eff = [z_start], [self._velocity(z_start)]
        for z, v in zip(self.Z, self.V):
            if z_start < z < z_end:
                z_eff.append(z)
                v_eff.append(v + self.delta_c)
        z_eff.append(z_end)
        v_eff.append(self._velocity(z_end))
        return np.array(z_eff), np.array(v_eff)

    def _solve_snell_eigenray(self, range_m: float, z_src: float, z_rcv: float) -> Tuple[float, float]:
        if range_m < 1e-3:
            _, t, s = self._shoot_ray_snell(0.0, *self._get_effective_ssp(z_src, z_rcv))
            return t, s

        z_eff, v_eff = self._get_effective_ssp(z_src, z_rcv)
        p_max = 1.0 / np.max(v_eff) - 1e-10

        def range_error(p):
            x, _, _ = self._shoot_ray_snell(p, z_eff, v_eff)
            return x - range_m

        try:
            if range_error(p_max) < 0:
                return np.nan, np.nan
        except Exception:
            pass

        try:
            res = root_scalar(range_error, bracket=[0.0, p_max], method='brentq', xtol=1e-12)
            if res.converged:
                _, t_opt, s_opt = self._shoot_ray_snell(res.root, z_eff, v_eff)
                return t_opt, s_opt
        except Exception:
            pass

        return np.nan, np.nan

    def _solve_snell_eigenray_batch(
        self, range_m: np.ndarray, z_src: float, z_rcv: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        'Batched Snell shooting by bisection on a common depth grid. Return (delay_s, arc_len_m), with NaN for shadow-zone or unconverged rays.'
        range_m = np.atleast_1d(np.asarray(range_m, dtype=np.float64))
        M = range_m.shape[0]
        delay = np.full(M, np.nan, dtype=np.float64)
        arc_len = np.full(M, np.nan, dtype=np.float64)

        if z_rcv < z_src:
            return delay, arc_len

        Z_eff, V_eff = self._get_effective_ssp(z_src, z_rcv)

        # Near-vertical ray (horizontal range approximately zero).
        vert = range_m < 1e-3
        if np.any(vert):
            _, t0, s0 = self._shoot_ray_snell(0.0, Z_eff, V_eff)
            delay[vert] = t0
            arc_len[vert] = s0

        sel = ~vert
        if not np.any(sel):
            return delay, arc_len

        p_max = 1.0 / np.max(V_eff) - 1e-10
        r_sel = range_m[sel]
        M2 = r_sel.shape[0]

        # Reject shadow-zone ranges beyond the limiting monotone ray.
        hi = np.full(M2, p_max)
        Xhi = self._shoot_ray_snell_batch(hi, Z_eff, V_eff)[0]
        active = Xhi >= r_sel

        lo = np.zeros(M2)
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            Xmid = self._shoot_ray_snell_batch(mid, Z_eff, V_eff)[0]
            go_up = (Xmid < r_sel) & active
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)

        p_opt = 0.5 * (lo + hi)
        if np.any(active):
            _, tt, ss = self._shoot_ray_snell_batch(p_opt, Z_eff, V_eff)
            idx = np.flatnonzero(sel)[active]
            delay[idx] = tt[active]
            arc_len[idx] = ss[active]
        return delay, arc_len

    @staticmethod
    def _shoot_ray_snell_batch(
        p: np.ndarray, Z_eff: np.ndarray, V_eff: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        'Trace constant-gradient layers for an array of ray parameters. Return horizontal distance X, time T, and arc length S, using the scalar Snell mapping.'
        p = np.atleast_1d(np.asarray(p, dtype=np.float64))
        Z_eff = np.asarray(Z_eff, dtype=np.float64)
        V_eff = np.asarray(V_eff, dtype=np.float64)

        G = np.diff(V_eff) / np.diff(Z_eff)          # (K,)
        v1 = V_eff[:-1]
        v2 = V_eff[1:]
        dz = np.abs(np.diff(Z_eff))

        pp = p[:, None]                              # (M,1)
        v1b = v1[None, :]                            # (1,K)
        v2b = v2[None, :]
        dzb = dz[None, :]
        Gb = G[None, :]

        near_vert = np.abs(pp) < 1e-12
        refl = (pp * v1b >= 1.0) | (pp * v2b >= 1.0)

        with np.errstate(all="ignore"):
            cos1 = np.sqrt(1.0 - (pp * v1b) ** 2)
            cos2 = np.sqrt(1.0 - (pp * v2b) ** 2)

            g_zero = np.abs(Gb) < 1e-6
            denom = pp * Gb
            small_denom = np.abs(denom) < 1e-12

            # Constant-speed layer.
            dx_eq = (pp * v1b / cos1) * dzb
            dt_eq = dzb / (v1b * cos1)

            # Constant-gradient layer.
            dx_gr = (cos1 - cos2) / np.where(small_denom, 1.0, denom)
            dt_gr = (1.0 / np.where(g_zero, 1.0, Gb)) * np.log(
                (v2b * (1.0 + cos1)) / (v1b * (1.0 + cos2))
            )

            # Near-vertical gradient layer: linearize with the mean layer speed.
            v_mid = 0.5 * (v1b + v2b)
            dx_small = pp * v_mid * dzb
            dt_small = dzb / v_mid

            dx = np.where(g_zero, dx_eq, np.where(small_denom, dx_small, dx_gr))
            dt = np.where(g_zero, dt_eq, np.where(small_denom, dt_small, dt_gr))

            # Vertical ray.
            dt_vert_eq = dzb / v1b
            dt_vert_gr = np.abs(
                (1.0 / np.where(g_zero, 1.0, Gb)) * np.log(v2b / v1b)
            )
            dt_vert = np.where(g_zero, dt_vert_eq, dt_vert_gr)
            dx = np.where(near_vert, 0.0, dx)
            dt = np.where(near_vert, dt_vert, dt)

            # Total reflection: return infinity.
            dx = np.where(refl, np.inf, dx)
            dt = np.where(refl, np.inf, dt)

        X = dx.sum(axis=1)
        T = dt.sum(axis=1)
        S = np.sqrt(dx * dx + dzb * dzb).sum(axis=1)
        return X, np.abs(T), S

    @staticmethod
    def _shoot_ray_snell(p: float, Z_eff: np.ndarray, V_eff: np.ndarray) -> Tuple[float, float, float]:
        'Trace constant-gradient Snell layers; return horizontal distance, travel time, and arc length.'
        G = np.diff(V_eff) / np.diff(Z_eff)
        X_total, T_total, S_total = 0.0, 0.0, 0.0

        if np.abs(p) < 1e-12:
            for i in range(len(G)):
                v1, v2 = V_eff[i], V_eff[i + 1]
                dz = np.abs(Z_eff[i + 1] - Z_eff[i])
                g = G[i]
                if np.abs(g) < 1e-6:
                    T_total += dz / v1
                else:
                    T_total += np.abs((1.0 / g) * np.log(v2 / v1))
                S_total += dz
            return 0.0, T_total, S_total

        for i in range(len(G)):
            v1, v2 = V_eff[i], V_eff[i + 1]
            dz = Z_eff[i + 1] - Z_eff[i]
            g = G[i]

            if p * v1 >= 1.0 or p * v2 >= 1.0:
                return np.inf, np.inf, np.inf

            cos1 = np.sqrt(1.0 - (p * v1) ** 2)
            cos2 = np.sqrt(1.0 - (p * v2) ** 2)

            if np.abs(g) < 1e-6:
                dx_i = (p * v1 / cos1) * dz
                dt_i = dz / (v1 * cos1)
            else:
                denom = p * g
                if abs(denom) < 1e-12:
                    v_mid = 0.5 * (v1 + v2)
                    dx_i = p * v_mid * dz
                    dt_i = dz / v_mid
                else:
                    dx_i = (cos1 - cos2) / denom
                    dt_i = (1.0 / g) * np.log((v2 * (1.0 + cos1)) / (v1 * (1.0 + cos2)))

            X_total += dx_i
            T_total += dt_i
            S_total += np.sqrt(dx_i * dx_i + dz * dz)

        return X_total, np.abs(T_total), S_total
