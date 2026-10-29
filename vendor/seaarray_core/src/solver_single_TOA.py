'One-way kinematic TOA solver with weighted least squares, session clocks, and MAD filtering. Form observations in travel time, adjust with measured-support C_EQ, and report residuals using local equivalent speed. lock_k fixes the zenith term to zero; otherwise fit k*d_h**2. Transducer delay is zero. lock_eof fixes c1=c2=0: EOF modes enter observation sigma only. States are [E,N,U,k] plus [B,D1,D2,c1,c2] for each session.'
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ray_tracing import resample_eof_modes, equivalent_sound_speed
from experiment_config import to_unix_seconds, SIGMA_AC

C_REF = 1496.32
SESSION_GAP_S = 1800.0
SESSION_LOCK_D2_S = 45.0 * 60.0
SESSION_LOCK_D1_S = 15.0 * 60.0
SNELL_FAIL_WEIGHT = 0.1
N_GEOM = 4
N_PER_SESS = 5
IDX_K = 3

SIGMA_EOF = 2.0
SIGMA_BIAS = 0.05
SIGMA_D1 = 1e-3
SIGMA_D2 = 1e-6
SIGMA_ZENITH = 0.05


def detect_sessions(t_seconds, gap_s=SESSION_GAP_S):
    'Split sessions by time gaps; return session indices matching t_seconds and the session count.'
    t = np.asarray(t_seconds, dtype=np.float64).reshape(-1)
    n = len(t)
    if n == 0:
        return np.zeros(0, dtype=int), 0
    order = np.argsort(t, kind="mergesort")
    sess_ord = np.zeros(n, dtype=int)
    n_sess = 1
    for i in range(1, n):
        if t[order[i]] - t[order[i - 1]] > gap_s:
            n_sess += 1
        sess_ord[i] = n_sess - 1
    session = np.empty(n, dtype=int)
    session[order] = sess_ord
    return session, n_sess


def session_breakpoints(t_seconds, gap_s=SESSION_GAP_S):
    'Derive shared session boundaries from timestamps.'
    t = np.sort(np.asarray(t_seconds, dtype=np.float64).reshape(-1))
    if len(t) == 0:
        return np.array([-np.inf, np.inf])
    breaks = [t[0] - 1.0]
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > gap_s:
            breaks.append(0.5 * (t[i - 1] + t[i]))
    breaks.append(t[-1] + 1.0)
    return np.asarray(breaks, dtype=np.float64)


def assign_sessions(t_seconds, breaks):
    t = np.asarray(t_seconds, dtype=np.float64).reshape(-1)
    if len(t) == 0:
        return np.zeros(0, dtype=int)
    sid = np.searchsorted(breaks, t, side="right") - 1
    return np.clip(sid, 0, len(breaks) - 2).astype(int)


def session_tau(t_seconds, session):
    'Center session time in hours and orthogonalize the quadratic drift term.'
    t = np.asarray(t_seconds, dtype=np.float64).reshape(-1)
    session = np.asarray(session, dtype=int).reshape(-1)
    tau = np.zeros(len(t), dtype=np.float64)
    tm2 = np.zeros(len(t), dtype=np.float64)
    for k in np.unique(session):
        m = session == k
        tc = np.mean(t[m])
        tau[m] = (t[m] - tc) / 3600.0
        tm2[m] = np.mean(tau[m] ** 2)
    return tau, tm2


def session_spans_s(t_seconds, session, n_sess):
    t = np.asarray(t_seconds, dtype=np.float64).reshape(-1)
    session = np.asarray(session, dtype=int).reshape(-1)
    spans = np.zeros(max(n_sess, 1), dtype=np.float64)
    for k in range(n_sess):
        m = session == k
        if np.any(m):
            spans[k] = float(t[m].max() - t[m].min())
    return spans


def sess_offset(k):
    return N_GEOM + N_PER_SESS * int(k)


def dominant_session(session, n_sess, t_seconds=None):
    'Choose the session with most pings, breaking ties by longest time span.'
    if n_sess <= 0:
        return 0
    counts = np.array([int(np.sum(session == k)) for k in range(n_sess)])
    if t_seconds is not None:
        spans = session_spans_s(t_seconds, session, n_sess)
        order = np.lexsort((-spans, -counts))
        return int(order[0])
    return int(np.argmax(counts)) if counts.max() > 0 else 0


class SingleWaySolver:
    def __init__(self, ray_tracer, eof_modes=None):
        self.tracer = ray_tracer
        self.Z_bg = ray_tracer.Z.copy()
        self.V_bg = ray_tracer.V.copy()

        if hasattr(ray_tracer, "Z_orig"):
            self.ssp_measured_max_depth = float(ray_tracer.Z_orig.max())
        else:
            self.ssp_measured_max_depth = 3000.0

        target_len = len(self.Z_bg)

        if eof_modes is not None:
            raw_eof = np.asarray(eof_modes, dtype=np.float64)
            eof_len = raw_eof.shape[1]
            if hasattr(ray_tracer, "Z_orig") and len(ray_tracer.Z_orig) == eof_len:
                z_src = ray_tracer.Z_orig
            elif eof_len == target_len:
                z_src = self.Z_bg
            else:
                z_src = np.arange(eof_len, dtype=np.float64)
            self.eof = resample_eof_modes(raw_eof, z_src, self.Z_bg)
        else:
            self.eof = self._generate_analytic_eofs(self.Z_bg)

    def _generate_analytic_eofs(self, Z: np.ndarray) -> np.ndarray:
        Z_clip = np.clip(Z, 1.0, None)
        eof1 = (Z_clip / 150.0) * np.exp(-(Z_clip - 150.0) ** 2 / (2 * 100 ** 2))
        eof1 = eof1 / np.max(np.abs(eof1))
        eof2 = np.sin(np.pi * Z_clip / max(2000.0, np.max(Z_clip))) * np.exp(-Z_clip / 1500.0)
        eof2 = eof2 / (np.max(np.abs(eof2)) + 1e-6)
        return np.vstack([eof1, eof2])

    def _c_global(self) -> float:
        v_save = self.tracer.V
        self.tracer.V = self.V_bg
        try:
            stored = getattr(self.tracer, "c_eq_measured", None)
            if stored is not None:
                return float(stored)
            fn = getattr(self.tracer, "equivalent_speed_global", None)
            if callable(fn):
                z_top = float(self.tracer.Z_orig[0]) if hasattr(self.tracer, "Z_orig") else None
                z_bot = float(self.tracer.Z_orig[-1]) if hasattr(self.tracer, "Z_orig") else None
                return float(fn(z_top=z_top, z_bot=z_bot))
        except Exception:
            pass
        finally:
            self.tracer.V = v_save
        return C_REF

    def _estimate_initial_position(self, df_beacon: pd.DataFrame, depth_constraint=None) -> np.ndarray:
        step = max(1, len(df_beacon) // 50)
        df_sub = df_beacon.iloc[::step]
        RxE, RxN, RxU = df_sub["Rx_E"].values, df_sub["Rx_N"].values, df_sub["Rx_U"].values
        r_obs = df_sub["delay"].values * C_REF

        mean_E, mean_N = np.mean(RxE), np.mean(RxN)
        max_r = np.max(r_obs)

        z_nom = float(depth_constraint[0]) if depth_constraint else self.ssp_measured_max_depth
        z_tol = float(depth_constraint[1]) if depth_constraint else 150.0

        z_min, z_max = -max(z_nom + z_tol, 1.0), -max(z_nom - z_tol, 1.0)
        U_grid = np.linspace(z_min, z_max, 5)
        z_bounds = (z_min, z_max)

        E_grid = np.linspace(mean_E - max_r, mean_E + max_r, 11)
        N_grid = np.linspace(mean_N - max_r, mean_N + max_r, 11)
        EE, NN, UU = np.meshgrid(E_grid, N_grid, U_grid, indexing="ij")
        pts = np.column_stack([EE.ravel(), NN.ravel(), UU.ravel()])

        best_var = np.inf
        best_p = pts[0]
        for p in pts:
            rng = np.sqrt((RxE - p[0]) ** 2 + (RxN - p[1]) ** 2 + (RxU - p[2]) ** 2)
            var = np.mean(np.abs((r_obs - rng) - np.mean(r_obs - rng)))
            if var < best_var:
                best_var = var
                best_p = p

        def resid(p):
            rng = np.sqrt((RxE - p[0]) ** 2 + (RxN - p[1]) ** 2 + (RxU - p[2]) ** 2)
            return (r_obs - rng) - np.mean(r_obs - rng)

        res = least_squares(
            resid, best_p, method="trf",
            bounds=([-np.inf, -np.inf, z_bounds[0]], [np.inf, np.inf, z_bounds[1]]),
            max_nfev=200,
        )
        return np.array([res.x[0], res.x[1], res.x[2]])

    def _clock_s(self, p, session, t_norm, tm2, n_sess):
        clock = np.zeros(len(session), dtype=np.float64)
        for k in range(n_sess):
            m = session == k
            if not np.any(m):
                continue
            j = sess_offset(k)
            clock[m] = p[j] + p[j + 1] * t_norm[m] + p[j + 2] * (t_norm[m] ** 2 - tm2[m])
        return clock

    def _apply_ssp(self, c1, c2):
        self.tracer.V = self.V_bg + c1 * self.eof[0] + c2 * self.eof[1]

    def solve_robust(self, df_beacon, initial_guess=None, depth_constraint=None,
                     lock_eof=True, lock_k=True, pos_prior=None, pos_tol=None,
                     progress_callback=None, session_gap_s=SESSION_GAP_S,
                     obs_sigma=None):
        rx_pos = df_beacon[["Rx_E", "Rx_N", "Rx_U"]].values
        t_obs = df_beacon["delay"].values
        absolute_seconds = np.asarray(to_unix_seconds(df_beacon.index), dtype=np.float64).reshape(-1)
        n_pings = len(t_obs)
        if len(absolute_seconds) != n_pings:
            raise ValueError(
                f"ping 时间与观测不等长: t={len(absolute_seconds)}, n_pings={n_pings}"
            )
        c_eq = self._c_global()
        obs_sigma = np.asarray(obs_sigma, dtype=np.float64).reshape(-1) if obs_sigma is not None else None
        if obs_sigma is not None and obs_sigma.size != n_pings:
            raise ValueError(f"obs_sigma length {obs_sigma.size} != n_pings {n_pings}")
        w_sigma = np.ones(n_pings, dtype=np.float64)
        if obs_sigma is not None:
            w_sigma = float(SIGMA_AC) / np.clip(obs_sigma, 1e-9, None)

        session, n_sess = detect_sessions(absolute_seconds, gap_s=session_gap_s)
        n_sess = max(1, n_sess)
        t_norm, tm2 = session_tau(absolute_seconds, session)
        spans = session_spans_s(absolute_seconds, session, n_sess)

        z_nom = float(depth_constraint[0]) if depth_constraint else self.ssp_measured_max_depth

        eof0_1 = eof0_2 = 0.0
        k0 = 0.0
        if initial_guess is not None and len(initial_guess) >= 10:
            core = np.asarray(initial_guess[:10], dtype=np.float64).copy()
            E0, N0, U0 = float(core[0]), float(core[1]), -abs(float(core[2]))
            bias0 = float(core[3])
            d1_0 = float(core[4]) * 3600.0
            d2_0 = float(core[5]) * (3600.0 ** 2)
            eof0_1, eof0_2 = float(core[6]), float(core[7])
            k0 = float(core[9]) if not lock_k else 0.0
        else:
            guess_pos = initial_guess if initial_guess is not None else self._estimate_initial_position(
                df_beacon, depth_constraint
            )
            E0, N0, U0 = float(guess_pos[0]), float(guess_pos[1]), -z_nom
            r_geom_approx = np.sqrt(
                (rx_pos[:, 0] - E0) ** 2 + (rx_pos[:, 1] - N0) ** 2 + (rx_pos[:, 2] - U0) ** 2
            )
            bias0 = np.clip(np.mean(t_obs - r_geom_approx / c_eq), -0.005, 0.005)
            d1_0 = d2_0 = 0.0

        x0 = np.zeros(N_GEOM + N_PER_SESS * n_sess, dtype=np.float64)
        x0[0], x0[1], x0[2], x0[IDX_K] = E0, N0, U0, k0
        for k in range(n_sess):
            j = sess_offset(k)
            x0[j] = bias0
            if k == 0:
                x0[j + 1], x0[j + 2] = d1_0, d2_0
                x0[j + 3], x0[j + 4] = eof0_1, eof0_2

        n_params = len(x0)
        lb = np.empty(n_params)
        ub = np.empty(n_params)
        lb[0], ub[0] = x0[0] - 20000.0, x0[0] + 20000.0
        lb[1], ub[1] = x0[1] - 20000.0, x0[1] + 20000.0
        lb[2], ub[2] = -10000.0, -10.0
        if lock_k:
            lb[IDX_K], ub[IDX_K] = -1e-12, 1e-12
            x0[IDX_K] = 0.0
        else:
            lb[IDX_K], ub[IDX_K] = -0.05, 0.05
        for k in range(n_sess):
            j = sess_offset(k)
            lb[j], ub[j] = -1e-1, 1e-1
            lb[j + 1], ub[j + 1] = -1e-3, 1e-3
            lb[j + 2], ub[j + 2] = -1e-6, 1e-6
            lb[j + 3], ub[j + 3] = -10.0, 10.0
            lb[j + 4], ub[j + 4] = -10.0, 10.0
            if spans[k] < SESSION_LOCK_D1_S:
                lb[j + 1], ub[j + 1] = -1e-12, 1e-12
                x0[j + 1] = 0.0
            if spans[k] < SESSION_LOCK_D2_S:
                lb[j + 2], ub[j + 2] = -1e-12, 1e-12
                x0[j + 2] = 0.0
            if lock_eof:
                lb[j + 3:j + 5] = -1e-12
                ub[j + 3:j + 5] = 1e-12
                x0[j + 3:j + 5] = 0.0
            elif spans[k] < SESSION_LOCK_D1_S:
                lb[j + 4], ub[j + 4] = -1e-12, 1e-12
                x0[j + 4] = 0.0

        if pos_prior is not None and pos_tol is not None:
            lb[0:2], ub[0:2] = pos_prior[0:2] - pos_tol[0:2], pos_prior[0:2] + pos_tol[0:2]
            u_prior = -abs(pos_prior[2])
            lb[2], ub[2] = u_prior - pos_tol[2], u_prior + pos_tol[2]

        bounds = (lb, ub)
        x0 = np.clip(x0, lb, ub)

        _batch_both_m = getattr(self.tracer, "calculate_travel_time_and_range_with_method_batch", None)
        _batch_both = getattr(self.tracer, "calculate_travel_time_and_range_batch", None)
        _rx_pos_arr = np.column_stack([rx_pos[:, 0], rx_pos[:, 1], rx_pos[:, 2]])
        # Single-station ray cache: clock-only changes do not alter the ray geometry.
        _rt_cache = {}
        _trace_mask = np.ones(n_pings, dtype=bool)

        def _trace_sessions(p, beacon_pos):
            t_ray = np.full(n_pings, np.nan, dtype=np.float64)
            s_arc = np.full(n_pings, np.nan, dtype=np.float64)
            methods = np.full(n_pings, "FAIL", dtype=object)
            # Use exact position keys: rounding could hide numerical Jacobian steps.
            P_key = (float(beacon_pos[0]), float(beacon_pos[1]), float(beacon_pos[2]))
            for k in range(n_sess):
                idx = np.flatnonzero((session == k) & _trace_mask)
                n_k = int(idx.size)
                if n_k == 0:
                    continue
                j = sess_offset(k)
                if lock_eof:
                    c1 = c2 = 0.0
                else:
                    c1, c2 = float(p[j + 3]), float(p[j + 4])
                key = (k, c1, c2, P_key, int(idx[0]), int(idx[-1]), n_k)
                hit = _rt_cache.get(key)
                if hit is not None and hit[0].shape[0] == n_k:
                    t_ray[idx] = hit[0]
                    s_arc[idx] = hit[1]
                    methods[idx] = hit[2]
                    continue
                if lock_eof:
                    self.tracer.V = self.V_bg
                else:
                    self._apply_ssp(c1, c2)
                rx_k = _rx_pos_arr[idx]
                if callable(_batch_both_m):
                    d, a, meth = _batch_both_m(rx_k, beacon_pos)
                elif callable(_batch_both):
                    d, a = _batch_both(rx_k, beacon_pos)
                    meth = np.where(np.isnan(d), "FAIL", "Snell")
                else:
                    d = np.empty(n_k, dtype=np.float64)
                    a = np.empty(n_k, dtype=np.float64)
                    meth = np.empty(n_k, dtype=object)
                    for ii in range(n_k):
                        d[ii] = self.tracer.calculate_travel_time(rx_k[ii], beacon_pos)
                        a[ii] = d[ii] * c_eq
                        meth[ii] = "Snell" if np.isfinite(d[ii]) else "FAIL"
                d = np.asarray(d, dtype=np.float64).reshape(-1)
                a = np.asarray(a, dtype=np.float64).reshape(-1)
                meth = np.asarray(meth, dtype=object).reshape(-1)
                if d.size != n_k or a.size != n_k or meth.size != n_k:
                    raise ValueError(
                        f"声线批处理长度不一致: n={n_k}, delay={d.size}, arc={a.size}, meth={meth.size}"
                    )
                packed = (d.copy(), a.copy(), meth.copy())
                _rt_cache[key] = packed
                t_ray[idx] = packed[0]
                s_arc[idx] = packed[1]
                methods[idx] = packed[2]
            return t_ray, s_arc, methods

        def _method_weight(methods):
            w = np.ones(len(methods), dtype=np.float64)
            w[np.asarray(methods) == "StraightLine"] = SNELL_FAIL_WEIGHT
            w[np.asarray(methods) == "FAIL"] = 0.0
            return w

        def _priors(p):
            rows = []
            if not lock_k:
                rows.append(p[IDX_K] / SIGMA_ZENITH)
            for k in range(n_sess):
                j = sess_offset(k)
                rows.extend([
                    p[j] / SIGMA_BIAS,
                    p[j + 1] / SIGMA_D1,
                    p[j + 2] / SIGMA_D2,
                ])
                if not lock_eof:
                    rows.extend([
                        p[j + 3] / SIGMA_EOF,
                        p[j + 4] / SIGMA_EOF,
                    ])
            return np.asarray(rows, dtype=np.float64)

        def _predict(p):
            true_rx_E = _rx_pos_arr[:, 0]
            true_rx_N = _rx_pos_arr[:, 1]
            beacon_pos = np.array([p[0], p[1], -abs(p[2])])
            t_ray, s, methods = _trace_sessions(p, beacon_pos)
            clock_s = self._clock_s(p, session, t_norm, tm2, n_sess)
            dist_sq = (true_rx_E - p[0]) ** 2 + (true_rx_N - p[1]) ** 2
            zenith_s = 0.0 if lock_k else p[IDX_K] * (dist_sq / 1e6)
            t_model = np.where(np.isnan(t_ray), 1e6, t_ray + clock_s + zenith_s)
            c_loc = equivalent_sound_speed(t_ray, s, fallback=c_eq)
            dz = np.abs(-abs(p[2]) - _rx_pos_arr[:, 2])
            slant = np.sqrt(dist_sq + (_rx_pos_arr[:, 2] + abs(p[2])) ** 2)
            w = (dz / np.clip(slant, 1.0, None)) * _method_weight(methods)
            return t_model, c_loc, w, methods, t_ray, s

        _nfev = [0]

        def objective(p):
            _nfev[0] += 1
            if _nfev[0] == 1 or _nfev[0] % 25 == 0:
                print(f"[Solver] nfev={_nfev[0]}", flush=True)
            t_model, _, w, *_ = _predict(p)
            ping_res = (t_obs - t_model) * c_eq * w * w_sigma
            return np.concatenate([ping_res, _priors(p)])

        def jacobian(p):
            """Clock/priors analytic; position 1 mm forward difference; locked EOF/k columns 0."""
            t_model, _, w, *_ = _predict(p)
            ping_res = (t_obs - t_model) * c_eq * w * w_sigma
            prior = _priors(p)
            n_par = len(p)
            J = np.zeros((n_pings + len(prior), n_par), dtype=np.float64)
            scale = -c_eq * w * w_sigma
            step = 1e-3
            for i in range(3):
                dp = np.array(p, copy=True)
                dp[i] = dp[i] + step
                t_p, _, w_p, *_ = _predict(dp)
                ping_p = (t_obs - t_p) * c_eq * w_p * w_sigma
                J[:n_pings, i] = (ping_p - ping_res) / step
            if not lock_k:
                dist_sq = (_rx_pos_arr[:, 0] - p[0]) ** 2 + (_rx_pos_arr[:, 1] - p[1]) ** 2
                J[:n_pings, IDX_K] = scale * (dist_sq / 1e6)
            for k in range(n_sess):
                j = sess_offset(k)
                m = session == k
                if not np.any(m):
                    continue
                idx = np.flatnonzero(m)
                J[idx, j] = scale[idx]
                J[idx, j + 1] = scale[idx] * t_norm[idx]
                J[idx, j + 2] = scale[idx] * (t_norm[idx] ** 2 - tm2[idx])
                if not lock_eof:
                    for ic in (3, 4):
                        dp = np.array(p, copy=True)
                        dp[j + ic] = dp[j + ic] + 1e-4
                        t_p, _, w_p, *_ = _predict(dp)
                        ping_p = (t_obs - t_p) * c_eq * w_p * w_sigma
                        J[:n_pings, j + ic] = (ping_p - ping_res) / 1e-4
            irow = n_pings
            if not lock_k:
                J[irow, IDX_K] = 1.0 / SIGMA_ZENITH
                irow += 1
            for k in range(n_sess):
                j = sess_offset(k)
                J[irow, j] = 1.0 / SIGMA_BIAS
                irow += 1
                J[irow, j + 1] = 1.0 / SIGMA_D1
                irow += 1
                J[irow, j + 2] = 1.0 / SIGMA_D2
                irow += 1
                if not lock_eof:
                    J[irow, j + 3] = 1.0 / SIGMA_EOF
                    irow += 1
                    J[irow, j + 4] = 1.0 / SIGMA_EOF
                    irow += 1
            return J

        print(f"[Solver] 启动纯净 TOA 内核寻优 (航次={n_sess}, C_EQ={c_eq:.2f} m/s, lock_k={lock_k}, lock_eof={lock_eof})...")
        res_coarse = least_squares(
            objective, x0, jac=jacobian, bounds=bounds, method="trf", loss="huber", f_scale=1.5,
            max_nfev=200, ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )

        t_model_c, _, _, methods_c, *_ = _predict(res_coarse.x)
        res_pure = (t_obs - t_model_c) * c_eq
        mad = np.median(np.abs(res_pure - np.median(res_pure)))
        robust_std = mad * 1.4826
        valid_mask = (np.abs(res_pure) <= max(3.0 * robust_std, 1.5)) & (np.asarray(methods_c) != "FAIL")
        num_outliers = n_pings - int(np.sum(valid_mask))

        _trace_mask = valid_mask
        _rt_cache.clear()

        def objective_fine(p):
            r = objective(p)
            return np.concatenate([r[:n_pings][valid_mask], r[n_pings:]])

        def jac_fine(p):
            J = jacobian(p)
            return np.concatenate([J[:n_pings][valid_mask], J[n_pings:]], axis=0)

        res_fine = least_squares(
            objective_fine, res_coarse.x, jac=jac_fine, bounds=bounds, method="trf", loss="linear",
            max_nfev=400, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        opt = res_fine.x

        _trace_mask = np.ones(n_pings, dtype=bool)
        t_model, c_loc, w, final_methods, t_ray, s = _predict(opt)
        res_s = t_obs - t_model
        residuals_m = res_s * c_loc

        J = getattr(res_fine, "jac", None)
        residuals_clean = res_fine.fun
        n_params = len(opt)
        n_obs_fine = len(residuals_clean)
        try:
            if J is None:
                raise ValueError("jacobian is None")
            dof = max(1, n_obs_fine - n_params)
            sigma0_sq = np.sum(residuals_clean ** 2) / dof
            cov_matrix = sigma0_sq * np.linalg.pinv(J.T @ J + np.eye(n_params) * 1e-12)
            m_E, m_N, m_U = np.sqrt(np.abs(np.diag(cov_matrix)[:3]))
        except (np.linalg.LinAlgError, ValueError, TypeError, AttributeError):
            m_E = m_N = m_U = sigma0_sq = np.nan

        inlier = valid_mask
        sigma0_time = np.sqrt(np.mean(res_s[inlier] ** 2)) if np.any(inlier) else np.nan
        sigma0_m = sigma0_time * c_eq
        wrms = np.sqrt(np.mean(residuals_m[inlier] ** 2)) if np.any(inlier) else np.nan

        if np.any(inlier):
            k_dom = dominant_session(session[inlier], n_sess, absolute_seconds[inlier])
        else:
            k_dom = dominant_session(session, n_sess, absolute_seconds)
        j_dom = sess_offset(k_dom)

        print(
            f"[Solver] TOA 内符合精度: E= ±{m_E:.4f}m, N= ±{m_N:.4f}m, U= ±{m_U:.4f}m, "
            f"航次={n_sess}, 主导航次={k_dom}, k={'0(锁)' if lock_k else f'{opt[IDX_K]:.4e}'}"
        )
        clock_by_session = []
        eof_by_session = []
        for k in range(n_sess):
            j = sess_offset(k)
            n_k = int(np.sum(session == k))
            clock_by_session.append({
                "session": k, "n_ping": n_k, "span_s": float(spans[k]),
                "B": float(opt[j]),
                "D1": float(opt[j + 1] / 3600.0),
                "D2": float(opt[j + 2] / (3600.0 ** 2)),
            })
            eof_by_session.append({
                "session": k, "EOF1": float(opt[j + 3]), "EOF2": float(opt[j + 4]),
            })
            print(
                f"      sess{k}: n={n_k}  span={spans[k]/60.0:.1f}min  "
                f"B={opt[j]*1e3:.2f}ms  EOF=({opt[j+3]:.3f},{opt[j+4]:.3f})"
            )

        method_stats = {
            "Snell": int(np.sum(final_methods == "Snell")),
            "StraightLine": int(np.sum(final_methods == "StraightLine")),
            "FAIL": int(np.sum(final_methods == "FAIL")),
            "outliers": int(num_outliers),
            "n_sessions": int(n_sess),
            "k_dom": int(k_dom),
        }

        final_params = np.array([
            opt[0], opt[1], -abs(opt[2]),
            opt[j_dom],
            opt[j_dom + 1] / 3600.0,
            opt[j_dom + 2] / (3600.0 ** 2),
            opt[j_dom + 3], opt[j_dom + 4],
            0.0, 0.0 if lock_k else float(opt[IDX_K]),
        ])
        precision_metrics = {
            "m_E": m_E, "m_N": m_N, "m_U": m_U, "sigma0": sigma0_m,
            "wrms": wrms, "n_sessions": int(n_sess), "n_params": int(n_params),
            "k_dom": int(k_dom), "c_eq": float(c_eq),
            "lock_k": bool(lock_k), "k_zenith": 0.0 if lock_k else float(opt[IDX_K]),
            "clock_by_session": clock_by_session,
            "eof_by_session": eof_by_session,
        }
        return final_params, residuals_m, method_stats, final_methods, precision_metrics
