import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import least_squares

# Import the v2.0 simulation generator
from simulation_generator import generate_simulation_data, load_ssp_from_csv

# Fallback only; prefer SSP harmonic mean when a profile is present.
C_REF = 1496.32


# =====================================================================
#  Multi-beacon Least-Squares solver  (time domain, ray‑tracing)
# =====================================================================

def solve_beacon_positions(
    obs_ship_pos: np.ndarray,          # (N, 3)  OBSERVED transducer [x, y, z] (with GNSS noise)
    obs_delays: np.ndarray,            # (N,)    OBSERVED acoustic delays (s, with acoustic noise)
    c_ref: float = C_REF,
    ray_tracer: Optional[Any] = None,  # Snell 打靶声线引擎 drop‑in replacement
    initial_guess: Optional[np.ndarray] = None,
    depth_prior: Optional[float] = None,   # 深度先验 (m, 正值)，锚定深度分量
    depth_weight: float = 1e3,
    progress_callback: Optional[object] = None,   # 每次目标求值回调 (0-100 进度)
) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """
    Estimate a single beacon's 3‑D position from N arrival‑time observations.

    Observation model (time domain)
    -------------------------------
        t_obs_i   = obs_delays[i]                          [s]
        t_model_i = ray_trace(obs_ship_pos[i], B_est)     [s]  (or ‖Δ‖ / c_ref)
        residual  = t_obs − t_model                        [s]

    The solver estimates ``B = [x, y, z]`` (3 parameters).  A
    constant clock offset (Δt) is recovered analytically.

    Parameters
    ----------
    obs_ship_pos : (N, 3) np.ndarray
        Observed transducer positions (noisy GNSS + lever arm).
    obs_delays : (N,) np.ndarray
        Observed acoustic travel times (s), with measurement noise.
    c_ref : float
        Reference sound speed (used only when ray_tracer is None).
    ray_tracer : object, optional
        An object that implements ``calculate_travel_time(P_trans, B) → τ``.
        Typically a Snell 打靶 ``RayTracer`` instance.
    initial_guess : np.ndarray, optional

    Returns
    -------
    B_est : np.ndarray
    rmse_ms : float        RMS time residual (ms)
    rmse_m : float         Equivalent range RMS (m)
    residuals_s : np.ndarray  Per-ping time residuals (s)
    calc_delays_s : np.ndarray  Final calculated (ray‑traced) delays (s)
    """
    t_obs = obs_delays

    if initial_guess is None:
        x0 = float(np.median(obs_ship_pos[:, 0]))
        y0 = float(np.median(obs_ship_pos[:, 1]))
        # 几何分解初始深度：斜距² = 水平半径² + 深度²。
        # 旧实现直接把 mean(t)·c 当深度（其实是斜距），导致深水深度被高估 ~ 1 个水平半径。
        h = float(np.median(np.hypot(obs_ship_pos[:, 0] - x0,
                                     obs_ship_pos[:, 1] - y0)))
        slant = float(np.mean(t_obs) * c_ref)
        z0 = -float(np.sqrt(max(slant * slant - h * h, 0.0)))
        initial_guess = np.array([x0, y0, z0])

    _eval = [0]

    def objective(B: np.ndarray) -> np.ndarray:
        """Time residuals [s] for given beacon position B."""
        if ray_tracer is not None:
            _batch_fn = getattr(ray_tracer, "calculate_travel_time_batch", None)
            if callable(_batch_fn):
                t_model = _batch_fn(obs_ship_pos, B)
            else:
                t_model = np.empty_like(t_obs)
                for i in range(len(obs_ship_pos)):
                    t_model[i] = ray_tracer.calculate_travel_time(obs_ship_pos[i], B)
        else:
            d_vec = obs_ship_pos - B[np.newaxis, :]
            r_model = np.sqrt(np.sum(d_vec ** 2, axis=1))
            t_model = r_model / c_ref
        raw_diff = t_obs - t_model
        dt = np.mean(raw_diff)
        res = raw_diff - dt
        # 深度软约束：单侧走航下深度与时钟偏差强耦合，必须由先验锚定。
        if depth_prior is not None:
            res = np.append(res, (abs(float(B[2])) - depth_prior) * depth_weight)
        # 每求值回调一次进度 (按估计求值上限 50 归一)，使进度条随计算推进。
        _eval[0] += 1
        if progress_callback:
            progress_callback(min(_eval[0] / 50.0, 1.0) * 100.0)
        return res

    bounds = (
        [-np.inf, -np.inf, -15000.0],
        [np.inf,  np.inf,  0.0],
    )

    print(f"[SOLVER] obs_ship_pos {obs_ship_pos.shape}, any NaN/Inf: "
          f"{np.any(np.isnan(obs_ship_pos))}/{np.any(np.isinf(obs_ship_pos))}")
    print(f"[SOLVER] obs_delays {obs_delays.shape}, "
          f"min={np.min(obs_delays)*1000:.2f} ms  max={np.max(obs_delays)*1000:.2f} ms, "
          f"ray_tracer={'LUT' if ray_tracer else 'none'}")

    try:
        result = least_squares(
            objective, initial_guess, bounds=bounds,
            method="trf", loss="soft_l1", f_scale=5.0, max_nfev=200,
        )
        B_est = result.x
        residuals_s = objective(B_est)
        rmse_ms = float(np.sqrt(np.mean((residuals_s * 1000) ** 2)))
        rmse_m  = rmse_ms / 1000.0 * c_ref

        # Compute final ray‑traced (or straight‑line) delays
        if ray_tracer is not None:
            _batch_fn = getattr(ray_tracer, "calculate_travel_time_batch", None)
            if callable(_batch_fn):
                calc_delays = _batch_fn(obs_ship_pos, B_est)
            else:
                calc_delays = np.array([
                    ray_tracer.calculate_travel_time(obs_ship_pos[i], B_est)
                    for i in range(len(obs_ship_pos))
                ])
        else:
            d_vec = obs_ship_pos - B_est[np.newaxis, :]
            calc_delays = np.sqrt(np.sum(d_vec ** 2, axis=1)) / c_ref

        print(f"[SOLVER] converged → B=({B_est[0]:.1f},{B_est[1]:.1f},{B_est[2]:.1f}) "
              f"rmse={rmse_ms:.3f} ms  (≈ {rmse_m:.3f} m)")
        return B_est, rmse_ms, rmse_m, residuals_s, calc_delays
    except Exception:
        res = objective(initial_guess)
        rmse_ms = float(np.sqrt(np.mean((res * 1000) ** 2)))
        rmse_m  = rmse_ms / 1000.0 * c_ref
        print(f"[SOLVER] LS failed — fallback, rmse={rmse_ms:.3f} ms")
        calc_delays = t_obs - res
        return initial_guess, rmse_ms, rmse_m, res, calc_delays


def solve_beacon_array(
    obs_ship_pos: np.ndarray,                        # (N, 3)  OBSERVED transducer positions
    obs_delays_per_beacon: np.ndarray,                # (N, M)  OBSERVED delays
    true_beacons: Optional[np.ndarray] = None,        # (M, 3)  ground truth for RMSE
    c_ref: float = C_REF,
    ray_tracer: Optional[object] = None,
    depth_priors: Optional[np.ndarray] = None,        # (M,) 每信标深度先验 (m, 正值)
    depth_weight: float = 1e3,
    progress_callback: Optional[object] = None,        # 信标内求解进度回调 (0-100)
) -> Dict[str, Any]:
    """
    Solve for all beacons using ONLY observed (noisy) ship positions and delays.

    Parameters
    ----------
    obs_ship_pos : (N, 3) np.ndarray
        Observed transducer positions (noisy GNSS + lever arm).
    obs_delays_per_beacon : (N, M) np.ndarray
        Observed delays for each ping (row) and each beacon (column), with noise.
    true_beacons : (M, 3) np.ndarray, optional
        Ground-truth positions for 3‑D RMSE computation.
    c_ref : float
    ray_tracer : optional
        Snell 打靶声线引擎实例，供 LS 迭代中在线计算旅行时。

    Returns
    -------
    dict with:
        estimated_beacons — per-beacon estimated [x,y,z] and RMS metrics
        position_errors_m — 3‑D Euclidean error vs ground truth
        rmse_3d_m         — overall RMS 3‑D position error
        residuals_ms      — per-beacon per-ping time residuals (ms)
    """
    n_pings, n_beacons = obs_delays_per_beacon.shape
    estimated = []
    pos_errors = []
    time_rmse_ms_list: list[float] = []
    distance_rmse_list: list[float] = []
    all_residuals_ms: list[np.ndarray] = []

    for j in range(n_beacons):
        dp = float(depth_priors[j]) if depth_priors is not None else None

        def _inner_cb(pct, _j=j):
            if progress_callback:
                progress_callback((_j + max(float(pct), 0.0) / 100.0) / n_beacons * 100.0)

        B_est, rmse_ms, rmse_m, _, calc_delays = solve_beacon_positions(
            obs_ship_pos=obs_ship_pos,
            obs_delays=obs_delays_per_beacon[:, j],
            c_ref=c_ref,
            ray_tracer=ray_tracer,
            depth_prior=dp,
            depth_weight=depth_weight,
            progress_callback=_inner_cb,
        )
        if progress_callback:
            progress_callback((j + 1) / n_beacons * 100.0)
        time_rmse_ms_list.append(rmse_ms)
        distance_rmse_list.append(rmse_m)

        estimated.append({
            "id": j,
            "x": float(B_est[0]),
            "y": float(B_est[1]),
            "z": float(B_est[2]),
            "rmse_time_ms": round(rmse_ms, 3),
            "rmse_equiv_m": round(rmse_m, 3),
        })

        # Compute per-ping time residuals for this beacon (ms)
        if true_beacons is not None and j < true_beacons.shape[0]:
            err_3d = float(np.linalg.norm(B_est - true_beacons[j]))
            pos_errors.append(err_3d)
        else:
            pos_errors.append(np.nan)

    # Overall 3‑D RMSE (vs ground truth)
    valid_errors = [e for e in pos_errors if not np.isnan(e)]
    rmse_3d = float(np.sqrt(np.mean(np.array(valid_errors) ** 2))) if valid_errors else np.nan

    # GDOP at array centre
    if true_beacons is not None:
        centre = np.mean(true_beacons, axis=0)
    else:
        centre = np.array([
            np.mean([b["x"] for b in estimated]),
            np.mean([b["y"] for b in estimated]),
            np.mean([b["z"] for b in estimated]),
        ])
    gdop_val = _compute_gdop_at(obs_ship_pos, centre)

    return {
        "estimated_beacons": estimated,
        "position_errors_m": pos_errors,
        "rmse_3d_m": round(rmse_3d, 4) if not np.isnan(rmse_3d) else None,
        "rmse_time_ms_mean": round(float(np.mean(time_rmse_ms_list)), 3),
        "rmse_equiv_m_mean": round(float(np.mean(distance_rmse_list)), 3),
        "gdop": round(gdop_val, 6),
        "array_centre": {
            "x": float(centre[0]), "y": float(centre[1]), "z": float(centre[2]),
        },
    }


def _compute_gdop_at(
    transducer_points: np.ndarray,   # (N, 3)
    beacon_position: np.ndarray,     # (3,)
) -> float:
    """
    Compute GDOP at a specific beacon position given transducer geometry.

    Jacobian of the range equation w.r.t. beacon (X, Y):
        ∂r_i/∂X = −(X − x_i) / r_i
        ∂r_i/∂Y = −(Y − y_i) / r_i

    For multi-beacon evaluation, we compute GDOP for the centre beacon.
    """
    x_t = transducer_points[:, 0]
    y_t = transducer_points[:, 1]
    z_t = transducer_points[:, 2]

    bx, by, bz = beacon_position

    # Slant range from each transducer to this beacon
    dx = bx - x_t
    dy = by - y_t
    dz = bz - z_t
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Jacobian — N×2 (we don't solve for Z, which is weakly constrained
    # by geometry; but including Z would make it 3×3 which we also do)
    A_xy = np.column_stack([dx / r, dy / r])           # N×2
    A_xyz = np.column_stack([dx / r, dy / r, dz / r]) # N×3

    try:
        N_xy = A_xy.T @ A_xy
        Q_xy = np.linalg.inv(N_xy)
        gdop_xy = np.sqrt(np.trace(Q_xy))

        N_xyz = A_xyz.T @ A_xyz
        Q_xyz = np.linalg.inv(N_xyz)
        gdop_xyz = np.sqrt(np.trace(Q_xyz))

        # Return the 3D GDOP (accounts for depth uncertainty too)
        return float(gdop_xyz)
    except np.linalg.LinAlgError:
        return np.inf


# =====================================================================
#  Calibration Trajectory Optimizer
# =====================================================================

class CalibrationTrajectoryOptimizer:
    """
    Sweep-based calibration trajectory optimizer.

    Generates candidate survey trajectories over a range of sizes,
    simulates noisy acoustic observations with full physical modelling
    (lever-arm, GNSS noise, SSP ray-tracing), solves for beacon positions
    via Least Squares, and selects three recommended survey plans.

    Parameters
    ----------
    array_points : list[list[float]]
        [[x, y, z], ...]  seafloor beacon positions (m).  Z ≤ 0.
    ssp_df : pd.DataFrame, optional
        Sound Speed Profile.  If None, constant sound speed is used.
    lever_arm : list[float], optional
        [dx, dy, dz]  (m).  Default [0, 0, -30].
    gnss_noise_std : float
        Per-axis GNSS position noise std (m).  Default 0.05.
    acoustic_noise_std : float
        Std of acoustic delay measurement noise (seconds).  Default 0.05.
    depth_H : float, optional
        Override nominal depth.  If None, use median |z| from array.
    min_size_ratio : float
        Minimum trajectory size = depth × ratio.  Default 0.5.
    max_size_ratio : float
        Maximum trajectory size = depth × ratio.  Default 2.0.
    size_step : float
        Size increment (m).  Default 200.
    n_pings : int
        Number of discrete pings per trajectory.  Default 360.
    use_ray_tracing : bool
        Enable acoustic ray tracing.  Default True.
    sv_grad_e, sv_grad_n : float
        Horizontal sound-speed gradient (m/s)/m applied only when
        generating observations.  The 9-parameter solver does not estimate it.
    modes : list[str], optional
        Trajectory families to sweep.
    """

    def __init__(
        self,
        array_points: List[List[float]],
        ssp_df: Optional[pd.DataFrame] = None,
        lever_arm: Optional[List[float]] = None,
        gnss_noise_std: float = 0.05,
        acoustic_noise_std: float = 0.05,
        depth_H: Optional[float] = None,
        min_size_ratio: float = 0.5,
        max_size_ratio: float = 2.0,
        size_step: float = 200.0,
        n_pings: int = 360,
        use_ray_tracing: bool = True,
        # 寻优固定走 9 参数（估天顶 k）。保留形参以免旧调用报错。
        solver_type: str = "9param",
        inject_system_errors: bool = False,
        system_clock_bias_s: float = 0.0,
        clock_drift_ss: float = 0.0,
        time_lag_s: float = 0.0,
        ntd_k: float = 0.0,
        lever_arm_error: Optional[List[float]] = None,
        sv_grad_e: float = 0.0,
        sv_grad_n: float = 4.0e-5,
        modes: Optional[List[str]] = None,
        size_ratios: Optional[List[float]] = None,
        rng_seed: Optional[int] = None,
    ):
        self.beacons = np.array(array_points, dtype=np.float64)   # (M, 3)
        self.n_beacons = self.beacons.shape[0]

        # --- Geometric centre ---
        self.centre = np.mean(self.beacons, axis=0)               # (3,)
        # Represent depth as a positive magnitude for size calcs
        self._depth_mag = abs(float(self.centre[2])) if depth_H is None else abs(depth_H)
        if self._depth_mag < 1.0:
            self._depth_mag = 3000.0   # safeguard

        # --- Physical parameters ---
        self.ssp_df = ssp_df
        self.lever_arm = lever_arm if lever_arm is not None else [0.0, 0.0, -30.0]
        self.gnss_noise_std = gnss_noise_std
        self.acoustic_noise_std = acoustic_noise_std
        self.use_ray_tracing = use_ray_tracing and ssp_df is not None
        self.n_pings = n_pings

        # 寻优使用 ABIC 求解器（估天顶 k 和水平梯度）。航程不进 ABIC。
        self.solver_type = "abic_grad"
        self.inject_system_errors = inject_system_errors
        self.system_clock_bias_s = system_clock_bias_s
        self.clock_drift_ss = clock_drift_ss
        self.time_lag_s = time_lag_s
        self.ntd_k = ntd_k
        self.lever_arm_error = lever_arm_error
        self.sv_grad_e = float(sv_grad_e)
        self.sv_grad_n = float(sv_grad_n)

        # --- Size sweep ---
        # size_ratios 给出相对水深的尺度（对数覆盖 0.1–10 时用这个，而不是等间距）。
        self.size_ratios = [float(r) for r in size_ratios if float(r) > 0] if size_ratios else None
        self.rng_seed = rng_seed
        self.min_size = min_size_ratio * self._depth_mag
        self.max_size = max_size_ratio * self._depth_mag
        self.size_step = size_step

        # --- Trajectory types ---
        allowed = ("circle", "circle_cross", "square", "asterisk")
        picked = [m for m in (modes or list(allowed)) if m in allowed]
        self.modes = picked or ["circle", "circle_cross", "square", "asterisk"]

        # --- Results cache ---
        self.results: List[Dict[str, Any]] = []

        if self.size_ratios:
            size_msg = f"size ratios={self.size_ratios}"
        else:
            size_msg = (
                f"size sweep=[{self.min_size:.0f}, {self.max_size:.0f}] "
                f"step={self.size_step:.0f} m"
            )
        print(
            f"[CALIB-OPT] Initialised with {self.n_beacons} beacons, "
            f"centre=({self.centre[0]:.1f}, {self.centre[1]:.1f}, {self.centre[2]:.1f}), "
            f"depth={self._depth_mag:.1f} m, {size_msg}, "
            f"solver=abic_grad, g_E={self.sv_grad_e:.3e}, g_N={self.sv_grad_n:.3e} (m/s)/m"
        )
        if self.inject_system_errors:
            print(f"[CALIB-OPT] Systemic error injection ENABLED: "
                  f"C={self.system_clock_bias_s*1000:.2f}ms, D={self.clock_drift_ss:.3e}s/s, "
                  f"lag={self.time_lag_s*1000:.2f}ms, k={self.ntd_k:.6f}, "
                  f"lever_err={self.lever_arm_error}")

    # ==================================================================
    #  Evaluate a single candidate
    # ==================================================================
    def evaluate_candidate(
        self, mode: str, size: float, progress_callback=None
    ) -> Dict[str, Any]:
        """
        Simulate a single candidate trajectory and compute its calibration accuracy.

        Returns
        -------
        dict with:
            type, size_m, path_length_m,
            rmse_position_m, gdop,
            estimated_beacons, position_errors_m,
            observations, trajectory,
        """
        def _sub(frac: float):
            if progress_callback:
                progress_callback(frac)
        _sub(0.0)

        # ------------------------------------------------------------------
        #  1.  Generate noised observations via the v2.0 simulation engine
        #
        #  The trajectory is centred at the beacon array's geometric centre.
        #  We translate the waypoints so that the ship circles around the
        #  array (instead of around (0,0)).
        # ------------------------------------------------------------------
        # Build translated beacon positions relative to the centre.
        # 仅平移水平 XY：深度是绝对量（以海面为基准），不能减去中心深度，
        # 否则信标深度被错误归零 → z_rcv < z_src，Snell 几何非法全线 FAIL。
        self._centre_xy = np.array([self.centre[0], self.centre[1], 0.0])
        translated_beacons = (self.beacons - self._centre_xy[np.newaxis, :]).tolist()

        try:
            data = generate_simulation_data(
                array_points=translated_beacons,
                radius_R=size,
                mode=mode,                                    # type: ignore[arg-type]
                n_points=self.n_pings,
                noise_std=self.acoustic_noise_std,
                lever_arm=self.lever_arm,
                gnss_noise_std=self.gnss_noise_std,
                ssp_df=self.ssp_df,
                use_ray_tracing=self.use_ray_tracing,
                # Inject systemic errors if requested
                system_clock_bias_s=self.system_clock_bias_s if self.inject_system_errors else 0.0,
                clock_drift_ss=self.clock_drift_ss if self.inject_system_errors else 0.0,
                time_lag_s=self.time_lag_s if self.inject_system_errors else 0.0,
                ntd_k=self.ntd_k if self.inject_system_errors else 0.0,
                lever_arm_error=self.lever_arm_error if self.inject_system_errors else None,
                sv_grad_e=self.sv_grad_e,
                sv_grad_n=self.sv_grad_n,
                rng_seed=self.rng_seed,
            )
        except Exception as exc:
            print(f"[CALIB-OPT] WARN — sim failed for {mode} R={size}: {exc}")
            return {
                "type": mode, "size_m": size,
                "path_length_m": np.nan, "rmse_position_m": np.nan,
                "gdop": np.inf, "valid": False, "error": str(exc),
            }

        observations = data["observations"]

        if len(observations) == 0:
            _sub(1.0)
            return {
                "type": mode, "size_m": size,
                "path_length_m": np.nan, "rmse_position_m": np.nan,
                "gdop": np.inf, "valid": False, "error": "No observations generated",
            }
        _sub(0.3)

        # ------------------------------------------------------------------
        #  2.  Extract OBSERVED data arrays (NO GROUND TRUTH)
        #
        #  Use observED transducer positions (noisy GNSS + lever arm)
        #  and observED delay times. These are the only inputs the
        #  solver is allowed to see.
        # ------------------------------------------------------------------
        n_pings_actual = len(observations)
        obs_ship = np.zeros((n_pings_actual, 3))
        obs_delays_mat = np.zeros((n_pings_actual, self.n_beacons))

        for i, obs in enumerate(observations):
            # Observed transducer = noisy GNSS + lever arm
            obs_ship[i, 0] = obs["transducer_x"]
            obs_ship[i, 1] = obs["transducer_y"]
            obs_ship[i, 2] = obs["transducer_z"]

            for j in range(self.n_beacons):
                obs_delays_mat[i, j] = obs["noisy_delay_per_beacon_s"][j]

        # Back-translate observed positions to absolute frame (仅回加水平 XY，深度不变)
        obs_ship_abs = obs_ship + self._centre_xy[np.newaxis, :]

        # ------------------------------------------------------------------
        #  3.  ABIC TOA：一维声线 + 钟差 + 天顶 k + 共用水平梯度。
        #  航程不进 ABIC，只在后面的方案比较里使用。
        # ------------------------------------------------------------------
        sys_err_comparison: Optional[Dict[str, Any]] = None

        if self.ssp_df is None:
            _sub(1.0)
            return {
                "type": mode, "size_m": size,
                "path_length_m": np.nan, "rmse_position_m": np.nan,
                "gdop": None, "valid": False,
                "error": "ABIC 求解需要声速剖面",
            }

        from abic_solver import solve_with_abic

        times_s = np.array([float(obs["time_s"]) for obs in observations], dtype=np.float64)

        def _beacon_cb(pct, _msg=None):
            _sub(0.3 + 0.7 * max(min(float(pct), 100.0), 0.0) / 100.0)

        cal_res = solve_with_abic(
            ship=obs_ship_abs,
            delays=obs_delays_mat,
            ssp_df=self.ssp_df,
            times_s=times_s,
            origin_xy=(float(self.centre[0]), float(self.centre[1])),
            progress_callback=_beacon_cb,
        )
        if cal_res.get("status") != "success":
            message = str(cal_res.get("message", "ABIC 求解失败"))
            print(f"[CALIB-OPT] ABIC FAILED: {message}")
            _sub(1.0)
            return {
                "type": mode, "size_m": size,
                "path_length_m": np.nan, "rmse_position_m": np.nan,
                "gdop": None, "valid": False, "error": message,
            }

        estimated: List[Dict[str, Any]] = []
        horizontal_errors: List[float] = []
        errors_3d: List[float] = []
        time_rmses: List[float] = []
        k_estimates: List[Any] = []
        for j, est in enumerate(cal_res["beacons"]):
            b_est = np.array([est["x"], est["y"], est["z"]], dtype=np.float64)
            truth = self.beacons[j]
            h_err = float(np.hypot(b_est[0] - truth[0], b_est[1] - truth[1]))
            e3 = float(np.linalg.norm(b_est - truth))
            horizontal_errors.append(h_err)
            errors_3d.append(e3)
            time_rmses.append(float(est.get("rmse_time_ms") or 0.0))
            k_estimates.append(est.get("zenith_k_ms_km2"))
            estimated.append({
                "id": j,
                "x": est["x"],
                "y": est["y"],
                "z": est["z"],
                "rmse_time_ms": est.get("rmse_time_ms", 0),
                "rmse_equiv_m": est.get("rmse_equiv_m"),
                "zenith_k_ms_km2": est.get("zenith_k_ms_km2"),
            })
            print(
                f"[CALIB-OPT] ABIC beacon {j} — "
                f"X={b_est[0]:.1f}, Y={b_est[1]:.1f}, Z={b_est[2]:.1f}, "
                f"dH={h_err:.2f} m, k={est.get('zenith_k_ms_km2')}"
            )
        print(
            f"[CALIB-OPT] ABIC g_E={cal_res['sv_grad_e_mps_per_km']:.4f} (m/s)/km, "
            f"g_N={cal_res['sv_grad_n_mps_per_km']:.4f} (m/s)/km, "
            f"λ={cal_res['abic_lambda']:.3g}, ABIC={cal_res['abic']}"
        )

        rmse_h = float(np.sqrt(np.mean(np.square(horizontal_errors))))
        rmse_3d = float(np.sqrt(np.mean(np.square(errors_3d))))
        sol = {
            "estimated_beacons": estimated,
            "position_errors_m": errors_3d,
            "horizontal_errors_m": horizontal_errors,
            "rmse_3d_m": rmse_3d,
            "rmse_horizontal_m": rmse_h,
            "gdop": None,
            "array_centre": {
                "x": float(np.mean([b["x"] for b in estimated])),
                "y": float(np.mean([b["y"] for b in estimated])),
                "z": float(np.mean([b["z"] for b in estimated])),
            },
            "rmse_time_ms_mean": float(np.mean(time_rmses)) if time_rmses else None,
            "zenith_k_ms_km2": k_estimates,
            "sv_grad_e_mps_per_km": cal_res["sv_grad_e_mps_per_km"],
            "sv_grad_n_mps_per_km": cal_res["sv_grad_n_mps_per_km"],
            "abic": cal_res["abic"],
            "abic_lambda": cal_res["abic_lambda"],
        }

        # ------------------------------------------------------------------
        #  4.  Path length
        # ------------------------------------------------------------------
        traj = data["trajectory"]
        if len(traj) > 1:
            pts = np.array([[p["x"], p["y"], p["z"]] for p in traj])
            diffs = np.diff(pts, axis=0)
            path_len = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
        else:
            path_len = 0.0

        # ------------------------------------------------------------------
        #  5.  Build result
        # ------------------------------------------------------------------
        result = {
            "type": mode,
            "size": round(size, 1),
            "size_m": round(size, 1),
            "length": round(path_len, 2),
            "path_length_m": round(path_len, 2),
            # 排序用水平误差：梯度造成的是水平平移，深度不拿来比方案。
            "rmse_position_m": sol.get("rmse_horizontal_m"),
            "rmse": sol.get("rmse_horizontal_m"),
            "rmse_horizontal_m": sol.get("rmse_horizontal_m"),
            "rmse_3d_m": sol.get("rmse_3d_m"),
            "gdop": sol.get("gdop"),
            "position_errors_m": sol.get("position_errors_m"),
            "horizontal_errors_m": sol.get("horizontal_errors_m"),
            "estimated_beacons": sol.get("estimated_beacons"),
            "rmse_time_ms_mean": sol.get("rmse_time_ms_mean"),
            "zenith_k_ms_km2": sol.get("zenith_k_ms_km2"),
            "sv_grad_e_mps_per_km": sol.get("sv_grad_e_mps_per_km"),
            "sv_grad_n_mps_per_km": sol.get("sv_grad_n_mps_per_km"),
            "abic": sol.get("abic"),
            "abic_lambda": sol.get("abic_lambda"),
            "sv_grad_e": self.sv_grad_e,
            "sv_grad_n": self.sv_grad_n,
            "solver_type": "abic_grad",
            "true_beacons": [
                {"id": j, "x": float(self.beacons[j, 0]),
                 "y": float(self.beacons[j, 1]), "z": float(self.beacons[j, 2])}
                for j in range(self.n_beacons)
            ],
            "array_centre": sol.get("array_centre"),
            "observations_count": n_pings_actual,
            "trajectory": traj,
            "valid": True,
        }
        if sys_err_comparison is not None:
            result["system_error_comparison"] = sys_err_comparison
        _sub(1.0)
        return result

    # ==================================================================
    #  Sweep all candidates
    # ==================================================================
    def evaluate_all(
        self,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all (mode, size) candidates.

        Parameters
        ----------
        progress_callback : callable, optional
            Called as ``callback(current, total, result)`` for progress tracking.

        Returns
        -------
        results : list[dict]
            All valid (and invalid) candidate evaluation results.
        """
        if self.size_ratios:
            sizes = np.array([r * self._depth_mag for r in self.size_ratios], dtype=np.float64)
        else:
            sizes = np.arange(self.min_size, self.max_size + self.size_step / 2, self.size_step)
        sizes = sizes[sizes > 0]

        total = len(self.modes) * len(sizes)
        self.results = []
        idx = 0

        print(f"[CALIB-OPT] Evaluating {total} candidates "
              f"({len(self.modes)} modes × {len(sizes)} sizes) ...")

        for mode in self.modes:
            for size in sizes:
                idx += 1

                def _cand_cb(frac, _idx=idx):
                    if progress_callback:
                        progress_callback(_idx - 1 + frac, total, None)

                r = self.evaluate_candidate(mode, float(size), progress_callback=_cand_cb)

                status = "OK" if r.get("valid") else "BAD"
                rmse_str = f"RMSE={r.get('rmse_position_m', 'N/A')}"
                len_str = f"L={r.get('path_length_m', 'N/A')}"
                gdop_str = f"GDOP={r.get('gdop', 'N/A')}"
                print(
                    f"  [{idx:3d}/{total}] {status} {mode:15s} "
                    f"R={float(size):7.1f} m | "
                    f"{len_str:>12s} | {rmse_str:>12s} | {gdop_str}"
                )

                self.results.append(r)

                if progress_callback:
                    progress_callback(idx, total, r)

        return self.results

    # ==================================================================
    #  Select top-3 plans
    # ==================================================================
    def select_top3(
        self, results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Select three recommended survey plans from evaluation results.

        - **Plan A** — 极致精度 : minimum RMSE position error.
        - **Plan B** — 经济最短 : shortest path among candidates whose
          RMSE ≤ median_RMSE of all valid candidates (precision floor).
        - **Plan C** — 综合均衡 : minimises normalized cost
          J = 0.5×RMSE_norm + 0.5×Length_norm

        Parameters
        ----------
        results : list[dict], optional
            If None, uses ``self.results`` (must have called evaluate_all() first).

        Returns
        -------
        dict with keys:
            plan_a, plan_b, plan_c, all_results, precision_floor_rmse
        """
        if results is None:
            results = self.results

        valid = [r for r in results if r.get("valid") and not np.isnan(r.get("rmse_position_m", np.nan))]
        if not valid:
            return {
                "plan_a": None, "plan_b": None, "plan_c": None,
                "all_results": results,
                "precision_floor_rmse": None,
                "message": "No valid candidates found.",
            }

        # ----------------------------------------------------------------
        #  Plan A — minimum RMSE
        # ----------------------------------------------------------------
        plan_a = min(valid, key=lambda r: (r["rmse_position_m"], r["path_length_m"]))

        # ----------------------------------------------------------------
        #  Plan B — shortest path satisfying RMSE ≤ median RMSE
        # ----------------------------------------------------------------
        rmses = np.array([r["rmse_position_m"] for r in valid])
        median_rmse = float(np.median(rmses))

        feasible_b = [r for r in valid if r["rmse_position_m"] <= median_rmse]
        if feasible_b:
            plan_b = min(feasible_b, key=lambda r: r["path_length_m"])
        else:
            # Fallback: just take the shortest overall
            plan_b = min(valid, key=lambda r: r["path_length_m"])

        # ----------------------------------------------------------------
        #  Plan C — minimises J = 0.5 * norm(RMSE) + 0.5 * norm(Length)
        # ----------------------------------------------------------------
        lengths = np.array([r["path_length_m"] for r in valid])
        gdops = np.array([r["gdop"] for r in valid])

        len_min, len_max = lengths.min(), lengths.max()
        rmse_min, rmse_max = rmses.min(), rmses.max()

        len_range = len_max - len_min if len_max > len_min else 1.0
        rmse_range = rmse_max - rmse_min if rmse_max > rmse_min else 1.0

        J = np.array([
            0.5 * ((r["rmse_position_m"] - rmse_min) / rmse_range)
            + 0.5 * ((r["path_length_m"] - len_min) / len_range)
            for r in valid
        ])
        plan_c = valid[int(np.argmin(J))]

        # ----------------------------------------------------------------
        #  Build response
        # ----------------------------------------------------------------
        plan_a_label = "方案 A (水平误差最小)"
        plan_b_label = f"方案 B (经济最短, 水平 RMSE ≤ median {median_rmse:.2f} m)"
        plan_c_label = "方案 C (综合均衡)"

        return {
            "plan_a": plan_a,
            "plan_a_label": plan_a_label,
            "plan_b": plan_b,
            "plan_b_label": plan_b_label,
            "plan_c": plan_c,
            "plan_c_label": plan_c_label,
            "all_results": results,
            "precision_floor_rmse": round(median_rmse, 4),
            "n_valid": len(valid),
            "n_total": len(results),
        }

    # ==================================================================
    #  Convenience: run full pipeline → evaluate + select
    # ==================================================================
    def run(self) -> Dict[str, Any]:
        """Run ``evaluate_all()`` → ``select_top3()`` and return combined result."""
        results = self.evaluate_all()
        selection = self.select_top3(results)
        return selection


# =====================================================================
#  Convenience function for direct API integration
# =====================================================================

def run_calibration_optimization(
    array_points: List[List[float]],
    ssp_path: str = "default",
    lever_arm: Optional[List[float]] = None,
    gnss_noise_std: float = 0.05,
    acoustic_noise_std: float = 0.05,
    min_size_ratio: float = 0.5,
    max_size_ratio: float = 2.0,
    size_step: float = 200.0,
    n_pings: int = 360,
    use_ray_tracing: bool = True,
    solver_type: str = "9param",
    inject_system_errors: bool = False,
    system_clock_bias_s: float = 0.0,
    clock_drift_ss: float = 0.0,
    time_lag_s: float = 0.0,
    ntd_k: float = 0.0,
    lever_arm_error: Optional[List[float]] = None,
    sv_grad_e: float = 0.0,
    sv_grad_n: float = 4.0e-5,
    modes: Optional[List[str]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    One-call wrapper: run the full calibration trajectory optimisation pipeline.

    Parameters
    ----------
    array_points : list[list[float]]
        Seafloor beacon positions [[x, y, z], ...].
    ssp_path : str
        Path to SSP CSV file, or "default" to use built-in template.
    lever_arm : list[float], optional
    gnss_noise_std : float
    acoustic_noise_std : float
    min_size_ratio, max_size_ratio, size_step : float
        Sweep parameters.
    n_pings : int
        Number of pings per trajectory.
    use_ray_tracing : bool
    solver_type : str
        Ignored. The sweep always uses the 9-parameter TOA solver.
    inject_system_errors : bool
    sv_grad_e, sv_grad_n : float
        Horizontal sound-speed gradient in (m/s)/m, forward model only.
    progress_callback : callable, optional

    Returns
    -------
    dict
        { plan_a, plan_b, plan_c, all_results, precision_floor_rmse, ... }
    """
    import os
    from munk_ssp import generate_munk_ssp_table

    # --- Load SSP ---
    ssp_df = None
    if use_ray_tracing:
        if ssp_path == "default":
            # Generate idealised Munk SSP on-the-fly
            # Determine max depth from beacon array
            max_z = 3000.0
            if array_points:
                depths = [abs(p[2]) for p in array_points if len(p) >= 3]
                if depths:
                    max_z = max(depths)
            try:
                ssp_df = generate_munk_ssp_table(H=max_z + 500.0, dz=10.0)
                print(f"[CALIB-OPT] Generated Munk SSP: {len(ssp_df)} layers")
            except Exception:
                pass
        elif os.path.isfile(ssp_path):
            try:
                ssp_df = load_ssp_from_csv(ssp_path)
            except Exception:
                pass

    opt = CalibrationTrajectoryOptimizer(
        array_points=array_points,
        ssp_df=ssp_df,
        lever_arm=lever_arm,
        gnss_noise_std=gnss_noise_std,
        acoustic_noise_std=acoustic_noise_std,
        min_size_ratio=min_size_ratio,
        max_size_ratio=max_size_ratio,
        size_step=size_step,
        n_pings=n_pings,
        use_ray_tracing=use_ray_tracing,
        solver_type=solver_type,
        inject_system_errors=inject_system_errors,
        system_clock_bias_s=system_clock_bias_s,
        clock_drift_ss=clock_drift_ss,
        time_lag_s=time_lag_s,
        ntd_k=ntd_k,
        lever_arm_error=lever_arm_error,
        sv_grad_e=sv_grad_e,
        sv_grad_n=sv_grad_n,
        modes=modes,
    )

    return opt.run()


# =====================================================================
#  Smoke test
# =====================================================================
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Sample beacon array — 3 beacons at ~3000 m depth
    test_beacons = [
        [0.0,    0.0,    -3000.0],
        [500.0,  200.0,  -3005.0],
        [0.0,    800.0,  -3010.0],
    ]

    print("=" * 70)
    print("  Calibration Trajectory Optimizer — Smoke Test")
    print("=" * 70)

    import time
    t0 = time.time()

    opt = CalibrationTrajectoryOptimizer(
        array_points=test_beacons,
        ssp_df=None,                    # skip ray tracing for speed
        lever_arm=[0.0, 0.0, -30.0],
        gnss_noise_std=0.05,
        acoustic_noise_std=0.05,
        min_size_ratio=0.5,
        max_size_ratio=1.5,
        size_step=1000.0,              # coarse step for demo
        n_pings=180,                    # fewer pings for demo
        use_ray_tracing=False,
    )

    result = opt.run()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.1f} seconds.")
    print(f"  Total candidates: {result['n_total']}, valid: {result['n_valid']}")
    print(f"  Precision floor (median RMSE): {result['precision_floor_rmse']} m")
    print(f"{'=' * 70}")

    for key in ["plan_a", "plan_b", "plan_c"]:
        plan = result[key]
        label = result.get(f"{key}_label", key)
        if plan is None:
            print(f"\n  {label}:  无可行方案")
        else:
            print(
                f"\n  {label}: "
                f"{plan['type']:15s} | R={plan['size_m']:7.1f} m | "
                f"L={plan['path_length_m']:10.1f} m | "
                f"水平RMSE={plan['rmse_position_m']:8.2f} m"
            )
            for j, err in enumerate(plan.get("position_errors_m", [])):
                print(f"      beacon {j}: 3D error = {err:.3f} m")