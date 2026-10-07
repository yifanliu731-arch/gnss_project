"""
Simulation Data Generator for SeaArrayCalib  v2.0.

Generates realistic synthetic survey datasets for algorithm validation.

Supports:
  - Multi-beacon arrays  (array of [x, y, z] seafloor positions)
  - GNSS-antenna-to-transducer lever-arm compensation
  - GNSS+INS position noise (configurable Gaussian std per axis)
  - Acoustic ray tracing through a constant-gradient SSP profile
    (Snell's-law layer integrator / shooting method)

Public API
----------
- generate_simulation_data(...)
    → dict with true beacon positions and noisy observation dataset.
- load_ssp_from_csv(csv_path)
    → pandas DataFrame formatted for RayTracer consumption.

Trajectory modes
----------------
  - "circle"       : pure circular survey at constant angular velocity
  - "circle_cross" : circle followed by two perpendicular diameter passes
  - "square"       : perimeter of a square with half-side R
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Literal, List, Tuple

from geo import apply_lever_arm_batch, heading_from_track


# ---------------------------------------------------------------------------
#  Physical constants & defaults
# ---------------------------------------------------------------------------

# Fallback only if SSP is missing; otherwise C_EQ comes from the profile.
DEFAULT_SOUND_SPEED = 1496.32

# Wave noise amplitude (m) added to ship Z coordinate
WAVE_NOISE_AMPLITUDE = 0.3

# Time between consecutive pings in seconds (≈ 1 Hz acoustic rate)
PING_INTERVAL = 1.0

# Default lever arm: transducer directly below GNSS antenna by 30 m
#   [dx, dy, dz] in body-frame (m).  Z-axis positive upward in NED convention,
#   so -30 means "30 m downward" (i.e. deeper than the antenna).
DEFAULT_LEVER_ARM = np.array([0.0, 0.0, -30.0])

# Default GNSS position noise std per axis (m) — ≈ 5 cm for RTK/PPK
DEFAULT_GNSS_NOISE_STD = 0.05


def apply_horizontal_sv_gradient(
    delays: np.ndarray,
    transducer: np.ndarray,
    beacons: np.ndarray,
    slants: np.ndarray,
    c_eq: float,
    grad_e: float,
    grad_n: float,
) -> np.ndarray:
    """一阶走时改正：水平声速梯度只进前向观测。

    声速场写成 C(z) + g_E (E − E0) + g_N (N − N0)，原点取阵列水平中心。
    沿直线路径，平均声速扰动等于中点处的值，因此

        δt = −δC_mid / C_EQ² · L

    反演仍用一维剖面并估计天顶系数 k，不估计 g_E、g_N。
    g 的单位是 (m/s)/m。
    """
    if abs(grad_e) < 1e-15 and abs(grad_n) < 1e-15:
        return delays
    c2 = float(c_eq) ** 2
    if c2 <= 0.0:
        return delays
    origin_e = float(np.mean(beacons[:, 0]))
    origin_n = float(np.mean(beacons[:, 1]))
    out = np.array(delays, dtype=np.float64, copy=True)
    for j in range(beacons.shape[0]):
        mid_e = 0.5 * (transducer[:, 0] + beacons[j, 0]) - origin_e
        mid_n = 0.5 * (transducer[:, 1] + beacons[j, 1]) - origin_n
        d_c = grad_e * mid_e + grad_n * mid_n
        out[:, j] = out[:, j] - d_c / c2 * slants[:, j]
    return out


def _travel_times(transducer: np.ndarray, beacons: np.ndarray, ray_tracer, sound_speed: float):
    """Travel time and geometric slant for every ping × beacon."""
    n_pings = transducer.shape[0]
    n_beacons = beacons.shape[0]
    delays = np.zeros((n_pings, n_beacons))
    slants = np.zeros((n_pings, n_beacons))
    batch = getattr(ray_tracer, "calculate_travel_time_batch", None) if ray_tracer is not None else None
    for j in range(n_beacons):
        d_vec = transducer - beacons[j]
        s_range = np.sqrt(np.sum(d_vec ** 2, axis=1))
        slants[:, j] = s_range
        if callable(batch):
            tau = np.asarray(batch(transducer, beacons[j]), dtype=np.float64)
            bad = ~np.isfinite(tau) | (tau <= 0)
            tau[bad] = s_range[bad] / sound_speed
            delays[:, j] = tau
        else:
            delays[:, j] = s_range / sound_speed
    return delays, slants


# ---------------------------------------------------------------------------
#  SSP loader
# ---------------------------------------------------------------------------

def load_ssp_from_csv(csv_path) -> pd.DataFrame:
    """Load SSP CSV (path or file-like) and normalise columns for RayTracer."""
    from ray_tracing import normalize_ssp_columns
    df = pd.read_csv(csv_path)
    return normalize_ssp_columns(df)[["Depth (m)", "SV (m/s)"]].dropna()


# ---------------------------------------------------------------------------
#  Physical simulation core
# ---------------------------------------------------------------------------

def generate_simulation_data(
    # --- Beacon array (new in v2.0) ---
    array_points: Optional[List[List[float]]] = None,
    depth_H: float = 3000.0,

    # --- Trajectory ---
    radius_R: float = 4000.0,
    mode: Literal["circle", "circle_cross", "square", "asterisk"] = "circle",
    n_points: int = 360,
    angular_velocity: Optional[float] = None,

    # --- Lever arm (new in v2.0) ---
    lever_arm: Optional[List[float]] = None,
    #   [dx, dy, dz]  (m)  body-frame offset from GNSS antenna to transducer.
    #   Default: [0, 0, -30] — transducer 30 m directly below the antenna.

    # --- GNSS position noise (new in v2.0) ---
    gnss_noise_std: float = DEFAULT_GNSS_NOISE_STD,
    #   Per-axis Gaussian std (m).  Applied independently to X, Y, Z.

    # --- Ray tracing (new in v2.0) ---
    ssp_df: Optional[pd.DataFrame] = None,
    #   Pandas DataFrame with columns 'Depth (m)' and 'SV (m/s)'.
    #   If provided and use_ray_tracing=True, travel times are computed
    #   via Snell's-law integration through the profile.
    use_ray_tracing: bool = True,

    # --- Acoustic delay noise (applied to all modes) ---
    noise_std: float = 0.00005,
    #   Std of acoustic delay measurement noise (seconds).  Default 0.05 ms.
    #   Applied as independent Gaussian noise to each ping's true delay.
    sound_speed: float = DEFAULT_SOUND_SPEED,
    #   Constant sound speed (m/s).  Only relevant when use_ray_tracing=False.

    # --- Systematic errors for 7‑param / 9‑param solver validation ---
    system_clock_bias_s: float = 0.0,
    #   Constant clock offset added to ALL delays (seconds).
    clock_drift_ss: float = 0.0,
    #   Linear clock drift (s/s).  Delay_i += drift × t_i.
    time_lag_s: float = 0.0,
    #   Hardware sync latency (s).  The ship position used for ray tracing
    #   is shifted forward/backward by this amount, simulating misalignment
    #   between GNSS timestamp and acoustic transmit time.
    ntd_k: float = 0.0,
    #   Zenith distance‑dependent delay coefficient (dimensionless).
    #   Additional delay = k × (horizontal_dist² / 1e6)  [seconds].
    lever_arm_error: Optional[List[float]] = None,
    # 水平声速梯度 (m/s)/m。只加在前向走时上，反演模型看不到。
    sv_grad_e: float = 0.0,
    sv_grad_n: float = 0.0,
    rng_seed: Optional[int] = None,
    #   [dx_err, dy_err, dz_err]  (m)  Residual lever-arm error on top
    #   of the nominal lever_arm.  This simulates unknown installation
    #   offsets.  Only meaningful for 9‑param solver validation.

) -> Dict[str, Any]:
    """
    Generate a synthetic survey dataset with realistic physical modelling.

    Physical model
    --------------
    1. **Trajectory generation** — ship GNSS antenna follows a prescribed
       geometric path (circle / circle+cross / square) at constant speed.

    2. **Lever-arm compensation** — the transducer is offset from the GNSS
       antenna by a body-frame vector ``lever_arm``::

           P_transducer = P_gnss + Rz(yaw)Ry(pitch)Rx(roll) · lever_RFU

    3. **GNSS position noise** — independent Gaussian noise is added to each
       axis of the antenna position::

           P_gnss_noisy ~ N(P_gnss_true, σ²)    σ = gnss_noise_std  [m]

    4. **Acoustic travel time** — two modes:

       a) **Ray tracing** (default):  travel time from transducer to each
          beacon is computed by integrating Snell's law through the SSP
          layers.  This accounts for ray bending caused by the vertical
          sound-speed gradient.

          The integration uses the constant-gradient (linear sound-speed)
          assumption within each layer, which is exact for an SSP composed
          of piecewise-linear segments.

       b) **Straight-line / constant speed** (fallback):::

            t = ‖P_transducer − P_beacon‖ / c


    Parameters
    ----------
    array_points : list of [x,y,z], optional
        Seafloor beacon positions.  Each entry is [easting, northing, depth]
        in metres (Z positive up → depth is negative).
        Default: single beacon at ``(0, 0, -depth_H)``.

    depth_H : float
        Nominal beacon depth (m, positive magnitude).  Only used when
        ``array_points`` is None.  Default 3000.

    radius_R : float
        Characteristic trajectory radius / half-side (m).  Default 4000.

    mode : str
        ``circle``, ``circle_cross``, or ``square``.

    n_points : int
        Number of discrete pings.  Default 360.

    angular_velocity : float, optional
        Ship angular velocity (rad/s).  Auto-computed if None.

    lever_arm : list of float, optional
        ``[dx, dy, dz]`` body-frame offset (m).  Default ``[0, 0, -30]``.

    gnss_noise_std : float
        Per-axis GNSS position noise std (m).  Default 0.05 (5 cm).

    ssp_df : pd.DataFrame, optional
        SSP with columns ``Depth (m)``, ``SV (m/s)``.  If None and
        ``use_ray_tracing`` is True, the function falls back to
        straight-line / constant-speed mode.

    use_ray_tracing : bool
        Enable Snell ray integration through the SSP.  Default True.

    noise_std : float
        Legacy acoustic delay noise std (seconds).  Only used when
        ``use_ray_tracing`` is False.  Default 0.05.

    sound_speed : float
        Fallback constant sound speed (m/s) when no SSP is given.
        If an SSP is provided, C_EQ is the harmonic mean of that profile.

    Returns
    -------
    result : dict
        {
            "true_beacons": [
                {"id": 0, "x": ..., "y": ..., "z": ...},
                ...
            ],
            "true_x", "true_y", "true_z":  (kept for backward compat),
            "mode", "radius",
            "lever_arm": [dx, dy, dz],
            "gnss_noise_std_m": float,
            "use_ray_tracing": bool,
            "observations": [
                {
                    "time_s": float,
                    "gnss_x", "gnss_y", "gnss_z":   (noisy GNSS antenna),
                    "gnss_true_x", ...,             (true GNSS antenna),
                    "transducer_x", "transducer_y", "transducer_z":  (computed),
                    "slant_range_m": float,          (geometric, transducer→beacon[0])
                    "delay_per_beacon_s": [t0, t1, ...],  (ray-traced or straight-line)
                    "noisy_delay_per_beacon_s": [...],    (delay + acoustic noise)
                },
                ...
            ],
            "trajectory": [{"x", "y", "z"}: transducer positions ...],
            "gnss_trajectory": [{"x", "y", "z"}: noisy GNSS positions ...],
            "residuals": [...],
        }
    """
    # ------------------------------------------------------------------
    #  0.  Sanitise inputs
    # ------------------------------------------------------------------
    if lever_arm is None:
        lever_arm_arr = DEFAULT_LEVER_ARM.copy()
    else:
        lever_arm_arr = np.array(lever_arm, dtype=np.float64)

    if array_points is None:
        array_points = [[0.0, 0.0, -abs(depth_H)]]

    beacons = np.array(array_points, dtype=np.float64)  # shape (M, 3)
    n_beacons = beacons.shape[0]
    print(f"[SIM-SELFCHECK] array_points shape: {beacons.shape}, "
          f"any NaN: {np.any(np.isnan(beacons))}, any Inf: {np.any(np.isinf(beacons))}")

    # ------------------------------------------------------------------
    #  1.  Initialise ray tracer (if requested)
    #
    #  唯一声线引擎：Snell 打靶 RayTracer（纯 Python，在线求解）。
    #  不再依赖静态 SQLite 查表（FastRayTracer）。
    # ------------------------------------------------------------------
    ray_tracer = None
    if ssp_df is not None:
        try:
            from ray_tracing import RayTracer, ssp_equivalent_speed
            if use_ray_tracing:
                ray_tracer = RayTracer(ssp_df)
                sound_speed = float(ray_tracer.c_eq_measured)
                print(f"[SIM] RayTracer (Snell 打靶) initialised with {len(ssp_df)} SSP layers, "
                      f"C_EQ={sound_speed:.2f} m/s.")
            else:
                sound_speed = float(ssp_equivalent_speed(ssp_df))
                print(f"[SIM] Ray tracing off; using SSP C_EQ={sound_speed:.2f} m/s.")
        except Exception as exc:
            print(f"[SIM] WARN — RayTracer init failed: {exc}. "
                  "Falling back to straight-line / constant-speed.")
            use_ray_tracing = False
            ray_tracer = None
            try:
                from ray_tracing import ssp_equivalent_speed
                sound_speed = float(ssp_equivalent_speed(ssp_df))
            except Exception:
                pass

    # ------------------------------------------------------------------
    #  2.  Generate ship GNSS-antenna trajectory (true positions)
    # ------------------------------------------------------------------
    generator_map = {
        "circle":       _generate_circle,
        "circle_cross": _generate_circle_cross,
        "square":       _generate_square,
        "asterisk":     _generate_asterisk,
    }
    generator = generator_map.get(mode)
    if generator is None:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: {list(generator_map.keys())}"
        )

    traj_xy, time_arr = generator(radius_R, n_points, angular_velocity)
    n_points = len(time_arr)
    print(f"[SIM-SELFCHECK] trajectory XY shape: {traj_xy.shape}, "
          f"any NaN: {np.any(np.isnan(traj_xy))}, any Inf: {np.any(np.isinf(traj_xy))}")

    # Small wave-induced Z noise around 0 for the antenna
    rng = np.random.default_rng(rng_seed)
    gnss_z_true = rng.normal(0.0, WAVE_NOISE_AMPLITUDE, size=n_points)
    gnss_true = np.column_stack([
        traj_xy[:, 0],           # X — easting
        traj_xy[:, 1],           # Y — northing
        gnss_z_true,             # Z — depth (≈ 0 at sea surface)
    ])  # shape (n_points, 3)

    # ------------------------------------------------------------------
    #  3.  Add GNSS position noise
    #
    #  Each axis gets independent Gaussian noise N(0, σ²_gnss).
    #  In practice GNSS+INS errors are correlated in time, but for
    #  simulation purposes white noise is a reasonable first-order model.
    # ------------------------------------------------------------------
    gnss_noise = rng.normal(0.0, gnss_noise_std, size=(n_points, 3))
    gnss_noisy = gnss_true + gnss_noise

    # ------------------------------------------------------------------
    #  4.  Lever-arm compensation:  GNSS antenna → transducer (RFU, yaw from track)
    # ------------------------------------------------------------------
    yaw = heading_from_track(gnss_true[:, 0], gnss_true[:, 1])
    zeros = np.zeros(n_points)
    transducer_true = apply_lever_arm_batch(gnss_true, lever_arm_arr, zeros, zeros, yaw)
    transducer_noisy = apply_lever_arm_batch(gnss_noisy, lever_arm_arr, zeros, zeros, yaw)

    # ------------------------------------------------------------------
    #  5.  Compute acoustic travel times
    # ------------------------------------------------------------------
    ideal_delays, slant_ranges = _travel_times(
        transducer_true, beacons,
        ray_tracer if use_ray_tracing else None,
        sound_speed,
    )

    # ------------------------------------------------------------------
    #  5b. Retrace if the transducer position itself is wrong, then add
    #  the horizontal sound-speed gradient on that path. Gradient is a
    #  forward-only mismatch: the 9-parameter solver never receives g.
    # ------------------------------------------------------------------
    if abs(system_clock_bias_s) > 0 or abs(clock_drift_ss) > 0 or abs(ntd_k) > 0 or abs(time_lag_s) > 0 or lever_arm_error is not None:
        print(f"[SIM] Injecting systematic errors: "
              f"C={system_clock_bias_s*1000:.3f}ms, D={clock_drift_ss:.3e}s/s, "
              f"lag={time_lag_s*1000:.3f}ms, k={ntd_k:.6f}")

    lever_arm_error_arr = np.zeros(3) if lever_arm_error is None else np.asarray(lever_arm_error, dtype=np.float64)
    if np.any(np.abs(lever_arm_error_arr) > 1e-9):
        print(f"[SIM] Lever‑arm error injected: {lever_arm_error_arr.tolist()}")

    need_retrace = abs(time_lag_s) > 1e-9 or np.any(np.abs(lever_arm_error_arr) > 1e-9)
    if need_retrace:
        trans_delay = apply_lever_arm_batch(
            gnss_true, lever_arm_arr + lever_arm_error_arr, zeros, zeros, yaw
        )
        if abs(time_lag_s) > 1e-9:
            dt = float(np.mean(np.diff(time_arr))) if n_points > 1 else 1.0
            trans_delay = trans_delay + np.gradient(trans_delay, axis=0) / max(dt, 1e-9) * time_lag_s
        transducer_true_lagged = trans_delay
        ideal_delays, slant_ranges = _travel_times(
            trans_delay, beacons,
            ray_tracer if use_ray_tracing else None,
            sound_speed,
        )
    else:
        transducer_true_lagged = transducer_true

    trace_pos = transducer_true_lagged if need_retrace else transducer_true
    if abs(sv_grad_e) > 1e-15 or abs(sv_grad_n) > 1e-15:
        print(f"[SIM] Horizontal SV gradient: g_E={sv_grad_e:.3e} (m/s)/m, "
              f"g_N={sv_grad_n:.3e} (m/s)/m")
        ideal_delays = apply_horizontal_sv_gradient(
            ideal_delays, trace_pos, beacons, slant_ranges, sound_speed,
            sv_grad_e, sv_grad_n,
        )

    noisy_delays = ideal_delays + rng.normal(0.0, noise_std, size=ideal_delays.shape)

    # Clock bias, drift and zenith k are added on top of the gradient delay.
    time_arr_sec = time_arr  # seconds from start
    _pos_for_ntd = trace_pos
    for j in range(n_beacons):
        bpos = beacons[j]
        for i in range(n_points):
            dx = _pos_for_ntd[i, 0] - bpos[0]
            dy = _pos_for_ntd[i, 1] - bpos[1]
            dist_sq = dx * dx + dy * dy
            ntd_correction = ntd_k * (dist_sq / 1000000.0)

            noisy_delays[i, j] += (
                system_clock_bias_s
                + clock_drift_ss * time_arr_sec[i]
                + ntd_correction
            )

    # ------------------------------------------------------------------
    #  6.  Build observation records  (per-ping)
    # ------------------------------------------------------------------
    observations: List[Dict[str, Any]] = []
    trajectory: List[Dict[str, float]] = []
    gnss_trajectory: List[Dict[str, float]] = []
    residuals: List[Dict[str, Any]] = []

    for i in range(n_points):
        obs = {
            "time_s": round(float(time_arr[i]), 6),

            # GNSS antenna — noisy (what the navigation system reports)
            "gnss_x": round(float(gnss_noisy[i, 0]), 6),
            "gnss_y": round(float(gnss_noisy[i, 1]), 6),
            "gnss_z": round(float(gnss_noisy[i, 2]), 6),

            # GNSS antenna — true (ground truth for validation)
            "gnss_true_x": round(float(gnss_true[i, 0]), 6),
            "gnss_true_y": round(float(gnss_true[i, 1]), 6),
            "gnss_true_z": round(float(gnss_true[i, 2]), 6),

            # Transducer — computed from noisy GNSS + lever arm
            "transducer_x": round(float(transducer_noisy[i, 0]), 6),
            "transducer_y": round(float(transducer_noisy[i, 1]), 6),
            "transducer_z": round(float(transducer_noisy[i, 2]), 6),

            # True transducer
            "transducer_true_x": round(float(transducer_true[i, 0]), 6),
            "transducer_true_y": round(float(transducer_true[i, 1]), 6),
            "transducer_true_z": round(float(transducer_true[i, 2]), 6),

            # Geometric slant range to first beacon (backward compat)
            "slant_range_m": round(float(slant_ranges[i, 0]), 6),

            # Ray-traced (or straight-line) delays per beacon
            "delay_per_beacon_s": [
                round(float(ideal_delays[i, j]), 10)
                for j in range(n_beacons)
            ],
            "noisy_delay_per_beacon_s": [
                round(float(noisy_delays[i, j]), 10)
                for j in range(n_beacons)
            ],
        }
        # Backward-compat shortcut fields (first beacon)
        if n_beacons >= 1:
            obs["ideal_delay_s"] = round(float(ideal_delays[i, 0]), 10)
            obs["noisy_delay_s"] = round(float(noisy_delays[i, 0]), 10)

        observations.append(obs)

        # Trajectory (transducer true positions — for front-end plotting)
        trajectory.append({
            "x": round(float(transducer_true[i, 0]), 6),
            "y": round(float(transducer_true[i, 1]), 6),
            "z": round(float(transducer_true[i, 2]), 6),
        })
        gnss_trajectory.append({
            "x": round(float(gnss_noisy[i, 0]), 6),
            "y": round(float(gnss_noisy[i, 1]), 6),
            "z": round(float(gnss_noisy[i, 2]), 6),
        })
        residuals.append({
            "epoch": i + 1,
            "residual": round(
                float(noisy_delays[i, 0] - ideal_delays[i, 0]), 10
            ),
        })

    # ------------------------------------------------------------------
    #  7.  Build response
    # ------------------------------------------------------------------
    true_beacons_list = [
        {"id": j, "x": float(beacons[j, 0]),
         "y": float(beacons[j, 1]), "z": float(beacons[j, 2])}
        for j in range(n_beacons)
    ]

    result: Dict[str, Any] = {
        "true_beacons": true_beacons_list,
        "n_beacons": n_beacons,
        # Backward-compat: first beacon
        "true_beacon": true_beacons_list[0] if n_beacons > 0 else None,
        "true_x": float(beacons[0, 0]) if n_beacons > 0 else 0.0,
        "true_y": float(beacons[0, 1]) if n_beacons > 0 else 0.0,
        "true_z": float(beacons[0, 2]) if n_beacons > 0 else 0.0,

        "mode": mode,
        "radius": radius_R,
        "lever_arm": lever_arm_arr.tolist(),
        "gnss_noise_std_m": gnss_noise_std,
        "use_ray_tracing": use_ray_tracing,
        "noise_std_s": noise_std,
        "sound_speed_mps": sound_speed,

        # Injected systematic errors (ground truth for solver validation)
        "injected_system_errors": {
            "clock_bias_s": system_clock_bias_s,
            "clock_drift_ss": clock_drift_ss,
            "time_lag_s": time_lag_s,
            "ntd_k": ntd_k,
            "lever_arm_error": lever_arm_error_arr.tolist() if lever_arm_error is not None else [0.0, 0.0, 0.0],
            "sv_grad_e": sv_grad_e,
            "sv_grad_n": sv_grad_n,
        },

        "observations": observations,
        "trajectory": trajectory,
        "gnss_trajectory": gnss_trajectory,
        "residuals": residuals,
    }
    return result


# =====================================================================
# Trajectory generators — return ( (N,2) array of XY, time array )
# =====================================================================

def _generate_circle(
    R: float, n: int, omega: Optional[float]
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Generate a circular trajectory at constant angular velocity.

    Parameters
    ----------
    R : float
        Circle radius (m).
    n : int
        Number of points.
    omega : float, optional
        Angular velocity (rad/s).  If None, auto-calc so that the circle
        completes in n × PING_INTERVAL seconds → ω = 2π / (n·Δt).

    Returns
    -------
    xy : np.ndarray shape (n, 2)
    time : np.ndarray shape (n,)
    """
    dt = PING_INTERVAL
    if omega is None:
        # Full circle exactly in (n-1)·dt seconds
        period = (n - 1) * dt
        omega = 2.0 * np.pi / max(period, 1e-6)

    time = np.arange(n) * dt
    theta = omega * time
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return np.column_stack([x, y]), time


def _generate_circle_cross(
    R: float, n: int, omega: Optional[float]
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Generate a circle + perpendicular diameter passes.

    n_circle = 2/3·n points on the circle,
    n_line   = 1/6·n on N-S line (x=0),
    n_ew     = 1/6·n on E-W line (y=0).

    The ship first completes the circle, then traverses the N-S diameter
    (south → north), then the E-W diameter (west → east).
    """
    n_circle = max(2, int(n * 2.0 / 3.0))
    n_line_each = max(2, (n - n_circle) // 2)

    # Circle portion
    xy_circle, t_circle = _generate_circle(R, n_circle, omega)
    t_circle_end = t_circle[-1] + PING_INTERVAL

    # N-S line: x = 0, y from -R → +R
    t_ns = t_circle_end + np.arange(n_line_each) * PING_INTERVAL
    y_ns = np.linspace(-R, R, n_line_each)
    x_ns = np.zeros_like(y_ns)

    t_ns_end = t_ns[-1] + PING_INTERVAL

    # E-W line: y = 0, x from -R → +R
    t_ew = t_ns_end + np.arange(n_line_each) * PING_INTERVAL
    x_ew = np.linspace(-R, R, n_line_each)
    y_ew = np.zeros_like(x_ew)

    xy = np.vstack([
        xy_circle,
        np.column_stack([x_ns, y_ns]),
        np.column_stack([x_ew, y_ew]),
    ])
    time = np.concatenate([t_circle, t_ns, t_ew])

    # Trim to exactly n
    if xy.shape[0] > n:
        xy, time = xy[:n], time[:n]

    return xy, time


def _generate_square(
    R: float, n: int, _omega: Optional[float]
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Generate points along the perimeter of a square with half-side R.

    Ship traverses: bottom edge (y=-R, L→R), right edge (x=R, -R→R),
    top edge (y=R, R→L), left edge (x=-R, R→-R).
    """
    pts_per_side = n // 4
    remainder = n - 4 * pts_per_side

    time = np.arange(n) * PING_INTERVAL
    all_xy: list[np.ndarray] = []

    def _append(x_arr, y_arr, n_p):
        all_xy.append(np.column_stack([x_arr, y_arr]))
        return n_p

    # Side 1: bottom
    n1 = pts_per_side + (1 if remainder > 0 else 0)
    r1 = remainder - (1 if remainder > 0 else 0)
    x1 = np.linspace(-R, R, n1, endpoint=False)
    y1 = np.full_like(x1, -R)
    _append(x1, y1, n1)

    # Side 2: right
    n2 = pts_per_side + (1 if r1 > 0 else 0)
    r2 = r1 - (1 if r1 > 0 else 0)
    y2 = np.linspace(-R, R, n2, endpoint=False)
    x2 = np.full_like(y2, R)
    _append(x2, y2, n2)

    # Side 3: top
    n3 = pts_per_side + (1 if r2 > 0 else 0)
    r3 = r2 - (1 if r2 > 0 else 0)
    x3 = np.linspace(R, -R, n3, endpoint=False)
    y3 = np.full_like(x3, R)
    _append(x3, y3, n3)

    # Side 4: left
    n4 = pts_per_side + r3
    y4 = np.linspace(R, -R, n4, endpoint=False)
    x4 = np.full_like(y4, -R)
    _append(x4, y4, n4)

    xy = np.vstack(all_xy)

    # Ensure exact n
    if xy.shape[0] > n:
        xy, time = xy[:n], time[:n]

    return xy, time


def _generate_asterisk(
    R: float, n: int, omega: Optional[float]
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Generate an Asterisk (米字形) trajectory:
    Circle + N-S + E-W + NE-SW + NW-SE passes.
    """
    n_circle = max(2, int(n * 0.4))
    n_line_each = max(1, (n - n_circle) // 4)

    # 1. Circle
    xy_circle, t_circle = _generate_circle(R, n_circle, omega)
    t_last = t_circle[-1] + PING_INTERVAL

    # 2. N-S line
    t_ns = t_last + np.arange(n_line_each) * PING_INTERVAL
    y_ns = np.linspace(-R, R, n_line_each)
    x_ns = np.zeros_like(y_ns)
    t_last = t_ns[-1] + PING_INTERVAL

    # 3. E-W line
    t_ew = t_last + np.arange(n_line_each) * PING_INTERVAL
    x_ew = np.linspace(-R, R, n_line_each)
    y_ew = np.zeros_like(x_ew)
    t_last = t_ew[-1] + PING_INTERVAL

    # 4. NE-SW diagonal
    t_ne = t_last + np.arange(n_line_each) * PING_INTERVAL
    x_ne = np.linspace(-R * 0.707, R * 0.707, n_line_each)
    y_ne = x_ne.copy()
    t_last = t_ne[-1] + PING_INTERVAL

    # 5. NW-SE diagonal
    t_nw = t_last + np.arange(n_line_each) * PING_INTERVAL
    x_nw = np.linspace(-R * 0.707, R * 0.707, n_line_each)
    y_nw = -x_nw.copy()

    xy = np.vstack([
        xy_circle,
        np.column_stack([x_ns, y_ns]),
        np.column_stack([x_ew, y_ew]),
        np.column_stack([x_ne, y_ne]),
        np.column_stack([x_nw, y_nw])
    ])
    time = np.concatenate([t_circle, t_ns, t_ew, t_ne, t_nw])

    if xy.shape[0] > n:
        xy, time = xy[:n], time[:n]

    return xy, time


# =====================================================================
# Smoke test
# =====================================================================
if __name__ == "__main__":
    for mode in ["circle", "circle_cross", "square"]:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")
        data = generate_simulation_data(
            depth_H=3000.0,
            radius_R=3000.0,
            mode=mode,          # type: ignore[arg-type]
            noise_std=0.05,
            n_points=100,       # small for smoke test
        )
        print(f"  True beacon : ({data['true_x']}, {data['true_y']}, {data['true_z']})")
        print(f"  Observations: {len(data['observations'])} pings")
        print(f"  First delay : ideal={data['observations'][0]['ideal_delay_s']:.6f}s  "
              f"noisy={data['observations'][0]['noisy_delay_s']:.6f}s")
        print(f"  Noise std (actual) : {np.std([o['noisy_delay_s'] - o['ideal_delay_s'] for o in data['observations']]):.6f}s")