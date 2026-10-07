"""实验配置：测线名、因子标签、水平基线先验、统一装载/预清洗、统一结果表。"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")
RAWDATA_DIR = os.path.join(DATA_DIR, "rawdata")
PAPER_DIR = os.path.join(DATA_DIR, "paper_tables")
MECH_DIR = os.path.join(DATA_DIR, "joint_mechanism_logic_chain")
FIG_DIR = os.path.join(DATA_DIR, "processed_picture", "paper")

REF_LAT = 18.75000000
REF_LON = 116.50000000
LEVER_ARM = np.array([1.585, 6.716, -29.050], dtype=np.float64)

# 旧设计水平距，仅对照；联合解刚体约束用站间单程时延 TAU*，不用这些数。
D52, D56, D26 = 3012.0, 3220.0, 4293.0
SIGMA_BASELINE = 8.0
# 时延伪观测 σ（秒）。与 8 m 同权，声速用深水段约 1512 m/s：8/1512 ≈ 5.29 ms。
SIGMA_TAU = SIGMA_BASELINE / 1512.0

# 站间单程时延中位数（秒）。跑 notebooks/extract_interbeacon_delay.py 后从 CSV 读入；未提取则为 nan。
INTERBEACON_MEDIAN_CSV = os.path.join(DATA_DIR, "interbeacon_delay_median.csv")
INTERBEACON_FILES = (
    "0501第一次相对测阵.txt",
    "0503第二次相对测阵.txt",
    "0503第三次相对测阵.txt",
    "0504第四次相对测阵.txt",
)
STATIC_GNSS_PATH = os.path.join(RAWDATA_DIR, "静态观测数据.txt")

DEPTHS = {2: 3150.0, 3: 3265.0, 4: 3265.0, 5: 3265.0, 6: 3180.0}
JOINT_BEACONS = (2, 5, 6)
SINGLE_BEACONS = (2, 5, 6)

# Isolated pick-error gate: |delay - 0.5*(prev+next)|. 0.3 s only caught
# steps of hundreds of ms; 50 ms still sits above smooth ranging curvature.
SPIKE_DELAY_S = 0.05

# Adjustment uses V = V_bg only (c1=c2 locked). EOF modes enter observation σ.
SIGMA_AC = 0.3
EOF_SIGMA_C1 = 2.0
EOF_N_MODES = 5
# NSCS SVD variance fractions (template_eof rebuild 2026-09-19). σ_c,k = σ_c1 √(λ_k/λ_1).
EOF_VAR_FRACTIONS = (0.8017387997231366, 0.08367211470739273, 0.0494383556903928, 0.024312129554247643, 0.016930038983076375)
TEMPLATE_EOF = os.path.join(DATA_DIR, "template_eof.csv")
TEMPLATE_SSP = os.path.join(DATA_DIR, "template_ssp.csv")
EOF_REBUILD_SUMMARY = os.path.join(DATA_DIR, "eof_nscs_rebuild_summary.json")

RESULTS_SINGLE = os.path.join(DATA_DIR, "results_single.csv")
RESULTS_JOINT = os.path.join(DATA_DIR, "results_joint.csv")
SUMMARY_SINGLE_LEGACY = os.path.join(DATA_DIR, "Calibration_Summary_Report_all.csv")
SUMMARY_JOINT_LEGACY = os.path.join(MECH_DIR, "Summary_Logic_Chain.csv")

UNIFIED_COLUMNS = [
    "traj", "beacon_id", "method", "use_baseline",
    "scale", "cross", "curvature",
    "n_sess", "k_dom", "n_ping_in", "n_ping_inlier",
    "E", "N", "Z",
    "sigma_E", "sigma_N", "sigma_Z", "WRMS",
    "B_dom_ms",
    "corr_Z_B0", "angle_Z_B0_deg", "kappa_Zclock",
    "D52_hat", "D56_hat", "D26_hat",
    "R52_geom", "R56_geom", "R26_geom",
    "R52_ssp", "R56_ssp", "R26_ssp",
    "TAU52_hat", "TAU56_hat", "TAU26_hat",
    "lock_k", "C_EQ",
]

# 规范名 + 因子。Scale 按包络：Small_* 为 Small，其余为 Large。
TRAJECTORIES = [
    {
        "name": "Big_Circle",
        "nav": "user_gnss_imu_circle_big_clean.csv",
        "ping": "user_ping_circle_big_clean.csv",
        "scale": "Large", "cross": 0, "curvature": "Circle",
    },
    {
        "name": "Small_Circle",
        "nav": "user_gnss_imu_circle_small_clean.csv",
        "ping": "user_ping_circle_small_clean.csv",
        "scale": "Small", "cross": 0, "curvature": "Circle",
    },
    {
        "name": "Square",
        "nav": "user_gnss_imu_square.csv",
        "ping": "user_ping_square.csv",
        "scale": "Large", "cross": 0, "curvature": "Square",
    },
    {
        "name": "Big_Rhombus",
        "nav": "user_gnss_imu_rhobus_big_clean.csv",
        "ping": "user_ping_rhobus_big_clean.csv",
        "scale": "Large", "cross": 0, "curvature": "Rhombus",
    },
    {
        "name": "Small_Rhombus",
        "nav": "user_gnss_imu_rhobus_small_clean.csv",
        "ping": "user_ping_rhobus_small_clean.csv",
        "scale": "Small", "cross": 0, "curvature": "Rhombus",
    },
    {
        "name": "Big_Circle_Cross",
        "nav": "user_gnss_imu_circle_big_cross_combined.csv",
        "ping": "user_ping_circle_big_cross_combined.csv",
        "scale": "Large", "cross": 1, "curvature": "Circle",
    },
    {
        "name": "Small_Circle_Cross",
        "nav": "user_gnss_imu_circle_small_cross_combined.csv",
        "ping": "user_ping_circle_small_cross_combined.csv",
        "scale": "Small", "cross": 1, "curvature": "Circle",
    },
    {
        "name": "Square_Cross",
        "nav": "user_gnss_imu_square_cross_combined.csv",
        "ping": "user_ping_square_cross_combined.csv",
        "scale": "Large", "cross": 1, "curvature": "Square",
    },
    {
        "name": "Square_No_Corners_Cross",
        "nav": "user_gnss_imu_square_no_corners_cross_clean.csv",
        "ping": "user_ping_square_no_corners_cross_clean.csv",
        "scale": "Large", "cross": 1, "curvature": "Square-cut",
    },
]

# 5×5 不进论文九测线 TRAJ_ORDER；单站/联合批处理单独并入。
EXTRA_TRAJECTORIES = [
    {
        "name": "Five_by_Five",
        "nav": "user_gnss_imu_55.csv",
        "ping": "user_ping_55.csv",
        "scale": "Large", "cross": 1, "curvature": "Grid",
    },
]

TRAJ_BY_NAME = {t["name"]: t for t in list(TRAJECTORIES) + list(EXTRA_TRAJECTORIES)}
TRAJ_ORDER = [t["name"] for t in TRAJECTORIES]
TRAJ_SHORT = {
    "Big_Circle": "Big Cir",
    "Small_Circle": "Small Cir",
    "Square": "Square",
    "Big_Rhombus": "Big Rhomb",
    "Small_Rhombus": "Small Rhomb",
    "Big_Circle_Cross": "Big Cir+X",
    "Small_Circle_Cross": "Small Cir+X",
    "Square_Cross": "Square+X",
    "Square_No_Corners_Cross": "Sq-cut+X",
    "Five_by_Five": "5x5",
}

_lat0 = np.radians(REF_LAT)
_a, _e2 = 6378137.0, 0.00669437999014
RM = _a * (1.0 - _e2) / (1.0 - _e2 * np.sin(_lat0) ** 2) ** 1.5
RN = _a / np.sqrt(1.0 - _e2 * np.sin(_lat0) ** 2)


def to_unix_seconds(values) -> np.ndarray:
    """Unix 秒。先转到 datetime64[ns]，避免 CSV 被读成 us 后 /1e9 把时间缩小 1000 倍。"""
    t = pd.to_datetime(values, errors="coerce")
    arr = np.asarray(t, dtype="datetime64[ns]")
    ns = arr.astype("datetime64[ns]").astype(np.int64)
    out = ns.astype(np.float64) / 1e9
    nat = np.isnat(arr)
    if np.any(nat):
        out = np.array(out, copy=True)
        out[nat] = np.nan
    return out


def horiz(P, Q) -> float:
    return float(np.hypot(float(P[0]) - float(Q[0]), float(P[1]) - float(Q[1])))


def posterior_baselines(pos):
    """pos: {2: (E,N,Z_or_U), 5:..., 6:...}，只用水平距。缺站则 nan。"""
    p2, p5, p6 = pos.get(2), pos.get(5), pos.get(6)
    d52 = horiz(p5, p2) if p2 is not None and p5 is not None else np.nan
    d56 = horiz(p5, p6) if p5 is not None and p6 is not None else np.nan
    d26 = horiz(p2, p6) if p2 is not None and p6 is not None else np.nan
    return d52, d56, d26


def _slant3(P, Q) -> float:
    a = np.asarray(P, dtype=np.float64).reshape(-1)[:3]
    b = np.asarray(Q, dtype=np.float64).reshape(-1)[:3]
    return float(np.linalg.norm(a - b))


def posterior_geom_ranges(pos):
    """解后三维几何斜距 ||P_i - P_j||。"""
    p2, p5, p6 = pos.get(2), pos.get(5), pos.get(6)
    r52 = _slant3(p5, p2) if p2 is not None and p5 is not None else np.nan
    r56 = _slant3(p5, p6) if p5 is not None and p6 is not None else np.nan
    r26 = _slant3(p2, p6) if p2 is not None and p6 is not None else np.nan
    return r52, r56, r26


def factor_of(traj_name: str) -> dict:
    meta = TRAJ_BY_NAME.get(traj_name, {})
    return {
        "scale": meta.get("scale", ""),
        "cross": int(meta.get("cross", 0)),
        "curvature": meta.get("curvature", ""),
    }


def eof_variance_fractions(n_modes: int = EOF_N_MODES) -> np.ndarray:
    fracs = list(EOF_VAR_FRACTIONS)
    path = EOF_REBUILD_SUMMARY
    if os.path.isfile(path):
        try:
            rec = json.loads(open(path, encoding="utf-8").read())
            ev = rec.get("explained_variance") or {}
            loaded = [float(ev[f"EOF{k}"]) for k in range(1, n_modes + 1) if f"EOF{k}" in ev]
            if loaded:
                fracs = loaded
        except (OSError, ValueError, TypeError, KeyError):
            pass
    out = np.asarray(fracs[:n_modes], dtype=np.float64)
    if out.size < n_modes:
        pad = np.full(n_modes - out.size, out[-1] * 0.5 if out.size else 1.0)
        out = np.concatenate([out, pad])
    return out


def eof_mode_amplitudes(sigma_c1: float = EOF_SIGMA_C1, n_modes: int = EOF_N_MODES) -> np.ndarray:
    """Peak-normalized mode amplitudes (m/s). Mode 1 = sigma_c1; others √(λ_k/λ_1)."""
    v = eof_variance_fractions(n_modes)
    v1 = max(float(v[0]), 1e-12)
    return float(sigma_c1) * np.sqrt(np.clip(v, 0.0, None) / v1)


def load_eof_phis(z_dst, n_modes: int = EOF_N_MODES, eof_path: str = TEMPLATE_EOF) -> list:
    """EOF1..n on z_dst. Adjustment does not estimate coefficients."""
    z_dst = np.asarray(z_dst, dtype=np.float64)
    if not os.path.isfile(eof_path) or n_modes <= 0:
        return []
    df = pd.read_csv(eof_path)
    z = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(np.float64)
    phis = []
    for k in range(1, int(n_modes) + 1):
        col = f"EOF{k}"
        if col not in df.columns:
            break
        phi = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
        ok = np.isfinite(z) & np.isfinite(phi)
        phis.append(np.interp(z_dst, z[ok], phi[ok]))
    return phis


def chord_kernel_phi(rx, P, z_grid, c_grid, phi_grid, n=128):
    """∂τ/∂a along the geometric chord: ∫ -φ/c² ds."""
    rx = np.asarray(rx, dtype=np.float64)
    if len(rx) == 0:
        return np.zeros(0, dtype=np.float64)
    p = np.asarray(P, dtype=np.float64).reshape(3)
    geom = np.linalg.norm(p[None, :] - rx, axis=1)
    ts = np.linspace(0.0, 1.0, int(n))
    depth = np.abs(rx[:, 2][:, None] + ts[None, :] * (p[2] - rx[:, 2][:, None]))
    c = np.clip(np.interp(depth, z_grid, c_grid), 1.0, None)
    phi = np.interp(depth, z_grid, phi_grid)
    ds = (geom / max(n - 1, 1))[:, None]
    c_mid = 0.5 * (c[:, :-1] + c[:, 1:])
    phi_mid = 0.5 * (phi[:, :-1] + phi[:, 1:])
    return np.sum((-phi_mid / (c_mid * c_mid)) * ds, axis=1)


def eof_obs_sigma(rx, P, z_grid, c_grid, phis, c_eq, sigma_ac: float = SIGMA_AC, sigma_c1: float = EOF_SIGMA_C1):
    """Range σ: hypot(σ_ac, RSS of C_EQ·k_k·σ_{c,k}). Frozen at initial X; c_k not estimated."""
    n = len(rx)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    sig = np.full(n, float(sigma_ac), dtype=np.float64)
    if sigma_c1 <= 0.0 or not phis:
        return sig
    amps = eof_mode_amplitudes(sigma_c1, len(phis))
    ssp2 = np.zeros(n, dtype=np.float64)
    for phi, amp in zip(phis, amps):
        k = chord_kernel_phi(rx, P, z_grid, c_grid, phi)
        ssp2 = ssp2 + (float(c_eq) * k * float(amp)) ** 2
    return np.hypot(sig, np.sqrt(ssp2))


def _find_time_col(df: pd.DataFrame):
    return next((c for c in df.columns if "time" in c.lower() or "utc" in c.lower()), None)


def _rename_nav(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        c = col.lower().strip()
        if c in ("lat", "latitude", "latitude_deg"):
            mapping[col] = "lat"
        elif c in ("lon", "longitude", "longitude_deg"):
            mapping[col] = "lon"
        elif c in ("alt", "altitude", "altitude_m", "height"):
            mapping[col] = "alt"
        elif c in ("pitch", "pitch_deg"):
            mapping[col] = "pitch"
        elif c in ("roll", "roll_deg"):
            mapping[col] = "roll"
        elif c in ("heading", "heading_deg", "yaw"):
            mapping[col] = "yaw"
    return df.rename(columns=mapping)


def _delay_to_seconds(delay: pd.Series) -> pd.Series:
    med = float(delay.median()) if len(delay) else 0.0
    if med > 10000.0:
        return delay / 1e6
    if med > 100.0:
        return delay / 1000.0
    return delay


def drop_isolated_delay_spikes(df_ping: pd.DataFrame, threshold_s: float = SPIKE_DELAY_S) -> pd.DataFrame:
    """Drop interior pings that jump off the line between neighbors.

    A range ramp makes consecutive |Δdelay| large on both sides, so requiring
    both neighbor diffs > threshold misses spikes and also false-flags turns.
    Comparing to 0.5*(prev+next) keeps ramps and cuts pick errors / step onsets.
    Endpoints are kept (no two-sided neighborhood).
    """
    if df_ping.empty or "delay" not in df_ping.columns:
        return df_ping
    parts = []
    n_drop = 0
    for _, g in df_ping.groupby("Beacon_ID", sort=False):
        g = g.sort_index()
        delay = g["delay"]
        pred = 0.5 * (delay.shift(1) + delay.shift(-1))
        err = (delay - pred).abs()
        keep = err.isna() | (err <= threshold_s)
        n_drop += int((~keep).sum())
        parts.append(g[keep])
    if not parts:
        return df_ping.iloc[0:0].copy()
    out = pd.concat(parts).sort_index()
    if n_drop:
        print(f"[QC] isolated delay spikes |d-0.5(prev+next)| > {threshold_s*1e3:.0f} ms: dropped {n_drop}")
    return out


def load_track(nav_filename: str, ping_filename: str, data_dir: str = DATA_DIR):
    """同一套装载 + 预清洗。返回 (df_nav, df_ping)，索引均为 UTC。"""
    nav_path = os.path.join(data_dir, nav_filename)
    ping_path = os.path.join(data_dir, ping_filename)
    if not os.path.exists(nav_path) or not os.path.exists(ping_path):
        return None, None

    df_nav = pd.read_csv(nav_path)
    tcol = _find_time_col(df_nav)
    if tcol is None:
        raise ValueError(f"导航文件无时间列: {nav_filename}")
    df_nav["Time_UTC"] = pd.to_datetime(df_nav[tcol], errors="coerce")
    df_nav = df_nav.set_index("Time_UTC")
    df_nav = _rename_nav(df_nav)
    for req in ("lat", "lon", "alt", "pitch", "roll", "yaw"):
        if req in df_nav.columns:
            df_nav[req] = pd.to_numeric(df_nav[req], errors="coerce")
        else:
            df_nav[req] = 0.0
    df_nav = df_nav.dropna(subset=["lat", "lon", "alt"])
    df_nav[["pitch", "roll", "yaw"]] = df_nav[["pitch", "roll", "yaw"]].fillna(0.0)
    df_nav = df_nav[df_nav.index.notnull()].sort_index()
    df_nav = df_nav[~df_nav.index.duplicated(keep="first")]
    df_nav["gnss_E"] = np.radians(df_nav["lon"] - REF_LON) * RN * np.cos(_lat0)
    df_nav["gnss_N"] = np.radians(df_nav["lat"] - REF_LAT) * RM
    df_nav["gnss_U"] = df_nav["alt"]

    df_ping = pd.read_csv(ping_path)
    tcol = _find_time_col(df_ping)
    if tcol is None:
        raise ValueError(f"Ping 文件无时间列: {ping_filename}")
    df_ping["Time_UTC"] = pd.to_datetime(df_ping[tcol], errors="coerce")
    df_ping = df_ping.set_index("Time_UTC")
    delay_col = next(
        (c for c in df_ping.columns if any(k in c.lower() for k in ("delay", "travel", "twt", "obs"))),
        None,
    )
    if delay_col and delay_col != "delay":
        df_ping = df_ping.rename(columns={delay_col: "delay"})
    df_ping["delay"] = pd.to_numeric(df_ping["delay"], errors="coerce")
    if "Beacon_ID" not in df_ping.columns:
        raise ValueError(f"Ping 文件无 Beacon_ID: {ping_filename}")
    df_ping = df_ping.dropna(subset=["delay", "Beacon_ID"])
    df_ping = df_ping[df_ping.index.notnull()].sort_index()
    df_ping = df_ping[~df_ping.index.duplicated(keep="first")]
    df_ping["delay"] = _delay_to_seconds(df_ping["delay"])
    t0, t1 = df_nav.index.min(), df_nav.index.max()
    df_ping = df_ping[(df_ping.index >= t0) & (df_ping.index <= t1)]
    df_ping = drop_isolated_delay_spikes(df_ping, SPIKE_DELAY_S)
    return df_nav, df_ping


def empty_unified_row() -> dict:
    return {k: np.nan for k in UNIFIED_COLUMNS}


def unified_row(**kwargs) -> dict:
    row = empty_unified_row()
    row.update(kwargs)
    return row


def backup_and_write_csv(path: str, df: pd.DataFrame, encoding: str = "utf-8-sig"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(path)
        shutil.copy2(path, f"{root}.bak_{ts}{ext}")
    df.to_csv(path, index=False, encoding=encoding)


def write_unified(path: str, records: list[dict]):
    df = pd.DataFrame(records)
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[UNIFIED_COLUMNS]
    backup_and_write_csv(path, df)
    return df


def load_interbeacon_tau(path: str = INTERBEACON_MEDIAN_CSV) -> dict:
    """
    读站间单程时延中位数。键 TAU52/TAU56/TAU26 为秒，TAU52_US 等为微秒（与原始记录同量纲）。
    CSV 不存在时全部为 nan。
    """
    out = {
        "TAU52": np.nan, "TAU56": np.nan, "TAU26": np.nan,
        "TAU52_US": np.nan, "TAU56_US": np.nan, "TAU26_US": np.nan,
        "n52": 0, "n56": 0, "n26": 0,
    }
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return out
    df = pd.read_csv(path)
    if "pair" not in df.columns:
        return out
    key_map = {
        "5-2": ("TAU52", "TAU52_US", "n52"),
        "2-5": ("TAU52", "TAU52_US", "n52"),
        "5-6": ("TAU56", "TAU56_US", "n56"),
        "6-5": ("TAU56", "TAU56_US", "n56"),
        "2-6": ("TAU26", "TAU26_US", "n26"),
        "6-2": ("TAU26", "TAU26_US", "n26"),
    }
    for _, row in df.iterrows():
        pair = str(row["pair"]).strip()
        if pair not in key_map:
            continue
        ks, kus, kn = key_map[pair]
        if "median_s" in df.columns and pd.notna(row.get("median_s")):
            out[ks] = float(row["median_s"])
        if "median_us" in df.columns and pd.notna(row.get("median_us")):
            out[kus] = float(row["median_us"])
        if "n" in df.columns and pd.notna(row.get("n")):
            out[kn] = int(row["n"])
    return out


_TAU = load_interbeacon_tau()
TAU52, TAU56, TAU26 = _TAU["TAU52"], _TAU["TAU56"], _TAU["TAU26"]
TAU52_US, TAU56_US, TAU26_US = _TAU["TAU52_US"], _TAU["TAU56_US"], _TAU["TAU26_US"]
