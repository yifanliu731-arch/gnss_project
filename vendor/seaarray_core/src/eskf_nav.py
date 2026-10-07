"""
eskf_nav.py — 误差状态扩展卡尔曼滤波 (ESKF) 组合导航引擎
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
from numba import njit

EARTH_A = 6378137.0          
EARTH_F = 1.0 / 298.257223563 
EARTH_B = EARTH_A * (1.0 - EARTH_F)
EARTH_E2 = (EARTH_A**2 - EARTH_B**2) / EARTH_A**2
EARTH_WIE = 7.2921151467e-5     
EARTH_G0 = 9.7803267714        

@njit
def earth_radius_meridian(lat_rad: float) -> float:
    s2 = np.sin(lat_rad) ** 2
    return EARTH_A * (1.0 - EARTH_E2) / (1.0 - EARTH_E2 * s2) ** 1.5

@njit
def earth_radius_prime_vertical(lat_rad: float) -> float:
    s2 = np.sin(lat_rad) ** 2
    return EARTH_A / np.sqrt(1.0 - EARTH_E2 * s2)

@njit
def earth_rate_ned(lat_rad: float) -> np.ndarray:
    return np.array([EARTH_WIE * np.cos(lat_rad), 0.0, -EARTH_WIE * np.sin(lat_rad)])

@njit
def transport_rate_ned(vn: np.ndarray, lat_rad: float, alt_m: float) -> np.ndarray:
    ve, vn_, vu = vn[0], vn[1], vn[2]  
    RM = earth_radius_meridian(lat_rad)
    RN = earth_radius_prime_vertical(lat_rad)
    return np.array([
         ve / (RN + alt_m),          
        -vn_ / (RM + alt_m),         
        -ve * np.tan(lat_rad) / (RN + alt_m),  
    ])

@njit
def skew_symmetric(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0,    -v[2],   v[1]],
        [v[2],   0.0,    -v[0]],
        [-v[1],  v[0],    0.0],
    ])

@njit
def update_gnss_21D_jit(X: np.ndarray, P: np.ndarray, pos: np.ndarray, vn: np.ndarray, time_delay: float, pos_measured: np.ndarray, std_pos_3d: np.ndarray):
    C_ned2enu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    pos_predicted = pos - vn * time_delay
    z = pos_measured - pos_predicted
    
    H = np.zeros((3, 21))
    H[0:3, 6:9]   = C_ned2enu
    H[0:3, 18]    = -vn 
    
    S = H @ P @ H.T + np.diag(std_pos_3d**2)
    K = P @ H.T @ np.linalg.inv(S)
    
    X_new = X + K @ z
    P_new = (np.eye(21) - K @ H) @ P
    P_new = 0.5 * (P_new + P_new.T)
    return X_new, P_new

@njit
def update_beacon_range_21D_jit(X: np.ndarray, P: np.ndarray, Cbn: np.ndarray, pos: np.ndarray, lever_arm: np.ndarray, sv_bias: float, sys_delay: float, beacon_enu: np.ndarray, measured_time: float, std_range: float, C_0: float):
    C_rfu2frd = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    C_ned2enu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    
    lever_ned = Cbn @ C_rfu2frd @ lever_arm
    lever_enu = C_ned2enu @ lever_ned
    transducer_pos = pos + lever_enu
    delta = transducer_pos - beacon_enu
    r_geom = np.linalg.norm(delta)
    
    if r_geom < 1e-3: 
        return X, P
        
    pred_pseudo_range = r_geom + sv_bias * (r_geom / C_0) + sys_delay * C_0
    z = measured_time * C_0 - pred_pseudo_range
    
    H = np.zeros((1, 21))
    e_enu = delta / r_geom  
    H[0, 6:9]   = e_enu @ C_ned2enu
    H[0, 0:3]   = e_enu @ (-C_ned2enu @ skew_symmetric(lever_ned))
    H[0, 15:18] = e_enu @ (C_ned2enu @ Cbn @ C_rfu2frd)
    H[0, 19]    = r_geom / C_0
    H[0, 20]    = C_0
    
    S = H @ P @ H.T + np.array([[std_range**2]])
    K = P @ H.T @ np.linalg.inv(S)
    
    X_new = X + (K * z).flatten()
    P_new = (np.eye(21) - K @ H) @ P
    P_new = 0.5 * (P_new + P_new.T)
    return X_new, P_new

@njit
def build_Phi_Gamma_21D_jit(Cbn: np.ndarray, fn_ned: np.ndarray, vn_ned: np.ndarray, lat: float, alt: float, dt: float):
    RM = earth_radius_meridian(lat)
    RN = earth_radius_prime_vertical(lat)
    vN, vE, vD = vn_ned[0], vn_ned[1], vn_ned[2]
    
    F = np.zeros((15, 15))
    win_n = earth_rate_ned(lat) + transport_rate_ned(np.array([vE, vN, -vD]), lat, alt)
    F[0:3, 0:3] = -skew_symmetric(win_n)
    F[3:6, 0:3] = -skew_symmetric(fn_ned)
    F[3, 6] = -2.0 * vE * EARTH_WIE * np.cos(lat) / (RM + alt) 
    
    F_22 = np.zeros((3, 3))
    F_22[0, 1] = - (2.0 * EARTH_WIE * np.sin(lat) + vE * np.tan(lat) / (RN + alt))
    F_22[0, 2] = vN / (RM + alt)
    F_22[1, 0] = 2.0 * EARTH_WIE * np.sin(lat) + vE * np.tan(lat) / (RN + alt)
    F_22[1, 2] = 2.0 * EARTH_WIE * np.cos(lat) + vE / (RN + alt)
    F_22[2, 0] = -vN / (RM + alt)
    F_22[2, 1] = - (2.0 * EARTH_WIE * np.cos(lat) + vE / (RN + alt))
    F[3:6, 3:6] = F_22
    
    F[6, 3] = 1.0 / (RM + alt)
    F[7, 4] = 1.0 / ((RN + alt) * np.cos(lat))
    F[8, 5] = -1.0
    F[0:3, 9:12] = -Cbn          
    F[3:6, 12:15] = Cbn          
    
    F_21 = np.zeros((21, 21))
    F_21[0:15, 0:15] = F
    Phi_21 = np.eye(21) + F_21 * dt + 0.5 * (F_21 @ F_21) * (dt**2)
    
    Gamma_21 = np.zeros((21, 12))
    Gamma_21[0:3, 0:3]   = -Cbn            
    Gamma_21[3:6, 3:6]   = Cbn             
    Gamma_21[9:12, 6:9]  = np.eye(3) * dt  
    Gamma_21[12:15, 9:12]= np.eye(3) * dt  
    return Phi_21, Gamma_21

@njit
def rts_smooth_jit(Xf_arr: np.ndarray, Pf_arr: np.ndarray, P_pred_arr: np.ndarray, Phi_arr: np.ndarray):
    N = Xf_arr.shape[0]
    Xs_arr = np.zeros_like(Xf_arr)
    Ps_arr = np.zeros_like(Pf_arr)
    
    Xs_arr[-1] = Xf_arr[-1].copy()
    Ps_arr[-1] = Pf_arr[-1].copy()
    
    eye21 = np.eye(21) * 1e-10
    
    for k in range(N - 2, -1, -1):
        Phi = Phi_arr[k + 1]           
        P_f_k  = Pf_arr[k]             
        P_pred_k1 = P_pred_arr[k + 1]  
        
        P_pred_reg = P_pred_k1 + eye21
        Ks = P_f_k @ Phi.T @ np.linalg.inv(P_pred_reg)
            
        Xs_arr[k] = Xf_arr[k] + Ks @ (Xs_arr[k + 1] - Phi @ Xf_arr[k])
        Ps_arr[k] = P_f_k + Ks @ (Ps_arr[k + 1] - P_pred_k1) @ Ks.T
        Ps_arr[k] = 0.5 * (Ps_arr[k] + Ps_arr[k].T)
        
    return Xs_arr, Ps_arr

def quat2att(q: np.ndarray) -> Tuple[float, float, float]:
    r = R.from_quat(q)  
    yaw, pitch, roll = r.as_euler('ZYX', degrees=False)
    return roll, pitch, yaw

def att2quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r = R.from_euler('ZYX', [yaw, pitch, roll], degrees=False)
    return r.as_quat()  

def quat2dcm(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()

class ESKF:
    def __init__(
        self, init_pos: np.ndarray, init_vel: np.ndarray, init_att: np.ndarray,      
        imu_dt: float = 0.1, arw: float = 0.01, vrw: float = 0.05,         
        gb_std: float = 1.0, ab_std: float = 100.0, 
        gnss_std_E: float = 2.0, gnss_std_N: float = 2.0, gnss_std_U: float = 2.0
    ):
        self.dt = imu_dt
        self.Qc = self._build_Qc(arw, vrw, gb_std, ab_std)  
        
        self.gnss_std_E = gnss_std_E 
        self.gnss_std_N = gnss_std_N
        self.gnss_std_U = gnss_std_U
        
        self.lever_arm = np.array([1.585, 6.716, -29.050])  
        self.time_delay = 0.0                         
        self.sv_bias = 0.0                            
        self.sys_delay = 0.0                          
        
        self.P = self._init_P_21D() 
        self.X = np.zeros(21)       
        self.Phi = np.eye(21)
        
        self.q   = att2quat(np.deg2rad(init_att[0]), np.deg2rad(init_att[1]), np.deg2rad(init_att[2]))
        self.vn  = init_vel.copy()     
        self.pos = init_pos.copy()     

    def _build_Qc(self, arw: float, vrw: float, gb_std: float, ab_std: float) -> np.ndarray:
        arw_si  = arw  * np.pi / 180.0 / 60.0       
        vrw_si  = vrw  * 9.80665 / 60.0              
        gb_si   = gb_std  * np.pi / 180.0 / 3600.0   
        ab_si   = ab_std  * 1e-6 * 9.80665            
        
        Qc = np.zeros((12, 12))
        Qc[0:3, 0:3]   = np.eye(3) * arw_si**2
        Qc[3:6, 3:6]   = np.eye(3) * vrw_si**2
        Qc[6:9, 6:9]   = np.eye(3) * (2.0 * gb_si**2 / 3600.0)
        Qc[9:12, 9:12] = np.eye(3) * (2.0 * ab_si**2 / 3600.0)
        return Qc

    def _init_P_21D(self) -> np.ndarray:
        P = np.zeros((21, 21))
        P[0:3, 0:3] = np.eye(3) * (np.deg2rad(1.0))**2
        P[3:6, 3:6] = np.eye(3) * 0.1**2
        P[6:9, 6:9] = np.diag(np.array([self.gnss_std_E**2, self.gnss_std_N**2, self.gnss_std_U**2]))
        P[9:12, 9:12] = np.eye(3) * (np.deg2rad(1.0) / 3600.0)**2
        P[12:15, 12:15] = np.eye(3) * (100e-6 * 9.80665)**2
        P[15:18, 15:18] = np.eye(3) * 0.5**2    
        P[18, 18]       = 0.01**2               
        P[19, 19]       = 5.0**2                
        P[20, 20]       = 0.005**2              
        return P

    def predict_21D(self, Cbn: np.ndarray, fn_ned: np.ndarray, vn_ned: np.ndarray, lat: float, alt: float):
        Phi, Gamma = build_Phi_Gamma_21D_jit(Cbn, fn_ned, vn_ned, lat, alt, self.dt)
        self.Phi = Phi  
        self.X = Phi @ self.X
        self.P = Phi @ self.P @ Phi.T + Gamma @ self.Qc @ Gamma.T

    def update_gnss_21D(self, pos_measured: np.ndarray):
        std_pos_3d = np.array([self.gnss_std_E, self.gnss_std_N, self.gnss_std_U])
        self.X, self.P = update_gnss_21D_jit(
            self.X, self.P, self.pos, self.vn, self.time_delay, pos_measured, std_pos_3d
        )

    def update_beacon_range_21D(self, beacon_enu: np.ndarray, measured_time: float, std_range: float = 1.0, C_0: float = 1500.0):
        Cbn = quat2dcm(self.q)
        self.X, self.P = update_beacon_range_21D_jit(
            self.X, self.P, Cbn, self.pos, self.lever_arm, self.sv_bias, self.sys_delay,
            beacon_enu, measured_time, std_range, C_0
        )

    def feedback(self):
        dphi, dv, dp = self.X[0:3], self.X[3:6], self.X[6:9]    
        dphi_norm = np.linalg.norm(dphi)
        if dphi_norm > 1e-15:
            axis = dphi / dphi_norm
            dq_corr = np.array([* (axis * np.sin(dphi_norm * 0.5)), np.cos(dphi_norm * 0.5)])
            qw = dq_corr[3] * self.q[3] - np.dot(dq_corr[:3], self.q[:3])
            qv = dq_corr[3] * self.q[:3] + self.q[3] * dq_corr[:3] + np.cross(dq_corr[:3], self.q[:3])
            self.q = np.array([*qv, qw]) / np.linalg.norm([*qv, qw])
            
        self.vn = self.vn + np.array([dv[1], dv[0], -dv[2]])
        self.pos = self.pos + np.array([dp[1], dp[0], -dp[2]])
        self.X[0:9] = 0.0

    def feedback_21D(self):
        self.feedback()  
        self.lever_arm  += self.X[15:18]
        self.time_delay += self.X[18]
        self.sv_bias    += self.X[19]
        self.sys_delay  += self.X[20]
        self.X[15:21] = 0.0

class ESKF_Navigator:
    def __init__(self, df_gnss: pd.DataFrame, df_imu: pd.DataFrame, 
                 gnss_std_E: float = 2.0, gnss_std_N: float = 2.0, gnss_std_U: float = 2.0):
        self.df_gnss = df_gnss.copy()
        self.df_imu  = df_imu.copy()
        self.gnss_std_E = gnss_std_E
        self.gnss_std_N = gnss_std_N
        self.gnss_std_U = gnss_std_U
        
        if "heading" not in self.df_imu.columns and "yaw" in self.df_imu.columns:
            self.df_imu = self.df_imu.rename(columns={"yaw": "heading"})
            
        if len(self.df_imu) > 1:
            self.imu_dt = float(np.median(np.diff(self.df_imu.index.astype(np.int64)) / 1e9))
        else:
            self.imu_dt = 0.1
            
        self._fused_forward: Optional[pd.DataFrame] = None
        self._fused_smoothed: Optional[pd.DataFrame] = None
        
        self._Xf_arr: Optional[np.ndarray] = None
        self._Pf_arr: Optional[np.ndarray] = None
        self._Phi_arr: Optional[np.ndarray] = None
        self._P_pred_arr: Optional[np.ndarray] = None
        
        self._prepare_gnss_enu()
        self.eskf: Optional[ESKF] = None
        self._init_eskf_21d()

    def _init_eskf_21d(self):
        init_pos = self.gnss_enu.iloc[0].to_numpy(dtype=float).copy()
        if len(self.gnss_enu) >= 2:
            dt = (self.gnss_enu.index[1] - self.gnss_enu.index[0]).total_seconds()
            ve0 = (self.gnss_enu["E"].iloc[1] - self.gnss_enu["E"].iloc[0]) / dt if dt > 0 else 0.0
            vn0 = (self.gnss_enu["N"].iloc[1] - self.gnss_enu["N"].iloc[0]) / dt if dt > 0 else 0.0
            vu0 = (self.gnss_enu["U"].iloc[1] - self.gnss_enu["U"].iloc[0]) / dt if dt > 0 else 0.0
        else:
            ve0, vn0, vu0 = 0.0, 0.0, 0.0
        init_vel = np.array([ve0, vn0, vu0])
        
        imu_head = self.df_imu.dropna(subset=["pitch", "roll"]).head(50)
        if len(imu_head) > 0:
            if "heading" in imu_head.columns:
                yaw_mean = imu_head["heading"].mean()
            elif "yaw" in imu_head.columns:
                yaw_mean = imu_head["yaw"].mean()
            else:
                yaw_mean = 0.0
                
            init_att = np.array([
                imu_head["roll"].mean(), 
                imu_head["pitch"].mean(), 
                yaw_mean
            ])
        else:
            init_att = np.array([0.0, 0.0, 0.0])
        
        self.eskf = ESKF(
            init_pos=init_pos, init_vel=init_vel, init_att=init_att, imu_dt=self.imu_dt, 
            gnss_std_E=self.gnss_std_E, gnss_std_N=self.gnss_std_N, gnss_std_U=self.gnss_std_U
        )

    @property
    def lever_arm(self) -> np.ndarray: return self.eskf.lever_arm
    @lever_arm.setter
    def lever_arm(self, value): self.eskf.lever_arm = np.asarray(value, dtype=float).copy()
    
    @property
    def time_delay(self) -> float: return self.eskf.time_delay
    @time_delay.setter
    def time_delay(self, value): self.eskf.time_delay = float(value)
    
    @property
    def sv_bias(self) -> float: return self.eskf.sv_bias
    @sv_bias.setter
    def sv_bias(self, value): self.eskf.sv_bias = float(value)
    
    @property
    def sys_delay(self) -> float: return self.eskf.sys_delay
    @sys_delay.setter
    def sys_delay(self, value): self.eskf.sys_delay = float(value)

    def _prepare_gnss_enu(self):
        df = self.df_gnss
        if {"E_m", "N_m", "U_m"}.issubset(df.columns):
            df = df.rename(columns={"E_m": "E", "N_m": "N", "U_m": "U"})
        if {"E", "N", "U"}.issubset(df.columns):
            sub = df.dropna(subset=["E", "N", "U"])
            self.gnss_enu = sub[["E", "N", "U"]].copy().sort_index()
            self.ref_lat = float(np.deg2rad(df["lat"].dropna().iloc[0])) if "lat" in df.columns else 0.0
            self.ref_alt = float(df["alt"].dropna().iloc[0]) if "alt" in df.columns else 0.0
            self.ref_lon = 0.0
            return
            
        df = df.dropna(subset=["lat", "lon", "alt"])
        self.ref_lat = np.deg2rad(df["lat"].iloc[0])
        self.ref_lon = np.deg2rad(df["lon"].iloc[0])
        self.ref_alt = df["alt"].iloc[0]
        
        lat_rad = np.deg2rad(df["lat"].values)
        lon_rad = np.deg2rad(df["lon"].values)
        RN = earth_radius_prime_vertical(self.ref_lat)
        RM = earth_radius_meridian(self.ref_lat)
        
        self.gnss_enu = pd.DataFrame({
            "E": (lon_rad - self.ref_lon) * RN * np.cos(self.ref_lat),
            "N": (lat_rad - self.ref_lat) * RM,
            "U": df["alt"].values, 
        }, index=df.index).sort_index()

    def run_forward_filter(self, ping_times_s: Optional[np.ndarray] = None, ping_delays_s: Optional[np.ndarray] = None, ping_enus: Optional[np.ndarray] = None) -> pd.DataFrame:
        t_grid = self.df_gnss.index.union(self.df_imu.index).sort_values().unique()
        t_start = max(self.df_gnss.index.min(), self.df_imu.index.min())
        t_end   = min(self.df_gnss.index.max(), self.df_imu.index.max())
        t_grid = t_grid[(t_grid >= t_start) & (t_grid <= t_end)]
        t_grid_num = t_grid.astype(np.int64) / 1e9
        n_pts = len(t_grid)
        
        t_gnss_num = self.gnss_enu.index.astype(np.int64) / 1e9
        gnss_E_raw = self.gnss_enu["E"].values
        gnss_N_raw = self.gnss_enu["N"].values
        gnss_U_raw = self.gnss_enu["U"].values
        
        gnss_E_interp = np.interp(t_grid_num, t_gnss_num, gnss_E_raw)
        gnss_N_interp = np.interp(t_grid_num, t_gnss_num, gnss_N_raw)
        gnss_U_interp = np.interp(t_grid_num, t_gnss_num, gnss_U_raw)
        Ve, Vn, Vu = np.gradient(gnss_E_interp, t_grid_num), np.gradient(gnss_N_interp, t_grid_num), np.gradient(gnss_U_interp, t_grid_num)
        
        t_imu_num = self.df_imu.index.astype(np.int64) / 1e9
        roll_interp  = np.interp(t_grid_num, t_imu_num, self.df_imu["roll"].ffill().bfill().values)
        pitch_interp = np.interp(t_grid_num, t_imu_num, self.df_imu["pitch"].ffill().bfill().values)
        heading_col = "heading" if "heading" in self.df_imu.columns else None
        
        if heading_col is not None:
            heading_interp = np.interp(t_grid_num, t_imu_num, self.df_imu[heading_col].ffill().bfill().values)
            hdg_raw = self.df_imu[heading_col].values
            hdg_is_zero = np.all(np.abs(hdg_raw[~np.isnan(hdg_raw)]) < 1e-6) if len(hdg_raw) > 0 else True
        else:
            heading_interp = np.full_like(roll_interp, 0.0)
            hdg_is_zero = True
            
        if hdg_is_zero:
            heading_interp = np.degrees(np.arctan2(Ve, Vn))
            
        if self.eskf is None:
            self._init_eskf_21d()
        else:
            _la, _td, _sv, _sd = self.eskf.lever_arm.copy(), self.eskf.time_delay, self.eskf.sv_bias, self.eskf.sys_delay
            self._init_eskf_21d()
            self.eskf.lever_arm, self.eskf.time_delay, self.eskf.sv_bias, self.eskf.sys_delay = _la, _td, _sv, _sd
            
        self._Xf_arr = np.zeros((n_pts, 21))
        self._Pf_arr = np.zeros((n_pts, 21, 21))
        self._Phi_arr = np.zeros((n_pts, 21, 21))
        self._P_pred_arr = np.zeros((n_pts, 21, 21))
        
        ping_idx_map: dict = {}
        if ping_times_s is not None and ping_delays_s is not None and ping_enus is not None:
            t_grid_s = t_grid_num - t_grid_num[0]
            pt, pd_ = np.asarray(ping_times_s, dtype=np.float64), np.asarray(ping_delays_s, dtype=np.float64)
            idx = np.clip(np.searchsorted(t_grid_s, pt), 1, n_pts - 1)
            left_diff, right_diff = pt - t_grid_s[idx - 1], t_grid_s[idx] - pt
            closest_idx = np.where(left_diff < right_diff, idx - 1, idx)
            for k, ii in enumerate(closest_idx):
                ping_idx_map.setdefault(int(ii), []).append((float(pd_[k]), ping_enus[k]))
                
        records = []
        for i in range(n_pts):
            self.eskf.dt = min(max(t_grid_num[i] - t_grid_num[i-1], 0.001), 1.0) if i > 0 else 0.1
                
            self.eskf.q = att2quat(np.deg2rad(roll_interp[i]), np.deg2rad(pitch_interp[i]), np.deg2rad(heading_interp[i]))
            Cbn = quat2dcm(self.eskf.q)
            vn_enu = np.array([Ve[i], Vn[i], Vu[i]])
            
            self.eskf.pos += self.eskf.vn * self.eskf.dt
            self.eskf.vn = vn_enu.copy()
            
            self.eskf.predict_21D(Cbn, np.zeros(3), np.array([Vn[i], Ve[i], -Vu[i]]), self.ref_lat, self.ref_alt)
            self._P_pred_arr[i] = self.eskf.P.copy()
            
            time_diff = np.abs(t_gnss_num - t_grid_num[i])
            min_diff_idx = np.argmin(time_diff)
            
            if time_diff[min_diff_idx] < 1e-3:
                real_E = gnss_E_raw[min_diff_idx]
                real_N = gnss_N_raw[min_diff_idx]
                real_U = gnss_U_raw[min_diff_idx]
                self.eskf.update_gnss_21D(np.array([real_E, real_N, real_U]))
                self.eskf.feedback_21D()
            
            self.eskf.vn = vn_enu.copy()
            
            if i in ping_idx_map:
                for dly, b_enu in ping_idx_map[i]:
                    self.eskf.update_beacon_range_21D(b_enu, dly, std_range=1.0, C_0=1500.0)
                    self.eskf.feedback_21D()
                    
            self._Xf_arr[i] = self.eskf.X.copy()
            self._Pf_arr[i] = self.eskf.P.copy()
            self._Phi_arr[i] = self.eskf.Phi.copy()
            
            _rr, _pp, _yy = quat2att(self.eskf.q)
            
            records.append({
                "time": t_grid[i], 
                "E": self.eskf.pos[0], 
                "N": self.eskf.pos[1], 
                "U": self.eskf.pos[2],
                "Ve": self.eskf.vn[0], 
                "Vn": self.eskf.vn[1], 
                "Vu": self.eskf.vn[2],
                "roll": np.rad2deg(_rr), 
                "pitch": np.rad2deg(_pp), 
                "yaw": np.rad2deg(_yy),
            })
            
        self._fused_forward = pd.DataFrame(records).set_index("time")
        print(f"[ESKF] Forward filter complete: {len(self._fused_forward)} discrete natural epochs processed.")
        return self._fused_forward

    def run_rts_smoother(self) -> pd.DataFrame:
        if self._Xf_arr is None: raise RuntimeError("请先运行 run_forward_filter()。")
            
        Xs_arr, Ps_arr = rts_smooth_jit(
            self._Xf_arr, self._Pf_arr, self._P_pred_arr, self._Phi_arr
        )
        
        df = self._fused_forward.copy()
        for i in range(len(self._Xf_arr)):
            dX = Xs_arr[i] - self._Xf_arr[i]
            
            if np.any(np.isnan(dX)) or np.any(np.isinf(dX)) or np.max(np.abs(dX[6:9])) > 50.0:
                continue 
                
            dphi, dv, dp = dX[0:3], dX[3:6], dX[6:9]
            if np.linalg.norm(dphi) > 1e-12:
                q_old = att2quat(np.deg2rad(df["roll"].iloc[i]), np.deg2rad(df["pitch"].iloc[i]), np.deg2rad(df["yaw"].iloc[i]))
                dq = np.array([* (dphi * 0.5), 1.0])
                dq = dq / np.linalg.norm(dq)
                q_new_w = dq[3] * q_old[3] - np.dot(dq[:3], q_old[:3])
                q_new_v = dq[3] * q_old[:3] + q_old[3] * dq[:3] + np.cross(dq[:3], q_old[:3])
                roll, pitch, yaw = quat2att(np.array([*q_new_v, q_new_w]) / np.linalg.norm([*q_new_v, q_new_w]))
                df.iloc[i, df.columns.get_loc("roll")]  = np.rad2deg(roll)
                df.iloc[i, df.columns.get_loc("pitch")] = np.rad2deg(pitch)
                df.iloc[i, df.columns.get_loc("yaw")]   = np.rad2deg(yaw)
                
            df.iloc[i, df.columns.get_loc("Ve")] += dv[1]
            df.iloc[i, df.columns.get_loc("Vn")] += dv[0]
            df.iloc[i, df.columns.get_loc("Vu")] -= dv[2]
            df.iloc[i, df.columns.get_loc("E")] += dp[1]
            df.iloc[i, df.columns.get_loc("N")] += dp[0]
            df.iloc[i, df.columns.get_loc("U")] -= dp[2]
            
        self._fused_smoothed = df
        print(f"[RTS] Smoothing complete: {len(df)} epochs")
        return df

    def get_fused_trajectory(self, smoothed: bool = True) -> pd.DataFrame:
        df = self._fused_smoothed if smoothed else self._fused_forward
        if df is None: raise RuntimeError("尚未运行滤波器。")
        return df[["E", "N", "U", "roll", "pitch", "yaw", "Ve", "Vn", "Vu"]]