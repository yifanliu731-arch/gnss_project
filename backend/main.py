"""
FastAPI Backend for SeaArrayCalib (Seafloor Beacon Positioning).

Endpoints
---------
POST /api/calculate
    - Content-Type: application/json  → simulation mode (async, returns task_id)
    - Content-Type: multipart/form-data → real data mode (sync)
GET  /api/progress/{task_id}
    → { "progress": 0-100, "status": "running|completed|failed", "result": {...} }
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import csv
import io
import traceback
import uuid
import numpy as np

import os
import sys
import pandas as pd
from simulation_generator import (
    generate_simulation_data,
    load_ssp_from_csv,
    _generate_circle,
    _generate_circle_cross,
    _generate_square,
    _generate_asterisk,
)
from munk_ssp import generate_munk_ssp_table
from real_solver import RealDataSolver
from seaarray_calib import run_toa_calibration
from ssp_preprocessor import load_and_parse_ssp
from ray_tracing import ssp_equivalent_speed
from geo import (
    origin_from_latlon,
    enu_matrix,
    apply_lever_arm_batch,
    find_column,
    column_values,
    latlonalt_to_enu,
    interp_heading,
)
from scipy.interpolate import CubicSpline
from track_display import mark_track_gaps, copy_gap_metadata, uploaded_times
from datetime import datetime
import re

def parse_beacon_ids(value):
    tokens=[v for v in re.split(r'[,，;；\s]+',str(value).strip()) if v]
    if any(not token.isdigit() or int(token)<=0 for token in tokens):
        raise ValueError('Beacon IDs must be positive integers separated by commas or spaces.')
    ids=list(dict.fromkeys(map(int,tokens)))
    if len(ids)>64:
        raise ValueError('Select at most 64 beacon IDs.')
    return ids

def delay_beacon_column(rows):
    return find_column(rows,('Beacon_ID','beacon_id','beaconID','BeaconID','Becon_ID','bid','信标ID','信标编号'))

def detected_beacon_ids(rows):
    column=delay_beacon_column(rows)
    if not column:
        return None,{}
    counts={}
    for row in rows:
        value=float(row[column])
        if not np.isfinite(value) or value<=0 or value!=int(value):
            raise ValueError('The travel-time file contains invalid beacon IDs.')
        bid=int(value);counts[bid]=counts.get(bid,0)+1
    return column,counts


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SeaArrayCalib API",
    version="1.0.0",
    description="Underwater beacon positioning — trajectory optimisation and simulation",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# CORS 加固：本地开发工具无 Cookie/认证依赖，优先读环境变量 CORS_ORIGINS；
# 未配置时放行所有来源，彻底规避 Vite 端口漂移与 localhost/127.0.0.1 不一致
# 导致的轮询请求被 CORS 拦截问题。
_cors_origins_raw = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,"
    "http://127.0.0.1:5174,http://localhost:3000",
)
if _cors_origins_raw.strip().lower() in ("*", "all", "any"):
    _cors_allow_origins = ["*"]
    _cors_allow_credentials = False
else:
    _cors_allow_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    _cors_allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global task store for async polling
# ---------------------------------------------------------------------------
TASK_STORE: dict = {}


def _json_safe(obj):
    """递归把 NaN/Infinity/NumPy 标量转为 JSON 兼容值（NaN/Inf → None）。"""
    import math as _math

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if _math.isfinite(val) else None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    # 兜底：其余类型交给默认处理（可能仍会报错，但已覆盖本项目数据类型）
    return obj


# =====================================================================
#  Background computation function  (runs in thread pool, NOT async)
# =====================================================================

def _run_simulation_background(task_id: str, body: dict):
    from joint_simulation import run_survey
    TASK_STORE[task_id].update(status="running", progress=2, message="Preparing joint simulation")
    def report(current, total, row):
        TASK_STORE[task_id].update(progress=round(5+40*(row.get('stage',0)+current/total)/row.get('rounds',1) if row.get('phase')=='search' else 45+54*current/total, 1),
            message=f"{row.get('phase','solve').capitalize()} {current}/{total} · {row['label']} · {row['size_m']:.0f} m",
            completed_candidates=current, total_candidates=total)
    try:
        def retain(rows):
            TASK_STORE[task_id]['_export'] = dict(kind='simulation', rows=rows)
        result = run_survey(body, report, export_sink=retain)
        TASK_STORE[task_id]['_export']['result'] = result
        TASK_STORE[task_id].update(status="completed", progress=100,
                                   message=result['message'], result=_json_safe(result))
    except Exception as exc:
        TASK_STORE[task_id].update(status="failed", error=str(exc),
                                  traceback=traceback.format_exc())


@app.post("/api/simulation/preview")
def simulation_preview(body: dict):
    from joint_simulation import configuration, initial_specs
    from closed_survey import SurveyTour
    try:
        c = configuration(body)
        radius = float(np.mean(-c['beacons'][:, 2]))*float(c['ratios'][0])
        tour = SurveyTour(c['modes'][0], radius, c['center'], c['angles'][0])
        return {"trajectory": tour.preview(), "length_m": tour.length,
                "duration_h": tour.length/c['speed']/3600,"seed_candidate_count":len(initial_specs(c)),
                "mean_depth_m":float(np.mean(-c['beacons'][:,2])),"scale_bounds_ratio":[float(c['ratios'][0]),float(c['ratios'][-1])],
                **tour.core.metadata()}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get('/api/simulation/profile-example')
def profile_example():
    from profile_data import example_payload
    return example_payload()


@app.get('/api/simulation/profile-example/download')
def profile_example_download():
    import zipfile
    from profile_data import EXAMPLE
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('background.csv','history.csv','provenance.json','README.md'):
            archive.writestr(name, (EXAMPLE/name).read_bytes())
    return Response(output.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition':'attachment; filename="south-china-sea-profiles.zip"'})


@app.post('/api/simulation/profiles/check')
def check_profiles(body: dict):
    from profile_data import load_profiles
    try:
        return load_profiles(body, fit=False)['metadata']
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


# =====================================================================
# POST  /api/calculate
# =====================================================================
@app.post("/api/calculate")
async def api_calculate(request: Request, background_tasks: BackgroundTasks):
    """
    Unified calculate endpoint.

    - JSON      → simulation mode (async, returns task_id, poll for progress)
    - multipart → real-data mode (sync, returns result directly)
    """
    content_type = request.headers.get("content-type", "")

    try:
        if "application/json" in content_type:
            # --- Simulation mode: launch background task ---
            body = await request.json()
            if not isinstance(body, dict):
                raise HTTPException(status_code=422, detail="Request must be a JSON object")
            # ---- 后端接收入口：打印收到的 JSON Payload 关键字段 ----
            _ap = body.get("array_points")
            if isinstance(_ap, list) and len(_ap) > 0:
                _cols = len(_ap[0]) if isinstance(_ap[0], (list, tuple)) else 0
                print(f"[SIM] 入口收到 array_points: shape=[{len(_ap)} x {_cols}], 数值={_ap}")
            else:
                print(f"[SIM] 入口收到 array_points: 缺失或格式异常 = {_ap}")
            print(f"[SIM] 入口收到 lever_arm = {body.get('lever_arm')}, "
                  f"modes = {body.get('modes')}")
            task_id = str(uuid.uuid4())[:12]
            TASK_STORE[task_id] = {
                "progress": 0,
                "status": "pending",
                "task_id": task_id,
            }
            from joint_simulation import configuration
            try:
                configuration(body)
                if body.get('sound_field_mode', 'munk') not in ('munk','profiles'):
                    raise ValueError('Unknown sound-field mode.')
                if body.get('sound_field_mode') == 'profiles':
                    from profile_data import load_profiles
                    load_profiles(body.get('sound_field'), fit=False)
            except (ValueError, TypeError) as exc:
                TASK_STORE.pop(task_id, None)
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            background_tasks.add_task(_run_simulation_background, task_id, body)
            print(f"[SIM] Created task {task_id}")
            return JSONResponse(content={"task_id": task_id})

        elif "multipart/form-data" in content_type:
            # --- Real-data mode: launch background task (supports progress bar) ---
            form = await request.form()
            gnss_file = form.get("gnss_file")
            delay_file = form.get("delay_file")
            svp_file = form.get("svp_file")
            lever_dx = float(form.get("lever_dx", 1.585))
            lever_dy = float(form.get("lever_dy", 6.716))
            lever_dz = float(form.get("lever_dz", -29.05))
            solver_type = str(form.get("solver_type", "joint_map")).strip().lower()
            if solver_type not in ("joint_map", "single_map", "7param", "9param", "eskf_21param"):
                raise HTTPException(status_code=422,detail='Unknown field solver.')
            try:
                beacon_ids = parse_beacon_ids(form.get("beacon_ids", ""))
                joint_noise_s = float(form.get('joint_noise_ms',.5))/1000
                joint_gnss_m = float(form.get('joint_gnss_m',.05))
                if not np.isfinite([lever_dx,lever_dy,lever_dz,joint_noise_s,joint_gnss_m]).all() or joint_noise_s<=0 or joint_gnss_m<0:
                    raise ValueError('Lever arm and joint noise settings must be finite; timing noise must be positive.')
            except ValueError as exc:
                raise HTTPException(status_code=422,detail=str(exc)) from exc
            if not all((gnss_file,delay_file,svp_file)):
                raise HTTPException(status_code=422,detail='Upload GNSS, travel-time and SSP files.')

            task_id = str(uuid.uuid4())[:12]
            TASK_STORE[task_id] = {
                "progress": 0,
                "status": "pending",
                "task_id": task_id,
                "message": "Task queued",
            }
            # Read file contents NOW (before response returns), since
            # BackgroundTasks run after the response and UploadFile handles
            # may already be closed.
            gnss_bytes = await gnss_file.read() if gnss_file else None
            delay_bytes = await delay_file.read() if delay_file else None
            svp_bytes = await svp_file.read() if svp_file else None
            # ---- 后端接收入口：打印实测 multipart 关键字段 ----
            print(f"[REAL] 入口收到 lever_arm=[{lever_dx}, {lever_dy}, {lever_dz}], "
                  f"solver_type={solver_type}, beacon_ids={beacon_ids}")
            print(f"[REAL] 入口收到文件: gnss={len(gnss_bytes) if gnss_bytes else 0}B, "
                  f"delay={len(delay_bytes) if delay_bytes else 0}B, "
                  f"svp={len(svp_bytes) if svp_bytes else 0}B")
            background_tasks.add_task(
                _run_realdatabackground, task_id,
                gnss_bytes, delay_bytes, svp_bytes,
                lever_dx, lever_dy, lever_dz, solver_type, beacon_ids, joint_noise_s, joint_gnss_m,
            )
            print(f"[REAL] Created task {task_id} — solver={solver_type}, beacons={beacon_ids}")
            return JSONResponse(content={"task_id": task_id})

        else:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Content-Type: {content_type}. "
                       "Use application/json or multipart/form-data.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[ERROR] /api/calculate failed:\n{tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(exc)}\n\nTraceback:\n{tb}",
        )


# =====================================================================
# GET  /api/progress/{task_id}
# =====================================================================
@app.get("/api/progress/{task_id}")
def get_progress(task_id: str):
    """Return the current progress and status of a background simulation task."""
    info = TASK_STORE.get(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    # 结果里可能含 NaN/inf（无效候选的 gdop=np.inf、rmse=np.nan），
    # 直接序列化会抛 ValueError → 500 且无 CORS 头，必须先清洗。
    return JSONResponse(content=_json_safe({k:v for k,v in info.items() if not k.startswith('_')}))


@app.post('/api/export/{task_id}')
async def export_result(task_id: str, request: Request):
    from result_export import make_archive
    info = TASK_STORE.get(task_id)
    if info is None:
        raise HTTPException(404, 'Saved run is unavailable. Run the calculation again.')
    if info.get('status') != 'completed' or '_export' not in info:
        raise HTTPException(409, 'Export is available after this run completes.')
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, 'Export options must be an object.')
    try:
        archive = make_archive(info['_export'], body, _json_safe)
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(422, str(error)) from error
    return Response(archive, media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename="seaarray-results.zip"'})


# ------------------------------------------------------------------
# Real-data background handler  (runs in thread pool)
# ------------------------------------------------------------------

def _run_realdatabackground(
    task_id: str,
    gnss_file,
    delay_file,
    svp_file,
    lever_dx: float,
    lever_dy: float,
    lever_dz: float,
    solver_type: str,
    beacon_ids: list,
    joint_noise_s: float = .0005,
    joint_gnss_m: float = .05,
):
    """Process real‑data in a background thread with progress reporting.
       Supports shared N-beacon joint calibration and legacy per-beacon solvers.
       Auto‑detects IMU columns (gyro/accel) in GNSS CSV and applies
       ESKF INS/GNSS loose coupling for high‑precision trajectory."""

    def _update(pct: int, msg: str = ""):
        TASK_STORE[task_id]["progress"] = pct
        if msg:
            TASK_STORE[task_id]["message"] = msg

    try:
        TASK_STORE[task_id]["status"] = "running"
        _update(5, "Reading uploaded files")

        # --- Read files (already bytes, passed from request handler) ---
        gnss_raw = gnss_file if isinstance(gnss_file, bytes) else None
        delay_raw = delay_file if isinstance(delay_file, bytes) else None
        svp_raw = svp_file if isinstance(svp_file, bytes) else None
        # Fallback: if still UploadFile objects (legacy path)
        if gnss_raw is None and gnss_file is not None and hasattr(gnss_file, 'file'):
            gnss_raw = gnss_file.file.read()
        if delay_raw is None and delay_file is not None and hasattr(delay_file, 'file'):
            delay_raw = delay_file.file.read()
        if svp_raw is None and svp_file is not None and hasattr(svp_file, 'file'):
            svp_raw = svp_file.file.read()

        _update(10, "Parsing GNSS and travel-time data")

        # --- Parse GNSS and delay ---
        gnss_data = _parse_csv_content(gnss_raw.decode("utf-8") if gnss_raw else "", "gnss")
        delay_data = _parse_csv_content(delay_raw.decode("utf-8") if delay_raw else "", "delay")

        if not gnss_data or not delay_data:
            TASK_STORE[task_id]["status"] = "failed"
            TASK_STORE[task_id]["error"] = "GNSS or delay data empty."
            return

        # --- Parse SSP ---
        from ssp_preprocessor import parse_raw_ssp
        ssp_df = None
        ssp_max_depth = None
        if svp_raw is not None:
            try:
                # Try the new flexible parser first
                ssp_arr = parse_raw_ssp(svp_raw.decode("utf-8"))
                if ssp_arr.shape[0] > 0:
                    ssp_df = pd.DataFrame({
                        "SV (m/s)": ssp_arr[:, 0],
                        "Depth (m)": ssp_arr[:, 1],
                    })
                    ssp_max_depth = float(ssp_df["Depth (m)"].max())
                    print(f"[REAL-BG] SSP loaded with new parser: {len(ssp_df)} layers, max_depth={ssp_max_depth:.1f} m")
                else:
                    # Fallback to legacy loader
                    ssp_df = load_ssp_from_csv(io.StringIO(svp_raw.decode("utf-8")))
                    ssp_max_depth = float(ssp_df["Depth (m)"].max())
            except Exception:
                print("[REAL-BG] WARN — Could not parse SSP; solvers will use simple LS")

        c_eq = float(ssp_equivalent_speed(ssp_df)) if ssp_df is not None else 1496.32
        print(f"[REAL-BG] C_EQ from SSP = {c_eq:.2f} m/s")

        _update(14, "Converting geographic coordinates to ENU")

        # --- Column detection ---
        lat_col = "Latitude_deg" if any("Latitude_deg" in r for r in gnss_data) else "lat"
        lon_col = "Longitude_deg" if any("Longitude_deg" in r for r in gnss_data) else "lon"
        alt_col = "Altitude_m" if any("Altitude_m" in r for r in gnss_data) else "altitude"
        dly_col = "Delay_s" if any("Delay_s" in r for r in delay_data) else "delay"

        # Detect beacon ID column in delay data
        bcn_col = delay_beacon_column(delay_data)
        if delay_data:
            sample = delay_data[0]
            for k in sample:
                kl = k.lower().strip()
                if 'beacon' in kl or 'becon' in kl or 'id' in kl or 'bid' in kl or '信标' in kl:
                    bcn_col = k
                    break
        # Fallback: check for integer column with values matching beacon_ids
        if bcn_col is None and delay_data:
            for k in list(delay_data[0].keys()):
                try:
                    test_vals = [int(d.get(k, -1)) for d in delay_data[:10]]
                    if any(v in beacon_ids for v in test_vals):
                        bcn_col = k
                        break
                except (ValueError, TypeError):
                    pass

        if solver_type in ('joint_map','single_map') or not beacon_ids:
            bcn_col, available = detected_beacon_ids(delay_data)
            if not beacon_ids:
                if not available:
                    raise ValueError('Enter one beacon ID when the travel-time file has no Beacon_ID column.')
                beacon_ids = sorted(available)
            if solver_type in ('joint_map','single_map') and not bcn_col and len(beacon_ids)>1:
                raise ValueError('Multi-beacon joint calibration requires a Beacon_ID column.')
            missing=set(beacon_ids)-set(available) if available else set()
            if missing:
                raise ValueError(f'Beacon IDs were not found in travel-time data: {sorted(missing)}')

        # --- Compute ENU origin from the uploaded track (0.1 deg) ---
        lat_col = find_column(gnss_data, ("Latitude_deg", "lat", "latitude")) or lat_col
        lon_col = find_column(gnss_data, ("Longitude_deg", "lon", "longitude")) or lon_col
        alt_col = find_column(gnss_data, ("Altitude_m", "alt", "altitude", "height")) or alt_col
        dly_col = find_column(delay_data, ("Delay_s", "delay", "travel_time")) or dly_col

        lats_all = column_values(gnss_data, lat_col)
        lons_all = column_values(gnss_data, lon_col)
        alts_all = column_values(gnss_data, alt_col)
        raw_lat = float(np.mean(lats_all)) if len(lats_all) else 0.0
        raw_lon = float(np.mean(lons_all)) if len(lons_all) else 0.0
        ref_lat, ref_lon = origin_from_latlon(lats_all, lons_all)
        print(f"[REAL-BG] Reference origin: ({ref_lat}, {ref_lon}) from centroid ({raw_lat}, {raw_lon}) of {len(lats_all)} points")

        # --- Parse GNSS timestamps --------------------------------------------------
        gnss_time_col = None
        if gnss_data:
            first = gnss_data[0]
            for k in first.keys():
                kl = (k or '').strip().lower()
                if 'time' in kl or 'utc' in kl or 'timestamp' in kl:
                    gnss_time_col = k
                    break
        if gnss_time_col is not None:
            try:
                _t0 = datetime.fromisoformat(str(gnss_data[0][gnss_time_col]).replace('Z', '+00:00'))
                _gnss_epoch_s = _t0.timestamp()
                gnss_times_s = np.array([
                    datetime.fromisoformat(str(g[gnss_time_col]).replace('Z', '+00:00')).timestamp() - _gnss_epoch_s
                    for g in gnss_data
                ])
                _use_time_align = True
                print(f"[REAL-BG] GNSS time column '{gnss_time_col}' detected, "
                      f"range={gnss_times_s[0]:.1f}s → {gnss_times_s[-1]:.1f}s")
            except Exception:
                gnss_times_s = np.arange(len(gnss_data), dtype=float)
                _use_time_align = False
                print(f"[REAL-BG] WARN — Could not parse GNSS timestamps, using index alignment")
        else:
            gnss_times_s = np.arange(len(gnss_data), dtype=float)
            _use_time_align = False

        # --- Build GNSS ENU array (antenna position, before lever arm) --------------
        n_gnss = len(gnss_data)
        gnss_ship_enu = enu_matrix(lats_all, lons_all, alts_all, ref_lat, ref_lon)
        #  ESKF INS/GNSS 松组合自动检测与融合  (v4.0)
        # ================================================================
        # Check for IMU sensor columns (gyro + accelerometer) in GNSS CSV
        IMU_GYRO_COLS  = ["gyro_x", "gyro_y", "gyro_z"]
        IMU_ACCEL_COLS = ["acc_x", "acc_y", "acc_z"]
        ATT_COLS       = ["pitch", "roll", "heading"]

        _gnss_keys_lower = {k.lower().strip(): k for k in gnss_data[0].keys()}
        _has_gyro  = all(c in _gnss_keys_lower for c in IMU_GYRO_COLS)
        _has_accel = all(c in _gnss_keys_lower for c in IMU_ACCEL_COLS)
        _has_att   = all(
            (c in _gnss_keys_lower) or (c + "_deg" in _gnss_keys_lower)
            for c in ATT_COLS
        )
        _has_raw_imu = _has_gyro and _has_accel

        fused_trajectory_enu = None   # will be set by ESKF if IMU data available
        esfk_used = False
        nav = None                    # ESKF_Navigator 实例 (21D 自标定需要)

        if _has_raw_imu:
            print(f"[REAL-BG] IMU columns detected: gyro={_has_gyro}, accel={_has_accel}, att={_has_att}")
            try:
                # --- Build DataFrames for ESKF_Navigator ---
                # GNSS DataFrame
                gnss_rows = []
                for g in gnss_data:
                    gnss_rows.append({
                        "lat": float(g.get(lat_col, 0)),
                        "lon": float(g.get(lon_col, 0)),
                        "alt": float(g.get(alt_col, 0)),
                    })
                df_gnss = pd.DataFrame(gnss_rows)
                if gnss_time_col is not None and _use_time_align:
                    _t0_dt = datetime.fromisoformat(str(gnss_data[0][gnss_time_col]).replace('Z', '+00:00'))
                    df_gnss.index = pd.to_datetime([g[gnss_time_col] for g in gnss_data], utc=True)
                else:
                    df_gnss.index = pd.date_range("2021-01-01", periods=len(df_gnss), freq="1s")

                # IMU DataFrame
                imu_rows = []
                for g in gnss_data:
                    row = {}
                    for c in ATT_COLS:
                        orig_key = (
                            _gnss_keys_lower.get(c)
                            or _gnss_keys_lower.get(c + "_deg")
                            or _gnss_keys_lower.get(c.capitalize() + "_deg")
                        )
                        row[c] = float(g.get(orig_key, 0.0)) if orig_key else 0.0
                    if _has_raw_imu:
                        for c in IMU_GYRO_COLS + IMU_ACCEL_COLS:
                            orig_key = _gnss_keys_lower.get(c, c)
                            row[c] = float(g.get(orig_key, 0.0))
                    else:
                        # Fill with zeros for gyro/accel when only attitude available
                        for c in IMU_GYRO_COLS + IMU_ACCEL_COLS:
                            row[c] = 0.0
                    imu_rows.append(row)
                df_imu = pd.DataFrame(imu_rows)
                df_imu.index = df_gnss.index  # same time base

                # --- Import and run ESKF_Navigator ---
                from research_bridge import RESEARCH_ROOT
                _SEAARRAY_SRC = str(RESEARCH_ROOT / "src")
                if _SEAARRAY_SRC not in sys.path:
                    sys.path.insert(0, _SEAARRAY_SRC)
                from eskf_nav import ESKF_Navigator

                nav = ESKF_Navigator(df_gnss, df_imu, output_rate_hz=1.0)
                nav.run_forward_filter()
                df_smoothed = nav.run_rts_smoother()
                df_fused = nav.get_fused_trajectory(smoothed=True)

                # ESKF outputs at 1 Hz, but we need positions at original GNSS timestamps
                # for consistent time-alignment with acoustics.
                # Re-sample fused trajectory back to GNSS epochs via numpy interp.
                if _use_time_align and len(gnss_times_s) >= 2:
                    t_fused_num = df_fused.index.astype(np.int64) / 1e9
                    t_fused_sec = t_fused_num - t_fused_num[0]  # relative seconds
                    t_gnss_sec = gnss_times_s  # already relative to same epoch
                    fused_E = np.interp(t_gnss_sec, t_fused_sec, df_fused["E"].values)
                    fused_N = np.interp(t_gnss_sec, t_fused_sec, df_fused["N"].values)
                    fused_U = np.interp(t_gnss_sec, t_fused_sec, df_fused["U"].values)
                    fused_trajectory_enu = np.column_stack([fused_E, fused_N, fused_U])
                    print(f"[REAL-BG] ESKF fused trajectory resampled to {len(fused_trajectory_enu)} GNSS epochs")
                else:
                    fused_trajectory_enu = np.column_stack([
                        df_fused["E"].values,
                        df_fused["N"].values,
                        df_fused["U"].values,
                    ])

                # Translate ESKF frame to match raw GNSS ENU reference frame.
                # ESKF uses its first GNSS point as origin (0,0,0); we need to
                # shift by the raw GNSS ENU of the first point to align with the
                # common reference centroid.
                _eskf_origin_E = float(gnss_ship_enu[0, 0] - fused_trajectory_enu[0, 0]) if len(gnss_ship_enu) > 0 else 0.0
                _eskf_origin_N = float(gnss_ship_enu[0, 1] - fused_trajectory_enu[0, 1]) if len(gnss_ship_enu) > 0 else 0.0
                _eskf_origin_U = float(gnss_ship_enu[0, 2] - fused_trajectory_enu[0, 2]) if len(gnss_ship_enu) > 0 else 0.0
                fused_trajectory_enu[:, 0] += _eskf_origin_E
                fused_trajectory_enu[:, 1] += _eskf_origin_N
                fused_trajectory_enu[:, 2] += _eskf_origin_U
                print(f"[REAL-BG] ESKF frame translated by E={_eskf_origin_E:.1f} N={_eskf_origin_N:.1f} U={_eskf_origin_U:.1f}")
                esfk_used = True
                _update(16, f"Loosely coupled ESKF complete ({'Full INS' if _has_raw_imu else 'Attitude aided'}mode); track smoothed")
                print(f"[REAL-BG] ESKF fused trajectory: {len(fused_trajectory_enu)} epochs, "
                      f"mode={'full INS' if _has_raw_imu else 'attitude-assisted'}")
            except Exception as _eskf_exc:
                _eskf_tb = traceback.format_exc()
                print(f"[REAL-BG] WARN — ESKF fusion failed, falling back to simple ENU:\n{_eskf_tb}")
                esfk_used = False
                fused_trajectory_enu = None
        else:
            print(f"[REAL-BG] No raw inertial measurements — using GNSS with available attitude for the lever arm")

        # --- Attitude for full-lever-arm rotation (roll / pitch / yaw) ---
        roll_col = find_column(gnss_data, ("Roll_deg", "roll", "Roll"))
        pitch_col = find_column(gnss_data, ("Pitch_deg", "pitch", "Pitch"))
        heading_col = find_column(
            gnss_data,
            ("Heading_deg", "heading", "Heading", "Yaw", "yaw", "CourseOverGround_deg", "cog"),
        )
        roll_gnss = column_values(gnss_data, roll_col, 0.0)
        pitch_gnss = column_values(gnss_data, pitch_col, 0.0)
        heading_gnss = column_values(gnss_data, heading_col, 0.0)
        print(
            f"[REAL-BG] 姿态列 roll={roll_col} pitch={pitch_col} yaw={heading_col}；"
            f"Lever arm按全姿态旋转"
        )

        # --- Parse delay timestamps if available, else index-based ------------------
        delay_time_col = None
        if delay_data and _use_time_align:
            first_d = delay_data[0]
            for k in first_d.keys():
                kl = (k or '').strip().lower()
                if 'time' in kl or 'utc' in kl or 'timestamp' in kl:
                    delay_time_col = k
                    break
        if delay_time_col is not None:
            try:
                _td0 = datetime.fromisoformat(str(delay_data[0][delay_time_col]).replace('Z', '+00:00'))
                _delay_epoch_s = _td0.timestamp()
                _offset_s = _delay_epoch_s - _gnss_epoch_s  # align to GNSS epoch
                delay_times_s = np.array([
                    datetime.fromisoformat(str(d[delay_time_col]).replace('Z', '+00:00')).timestamp()
                    - _gnss_epoch_s
                    for d in delay_data
                ])
                _use_delay_align = True
                print(f"[REAL-BG] Delay time column '{delay_time_col}' detected, "
                      f"offset={_offset_s:.1f}s, range={delay_times_s[0]:.1f}s → {delay_times_s[-1]:.1f}s")
            except Exception:
                delay_times_s = np.arange(len(delay_data), dtype=float) * 20.0  # assume 20 s interval
                _use_delay_align = False
                print(f"[REAL-BG] WARN — Could not parse delay timestamps, using 20 s interval fallback")
        else:
            delay_times_s = np.arange(len(delay_data), dtype=float) * 20.0  # assume 20 s interval
            _use_delay_align = False

        # --- Build interpolator for E, N, U over time -------------------------------
        # Tier 1: ESKF fused trajectory (physics-based, full INS or COG + RTS smoothed)
        # Tier 2: Dead-reckoning interpolation  (COG + SOG from GNSS, respects kinematics)
        # Tier 3: Index-based alignment          (fallback, no time info)
        #
        # Choose reference ENU array:
        if esfk_used and fused_trajectory_enu is not None:
            _ref_enu = fused_trajectory_enu
            _interp_method = "ESKF"
            print(f"[REAL-BG] Using ESKF-fused trajectory ({len(_ref_enu)} pts) for time alignment")
        else:
            _ref_enu = gnss_ship_enu
            _interp_method = "DR"  # dead-reckoning (replaces CubicSpline)

        _gnss_has_time = _use_time_align and len(gnss_times_s) >= 2
        _use_delay_align = _use_delay_align and _gnss_has_time

        if _gnss_has_time:
            # Ensure len(ref_enu) == len(gnss_times_s)
            if len(_ref_enu) != len(gnss_times_s):
                print(f"[REAL-BG] WARN — Reference trajectory length ({len(_ref_enu)}) ≠ "
                      f"GNSS times ({len(gnss_times_s)}), using index alignment")
                _gnss_has_time = False
                _use_delay_align = False
            else:
                _sort_idx = np.argsort(gnss_times_s)
                _t_sorted = gnss_times_s[_sort_idx].astype(np.float64)
                _e_sorted = _ref_enu[_sort_idx, 0].astype(np.float64)
                _n_sorted = _ref_enu[_sort_idx, 1].astype(np.float64)
                _u_sorted = _ref_enu[_sort_idx, 2].astype(np.float64)

                if _interp_method == "ESKF":
                    # ESKF trajectory: use CubicSpline (already physically consistent, spline is safe)
                    try:
                        _cs_e = CubicSpline(_t_sorted, _e_sorted, extrapolate=True)
                        _cs_n = CubicSpline(_t_sorted, _n_sorted, extrapolate=True)
                        _cs_u = CubicSpline(_t_sorted, _u_sorted, extrapolate=True)
                        print(f"[REAL-BG] CubicSpline built on ESKF-fused trajectory")
                    except Exception:
                        _gnss_has_time = False
                        _use_delay_align = False
                        print(f"[REAL-BG] WARN — CubicSpline failed on ESKF trajectory, using index alignment")
                else:
                    # GNSS-only: use windowed polynomial fitting for smooth kinematics
                    # Instead of 2-point dead-reckoning, use 10 nearest GNSS fixes
                    # to fit a local 2nd-order polynomial in E(t) and N(t).
                    # This borrows information from ±5 surrounding fixes to produce
                    # smooth, physically plausible intermediate positions.
                    _n_pts = len(_t_sorted)
                    _window_half = 5
                    _min_window = max(4, min(5, _n_pts // 2))
                    _actual_half = max(_min_window, _window_half)

                    print(f"[REAL-BG] Windowed-poly interpolator built "
                          f"(window=±{_actual_half} pts, {_n_pts} GNSS fixes)")

                    def _win_poly(t_q_local: np.ndarray,
                                  times: np.ndarray, values: np.ndarray) -> np.ndarray:
                        """
                        For each query time, select the nearest ±half_window GNSS points,
                        fit a 2nd-order polynomial p(t) = a₀ + a₁·t + a₂·t²,
                        and evaluate at t_q.
                        """
                        n_vals = len(times)
                        if n_vals < 3:
                            # Fallback: linear interpolation
                            return np.interp(t_q_local, times, values)
                        half = min(_actual_half, n_vals // 2 - 1)
                        if half < 2:
                            half = min(2, n_vals // 2)
                        result = np.zeros(len(t_q_local), dtype=np.float64)
                        for _ki, _tq_f in enumerate(t_q_local.flat):
                            _qi = float(_tq_f)
                            # Find the closest index
                            _ci = int(np.searchsorted(times, _qi))
                            _ci = max(half, min(_ci, n_vals - 1 - half))
                            _lo = _ci - half
                            _hi = _ci + half + 1  # inclusive
                            _t_win = times[_lo:_hi].astype(np.float64)
                            _v_win = values[_lo:_hi].astype(np.float64)
                            # Shift time to improve conditioning
                            _t_mean = float(np.mean(_t_win))
                            _t_centered = _t_win - _t_mean
                            # Fit 2nd-order polynomial: v ≈ a₀ + a₁·(t-t̄) + a₂·(t-t̄)²
                            _A = np.column_stack([
                                np.ones(len(_t_win)),
                                _t_centered,
                                _t_centered ** 2,
                            ])
                            try:
                                _coeff, _resid, _rank, _sv = np.linalg.lstsq(_A, _v_win, rcond=None)
                                _t_q_centered = _qi - _t_mean
                                result[_ki] = _coeff[0] + _coeff[1] * _t_q_centered + _coeff[2] * _t_q_centered ** 2
                            except np.linalg.LinAlgError:
                                # Degenerate: fall back to linear interpolation
                                result[_ki] = np.interp(_qi, times, values)
                        return result

                    # Build callables matching the _cs_e/_cs_n/_cs_u interface
                    _cs_e = lambda tq: _win_poly(np.asarray(tq, dtype=np.float64), _t_sorted, _e_sorted)
                    _cs_n = lambda tq: _win_poly(np.asarray(tq, dtype=np.float64), _t_sorted, _n_sorted)
                    _cs_u = lambda tq: _win_poly(np.asarray(tq, dtype=np.float64), _t_sorted, _u_sorted)

        if solver_type in ('joint_map','single_map'):
            if ssp_df is None:
                raise ValueError('Joint calibration requires a valid measured sound-speed profile.')
            if not (_use_delay_align and _gnss_has_time):
                raise ValueError('Joint calibration requires valid UTC timestamps in GNSS and travel-time files.')
            # The research field path uses CubicSpline on time-aligned GNSS.
            # Do not smooth the observations or extrapolate beyond navigation support.
            unique=np.r_[True,np.diff(_t_sorted)>0]
            _t_sorted=_t_sorted[unique];_sort_idx=_sort_idx[unique]
            if len(_t_sorted)<2:
                raise ValueError('At least two distinct GNSS UTC timestamps are required.')
            _cs_e=CubicSpline(_t_sorted,_e_sorted[unique],extrapolate=False)
            _cs_n=CubicSpline(_t_sorted,_n_sorted[unique],extrapolate=False)
            _cs_u=CubicSpline(_t_sorted,_u_sorted[unique],extrapolate=False)

        # --- Filter delay by Beacon ID ---
        if bcn_col is not None:
            delay_by_beacon = {}
            for d in delay_data:
                bid = int(float(d.get(bcn_col, -1)))
                if bid not in delay_by_beacon:
                    delay_by_beacon[bid] = []
                delay_by_beacon[bid].append(d)
            bcn_counts = {k: len(v) for k, v in delay_by_beacon.items()}
            print(f"[REAL-BG] Found beacon IDs in data: {list(delay_by_beacon.keys())} "
                  f"counts={bcn_counts}")
        else:
            # No beacon column: treat all as single beacon
            delay_by_beacon = {beacon_ids[0]: delay_data}

        ls_tracer = None
        if ssp_df is not None:
            try:
                from ray_tracing import RayTracer
                ls_tracer = RayTracer(ssp_df, use_cache=False)
                c_eq = float(getattr(ls_tracer, "c_eq_measured", None) or c_eq)
            except Exception:
                ls_tracer = None

        # --- Process each requested beacon ---
        total_beacons = len(beacon_ids)
        beacon_results = []
        export_rows = []
        all_residuals_ms: list = []
        all_residuals_m: list = []
        all_outlier_mask: list = []
        residual_observations: list = []
        shared_trajectory = None
        shared_trajectory_latlon = None
        shared_trajectory_times = None
        shared_latlon_times = None
        depth_prior = None
        joint_groups = {}
        joint_solution = None

        for idx, bid in enumerate(beacon_ids):
            if bid not in delay_by_beacon:
                beacon_results.append({
                    "beacon_id": bid,
                    "status": "skipped",
                    "message": f"Beacon {bid} was not found in travel-time data",
                })
                continue

            b_delays = delay_by_beacon[bid]
            n_dly = len(b_delays)

            # --- Time-aligned or index-based alignment -------------------------------
            if _use_delay_align and _gnss_has_time:
                # Extract delay times for this beacon (same indexing into delay_times_s)
                b_delay_indices = [di for di,d in enumerate(delay_data) if bcn_col is None or int(float(d[bcn_col]))==bid]
                # If delay_data shares same row refs as b_delays, build by index
                if len(b_delay_indices) == n_dly:
                    b_times = delay_times_s[b_delay_indices]
                else:
                    # Fallback: filter delay_times_s by matching row content
                    b_times = np.array([
                        delay_times_s[di] for di, d in enumerate(delay_data)
                        if int(float(d.get(bcn_col, -1))) == bid
                    ])
                    if len(b_times) != n_dly:
                        b_times = np.arange(n_dly, dtype=float) * 20.0

                t_query = np.asarray(b_times, dtype=float)
                n = n_dly
                pct_start = 20 + int(idx / total_beacons * 70)
                _update(pct_start, f"Beacon {bid}: Preparing fit ({n} observations; CubicSpline time alignment)")

                # Interpolate GNSS antenna ENU at each delay time, then apply full-attitude lever
                antenna = np.column_stack([_cs_e(t_query), _cs_n(t_query), _cs_u(t_query)])
                roll_q = np.interp(t_query, _t_sorted, roll_gnss[_sort_idx])
                pitch_q = np.interp(t_query, _t_sorted, pitch_gnss[_sort_idx])
                yaw_q = interp_heading(t_query, _t_sorted, heading_gnss[_sort_idx])
                obs_ship_b = apply_lever_arm_batch(
                    antenna, [lever_dx, lever_dy, lever_dz], roll_q, pitch_q, yaw_q
                )
                obs_delays_b = np.array([float(d.get(dly_col, 0)) for d in b_delays])

                # Shared trajectory: return interpolated ENU at delay times (with lever arm for display)
                if shared_trajectory is None:
                    shared_trajectory_times = t_query
                    shared_latlon_times = t_query
                    shared_trajectory = [{"x": float(obs_ship_b[i, 0]), "y": float(obs_ship_b[i, 1]), "z": float(obs_ship_b[i, 2])} for i in range(n)]
                    # latlon: interpolate lat/lon from GNSS data at query times (simplified: nearest or use original GNSS lat/lon)
                    shared_trajectory_latlon = []
                    for i in range(n):
                        # Find nearest GNSS index for this delay time
                        _tq = t_query[i]
                        _ji = int(np.argmin(np.abs(gnss_times_s - _tq)))
                        shared_trajectory_latlon.append({
                            "lat": float(gnss_data[_ji].get(lat_col, 0)),
                            "lon": float(gnss_data[_ji].get(lon_col, 0)),
                        })
            else:
                # --- Legacy: index-based alignment ------------------------------------
                n = min(n_dly, n_gnss)
                pct_start = 20 + int(idx / total_beacons * 70)
                _update(pct_start, f"Beacon {bid}: Preparing fit ({n} observations; index alignment)")
                t_query = gnss_times_s[:n] if len(gnss_times_s) >= n else np.arange(n, dtype=float)
                gi = np.arange(n) % n_gnss
                obs_ship_b = apply_lever_arm_batch(
                    gnss_ship_enu[gi],
                    [lever_dx, lever_dy, lever_dz],
                    roll_gnss[gi], pitch_gnss[gi], heading_gnss[gi],
                )
                obs_delays_b = np.array([float(b_delays[i % n_dly].get(dly_col, 0)) for i in range(n)])

                if shared_trajectory is None:
                    shared_trajectory_times = t_query
                    shared_latlon_times = gnss_times_s[:n]
                    shared_trajectory = [{"x": float(obs_ship_b[i, 0]), "y": float(obs_ship_b[i, 1]), "z": float(obs_ship_b[i, 2])} for i in range(n)]
                    shared_trajectory_latlon = [
                        {"lat": float(gnss_data[i % n_gnss].get(lat_col, 0)), "lon": float(gnss_data[i % n_gnss].get(lon_col, 0))}
                        for i in range(n)
                    ]
            n = len(obs_delays_b)  # ensure n matches actual size

            if solver_type in ('joint_map','single_map'):
                if not np.isfinite(obs_ship_b).all():
                    raise ValueError(f'Beacon {bid}: observation UTC is outside valid GNSS support.')
                joint_groups[bid]=dict(ship=obs_ship_b,times=np.asarray(t_query),delays=obs_delays_b,
                    source_times=[d.get(delay_time_col) for d in b_delays])
                continue

            depth_prior = abs(float(np.mean(obs_delays_b) * c_eq))
            ping_times = np.asarray(t_query, dtype=float) if t_query is not None else np.arange(n, dtype=float)

            # --- Run solver ---
            solver_pct_start = pct_start
            solver_pct_range = 70 / total_beacons
            result = None

            if solver_type == "9param" and ssp_df is not None:
                from abic_solver import solve_with_abic

                def _cb(pct, msg):
                    ov = solver_pct_start + int(pct * solver_pct_range / 100)
                    _update(min(ov, 98), f"Beacon {bid}: {msg}")

                result = solve_with_abic(
                    ship=obs_ship_b,
                    delays=obs_delays_b,
                    ssp_df=ssp_df,
                    times_s=ping_times,
                    progress_callback=_cb,
                )
                if result.get("status") != "success":
                    print(f"[REAL] ABIC solver failed: {result.get('message')}; falling back to simple LS")
                    result = None
            elif solver_type == "7param" and ssp_df is not None:
                def _cb(pct, msg):
                    ov = solver_pct_start + int(pct * solver_pct_range / 100)
                    _update(min(ov, 98), f"Beacon {bid}: {msg}")

                result = run_toa_calibration(
                    obs_ship_pos=obs_ship_b,
                    obs_delays=obs_delays_b,
                    ssp_df=ssp_df,
                    progress_callback=_cb,
                    lock_eof=True,
                    lock_k=(solver_type != "9param"),
                    times_s=ping_times,
                    solver_type=solver_type,
                )
                if result.get("status") != "success":
                    print(f"[REAL] TOA solver failed: {result.get('message')}; falling back to simple LS")
                    result = None

            if result is None and solver_type == "eskf_21param" and esfk_used and nav is not None:
                _update(pct_start + 5, f"Beacon {bid}: Running 21-state ESKF calibration")
                print(f"[REAL] Using 21-state ESKF SLAM (beacon {bid})")

                solver = RealDataSolver(ray_tracer=ls_tracer, c_ref=c_eq)
                B_est, rmse_ms, xy_m, z_m, residuals_ms, outlier_mask = solver.solve(
                    obs_ship_pos=obs_ship_b,
                    obs_delays=obs_delays_b,
                    depth_prior=depth_prior,
                    depth_weight=1e3,
                    sigma_threshold=3.0,
                )

                nav.lever_arm = np.array([lever_dx, lever_dy, lever_dz])
                _nav_origin = gnss_ship_enu[0] if n_gnss > 0 else np.zeros(3)

                for _iter in range(3):
                    beacon_enu = B_est - _nav_origin
                    nav.run_forward_filter(
                        ping_times_s=np.asarray(ping_times, dtype=float),
                        ping_delays_s=obs_delays_b,
                        beacon_enu=beacon_enu,
                    )
                    df_smoothed = nav.run_rts_smoother()
                    t_grid_sec = df_smoothed.index.astype(np.int64) / 1e9
                    t_grid_sec = t_grid_sec - t_grid_sec[0]
                    _pt = np.asarray(ping_times, dtype=float)
                    E_s = np.interp(_pt, t_grid_sec, df_smoothed["E"].values)
                    N_s = np.interp(_pt, t_grid_sec, df_smoothed["N"].values)
                    U_s = np.interp(_pt, t_grid_sec, df_smoothed["U"].values)
                    ship_smooth = np.column_stack([E_s, N_s, U_s])
                    B_est, rmse_ms, xy_m, z_m, residuals_ms, outlier_mask = solver.solve(
                        obs_ship_pos=ship_smooth,
                        obs_delays=obs_delays_b,
                        depth_prior=depth_prior,
                        depth_weight=1e3,
                        sigma_threshold=3.0,
                    )

                result = {
                    "status": "success",
                    "result_x": round(float(B_est[0]), 2),
                    "result_y": round(float(B_est[1]), 2),
                    "result_z": round(float(B_est[2]), 2),
                    "rmse_time_ms": round(rmse_ms, 4),
                    "solver_type": "eskf_21param",
                    "residuals_ms": residuals_ms.tolist() if residuals_ms is not None else [],
                    "outlier_mask": outlier_mask.tolist() if outlier_mask is not None else [],
                    "outlier_rate": f"{outlier_mask.mean() * 100:.1f}%" if outlier_mask is not None else "N/A",
                    "rmse_distance_m": round(rmse_ms / 1000.0 * c_eq, 4),
                    "c_eq_mps": round(c_eq, 3),
                    "sound_speed_bias_mps": round(float(nav.sv_bias), 3),
                    "system_delay_ms": round(float(nav.sys_delay) * 1000, 3),
                    "time_sync_delay_ms": round(float(nav.time_delay) * 1000, 3),
                    "lever_arm_z_down_m": round(float(nav.lever_arm[2]), 3),
                }

            if result is None:
                _update(pct_start + 5, f"Beacon {bid}: Running basic least squares")
                solver = RealDataSolver(ray_tracer=ls_tracer, c_ref=c_eq)
                B_est, rmse_ms, xy_m, z_m, residuals_ms, outlier_mask = solver.solve(
                    obs_ship_pos=obs_ship_b,
                    obs_delays=obs_delays_b,
                    depth_prior=depth_prior,
                    depth_weight=1e3,
                    sigma_threshold=3.0,
                )
                result = {
                    "status": "success",
                    "result_x": round(float(B_est[0]), 2),
                    "result_y": round(float(B_est[1]), 2),
                    "result_z": round(float(B_est[2]), 2),
                    "rmse_time_ms": round(rmse_ms, 4),
                    "precision_xy_m": round(xy_m, 4) if not np.isnan(xy_m) else None,
                    "precision_z_m": round(z_m, 4) if not np.isnan(z_m) else None,
                    "outlier_rate": f"{outlier_mask.mean() * 100:.1f}%" if outlier_mask is not None else "N/A",
                    "residuals_ms": residuals_ms.tolist() if residuals_ms is not None else [],
                    "residuals_m": (residuals_ms / 1000.0 * c_eq).tolist() if residuals_ms is not None else [],
                    "outlier_mask": outlier_mask.tolist() if outlier_mask is not None else [],
                    "solver_type": "simple_ls",
                    "rmse_distance_m": round(rmse_ms / 1000.0 * c_eq, 4),
                    "c_eq_mps": round(c_eq, 3),
                }

            # --- Aggregate residuals ---
            if result.get("residuals_ms"):
                all_residuals_ms.extend(result["residuals_ms"])
                if result.get("outlier_mask"):
                    all_outlier_mask.extend(result["outlier_mask"])
            if result.get("residuals_m"):
                all_residuals_m.extend(result["residuals_m"])

            # --- Build per-beacon entry ---
            beacon_entry = {
                "beacon_id": bid,
                "status": result.get("status", "success"),
                "x": result.get("result_x"),
                "y": result.get("result_y"),
                "z": result.get("result_z"),
                "rmse_time_ms": result.get("rmse_time_ms"),
                "rmse_distance_m": result.get("rmse_distance_m"),
                "outlier_rate": result.get("outlier_rate"),
                "n_total": result.get("n_total", n),
                "n_clean": result.get("n_clean", n),
                "solver_type": result.get("solver_type", solver_type),
            }
            # System error fields
            if result.get("system_clock_bias_ms") is not None:
                beacon_entry["system_clock_bias_ms"] = result.get("system_clock_bias_ms")
                beacon_entry["clock_drift_ss"] = result.get("clock_drift_ss")
                beacon_entry["clock_drift2_ss2"] = result.get("clock_drift2_ss2")
                beacon_entry["zenith_k_ms_km2"] = result.get("zenith_k_ms_km2")
                beacon_entry["sv_grad_e_mps_per_km"] = result.get("sv_grad_e_mps_per_km")
                beacon_entry["sv_grad_n_mps_per_km"] = result.get("sv_grad_n_mps_per_km")
                beacon_entry["abic"] = result.get("abic")
                beacon_entry["abic_lambda"] = result.get("abic_lambda")
            if result.get("c_eq_mps") is not None:
                beacon_entry["c_eq_mps"] = result.get("c_eq_mps")
            if result.get("sigma_e_m") is not None:
                beacon_entry["sigma_e_m"] = result.get("sigma_e_m")
                beacon_entry["sigma_n_m"] = result.get("sigma_n_m")
                beacon_entry["sigma_u_m"] = result.get("sigma_u_m")
                beacon_entry["n_sessions"] = result.get("n_sessions")
            # 21D 自标定字段
            if result.get("sound_speed_bias_mps") is not None:
                beacon_entry["sound_speed_bias_mps"] = result.get("sound_speed_bias_mps")
                beacon_entry["system_delay_ms"] = result.get("system_delay_ms")
                beacon_entry["time_sync_delay_ms"] = result.get("time_sync_delay_ms")
                beacon_entry["lever_arm_z_down_m"] = result.get("lever_arm_z_down_m")

            beacon_results.append(beacon_entry)
            export_rows.append(dict(beacon_entry,
                trajectory=[dict(x=float(p[0]),y=float(p[1]),z=float(p[2])) for p in obs_ship_b],
                observations=[dict(observation_index=i+1,beacon_id=bid,time_s=float(ping_times[i]),
                    source_time=b_delays[i].get(delay_time_col) if delay_time_col else None,
                    delay_s=float(obs_delays_b[i]),e_m=float(obs_ship_b[i,0]),n_m=float(obs_ship_b[i,1]),u_m=float(obs_ship_b[i,2])) for i in range(n)],
                residuals_ms=result.get('residuals_ms') or [],residuals_m=result.get('residuals_m') or [],
                outlier_mask=result.get('outlier_mask') or []))

        if solver_type in ('joint_map','single_map'):
            from joint_field import solve_field_joint
            if solver_type == 'joint_map':
                beacon_results,export_rows,joint_solution=solve_field_joint(joint_groups,ssp_df,joint_noise_s,joint_gnss_m,_update)
            else:
                for index,bid in enumerate(sorted(joint_groups)):
                    def report(pct,msg):
                        _update(40+int((index+pct/100)*56/len(joint_groups)),f'Beacon {bid}: {msg}')
                    records,rows,summary=solve_field_joint({bid:joint_groups[bid]},ssp_df,joint_noise_s,joint_gnss_m,report)
                    for row in records+rows:row['solver_type']='single_map'
                    for row in records:row['single_solution']=summary
                    for row in rows:row['single_solution']=summary
                    beacon_results.extend(records);export_rows.extend(rows)
            observations=[]
            for row in export_rows:
                observations.extend(zip(row['observations'],row['residuals_ms'],row['residuals_m']))
            observations.sort(key=lambda item:item[0]['time_s'])
            shared_trajectory=[dict(x=o['e_m'],y=o['n_m'],z=o['u_m']) for o,_,_ in observations]
            shared_trajectory_times=np.array([o['time_s'] for o,_,_ in observations])
            all_residuals_ms=[r for _,r,_ in observations]
            all_residuals_m=[r for _,_,r in observations]
            all_outlier_mask=[False]*len(observations)
            residual_observations=[dict(beacon_id=o['beacon_id'],time_s=o['time_s'],source_time=o['source_time']) for o,_,_ in observations]
            shared_trajectory_latlon=[dict(lat=float(g[lat_col]),lon=float(g[lon_col])) for g in gnss_data]
            shared_latlon_times=gnss_times_s

        # Display segmentation uses original navigation gaps; no fit input changes.
        nav_e,nav_n,_=latlonalt_to_enu(lats_all,lons_all,alts_all,ref_lat,ref_lon)
        nav_marked=mark_track_gaps([dict(x=float(e),y=float(n)) for e,n in zip(nav_e,nav_n)],
                                   gnss_times_s if _use_time_align else None)
        break_times=[p['time_s'] for p in nav_marked if p['gap_before'] and 'time_s' in p]
        shared_trajectory=mark_track_gaps(shared_trajectory or [], shared_trajectory_times if _use_time_align else None, break_times)
        if shared_trajectory_latlon:
            geo_e,geo_n,_=latlonalt_to_enu([p['lat'] for p in shared_trajectory_latlon],
                [p['lon'] for p in shared_trajectory_latlon],np.zeros(len(shared_trajectory_latlon)),ref_lat,ref_lon)
            geo_marked=mark_track_gaps([dict(x=float(e),y=float(n)) for e,n in zip(geo_e,geo_n)],
                shared_latlon_times if _use_time_align else None, break_times)
            shared_trajectory_latlon=copy_gap_metadata(shared_trajectory_latlon,geo_marked)

        # --- Final response ---
        _update(98, "Preparing results")
        response = {
            "ref_lat": round(ref_lat, 8),
            "ref_lon": round(ref_lon, 8),
            "depth_prior_m": round(depth_prior, 1) if depth_prior is not None else None,
            "c_eq_mps": round(c_eq, 3),
            "solver_type": solver_type,
            "joint_solution": joint_solution,
            "trajectory": shared_trajectory or [],
            "trajectory_latlon": shared_trajectory_latlon or [],
            "residuals_ms": all_residuals_ms,
            "residuals_m": all_residuals_m,
            "outlier_mask": all_outlier_mask,
            "residual_observations": residual_observations,
            "beacon_results": beacon_results,
            # Legacy compat: first beacon
            "result_x": next((b.get("x") for b in beacon_results if b.get("status")=="success"),None),
            "result_y": next((b.get("y") for b in beacon_results if b.get("status")=="success"),None),
            "result_z": next((b.get("z") for b in beacon_results if b.get("status")=="success"),None),
            "rmse_time_ms": next((b.get("rmse_time_ms") for b in beacon_results if b.get("status")=="success"),None),
            "outlier_rate": next((b.get("outlier_rate") for b in beacon_results if b.get("status")=="success"),None),
            # ESKF fusion metadata
            "eskf_used": esfk_used,
            "eskf_mode": "full_ins" if (_has_raw_imu and esfk_used) else ("attitude_assisted" if esfk_used else "none"),
            "imu_columns_detected": {
                "gyro": _has_gyro,
                "accel": _has_accel,
                "attitude": _has_att,
            },
        }

        TASK_STORE[task_id]["progress"] = 100
        TASK_STORE[task_id]["status"] = "completed"
        TASK_STORE[task_id]["result"] = response
        exported_ids = {r['beacon_id'] for r in export_rows}
        export_rows.extend(r for r in beacon_results if r['beacon_id'] not in exported_ids)
        TASK_STORE[task_id]['_export'] = dict(kind='field', rows=export_rows, result=response,
            config=dict(lever_arm=[lever_dx,lever_dy,lever_dz],lever_arm_frame="antenna-to-transducer, body right/forward/up",solver_type=solver_type,beacon_ids=beacon_ids,joint_noise_s=joint_noise_s,joint_gnss_m=joint_gnss_m),
            raw={'gnss.csv':gnss_raw,'delay.csv':delay_raw,'ssp.csv':svp_raw})
        TASK_STORE[task_id]["message"] = f"Complete: {len(beacon_results)} beacons processed"
        print(f"[REAL-BG] task={task_id} completed — {len(beacon_results)} beacons, ESKF={esfk_used}")

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[REAL-BG] task={task_id} FAILED:\n{tb}")
        TASK_STORE[task_id]["status"] = "failed"
        TASK_STORE[task_id]["error"] = str(exc)
        TASK_STORE[task_id]["traceback"] = tb



# ---------------------------------------------------------------------------
# CSV parsing helper
# ---------------------------------------------------------------------------
@app.post('/api/delay/parse')
async def parse_delay_ids(request: Request):
    form=await request.form();upload=form.get('delay_file')
    if upload is None:
        raise HTTPException(status_code=422,detail='Upload a travel-time CSV file.')
    try:
        rows=_parse_csv_content((await upload.read()).decode('utf-8-sig'),'delay')
        column,counts=detected_beacon_ids(rows)
        return dict(status='ok',beacon_ids=sorted(counts),counts=counts,n_observations=len(rows),has_beacon_column=bool(column))
    except (ValueError,KeyError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc

def _parse_csv_content(csv_text: str, file_type: str):
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
    except Exception as exc:
        raise ValueError(f"Failed to read {file_type} CSV: {str(exc)}")

    if not rows:
        print(f"[REAL] WARNING — {file_type} CSV appears to be empty or header-only.")
        return []

    parsed = []
    for row in rows:
        clean = {}
        for key, val in row.items():
            key_stripped = key.strip() if key else key
            val_stripped = val.strip() if isinstance(val, str) else val
            try:
                clean[key_stripped] = float(val_stripped)
                if clean[key_stripped] == int(clean[key_stripped]):
                    clean[key_stripped] = int(clean[key_stripped])
            except (ValueError, TypeError):
                clean[key_stripped] = val_stripped
        parsed.append(clean)

    print(f"[REAL] Parsed {file_type} CSV: {len(parsed)} rows, columns={list(parsed[0].keys()) if parsed else '[]'}")
    return parsed




# =====================================================================
# POST  /api/gnss/parse  — Parse GNSS file and return trajectory preview
# =====================================================================
@app.post("/api/gnss/parse")
async def api_gnss_parse(request: Request):
    """
    Upload a GNSS CSV file, parse lat/lon/alt, convert to ENU trajectory.
    Also detects IMU columns (gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z,
    pitch, roll, heading) and returns their availability status.

    Returns:
      - trajectory:       [{x, y}]  ENU coordinate
      - trajectory_latlon: [{lat, lon}]  经Latitude
      - ref_lat, ref_lon: Reference origin
      - n_points:         数据点数
      - has_imu:          是否包含IMU传感器数据
      - imu_columns:      检测到的IMU相关列名列表
      - gnss_columns:     检测到的GNSS列名列表
    """
    try:
        form = await request.form()
        gnss_file = form.get("gnss_file")
        if gnss_file is None:
            raise HTTPException(status_code=400, detail="'gnss_file' is required")

        raw = await gnss_file.read()
        content = raw.decode("utf-8")
        gnss_file.file.seek(0)

        rows = _parse_csv_content(content, "gnss")
        if not rows:
            return JSONResponse(content={
                "status": "ok", "trajectory": [], "trajectory_latlon": [],
                "ref_lat": 0, "ref_lon": 0, "n_points": 0,
                "has_imu": False, "imu_columns": [], "gnss_columns": [],
            })

        lat_col = "Latitude_deg" if any("Latitude_deg" in r for r in rows) else "lat"
        lon_col = "Longitude_deg" if any("Longitude_deg" in r for r in rows) else "lon"

        # --- Detect all available columns ---
        all_columns = list(rows[0].keys())
        columns_lower = {k.lower().strip(): k for k in all_columns}

        # GNSS core columns
        gnss_core = ["latitude_deg", "longitude_deg", "altitude_m", "time_utc",
                     "lat", "lon", "alt", "altitude"]
        gnss_detected = [k for k in all_columns if k.lower().strip() in gnss_core]

        # IMU gyro columns
        imu_gyro = ["gyro_x", "gyro_y", "gyro_z"]
        imu_accel = ["acc_x", "acc_y", "acc_z"]
        imu_att = ["pitch", "roll", "heading"]

        has_gyro = all(c in columns_lower for c in imu_gyro)
        has_accel = all(c in columns_lower for c in imu_accel)
        has_att = all(
            (c in columns_lower) or (c + "_deg" in columns_lower)
            for c in imu_att
        )

        imu_detected = []
        if has_gyro:
            imu_detected.extend([columns_lower[c] for c in imu_gyro])
        if has_accel:
            imu_detected.extend([columns_lower[c] for c in imu_accel])
        if has_att:
            for c in imu_att:
                orig = columns_lower.get(c) or columns_lower.get(c + "_deg")
                if orig:
                    imu_detected.append(orig)

        has_raw_imu = has_gyro and has_accel
        has_any_imu = has_raw_imu or has_att

        # --- Check for non-zero IMU data (not just column headers with zeros) ---
        imu_has_data = False
        if has_raw_imu:
            # Sample first 10 rows to check if any gyro/accel value is non-zero
            for r in rows[:min(10, len(rows))]:
                for c in imu_gyro + imu_accel:
                    orig_key = columns_lower.get(c, c)
                    val = float(r.get(orig_key, 0))
                    if abs(val) > 1e-12:
                        imu_has_data = True
                        break
                if imu_has_data:
                    break
        elif has_att:
            for r in rows[:min(10, len(rows))]:
                for c in imu_att:
                    orig_key = columns_lower.get(c) or columns_lower.get(c + "_deg")
                    if orig_key is None:
                        continue
                    val = float(r.get(orig_key, 0))
                    if abs(val) > 1e-12:
                        imu_has_data = True
                        break
                if imu_has_data:
                    break

        # Use uploaded-track centroid, rounded to 0.1 deg
        lat_col = find_column(rows, ("Latitude_deg", "lat", "latitude")) or lat_col
        lon_col = find_column(rows, ("Longitude_deg", "lon", "longitude")) or lon_col
        lats = column_values(rows, lat_col)
        lons = column_values(rows, lon_col)
        alts = column_values(rows, find_column(rows, ("Altitude_m", "alt", "altitude", "height")), 0.0)
        ref_lat, ref_lon = origin_from_latlon(lats, lons)
        east_preview, north_preview, _ = latlonalt_to_enu(lats, lons, alts, ref_lat, ref_lon)

        n = len(rows)
        trajectory = [{"x": round(float(east_preview[i]), 2), "y": round(float(north_preview[i]), 2)} for i in range(n)]
        trajectory_latlon = [
            {"lat": round(float(lats[i]), 8), "lon": round(float(lons[i]), 8)} for i in range(n)
        ]
        trajectory = mark_track_gaps(trajectory, uploaded_times(rows))
        trajectory_latlon = copy_gap_metadata(trajectory_latlon, trajectory)

        return JSONResponse(content={
            "status": "ok",
            "trajectory": trajectory,
            "trajectory_latlon": trajectory_latlon,
            "ref_lat": round(ref_lat, 8),
            "ref_lon": round(ref_lon, 8),
            "n_points": n,
            "has_imu": has_any_imu and imu_has_data,
            "has_raw_imu": has_raw_imu and imu_has_data,
            "has_attitude": has_att,
            "imu_columns": imu_detected,
            "gnss_columns": gnss_detected,
            "all_columns": all_columns,
        })
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[ERROR] /api/gnss/parse failed:\n{tb}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(exc), "traceback": tb},
        )


# =====================================================================
# POST  /api/ssp/parse  — Parse SSP file unchanged and return profile data
# =====================================================================
@app.post("/api/ssp/parse")
async def api_ssp_parse(request: Request):
    """Parse uploaded SSP values unchanged; no smoothing or decimation."""
    try:
        form = await request.form()
        svp_file = form.get("svp_file")
        if svp_file is None:
            raise HTTPException(status_code=400, detail="'svp_file' is required")

        file_content = (await svp_file.read()).decode("utf-8")
        # Rewind for potential later re-reads
        svp_file.file.seek(0)

        result = load_and_parse_ssp(file_content)

        return JSONResponse(content={**result, "status": "ok"})
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[ERROR] /api/ssp/parse failed:\n{tb}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(exc), "traceback": tb},
        )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running!", "service": "SeaArrayCalib API"}


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
