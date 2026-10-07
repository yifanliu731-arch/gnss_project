"""Observed N-beacon joint calibration, using the simulation's shared estimator.

Coordinates have no truth or depth priors. Report internal precision only.
"""
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from joint_simulation import joint_fit
from research_bridge import ResearchRayTracer
from ray_tracing import normalize_ssp_columns, ssp_equivalent_speed


def initial_position(ship, delays, sound_speed, model_depth):
    ranges=np.asarray(delays)*sound_speed
    center=np.mean(ship[:,:2],axis=0)
    xy=ship[:,:2]-center
    target=ranges**2-np.sum(xy**2,axis=1)
    a=np.c_[-2*xy,np.ones(len(xy))]
    solution=np.linalg.lstsq(a,target,rcond=1e-10)[0]
    horizontal=center+solution[:2]
    vertical=np.sqrt(max(100.,float(np.median(ranges**2-np.sum((ship[:,:2]-horizontal)**2,axis=1)))))
    guess=np.r_[horizontal,np.median(ship[:,2])-vertical]
    lower=np.array([-np.inf,-np.inf,-model_depth+.02])
    upper=np.array([np.inf,np.inf,min(-1.,float(ship[:,2].min())-1.)])
    guess=np.clip(guess,lower+1e-4,upper-1e-4)
    fit=least_squares(lambda b:np.linalg.norm(ship-b,axis=1)-ranges,guess,
                      bounds=(lower,upper),loss='soft_l1',f_scale=5.,max_nfev=100)
    return fit.x


def solve_field_joint(groups, profile, noise_s=.0005, gnss_noise_m=.05, progress=None):
    if not groups:
        raise ValueError('Select at least one beacon present in the uploaded observations.')
    noise_s=float(noise_s);gnss_noise_m=float(gnss_noise_m)
    if not np.isfinite(noise_s) or noise_s<=0 or not np.isfinite(gnss_noise_m) or gnss_noise_m<0:
        raise ValueError('Joint timing noise must be positive and GNSS noise nonnegative.')
    profile=normalize_ssp_columns(profile).dropna().sort_values('Depth (m)')
    measured_depth=float(profile['Depth (m)'].max())
    model_depth=max(measured_depth*1.5,6000.)
    # Preserve every measured profile value. Declare only the appended extension.
    extra=pd.DataFrame({'Depth (m)':np.linspace(measured_depth+100,model_depth,10),
                        'SV (m/s)':profile['SV (m/s)'].iloc[-1]+.016*(np.linspace(measured_depth+100,model_depth,10)-measured_depth)})
    tracer=ResearchRayTracer(pd.concat([profile,extra],ignore_index=True))
    c_eq=float(ssp_equivalent_speed(profile))
    # Match the existing field convention when B5 is available.
    reference=5 if 5 in groups else sorted(groups)[0]
    beacon_ids=[reference]+[bid for bid in sorted(groups) if bid!=reference]
    blocks=[];initial=[]
    initialized_at=perf_counter()
    for index,bid in enumerate(beacon_ids):
        g=groups[bid]
        ship=np.asarray(g['ship'],float);times=np.asarray(g['times'],float);delays=np.asarray(g['delays'],float)
        if len(delays)<8 or ship.shape!=(len(delays),3) or times.shape!=delays.shape:
            raise ValueError(f'Beacon {bid} needs at least 8 time-aligned valid observations.')
        if not np.isfinite(ship).all() or not np.isfinite(times).all() or not np.isfinite(delays).all() or np.any(delays<=0):
            raise ValueError(f'Beacon {bid} contains invalid aligned observations.')
        if np.any(ship[:,2]>=0):
            raise ValueError('The joint refracted-ray model requires the transducer below sea level. Check GNSS altitude and lever arm.')
        initial.append(initial_position(ship,delays,c_eq,model_depth))
        blocks.append(dict(ship=ship,times=times,delays=delays,ids=np.full(len(delays),index,dtype=int)))
    data={key:np.concatenate([b[key] for b in blocks]) for key in ('ship','times','delays','ids')}
    order=np.argsort(data['times'],kind='stable')
    data={key:value[order] for key,value in data.items()}
    if len(data['times'])<=4*len(beacon_ids)+2:
        raise ValueError('There are too few observations for the joint parameter model.')
    if progress:progress(40,'Jointly fitting all selected beacons')
    fit_started=perf_counter()
    fitted=joint_fit(data,dict(beacons=np.asarray(initial),sw=noise_s,gnss=gnss_noise_m),tracer,simulation_truth=False)
    fit_seconds=perf_counter()-fit_started
    records=[];export_rows=[]
    for index,bid in enumerate(beacon_ids):
        mask=data['ids']==index
        g=groups[bid];r=fitted['estimated_beacons'][index]
        # Recover the original order within this beacon for UTC and residual export.
        residual=np.asarray(fitted['residuals_ms'])[mask]
        local_order=np.argsort(g['times'],kind='stable')
        residual_original=np.empty_like(residual);residual_original[local_order]=residual
        sigma=fitted['internal_axis_sd_m'][index]
        entry=dict(beacon_id=bid,status='success',x=r['x'],y=r['y'],z=r['z'],
                   rmse_time_ms=r['rmse_time_ms'],rmse_distance_m=r['rmse_time_ms']/1000*c_eq,
                   n_total=len(g['times']),n_clean=len(g['times']),outlier_rate='0.0%',solver_type='joint_map',
                   sigma_e_m=sigma[0],sigma_n_m=sigma[1],sigma_u_m=sigma[2],c_eq_mps=c_eq,
                   relative_delay_ms=fitted['relative_delays_s'][index]*1000)
        records.append(entry)
        observations=[dict(observation_index=i+1,beacon_id=bid,time_s=float(g['times'][i]),source_time=g['source_times'][i],
                           delay_s=float(g['delays'][i]),e_m=float(p[0]),n_m=float(p[1]),u_m=float(p[2])) for i,p in enumerate(g['ship'])]
        export_rows.append(dict(entry,trajectory=[dict(x=float(p[0]),y=float(p[1]),z=float(p[2])) for p in g['ship']],
                                observations=observations,residuals_ms=residual_original.tolist(),
                                residuals_m=(residual_original/1000*c_eq).tolist(),outlier_mask=[False]*len(g['times'])))
    records.sort(key=lambda r:r['beacon_id']);export_rows.sort(key=lambda r:r['beacon_id'])
    summary={k:v for k,v in fitted.items() if k not in ('estimated_beacons','internal_axis_sd_m','residuals_ms')}
    summary.update(initialization_seconds=fit_started-initialized_at,fit_seconds=fit_seconds,measured_ssp_points=len(profile),reference_beacon_id=reference,beacon_ids=beacon_ids,n_observations=len(data['times']),
                   nominal_timing_noise_ms=noise_s*1000,gnss_noise_m=gnss_noise_m,
                   measured_ssp_max_depth_m=measured_depth,model_ssp_max_depth_m=model_depth,
                   deep_extension_gradient_mps_per_m=.016,
                   noise_priors=dict(sound_offset_mps=2.,common_delay_s=.005,drift_s_per_hour=.001,relative_delay_s=.005),
                   interpretation='Joint free-coordinate MAP estimate. Standard deviations are residual-scaled conditional internal precision; no coordinate truth or depth prior is used. Original UTC and all retained observations are preserved. No automatic residual clipping.')
    if progress:progress(96,'Preparing joint estimates and export data')
    return records,export_rows,summary
