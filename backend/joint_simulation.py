"""N-beacon complete-tour simulation and joint MAP estimation.

One-way equivalent observations, serial acoustic occupancy, a shared depth mode
with OU amplitude and affine gradients. No coordinate truth enters the objective.
"""
import hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research_bridge import ResearchRayTracer, RESEARCH_ROOT
from closed_survey import SurveyTour, FAMILIES
from risk_design.closed_tours import scale_grid, refinements, VERSION
from munk_ssp import generate_munk_ssp_table

def number(body, key, default, low, high):
    value = float(body.get(key, default))
    if not np.isfinite(value) or value < low or value > high:
        raise ValueError(f'{key} must be between {low} and {high}.')
    return value

def configuration(body):
    if not isinstance(body, dict):
        raise ValueError('Request must be an object containing array and simulation parameters.')
    b = np.asarray(body.get('array_points', []), dtype=float)
    if b.ndim != 2 or b.shape[1] != 3 or not 1 <= len(b) <= 64 or not np.isfinite(b).all():
        raise ValueError('Enter 1 to 64 finite ENU beacon coordinate triples.')
    lever = np.asarray(body.get('lever_arm', [1.585, 6.716, -29.05]), float)
    if lever.shape != (3,) or not np.isfinite(lever).all() or not -1000 <= lever[2] < 0:
        raise ValueError('The lever arm requires three finite values and a negative Up component.')
    if np.any(b[:, 2] >= lever[2]-10) or np.any(b[:, 2] < -12000):
        raise ValueError('Beacons must be at least 10 m below the transducer and no deeper than 12000 m.')
    modes = body.get('modes', list(FAMILIES))
    if not isinstance(modes, list) or not modes or any(m not in FAMILIES for m in modes):
        raise ValueError('Select complete survey families: circle, figure_eight, clover, square, rose, circle_cross. Circle plus cross includes its straight transfer in full duration.')
    if 'rotation_divisions' in body:
        divisions=number(body,'rotation_divisions',1,1,180)
        if isinstance(body['rotation_divisions'],bool) or divisions != int(divisions):
            raise ValueError('180-degree divisions must be an integer from 1 to 180.')
        if 'angles_deg' in body:
            raise ValueError('Specify rotation_divisions or angles_deg, not both.')
        angles=np.arange(int(divisions),dtype=float)*(180./divisions)
    else:
        angles = np.asarray(body.get('angles_deg', [0.]), float)
    if 'scale_min_ratio' in body or 'scale_max_ratio' in body:
        if 'scale_min_ratio' not in body or 'scale_max_ratio' not in body:
            raise ValueError('Enter both scale limits.')
        low_ratio=number(body,'scale_min_ratio',.5,.01,100.)
        high_ratio=number(body,'scale_max_ratio',10.,.01,100.)
        count=number(body,'scale_count',8,1,12)
        if count!=int(count) or low_ratio>high_ratio or (low_ratio<high_ratio and count<2):
            raise ValueError('Use ordered scale limits and at least two seed scales for a nonzero range.')
        ratios=np.geomspace(low_ratio,high_ratio,int(count))
    else:
        ratios = np.asarray(body.get('size_ratios', np.geomspace(.5, 10., 8)), float)
    if angles.ndim != 1 or not 1 <= len(angles) <= 180 or not np.isfinite(angles).all():
        raise ValueError('Invalid orientation configuration.')
    if ratios.ndim != 1 or not 1 <= len(ratios) <= 12 or not np.isfinite(ratios).all() or np.any((ratios <= 0) | (ratios > 100)):
        raise ValueError('Scales must be positive and at most 100 times mean depth.')
    rounds=number(body,'refinement_rounds',1,0,2)
    if rounds!=int(rounds):raise ValueError('Refinement rounds must be 0, 1 or 2.')
    c = dict(beacons=b, lever=lever, modes=list(dict.fromkeys(modes)), angles=np.unique(angles % 360), ratios=np.unique(ratios), refinement_rounds=int(rounds))
    c.update(speed=number(body, 'ship_speed_kn', 3, .1, 30)*.514444,
             interval=1/number(body, 'sample_freq_hz', .05, .001, 10),
             sw=number(body, 'noise', .0001, 0., .1),
             gnss=number(body, 'gnss_noise_std', .05, 0., 10),
             budget=number(body, 'time_budget_h', 7, .1, 48)*3600,
             max_range=number(body, 'max_range_km', 10, .1, 200)*1000,
             dropout=number(body, 'dropout_probability', 0, 0., .95),
             sa=number(body, 'temporal_sigma_mps', 0.3, 0., 10),
             tau=number(body, 'correlation_time_s', 1800., 1., 86400.),
             ge=number(body, 'sv_grad_e', 0., -.01, .01)*1000,
             gn=number(body, 'sv_grad_n', .00004, -.01, .01)*1000,
             seed=int(number(body, 'rng_seed', 20261002, 0, 2**32-1)),
             gdop=number(body, 'gdop_threshold', 2.5, .01, 100),
             criterion=body.get('criterion', 'c3'))
    if c['criterion'] not in ('c3', 'c1'):
        raise ValueError('The criterion must be c1 or c3.')
    if body.get('use_ray_tracing', True) is not True:
        raise ValueError('Joint simulation requires ray tracing.')
    if 'time_budget_min_h' in body or 'time_budget_max_h' in body:
        if 'time_budget_min_h' not in body or 'time_budget_max_h' not in body:
            raise ValueError('Enter both minimum and maximum time budgets.')
        low = number(body, 'time_budget_min_h', 2., .1, 48.)
        high = number(body, 'time_budget_max_h', 7., .1, 48.)
        if low >= high:
            raise ValueError('Minimum time budget must be less than maximum time budget.')
        c['budgets_h'] = np.geomspace(low, high, 10)
        c['budgets_h'][[0, -1]] = [low, high]
        c['budget'] = high*3600
    else:
        # Older clients can still request a single time cap.
        c['budgets_h'] = np.array([c['budget']/3600])
    if not isinstance(body.get('collect_transfer', False), bool):
        raise ValueError('collect_transfer must be true or false.')
    c['collect_transfer'] = body.get('collect_transfer', False)
    c['center'] = b[:, :2].mean(axis=0)
    return c

def ou_sample(times, rng, sigma, tau):
    values = np.empty(len(times)); values[0] = rng.normal()*sigma
    for i in range(1, len(times)):
        r = np.exp(-(times[i]-times[i-1])/tau)
        values[i] = r*values[i-1]+sigma*np.sqrt(-np.expm1(-2*(times[i]-times[i-1])/tau))*rng.normal()
    return values

def ou_product(times, values, tau):
    """O(n*p) exponential covariance product; never materialize n by n."""
    out = values.copy(); forward = np.zeros(values.shape[1]); backward = forward.copy()
    for i in range(1, len(times)):
        forward = np.exp(-(times[i]-times[i-1])/tau)*(forward+values[i-1])
        out[i] += forward
    for i in range(len(times)-2, -1, -1):
        backward = np.exp(-(times[i+1]-times[i])/tau)*(backward+values[i+1])
        out[i] += backward
    return out

def nominal_rows(tracer, ship, ids, beacons, times, theta=None):
    n = len(beacons); p = 4*n+2
    if theta is None:
        theta = np.r_[beacons.ravel(), np.zeros(n+2)]
    xyz = theta[:3*n].reshape(n, 3); a0 = theta[3*n]
    q = (times-np.median(times))/3600
    y = np.zeros(len(times)); j = np.zeros((len(times), p))
    for b in range(n):
        mask = ids == b
        if not mask.any():
            continue
        travel, jac = tracer.varying_offset(ship[mask], xyz[b], a0)
        y[mask] = travel + theta[3*n+1] + theta[3*n+2]*q[mask]
        if b:
            y[mask] += theta[3*n+2+b]
            j[mask, 3*n+2+b] = 1.
        j[mask, 3*b:3*b+3] = jac[:, :3]
        j[mask, 3*n] = jac[:, 3]
    j[:, 3*n+1] = 1.; j[:, 3*n+2] = q
    return y, j

def schedule(tour, c, tracer):
    maxdepth = float(np.max(-c['beacons'][:, 2]))
    max_horizontal = tour.radius*1.001 + np.linalg.norm(c['beacons'][:, :2]-c['center'], axis=1).max() + np.linalg.norm(c['lever'][:2])
    # On a monotone ray, arc length <= horizontal extent + vertical extent.
    # Use the smaller of requested coverage and geometric extent; longer
    # device range alone must not reduce the rate on an unchanged short tour.
    occupancy = 2*(min(c['max_range'], max_horizontal)+maxdepth)/float(np.min(tracer.V)) + 1.
    slot = max(c['interval'], occupancy)
    times = np.arange(0., tour.length/c['speed']-slot+1e-9, slot)
    return slot, times

def generate_observations(tour, c, tracer):
    b = c['beacons']; duration = tour.length/c['speed']; n = len(b)
    # Conservative full-path occupancy bound. Boundary/sound-path failures are
    # tested per observation below; no straight-line replacement is allowed.
    slot, times = schedule(tour, c, tracer)
    if len(times) > 12000:
        raise ValueError('This candidate exceeds 12000 response slots. Increase the interval or reduce duration. The route was not truncated.')
    if len(times) < n*8:
        raise ValueError('Too few response slots: each beacon needs at least 8 valid observations.')
    ids = np.arange(len(times)) % n
    antenna, ship = tour.vessel(times, c['speed'], c['lever'])
    acquisition = (np.ones(len(times), dtype=bool) if c['collect_transfer'] or not hasattr(tour.core, 'acquisition_mask') else
                   tour.core.acquisition_mask(times*c['speed'], (times+slot)*c['speed']))
    range_supported = np.linalg.norm(ship-b[ids], axis=1) <= c['max_range']
    supported = acquisition & range_supported
    skipped_transfer = int((~acquisition).sum())
    rejected_range = int((acquisition & ~range_supported).sum()); ray_fail = 0
    travel = np.full(len(times), np.nan); mapping = np.zeros((len(times), 3))
    for index in range(n):
        ix = np.flatnonzero((ids == index) & supported)
        if not len(ix):
            continue
        try:
            travel[ix], _ = tracer.varying_offset(ship[ix], b[index], 0.)
            mapping[ix] = tracer.mapping(ship[ix], b[index], c['center'])
        except ValueError:
            for k in ix:
                try:
                    travel[k] = tracer.varying_offset(ship[k:k+1], b[index], 0.)[0][0]
                    mapping[k] = tracer.mapping(ship[k:k+1], b[index], c['center'])[0]
                except ValueError:
                    supported[k] = False; ray_fail += 1
    token = f"{getattr(tour,'version','legacy')}:{tour.family}:{tour.radius:.8f}:{tour.angle:.8f}".encode()
    identity = int.from_bytes(hashlib.sha256(token).digest()[:4], 'little')
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence([c['seed'], identity]).spawn(4)]
    drop = streams[0].random(len(times)) < c['dropout']
    rejected_drop = int((supported & drop).sum()); supported &= ~drop
    counts = np.bincount(ids[supported], minlength=n)
    if np.any(counts < 8):
        raise ValueError(f'Insufficient coverage: per-beacon counts {counts.tolist()}. Adjust range, route or sampling.')
    temporal = (np.interp(times, c['_field_times'], c['_field_values']) if '_field_times' in c
                else ou_sample(times, streams[1], c['sa'], c['tau']))
    field = mapping[:, 0]*temporal + mapping[:, 1]*c['ge'] + mapping[:, 2]*c['gn']
    noise = streams[2].normal(0., c['sw'], len(times))
    observed_ship = ship + streams[3].normal(0., c['gnss'], ship.shape)
    if np.any(observed_ship[supported, 2] >= 0):
        raise ValueError('GNSS noise placed the transducer above sea level. Check lever arm and noise.')
    ix = np.flatnonzero(supported)
    return dict(times=times[ix], ids=ids[ix], ship=observed_ship[ix],
                true_ship=ship[ix], antenna=antenna[ix], delays=(travel+field+noise)[ix],
                travel=travel[ix], field=field[ix], h=mapping[ix], slot_s=slot,
                per_beacon_counts=counts.tolist(), attempted=len(times),
                rejected_range=rejected_range, rejected_ray=ray_fail, rejected_dropout=rejected_drop,
                skipped_transfer=skipped_transfer, slot_indices=ix)

def design_metrics(data,c,tracer):
    b=c['beacons'];n=len(b);p=4*n+2
    sw=max(np.hypot(c['sw'],c['gnss']/1500.),1e-7)
    scales=np.r_[np.full(3*n,100.),1.,.001,.001,np.full(n-1,.001)]
    prior=np.zeros((n+2,p));prior[:,3*n:]=np.diag(1/np.r_[2.,.005,.001,np.full(n-1,.005)])
    # Design evaluation uses nominal geometry and fixed noise, not fitted residuals.
    _, j0 = nominal_rows(tracer, data['true_ship'], data['ids'], b, data['times'])
    aug = np.vstack([j0/sw, prior])*scales
    _, singular, vt = np.linalg.svd(aug, full_matrices=False)
    if len(singular) < p or singular[-1] <= singular[0]*1e-10:
        raise ValueError('The joint parameters are not identifiable; finite coordinate risk is unavailable.')
    cov = scales[:, None]*((vt.T/singular**2)@vt)*scales[None, :]
    gain = cov[:3*n]@j0.T/sw**2
    c1 = float(np.trace(cov[:3*n, :3*n])/(3*n))
    a = (gain*data['h'][:, 0]).T
    bias_cov = c['sa']**2 * (a.T@ou_product(data['times'], a, c['tau']))
    g = gain@data['h'][:, 1:]
    bias_cov += (g*np.array([c['ge']**2, c['gn']**2]))@g.T
    worst = float(max(0., np.linalg.eigvalsh((bias_cov+bias_cov.T)/2)[-1])/(3*n))
    gdops = []
    for i in range(n):
        rows = j0[data['ids'] == i, 3*i:3*i+3]*1500
        sv = np.linalg.svd(rows, compute_uv=False)
        gdops.append(float(np.sqrt(np.sum(1/sv**2))) if len(sv)==3 and sv[-1]>sv[0]*1e-10 else float('inf'))
    return dict(c1_m2=c1,c3_m2=c1+worst,worst_bias_m2=worst,gdop=max(gdops),condition_number=float(singular[0]/singular[-1]))


def joint_fit(data, c, tracer, *, simulation_truth=True):
    b = c['beacons']; n = len(b); p = 4*n+2
    sw = max(np.hypot(c['sw'], c['gnss']/1500.), 1e-7)
    scales = np.r_[np.full(3*n, 100.), 1., .001, .001, np.full(n-1, .001)]
    # Nuisance priors match the current study's structure; coordinates are free.
    sigma_prior = np.r_[2., .005, .001, np.full(n-1, .005)]
    prior = np.zeros((n+2, p)); prior[:, 3*n:] = np.diag(1/sigma_prior)
    theta0 = np.r_[b.ravel(), np.zeros(n+2)]
    # Start from a reproducible displaced survey approximation, never fix true depths.
    if simulation_truth:
        theta0[:3*n] += np.tile([15., -12., 8.], n)
    cache = {}
    def evaluate(z):
        if 'z' not in cache or not np.array_equal(z, cache['z']):
            theta = z*scales
            y, j = nominal_rows(tracer, data['ship'], data['ids'], b, data['times'], theta)
            cache.update(z=z.copy(), residual=np.r_[(y-data['delays'])/sw, prior@theta],
                         jac=np.vstack([j/sw, prior])*scales)
        return cache
    lower = np.full(p, -np.inf); upper = np.full(p, np.inf)
    lower[np.arange(n)*3+2] = -float(tracer.Z[-1])+.01
    upper[np.arange(n)*3+2] = min(-1., float(data['ship'][:, 2].min())-1.)
    lower[3*n] = -20.; upper[3*n] = 20.
    fit = least_squares(lambda z: evaluate(z)['residual'], theta0/scales,
                        jac=lambda z: evaluate(z)['jac'], bounds=(lower/scales, upper/scales),
                        max_nfev=100 if simulation_truth else 200, ftol=1e-10, xtol=1e-10, gtol=1e-7)
    # Raw cost gradients grow as 1/sigma^2, even at sub-mm numerical closure.
    # Check the remaining Gauss-Newton correction in declared parameter scales:
    # 1e-5 corresponds to 1 mm coordinates / 1e-5 m/s / 1e-8 s delay.
    correction = np.linalg.lstsq(fit.jac, -fit.fun, rcond=1e-12)[0]
    correction_max = float(np.max(np.abs(correction)))
    if not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(correction_max) or correction_max > 1e-5:
        raise ValueError(f'Joint fit failed the remaining-step check: {fit.message}; scaled step {correction_max:.3g}')
    metrics=design_metrics(data,c,tracer) if simulation_truth else {}
    theta = fit.x*scales
    y, final_j = nominal_rows(tracer, data['ship'], data['ids'], b, data['times'], theta)
    residual = data['delays']-y
    est = theta[:3*n].reshape(n, 3); delta = est-b
    errors = np.linalg.norm(delta, axis=1); horizontal = np.linalg.norm(delta[:, :2], axis=1)
    records = []
    for i in range(n):
        r = residual[data['ids']==i]; rms = float(np.sqrt(np.mean(r*r)))
        records.append(dict(id=i, x=float(est[i,0]), y=float(est[i,1]), z=float(est[i,2]),
                            rmse_time_ms=rms*1000, rmse_equiv_m=rms*1500,
                            observations=int((data['ids']==i).sum())))
    if not simulation_truth:
        # Same scaled augmented information matrix and effective residual degrees
        # of freedom as the SeaArrayCalib field diagnostics. No coordinate truth.
        _, singular, vt = np.linalg.svd(fit.jac, full_matrices=False)
        if singular[-1] <= singular[0]*1e-12:
            raise ValueError('The joint augmented Jacobian is not estimable.')
        cov_z = (vt.T/singular**2)@vt
        covariance = scales[:,None]*cov_z*scales[None,:]
        jw = final_j/sw
        information = jw.T@jw
        vh = covariance@information
        dof = float(len(residual)-2*np.trace(vh)+np.trace(vh@vh))
        if not np.isfinite(dof) or dof <= 0:
            raise ValueError('The joint effective residual degrees of freedom are invalid.')
        factor = float(np.sum((residual/sw)**2)/dof)
        internal = covariance*factor
        return dict(estimated_beacons=records,
            internal_axis_sd_m=np.sqrt(np.maximum(0,np.diag(internal)[:3*n])).reshape(n,3).tolist(),
            conditional_covariance=internal.tolist(),
            formal_c1_m2=float(np.trace(covariance[:3*n,:3*n])/(3*n)),
            residual_variance_factor=factor, effective_residual_dof=dof,
            rmse_time_ms_mean=float(np.sqrt(np.mean(residual**2))*1000),
            solver_optimality=float(fit.optimality), solver_remaining_scaled_step=correction_max,
            effective_noise_s=float(sw), joint_parameter_count=p, augmented_rank=p,
            sound_speed_offset_mps=float(theta[3*n]), common_delay_s=float(theta[3*n+1]),
            delay_rate_s_per_hour=float(theta[3*n+2]), time_reference_s=float(np.median(data['times'])),
            relative_delays_s=[0., *theta[3*n+3:].tolist()], residuals_ms=(residual*1000).tolist())
    return dict(estimated_beacons=records, position_errors_m=errors.tolist(),
                horizontal_errors_m=horizontal.tolist(), rmse_horizontal_m=float(np.sqrt(np.mean(horizontal**2))),
                rmse_3d_m=float(np.sqrt(np.mean(errors**2))), rmse_time_ms_mean=float(np.sqrt(np.mean(residual**2))*1000),
                **metrics, solver_optimality=float(fit.optimality),
                solver_remaining_scaled_step=correction_max, effective_noise_s=float(sw),
                joint_parameter_count=p, sound_speed_offset_mps=float(theta[3*n]),
                common_delay_s=float(theta[3*n+1]), delay_rate_s_per_hour=float(theta[3*n+2]),
                relative_delays_s=[0., *theta[3*n+3:].tolist()], residuals_ms=(residual*1000).tolist())

def initial_specs(c):
    depth=float(np.mean(-c['beacons'][:,2]))
    return [(m,r,float(a),source) for m in sorted(c['modes'])
        for r,source in scale_grid(m,depth,c['speed'],c['budgets_h'],c['ratios']) for a in c['angles']]


def run_survey(body, progress=None, export_sink=None):
    c = configuration(body)
    profile = generate_munk_ssp_table(H=float(np.max(-c['beacons'][:,2]))+500, dz=20.)
    if body.get('ssp_profile'):
        import pandas as pd
        profile = pd.DataFrame(body['ssp_profile'])
    tracer = ResearchRayTracer(profile)
    if tracer.Z[-1] < np.max(-c['beacons'][:,2])+100:
        raise ValueError('The SSP must extend at least 100 m below the deepest beacon for free-depth estimation.')
    rows = []; depth = float(np.mean(-c['beacons'][:,2]))
    specs=initial_specs(c)
    # Search depends only on nominal geometry, declared risk and coverage.
    # Freeze the final pool before drawing its shared environmental realization.
    seen=set();round_counts=[]
    for stage in range(c['refinement_rounds']+1):
        pending=[x for x in specs if x[:3] not in seen]
        if len(rows)+len(pending)>2000:
            raise ValueError('Scale search exceeds 2000 candidates. Reduce families, angles, seed scales or refinement rounds; no candidates were silently dropped.')
        round_counts.append(len(pending))
        for index,(mode,ratio,angle,source) in enumerate(pending):
            seen.add((mode,ratio,angle));tour=SurveyTour(mode,depth*ratio,c['center'],angle)
            row=dict(candidate_id=f'{VERSION}:{mode}:{float(ratio):.15g}:{float(angle):.15g}',
                type=mode,label=FAMILIES[mode],ratio=float(ratio),size=float(depth*ratio),size_m=float(depth*ratio),
                angle_deg=float(angle),path_length_m=tour.length,length=tour.length,
                duration_h=tour.length/c['speed']/3600,valid=False,design_valid=False,
                search_source=source,search_round=stage,**tour.core.metadata())
            try:
                if tour.length/c['speed']>c['budget']+1e-7:raise ValueError('The complete route exceeds the time budget.')
                d=generate_observations(tour,c,tracer)
                row.update(design_metrics(d,c,tracer));row['design_valid']=True
            except (ValueError,np.linalg.LinAlgError) as error:row['reason']=str(error)
            rows.append(row)
            if progress:progress(index+1,len(pending),dict(row,phase='search',stage=stage,rounds=c['refinement_rounds']+1))
        specs=refinements(rows,c['budgets_h'],c['gdop']) if stage<c['refinement_rounds'] else []
    epochs=[]
    for row in rows:
        if row['design_valid']:
            epochs.append(schedule(SurveyTour(row['type'],row['size'],c['center'],row['angle_deg']),c,tracer)[1])
    if epochs:
        c['_field_times']=np.unique(np.concatenate(epochs))
        c['_field_values']=ou_sample(c['_field_times'],np.random.default_rng(c['seed']),c['sa'],c['tau'])
    for index,row in enumerate(rows):
        tour=SurveyTour(row['type'],row['size'],c['center'],row['angle_deg'])
        row['trajectory']=tour.preview()
        if row['design_valid']:
            try:
                d=generate_observations(tour,c,tracer)
                metrics=joint_fit(d,c,tracer)
                for key in ('c1_m2','c3_m2','gdop'):
                    if not np.isclose(metrics[key],row[key],rtol=1e-10,atol=1e-10):raise ValueError('Design changed after the search pool was frozen.')
                row.update(metrics)
                row.update(valid=True, observations_count=len(d['times']), attempted_slots=d['attempted'],
                    slot_s=d['slot_s'], per_beacon_counts=d['per_beacon_counts'],
                    rejected_range=d['rejected_range'], rejected_ray=d['rejected_ray'],
                    rejected_dropout=d['rejected_dropout'], skipped_transfer=d['skipped_transfer'], collect_transfer=c['collect_transfer'],rmse=row['rmse_horizontal_m'],
                    rmse_position_m=row['rmse_horizontal_m'],solver_type='joint_map')
                row['_observations']=[dict(time_s=float(d['times'][k]),beacon_id=int(d['ids'][k])+1,
                    e_m=float(d['ship'][k,0]),n_m=float(d['ship'][k,1]),u_m=float(d['ship'][k,2]),
                    delay_s=float(d['delays'][k]),true_delay_s=float(d['travel'][k]),field_delay_s=float(d['field'][k])) for k in range(len(d['times']))]
            except (ValueError,np.linalg.LinAlgError) as error:row['reason']=str(error)
        if progress:progress(index+1,len(rows),dict(row,phase='solve'))
    if export_sink is not None:
        export_sink(rows)
    score = c['criterion']+'_m2'
    valid = [r for r in rows if r['valid'] and np.isfinite(r[score])]
    eligible = [r for r in valid if r['gdop'] <= c['gdop']]
    def choose(candidates):
        a = b = balanced = None
        if candidates:
            a = min(candidates, key=lambda r: (r[score], r['duration_h']))
            median = float(np.median([r[score] for r in candidates]))
            b = min([r for r in candidates if r[score] <= median], key=lambda r: r['duration_h'])
            scores = np.array([r[score] for r in candidates]); durations = np.array([r['duration_h'] for r in candidates])
            cost = .5*(scores-scores.min())/max(float(np.ptp(scores)),1e-15)+.5*(durations-durations.min())/max(float(np.ptp(durations)),1e-15)
            balanced = candidates[int(np.argmin(cost))]
        return dict(zip(('plan_a','plan_b','plan_c'), (a,b,balanced)))

    plans_by_id = {}; budget_results = []
    for budget_h in c['budgets_h']:
        within = [r for r in valid if r['duration_h'] <= budget_h+1e-10]
        feasible = [r for r in within if r['gdop'] <= c['gdop']]
        chosen = choose(feasible)
        item = dict(budget_h=float(budget_h), n_valid=len(within), n_eligible=len(feasible),
                    message='Complete' if feasible else 'No complete eligible route within this time budget.')
        for key, row in chosen.items():
            item[key] = None if row is None else row['candidate_id']
            if row is not None and row['candidate_id'] not in plans_by_id:
                plans_by_id[row['candidate_id']] = {k:v for k,v in row.items() if not k.startswith('_')}
                plans_by_id[row['candidate_id']]['observations'] = row['_observations']
        budget_results.append(item)
    # Retain the old top-level contract as the largest budget's choices.
    plans = {key:plans_by_id.get(budget_results[-1][key]) for key in ('plan_a','plan_b','plan_c')}
    return dict(**plans, budget_values_h=c['budgets_h'].tolist(), budget_results=budget_results, plans_by_id=plans_by_id, all_results=[{k:v for k,v in r.items() if k not in ('trajectory','residuals_ms','_observations')} for r in rows],
                provenance={str(p.relative_to(RESEARCH_ROOT)) if p.is_relative_to(RESEARCH_ROOT) else p.name:
                            hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [Path(__file__),Path(__file__).with_name('research_bridge.py'),Path(__file__).with_name('closed_survey.py'),
                             *[RESEARCH_ROOT/'research/src/risk_design'/name for name in
                               ['trajectories.py','closed_tours.py','fast_precision.py','profile_gradient.py']]]},
                plan_trajectories={k:p['trajectory'] for k,p in plans.items() if p},
                n_total=len(rows), n_valid=len(valid), n_eligible=len(eligible), n_beacons=len(c['beacons']),
                criterion=c['criterion'], rng_seed=c['seed'], config=body,
                search=dict(route_version=VERSION,scale_bounds_ratio=[float(c['ratios'][0]),float(c['ratios'][-1])],mean_depth_m=depth,round_counts=round_counts,refinement_rounds=c['refinement_rounds'],pool_frozen_before_shared_field=True),
                message='Complete' if eligible else 'No candidate meets coverage, identifiability, duration and GDOP requirements. Review the candidate reasons.',
                model_notes=['User minimum and maximum budgets define ten inclusive logarithmically spaced caps. Each complete candidate is simulated once; all caps select from these same results without truncating routes or rerandomizing observations.',
                  'Five families use one closed traversal. Circle plus cross runs the circle, first diameter, straight transfer and perpendicular diameter, ending without a forced return. Transfer travel counts in duration; transfer sampling is off by default. No extra laps are added. Transit from an unspecified port is excluded.',
                  'User scale bounds are strict. Search adds in-range budget-boundary scales and log-midpoints near both C1 and C3 winners. It is a finite refined search, not a global continuous optimum. Circle angle controls starting phase; other family angles control orientation.',
                  'Serial polling addresses one beacon per slot. Slots use the larger of the requested interval and conservative two-way acoustic occupancy.',
                  'Layered Snell rays with perturbations integrated along the reference ray: exp(-depth/1000 m), shared OU amplitude and fixed horizontal gradients.',
                  'Jointly estimates all beacon coordinates, common sound-speed offset, delay, linear drift and relative beacon delays. No coordinate truth priors.',
                  'C1/C3 are declared local risks with C3 radius 1. Position errors are from one simulation realisation, not repeated-trial MSE.',
                  'Nuisance prior SDs: sound offset 2 m/s, common/relative delays 5 ms, drift 1 ms/hour. Effective timing noise has a 0.1 microsecond numerical floor. C1 includes prior uncertainty.',
                  'Range is an assumed slant-range cutoff, without a device link budget, separate transmit/receive positions or full 3D acoustics. Scoring approximates GNSS errors as independent time noise.'])
