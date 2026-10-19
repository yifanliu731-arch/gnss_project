"""Train-only route selection followed by exact held-out profile ray tracing."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import joint_simulation as js
from closed_survey import SurveyTour, FAMILIES, VERSION
from research_bridge import ResearchRayTracer, RESEARCH_ROOT
from risk_design.closed_tours import refinements
from risk_design.safe_monotone import solve_safe, safe_varying_offset
from risk_design.precision import _stable_shoot
from risk_design.profile_ensemble import profile_mapping
from profile_data import load_profiles


class ProfileTracer(ResearchRayTracer):
    def varying_offset(self, ship, beacon, offsets):
        return safe_varying_offset(self, ship, beacon, offsets)

    def mapping(self, ship, beacon, center, mode_depth=1000.):
        # This mode uses the complete uploaded training ensemble below.
        # The analytic OU/gradient channels in the legacy generator are off.
        return np.zeros((len(ship), 3))


def tracer_for(depth, speed):
    return ProfileTracer(pd.DataFrame({'Depth (m)':depth, 'SV (m/s)':speed}))


def profile_metrics(data, c, tracer, fields):
    metrics, gain = js.design_metrics(data, c, tracer, return_gain=True)
    if len(fields['basis']):
        h = profile_mapping(tracer, data['true_ship'], data['ids'], c['beacons'], fields['depth'], fields['basis'])
        a = gain @ h / np.sqrt(3*len(c['beacons']))
        matrix = a @ a.T
        beta = max(0., float(np.linalg.eigvalsh((matrix+matrix.T)/2)[-1]))
        average = float(np.trace(matrix))
    else:
        beta = average = 0.
    scale = (fields['alpha']*c['rho'])**2
    metrics.update(c2_m2=metrics['c1_m2']+scale*average,
                   c3_m2=metrics['c1_m2']+scale*beta,
                   worst_bias_m2=scale*beta,
                   mean_risk_m2=metrics['c1_m2']+scale*average,
                   unit_c2_m2=metrics['c1_m2']+fields['alpha']**2*average,
                   unit_c3_m2=metrics['c1_m2']+fields['alpha']**2*beta)
    return metrics


def truth_observations(data, c, fields, day, candidate_id):
    index = int(np.flatnonzero(fields['dates']==day)[0])
    tracer = tracer_for(fields['depth'], fields['nominal']+fields['alpha']*fields['delta'][index])
    travel = np.full(len(data['times']), np.nan)
    for b, beacon in enumerate(c['beacons']):
        ix = np.flatnonzero(data['ids']==b)
        if not len(ix):
            continue
        ship = data['true_ship'][ix]; source = -ship[:,2]
        tracer.delta_c = 0.
        z, speed = tracer._get_effective_ssp(float(source.min()), -float(beacon[2]))
        ranges = np.linalg.norm(ship[:,:2]-beacon[:2],axis=1)
        p = solve_safe(ranges, source, z, speed, np.zeros(len(ix)))
        good = np.isfinite(p)
        if good.any():
            x, t, _ = _stable_shoot(p[good], z, speed, source[good], np.zeros(int(good.sum())))
            closed = np.isfinite(t) & (np.abs(x-ranges[good])<=1e-4)
            travel[ix[good][closed]] = t[closed]
    reachable = np.isfinite(travel)
    keep = reachable & (2*travel+1 <= data['slot_s']+1e-8)
    counts = np.bincount(data['ids'][keep], minlength=len(c['beacons']))
    details = dict(planned_observations=len(travel), retained_observations=int(keep.sum()),
                   missing_truth_rays=int((~reachable).sum()), occupancy_rejections=int((reachable & ~keep).sum()),
                   per_beacon_counts=counts.tolist(), duration_unchanged=True)
    # Draw on the full planned set, then mask: no noise redraw after ray loss.
    token = int.from_bytes(hashlib.sha256(candidate_id.encode()).digest()[:4], 'little')
    calendar = int(day.replace('-', ''))
    rng = np.random.default_rng(np.random.SeedSequence([c['seed'], token, calendar]))
    white = rng.normal(0., c['sw'], len(travel))
    ship = data['true_ship']+rng.normal(0., c['gnss'], data['true_ship'].shape)
    if np.any(ship[keep,2]>=0):
        raise ValueError('GNSS noise placed the transducer above sea level. Check lever arm and noise.')
    masked = {k:(v[keep] if isinstance(v,np.ndarray) and v.ndim>0 and len(v)==len(travel) else v) for k,v in data.items()}
    masked.update(ship=ship[keep], delays=(travel+white)[keep], travel=travel[keep],
                  nominal_travel=data['travel'][keep], field=(travel-data['travel'])[keep],
                  per_beacon_counts=counts.tolist())
    return masked, details


def choose_plans(candidates, score):
    return js.choose_plans(candidates, score)


def run_profile_survey(body, progress=None, export_sink=None):
    c = js.configuration(body)
    c.update(sa=0.,ge=0.,gn=0.)
    fields = load_profiles(body.get('sound_field'))
    tracer = tracer_for(fields['depth'], fields['nominal'])
    if tracer.Z[-1] < np.max(-c['beacons'][:,2])+100:
        raise ValueError('The SSP must extend at least 100 m below the deepest beacon for free-depth estimation.')
    rows = []; seen = set(); round_counts = []; depth = float(np.mean(-c['beacons'][:,2]))
    specs = js.initial_specs(c)
    for stage in range(c['refinement_rounds']+1):
        pending = [x for x in specs if x[:3] not in seen]
        if len(rows)+len(pending)>2000:
            raise ValueError('Scale search exceeds 2000 candidates. Reduce families, angles, seed scales or refinement rounds; no candidates were silently dropped.')
        round_counts.append(len(pending))
        for index,(mode,ratio,angle,source) in enumerate(pending):
            seen.add((mode,ratio,angle)); tour = SurveyTour(mode,depth*ratio,c['center'],angle)
            row = dict(candidate_id=f'{VERSION}:{mode}:{float(ratio):.15g}:{float(angle):.15g}',
                       type=mode,label=FAMILIES[mode],ratio=float(ratio),size=float(depth*ratio),size_m=float(depth*ratio),
                       angle_deg=float(angle),path_length_m=tour.length,length=tour.length,
                       duration_h=tour.length/c['speed']/3600,valid=False,design_valid=False,
                       validation_status='not_selected',search_source=source,search_round=stage,**tour.core.metadata())
            try:
                if tour.length/c['speed']>c['budget']+1e-7:
                    raise ValueError('The complete route exceeds the time budget.')
                data = js.generate_observations(tour,c,tracer)
                row.update(profile_metrics(data,c,tracer,fields))
                row.update(valid=True,design_valid=True,observations_count=len(data['times']),
                           slot_s=data['slot_s'],attempted_slots=data['attempted'],per_beacon_counts=data['per_beacon_counts'],
                           rejected_range=data['rejected_range'],rejected_ray=data['rejected_ray'],
                           rejected_dropout=data['rejected_dropout'],skipped_transfer=data['skipped_transfer'])
            except (ValueError,np.linalg.LinAlgError) as exc:
                row['reason'] = str(exc)
            rows.append(row)
            if progress:
                progress(index+1,len(pending),dict(row,phase='search',stage=stage,rounds=c['refinement_rounds']+1))
        specs = js.risk_refinements(rows,c['budgets_h'],c['gdop']) if stage<c['refinement_rounds'] else []
    # Freeze all budget choices using training-only scores, before any test fits.
    valid = [r for r in rows if r['design_valid']]
    eligible = [r for r in valid if r['gdop']<=c['gdop']]
    selected = {}; comparisons = []; budgets = []; comparison_ids = set()
    score = c['criterion']+'_m2'
    for cap in c['budgets_h']:
        within = [r for r in valid if r['duration_h']<=cap+1e-10]
        feasible = [r for r in within if r['gdop']<=c['gdop']]
        item = dict(budget_h=float(cap),n_valid=len(within),n_eligible=len(feasible),
                    message='Complete' if feasible else 'No complete eligible route within this time budget.')
        for name,row in choose_plans(feasible,score).items():
            item[name] = row['candidate_id'] if row else None
            if row:
                selected[row['candidate_id']] = row
        compare = dict(budget_h=float(cap))
        for criterion in ('c1','c2','c3'):
            row = min(feasible,key=lambda r:(r[criterion+'_m2'],r['duration_h'],r['candidate_id'])) if feasible else None
            compare[criterion+'_id'] = row['candidate_id'] if row else None
            if row:
                selected[row['candidate_id']] = row; comparison_ids.add(row['candidate_id'])
        comparisons.append(compare); budgets.append(item)
    frozen = json.dumps(dict(budgets=budgets,comparisons=comparisons),sort_keys=True).encode()
    validation_dates = fields['dates'][fields['test']].tolist() if fields['scope']=='all' else [fields['detail_date']]
    jobs = [(identity,day) for identity in sorted(selected) for day in
            (validation_dates if identity in comparison_ids else [fields['detail_date']])]
    # The detail day is always present for every selected route.
    jobs += [(identity,fields['detail_date']) for identity in sorted(selected) if (identity,fields['detail_date']) not in jobs]
    if len(jobs)>20000:
        raise ValueError('Validation exceeds 20000 route/date fits. Reduce dates or budgets; no cases were silently skipped.')
    trials = []; cache = {}; plans = {}
    for index,(identity,day) in enumerate(jobs):
        row = selected[identity]
        trial = dict(candidate_id=identity,date=day,success=False)
        data = None
        try:
            if identity not in cache:
                tour = SurveyTour(row['type'],row['size'],c['center'],row['angle_deg'])
                cache[identity] = js.generate_observations(tour,c,tracer)
                row['trajectory'] = tour.preview()
            data, details = truth_observations(cache[identity],c,fields,day,identity)
            trial.update(details)
            if min(details['per_beacon_counts'])<8:
                raise ValueError('Too few retained observations: each beacon needs at least 8.')
            retained = js.design_metrics(data,c,tracer)
            trial['retained_gdop'] = retained['gdop']
            if retained['gdop']>c['gdop']:
                raise ValueError('Retained geometry exceeds the GDOP limit.')
            fit = js.joint_fit(data,c,tracer)
            trial.update(success=True,coordinate_mse_m2=fit['rmse_3d_m']**2/3,
                         rmse_3d_m=fit['rmse_3d_m'],residual_rms_ms=fit['rmse_time_ms_mean'],
                         solver_remaining_scaled_step=fit['solver_remaining_scaled_step'])
            if day==fields['detail_date']:
                # Preserve the frozen scores: retained-geometry risk is a diagnostic.
                row.update({k:v for k,v in fit.items() if k not in ('c1_m2','c2_m2','c3_m2','worst_bias_m2','unit_c2_m2','unit_c3_m2','gdop','condition_number')})
                row.update(validation_status='success',validation_date=day,**details,observations_count=len(data['times']),
                           retained_gdop=retained['gdop'],retained_c1_m2=retained['c1_m2'],
                           rmse=fit['rmse_horizontal_m'],rmse_position_m=fit['rmse_horizontal_m'],solver_type='joint_map')
        except (ValueError,np.linalg.LinAlgError) as exc:
            trial['reason'] = str(exc)
            if day==fields['detail_date']:
                row.update(validation_status='failed',validation_date=day,validation_reason=str(exc),
                           **{k:v for k,v in trial.items() if k in ('planned_observations','retained_observations','missing_truth_rays','occupancy_rejections')})
        if day==fields['detail_date']:
            row['_observations'] = [] if data is None else [dict(time_s=float(data['times'][k]),
                beacon_id=int(data['ids'][k])+1,slot_index=int(data['slot_indices'][k]),date=day,
                e_m=float(data['ship'][k,0]),n_m=float(data['ship'][k,1]),u_m=float(data['ship'][k,2]),
                delay_s=float(data['delays'][k]),true_delay_s=float(data['travel'][k]),
                nominal_delay_s=float(data['nominal_travel'][k]),field_delay_s=float(data['field'][k])) for k in range(len(data['times']))]
            plans[identity] = row
        trials.append(trial)
        if progress:
            progress(index+1,len(jobs),dict(row,phase='validate'))
    for comparison in comparisons:
        sets = {}
        for criterion in ('c1','c2','c3'):
            identity = comparison[criterion+'_id']
            subset = [r for r in trials if r['candidate_id']==identity and r['date'] in validation_dates]
            successful = [r for r in subset if r['success']]
            sets[criterion] = {r['date']:r for r in successful}
            route = selected.get(identity, {})
            comparison[criterion] = dict(label=route.get('label'),size_m=route.get('size_m'),duration_h=route.get('duration_h'),
                attempted=len(subset),successful=len(successful),failed=len(subset)-len(successful),
                mean_mse_m2=float(np.mean([r['coordinate_mse_m2'] for r in successful])) if successful else None,
                missing_rays=sum(r.get('missing_truth_rays',0) for r in subset),
                mean_scope='all_requested_dates' if subset and len(successful)==len(subset) else 'successful_dates_only')
        paired = sorted(sets['c1'].keys() & sets['c3'].keys())
        comparison['paired_dates'] = len(paired)
        if paired:
            comparison['paired_c1_mse_m2'] = float(np.mean([sets['c1'][d]['coordinate_mse_m2'] for d in paired]))
            comparison['paired_c3_mse_m2'] = float(np.mean([sets['c3'][d]['coordinate_mse_m2'] for d in paired]))
    source_paths = [Path(__file__),Path(js.__file__),Path(__file__).with_name('profile_data.py'),
                    Path(__file__).with_name('research_bridge.py'),Path(__file__).with_name('closed_survey.py'),
                    *[RESEARCH_ROOT/'research/src/risk_design'/name for name in ('safe_monotone.py','profile_ensemble.py','fast_precision.py','precision.py','closed_tours.py')]]
    provenance = {str(p.relative_to(RESEARCH_ROOT)) if p.is_relative_to(RESEARCH_ROOT) else p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    plans_by_id = {key:{**{k:v for k,v in row.items() if not k.startswith('_')},'observations':row.get('_observations',[])} for key,row in plans.items()}
    if export_sink is not None:
        export_sink(rows)
    top = {key:plans_by_id.get(budgets[-1][key]) for key in ('plan_a','plan_b','plan_c')}
    return dict(**top, budget_values_h=c['budgets_h'].tolist(),budget_results=budgets,plans_by_id=plans_by_id,
                all_results=[{k:v for k,v in row.items() if k not in ('trajectory','residuals_ms','_observations')} for row in rows],
                provenance=provenance,plan_trajectories={k:p.get('trajectory',[]) for k,p in top.items() if p},
                n_total=len(rows),n_valid=len(valid),n_eligible=len(eligible),n_beacons=len(c['beacons']),
                criterion=c['criterion'],rng_seed=c['seed'],risk_radius=c['rho'],config={**body,'risk_radius':c['rho']},
                search=dict(route_version=VERSION,refinement_risk_radius=1.,round_counts=round_counts,refinement_rounds=c['refinement_rounds'],
                            pool_frozen_before_validation=True,selection_sha256=hashlib.sha256(frozen).hexdigest()),
                field_validation=dict(**fields['metadata'],comparisons=comparisons,trials=trials,
                    dates_evaluated=validation_dates,fits=len(trials),failed_fits=sum(not r['success'] for r in trials)),
                message='Complete' if eligible else 'No candidate meets coverage, identifiability, duration and GDOP requirements. Review the candidate reasons.',
                model_notes=[
                    'Profile mode uses background plus amplified historical-minus-background differences. The uploaded background is fixed; only training dates define the uncentered risk model.',
                    'C1, C2 and C3 choices are frozen before validation. Failed tests remain attached to the original choices; validation never substitutes a better-performing route.',
                    'C2 and C3 scale their environmental penalties by the squared risk radius. Keep radius 1 to use the uploaded training scale; changing the radius does not amplify validation profiles. The separate profile multiplier scales both design and validation perturbations.',
                    'Each validation date represents one static layered ocean for the complete survey. Daily profiles do not specify subdaily dynamics or a full three-dimensional field.',
                    'C3 is a local ellipsoid score based on the training second moment and the chosen risk radius, not a calibrated confidence envelope or guaranteed error bound. Refinement uses radius 1 to keep the candidate pool independent of the chosen radius.',
                    'Validation retraces layered rays through the perturbed profile. Unreachable arrivals are omitted without shortening the route; retained coverage, GDOP and solver checks still apply.',
                    'Route errors are simulated coordinate errors. MSE averages squared errors over all 3N coordinates; 3D RMS is sqrt(3 MSE). Correlated dates are not independent Monte Carlo repetitions.',
                    'Detailed solutions and observation exports use the selected detail date. Other dates are retained in the validation comparison and trial exports.',
                    'Analytic OU and horizontal-gradient settings are inactive in profile mode. The joint estimator retains its shared constant sound-speed offset, delay, drift and relative delays.',
                    'Uploaded knots are preserved through a common piecewise-linear grid. No smoothing, decimation or implicit deep extension is applied.',
                    'The finite search uses complete routes within the time caps. Beacons respond serially; device slant range and monotone-ray support are separate constraints.'
                ])
