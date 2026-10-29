"""Recompute the frozen paper simulation from the public profile CSV files.

smoke: two rose routes and the four-circle baseline on the first held-out day.
scores: all 702 frozen geometries and ten-budget C1/C2/C3 selections.
daily: all 170 held-out days for the two roses and four-circle baseline.
all: scores and daily. Results go to an ignored local cache, with source hashes.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache/numba'))
import sys
sys.path.insert(0, str(ROOT / 'backend'))
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import lru_cache
import hashlib
import json
import numpy as np
import joint_simulation as js
from closed_survey import SurveyTour
from profile_data import load_profiles, example_payload
from profile_simulation import tracer_for, profile_metrics
from risk_design.safe_monotone import solve_safe
from risk_design.precision import _stable_shoot
from paper_station import StationTour, station_data

PROTOCOL_PATH = ROOT / 'examples/paper_protocol.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())+'\n', encoding='utf-8')
    temporary.replace(path)


@lru_cache(maxsize=1)
def fields():
    return load_profiles(example_payload())


def route(identifier):
    c = js.configuration(PROTOCOL['config'])
    c.update(sa=0., ge=0., gn=0.)
    f = fields()
    tracer = tracer_for(f['depth'], f['nominal'])
    if identifier == PROTOCOL['station_baseline']['id']:
        spec = PROTOCOL['station_baseline']
        tour = StationTour(spec['type'], spec['size'], c['beacons'], spec['angle_deg'], spec['order'])
        data = station_data(tour, c, tracer)
    else:
        spec = PROTOCOL['geometry_pool'][identifier]
        tour = SurveyTour(spec['type'], spec['size'], c['center'], spec['angle_deg'])
        data = js.generate_observations(tour, c, tracer)
    np.testing.assert_allclose(tour.length/c['speed']/3600, spec['duration_h'], atol=1e-8, rtol=0)
    return data, c, tracer


def score(identifier):
    spec = PROTOCOL['geometry_pool'][identifier]
    row = dict(id=identifier, duration_h=spec['duration_h'], eligible=False)
    try:
        if spec['duration_h'] > 25+1e-10:
            raise ValueError('Complete duration exceeds 25 h')
        data, c, tracer = route(identifier)
        metrics = profile_metrics(data, c, tracer, fields())
        if metrics['gdop'] > c['gdop']:
            raise ValueError('GDOP limit')
        if data['rejected_ray'] or data['rejected_range']:
            raise ValueError('Incomplete nominal ray/range acquisition')
        row.update(eligible=True, c1=metrics['c1_m2'], beta=metrics['worst_bias_m2'],
                   mean_penalty=metrics['c2_m2']-metrics['c1_m2'])
    except ValueError as exc:
        row['reason'] = str(exc)
    reference = PROTOCOL['reference_scores'][identifier]
    if row['eligible'] != reference['eligible']:
        raise AssertionError(f'Eligibility changed for route {identifier}: {row}')
    if row['eligible']:
        for key in ('c1', 'beta', 'mean_penalty'):
            np.testing.assert_allclose(row[key], reference[key],
                rtol=PROTOCOL['tolerance']['score_rtol'], atol=PROTOCOL['tolerance']['score_atol'],
                err_msg=f'route {identifier}, {key}')
    return row


def truth(data, c, profile):
    f = fields()
    tracer = tracer_for(f['depth'], profile)
    travel = np.full(len(data['times']), np.nan)
    for index, beacon in enumerate(c['beacons']):
        ix = np.flatnonzero(data['ids'] == index)
        ship = data['true_ship'][ix]
        source = -ship[:, 2]
        z, speed = tracer._get_effective_ssp(float(source.min()), -float(beacon[2]))
        ranges = np.linalg.norm(ship[:, :2]-beacon[:2], axis=1)
        p = solve_safe(ranges, source, z, speed, np.zeros(len(ix)))
        valid = np.isfinite(p)
        if valid.any():
            x, t, _ = _stable_shoot(p[valid], z, speed, source[valid], np.zeros(int(valid.sum())))
            closed = np.isfinite(t) & (abs(x-ranges[valid]) <= 1e-4)
            travel[ix[valid][closed]] = t[closed]
    keep = np.isfinite(travel) & (2*travel+1 <= data['slot_s']+1e-8)
    return travel, keep


def trial(data, c, tracer, identifier, day):
    f = fields()
    travel, keep = truth(data, c, f['nominal']+f['delta'][day])
    row = dict(id=identifier, day=day, date=str(f['dates'][day]), success=False,
               planned_observations=len(keep), missing_observations=int((~keep).sum()))
    rng = np.random.default_rng(np.random.SeedSequence([PROTOCOL['seed'], identifier, day]))
    # Draw on the full planned grid before omitting unsupported arrivals.
    white = rng.normal(0., c['sw'], len(travel))
    navigation = rng.normal(0., c['gnss'], data['true_ship'].shape)
    try:
        counts = np.bincount(data['ids'][keep], minlength=len(c['beacons']))
        if np.any(counts < 8):
            raise ValueError('Fewer than eight retained observations per beacon')
        masked = {k: v[keep] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(travel) else v
                  for k, v in data.items()}
        masked.update(ship=(data['true_ship']+navigation)[keep], delays=(travel+white)[keep],
                      per_beacon_counts=counts.tolist())
        if js.design_metrics(masked, c, tracer)['gdop'] > c['gdop']:
            raise ValueError('GDOP limit')
        fit = js.joint_fit(masked, c, tracer)
        positions = np.array([[r['x'], r['y'], r['z']] for r in fit['estimated_beacons']])
        error = (positions-c['beacons']).ravel()
        row.update(success=True, error=error.tolist(), mse=float(np.mean(error**2)),
                   remaining_scaled_step=fit['solver_remaining_scaled_step'])
    except ValueError as exc:
        row['reason'] = str(exc)
    return row


def daily_job(args):
    identifier, days, output = args
    data, c, tracer = route(identifier)
    records = []
    references = {(r['id'], r['day']): r for r in PROTOCOL['native_daily_reference']}
    for day in days:
        path = Path(output) / 'daily' / f'{identifier}_{day}.json'
        if path.exists():
            row = json.loads(path.read_text(encoding='utf-8'))
        else:
            row = trial(data, c, tracer, identifier, day)
            write(path, row)
        reference = references[identifier, day]
        if row['success'] != reference['success']:
            raise AssertionError(f'Fit acceptance changed: route {identifier}, day {day}: {row}')
        if row['success']:
            difference = float(np.max(np.abs(np.asarray(row['error'])-reference['error'])))
            if difference > PROTOCOL['tolerance']['coordinate_m']:
                raise AssertionError(f'Coordinate regression: route {identifier}, day {day}, {difference} m')
        records.append(row)
        if len(records) % 20 == 0:
            print(f'Route {identifier}: {len(records)}/{len(days)} days checked', flush=True)
    return records


def source_manifest():
    paths = [PROTOCOL_PATH, *sorted((ROOT/'backend').glob('*.py')),
             *sorted((ROOT/'scripts').glob('*.py')), ROOT/'vendor/seaarray_core/manifest.json',
             ROOT/'examples/south_china_sea_profiles/background.csv', ROOT/'examples/south_china_sea_profiles/history.csv']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['smoke', 'scores', 'daily', 'all'], nargs='?', default='smoke')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/paper-reproduction')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be from 1 to 8')
    args.output = args.output.resolve()
    manifest = source_manifest()
    manifest_path = args.output/'inputs.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise SystemExit('Sources or inputs changed. Choose a fresh --output directory for this run.')
    write(manifest_path, manifest)
    f = fields()
    assert (int(f['train'].sum()), int(f['test'].sum()), len(f['basis'])) == (182, 170, 32)
    report = dict(passed=False, phase=args.phase, scope=PROTOCOL['scope'],
                  training_profiles=182, gap_days=14, heldout_profiles=170, risk_rank=32)
    if args.phase in ('scores', 'all'):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(score, range(702), chunksize=4))
        by_id = {r['id']: r for r in rows}
        for plan in PROTOCOL['selections']:
            penalty = {'C1': None, 'C2': 'mean_penalty', 'C3': 'beta'}[plan['method']]
            allowed = [r for r in rows if r['eligible'] and r['duration_h'] <= plan['budget_h']+1e-8]
            best = min(allowed, key=lambda r: (r['c1']+(r[penalty] if penalty else 0.), r['duration_h'], r['id']))
            assert best['id'] == plan['id'], (plan, best)
        write(args.output/'scores.json', rows)
        report.update(geometry_scores_checked=len(by_id), budget_selections_checked=len(PROTOCOL['selections']))
    if args.phase in ('smoke', 'daily', 'all'):
        days = np.flatnonzero(f['test']).tolist()
        if args.phase == 'smoke':
            days = days[:1]
        jobs = [(identifier, days, str(args.output)) for identifier in (326, 338, 20002)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records = [r for group in pool.map(daily_job, jobs) for r in group]
        report.update(trials_checked=len(records), accepted=sum(r['success'] for r in records),
                      rejected=sum(not r['success'] for r in records), days_per_route=len(days),
                      tolerance_m=PROTOCOL['tolerance']['coordinate_m'])
        if len(days) == 170:
            compared = {}
            for baseline in (338, 20002):
                left = {r['day']: r for r in records if r['id'] == baseline}
                right = {r['day']: r for r in records if r['id'] == 326}
                common = [day for day in days if left[day]['success'] and right[day]['success']]
                a = float(np.sqrt(3*np.mean([left[d]['mse'] for d in common])))
                b = float(np.sqrt(3*np.mean([right[d]['mse'] for d in common])))
                compared[str(baseline)] = dict(common_days=len(common), baseline_3d_rmse_m=a,
                    risk_3d_rmse_m=b, reduction_percent=100*(1-b/a), conditional=len(common)!=170)
            report['native_comparisons'] = compared
    report['passed'] = True
    write(args.output/f'{args.phase}_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
