"""Export saved results without rerunning the numerical workflow."""
import csv
import io
import json
import math
import zipfile


def select_rows(saved, request):
    rows = saved['rows']
    scope = request.get('scope', 'all')
    kind = saved['kind']
    if scope == 'all':
        return rows
    if scope == 'custom':
        ids = request.get('ids', [])
        if not isinstance(ids, list) or not ids:
            raise ValueError('Select at least one result.')
        key = 'candidate_id' if kind == 'simulation' else 'beacon_id'
        wanted = set(map(str, ids))
        if not wanted.issubset({str(r[key]) for r in rows}):
            raise ValueError('A selected result does not belong to this run.')
        return [r for r in rows if str(r[key]) in wanted]
    if scope != 'top10':
        raise ValueError('Unknown export scope.')
    if kind == 'simulation':
        criterion = saved['result']['criterion'] + '_m2'
        cap = float(request.get('budget_h', saved['result']['budget_values_h'][-1]))
        if not math.isfinite(cap) or cap not in saved['result']['budget_values_h']:
            raise ValueError('Select a time budget from this run.')
        gdop = float(saved['result']['config'].get('gdop_threshold', 2.5))
        eligible = [r for r in rows if r.get('design_valid', r.get('valid')) and r.get('gdop', math.inf) <= gdop
                    and r['duration_h'] <= cap + 1e-10 and math.isfinite(r.get(criterion, math.inf))]
        selected = sorted(eligible, key=lambda r: (r[criterion], r['duration_h'], r['candidate_id']))[:10]
    else:
        eligible = [r for r in rows if r.get('status') not in ('skipped', 'failed')
                    and isinstance(r.get('rmse_time_ms'), (int, float)) and math.isfinite(r['rmse_time_ms'])]
        selected = sorted(eligible, key=lambda r: (r['rmse_time_ms'], r['beacon_id']))[:10]
    if not selected:
        raise ValueError('No eligible results for this selection.')
    return selected


def csv_text(rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return '\ufeff' + out.getvalue()


def make_archive(saved, request, json_safe):
    selected = select_rows(saved, request)
    parts = request.get('parts', ['trajectory', 'observations', 'residuals', 'raw'])
    if not isinstance(parts, list) or not set(parts) <= {'trajectory', 'observations', 'residuals', 'raw'}:
        raise ValueError('Unknown export data selection.')
    kind = saved['kind']
    id_key = 'candidate_id' if kind == 'simulation' else 'beacon_id'
    summary = [{k:v for k,v in row.items() if not k.startswith('_') and k not in
                ('trajectory', 'observations', 'residuals_ms', 'residuals_m', 'outlier_mask')} for row in selected]
    manifest = dict(format_version=1, kind=kind, scope=request.get('scope','all'),
                    selected_ids=[r[id_key] for r in selected], parts=parts,
                    selection_budget_h=request.get('budget_h'),
                    ranking=(saved['result']['criterion'] if kind == 'simulation' else 'residual_rms_ms'),
                    config=saved['result'].get('config', saved.get('config')),
                    provenance=saved['result'].get('provenance'),
                    note='Exports saved data only; no simulation or fit is rerun. Field ranking is per-beacon residual RMS, not trajectory C3 or absolute coordinate accuracy.')
    manifest['run_metadata'] = {k:v for k,v in saved['result'].items() if k not in
        ('all_results','plans_by_id','plan_a','plan_b','plan_c','plan_trajectories','trajectory',
         'trajectory_latlon','residuals_ms','residuals_m','outlier_mask','beacon_results','residual_observations')}
    content = dict(manifest=manifest, results=[])
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('summary.csv', csv_text(json_safe(summary)))
        if kind == 'simulation' and saved['result'].get('field_validation'):
            validation = saved['result']['field_validation']
            selected_ids = {r[id_key] for r in selected}
            z.writestr('validation_trials.csv', csv_text(json_safe([r for r in validation['trials'] if r['candidate_id'] in selected_ids])))
            z.writestr('validation_comparisons.json', json.dumps(json_safe(validation),ensure_ascii=False,indent=2,allow_nan=False))
            if 'raw' in parts:
                settings = saved['result']['config']['sound_field']
                z.writestr('original_inputs/background.csv', settings['background_csv'])
                z.writestr('original_inputs/history.csv', settings['history_csv'])
        for index, row in enumerate(selected, 1):
            item = dict(summary[index-1])
            folder = f'results/{index:04d}'
            observations = row.get('observations', row.get('_observations', []))
            if 'trajectory' in parts:
                item['trajectory'] = row.get('trajectory', [])
                z.writestr(folder+'/trajectory.csv', csv_text(json_safe(item['trajectory'])))
            if 'observations' in parts:
                item['observations'] = observations
                z.writestr(folder+'/observations.csv', csv_text(json_safe(observations)))
            if 'residuals' in parts:
                residuals = [dict(observation_index=i+1, beacon_id=observations[i].get('beacon_id') if i<len(observations) else row.get('beacon_id'),
                                  time_s=observations[i].get('time_s') if i<len(observations) else None,
                                  residual_ms=v, residual_m=(row.get('residuals_m') or [])[i] if i<len(row.get('residuals_m') or []) else None,
                                  outlier=(row.get('outlier_mask') or [])[i] if i<len(row.get('outlier_mask') or []) else None)
                             for i,v in enumerate(row.get('residuals_ms') or [])]
                item['residuals'] = residuals
                z.writestr(folder+'/residuals.csv', csv_text(json_safe(residuals)))
            content['results'].append(item)
        if 'raw' in parts and kind == 'field':
            # Raw source files are complete originals, not filtered copies.
            for name, data in saved.get('raw', {}).items():
                if data is not None:
                    z.writestr('original_inputs/'+name, data)
            manifest['raw_input_scope'] = 'Complete original uploads, including unselected beacon rows.'
        z.writestr('manifest.json', json.dumps(json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False))
        z.writestr('results.json', json.dumps(json_safe(content), ensure_ascii=False, indent=2, allow_nan=False))
    return out.getvalue()
