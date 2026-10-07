"""Explicit two-file profile input, calendar separation and training-only risk."""
from datetime import date
from hashlib import sha256
from io import StringIO
from pathlib import Path
import json
import numpy as np
import pandas as pd

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/south_china_sea_profiles'
MAX_BYTES = 16 * 1024 * 1024


def read_csv(text, required):
    if not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > MAX_BYTES:
        raise ValueError('Upload both profile CSV files, each at most 16 MiB.')
    try:
        table = pd.read_csv(StringIO(text.lstrip('\ufeff')), dtype={'date': str}, float_precision='round_trip')
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError('Cannot read the profile CSV file.') from exc
    if not set(required) <= set(table.columns):
        raise ValueError('Required profile CSV columns: ' + ', '.join(required))
    for column in ('depth_m', 'sound_speed_mps'):
        table[column] = pd.to_numeric(table[column], errors='coerce')
    if not np.isfinite(table[['depth_m','sound_speed_mps']].to_numpy()).all():
        raise ValueError('Profile CSV contains missing or nonfinite depths or speeds.')
    if (table['depth_m'] < 0).any() or (table['sound_speed_mps'] <= 0).any():
        raise ValueError('Depths must be nonnegative and sound speeds positive.')
    return table


def iso_date(value):
    try:
        if not isinstance(value, str) or len(value) != 10:
            raise ValueError()
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError('Use ISO dates YYYY-MM-DD for profiles and split boundaries.') from exc


def load_profiles(settings, *, fit=True):
    if not isinstance(settings, dict):
        raise ValueError('Sound-field settings must be an object.')
    background = read_csv(settings.get('background_csv'), ['depth_m','sound_speed_mps']).sort_values('depth_m')
    history = read_csv(settings.get('history_csv'), ['date','depth_m','sound_speed_mps'])
    if len(background) < 2 or len(background) > 6000 or background['depth_m'].duplicated().any() or background['depth_m'].iloc[0] != 0:
        raise ValueError('Background requires 2–6000 distinct depths starting at sea level.')
    history['date'] = history['date'].map(iso_date)
    if history.duplicated(['date','depth_m']).any():
        raise ValueError('Historical profiles contain duplicate date/depth pairs.')
    dates = sorted(history['date'].unique())
    if len(dates) < 2 or len(dates) > 1500:
        raise ValueError('Historical data require 2–1500 dated profiles.')
    # Retain every supplied knot in either file; interpolation only adds knots.
    depth = np.unique(np.r_[background['depth_m'].to_numpy(), history['depth_m'].to_numpy()])
    if len(depth) > 6000 or len(depth)*len(dates) > 2_000_000:
        raise ValueError('Profile grid exceeds the supported size; no points were silently removed.')
    if depth[-1] > background['depth_m'].iloc[-1]:
        raise ValueError('Background and historical profiles must cover the same full depth interval.')
    profiles = []
    for day, group in history.groupby('date', sort=True):
        group = group.sort_values('depth_m')
        if len(group) < 2 or group['depth_m'].iloc[0] != 0 or group['depth_m'].iloc[-1] < depth[-1]:
            raise ValueError('Every historical profile must cover sea level through the full background depth; no extrapolation is applied.')
        profiles.append(np.interp(depth, group['depth_m'], group['sound_speed_mps']))
    nominal = np.interp(depth, background['depth_m'], background['sound_speed_mps'])
    profiles = np.asarray(profiles); dates = np.asarray(dates)
    train_end = iso_date(settings.get('training_end')); validation_start = iso_date(settings.get('validation_start'))
    if train_end >= validation_start:
        raise ValueError('Training end must precede validation start.')
    train = dates <= train_end; test = dates >= validation_start
    if not train.any() or not test.any():
        raise ValueError('The split must leave at least one training and one validation profile.')
    alpha = float(settings.get('amplitude', 1.))
    if alpha not in (.5, 1., 2., 5.):
        raise ValueError('Select a perturbation multiplier of 0.5, 1, 2 or 5.')
    scope = settings.get('validation_scope', 'single')
    if scope not in ('single', 'all'):
        raise ValueError('Validation scope must be single or all.')
    selected_date = iso_date(settings.get('validation_date', str(dates[test][0])))
    if selected_date not in dates[test]:
        raise ValueError('Select a validation date outside the training and gap periods.')
    delta = profiles-nominal
    if np.any(nominal+alpha*delta[test] <= 0):
        raise ValueError('The amplified validation profiles contain nonpositive sound speeds.')
    basis = None
    if fit:
        _, singular, vt = np.linalg.svd(delta[train], full_matrices=False)
        keep = singular > (singular[0]*1e-10 if singular[0] > 0 else 0)
        basis = vt[keep]*singular[keep,None]/np.sqrt(int(train.sum()))
    metadata = dict(training_dates=dates[train].tolist(), validation_dates=dates[test].tolist(),
                    gap_dates=dates[~(train|test)].tolist(), training_count=int(train.sum()), validation_count=int(test.sum()),
                    gap_count=int((~(train|test)).sum()), depth_count=len(depth), max_depth_m=float(depth[-1]),
                    original_background_knots=len(background), amplitude=alpha,
                    background_sha256=sha256(settings['background_csv'].encode('utf-8')).hexdigest(),
                    history_sha256=sha256(settings['history_csv'].encode('utf-8')).hexdigest(),
                    validation_scope=scope, detail_date=selected_date,
                    risk='Uncentered training second moment; unit ellipsoid, not a confidence envelope.',
                    interpolation='Union of all supplied knots, piecewise linear, no smoothing, decimation or extrapolation.')
    if basis is not None:
        metadata['risk_rank'] = len(basis)
    return dict(depth=depth, nominal=nominal, delta=delta, dates=dates, train=train, test=test,
                basis=basis, alpha=alpha, scope=scope, detail_date=selected_date, metadata=metadata)


def example_payload():
    return dict(background_csv=(EXAMPLE/'background.csv').read_text(encoding='utf-8'),
                history_csv=(EXAMPLE/'history.csv').read_text(encoding='utf-8'),
                provenance=json.loads((EXAMPLE/'provenance.json').read_text(encoding='utf-8')),
                training_end='2020-06-30', validation_start='2020-07-15', validation_date='2020-07-15',
                validation_scope='single', amplitude=1.)
