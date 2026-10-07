"""Display-only discontinuities; never filter, interpolate, or reorder observations."""
from datetime import datetime
import numpy as np


def uploaded_times(rows):
    if not rows:
        return None
    key = next((k for k in rows[0] if any(s in k.lower() for s in ('time', 'utc'))), None)
    if key is None:
        return None
    try:
        values = np.array([datetime.fromisoformat(str(r[key]).replace('Z', '+00:00')).timestamp() for r in rows])
        return values - values[0]
    except (ValueError, TypeError, KeyError):
        return None


def mark_track_gaps(points, times=None, source_break_times=()):
    """Mark the incoming edge after a gap without changing any coordinate.

    Time gaps exceed max(120 s, 10 median positive intervals). A large jump
    exceeds max(250 m, 20 median positive horizontal steps) and, when UTC is
    available, implies >30 m/s. These are display heuristics, not data QC.
    Explicit gaps in the source navigation also propagate to sampled tracks.
    """
    if not points:
        return []
    xy = np.array([[p['x'], p['y']] for p in points], float)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    jump_limit = max(250., 20*float(np.median(positive))) if len(positive) else 250.
    have_times = times is not None and len(times) == len(points) and np.isfinite(times).all()
    dt = np.diff(times) if have_times else np.zeros(len(steps))
    positive_dt = dt[dt > 0]
    gap_limit = max(120., 10*float(np.median(positive_dt))) if len(positive_dt) else 120.
    breaks = np.sort(np.asarray(source_break_times, float))
    result = []
    segment = 0
    for i, point in enumerate(points):
        gap = False
        if i:
            large_jump = steps[i-1] > jump_limit
            gap = bool(large_jump and (not have_times or dt[i-1] <= 0 or steps[i-1]/dt[i-1] > 30))
            if have_times:
                gap |= bool(dt[i-1] < 0 or dt[i-1] > gap_limit)
                gap |= bool(np.searchsorted(breaks, times[i], side='right') > np.searchsorted(breaks, times[i-1], side='right'))
        segment += int(gap)
        record = dict(point, gap_before=gap, segment_id=segment)
        if have_times: record['time_s'] = float(times[i])
        result.append(record)
    return result


def copy_gap_metadata(geographic, marked):
    if len(geographic) != len(marked):
        raise ValueError('Display coordinates must have matching point counts.')
    return [dict(p, **{k:q[k] for k in ('gap_before','segment_id','time_s') if k in q})
            for p,q in zip(geographic, marked)]
