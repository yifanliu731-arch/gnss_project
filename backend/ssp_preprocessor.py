"""Parse uploaded SSP values without smoothing, decimation or rounding."""
import csv
import io
import numpy as np
from typing import Optional, Tuple


# =====================================================================
#  Phase 1: Parse raw SSP file → DataFrame
# =====================================================================

def _find_data_start(lines: list[str]) -> int:
    """Locate the row immediately after ``[data]`` section header."""
    for i, line in enumerate(lines):
        if '[data]' in line.lower():
            return i + 1
    return 0   # fallback: no header


def parse_raw_ssp(file_content: str) -> np.ndarray:
    """
    Parse a raw .csv SSP file.

    Returns
    -------
    data : (N, 2) np.ndarray
        [[SV (m/s), Depth (m)], ...]  sorted shallow → deep.
    """
    lines = file_content.splitlines()
    skip = _find_data_start(lines)
    reader = csv.DictReader(io.StringIO('\n'.join(lines[skip:])))

    rows = []
    if reader.fieldnames is None:
        return np.empty((0, 2))

    # Detect column names flexibly
    cols = [c.strip() for c in reader.fieldnames]
    sv_col = depth_col = None

    # Normalise column names for matching (exact + substring)
    col_lower = [c.lower().strip() for c in cols]
    for i, c in enumerate(col_lower):
        # Exact match first
        if c in ('sv (m/s)', 'velocity_mps', 'sv', 'velocity', 'sound_speed', 'sv_mps', '声速'):
            if sv_col is None:
                sv_col = cols[i]
        elif c in ('depth (m)', 'depth_m', 'depth', 'z', '深度', 'dep'):
            if depth_col is None:
                depth_col = cols[i]
        # Substring fuzzy match (e.g. "sound speed [m/s]" contains "speed")
        elif sv_col is None and any(kw in c for kw in ('speed', 'sv', 'veloc', '声速')):
            sv_col = cols[i]
        elif depth_col is None and any(kw in c for kw in ('depth', 'dep', 'z', '深度')):
            depth_col = cols[i]

    # Fallback: if only 2 numeric columns, assume [sv, depth] or [depth, sv]
    if sv_col is None and depth_col is None and len(cols) >= 2:
        # Check first 5 rows to guess which column is depth vs velocity
        # depth is typically less than 5000, velocity around 1500
        sv_col = cols[0]
        depth_col = cols[1]
    elif sv_col is None and depth_col is not None and len(cols) >= 2:
        sv_col = cols[0] if cols[0] != depth_col else cols[1]
    elif depth_col is None and sv_col is not None and len(cols) >= 2:
        depth_col = cols[0] if cols[0] != sv_col else cols[1]

    if sv_col is None or depth_col is None:
        # Last resort: use first two columns
        if len(cols) >= 2:
            sv_col = cols[0]
            depth_col = cols[1]
        else:
            return np.empty((0, 2))

    # Detect if header row is actually data (no real headers)
    sample_vals = []
    for row in reader:
        sample_vals.append(row)
        if len(sample_vals) >= 3:
            break
    # Check if the "headers" look like numeric values
    _header_looks_numeric = True
    if sv_col and depth_col:
        for sv_row in sample_vals:
            try:
                _v = float(sv_row.get(sv_col, 'x'))
                if _v < 500 or _v > 5000:
                    _header_looks_numeric = False
                break
            except (ValueError, TypeError):
                _header_looks_numeric = False
                break

    if _header_looks_numeric:
        # The first row IS data, not headers — re-parse without DictReader
        all_lines = file_content.splitlines()
        data_lines = all_lines[skip:]  # skip [data] header if present
        # Skip first line (was used as header), parse remaining as plain CSV
        sv_idx = cols.index(sv_col) if sv_col in cols else 0
        depth_idx = cols.index(depth_col) if depth_col in cols else 1
        for line in data_lines[1:]:
            parts = line.strip().split(',')
            if len(parts) < max(sv_idx, depth_idx) + 1:
                continue
            try:
                sv = float(parts[sv_idx].strip())
                depth = float(parts[depth_idx].strip())
            except (ValueError, IndexError):
                continue
            depth = abs(depth)
            if sv < 1000 or sv > 2000:
                continue
            rows.append([sv, depth])
    else:
        # Normal DictReader flow
        for row in sample_vals:
            try:
                sv = float(row.get(sv_col, 0))
                depth = float(row.get(depth_col, 0))
                depth = abs(depth)
                if sv < 1000 or sv > 2000:
                    continue
                rows.append([sv, depth])
            except (ValueError, TypeError):
                continue
        # Read remaining rows
        for row in reader:
            try:
                sv = float(row.get(sv_col, 0))
                depth = float(row.get(depth_col, 0))
                depth = abs(depth)
                if sv < 1000 or sv > 2000:
                    continue
                rows.append([sv, depth])
            except (ValueError, TypeError):
                continue

    if not rows:
        return np.empty((0, 2))

    arr = np.array(rows)
    # Sort by depth ascending
    arr = arr[arr[:, 1].argsort()]
    return arr


def load_and_parse_ssp(file_content: str, **_legacy_options) -> dict:
    """Keep every parsed profile row; legacy smoothing flags have no effect."""
    raw = parse_raw_ssp(file_content)
    points = [{"depth": float(row[1]), "sv": float(row[0])} for row in raw]
    return {"raw_points": points, "filtered_points": [], "optimized_points": [],
            "n_raw": len(points), "n_filtered": 0, "n_optimized": 0,
            "processing": "uploaded_values_unchanged"}
