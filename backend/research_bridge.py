"""Reuse current research primitives, without the three-beacon study parameterization."""
from pathlib import Path
import os
import sys
import numpy as np

def find_research_root():
    explicit = os.environ.get('SEAARRAY_ROOT')
    if explicit:
        root = Path(explicit).resolve()
        if not (root / 'research/src/risk_design/closed_tours.py').is_file():
            raise RuntimeError('SEAARRAY_ROOT is not a valid research checkout.')
        return root
    root = Path(__file__).resolve().parents[1] / 'vendor/seaarray_core'
    if (root / 'research/src/risk_design/trajectories.py').is_file():
        return root
    raise RuntimeError('Bundled numerical core is missing. Restore vendor/seaarray_core or explicitly set SEAARRAY_ROOT for development.')

RESEARCH_ROOT = find_research_root()
for relative in ('research/src', 'src'):
    path = str(RESEARCH_ROOT / relative)
    if path not in sys.path:
        sys.path.append(path)

from risk_design.trajectories import CompleteTour
from risk_design.fast_precision import FastPreciseRayTracer, solve_parameters
from risk_design.profile_gradient import _integrals
from ray_tracing import RayTracer, normalize_ssp_columns

class ResearchRayTracer(FastPreciseRayTracer):
    def __init__(self, profile):
        profile = normalize_ssp_columns(profile).dropna().sort_values('Depth (m)')
        if len(profile) < 2 or profile['Depth (m)'].duplicated().any():
            raise ValueError('The SSP requires at least two distinct depths.')
        if not np.isfinite(profile[['Depth (m)', 'SV (m/s)']]).all().all():
            raise ValueError('The SSP contains invalid values.')
        if (profile['SV (m/s)'] <= 0).any() or profile['Depth (m)'].min() > 0:
            raise ValueError('The SSP must cover sea level and contain positive sound speeds.')
        self.max_depth_m = float(profile['Depth (m)'].max())
        self.deep_gradient = 0.
        RayTracer.__init__(self, profile, use_cache=False)

    def mapping(self, ship, beacon, center, mode_depth=1000.):
        source = -ship[:, 2]
        self.delta_c = 0.
        z, c = self._get_effective_ssp(float(source.min()), -float(beacon[2]))
        p = solve_parameters(np.linalg.norm(ship[:, :2]-beacon[:2], axis=1),
                             source, z, c, np.zeros(len(ship)))
        if not np.isfinite(p).all():
            raise ValueError('The ray model does not support this horizontal range.')
        nodes, weights = np.polynomial.legendre.leggauss(8)
        mode = np.exp(-z / mode_depth)
        return _integrals(p, source, z, c, mode, ship, beacon,
                          nodes, weights, np.asarray(center), 1000.)

FAMILIES = {
    'circle': 'Circle', 'figure_eight': 'Figure eight', 'clover': 'Clover',
    'square': 'Square', 'circle_cross': 'Circle + cross',
    'square_cross': 'Square + cross', 'rose': 'Four-petal rose', 'asterisk': 'Eight-spoke star',
}

class SurveyTour:
    """Common maximum radial extent; complete outward/return segments included."""
    def __init__(self, family, radius, center, angle_deg=0):
        self.family, self.radius = family, float(radius)
        self.center = np.asarray(center, float)
        self.angle = float(angle_deg)
        if family != 'asterisk':
            self.core = CompleteTour(family, radius, self.center,
                                     np.deg2rad(angle_deg), turn_radius=0.)
            self.length = self.core.length()
            self.connector_length = self.core.connector_length
        else:
            points = [self.center]
            for a in np.arange(8)*np.pi/4 + np.deg2rad(angle_deg):
                points.extend([self.center + radius*np.array([np.cos(a), np.sin(a)]), self.center])
            self.points = np.asarray(points)
            self.distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
            self.length = float(self.distance[-1])
            self.connector_length = 0.
            self.core = None

    def positions(self, distance):
        s = np.clip(np.atleast_1d(distance), 0., self.length)
        if self.core is not None:
            return self.core.positions(s)[:, :2]
        return np.c_[np.interp(s, self.distance, self.points[:, 0]),
                     np.interp(s, self.distance, self.points[:, 1])]

    def vessel(self, times, speed, lever):
        s = np.asarray(times)*speed
        xy = self.positions(s)
        d = self.positions(s+.1)-self.positions(s-.1)
        # Heading clockwise from north; RFU lever arm, level vessel.
        yaw = np.arctan2(d[:, 0], d[:, 1])
        antenna = np.c_[xy, np.zeros(len(xy))]
        offset = np.c_[lever[0]*np.cos(yaw)+lever[1]*np.sin(yaw),
                       -lever[0]*np.sin(yaw)+lever[1]*np.cos(yaw),
                       np.full(len(xy), lever[2])]
        return antenna, antenna+offset

    def preview(self):
        points = self.positions(np.linspace(0., self.length, 801))
        return [{'x': float(p[0]), 'y': float(p[1])} for p in points]
