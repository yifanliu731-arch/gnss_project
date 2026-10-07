"""Complete survey routes without home connectors or budget padding.

The five simple families retain their closed curves. Circle plus cross runs
one circle, one diameter, a straight transfer and the perpendicular diameter.
The historical CompleteTour remains unchanged.
"""
from functools import lru_cache
import numpy as np
from .trajectories import CompleteTour

FAMILIES = {'circle':'Circle', 'figure_eight':'Figure eight', 'clover':'Clover',
            'square':'Square', 'rose':'Four-petal rose', 'circle_cross':'Circle + cross'}
VERSION = 'complete-survey-v2'


class CircleCrossUnit:
    """Unit circle followed by two diameters and a straight transfer.

    Order: one CCW perimeter, E→W diameter, W→N diagonal transfer,
    N→S diameter. Ends at S without a forced return. Total length is
    2*pi+4+sqrt(2). The transfer counts in time and normally has no pings.
    """
    loop_length = 2*np.pi+4+np.sqrt(2)

    def _loop_point(self, local):
        s = min(max(float(local), 0.), self.loop_length)
        if s <= 2*np.pi:
            return np.array([np.cos(s), np.sin(s)])
        s -= 2*np.pi
        if s <= 2:
            return np.array([1-s, 0.])
        s -= 2
        if s <= np.sqrt(2):
            return np.array([-1+s/np.sqrt(2), s/np.sqrt(2)])
        s -= np.sqrt(2)
        return np.array([0., 1-s])

@lru_cache(maxsize=6)
def unit_curve(family):
    if family not in FAMILIES:
        raise ValueError('Unsupported complete closed survey family.')
    if family == 'circle_cross':
        return CircleCrossUnit()
    return CompleteTour(family, 1., np.zeros(2), 0., turn_radius=0.)

class ClosedTour:
    def __init__(self, family, radius, center, angle=0.):
        center=np.asarray(center,float)
        if center.shape!=(2,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0 or not np.isfinite(angle):
            raise ValueError('A closed tour requires a finite centre, angle and positive scale.')
        self.family=family;self.radius=float(radius);self.center=center.copy();self.angle=float(angle)
        self.unit=unit_curve(family);self.loop_length=self.unit.loop_length*self.radius
        self.connector_length=np.sqrt(2)*self.radius if family=='circle_cross' else 0.
        self.rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

    def length(self):return self.loop_length

    def transfer_intervals(self):
        if self.family!='circle_cross':return []
        start=(2*np.pi+2)*self.radius
        return [(start,start+self.connector_length)]

    def acquisition_mask(self, distance, end_distance=None):
        start=np.atleast_1d(distance).astype(float)
        end=start if end_distance is None else np.atleast_1d(end_distance).astype(float)
        keep=np.ones(len(start),dtype=bool)
        for a,b in self.transfer_intervals():
            keep &= ~((start < b-1e-8) & (end >= a-1e-8))
        return keep

    def positions(self, distance):
        s=np.atleast_1d(distance).astype(float)
        if not np.isfinite(s).all() or np.any(s < -1e-8) or np.any(s>self.loop_length+1e-7):
            raise ValueError('Distance is outside the complete closed route.')
        xy=np.array([self.unit._loop_point(float(x/self.radius)) for x in s])
        return np.c_[xy@self.rotation.T*self.radius+self.center,np.full(len(s),-10.)]

    def metadata(self):
        return dict(route_version=VERSION,closed=self.family!='circle_cross',traversals=1,connector_length_m=float(self.connector_length),
            retraced_length_m=0.,unique_path_length_m=float(self.loop_length),
            acquisition_path_length_m=float(self.loop_length-self.connector_length),
            transfer_intervals_m=self.transfer_intervals(),
            scale_parameter='side_length_m' if self.family=='square' else 'radial_extent_m',
            scale_value_m=float(self.radius*np.sqrt(2) if self.family=='square' else self.radius),
            start_position_m=self.positions([0])[0,:2].tolist(),
            angle_role='start_phase' if self.family=='circle' else 'orientation',
            direction=1,start_phase_deg=float(np.rad2deg(self.angle)) if self.family=='circle' else 0.)

def scale_grid(family, depth, speed, budgets_h, ratios):
    """User seed scales plus exact complete-route budget boundaries in bounds."""
    ratios=np.unique(np.asarray(ratios,float));lo,hi=float(ratios[0]),float(ratios[-1])
    values={float(r):'user_scale' for r in ratios}
    unit_length=unit_curve(family).loop_length
    for budget in budgets_h:
        ratio=float(budget*3600*speed/(unit_length*depth))
        if lo<=ratio<=hi and not any(np.isclose(ratio,r,rtol=1e-12,atol=0) for r in values):
            values[ratio]='budget_boundary'
    return sorted(values.items())

def refinements(records, budgets_h, gdop_limit):
    """Bisect log-scale intervals around each family's C1/C3 winners.

    Both criteria use the same expanded pool. Invalid designs stay as boundaries
    and their neighbouring midpoints can be evaluated, but never win selection.
    """
    proposals={}
    groups=sorted({(r['type'],r['angle_deg']) for r in records})
    for family,angle in groups:
        group=sorted([r for r in records if r['type']==family and r['angle_deg']==angle],key=lambda r:r['ratio'])
        for budget in budgets_h:
            valid=[r for r in group if r.get('design_valid') and r['duration_h']<=budget+1e-10 and r['gdop']<=gdop_limit]
            for metric in ('c1_m2','c3_m2'):
                if not valid:continue
                winner=min(valid,key=lambda r:(r[metric],r['duration_h']))
                index=group.index(winner)
                for neighbour in (index-1,index+1):
                    if 0<=neighbour<len(group):
                        other=group[neighbour]
                        ratio=float(np.sqrt(winner['ratio']*other['ratio']))
                        # Existing near-identical scales must not spawn duplicates.
                        if not any(np.isclose(ratio,r['ratio'],rtol=1e-10,atol=0) for r in group):
                            proposals[(family,ratio,angle)]='risk_refinement'
    return [(f,r,a,source) for (f,r,a),source in sorted(proposals.items())]
