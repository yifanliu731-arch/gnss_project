"""Static profile-ensemble sensitivities using the layered Snell ray model.

The knot integral is the same eight-point Fermat quadrature validated in the
time-holdout experiment. Profiles are never smoothed or extrapolated here.
"""
import numpy as np
from numba import njit
from risk_design.safe_monotone import solve_safe


@njit(cache=True)
def node_weights(p, source, z, c, nodes, weights):
    out = np.zeros((len(p), len(z)))
    for k in range(len(p)):
        for j in range(len(z)-1):
            za = max(source[k], z[j]); zb = z[j+1]
            if za >= zb:
                continue
            frac = (za-z[j])/(zb-z[j]); ca = c[j]+(c[j+1]-c[j])*frac; cb = c[j+1]
            for l in range(len(nodes)):
                f = (nodes[l]+1)/2; cq = ca+(cb-ca)*f
                factor = -weights[l]*(zb-za)/2/(cq*cq*np.sqrt(1-(p[k]*cq)**2))
                out[k,j] += factor*(1-f)*(1-frac)
                out[k,j+1] += factor*(frac*(1-f)+f)
    return out


def profile_mapping(tracer, ship, ids, beacons, depth, shapes):
    """Return dT/d(profile coefficient); rows are supplied observations."""
    result = np.empty((len(ship), len(shapes)))
    nodes, weights = np.polynomial.legendre.leggauss(8)
    for index, beacon in enumerate(beacons):
        mask = ids == index
        if not mask.any():
            continue
        positions = ship[mask]; source = -positions[:,2]
        tracer.delta_c = 0.
        z, c = tracer._get_effective_ssp(float(source.min()), -float(beacon[2]))
        if z[0] < depth[0] or z[-1] > depth[-1]:
            raise ValueError('Profile shapes do not cover the reference ray depths.')
        p = solve_safe(np.linalg.norm(positions[:,:2]-beacon[:2],axis=1), source, z, c, np.zeros(len(source)))
        if not np.isfinite(p).all():
            raise ValueError('No supported monotone reference ray')
        result[mask] = node_weights(p, source, z, c, nodes, weights) @ np.array([np.interp(z, depth, s) for s in shapes]).T
    return result
