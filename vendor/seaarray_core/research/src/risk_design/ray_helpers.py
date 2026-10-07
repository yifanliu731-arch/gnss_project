"""SSP loading and diagnostic chord integration; no ray fallback in fitting."""
import numpy as np
import pandas as pd
from ray_tracing import RayTracer
from experiment_config import TEMPLATE_SSP

def load_tracer() -> RayTracer:
    ssp = pd.read_csv(TEMPLATE_SSP)
    return RayTracer(ssp, use_cache=False)

def chord_time(pa: np.ndarray, pb: np.ndarray, tracer: RayTracer, a0: float, n: int = 64) -> float:
    """Straight-chord travel time on the piecewise-linear SSP.

    Within each depth layer, 1/c is integrated in closed form. `n` is retained
    so callers can still request the sampled quadrature for comparison.
    """
    del n
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    geom = float(np.linalg.norm(pa - pb))
    if geom < 1e-6:
        return float("nan")
    z0 = abs(float(pa[2]))
    z1 = abs(float(pb[2]))
    if abs(z1 - z0) < 1e-6:
        c = float(np.interp(z0, tracer.Z, tracer.V)) + a0
        return geom / c
    z_lo, z_hi = (z0, z1) if z0 < z1 else (z1, z0)
    nodes = [z_lo]
    for z in tracer.Z:
        if z_lo < z < z_hi:
            nodes.append(float(z))
    nodes.append(z_hi)
    # Path length per metre of depth is constant on a straight chord.
    ds_dz = geom / abs(z1 - z0)
    total = 0.0
    for za, zb in zip(nodes[:-1], nodes[1:]):
        ca = float(np.interp(za, tracer.Z, tracer.V)) + a0
        cb = float(np.interp(zb, tracer.Z, tracer.V)) + a0
        dz = zb - za
        if abs(cb - ca) < 1e-8:
            total += ds_dz * dz / ca
        else:
            total += ds_dz * dz * math_log(cb / ca) / (cb - ca)
    return float(total)

def math_log(value: float) -> float:
    return float(np.log(value))
