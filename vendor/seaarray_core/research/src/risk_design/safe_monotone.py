"""Research-only bracket fallback; never extend the physical monotone branch.

Existing fast solutions are bitwise retained. Only failed roots are bisected,
inside their actual Snell range bound, with10um range acceptance. The caller's
existing100um final range gate is unchanged. No turning/reflected rays added.
"""
import numpy as np
from numba import njit
from risk_design.fast_precision import solve_parameters
from risk_design.precision import _stable_shoot, _offset_sensitivity

@njit(cache=True)
def solve_safe(ranges,sources,depths,speeds,offsets):
    result=solve_parameters(ranges,sources,depths,speeds,offsets)
    for k in range(len(result)):
        if np.isfinite(result[k]):continue
        hi=np.nextafter(1/np.max(speeds+offsets[k]),0.)
        src=sources[k:k+1];off=offsets[k:k+1]
        bound=_stable_shoot(np.array([hi]),depths,speeds,src,off)[0][0]
        if np.isnan(bound) or ranges[k]>bound:continue
        lo=0.;best=hi;error=abs(bound-ranges[k])
        for _ in range(120):
            mid=(lo+hi)/2
            x=_stable_shoot(np.array([mid]),depths,speeds,src,off)[0][0]
            if abs(x-ranges[k])<error:best=mid;error=abs(x-ranges[k])
            if x<ranges[k]:lo=mid
            else:hi=mid
        if error<=1e-5:result[k]=best
    return result

def safe_varying_offset(tracer,ship,beacon,offsets):
    ship=np.asarray(ship);beacon=np.asarray(beacon);offsets=np.broadcast_to(offsets,(len(ship),)).copy()
    source=np.abs(ship[:,2]);receiver=abs(beacon[2])
    if np.any(source>receiver):raise ValueError('invalid_ship_ray: reversed depths')
    tracer.delta_c=0.;z,c=tracer._get_effective_ssp(float(np.min(source)),receiver)
    if np.any(np.min(c)+offsets<=0):raise ValueError('invalid_ship_ray: nonpositive speed')
    difference=beacon[:2]-ship[:,:2];ranges=np.linalg.norm(difference,axis=1)
    p=solve_safe(ranges,source,z,c,offsets)
    if not np.isfinite(p).all():raise ValueError('invalid_ship_ray: no accepted monotone root (range bound enforced)')
    x,t,_=_stable_shoot(p,z,c,source,offsets)
    if np.any(abs(x-ranges)>1e-4) or not np.isfinite(t).all():raise ValueError('invalid_ship_ray: range closure')
    jac=np.zeros((len(ship),4));jac[:,:2]=p[:,None]*difference/np.maximum(ranges[:,None],1e-30)
    jac[:,2]=-np.sqrt(np.maximum(0.,1-(p*(c[-1]+offsets))**2))/(c[-1]+offsets)
    jac[:,3]=_offset_sensitivity(p,z,c,False,source,offsets)
    return t,jac
