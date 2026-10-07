"""Safeguarded Newton shooting for the same analytic layered-ray equation.

Only varying_offset is accelerated. Bracketing and final range checks remain.
The reference bisection tracer is retained for regression verification.
"""
import math
import numpy as np
from numba import njit
from risk_design.precision import PreciseRayTracer,_stable_shoot,_offset_sensitivity

@njit(cache=True)
def solve_parameters(ranges,sources,depths,speeds,offsets):
    result=np.empty(len(ranges))
    for k in range(len(ranges)):
        cs=speeds+offsets[k];lo=0.;hi=np.nextafter(1/np.max(cs),0.)
        height=depths[-1]-sources[k]
        p=min(hi*.99,ranges[k]/math.sqrt(ranges[k]**2+height**2)/np.mean(cs))
        converged=False
        for it in range(65):
            x=0.;dx=0.
            for i in range(len(depths)-1):
                if sources[k]>=depths[i+1]:continue
                dz=depths[i+1]-depths[i];a=cs[i];b=cs[i+1]
                if sources[k]>depths[i]:
                    a+=(b-a)*(sources[k]-depths[i])/dz;dz=depths[i+1]-sources[k]
                ca=math.sqrt(max(1e-30,1-(p*a)**2));cb=math.sqrt(max(1e-30,1-(p*b)**2))
                denominator=ca+cb;factor=(a+b)*dz
                x+=p*factor/denominator
                dx+=factor/denominator+p*p*factor*(a*a/ca+b*b/cb)/denominator**2
            if abs(x-ranges[k])<1e-7:
                converged=True;break
            if x<ranges[k]:lo=p
            else:hi=p
            proposal=p-(x-ranges[k])/dx
            p=proposal if lo<proposal<hi else (lo+hi)/2
        result[k]=p if converged else np.nan
    return result

class FastPreciseRayTracer(PreciseRayTracer):
    def varying_offset(self,ship,beacon,offsets):
        ship=np.asarray(ship);beacon=np.asarray(beacon);offsets=np.broadcast_to(offsets,(len(ship),)).copy()
        source=np.abs(ship[:,2]);receiver=abs(beacon[2])
        if np.any(source>receiver):raise ValueError('invalid_ship_ray: reversed depths')
        self.delta_c=0.
        z,c=self._get_effective_ssp(float(np.min(source)),receiver)
        if np.any(np.min(c)+offsets<=0):raise ValueError('invalid_ship_ray: nonpositive speed')
        difference=beacon[:2]-ship[:,:2];ranges=np.linalg.norm(difference,axis=1)
        p=solve_parameters(ranges,source,z,c,offsets)
        if not np.all(np.isfinite(p)):raise ValueError('invalid_ship_ray: root closure')
        x,t,_=_stable_shoot(p,z,c,source,offsets)
        if np.any(abs(x-ranges)>1e-4) or not np.all(np.isfinite(t)):raise ValueError('invalid_ship_ray: range closure')
        jac=np.zeros((len(ship),4));jac[:,:2]=p[:,None]*difference/np.maximum(ranges[:,None],1e-30)
        jac[:,2]=-np.sqrt(np.maximum(0.,1-(p*(c[-1]+offsets))**2))/(c[-1]+offsets)
        jac[:,3]=_offset_sensitivity(p,z,c,False,source,offsets)
        return t,jac
