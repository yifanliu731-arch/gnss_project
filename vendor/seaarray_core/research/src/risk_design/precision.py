"""Tighter scalar eigenray root for derivative and stationarity checks.

The legacy absolute p tolerance 1e-12 can be amplified near grazing incidence.
This implementation records the range residual; it does not alter SSP data.
"""
import numpy as np
import pandas as pd
import math
from numba import njit
from scipy.optimize import brentq
from ray_tracing import RayTracer
from experiment_config import TEMPLATE_SSP


@njit(cache=True)
def _ranges(parameters, depths, speeds, sources=None, offsets=None):
    out=np.zeros(len(parameters))
    for k in range(len(parameters)):
        p=parameters[k]
        for i in range(len(depths)-1):
            dz=depths[i+1]-depths[i]; c1=speeds[i]; c2=speeds[i+1]
            if offsets is not None:c1+=offsets[k];c2+=offsets[k]
            if sources is not None:
                if sources[k]>=depths[i+1]:continue
                if sources[k]>depths[i]:
                    c1+=(c2-c1)*(sources[k]-depths[i])/dz
                    dz=depths[i+1]-sources[k]
            co1=math.sqrt(max(0.,1-(p*c1)**2));co2=math.sqrt(max(0.,1-(p*c2)**2))
            denominator=co1+co2
            if denominator<=0: out[k]=np.inf;break
            out[k]+=p*(c1+c2)*dz/denominator
    return out


@njit(cache=True)
def _offset_sensitivity(parameters, depths, speeds, terminal_turn=False, sources=None, offsets=None):
    out=np.zeros(len(parameters))
    for k in range(len(parameters)):
        p=parameters[k]
        for i in range(len(depths)-1):
            dz=depths[i+1]-depths[i];c1=speeds[i];c2=speeds[i+1]
            if offsets is not None:c1+=offsets[k];c2+=offsets[k]
            if sources is not None:
                if sources[k]>=depths[i+1]:continue
                if sources[k]>depths[i]:
                    c1+=(c2-c1)*(sources[k]-depths[i])/dz
                    dz=depths[i+1]-sources[k]
            co1=math.sqrt(max(0.,1-(p*c1)**2));co2=math.sqrt(max(0.,1-(p*c2)**2))
            if terminal_turn and i==len(depths)-2:co2=0.
            denominator=co1+co2
            out[k]-=dz*(co1+c1*p*p*(c2+c1)/denominator)/(c1*c2)
    return out


@njit(cache=True)
def _stable_shoot(parameters, depths, speeds, sources=None, offsets=None):
    x=np.zeros(len(parameters));t=np.zeros(len(parameters));arc=np.zeros(len(parameters))
    for k in range(len(parameters)):
        p=parameters[k]
        for i in range(len(depths)-1):
            dz=depths[i+1]-depths[i]; c1=speeds[i]; c2=speeds[i+1]; dc=c2-c1
            if offsets is not None:c1+=offsets[k];c2+=offsets[k]
            if sources is not None:
                if sources[k]>=depths[i+1]:continue
                if sources[k]>depths[i]:
                    c1+=(c2-c1)*(sources[k]-depths[i])/dz
                    dz=depths[i+1]-sources[k];dc=c2-c1
            if dz<=0: continue
            if abs(p)<1e-12:
                t[k]+=dz/c1 if abs(dc)<1e-12 else math.log1p(dc/c1)*dz/dc
                arc[k]+=dz
                continue
            co1=math.sqrt(max(0.,1-(p*c1)**2));co2=math.sqrt(max(0.,1-(p*c2)**2))
            if abs(dc)<1e-12:
                if co1<=0: return np.full(len(parameters),np.nan),np.full(len(parameters),np.nan),np.full(len(parameters),np.nan)
                x[k]+=p*c1*dz/co1;t[k]+=dz/(c1*co1);arc[k]+=dz/co1
            else:
                denominator=co1+co2
                if denominator<=0: return np.full(len(parameters),np.nan),np.full(len(parameters),np.nan),np.full(len(parameters),np.nan)
                difference=p*p*dc*(c2+c1)/denominator
                x[k]+=p*(c1+c2)*dz/denominator
                t[k]+=(math.log1p(dc/c1)+math.log1p(difference/(1+co2)))*dz/dc
                angle=math.atan2(p*(dc*co1+c1*difference),co1*co2+p*p*c1*c2)
                arc[k]+=angle*dz/(p*dc)
    return x,t,arc


class PreciseRayTracer(RayTracer):
    def __init__(self, max_depth_m=3300., deep_gradient=.016):
        self.max_depth_m=float(max_depth_m)
        self.deep_gradient=float(deep_gradient)
        super().__init__(pd.read_csv(TEMPLATE_SSP), use_cache=False)

    def _extend_ssp_to_deep(self):
        """Explicit deep extrapolation scenario; original observations untouched."""
        deepest=float(self.Z[-1]);cap=self.max_depth_m
        if cap<=deepest:return
        grid=np.linspace(deepest,cap,int(np.ceil(cap-deepest))+1)[1:]
        self.V=np.r_[self.V,self.V[-1]+self.deep_gradient*(grid-deepest)]
        self.Z=np.r_[self.Z,grid]
        print(f'[PreciseRayTracer] assumed deep gradient {self.deep_gradient:g} s^-1 to {cap:g} m')

    def varying_offset(self, ship, beacon, offsets):
        """Exact 1-D rays with a separate uniform SSP offset at each event.

        Each field is frozen during propagation. This is not a 3-D ocean model.
        Returns travel times and endpoint/offset derivatives at the same roots.
        """
        ship=np.asarray(ship);beacon=np.asarray(beacon);offsets=np.asarray(offsets)
        source=np.abs(ship[:,2]);receiver=abs(float(beacon[2]))
        if np.any(source>receiver):raise ValueError('invalid_ship_ray: reversed depths')
        self.delta_c=0.
        z,c=self._get_effective_ssp(float(np.min(source)),receiver)
        if np.any(np.min(c)+offsets<=0):raise ValueError('invalid_ship_ray: nonpositive speed')
        difference=beacon[:2]-ship[:,:2];ranges=np.linalg.norm(difference,axis=1)
        hi=np.nextafter(1/(np.max(c)+offsets),0.);lo=np.zeros(len(ship))
        if np.any(_ranges(hi,z,c,source,offsets)<ranges):
            raise ValueError('invalid_ship_ray: no monotone root')
        for _ in range(52):
            mid=(lo+hi)/2;below=_ranges(mid,z,c,source,offsets)<ranges
            lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        p=(lo+hi)/2;x,t,_=_stable_shoot(p,z,c,source,offsets)
        if np.any(abs(x-ranges)>1e-4) or not np.all(np.isfinite(t)):
            raise ValueError('invalid_ship_ray: range closure')
        jac=np.zeros((len(ship),4))
        jac[:,:2]=p[:,None]*difference/np.maximum(ranges[:,None],1e-30)
        jac[:,2]=-np.sqrt(np.maximum(0.,1-(p*(c[-1]+offsets))**2))/(c[-1]+offsets)
        jac[:,3]=_offset_sensitivity(p,z,c,False,source,offsets)
        return t,jac

    def _solve_snell_eigenray(self, range_m, z_src, z_rcv):
        z, c = self._get_effective_ssp(z_src,z_rcv)
        if range_m < 1e-3:
            _,t,s=self._shoot_ray_snell(0.,z,c)
            return t,s
        upper=np.nextafter(1/np.max(c),0.)
        def error(p): return _ranges(np.array([p]),z,c)[0]-range_m
        if error(upper)<0: return np.nan,np.nan
        p=brentq(error,0.,upper,xtol=1e-18,rtol=1e-14)
        reached,t,s=self._shoot_ray_snell(p,z,c)
        if abs(reached-range_m)>1e-4: return np.nan,np.nan
        return t,s

    def _solve_snell_eigenray_batch(self, ranges, z_src, z_rcv):
        ranges=np.atleast_1d(ranges).astype(float)
        if z_rcv<z_src: return np.full(len(ranges),np.nan),np.full(len(ranges),np.nan)
        z,c=self._get_effective_ssp(z_src,z_rcv)
        hi=np.full(len(ranges),np.nextafter(1/np.max(c),0.))
        feasible=_ranges(hi,z,c)>=ranges
        lo=np.zeros(len(ranges))
        for _ in range(52):
            mid=(lo+hi)/2
            below=_ranges(mid,z,c)<ranges
            lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        x,t,s=_stable_shoot((lo+hi)/2,z,c)
        feasible &= np.abs(x-ranges)<1e-4
        return np.where(feasible,t,np.nan),np.where(feasible,s,np.nan)

    def parameters_batch(self,ranges,z_src,z_rcv,sources=None):
        z,c=self._get_effective_ssp(z_src,z_rcv)
        hi=np.full(len(ranges),np.nextafter(1/np.max(c),0.));lo=np.zeros(len(ranges))
        if np.any(_ranges(hi,z,c,sources)<ranges):raise ValueError('no monotone eigenray for derivatives')
        for _ in range(52):
            mid=(lo+hi)/2;below=_ranges(mid,z,c,sources)<ranges
            lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        return (lo+hi)/2,z,c

    def _resolve_batch(self, ship, beacon):
        """Exact individual source depths; grouping only shares the layer grid."""
        ship=np.asarray(ship);beacon=np.asarray(beacon)
        source=np.abs(ship[:,2]);receiver=abs(beacon[2])
        ranges=np.linalg.norm(ship[:,:2]-beacon[:2],axis=1)
        travel=np.full(len(ship),np.nan);length=travel.copy();ok=np.zeros(len(ship),bool)
        for level in np.unique(np.round(source,1)):
            mask=(np.round(source,1)==level)&(source<=receiver)
            if not np.any(mask):continue
            src=source[mask]
            grid,speed=self._get_effective_ssp(float(np.min(src)),receiver)
            limit=_ranges(np.full(len(src),np.nextafter(1/np.max(speed),0.)),grid,speed,src)
            candidate=np.flatnonzero(mask)[limit>=ranges[mask]]
            if not len(candidate):continue
            mask=np.zeros(len(ship),bool);mask[candidate]=True;src=source[mask]
            try:p,z,c=self.parameters_batch(ranges[mask],float(np.min(src)),receiver,src)
            except ValueError:continue
            x,t,s=_stable_shoot(p,z,c,src)
            success=np.abs(x-ranges[mask])<1e-4
            indices=np.flatnonzero(mask)
            travel[indices[success]]=t[success];length[indices[success]]=s[success];ok[indices[success]]=True
        return travel,length,ok

    def endpoint_derivative(self,ship,beacon,offset):
        """Fermat endpoint/slowness derivatives on the accepted monotone branch."""
        self.delta_c=float(offset)
        ship=np.asarray(ship);beacon=np.asarray(beacon)
        jac=np.zeros((len(ship),4));source=np.abs(ship[:,2]);receiver=abs(beacon[2])
        if np.any(source>receiver):raise ValueError('invalid monotone endpoints')
        difference=beacon[:2]-ship[:,:2];ranges=np.linalg.norm(difference,axis=1)
        for level in np.unique(np.round(source,1)):
            mask=np.round(source,1)==level
            src=source[mask]
            p,z,c=self.parameters_batch(ranges[mask],float(np.min(src)),receiver,src)
            jac[mask,:2]=p[:,None]*difference[mask]/np.maximum(ranges[mask,None],1e-30)
            jac[mask,2]=-np.sqrt(np.maximum(0.,1-(p*c[-1])**2))/c[-1]
            jac[mask,3]=_offset_sensitivity(p,z,c,False,src)
        return jac

    def edge_derivative(self,left,right,offset,ray):
        """Derivatives for two endpoints and uniform offset, incl. turning rays."""
        self.delta_c=float(offset)
        left=np.asarray(left);right=np.asarray(right)
        zl,zr=abs(left[2]),abs(right[2]);cl=self._velocity(zl);cr=self._velocity(zr)
        difference=right[:2]-left[:2];horizontal=np.linalg.norm(difference)
        if ray['method']=='snell_monotone':
            p,z,c=self.parameters_batch(np.array([horizontal]),min(zl,zr),max(zl,zr));p=float(p[0])
            speed=float(_offset_sensitivity(np.array([p]),z,c)[0])
            vertical_left=(1 if zl<zr else -1)*math.sqrt(max(0.,1-(p*cl)**2))/cl
            vertical_right=(1 if zr<zl else -1)*math.sqrt(max(0.,1-(p*cr)**2))/cr
        elif ray['method']=='snell_turning':
            vertex=ray['vertex_m'];p=1/self._velocity(vertex);speed=0.
            for endpoint in (zl,zr):
                z,c=self._get_effective_ssp(endpoint,vertex)
                speed+=float(_offset_sensitivity(np.array([p]),z,c,True)[0])
            vertical_left=math.sqrt(max(0.,1-(p*cl)**2))/cl
            vertical_right=math.sqrt(max(0.,1-(p*cr)**2))/cr
        else:raise ValueError('no derivative for fallback ray')
        direction=p*difference/max(horizontal,1e-30)
        return np.r_[-direction,vertical_left,direction,vertical_right,speed]

    @staticmethod
    def _shoot_ray_snell(p, z, c):
        arrays=_stable_shoot(np.atleast_1d(p).astype(float),np.asarray(z,float),np.asarray(c,float))
        return tuple(float(a[0]) for a in arrays)

    @staticmethod
    def _shoot_ray_snell_batch(p, z, c):
        return _stable_shoot(np.atleast_1d(p).astype(float),np.asarray(z,float),np.asarray(c,float))
