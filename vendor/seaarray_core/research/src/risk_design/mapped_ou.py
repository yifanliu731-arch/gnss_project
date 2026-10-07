"""Frozen-reference ray sensitivity mapped common OU sound-speed process.

Cov(y_i,y_j)=h_i h_j sigma_c^2 exp(-abs(t_i-t_j)/tau)+sigma_w^2 I.
h is dT/dc (s per m/s), frozen at a training-only nonlinear reference.
The stationary zero-mean process has a proper prior; no free duplicate offset.
"""
import numpy as np
from numba import njit
from risk_design.joint import JointProblem


@njit(cache=True)
def factors(t, h, sw, sc, tau):
    n=len(t); phi=np.ones(n); gain=np.zeros(n); sd=np.zeros(n); p=sc*sc
    for i in range(n):
        if i:
            phi[i]=np.exp(-(t[i]-t[i-1])/tau)
            p=phi[i]**2*p+sc*sc*(1-phi[i]**2)
        v=h[i]*h[i]*p+sw*sw
        sd[i]=np.sqrt(v); gain[i]=p*h[i]/v
        p=max(0.,p-gain[i]*h[i]*p)
    return phi,gain,sd


@njit(cache=True)
def apply_white(x,h,phi,gain,sd):
    out=np.empty_like(x); state=np.zeros(x.shape[1])
    for i in range(len(h)):
        state=phi[i]*state
        e=x[i]-h[i]*state
        out[i]=e/sd[i]
        state+=gain[i]*e
    return out


class MappedWhitener:
    def __init__(self,t,h,sw,sc,tau):
        self.order=np.argsort(t,kind='stable');self.h=np.asarray(h)[self.order]
        if sw<=0 or sc<0 or tau<=0:raise ValueError('invalid covariance')
        self.phi,self.gain,self.sd=factors(np.asarray(t)[self.order],self.h,sw,sc,tau)
        self.logdet=float(2*np.log(self.sd).sum())
    def __call__(self,x):
        x=np.asarray(x);vector=x.ndim==1
        if vector:x=x[:,None]
        out=np.empty_like(x);out[self.order]=apply_white(x[self.order],self.h,self.phi,self.gain,self.sd)
        return out[:,0] if vector else out


class MappedJointProblem(JointProblem):
    def __init__(self,base,h,sw,sc,tau):
        if base.n!=14:raise ValueError('ship-only development model')
        self.__dict__.update(base.__dict__);self.base=base
        self.mapped_white=MappedWhitener(base.times,h,sw,sc,tau)
    def mean(self,theta):return self.base.mean(theta)
    def derivative(self,theta):return self.base.derivative(theta)
    def white(self,x):return self.mapped_white(x)


@njit(cache=True)
def smooth(t,h,y,observed,sw,sc,tau):
    n=len(t);mp=np.zeros_like(y);mf=np.zeros_like(y)
    pp=np.zeros(n);pf=np.zeros(n);phi=np.ones(n)
    for i in range(n):
        if i:phi[i]=np.exp(-(t[i]-t[i-1])/tau)
        pp[i]=sc*sc if i==0 else phi[i]**2*pf[i-1]+sc*sc*(1-phi[i]**2)
        if i:mp[i]=phi[i]*mf[i-1]
        gain=pp[i]*h[i]/(h[i]*h[i]*pp[i]+sw*sw) if observed[i] else 0.
        mf[i]=mp[i]+gain*(y[i]-h[i]*mp[i]);pf[i]=max(0.,pp[i]-gain*h[i]*pp[i])
    ms=mf.copy();ps=pf.copy()
    for i in range(n-2,-1,-1):
        gain=pf[i]*phi[i+1]/pp[i+1] if pp[i+1]>0 else 0.
        ms[i]+=gain*(ms[i+1]-mp[i+1]);ps[i]+=gain*gain*(ps[i+1]-pp[i+1])
    return ms,ps


def conditional(t,h,values,nt,nh,sw,sc,tau):
    values=np.asarray(values);vector=values.ndim==1
    if vector:values=values[:,None]
    n=len(t);order=np.argsort(np.r_[t,nt],kind='stable')
    y=np.zeros((n+len(nt),values.shape[1]));y[:n]=values
    hh=np.r_[h,nh][order]
    ms,ps=smooth(np.r_[t,nt][order],hh,y[order],order<n,sw,sc,tau)
    mask=order>=n;idx=order[mask]-n
    mean=np.empty((len(nt),values.shape[1]));var=np.empty(len(nt))
    mean[idx]=hh[mask,None]*ms[mask];var[idx]=np.maximum(ps[mask],0)*hh[mask]**2+sw*sw
    return (mean[:,0] if vector else mean),var
