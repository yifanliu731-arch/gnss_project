"""One profile mode with a common OU coefficient and constant horizontal gradients.

Gradient coefficients have proper Gaussian priors and are marginalized exactly.
The static 14-parameter mean remains nonlinear; ray sensitivities are frozen at
a training-only reference. No independent bias penalty duplicates this covariance.
"""
import numpy as np
from numba import njit
from risk_design.mapped_ou import MappedWhitener, MappedJointProblem, conditional
from risk_design.fast_precision import solve_parameters


@njit(cache=True)
def _integrals(p, source, z, c, mode, ship, beacon, nodes, weights, center, length):
    out = np.zeros((len(p), 3))
    for k in range(len(p)):
        distance = np.sqrt(np.sum((beacon[:2] - ship[k, :2]) ** 2))
        direction = (beacon[:2] - ship[k, :2]) / max(distance, 1e-30)
        horizontal = 0.
        for i in range(len(z) - 1):
            if source[k] >= z[i + 1]:
                continue
            za = max(z[i], source[k]); zb = z[i + 1]
            frac0 = (za-z[i])/(zb-z[i])
            ca = c[i] + (c[i+1]-c[i])*frac0; cb = c[i+1]
            ma = mode[i] + (mode[i+1]-mode[i])*frac0; mb = mode[i+1]
            cosa = np.sqrt(1-(p[k]*ca)**2)
            cosb = np.sqrt(1-(p[k]*cb)**2)
            for j in range(len(nodes)):
                f = (nodes[j]+1)/2; dz = f*(zb-za)
                cq = ca+(cb-ca)*f; mq = ma+(mb-ma)*f
                cosq = np.sqrt(1-(p[k]*cq)**2)
                x = horizontal+p[k]*(ca+cq)*dz/(cosa+cosq)
                factor = -weights[j]*(zb-za)/2*mq/(cq*cq*cosq)
                out[k, 0] += factor
                out[k, 1] += factor*(ship[k,0]+direction[0]*x-center[0])/length
                out[k, 2] += factor*(ship[k,1]+direction[1]*x-center[1])/length
            horizontal += p[k]*(ca+cb)*(zb-za)/(cosa+cosb)
    return out


def profile_mapping(problem, theta, profile_z, profile_mode, center, length=5000., order=8):
    """Integrate the piecewise-linear mode on the existing curved reference ray.

    Retain the tracer's profile discretization for consistency with nonlinear
    profile perturbation checks. Endpoint continuation is a declared assumption.
    """
    if length <= 0:
        raise ValueError('positive spatial scale required')
    tracer = problem.model.tracer
    tracer_mode = np.interp(tracer.Z, profile_z, profile_mode)
    out = np.zeros((len(problem.times), 3))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for j, beacon in enumerate((2, 5, 6)):
        mask = problem.beacons == beacon
        if not mask.any():
            continue
        ship = problem.ship[mask]; position = theta[3*j:3*j+3]
        source = np.abs(ship[:, 2]); tracer.delta_c = 0.
        z, c = tracer._get_effective_ssp(float(source.min()), abs(float(position[2])))
        mode = np.interp(z, tracer.Z, tracer_mode)
        c = c + theta[9]
        p = solve_parameters(np.linalg.norm(ship[:,:2]-position[:2],axis=1), source,z,c,np.zeros(len(ship)))
        if not np.isfinite(p).all():
            raise ValueError('invalid profile-gradient reference ray')
        out[mask] = _integrals(p,source,z,c,mode,ship,position,nodes,weights,np.asarray(center),length)
    return out


class ProfileGradientWhitener:
    def __init__(self, times, h, sw, sa, tau, sg):
        if sg < 0:
            raise ValueError('negative gradient prior scale')
        self.ou = MappedWhitener(times, h[:,0], sw, sa, tau)
        self.g = np.asarray(h[:,1:]); self.sg = sg
        self.wg = self.ou(self.g)
        u, s, _ = np.linalg.svd(self.wg*sg, full_matrices=False)
        self.u = u; self.adjust = 1/np.sqrt(1+s*s)-1
        self.logdet = self.ou.logdet + float(np.log1p(s*s).sum())
        self.gradient_covariance = sg*sg*np.linalg.inv(np.eye(2)+sg*sg*(self.wg.T@self.wg))

    def __call__(self, values):
        w = self.ou(values)
        if w.ndim == 1:
            return w + self.u@(self.adjust*(self.u.T@w))
        return w + self.u@(self.adjust[:,None]*(self.u.T@w))


class ProfileGradientProblem(MappedJointProblem):
    def __init__(self, base, h, sw, sa, tau, sg):
        if base.n != 14:
            raise ValueError('ship-only experiment')
        self.__dict__.update(base.__dict__); self.base = base
        self.mapped_white = ProfileGradientWhitener(base.times,h,sw,sa,tau,sg)


def predict(times, h, values, new_times, new_h, sw, sa, tau, sg):
    """Exact conditional Gaussian prediction including marginalized gradients."""
    values = np.asarray(values); vector = values.ndim == 1
    if vector:
        values = values[:,None]
    combined, variance = conditional(times,h[:,0],np.c_[values,h[:,1:]],new_times,new_h[:,0],sw,sa,tau)
    w = ProfileGradientWhitener(times,h,sw,sa,tau,sg)
    vg = w.gradient_covariance
    coefficient = vg@w.wg.T@w.ou(values)
    remaining = new_h[:,1:]-combined[:,-2:]
    mean = combined[:,:values.shape[1]]+remaining@coefficient
    variance = variance+np.einsum('ij,jk,ik->i',remaining,vg,remaining)
    return (mean[:,0] if vector else mean), variance
