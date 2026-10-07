"""MODEL sections 1--4: scaled nonlinear MAP with fixed observation support.

Independent bottom sound speed and edge delays are explicit nuisance parameters
when inter-beacon data are used. No straight-line fallback enters the fit.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares

from risk_design.estimation import predict_mean
from risk_design.parameters import unpack_beacons, prior_precision


def whiten(covariance, values):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 1:
        return values / np.sqrt(covariance).reshape((-1,) + (1,) * (values.ndim - 1))
    return solve_triangular(cholesky(covariance, lower=True), values, lower=True)


class JointProblem:
    """14 ship parameters; optional bottom offset and three uncertain edge delays.

    Priors are declared assumptions. They must be calibrated before field claims.
    The ray model assumes the supplied water column; bottom clearance is not
    established by the deepest beacon alone.
    """
    def __init__(self, model, ship, beacons, times, covariance, edge_covariance=None,
                 bottom_sigma=2.0, edge_bias_sigma=0.002, t0=None):
        self.model = model
        self.ship = np.asarray(ship)
        self.beacons = np.asarray(beacons)
        self.times = np.asarray(times)
        self.t0 = float(np.median(times) if t0 is None else t0)
        self.covariance = np.asarray(covariance)
        self.edge_covariance = None if edge_covariance is None else np.asarray(edge_covariance)
        self._ship_chol = None if self.covariance.ndim == 1 else cholesky(self.covariance, lower=True)
        self._edge_chol = None if self.edge_covariance is None or self.edge_covariance.ndim == 1 else cholesky(self.edge_covariance, lower=True)
        self.n = 14 if edge_covariance is None else 18
        self.scales = np.array([100.] * 9 + [1., .001, .001, .001, .001] +
                               ([] if self.n == 14 else [1., .001, .001, .001]))
        self.steps = np.array([.01] * 9 + [.001, 1e-6, 1e-6, 1e-6, 1e-6] +
                              ([] if self.n == 14 else [.001, 1e-6, 1e-6, 1e-6]))
        self.precision = np.r_[np.zeros(9), prior_precision(),
                               [] if self.n == 14 else [bottom_sigma**-2] + [edge_bias_sigma**-2]*3]
        self.center = np.zeros(self.n)
        self.calls = 0

    def edge_mean(self, theta):
        rows = self.model.inter_beacon(unpack_beacons(theta[:9]), theta[14])
        if not all(row['snell_ok'] and np.isfinite(row['delay_s']) for row in rows):
            raise ValueError('invalid_interbeacon_ray: no fallback permitted')
        return np.array([row['delay_s'] for row in rows]) + theta[15:18]

    def mean(self, theta):
        self.calls += 1
        pred, ok, _ = predict_mean(self.model, theta[:9], theta[9:14], self.ship,
                                  self.beacons, self.times, self.t0)
        if not np.all(ok) or not np.all(np.isfinite(pred)):
            raise ValueError('invalid_ship_ray: observation support must not change')
        return pred if self.n == 14 else np.r_[pred, self.edge_mean(theta)]

    def derivative(self, theta):
        # Coordinate and uniform-SSP central differences; clock columns exact.
        raw = np.zeros((len(self.times), 14))
        positions = unpack_beacons(theta[:9])
        for index, code in enumerate((2, 5, 6)):
            mask = self.beacons == code
            ship = self.ship[mask]
            if hasattr(self.model.tracer, 'endpoint_derivative'):
                endpoint = self.model.tracer.endpoint_derivative(ship,positions[code],theta[9])
                raw[mask,3*index:3*index+3]=endpoint[:,:3]
                raw[mask,9]=endpoint[:,3]
                continue
            for axis in range(4):
                col = 3*index+axis if axis < 3 else 9
                step = self.steps[col]
                vals = []
                for sign in (1, -1):
                    position = positions[code].copy(); offset = theta[9]
                    if axis < 3: position[axis] += sign*step
                    else: offset += sign*step
                    result = self.model.ship(ship, position, offset)
                    if not np.all(result['snell_ok']):
                        raise ValueError('invalid_derivative_ray: no zero-column substitution')
                    vals.append(result['delay_s'])
                raw[mask, col] = (vals[0]-vals[1])/(2*step)
        raw[:, 10] = 1
        raw[:, 11] = (self.times-self.t0)/3600
        raw[:, 12] = self.beacons == 2
        raw[:, 13] = self.beacons == 6
        if self.n == 14:
            return raw
        jac = np.zeros((len(self.times)+3, self.n)); jac[:len(self.times), :14] = raw
        if hasattr(self.model.tracer, 'edge_derivative'):
            rows=self.model.inter_beacon(positions,theta[14])
            for i,((left,right),ray) in enumerate(zip(((5,2),(5,6),(2,6)),rows)):
                if not ray['snell_ok']:raise ValueError('invalid interbeacon derivative')
                endpoint=self.model.tracer.edge_derivative(positions[left],positions[right],theta[14],ray)
                li=(2,5,6).index(left)*3;ri=(2,5,6).index(right)*3
                jac[len(self.times)+i,li:li+3]=endpoint[:3]
                jac[len(self.times)+i,ri:ri+3]=endpoint[3:6]
                jac[len(self.times)+i,14]=endpoint[6]
            jac[-3:,15:18]=np.eye(3)
            return jac
        for col in list(range(9)) + [14]:
            d = np.zeros(self.n); d[col] = self.steps[col]
            jac[-3:, col] = (self.edge_mean(theta+d)-self.edge_mean(theta-d))/(2*self.steps[col])
        jac[-3:, 15:18] = np.eye(3)
        return jac

    def white(self, values):
        n = len(self.times)
        first = (whiten(self.covariance, values[:n]) if self._ship_chol is None else
                 solve_triangular(self._ship_chol, values[:n], lower=True))
        if self.edge_covariance is None:
            return first
        last = (whiten(self.edge_covariance, values[n:]) if self._edge_chol is None else
                solve_triangular(self._edge_chol, values[n:], lower=True))
        return np.concatenate([first, last], axis=0)

    def fit(self, observed, initial, max_nfev=100, stationarity_tol=1e-4,
            loss='linear', f_scale=1.0):
        observed = np.asarray(observed); initial = np.asarray(initial)
        root_prior = np.sqrt(self.precision)
        history = []
        rejected_steps = []
        # Invalid starting data/model support is an error, not an optimization step.
        self.mean(initial)

        def residual(z):
            theta = initial + self.scales*z
            try:
                prediction = self.mean(theta)
            except ValueError as exc:
                if not str(exc).startswith(('invalid_ship_ray:', 'invalid_interbeacon_ray:')):
                    raise
                # TRF rejects a nonfinite trial residual and contracts its radius.
                # Keep fixed observation support; never substitute chord travel time.
                rejected_steps.append(dict(reason=str(exc), theta=theta.copy()))
                return np.full(len(observed)+self.n, np.inf)
            r = np.r_[self.white(prediction-observed), root_prior*(theta-self.center)]
            history.append(float(r@r))
            return r

        def derivative(z):
            theta = initial+self.scales*z
            return np.vstack([self.white(self.derivative(theta)), np.diag(root_prior)])*self.scales

        result = least_squares(residual, np.zeros(self.n), jac=derivative, method='trf',
                               x_scale='jac', loss=loss, f_scale=f_scale, max_nfev=max_nfev,
                               ftol=1e-11, xtol=1e-11, gtol=1e-8)
        theta = initial+self.scales*result.x
        jw = self.white(self.derivative(theta))
        aug = np.vstack([jw, np.diag(root_prior)])*self.scales
        r = residual(result.x)
        # Gradient in column-normalized units: invariant to physical parameter units.
        gradient = aug.T@r
        stationarity = float(np.max(np.abs(gradient)/np.maximum(np.linalg.norm(aug, axis=0), 1e-30)))
        _, singular, vt = np.linalg.svd(aug, full_matrices=False)
        rank = int(np.sum(singular > singular[0]*1e-12))
        covariance = None
        if rank == self.n and loss == 'linear':
            inverse = (vt.T/singular**2)@vt
            covariance = self.scales[:, None]*inverse*self.scales[None, :]
        return dict(theta=theta, success=bool(result.success and rank == self.n and stationarity <= stationarity_tol and loss=='linear'),
                    inference_valid=bool(loss=='linear'),loss=loss,
                    optimizer_success=bool(result.success), status=int(result.status), message=str(result.message),
                    stationarity=stationarity, rank=rank, nfev=result.nfev, cost=2*result.cost,
                    residual_s=self.mean(theta)-observed, covariance=covariance,
                    jacobian=self.derivative(theta), history=history,
                    rejected_steps=rejected_steps)


def sampling_risk(problem, theta, q, mismatch=None, true_covariance=None):
    """Fixed-truth sampling MSE, including nonzero prior-center bias (MODEL 4)."""
    jac = problem.derivative(theta)
    jw = problem.white(jac)
    sc = problem.scales
    a = np.vstack([jw, np.diag(np.sqrt(problem.precision))])*sc
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    if s[-1] <= s[0]*1e-12:
        raise ValueError('non_estimable_joint_problem')
    covariance = sc[:, None]*((vt.T/s**2)@vt)*sc[None, :]
    kw = covariance@jw.T
    mis = np.zeros(jac.shape[0]) if mismatch is None else np.asarray(mismatch)
    bias = kw@problem.white(mis)-covariance@(problem.precision*(theta-problem.center))
    if true_covariance is None:
        freq = kw@kw.T
    else:
        w = problem.white(np.eye(jac.shape[0]))
        k = kw@w
        freq = k@true_covariance@k.T
    q = np.atleast_2d(q)
    return dict(variance=float(np.trace(q@freq@q.T)), bias_squared=float(np.sum((q@bias)**2)),
                mse=float(np.trace(q@freq@q.T)+np.sum((q@bias)**2)),
                posterior_risk=float(np.trace(q@covariance@q.T)), covariance=covariance)
