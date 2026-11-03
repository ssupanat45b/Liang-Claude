"""Constrained multi-objective Bayesian optimisation (2 objectives, minimisation).

* Objective 1 (active mass) is analytic and exact -> no surrogate needed.
* Objective 2 (EM losses) and the FEM constraints (Q, magnet B at short circuit,
  tooth B) are modelled with Gaussian processes (Matern-5/2, ARD, noise term).
* Acquisition: expected hypervolume improvement (Monte-Carlo over the GP of
  objective 2) x probability of feasibility.  Before the first feasible design
  is found the acquisition is the probability of feasibility alone.
* Batch of q points per iteration by greedy selection with a "Kriging believer"
  update of the Pareto front and a distance penalty, so the q FEM runs can go
  in parallel.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------------- Pareto / HV
def pareto_mask(F):
    """Non-dominated rows of F (minimisation)."""
    F = np.asarray(F, float)
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            keep[i] = False
    return keep


def hypervolume_2d(F, ref):
    """Hypervolume dominated by the 2-D point set F w.r.t. the reference point (minimisation)."""
    F = np.asarray(F, float)
    if len(F) == 0:
        return 0.0
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[pareto_mask(F)]
    F = F[np.argsort(F[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in F:
        hv += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return float(hv)


# ----------------------------------------------------------------------------- GP
class GP:
    def __init__(self, log=False):
        self.log = log
        self.model = None

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        y = np.asarray(y, float)
        if self.log:
            y = np.log(np.maximum(y, 1e-9))
        d = X.shape[1]
        k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(d, 0.5),
                                                       length_scale_bounds=(0.03, 30.0), nu=2.5) \
            + WhiteKernel(1e-4, (1e-8, 1e-1))
        self.model = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=3,
                                              random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        return self

    def predict(self, X):
        mu, sd = self.model.predict(X, return_std=True)
        sd = np.maximum(sd, 1e-9)
        if self.log:                     # lognormal moments of the back-transformed variable
            m = np.exp(mu + 0.5 * sd**2)
            s = np.sqrt((np.exp(sd**2) - 1) * np.exp(2 * mu + sd**2))
            return m, s
        return mu, sd


# ----------------------------------------------------------------------------- acquisition
def ehvi_mc(f1, mu2, sd2, front, ref, n_mc=48, rng=None):
    """Expected hypervolume improvement for candidates with exact f1 and Gaussian f2."""
    rng = rng or np.random.default_rng(0)
    base = hypervolume_2d(front, ref) if len(front) else 0.0
    z = rng.standard_normal(n_mc)
    out = np.zeros(len(f1))
    for i in range(len(f1)):
        if f1[i] >= ref[0]:
            continue
        samples = mu2[i] + sd2[i] * z
        hv = 0.0
        for y in samples:
            if y >= ref[1]:
                continue
            pts = np.vstack([front, [[f1[i], y]]]) if len(front) else np.array([[f1[i], y]])
            hv += hypervolume_2d(pts, ref) - base
        out[i] = hv / n_mc
    return out


def prob_feasible(models, X, limits):
    """Product of per-constraint probabilities.  models: dict name -> GP."""
    p = np.ones(len(X))
    mu, sd = models["Q"].predict(X)
    p *= norm.cdf((mu - limits["Q_min"]) / sd) - norm.cdf((mu - limits["Q_max"]) / sd)
    mu, sd = models["Bd"].predict(X)
    p *= norm.cdf((mu - limits["Bd_min"]) / sd)
    mu, sd = models["Bt"].predict(X)
    p *= norm.cdf((limits["B_tooth_max"] - mu) / sd)
    return np.clip(p, 0, 1)


def select_batch(U, f1, models, front, ref, limits, q, min_dist=0.08, rng=None):
    """Greedy batch: maximise EHVI*PoF, believer update of the front, distance penalty."""
    pof = prob_feasible(models, U, limits)
    mu2, sd2 = models["P"].predict(U)
    chosen = []
    front = np.array(front, float).reshape(-1, 2)
    for _ in range(q):
        if len(front) == 0:
            acq = pof.copy()
        else:
            # pre-screen with a cheap optimistic bound, then exact MC-EHVI on the best few hundred
            opt = np.maximum(0.0, ref[1] - (mu2 - 2 * sd2)) * np.maximum(0.0, ref[0] - f1) * pof
            idx = np.argsort(-opt)[:300]
            acq = np.zeros(len(U))
            acq[idx] = ehvi_mc(f1[idx], mu2[idx], sd2[idx], front, ref, rng=rng) * pof[idx]
            if acq.max() <= 0:                           # nothing improves: explore feasibility
                acq = pof * sd2
        for c in chosen:                                 # keep the batch diverse
            dist = np.linalg.norm(U - U[c], axis=1)
            acq[dist < min_dist] = -1
        j = int(np.argmax(acq))
        if acq[j] < 0:
            break
        chosen.append(j)
        if pof[j] > 0.5:                                 # Kriging believer
            front = np.vstack([front, [[f1[j], mu2[j]]]])
            front = front[pareto_mask(front)]
    return chosen
