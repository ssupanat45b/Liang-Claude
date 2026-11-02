"""Nonlinear 2-D magnetostatic finite-element solver (A_z formulation).

    -div( nu(|B|) grad A ) = J ,   B = curl(A e_z) ,   A = 0 on the stator outer surface

Linear (first-order) triangles; Newton-Raphson iteration on the nonlinear
reluctivity with backtracking line search.  Flux linkages are obtained from
the average vector potential over each coil side; torque from Arkkio's
air-gap Maxwell-stress method.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from . import machine as mc
from .materials import NU0, ROTOR_FORGING, STATOR_LAMINATION


@dataclass
class Solution:
    A: np.ndarray
    Bx: np.ndarray
    By: np.ndarray
    iterations: int
    psi_abc: np.ndarray          # phase flux linkages [Wb-turn] (stack length incl.)
    psi_f: float                 # field flux linkage
    torque: float                # N m (CCW positive, acting on the rotor)
    k_shift: int

    @property
    def Bmag(self):
        return np.hypot(self.Bx, self.By)


class MagnetostaticModel:
    def __init__(self, mesh: mc.Mesh):
        self.mesh = mesh
        self.d = mesh.design
        self._cache: dict[int, dict] = {}

    # ------------------------------------------------------------------
    def _setup(self, k: int) -> dict:
        if k in self._cache:
            return self._cache[k]
        m, d = self.mesh, self.d
        tri, xy, reg, part, nn, outer = m.connectivity(k)
        x, y = xy[..., 0], xy[..., 1]
        b = np.c_[y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]]
        c = np.c_[x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]]
        area = 0.5 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
        if np.any(area <= 0):
            raise RuntimeError("mesh contains inverted elements")
        G = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4 * area**2)[:, None, None]
        rows = np.repeat(tri, 3, axis=1).ravel()
        cols = np.tile(tri, (1, 3)).ravel()

        # --- coil coupling vectors: F = sum_x I_x c_x ;  psi_x = L * c_x . A
        ne = len(tri)
        w_ph = np.zeros((3, ne))
        lay = mc.winding_layout(d)
        s_off = len(m.r_tri)
        tag = m.s_tag
        side_area = {}
        for e in np.nonzero(tag >= 0)[0]:
            side_area[tag[e]] = side_area.get(tag[e], 0.0) + area[s_off + e]
        for e in np.nonzero(tag >= 0)[0]:
            slot, layer = divmod(int(tag[e]), 2)
            ph, sg = lay[layer][slot]
            w_ph[ph, s_off + e] = sg * d.turns_per_coil / (d.parallel_paths * side_area[tag[e]])
        w_f = np.zeros(ne)
        rt = m.r_tag
        fs = mc.field_sign(d, np.arange(d.Qr_pitches))
        slot_area = {}
        for e in np.nonzero(rt >= 0)[0]:
            slot_area[rt[e]] = slot_area.get(rt[e], 0.0) + area[e]
        for e in np.nonzero(rt >= 0)[0]:
            # each slot side holds Nf conductors; summing sign*Nf*<A> over all sides gives
            # the linkage of the complete (series-connected) field winding
            w_f[e] = fs[rt[e]] * d.Nf_per_slot / slot_area[rt[e]]

        def to_nodes(w):
            v = np.zeros(nn)
            np.add.at(v, tri.ravel(), np.repeat(w * area / 3, 3))
            return v

        c_ph = np.array([to_nodes(w_ph[i]) for i in range(3)])
        c_f = to_nodes(w_f)

        free = np.ones(nn, bool); free[outer] = False
        # gap elements for Arkkio torque
        cen = xy.mean(axis=1)
        rc = np.hypot(cen[:, 0], cen[:, 1])
        gap = (rc > d.R_rotor) & (rc < d.R_bore)
        iron_r = reg == mc.ROTOR_IRON
        iron_s = reg == mc.STATOR_IRON
        setup = dict(tri=tri, xy=xy, reg=reg, part=part, nn=nn, b=b, c=c, area=area, G=G,
                     rows=rows, cols=cols, c_ph=c_ph, c_f=c_f, free=free, gap=gap,
                     cen=cen, iron_r=iron_r, iron_s=iron_s, side_area=side_area,
                     slot_area=slot_area)
        self._cache[k] = setup
        return setup

    # ------------------------------------------------------------------
    def _B(self, s, A):
        a = A[s["tri"]]
        dAdx = (s["b"] * a).sum(1) / (2 * s["area"])
        dAdy = (s["c"] * a).sum(1) / (2 * s["area"])
        return dAdy, -dAdx

    def _nu(self, s, b2, linear_mu_r=None):
        nu = np.full(len(b2), NU0)
        dnu = np.zeros(len(b2))
        for mask, mat in ((s["iron_r"], ROTOR_FORGING), (s["iron_s"], STATOR_LAMINATION)):
            if linear_mu_r is not None:
                nu[mask] = NU0 / linear_mu_r
            else:
                nu[mask], dnu[mask] = mat.nu(b2[mask])
        return nu, dnu

    def solve(self, i_abc=(0.0, 0.0, 0.0), i_f=0.0, k_shift=0, linear_mu_r=None,
              tol=1e-6, max_iter=40, A0=None) -> Solution:
        s = self._setup(k_shift)
        F = np.asarray(i_abc, float) @ s["c_ph"] + i_f * s["c_f"]
        free = s["free"]
        A = np.zeros(s["nn"]) if A0 is None else A0.copy()
        G, area = s["G"], s["area"]

        def residual(A):
            a = A[s["tri"]]
            Ga = np.einsum("eij,ej->ei", G, a)
            b2 = (a * Ga).sum(1)
            nu, dnu = self._nu(s, b2, linear_mu_r)
            re = (nu * area)[:, None] * Ga
            R = np.zeros(s["nn"])
            np.add.at(R, s["tri"].ravel(), re.ravel())
            return R - F, Ga, nu, dnu

        it = 0
        R, Ga, nu, dnu = residual(A)
        rnorm0 = max(np.linalg.norm(F[free]), 1e-30)
        for it in range(1, max_iter + 1):
            Ke = (nu * area)[:, None, None] * G \
                + (2 * area * dnu)[:, None, None] * Ga[:, :, None] * Ga[:, None, :]
            K = sp.csr_matrix((Ke.ravel(), (s["rows"], s["cols"])), shape=(s["nn"],) * 2)
            Kff = K[free][:, free]
            dA = np.zeros_like(A)
            dA[free] = spsolve(Kff.tocsc(), -R[free])
            # backtracking line search on the residual norm
            r_old = np.linalg.norm(R[free])
            step = 1.0
            for _ in range(8):
                An = A + step * dA
                Rn, Gan, nun, dnun = residual(An)
                if np.linalg.norm(Rn[free]) < r_old or step < 0.02:
                    break
                step *= 0.5
            A, R, Ga, nu, dnu = An, Rn, Gan, nun, dnun
            if linear_mu_r is not None:
                break
            if (np.linalg.norm(step * dA) <= tol * max(np.linalg.norm(A), 1e-30)
                    and np.linalg.norm(R[free]) <= 1e-4 * rnorm0):
                break
        Bx, By = self._B(s, A)
        L = self.d.L_stack
        psi_abc = L * s["c_ph"] @ A
        psi_f = L * (s["c_f"] @ A)
        return Solution(A, Bx, By, it, psi_abc, psi_f, self.torque(s, Bx, By), k_shift)

    # ------------------------------------------------------------------
    def torque(self, s, Bx, By) -> float:
        """Arkkio: T = L/(mu0 (ro-ri)) * sum_gap r Br Btheta dS (CCW on rotor)."""
        g = s["gap"]
        cx, cy = s["cen"][g, 0], s["cen"][g, 1]
        r = np.hypot(cx, cy)
        ct, st = cx / r, cy / r
        Br = Bx[g] * ct + By[g] * st
        Bt = -Bx[g] * st + By[g] * ct
        return float(self.d.L_stack * NU0 / self.d.gap * np.sum(r * Br * Bt * s["area"][g]))

    def gap_flux_density(self, sol: Solution, radius=None, npts=720):
        """Radial/tangential B on a circle in the stator-side air gap (stator frame)."""
        s = self._setup(sol.k_shift)
        d = self.d
        if radius is None:
            radius = 0.5 * (d.R_slide + d.R_bore)
        th = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        cen = s["cen"]
        rc = np.hypot(cen[:, 0], cen[:, 1])
        tc = np.mod(np.arctan2(cen[:, 1], cen[:, 0]), 2 * np.pi)
        sel = np.nonzero((s["part"] == 1) & (rc > d.R_slide) & (rc < d.R_bore))[0]
        # pick, for every sample angle, the element of the layer nearest to 'radius'
        layer_r = rc[sel]
        best = sel[np.abs(layer_r - radius) <= np.abs(layer_r - radius).min() + 1e-4]
        order = np.argsort(tc[best])
        tb = tc[best][order]
        Bx, By = sol.Bx[best][order], sol.By[best][order]
        Br_e = Bx * np.cos(tb) + By * np.sin(tb)
        Bt_e = -Bx * np.sin(tb) + By * np.cos(tb)
        Br = np.interp(th, tb, Br_e, period=2 * np.pi)
        Bt = np.interp(th, tb, Bt_e, period=2 * np.pi)
        return th, Br, Bt

    def element_data(self, k_shift=0):
        return self._setup(k_shift)
