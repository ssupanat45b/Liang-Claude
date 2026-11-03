"""Nonlinear 2-D magnetostatic finite-element solver (A_z formulation).

    -div( nu(|B|) grad A ) = J + curl(nu_m Br) ,   B = curl(A e_z) ,   A = 0 on the stator outer surface

Permanent magnets are linear (recoil permeability mu_rm, remanence Br(T)), radially magnetised.

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

import os
import warnings

# ---------------------------------------------------------------------------
# Linear-solver backends (select with set_backend() or the env. variable PM_FEM_SOLVER)
#   gpu      : NVIDIA GPU via CuPy - Jacobi-preconditioned conjugate gradient on the
#              device (install:  pip install cupy-cuda12x ).  The FE Jacobian is
#              symmetric positive definite (monotone B-H curves), so CG is exact up to tol.
#   pardiso  : Intel MKL PARDISO via 'pypardiso' (multithreaded CPU direct solver)
#   superlu  : SciPy built-in direct solver (always available)
#   auto     : pardiso if installed, else superlu (CPU).  Measured on an RTX 4060 vs a
#              Core Ultra 7 265K, the sparse direct CPU solver is 2-8x faster than GPU PCG for
#              these 2-D meshes (10k-100k nodes), so the GPU is used only when requested.
# Any GPU failure (no device, out of memory, CG not converged) falls back to the CPU.
# ---------------------------------------------------------------------------
_state = {"backend": None, "cp": None, "csp": None, "cspl": None, "pardiso": None,
          "gpu_name": "", "gpu_error": "", "fallbacks": 0, "cg_iters": 0}


def _try_gpu():
    try:
        import warnings as _w
        with _w.catch_warnings():                           # CuPy warns about CUDA_PATH even when
            _w.simplefilter("ignore")                       # pip-installed CUDA libraries work
            import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as cspl
        if cp.cuda.runtime.getDeviceCount() < 1:
            _state["gpu_error"] = "no CUDA device found"
            return False
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        # smoke test: a real sparse kernel (fails here if the CUDA libraries cannot be loaded)
        csp.identity(4, format="csr").dot(cp.ones(4))
        _state.update(cp=cp, csp=csp, cspl=cspl, gpu_name=name, gpu_error="")
        return True
    except ImportError as e:
        _state["gpu_error"] = f"CuPy not installed ({e})"
        return False
    except Exception as e:                                  # noqa: BLE001
        _state["gpu_error"] = f"{type(e).__name__}: {e}"
        return False


def gpu_help() -> str:
    """Why the GPU is not used, and how to fix it ('' if the GPU is active)."""
    if _state["backend"] == "gpu" or not _state["gpu_error"]:
        return ""
    return ("GPU not used -> " + _state["gpu_error"] + "\n"
            "  fix: pip install \"cupy-cuda12x[ctk]\"   (adds the CUDA 12 runtime libraries)\n"
            "       or install the CUDA Toolkit 12.x and set CUDA_PATH; update the NVIDIA driver;\n"
            "  test: python -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount(), cupy.arange(3) * 2)\"")


def _try_pardiso():
    try:
        import pypardiso
        _state["pardiso"] = pypardiso
        return True
    except Exception:                                       # noqa: BLE001
        return False


def set_backend(name: str = "auto") -> str:
    """Choose the linear solver: auto | gpu | pardiso | superlu.  Returns the active name."""
    name = (name or "auto").lower()
    if name == "gpu" and _try_gpu():
        _state["backend"] = "gpu"
    elif name == "gpu":
        warnings.warn("GPU backend requested but CuPy/CUDA device not available - using CPU")
        _state["backend"] = "pardiso" if _try_pardiso() else "superlu"
    elif name in ("auto", "pardiso") and _try_pardiso():
        _state["backend"] = "pardiso"
    else:
        _state["backend"] = "superlu"
    return backend_name()


def backend_name() -> str:
    b = _state["backend"]
    if b == "gpu":
        return f"GPU {_state['gpu_name']} (CuPy, PCG)"
    return {"pardiso": "MKL PARDISO (pypardiso)", "superlu": "SciPy SuperLU"}.get(b, "unset")


def _gpu_solve(K, b, rtol=1e-10):
    cp, csp, cspl = _state["cp"], _state["csp"], _state["cspl"]
    A = csp.csr_matrix(K.tocsr().astype(np.float64))
    bd = cp.asarray(b, dtype=cp.float64)
    inv_diag = 1.0 / A.diagonal()
    M = cspl.LinearOperator(A.shape, matvec=lambda x: inv_diag * x, dtype=cp.float64)
    count = {"n": 0}

    def cb(_):
        count["n"] += 1
    try:                                                    # CuPy >= 13 uses rtol, older tol
        x, info = cspl.cg(A, bd, M=M, rtol=rtol, maxiter=20 * A.shape[0], callback=cb)
    except TypeError:
        x, info = cspl.cg(A, bd, M=M, tol=rtol, maxiter=20 * A.shape[0], callback=cb)
    _state["cg_iters"] += count["n"]
    if info != 0:
        raise RuntimeError(f"GPU CG did not converge (info={info})")
    return cp.asnumpy(x)


def linear_solve(K, b):
    if _state["backend"] is None:
        set_backend(os.environ.get("PM_FEM_SOLVER", "auto"))
    if _state["backend"] == "gpu":
        try:
            return _gpu_solve(K, b)
        except Exception:                                   # noqa: BLE001
            _state["fallbacks"] += 1                        # robust CPU fallback
    if _state["backend"] == "pardiso" or (_state["backend"] == "gpu" and _state["pardiso"]):
        return _state["pardiso"].spsolve(K.tocsr(), b)
    return spsolve(K.tocsc(), b)


def solver_stats() -> dict:
    return dict(backend=backend_name(), gpu_fallbacks=_state["fallbacks"], cg_iterations=_state["cg_iters"])


from . import machine as mc
from .materials import NU0, ROTOR_FORGING, STATOR_LAMINATION


@dataclass
class Solution:
    A: np.ndarray
    Bx: np.ndarray
    By: np.ndarray
    iterations: int
    psi_abc: np.ndarray          # phase flux linkages [Wb-turn] (stack length incl.)
    pm: float                    # magnet remanence scale used (1 = Br at T_mag)
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
        def to_nodes(w):
            v = np.zeros(nn)
            np.add.at(v, tri.ravel(), np.repeat(w * area / 3, 3))
            return v

        c_ph = np.array([to_nodes(w_ph[i]) for i in range(3)])

        # --- permanent magnets: H = nu_m (B - Br)  ->  RHS  f_i = nu_m (Br_x c_i - Br_y b_i) / 2
        # (per tesla of remanence; radial magnetisation, sign +1 N / -1 S, rotor-frame coordinates)
        mag = reg == mc.MAGNET
        cen0 = xy.mean(axis=1)
        th0 = np.arctan2(cen0[:, 1], cen0[:, 0])
        sgn = np.zeros(ne)
        nr = len(m.r_tri)                                  # rotor elements come first
        sgn[:nr][mag[:nr]] = mc.magnet_sign(d, m.r_tag[mag[:nr]])
        ux, uy = sgn * np.cos(th0), sgn * np.sin(th0)              # radial (arc magnets)
        if getattr(m, "r_mdir", None) is not None:                 # parallel (flat magnets)
            ux[:nr] = np.where(mag[:nr], m.r_mdir[:, 0], ux[:nr])
            uy[:nr] = np.where(mag[:nr], m.r_mdir[:, 1], uy[:nr])
        nu_m = NU0 / d.mu_rm
        fe = 0.5 * nu_m * (ux[:, None] * c - uy[:, None] * b)       # (ne, 3)
        fe[~mag] = 0.0
        c_pm = np.zeros(nn)
        np.add.at(c_pm, tri.ravel(), fe.ravel())

        free = np.ones(nn, bool); free[outer] = False
        # gap elements for Arkkio torque
        cen = xy.mean(axis=1)
        rc = np.hypot(cen[:, 0], cen[:, 1])
        gap = (rc > d.R_rotor) & (rc < d.R_bore)
        iron_r = reg == mc.ROTOR_IRON
        iron_s = reg == mc.STATOR_IRON
        setup = dict(tri=tri, xy=xy, reg=reg, part=part, nn=nn, b=b, c=c, area=area, G=G,
                     rows=rows, cols=cols, c_ph=c_ph, c_pm=c_pm, free=free, gap=gap,
                     cen=cen, iron_r=iron_r, iron_s=iron_s, side_area=side_area,
                     mag=mag, mag_u=(ux, uy))
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
        nu[s["mag"]] = NU0 / self.d.mu_rm                    # linear recoil line
        return nu, dnu

    def solve(self, i_abc=(0.0, 0.0, 0.0), pm=1.0, k_shift=0, linear_mu_r=None,
              tol=1e-6, max_iter=40, A0=None) -> Solution:
        """``pm`` scales the magnet remanence: 1.0 = Br at the operating temperature T_mag,
        0.0 = magnets switched off (armature-only fields)."""
        s = self._setup(k_shift)
        F = np.asarray(i_abc, float) @ s["c_ph"] + pm * self.d.Br() * s["c_pm"]
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
            dA[free] = linear_solve(Kff, -R[free])
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
        # stator skew (2-D + analytic correction): reduces the fundamental linkage by k_sk1
        psi_abc = L * s["c_ph"] @ A * self.d.skew_factor(1)
        return Solution(A, Bx, By, it, psi_abc, pm, self.torque(s, Bx, By), k_shift)

    def magnet_Bd(self, sol: Solution) -> np.ndarray:
        """Flux density along the magnetisation direction in every magnet element [T].
        Irreversible demagnetisation risk where this falls below the knee (B_demag_limit)."""
        s = self._setup(sol.k_shift)
        ux, uy = s["mag_u"]
        m = s["mag"]
        return sol.Bx[m] * ux[m] + sol.By[m] * uy[m]

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
