"""Performance analyses of a permanent-magnet (SPM) synchronous condenser.

Conventions: generator convention; positive reactive current I > 0 means the
condenser DELIVERS vars (over-excited behaviour, armature MMF on the -d axis,
demagnetising the magnets); I < 0 means it absorbs vars.

With magnets the internal EMF is fixed, so the operating point is set by the
terminal voltage alone:  E(I) = V + X_end * I  (E from the FEM air-gap linkage).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from . import machine as mc
from .machine import fmt_q, fmt_v
from .solver import MagnetostaticModel, Solution


def fundamental_axis(theta, Br, order=1):
    """Return (amplitude, axis angle) of space harmonic ``order`` of Br(theta).

    Use ``order = pole pairs`` for the working (fundamental) wave; the angle is then
    an *electrical* angle."""
    c = 2 * np.mean(Br * np.exp(-1j * order * theta))
    return abs(c), float(np.angle(c))


def harmonics(theta, Br, nmax=15):
    c = 2 * np.fft.rfft(Br) / len(Br)
    return np.abs(c[:nmax + 1])


@dataclass
class PMResults:
    design: mc.PMDesign
    mesh_stats: dict = field(default_factory=dict)
    axes: dict = field(default_factory=dict)
    emf: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    rated: dict = field(default_factory=dict)
    qv: dict = field(default_factory=dict)
    fault: dict = field(default_factory=dict)
    waveform: dict = field(default_factory=dict)
    torque_angle: dict = field(default_factory=dict)
    losses: dict = field(default_factory=dict)
    gap_field: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    solutions: dict = field(default_factory=dict)


class PMAnalysis:
    def __init__(self, design: mc.PMDesign | None = None, verbose: bool = True):
        self.d = design or mc.design_preset()
        self.mesh = mc.build_mesh(self.d)
        self.model = MagnetostaticModel(self.mesh)
        self.verbose = verbose
        self.res = PMResults(self.d)
        s = self.model.element_data(0)
        self.res.mesh_stats = dict(nodes=int(s["nn"]), elements=int(len(s["tri"])),
                                   n_theta=self.d.n_theta)
        self._A_warm = None

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ----------------------------------------------------------- helpers
    def solve(self, i_abc=(0, 0, 0), pm=1.0, k=0, **kw) -> Solution:
        sol = self.model.solve(i_abc, pm, k_shift=k, A0=self._A_warm, **kw)
        self._A_warm = sol.A
        return sol

    def currents(self, I_peak, phi):
        th = self.res.axes["phase"]
        return [I_peak * math.cos(phi - t) for t in th]

    def reactive_currents(self, I_rms):
        """Phase currents for a purely reactive current I_rms (>0 delivers vars: -d axis)."""
        phi = self.theta_d() + (math.pi if I_rms > 0 else 0.0)
        return self.currents(abs(I_rms) * math.sqrt(2), phi)

    def theta_d(self, k=0):
        return self.res.axes["d"] + self.d.pp * 2 * math.pi * k / self.d.n_theta

    def psi_dq(self, sol: Solution):
        td = self.theta_d(sol.k_shift)
        th = self.res.axes["phase"]
        pd = 2 / 3 * sum(p * math.cos(td - t) for p, t in zip(sol.psi_abc, th))
        pq = 2 / 3 * sum(p * math.cos(td + math.pi / 2 - t) for p, t in zip(sol.psi_abc, th))
        return pd, pq

    def E_phase(self, psi):
        return self.d.omega * psi / math.sqrt(2)

    @property
    def X_end(self):
        return self.d.X_end_pu * self.d.Z_base

    # --------------------------------------------------------------- axes
    def find_axes(self):
        """Rotor d-axis from the magnet field; phase axes from the conductor-current angle,
        calibrated with the magnets' equivalent current distribution (same physics)."""
        self.log("[1] locating rotor d-axis and phase axes ...")
        ax = {}
        sol = self.model.solve(pm=1.0, linear_mu_r=1000.0)
        th, Br, _ = self.model.gap_flux_density(sol)
        ax["d"] = fundamental_axis(th, Br, self.d.pp)[1]          # electrical angle
        s = self.model.element_data(0)
        P = self.mesh.node_coords(0)
        ej = np.exp(1j * self.d.pp * np.arctan2(P[:, 1], P[:, 0]))
        off = ax["d"] - np.angle(np.sum(s["c_pm"] * ej))
        ax["phase"] = [float(np.angle(np.exp(1j * (np.angle(np.sum(s["c_ph"][p] * ej)) + off))))
                       for p in range(3)]
        self.res.axes = ax
        deg = lambda a: (math.degrees(a) + 360) % 360
        self.log(f"    d-axis {deg(ax['d']):.1f} deg, phase axes "
                 + ", ".join(f"{deg(a):.1f}" for a in ax["phase"]))
        return ax

    # ---------------------------------------------------------------- EMF
    def no_load_emf(self):
        """Open-circuit EMF from the magnets at operating and cold (20 degC) temperature."""
        self.log("[2] magnet EMF (open circuit) ...")
        d = self.d
        out = {}
        for key, T in (("hot", d.T_mag), ("cold", 20.0)):
            sol = self.solve(pm=d.Br(T) / d.Br())
            pd, pq = self.psi_dq(sol)
            out[f"E0_line_{key}"] = math.sqrt(3) * self.E_phase(math.hypot(pd, pq))
            out[f"psi_pm_{key}"] = pd
            if key == "hot":
                self.res.solutions["no_load"] = sol
        out["E0_pu_hot"] = out["E0_line_hot"] / d.V_line
        out["E0_pu_cold"] = out["E0_line_cold"] / d.V_line
        out["Br_hot"], out["Br_cold"] = d.Br(), d.Br(20.0)
        self.res.emf = out
        self.log(f"    E0 = {fmt_v(out['E0_line_hot'])} ({out['E0_pu_hot']:.3f} pu) at {d.T_mag:.0f} C, "
                 f"{fmt_v(out['E0_line_cold'])} ({out['E0_pu_cold']:.3f} pu) at 20 C")
        return out

    def reactances(self):
        """Incremental synchronous reactances around the magnet operating point."""
        self.log("[3] synchronous reactances ...")
        d = self.d
        Ipk = 0.10 * d.I_rated * math.sqrt(2)
        out = {}
        for name, off, idx in (("Xd", 0.0, 0), ("Xq", math.pi / 2, 1)):
            plus = self.psi_dq(self.solve(i_abc=self.currents(Ipk, self.theta_d() + off)))[idx]
            minus = self.psi_dq(self.solve(i_abc=self.currents(-Ipk, self.theta_d() + off)))[idx]
            L = (plus - minus) / (2 * Ipk)
            out[name + "_pu"] = d.omega * L / d.Z_base + d.X_end_pu
        self.res.params = out
        self.log(f"    Xd = {out['Xd_pu']:.3f} pu, Xq = {out['Xq_pu']:.3f} pu (incl. end leakage)")
        return out

    # -------------------------------------------------- operating points
    def current_for_voltage(self, V_phase, pm=1.0):
        """Reactive current I_rms (>0 delivering vars) at terminal phase voltage V_phase."""
        d = self.d

        def f(I):
            sol = self.solve(i_abc=self.reactive_currents(I), pm=pm)
            return self.E_phase(self.psi_dq(sol)[0]) - (V_phase + self.X_end * I)
        I = brentq(f, -4 * d.I_rated, 8 * d.I_rated, xtol=0.01, rtol=1e-5)
        return I, self.solve(i_abc=self.reactive_currents(I), pm=pm)

    def rated_point(self):
        self.log("[4] operating point at rated voltage ...")
        d = self.d
        I, sol = self.current_for_voltage(d.V_phase)
        Q = 3 * d.V_phase * I
        Ic, _ = self.current_for_voltage(d.V_phase, pm=d.Br(20.0) / d.Br())
        Bd = self.model.magnet_Bd(sol)
        self.res.rated = dict(I=I, Q_kVAr=Q / 1e3, Q_pu=Q / d.S_rated, I_pu=I / d.I_rated,
                              Q_cold_kVAr=3 * d.V_phase * Ic / 1e3, I_cold=Ic, I_cold_pu=Ic / d.I_rated,
                              torque=sol.torque,
                              Bd_min=float(Bd.min()), Bd_mean=float(Bd.mean()))
        self.res.solutions["rated"] = sol
        self.log(f"    at {fmt_v(d.V_line)}: I = {I:.1f} A, Q = {fmt_q(Q / 1e3)} ({Q / d.S_rated:.2f} pu), "
                 f"cold magnets: {fmt_q(3 * d.V_phase * Ic / 1e3)}")
        return self.res.rated

    def q_v_characteristic(self, v_pu=(0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15)):
        """Reactive power vs terminal voltage (the only way Q changes on a PM condenser)."""
        self.log("[5] Q-V characteristic ...")
        d = self.d
        V, I, Q = [], [], []
        for v in v_pu:
            Ii, _ = self.current_for_voltage(v * d.V_phase)
            V.append(v); I.append(Ii); Q.append(3 * v * d.V_phase * Ii / 1e3)
            self.log(f"    V = {v:.2f} pu ({v * d.V_line:.0f} V): I = {Ii:7.1f} A, Q = {fmt_q(Q[-1]):>10s}")
        V, I, Q = map(np.array, (V, I, Q))
        # voltage at which Q = 0 (floating voltage) and slope dQ/dV around rated voltage
        # Q = 0 means I = 0, i.e. V = E0 (exact): the condenser absorbs vars only above this voltage
        V0 = self.res.emf["E0_pu_hot"] if self.res.emf else float(np.interp(0.0, Q[::-1], V[::-1]))
        slope = float(np.polyfit(V[(V > 0.94) & (V < 1.06)], Q[(V > 0.94) & (V < 1.06)], 1)[0])
        self.res.qv = dict(V_pu=V, I=I, Q_kVAr=Q, V_Q0_pu=V0, dQdV_kVAr_per_pu=slope)
        self.log(f"    dQ/dV = {slope:.0f} kVAr per pu voltage, Q = 0 at V = {V0:.3f} pu")
        return self.res.qv

    def faults_and_demag(self):
        """Steady short circuit (V = 0) and demagnetisation checks."""
        self.log("[6] short circuit and magnet demagnetisation check ...")
        d = self.d
        I_sc, sol_sc = self.current_for_voltage(0.0)
        Bd_sc = self.model.magnet_Bd(sol_sc)
        I_sc2, sol_sc2 = self.current_for_voltage(0.0, pm=d.Br(20.0) / d.Br())
        out = dict(I_sc=I_sc, I_sc_pu=I_sc / d.I_rated, I_sc_cold=I_sc2,
                   Bd_min_sc=float(Bd_sc.min()), Bd_min_rated=self.res.rated["Bd_min"],
                   Bd_min_noload=float(self.model.magnet_Bd(self.res.solutions["no_load"]).min()),
                   limit=d.B_demag_limit)
        out["demag_ok_rated"] = out["Bd_min_rated"] > d.B_demag_limit
        out["demag_ok_sc"] = out["Bd_min_sc"] > d.B_demag_limit
        self.res.fault = out
        self.res.solutions["short_circuit"] = sol_sc
        self.log(f"    steady 3-ph short circuit: I = {I_sc:.0f} A ({I_sc / d.I_rated:.2f} pu)")
        self.log(f"    min magnet B along magnetisation: no-load {out['Bd_min_noload']:.2f} T, rated "
                 f"{out['Bd_min_rated']:.2f} T, short circuit {out['Bd_min_sc']:.2f} T "
                 f"(limit {d.B_demag_limit:.2f} T) -> "
                 + ("OK" if out["demag_ok_sc"] else "DEMAGNETISATION RISK at short circuit"))
        return out

    # ---------------------------------------------------- rotating sweeps
    def no_load_waveform(self, step=None):
        """Rotate the rotor over half an electrical period (magnets only, no current) and record psi_A, EMF,
        torque ripple (cogging) and peak stator-element flux density."""
        self.log("[7] no-load EMF waveform and cogging torque (rotor rotation sweep) ...")
        d = self.d
        n = d.n_theta
        half = n // (2 * d.pp)                   # mesh divisions per half electrical period
        if step is None:                         # >= 96 samples per half period (no slot aliasing)
            step = max(1, half // 96)
        ks = np.arange(0, half + 1, step)
        psiA, T = [], []
        s0 = self.model.element_data(0)
        Bmax = np.zeros(len(s0["tri"]))
        for k in ks:
            sol = self.solve(pm=1.0, k=int(k))
            psiA.append(sol.psi_abc[0]); T.append(sol.torque)
            Bmax = np.maximum(Bmax, sol.Bmag)
            if k == 0:
                self.res.solutions["no_load"] = sol
        psiA = np.array(psiA)
        ang = d.pp * 2 * np.pi * ks / n                      # electrical angle
        # half-wave symmetry: psi(theta + pi) = -psi(theta)
        ang_full = np.r_[ang[:-1], ang[:-1] + np.pi, 2 * np.pi]
        psi_full = np.r_[psiA[:-1], -psiA[:-1], psiA[0]]
        # EMF from the flux-linkage spectrum (e_n = n * omega * psi_n), exact for the sampled
        # waveform.  Stator skew: harmonic nu is scaled by k_sk(nu)/k_sk(1) (k_sk(1) is already
        # contained in the solver's flux linkage).
        Psi = np.fft.rfft(psi_full[:-1])
        nh_all = np.arange(len(Psi))
        Psi = Psi * d.skew_factor(nh_all) / d.skew_factor(1)
        E = -1j * nh_all * d.omega * Psi
        e = np.fft.irfft(E, len(psi_full) - 1)
        e = np.r_[e, e[0]]
        nmax = min(25, len(Psi) - 1)
        e_h = 2 * np.abs(E[:nmax + 1]) / (len(psi_full) - 1)
        nh = np.arange(len(e_h))
        thd = math.sqrt(np.sum(e_h[2:] ** 2)) / e_h[1]
        non_triplen = (nh >= 2) & (nh % 3 != 0)          # triplen harmonics cancel line-to-line
        thd_line = math.sqrt(np.sum(e_h[non_triplen] ** 2)) / e_h[1]
        self.res.waveform = dict(angle=ang_full, psiA=psi_full, e=e, e_harm=e_h, thd=thd,
                                 thd_line=thd_line,
                                 E_line_rms_fund=math.sqrt(3) * e_h[1] / math.sqrt(2),
                                 cogging=np.array(T), cogging_angle=ang, Bmax_stator=Bmax)
        self.log(f"    EMF THD phase {100 * thd:.2f} % / line-to-line {100 * thd_line:.2f} %, "
                 f"fundamental E_line = {mc.fmt_v(self.res.waveform['E_line_rms_fund'])}")
        return self.res.waveform

    def torque_vs_angle(self, n=13):
        """Electromagnetic torque vs current angle at rated current (static pull-out torque)."""
        self.log("[8] torque vs current angle (static stability) ...")
        d = self.d
        gam = np.linspace(0, 2 * np.pi, n)
        T = np.array([self.solve(i_abc=self.currents(d.I_rated * math.sqrt(2), self.theta_d() + g)).torque
                      for g in gam])
        T_base = d.S_rated / (d.omega / d.pp)
        self.res.torque_angle = dict(gamma=gam, T=T, T_base=T_base, T_max_pu=np.abs(T).max() / T_base)
        self.log(f"    max torque {np.abs(T).max():.0f} Nm = {np.abs(T).max() / T_base:.2f} pu")
        return self.res.torque_angle

    # --------------------------------------------------------------- loss
    def losses(self):
        self.log("[9] loss estimate ...")
        d = self.d
        s = self.model.element_data(0)
        iron = s["iron_s"]
        if "Bmax_stator" in self.res.waveform:
            Bm = self.res.waveform["Bmax_stator"]
        else:
            Bm = (self.res.solutions.get("no_load") or self.solve(pm=1.0)).Bmag
        mass_e = s["area"][iron] * d.L_stack * d.stacking * d.rho_fe
        p = (d.kh * d.f + d.ke * d.f**2) * Bm[iron] ** 2
        P_fe = d.build_factor * float(np.sum(p * mass_e))
        side = np.mean(list(s["side_area"].values()))
        A_bar = side * d.fill_arm / d.turns_per_coil
        turns_path = d.Qs / 3 * d.turns_per_coil / d.parallel_paths
        R_ph = d.rho_cu * turns_path * 2 * (d.L_stack + d.end_arm) / A_bar / d.parallel_paths
        I = abs(self.res.rated.get("I", d.I_rated))
        P_cu = 3 * I**2 * R_ph
        out = dict(P_iron_kW=P_fe / 1e3, stator_core_mass_t=float(mass_e.sum()) / 1e3,
                   R_phase_mohm=R_ph * 1e3, P_arm_cu_kW=P_cu / 1e3, P_field_cu_kW=0.0,
                   P_total_kW=(P_fe + P_cu) / 1e3,
                   J_arm_A_mm2=I / d.parallel_paths / (A_bar * 1e6))
        out["loss_pct"] = out["P_total_kW"] * 1e3 / d.S_rated * 100
        self.res.losses = out
        self.log(f"    iron {out['P_iron_kW']:.2f} kW, armature Cu {out['P_arm_cu_kW']:.2f} kW, "
                 f"no field loss  (magnet eddy, friction/windage, stray excluded)")
        return out

    def gap_field(self):
        sol = self.res.solutions["no_load"]
        th, Br, Bt = self.model.gap_flux_density(sol)
        pp = self.d.pp
        h = harmonics(th, Br, 25 * pp)[::pp]       # index = electrical harmonic order
        self.res.gap_field = dict(theta=th, Br=Br, Bt=Bt, harm=h, B1=h[1])
        return self.res.gap_field

    def design_summary(self):
        from .dimensions import geometry
        d, r = self.d, self.res
        g = geometry(d)
        s = self.model.element_data(0)
        sol = r.solutions.get("no_load") or self.solve(pm=1.0)
        B = sol.Bmag
        cen = s["cen"]; rc = np.hypot(cen[:, 0], cen[:, 1])
        p98 = lambda m: float(np.percentile(B[m], 98)) if np.any(m) else float("nan")
        D = 2 * d.R_bore
        I = abs(r.rated.get("I", d.I_rated))
        B1 = r.gap_field["B1"] if r.gap_field else fundamental_axis(*self.model.gap_flux_density(sol)[:2], d.pp)[0]
        rho_cu, rho_st = 8900.0, 7850.0
        side = np.mean(list(s["side_area"].values()))
        m_cu = d.Qs * 2 * side * d.fill_arm * (d.L_stack + d.end_arm) * rho_cu
        m_core = float(np.sum(s["area"][s["iron_r"]])) * d.L_stack * rho_st
        m_mag = float(np.sum(s["area"][s["mag"]])) * d.L_stack * d.rho_mag
        m_stat = float(np.sum(s["area"][s["iron_s"]])) * d.L_stack * d.stacking * d.rho_fe
        J_rot = 0.5 * m_core * d.R_core**2 + 0.5 * m_mag * (d.R_core**2 + d.R_rotor**2)
        wm = d.omega / d.pp
        sol_r = r.solutions.get("rated")
        out = dict(
            D_bore=D, D_rotor=2 * d.R_rotor, D_out=2 * d.R_stator_out, L_stack=d.L_stack,
            surface_speed=wm * d.R_rotor, electric_loading_kA_m=6 * d.series_turns * I / (math.pi * D) / 1e3,
            B_gap1=B1, flux_per_pole_Wb=2 * B1 * D * d.L_stack / d.poles,
            esson_C_kVAmin_m3=d.S_rated / 1e3 / (D**2 * d.L_stack * d.rpm),
            B_stator_tooth_max=p98(s["iron_s"] & (rc < g["s3"])),
            B_stator_yoke_max=p98(s["iron_s"] & (rc >= g["s3"])),
            B_rotor_core_max=p98(s["iron_r"]),
            B_stator_tooth_rated=float(np.percentile(sol_r.Bmag[s["iron_s"] & (rc < g["s3"])], 98)) if sol_r else float("nan"),
            B_stator_yoke_rated=float(np.percentile(sol_r.Bmag[s["iron_s"] & (rc >= g["s3"])], 98)) if sol_r else float("nan"),
            B_rotor_core_rated=float(np.percentile(sol_r.Bmag[s["iron_r"]], 98)) if sol_r else float("nan"),
            J_arm=r.losses.get("J_arm_A_mm2", float("nan")),
            mass_stator_core_t=m_stat / 1e3, mass_rotor_core_t=m_core / 1e3, mass_magnet_t=m_mag / 1e3,
            mass_arm_cu_t=m_cu / 1e3, J_rotor_kgm2=J_rot, H_inertia_s=0.5 * J_rot * wm**2 / d.S_rated,
            A_bar_mm2=side * d.fill_arm / d.turns_per_coil * 1e6,
        )
        self.res.summary = out
        return out

    # ---------------------------------------------------------------- all
    def run_all(self, quick=False):
        d = self.d
        self.log(f"PM synchronous condenser {fmt_q(d.S_rated / 1e3)}, {fmt_v(d.V_line)}, {d.f:.0f} Hz, "
                 f"{d.rpm:.0f} rpm, magnets {d.h_m * 1e3:.1f} mm | mesh: {self.res.mesh_stats}")
        self.find_axes()
        self.no_load_emf()
        self.reactances()
        self.rated_point()
        self.q_v_characteristic((0.90, 0.95, 1.0, 1.05, 1.10) if quick else
                                (0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15))
        self.faults_and_demag()
        self.no_load_waveform(step=max(1, d.n_theta // (2 * d.pp) // 48) if quick else None)
        self.gap_field()
        self.torque_vs_angle(9 if quick else 13)
        self.losses()
        self.design_summary()
        return self.res


# ---------------------------------------------------------------------------
def size_magnets(design: mc.PMDesign, Q_target_kVAr=None, h_lo=0.003, h_hi=0.030, verbose=True):
    """Magnet thickness h_m giving the target reactive power at rated voltage (hot magnets).

    The stator bore and air gap are kept; the rotor core shrinks as the magnets get thicker."""
    import copy
    Qt = (design.S_rated / 1e3) if Q_target_kVAr is None else Q_target_kVAr
    hist = []

    def q_of(h):
        d = copy.deepcopy(design); d.h_m = h
        an = PMAnalysis(d, verbose=False); an.find_axes()
        I, _ = an.current_for_voltage(d.V_phase)
        Q = 3 * d.V_phase * I / 1e3
        hist.append((h, Q))
        if verbose:
            print(f"    h_m = {h * 1e3:6.2f} mm -> Q = {Q:8.1f} kVAr", flush=True)
        return Q - Qt
    if verbose:
        print(f"[0] sizing magnets for Q = {Qt:.0f} kVAr at rated voltage ...", flush=True)
    h = brentq(q_of, h_lo, h_hi, xtol=5e-5)
    return h, hist
