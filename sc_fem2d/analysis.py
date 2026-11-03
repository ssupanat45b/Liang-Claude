"""Synchronous-condenser performance analyses built on the 2-D FEM solver.

Conventions: generator convention for the armature; the over-excited
condenser delivers lagging (inductive) vars, its armature MMF lies on the
negative d-axis (demagnetising).  The 2-D model gives the air-gap and slot
fields; end-winding leakage is added through ``X_end_pu``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from . import machine as mc
from .machine import fmt_q
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
class SCResults:
    design: mc.SCDesign
    mesh_stats: dict = field(default_factory=dict)
    axes: dict = field(default_factory=dict)
    occ: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    rated: dict = field(default_factory=dict)
    vcurve: dict = field(default_factory=dict)
    waveform: dict = field(default_factory=dict)
    torque_angle: dict = field(default_factory=dict)
    losses: dict = field(default_factory=dict)
    gap_field: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    solutions: dict = field(default_factory=dict)


class SCAnalysis:
    def __init__(self, design: mc.SCDesign | None = None, verbose: bool = True):
        self.d = design or mc.SCDesign()
        self.mesh = mc.build_mesh(self.d)
        self.model = MagnetostaticModel(self.mesh)
        self.verbose = verbose
        self.res = SCResults(self.d)
        s = self.model.element_data(0)
        self.res.mesh_stats = dict(nodes=int(s["nn"]), elements=int(len(s["tri"])),
                                   n_theta=self.d.n_theta)
        self._A_warm = None

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ----------------------------------------------------------- helpers
    def solve(self, i_abc=(0, 0, 0), i_f=0.0, k=0, **kw) -> Solution:
        sol = self.model.solve(i_abc, i_f, k_shift=k, A0=self._A_warm, **kw)
        self._A_warm = sol.A
        return sol

    def currents(self, I_peak, phi):
        th = self.res.axes["phase"]
        return [I_peak * math.cos(phi - t) for t in th]

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

    # --------------------------------------------------------------- axes
    def find_axes(self):
        """Magnetic axes of the rotor (d) and of each phase, from linear solutions."""
        self.log("[1] locating rotor d-axis and phase axes ...")
        ax = {}
        sol = self.model.solve(i_f=100.0, linear_mu_r=1000.0)
        th, Br, _ = self.model.gap_flux_density(sol)
        ax["d"] = fundamental_axis(th, Br, self.d.pp)[1]          # electrical angle
        # The anisotropic (slotted, salient) rotor biases a field-based measurement of the
        # phase axes, so they are taken from the angle of the conductor-current
        # distribution.  The fixed offset between "current centroid angle" and "MMF axis"
        # is calibrated on the field winding, whose axis coincides with the rotor symmetry.
        s = self.model.element_data(0)
        P = self.mesh.node_coords(0)
        ej = np.exp(1j * self.d.pp * np.arctan2(P[:, 1], P[:, 0]))
        off = ax["d"] - np.angle(np.sum(s["c_f"] * ej))
        ax["phase"] = [float(np.angle(np.exp(1j * (np.angle(np.sum(s["c_ph"][p] * ej)) + off))))
                       for p in range(3)]
        self.res.axes = ax
        deg = lambda a: (math.degrees(a) + 360) % 360
        self.log(f"    d-axis {deg(ax['d']):.1f} deg, phase axes "
                 + ", ".join(f"{deg(a):.1f}" for a in ax["phase"]))
        return ax

    # ---------------------------------------------------------------- OCC
    def open_circuit(self, If_values):
        self.log("[2] open-circuit characteristic ...")
        E = []
        for If in If_values:
            sol = self.solve(i_f=If)
            pd, pq = self.psi_dq(sol)
            E.append(math.sqrt(3) * self.E_phase(math.hypot(pd, pq)))
            self.log(f"    If = {If:7.1f} A  ->  E_line = {E[-1] / 1e3:7.2f} kV  ({sol.iterations} NR it)")
        E = np.array(E)
        # air-gap line from the lowest (unsaturated) point
        slope = E[1] / If_values[1]
        # exact root E(If0) = V_rated, bracketed by the OCC points
        i = int(np.searchsorted(E, self.d.V_line))
        lo, hi = If_values[max(i - 1, 0)], If_values[min(i, len(If_values) - 1)]

        def f(If):
            pd, pq = self.psi_dq(self.solve(i_f=If))
            return math.sqrt(3) * self.E_phase(math.hypot(pd, pq)) - self.d.V_line
        If0 = float(brentq(f, lo, hi, rtol=1e-5)) if lo < hi else float(np.interp(self.d.V_line, E, If_values))
        self.res.occ = dict(If=np.asarray(If_values), E_line=E, airgap_slope=slope,
                            If_noload=If0, If_airgap=self.d.V_line / slope,
                            saturation_factor=If0 / (self.d.V_line / slope))
        self.log(f"    no-load field current for rated voltage: If0 = {If0:.1f} A")
        return self.res.occ

    # -------------------------------------------------- operating points
    def field_for_point(self, I_rms: float, V_phase: float, If_hi: float):
        """Field current giving terminal voltage V for a purely reactive armature
        current I_rms (>0 over-excited / lagging, <0 under-excited / leading)."""
        X_end = self.d.X_end_pu * self.d.Z_base
        Ipk = abs(I_rms) * math.sqrt(2)
        phi = self.theta_d() + (math.pi if I_rms > 0 else 0.0)
        target = V_phase + X_end * I_rms        # internal (air-gap) EMF
        cur = self.currents(Ipk, phi)

        def f(If):
            sol = self.solve(i_abc=cur, i_f=If)
            pd, _ = self.psi_dq(sol)
            return self.E_phase(pd) - target
        If = brentq(f, -If_hi, If_hi, xtol=0.5, rtol=1e-4)
        sol = self.solve(i_abc=cur, i_f=If)
        return If, sol

    def under_excited_limit(self, If=0.0):
        """Max leading current at rated voltage with field current ``If`` (default 0)."""
        X_end = self.d.X_end_pu * self.d.Z_base
        V = self.d.V_phase

        def f(I):
            sol = self.solve(i_abc=self.currents(I * math.sqrt(2), self.theta_d()), i_f=If)
            pd, _ = self.psi_dq(sol)
            return self.E_phase(pd) - (V - X_end * I)
        I = brentq(f, 1.0, 4 * self.d.I_rated, xtol=1.0, rtol=1e-4)
        return I

    def reactances(self):
        """Unsaturated synchronous reactances from small-current FEM solutions."""
        self.log("[3] synchronous reactances & short-circuit ratio ...")
        d = self.d
        Ipk = 0.05 * d.I_rated * math.sqrt(2)
        out = {}
        for name, off in (("Xd", 0.0), ("Xq", math.pi / 2)):
            sol = self.solve(i_abc=self.currents(Ipk, self.theta_d() + off))
            pd, pq = self.psi_dq(sol)
            L = (pd if name == "Xd" else pq) / Ipk
            out[name + "_2D_ohm"] = d.omega * L
            out[name + "_pu"] = d.omega * L / d.Z_base + d.X_end_pu
        # short-circuit: rated current, V = 0
        If_sc, _ = self.field_for_point(d.I_rated, 0.0, If_hi=20 * self.res.occ["If_noload"])
        out["If_sc"] = If_sc
        out["SCR"] = self.res.occ["If_noload"] / If_sc
        out["Xd_sat_pu_from_SCR"] = 1.0 / out["SCR"]
        self.res.params = out
        self.log(f"    Xd = {out['Xd_pu']:.3f} pu, Xq = {out['Xq_pu']:.3f} pu, "
                 f"If_sc = {If_sc:.0f} A, SCR = {out['SCR']:.3f}")
        return out

    def rated_points(self):
        self.log("[4] rated over-excited and under-excited operation ...")
        d = self.d
        If0 = self.res.occ["If_noload"]
        If_r, sol_over = self.field_for_point(d.I_rated, d.V_phase, If_hi=6 * If0)
        I_u = self.under_excited_limit(0.0)
        sol_under = self.solve(i_abc=self.currents(I_u * math.sqrt(2), self.theta_d()), i_f=0.0)
        self.res.rated = dict(If_rated_over=If_r, Q_over_kVAr=d.S_rated / 1e3,
                              I_under_If0=I_u,
                              Q_under_If0_kVAr=-3 * d.V_phase * I_u / 1e3,
                              torque_over=sol_over.torque)
        self.res.solutions["rated_over"] = sol_over
        self.res.solutions["under_If0"] = sol_under
        self.log(f"    over-excited  +{fmt_q(d.S_rated / 1e3)}: If = {If_r:.1f} A "
                 f"({If_r / If0:.2f} x If0)")
        self.log(f"    under-excited at If = 0: I = {I_u:.0f} A -> Q = "
                 f"{fmt_q(-3 * d.V_phase * I_u / 1e3)}")
        return self.res.rated

    def v_curve(self, n=7):
        self.log("[5] V-curve at rated voltage ...")
        d = self.d
        I_u = self.res.rated["I_under_If0"]
        Is = np.linspace(-0.98 * I_u, d.I_rated, n)
        Ifs = []
        for I in Is:
            if abs(I) < 1e-6 * d.I_rated:
                Ifs.append(self.res.occ["If_noload"]); continue
            If, _ = self.field_for_point(I, d.V_phase, If_hi=6 * self.res.occ["If_noload"])
            Ifs.append(If)
            self.log(f"    I = {I:8.1f} A  Q = {fmt_q(3 * d.V_phase * I / 1e3):>12s}  If = {If:8.1f} A")
        self.res.vcurve = dict(I=Is, Q_kVAr=3 * d.V_phase * Is / 1e3, If=np.array(Ifs))
        return self.res.vcurve

    # ---------------------------------------------------- rotating sweeps
    def no_load_waveform(self, step=None):
        """Rotate the rotor over half an electrical period at If0 and record psi_A, EMF,
        torque ripple (cogging) and peak stator-element flux density."""
        self.log("[6] no-load EMF waveform (rotor rotation sweep) ...")
        d = self.d
        If0 = self.res.occ["If_noload"]
        n = d.n_theta
        half = n // (2 * d.pp)                   # mesh divisions per half electrical period
        if step is None:                         # >= 96 samples per half period (no slot aliasing)
            step = max(1, half // 96)
        ks = np.arange(0, half + 1, step)
        psiA, T = [], []
        s0 = self.model.element_data(0)
        Bmax = np.zeros(len(s0["tri"]))
        for k in ks:
            sol = self.solve(i_f=If0, k=int(k))
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
        """Electromagnetic torque vs stator-current angle at rated I and rated If."""
        self.log("[7] torque vs current angle (static stability) ...")
        d = self.d
        If = self.res.rated["If_rated_over"]
        gam = np.linspace(0, 2 * np.pi, n)
        T = []
        for g in gam:
            sol = self.solve(i_abc=self.currents(d.I_rated * math.sqrt(2), self.theta_d() + g), i_f=If)
            T.append(sol.torque)
        T = np.array(T)
        T_base = d.S_rated / (d.omega / (d.poles / 2))
        self.res.torque_angle = dict(gamma=gam, T=T, T_base=T_base, T_max_pu=np.abs(T).max() / T_base)
        self.log(f"    max torque {np.abs(T).max() / 1e3:.3g} kNm = {np.abs(T).max() / T_base:.2f} pu")
        return self.res.torque_angle

    # --------------------------------------------------------------- loss
    def losses(self):
        self.log("[8] loss estimate ...")
        d = self.d
        s = self.model.element_data(0)
        iron = s["iron_s"]
        if "Bmax_stator" in self.res.waveform:
            Bm = self.res.waveform["Bmax_stator"]
        else:   # no rotation sweep: use the no-load snapshot (if any) as an estimate
            sol0 = self.res.solutions.get("no_load") or self.solve(i_f=self.res.occ["If_noload"])
            Bm = sol0.Bmag
        mass_e = s["area"][iron] * d.L_stack * d.stacking * d.rho_fe
        p = (d.kh * d.f + d.ke * d.f**2) * Bm[iron] ** 2
        P_fe = d.build_factor * float(np.sum(p * mass_e))
        # armature copper
        side = np.mean(list(s["side_area"].values()))
        A_bar = side * d.fill_arm / d.turns_per_coil
        l_half = d.L_stack + d.end_arm
        turns_path = d.Qs / 3 * d.turns_per_coil / d.parallel_paths
        R_path = d.rho_cu * turns_path * 2 * l_half / A_bar
        R_ph = R_path / d.parallel_paths
        P_cu_a = 3 * d.I_rated**2 * R_ph
        # field copper at rated over-excitation
        slot = np.mean(list(s["slot_area"].values()))
        A_f = slot * d.fill_field / d.Nf_per_slot
        n_slots = len(s["slot_area"])
        turns = n_slots * d.Nf_per_slot / 2
        R_f = d.rho_cu * turns * 2 * (d.L_stack + d.end_field) / A_f
        If = self.res.rated["If_rated_over"]
        P_cu_f = If**2 * R_f
        out = dict(P_iron_kW=P_fe / 1e3, stator_core_mass_t=float(mass_e.sum()) / 1e3,
                   R_phase_mohm=R_ph * 1e3, P_arm_cu_kW=P_cu_a / 1e3,
                   R_field_ohm=R_f, P_field_cu_kW=P_cu_f / 1e3,
                   V_field_rated=If * R_f,
                   P_total_kW=(P_fe + P_cu_a + P_cu_f) / 1e3,
                   J_arm_A_mm2=d.I_rated / d.parallel_paths / (A_bar * 1e6),
                   J_field_A_mm2=If / (A_f * 1e6))
        out["loss_pct"] = out["P_total_kW"] * 1e3 / d.S_rated * 100
        self.res.losses = out
        self.log(f"    iron {out['P_iron_kW']:.0f} kW, armature Cu {out['P_arm_cu_kW']:.0f} kW, "
                 f"field Cu {out['P_field_cu_kW']:.0f} kW  (excl. friction/windage/stray)")
        return out

    def design_summary(self):
        """Key design parameters: loadings, flux densities, current densities, masses, inertia."""
        from .dimensions import geometry
        d, r = self.d, self.res
        g = geometry(d)
        s = self.model.element_data(0)
        sol = r.solutions.get("no_load") or self.solve(i_f=r.occ["If_noload"])
        B = sol.Bmag
        cen = s["cen"]; rc = np.hypot(cen[:, 0], cen[:, 1])
        p98 = lambda m: float(np.percentile(B[m], 98)) if np.any(m) else float("nan")
        teeth = s["iron_s"] & (rc < g["s3"])
        yoke = s["iron_s"] & (rc >= g["s3"])
        r_teeth = s["iron_r"] & (rc > g["r_slot_bottom"])
        r_body = s["iron_r"] & (rc <= g["r_slot_bottom"])
        D = 2 * d.R_bore
        A_el = 6 * d.series_turns * d.I_rated / (math.pi * D)
        B1 = r.gap_field.get("B1") if r.gap_field else None
        if B1 is None:
            B1 = float(fundamental_axis(*self.model.gap_flux_density(sol)[:2], d.pp)[0])
        Phi = 2 * B1 * D * d.L_stack / d.poles          # flux per pole
        # masses
        rho_cu, rho_st = 8900.0, 7850.0
        L = r.losses or {}
        side = np.mean(list(s["side_area"].values()))
        slot = np.mean(list(s["slot_area"].values()))
        m_cu_a = d.Qs * 2 * side * d.fill_arm * (d.L_stack + d.end_arm) * rho_cu
        m_cu_f = len(s["slot_area"]) * slot * d.fill_field * (d.L_stack + d.end_field) * rho_cu
        m_rot_fe = float(np.sum(s["area"][s["iron_r"]])) * d.L_stack * rho_st
        m_stat_fe = float(np.sum(s["area"][s["iron_s"]])) * d.L_stack * d.stacking * d.rho_fe
        J_rot = 0.5 * (m_rot_fe + m_cu_f * d.L_stack / (d.L_stack + d.end_field)) * d.R_rotor**2
        wm = d.omega / (d.poles / 2)
        out = dict(
            D_bore=D, D_rotor=2 * d.R_rotor, D_out=2 * d.R_stator_out, L_stack=d.L_stack,
            surface_speed=wm * d.R_rotor, electric_loading_kA_m=A_el / 1e3, B_gap1=B1,
            flux_per_pole_Wb=Phi,
            esson_C_kVAmin_m3=d.S_rated / 1e3 / (D**2 * d.L_stack * d.rpm),
            B_stator_tooth_max=p98(teeth), B_stator_yoke_max=p98(yoke),
            B_rotor_tooth_max=p98(r_teeth), B_rotor_body_max=p98(r_body),
            J_arm=L.get("J_arm_A_mm2", float("nan")), J_field=L.get("J_field_A_mm2", float("nan")),
            mass_stator_core_t=m_stat_fe / 1e3, mass_rotor_iron_active_t=m_rot_fe / 1e3,
            mass_arm_cu_t=m_cu_a / 1e3, mass_field_cu_t=m_cu_f / 1e3,
            J_rotor_kgm2=J_rot, H_inertia_s=0.5 * J_rot * wm**2 / d.S_rated,
            A_bar_mm2=side * d.fill_arm / d.turns_per_coil * 1e6,
            ampere_turns_pole=r.rated.get("If_rated_over", float("nan")) * len(s["slot_area"]) * d.Nf_per_slot / 2 / d.poles,
        )
        self.res.summary = out
        return out

    def gap_field(self):
        sol = self.res.solutions["no_load"]
        th, Br, Bt = self.model.gap_flux_density(sol)
        pp = self.d.pp
        h = harmonics(th, Br, 25 * pp)[::pp]       # index = electrical harmonic order
        self.res.gap_field = dict(theta=th, Br=Br, Bt=Bt, harm=h, B1=h[1])
        return self.res.gap_field

    def occ_auto(self, quick=False):
        """OCC with the If range scaled from an unsaturated air-gap-line estimate."""
        sol = self.model.solve(i_f=1.0, linear_mu_r=5000.0)     # unsaturated probe
        If_ag = 1.0 * self.d.V_phase / self.E_phase(self.psi_dq(sol)[0])
        frac = [0.05, 0.4, 0.7, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0]
        if quick:
            frac = [0.05, 0.5, 0.9, 1.05, 1.2, 1.5, 2.0]
        return self.open_circuit(np.array(frac) * If_ag)

    # ---------------------------------------------------------------- all
    def run_all(self, quick=False):
        d = self.d
        self.log(f"Synchronous condenser {fmt_q(d.S_rated / 1e3)}, {mc.fmt_v(d.V_line)}, "
                 f"{d.f:.0f} Hz, {d.rpm:.0f} rpm | mesh: {self.res.mesh_stats}")
        self.find_axes()
        self.occ_auto(quick)
        self.reactances()
        self.rated_points()
        self.v_curve(5 if quick else 7)
        self.no_load_waveform(step=max(1, self.d.n_theta // (2 * self.d.pp) // 48) if quick else None)
        self.gap_field()
        self.torque_vs_angle(9 if quick else 13)
        self.losses()
        self.design_summary()
        return self.res
