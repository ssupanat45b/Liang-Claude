"""Parametric sizing of a 300 kVAr / 400 V IPM (interior PM) synchronous condenser.

``make_design(poles, D_bore)`` builds a complete, self-consistent machine from
two inputs using classical sizing rules; ``evaluate(case)`` then sizes the
magnets with the nonlinear FEM so that rated kVAr is delivered at rated voltage
and computes the key performance figures.

Sizing rules (all adjustable in SIZING below)
---------------------------------------------
* Winding: q = 3 slots/pole/phase  ->  Qs = 9 p, double layer, coil pitch 7/9.
* Core length from the output coefficient of the validated ipm300 design:
      L = S / (C0 * D^2 * n)          (C0 = 2.78 kVA.min/m^3)
* Series turns from the voltage:  N = k_E V_ph / (4.44 f k_w Phi),
  then the core length is corrected by N_target/N so the turns quantisation
  does not distort the comparison,
      Phi = 2 B1 D L / p  with the design air-gap fundamental B1.
  Turns are realised with an integer turns/coil and a parallel-path count a | p.
* Slot depth from the copper area at current density J and fill factor.
* Stator yoke from the flux per pole at B_yoke; outer diameter follows.
* Air gap ~ 0.625 % of D (min 1.5 mm), no sleeve; pole shoe 1.25 % of D (min 3 mm),
  bridge 0.3 % of D (min 0.8 mm), flux barriers 6 % of the pole pitch at each magnet end.
* End-winding length ~ 1.2 x coil span + 50 mm; end leakage scaled from ipm300.
* Buried NdFeB N38SH arc magnets, 80 % pole arc, thickness from FEM (brentq on Q(h_m)).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

SIZING = dict(
    S=300e3, V=400.0, f=50.0,
    C0=2.778,            # kVA.min/m^3   (ipm300: 300 / (0.4^2 * 0.45 * 1500))
    B1=0.85,             # design air-gap fundamental for the turns calculation [T]
    kE=1.06,             # air-gap EMF / terminal voltage at rated load
    B_yoke=1.40,         # stator yoke flux density [T]
    J=3.7e6,             # armature current density [A/m^2]
    fill=0.45,           # copper fill of each slot layer
    gap_frac=0.00625,    # mechanical gap / bore diameter
    gap_min=0.0015,
    shoe_frac=0.0125, shoe_min=0.003, bridge_frac=0.003, bridge_min=0.0008, barrier_frac=0.06,
    q=3, pitch_num=7,    # slots per pole per phase, coil pitch in slots (of 3q = 9)
    alpha_p=0.80,
    n_theta_target=432,  # angular mesh divisions (kept the same for every pole count)
    # material cost index [USD/kg] for a rough comparison only
    cost_mag=60.0, cost_cu=10.0, cost_fe=2.0,
)


def make_design(poles: int, D_bore: float, S=None):
    """Return a PMDesign for the given pole number and stator bore diameter [m]."""
    from ipm_fem2d.machine import PMDesign
    z = SIZING
    S = S or z["S"]
    p = poles
    pp = p // 2
    rpm = 120 * z["f"] / p
    Qs = 3 * z["q"] * p
    div = max(2, int(round(z["n_theta_target"] / Qs)))
    div += div % 2                                     # even divisions per slot
    L = S / 1e3 / (z["C0"] * D_bore**2 * rpm)
    # winding factor (q, pitch 7/9)
    q = z["q"]; alpha = math.pi * p / Qs
    kd = math.sin(q * alpha / 2) / (q * math.sin(alpha / 2))
    kp = math.sin(math.pi / 2 * z["pitch_num"] / (3 * q))
    kw = kd * kp
    Phi = 2 * z["B1"] * D_bore * L / p
    V_ph = z["V"] / math.sqrt(3)
    N_target = z["kE"] * V_ph / (4.44 * z["f"] * kw * Phi)
    best = None
    for a in [k for k in range(1, p + 1) if p % k == 0]:
        tpc = max(1, int(round(N_target * 3 * a / Qs)))
        N = Qs * tpc / (3 * a)
        err = abs(N - N_target) / N_target
        if best is None or err < best[0] - 1e-9:
            best = (err, a, tpc, N)
    _, a, tpc, N = best
    # designer's practice: absorb the turns quantisation in the core length (keeps N*Phi)
    L *= N_target / N
    I = S / (math.sqrt(3) * z["V"])
    # slots: annular sectors, 50 % of the slot pitch
    theta_s = 2 * math.pi / Qs * 0.5
    g_mech = max(z["gap_min"], z["gap_frac"] * D_bore)
    gap = g_mech                                       # IPM: no sleeve
    R_bore = D_bore / 2
    s_wedge = max(0.002, 0.0075 * D_bore)
    A_cond = I / (a * z["J"])
    A_side = tpc * A_cond / z["fill"]
    r1 = R_bore + s_wedge
    r2 = math.sqrt(r1**2 + 2 * A_side / theta_s)
    s_layer = r2 - r1                                  # use the top layer depth for both layers
    # yoke carries half the flux per pole: (2 B1 D L / p) / 2 = B_yoke * h * L * k_stack
    h_yoke = z["B1"] * D_bore / (p * z["B_yoke"] * 0.95)
    R_out = R_bore + s_wedge + 2 * s_layer + h_yoke
    # end winding and its leakage, scaled from ipm300 (l_end 0.30 m, L 0.45 m, 0.06 pu)
    tau_p = math.pi * D_bore / p
    end_arm = 1.2 * tau_p * z["pitch_num"] / (3 * q) + 0.05
    X_end = float(np.clip(0.06 * (end_arm / L) / (0.30 / 0.45), 0.02, 0.30))
    R_rotor = R_bore - gap
    d = PMDesign(
        S_rated=S, V_line=z["V"], f=z["f"], poles=p,
        L_stack=L, R_rotor=R_rotor, gap=gap, t_sleeve=0.0, R_stator_out=R_out,
        t_shoe=max(z["shoe_min"], z["shoe_frac"] * D_bore),
        t_bridge=max(z["bridge_min"], z["bridge_frac"] * D_bore), barrier_frac=z["barrier_frac"],
        Qs=Qs, div_per_slot=div, slot_divs=div // 2, s_wedge=s_wedge, s_layer=s_layer,
        coil_pitch=z["pitch_num"], parallel_paths=a, turns_per_coil=tpc,
        h_m=0.012, alpha_p=z["alpha_p"],
        X_end_pu=X_end, skew_slots=1.0, fill_arm=z["fill"], end_arm=end_arm,
        gap_layers=12, radial_refine=1)
    info = dict(poles=p, D_bore=D_bore, rpm=rpm, L=L, L_over_D=L / D_bore, Qs=Qs, a=a,
                turns_per_coil=tpc, N_series=N, N_target=N_target, kw=kw, D_out=2 * R_out,
                g_mech=g_mech, s_layer=s_layer, h_yoke=h_yoke, X_end_pu=X_end, end_arm=end_arm)
    return d, info


def evaluate(case):
    """Worker: size magnets, analyse, return a flat result dict (+ the design for plots)."""
    poles, D_bore, solver_name, waveform = case
    from ipm_fem2d import solver
    from ipm_fem2d.analysis import PMAnalysis
    solver.set_backend(solver_name)
    t0 = time.time()
    d, info = make_design(poles, D_bore)
    res = dict(info, status="ok", reason="")
    # ---- magnet sizing: Q(h_m) = rated at rated voltage ---------------------------
    h_hi = min(0.060, 0.45 * d.R_mag)

    def q_of(h):
        dd = copy.deepcopy(d); dd.h_m = h
        an = PMAnalysis(dd, verbose=False); an.find_axes()
        I, _ = an.current_for_voltage(dd.V_phase)
        return 3 * dd.V_phase * I / 1e3 - d.S_rated / 1e3
    from scipy.optimize import brentq
    try:
        f_lo, f_hi = q_of(0.002), q_of(h_hi)
        if f_hi < 0:
            res.update(status="infeasible", reason=f"Q < rated even with {h_hi * 1e3:.0f} mm magnets")
            d.h_m = h_hi
        elif f_lo > 0:
            d.h_m = 0.002
        else:
            d.h_m = brentq(q_of, 0.002, h_hi, xtol=1e-4)
    except Exception as e:                                # noqa: BLE001
        res.update(status="error", reason=f"sizing: {type(e).__name__}: {e}")
        res["runtime_s"] = time.time() - t0
        return res, d
    # ---- performance -------------------------------------------------------------
    try:
        an = PMAnalysis(d, verbose=False)
        an.find_axes(); an.no_load_emf(); an.reactances(); an.rated_point()
        an.q_v_characteristic((0.95, 1.0, 1.05)); an.faults_and_demag()
        if waveform:
            half = d.n_theta // (2 * d.pp)
            an.no_load_waveform(step=max(1, half // 48))
        an.gap_field(); an.losses(); sm = an.design_summary()
        r, fl, em, L = an.res.rated, an.res.fault, an.res.emf, an.res.losses
        z = SIZING
        m_fe = sm["mass_stator_core_t"] + sm["mass_rotor_core_t"]
        cost = (sm["mass_magnet_t"] * z["cost_mag"] + sm["mass_arm_cu_t"] * z["cost_cu"]
                + m_fe * z["cost_fe"]) * 1e3
        T_rated = d.S_rated / (d.omega / d.pp)
        res.update(
            h_m_mm=d.h_m * 1e3, Q_kVAr=r["Q_kVAr"], Q_cold_kVAr=r["Q_cold_kVAr"], I_cold_pu=r["I_cold_pu"],
            E0_pu=em["E0_pu_hot"], E0_cold_pu=em["E0_pu_cold"], Xd_pu=an.res.params["Xd_pu"],
            Xq_pu=an.res.params["Xq_pu"], saliency=an.res.params["Xq_pu"] / an.res.params["Xd_pu"], dQ_per_pct=an.res.qv["dQdV_kVAr_per_pu"] / 100,
            I_sc_pu=fl["I_sc_pu"], Bd_min_sc=fl["Bd_min_sc"], Bd_min_rated=fl["Bd_min_rated"],
            demag_ok=bool(fl["demag_ok_sc"]),
            B_gap1=sm["B_gap1"], B_tooth_rated=sm["B_stator_tooth_rated"], B_yoke_rated=sm["B_stator_yoke_rated"],
            J_arm=L["J_arm_A_mm2"], P_iron_kW=L["P_iron_kW"], P_cu_kW=L["P_arm_cu_kW"],
            P_loss_kW=L["P_total_kW"], loss_pct=L["loss_pct"],
            m_magnet_kg=sm["mass_magnet_t"] * 1e3, m_cu_kg=sm["mass_arm_cu_t"] * 1e3, m_fe_kg=m_fe * 1e3,
            m_active_kg=(sm["mass_magnet_t"] + sm["mass_arm_cu_t"] + m_fe) * 1e3,
            cost_usd=cost, H_s=sm["H_inertia_s"], J_rotor=sm["J_rotor_kgm2"],
            surface_speed=sm["surface_speed"],
            thd_line_pct=100 * an.res.waveform["thd_line"] if waveform else float("nan"),
            cogging_pct=100 * float(np.abs(an.res.waveform["cogging"]).max()) / T_rated if waveform else float("nan"),
            T_max_pu=float("nan"))
        if res["status"] == "ok" and not res["demag_ok"]:
            res["reason"] = "demagnetisation risk at short circuit"
    except Exception as e:                                # noqa: BLE001
        res.update(status="error", reason=f"analysis: {type(e).__name__}: {e}")
    res["runtime_s"] = time.time() - t0
    res["backend"] = solver.backend_name()
    return res, d
