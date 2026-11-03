"""Design space, machine generator and FEM evaluation for the IPM (flat-magnet) MOBO.

Design variables
----------------
    poles   : {2, 4, 6, 8}                (categorical)
    D       : stator bore diameter [m]
    L       : active core length [m]
    h_m     : magnet thickness [m]
    d_frac  : magnet depth below the rotor surface / pole pitch
    w_frac  : magnet width / widest width allowed (bridge corner and q-axis rib)

Everything else follows from classical sizing rules (q = 3 winding with 7/9
pitch, turns from the voltage, slot depth from the current density, yoke from
the flux density, gap 0.625 % of D, bridges 0.3 % of D, barriers scaled with
the pole pitch) - the same rules as the sizing study in Liang_IPM_SynCon.

Cheap quantities (geometry validity, L/D, masses, copper loss) are computed
analytically; reactive power, magnet working point, tooth flux density and
iron loss come from the nonlinear FEM (``fem_evaluate``).
"""
from __future__ import annotations

import math
import time

import numpy as np

POLES = (2, 4, 6, 8)
BOUNDS = dict(D=(0.25, 0.55), L=(0.10, 1.20), h_m=(0.008, 0.070), d_frac=(0.06, 0.20), w_frac=(0.60, 1.00))
CONT = ("D", "L", "h_m", "d_frac", "w_frac")

RULES = dict(
    S=300e3, V=400.0, f=50.0, B1=0.85, kE=1.06, B_yoke=1.40, J=3.7e6, fill=0.45,
    gap_frac=0.00625, gap_min=0.0015, bridge_frac=0.003, bridge_min=0.0008,
    corner_frac=0.0075, barrier_frac_rect=0.044, barrier_gap_frac=0.011, q=3, pitch_num=7,
    rho_fe=7650.0, rho_rotor=7850.0, rho_mag=7500.0, rho_cu=8900.0, stacking=0.95,
    cost_mag=60.0, cost_cu=10.0, cost_fe=2.0,
)

LIMITS = dict(Q_min=300.0, Q_max=330.0,   # Q_max: current <= 1.1 pu (thermal)
              Bd_min=0.10, B_tooth_max=2.0, LD_min=0.3, LD_max=3.0)


# ----------------------------------------------------------------------------- encoding
def to_unit(x):
    """Design dict -> GP input vector (continuous in [0,1] + one-hot poles)."""
    u = [(x[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0]) for k in CONT]
    return np.array(u + [1.0 if x["poles"] == p else 0.0 for p in POLES])


def from_unit(u, poles):
    x = {k: BOUNDS[k][0] + float(np.clip(u[i], 0, 1)) * (BOUNDS[k][1] - BOUNDS[k][0]) for i, k in enumerate(CONT)}
    x["poles"] = int(poles)
    return x


# ----------------------------------------------------------------------------- machine
def make_design(x):
    """Return (PMDesign, info) for a design dict."""
    from ipm_fem2d.machine import PMDesign
    z = RULES
    p, D, L = int(x["poles"]), float(x["D"]), float(x["L"])
    rpm = 120 * z["f"] / p
    Qs = 3 * z["q"] * p
    div = max(2, int(round(432 / Qs))); div += div % 2
    q = z["q"]; alpha = math.pi * p / Qs
    kw = math.sin(q * alpha / 2) / (q * math.sin(alpha / 2)) * math.sin(math.pi / 2 * z["pitch_num"] / (3 * q))
    Phi = 2 * z["B1"] * D * L / p
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
    I = z["S"] / (math.sqrt(3) * z["V"])
    theta_s = math.pi / Qs                              # slot = 50 % of the slot pitch
    gap = max(z["gap_min"], z["gap_frac"] * D)
    R_bore = D / 2
    s_wedge = max(0.002, 0.0075 * D)
    A_side = tpc * I / (a * z["J"]) / z["fill"]
    r1 = R_bore + s_wedge
    s_layer = math.sqrt(r1**2 + 2 * A_side / theta_s) - r1
    h_yoke = z["B1"] * D / (p * z["B_yoke"] * 0.95)
    R_out = R_bore + s_wedge + 2 * s_layer + h_yoke
    tau_p = math.pi * D / p
    end_arm = 1.2 * tau_p * z["pitch_num"] / (3 * q) + 0.05
    X_end = float(np.clip(0.06 * (end_arm / L) / (0.30 / 0.45), 0.02, 0.30))
    d = PMDesign(
        S_rated=z["S"], V_line=z["V"], f=z["f"], poles=p,
        L_stack=L, R_rotor=R_bore - gap, gap=gap, t_sleeve=0.0, R_stator_out=R_out,
        t_bridge=max(z["bridge_min"], z["bridge_frac"] * D),
        Qs=Qs, div_per_slot=div, slot_divs=div // 2, s_wedge=s_wedge, s_layer=s_layer,
        coil_pitch=z["pitch_num"], parallel_paths=a, turns_per_coil=tpc,
        h_m=float(x["h_m"]), magnet_shape="rect", d_m=float(x["d_frac"]) * tau_p,
        w_frac=float(x["w_frac"]), corner_margin=max(0.002, z["corner_frac"] * D),
        barrier_deg=z["barrier_frac_rect"] * 360.0 / p, barrier_gap_deg=z["barrier_gap_frac"] * 360.0 / p,
        X_end_pu=X_end, skew_slots=1.0, fill_arm=z["fill"], end_arm=end_arm, gap_layers=12, radial_refine=1)
    info = dict(rpm=rpm, Qs=Qs, a=a, turns_per_coil=tpc, N_series=N, D_out=2 * R_out, L_over_D=L / D,
                X_end_pu=X_end, gap=gap, s_layer=s_layer, h_yoke=h_yoke)
    return d, info


def _shoelace(P):
    P = np.asarray(P)
    return 0.5 * abs(np.dot(P[:, 0], np.roll(P[:, 1], 1)) - np.dot(P[:, 1], np.roll(P[:, 0], 1)))


def cheap(x):
    """Analytic quantities, no FEM.  Returns dict with 'valid' and a 'reason' if not."""
    from ipm_fem2d.rotor_rect import layout
    z = RULES
    out = dict(x)
    try:
        d, info = make_design(x)
        d.check()
        g = layout(d)
    except Exception as e:                               # noqa: BLE001
        out.update(valid=False, reason=f"geometry: {e}")
        return out
    out.update(info, valid=True, reason="", w_m=g["w"], d_m=d.d_m)
    L = d.L_stack
    # masses (active length; copper incl. end windings)
    R_bore, R_out = d.R_bore, d.R_stator_out
    slots = d.Qs * 0.5 * (math.pi / d.Qs) * ((R_bore + d.s_wedge + 2 * d.s_layer) ** 2 - R_bore**2)
    m_stat = (math.pi * (R_out**2 - R_bore**2) - slots) * L * z["stacking"] * z["rho_fe"]
    a_mag = d.poles * g["w"] * d.h_m
    a_bar = 2 * d.poles * _shoelace(g["barrier_upper"]) - 0  # magnet end edge is shared
    m_mag = a_mag * L * z["rho_mag"]
    m_rot = (math.pi * d.R_rotor**2 - a_mag - a_bar) * L * z["rho_rotor"]
    A_side = d.turns_per_coil * (d.S_rated / (math.sqrt(3) * d.V_line)) / (d.parallel_paths * z["J"])
    m_cu = d.Qs * 2 * A_side * (L + d.end_arm) * z["rho_cu"]
    # armature copper loss at rated current (75 C)
    I = d.S_rated / (math.sqrt(3) * d.V_line)
    turns_path = d.Qs / 3 * d.turns_per_coil / d.parallel_paths
    R_ph = d.rho_cu * turns_path * 2 * (L + d.end_arm) / (A_side / d.turns_per_coil) / d.parallel_paths
    out.update(m_stator=m_stat, m_rotor=m_rot, m_magnet=m_mag, m_cu=m_cu,
               m_active=m_stat + m_rot + m_mag + m_cu,
               cost=m_mag * z["cost_mag"] + m_cu * z["cost_cu"] + (m_stat + m_rot) * z["cost_fe"],
               P_cu_kW=3 * I**2 * R_ph / 1e3)
    lo, hi = LIMITS["LD_min"], LIMITS["LD_max"]
    if not lo <= out["L_over_D"] <= hi:
        out.update(valid=False, reason=f"L/D {out['L_over_D']:.2f} outside [{lo}, {hi}]")
    return out


# ----------------------------------------------------------------------------- FEM
def fem_evaluate(args):
    """Worker: full nonlinear FEM evaluation of one design.  args = (x, solver_name)."""
    x, solver_name = args
    from ipm_fem2d import solver
    from ipm_fem2d.analysis import PMAnalysis
    solver.set_backend(solver_name)
    t0 = time.time()
    r = cheap(x)
    if not r["valid"]:
        r.update(fem_ok=False, runtime_s=time.time() - t0)
        return r
    try:
        d, _ = make_design(x)
        an = PMAnalysis(d, verbose=False)
        an.find_axes()
        I, sol_r = an.current_for_voltage(d.V_phase)
        an.res.rated = dict(I=I)
        an.res.solutions["rated"] = sol_r
        sol0 = an.solve(pm=1.0)
        an.res.solutions["no_load"] = sol0
        pd0, _ = an.psi_dq(sol0)
        I_sc, sol_sc = an.current_for_voltage(0.0)
        Bd_sc = float(an.model.magnet_Bd(sol_sc).min())
        an.losses()
        sm = an.design_summary()
        L = an.res.losses
        r.update(fem_ok=True, Q_kVAr=3 * d.V_phase * I / 1e3, I_pu=I / d.I_rated, I_sc_pu=I_sc / d.I_rated,
                 E0_pu=math.sqrt(3) * an.E_phase(abs(pd0)) / d.V_line, Bd_min_sc=Bd_sc,
                 Bd_min_rated=float(an.model.magnet_Bd(sol_r).min()),
                 B_tooth_rated=sm["B_stator_tooth_rated"], B_yoke_rated=sm["B_stator_yoke_rated"],
                 P_iron_kW=L["P_iron_kW"], P_cu_fem_kW=L["P_arm_cu_kW"],
                 P_loss_kW=L["P_iron_kW"] + r["P_cu_kW"],
                 m_magnet_fem=sm["mass_magnet_t"] * 1e3, H_s=sm["H_inertia_s"])
    except Exception as e:                               # noqa: BLE001
        r.update(fem_ok=False, reason=f"FEM: {type(e).__name__}: {e}")
    r["runtime_s"] = time.time() - t0
    return r


def is_feasible(r):
    if not (r.get("valid") and r.get("fem_ok")):
        return False
    z = LIMITS
    return (z["Q_min"] <= r["Q_kVAr"] <= z["Q_max"] and r["Bd_min_sc"] >= z["Bd_min"]
            and r["B_tooth_rated"] <= z["B_tooth_max"])
