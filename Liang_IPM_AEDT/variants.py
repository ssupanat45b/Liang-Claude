"""Base design (fixed main dimensions) and the magnet / air-gap variables.

The machine is the best-compromise design of the multi-objective BO
(Liang_IPM_BO): 4-pole, bore 376 mm, core 546 mm, flat magnets with their outer
face 35.1 mm below the rotor surface.  If ../Liang_IPM_BO/results/pareto.csv is
present, the exact compromise design is read from it instead of the rounded
defaults below.

Only three quantities change (mm resolution):
    h_m  magnet thickness   (step 0.5 mm, outer face fixed -> grows inward)
    w_m  magnet width       (step 1 mm, must fit under the bridges / leave the q-axis rib)
    gap  air gap            (step 0.1 mm, stator bore fixed -> rotor OD changes;
                             the magnet keeps its depth below the rotor surface)
"""
from __future__ import annotations

import copy
import csv
import math
import os
import time

import numpy as np

from design_space import make_design

BASE_DEFAULT = dict(poles=4, D=0.376, L=0.546, d_frac=0.0351 / (math.pi * 0.376 / 4), w_frac=0.968, h_m=0.0364)
VARS = ("h_m", "w_m", "gap")
BOUNDS = dict(h_m=(0.020, 0.055), w_m=(0.150, 0.230), gap=(0.0015, 0.0040))
STEP = dict(h_m=0.0005, w_m=0.001, gap=0.0001)

LIMITS = dict(Q_min=300.0, Q_max=330.0, Bd_min=0.10, B_tooth_max=2.0)
OBJECTIVES = [  # key, label, unit
    ("thd_line_pct", "EMF THD (line-line)", "%"),
    ("P_loss_kW", "EM losses (heat)", "kW"),
    ("cogging_Nm", "cogging torque (peak, unskewed)", "Nm"),
    ("I_cold_pu", "current with cold magnets", "pu"),
]


def load_base(bo_results="../Liang_IPM_BO/results/pareto.csv"):
    """Best-compromise (knee) design of the BO run if available, else the rounded defaults."""
    if os.path.exists(bo_results):
        with open(bo_results) as fh:
            pf = [r for r in csv.DictReader(fh)]
        if pf:
            F = np.array([[float(r["m_active"]), float(r["P_loss_kW"])] for r in pf])
            lo, hi = F.min(0), F.max(0)
            N = (F - lo) / np.where(hi > lo, hi - lo, 1)
            r = pf[int(np.argmin(np.hypot(N[:, 0], N[:, 1])))]
            base = dict(poles=int(float(r["poles"])), **{k: float(r[k]) for k in ("D", "L", "d_frac", "w_frac", "h_m")})
            return base, f"BO compromise from {bo_results}"
    return dict(BASE_DEFAULT), "built-in defaults (BO compromise, rounded)"


def base_values(base):
    """The three variables of the base design (for reference / plotting)."""
    from ipm_fem2d.rotor_rect import layout
    d, _ = make_design(base)
    return dict(h_m=d.h_m, w_m=layout(d)["w"], gap=d.gap)


def snap(v):
    return {k: round(round(v[k] / STEP[k]) * STEP[k], 6) for k in VARS}


def to_unit(v):
    return np.array([(v[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0]) for k in VARS])


def from_unit(u):
    return snap({k: BOUNDS[k][0] + float(np.clip(u[i], 0, 1)) * (BOUNDS[k][1] - BOUNDS[k][0])
                 for i, k in enumerate(VARS)})


def make_variant(base, v):
    """PMDesign of the base machine with magnet thickness/width and air gap v."""
    d, _ = make_design(base)
    d = copy.deepcopy(d)
    d.h_m = float(v["h_m"])
    d.w_m_abs = float(v["w_m"])
    bore = d.R_bore
    d.gap = float(v["gap"])
    d.R_rotor = bore - d.gap
    return d


def valid(base, v):
    try:
        make_variant(base, v).check()
        return True, ""
    except Exception as e:                               # noqa: BLE001
        return False, str(e)


def cogging_period_divs(d):
    """Mesh divisions per cogging period (360 deg / lcm(Qs, 2p) mechanical)."""
    lcm = d.Qs * d.poles // math.gcd(d.Qs, d.poles)
    return d.n_theta // lcm


def evaluate(args):
    """Worker: FEM evaluation of one magnet/gap variant.  args = (base, v, solver_name)."""
    base, v, solver_name = args
    from ipm_fem2d import solver
    from ipm_fem2d.analysis import PMAnalysis
    from ipm_fem2d.rotor_rect import layout
    solver.set_backend(solver_name)
    t0 = time.time()
    r = dict(v)
    ok, why = valid(base, v)
    if not ok:
        r.update(valid=False, fem_ok=False, reason=f"geometry: {why}", runtime_s=time.time() - t0)
        return r
    r["valid"] = True
    try:
        d = make_variant(base, v)
        an = PMAnalysis(d, verbose=False)
        an.find_axes()
        an.rated_point()                                  # hot and cold-magnet operating points
        ra = an.res.rated
        I_sc, sol_sc = an.current_for_voltage(0.0)
        an.no_load_waveform(step=max(1, d.n_theta // (2 * d.pp) // 36))
        an.gap_field()
        an.losses()
        sm = an.design_summary()
        # cogging torque over one cogging period, every mesh division (no current)
        nc = cogging_period_divs(d)
        T = np.array([an.solve(pm=1.0, k=k).torque for k in range(nc + 1)])
        T_rated = d.S_rated / (d.omega / d.pp)
        m_mag = sm["mass_magnet_t"] * 1e3
        r.update(fem_ok=True, reason="", Q_kVAr=ra["Q_kVAr"], Q_cold_kVAr=ra["Q_cold_kVAr"],
                 I_pu=ra["I_pu"], I_cold_pu=ra["I_cold_pu"], I_sc_pu=I_sc / d.I_rated,
                 Bd_min_sc=float(an.model.magnet_Bd(sol_sc).min()), Bd_min_rated=ra["Bd_min"],
                 thd_line_pct=100 * an.res.waveform["thd_line"], thd_phase_pct=100 * an.res.waveform["thd"],
                 cogging_Nm=0.5 * float(T.max() - T.min()), cogging_pct=50 * float(T.max() - T.min()) / T_rated,
                 P_loss_kW=an.res.losses["P_total_kW"], P_iron_kW=an.res.losses["P_iron_kW"],
                 P_cu_kW=an.res.losses["P_arm_cu_kW"], B_tooth_rated=sm["B_stator_tooth_rated"],
                 m_magnet_kg=m_mag, m_active_kg=(sm["mass_magnet_t"] + sm["mass_arm_cu_t"] + sm["mass_stator_core_t"]
                                                 + sm["mass_rotor_core_t"]) * 1e3,
                 E0_pu=an.res.emf.get("E0_pu_hot", float("nan")) if an.res.emf else float("nan"),
                 w_lim=layout(d)["w_lim"], Xd_pu=float("nan"))
    except Exception as e:                               # noqa: BLE001
        r.update(fem_ok=False, reason=f"FEM: {type(e).__name__}: {e}")
    r["runtime_s"] = time.time() - t0
    return r


def is_feasible(r):
    if not r.get("fem_ok"):
        return False
    z = LIMITS
    return (z["Q_min"] <= r["Q_kVAr"] <= z["Q_max"] and r["Bd_min_sc"] >= z["Bd_min"]
            and r["B_tooth_rated"] <= z["B_tooth_max"])
