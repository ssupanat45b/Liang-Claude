#!/usr/bin/env python3
"""2-D electromagnetic FEM study of a permanent-magnet (SPM) synchronous condenser.

Usage:
    python run_pm_fem.py                     # full study, GPU if available (~3-5 min on CPU)
    python run_pm_fem.py --quick             # coarser sweeps
    python run_pm_fem.py --size-magnets      # first size the magnet thickness for rated kVAr
    python run_pm_fem.py --solver gpu        # force GPU (CuPy) / cpu: pardiso | superlu
    python run_pm_fem.py --hm 0.020 --mesh fine

Outputs (results_pm300/): PNG figures 00-09, results.json, report.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from pm_fem2d import plots, solver
from pm_fem2d.analysis import PMAnalysis, size_magnets
from pm_fem2d.machine import DESIGN_PRESETS, apply_mesh_preset, design_preset, fmt_q, fmt_v


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def write_report(an, files, path, runtime, backend):
    d, r = an.d, an.res
    em, p, ra, qv, fl, L, w, sm = r.emf, r.params, r.rated, r.qv, r.fault, r.losses, r.waveform, r.summary
    rel = os.path.basename
    ok = lambda c: "✓" if c else "⚠"
    qv_rows = "\n".join(f"| {v:.2f} | {v * d.V_line:.0f} | {i:.1f} | {i / d.I_rated:.2f} | {q:.1f} |"
                        for v, i, q in zip(qv["V_pu"], qv["I"], qv["Q_kVAr"]))
    txt = f"""# PM Synchronous Condenser (SPM rotor) — 2-D Electromagnetic FEM Study

Machine: **{fmt_q(d.S_rated / 1e3)}, {fmt_v(d.V_line)}, {d.f:.0f} Hz, {d.poles}-pole, {d.rpm:.0f} rpm**,
surface-mounted radially magnetised NdFeB magnets, no field winding.
Nonlinear magnetostatic A<sub>z</sub> FEM with linear magnets (recoil μ<sub>r</sub> = {d.mu_rm}),
{r.mesh_stats['elements']} elements. Solver: {backend}. Runtime {runtime:.0f} s.

![design summary]({rel(files['summary'])})

## 1. Key consequence of permanent-magnet excitation

The EMF is fixed by the magnets, so **reactive power cannot be set by excitation**; it follows the
terminal voltage: Q ≈ 3·V·(E − V)/X<sub>s</sub>.

| Item | Value |
|---|---|
| Q at rated voltage, magnets {d.T_mag:.0f} °C | **{fmt_q(ra['Q_kVAr'])}** (I = {ra['I']:.1f} A, {ra['I_pu']:.2f} pu) |
| Q at rated voltage, cold magnets 20 °C | {fmt_q(ra['Q_cold_kVAr'])} — current {ra['I_cold_pu']:.2f} pu {ok(ra['I_cold_pu'] <= 1.05)} (thermal: cold start-up) |
| Natural voltage support dQ/dV | {qv['dQdV_kVAr_per_pu'] / 100:.1f} kVAr per 1 % voltage (Q rises when V falls) |
| Terminal voltage for Q = 0 | {qv['V_Q0_pu']:.3f} pu ({qv['V_Q0_pu'] * d.V_line:.0f} V) — absorbs vars only above this |

| V [pu] | V [V] | I [A] | I [pu] | Q [kVAr] |
|---|---|---|---|---|
{qv_rows}

![Q-V]({rel(files['qv'])})

## 2. Main dimensions

| Item | Value |
|---|---|
| Stator outer / bore diameter | {2 * d.R_stator_out * 1e3:.0f} / {2 * d.R_bore * 1e3:.0f} mm |
| Rotor diameter (magnet OD) / core diameter | {2 * d.R_rotor * 1e3:.1f} / {2 * d.R_core * 1e3:.1f} mm |
| Magnetic gap = mechanical + sleeve | {d.gap * 1e3:.1f} = {d.g_mech * 1e3:.1f} + {d.t_sleeve * 1e3:.1f} mm |
| Active length | {d.L_stack * 1e3:.0f} mm |
| Magnets | {d.poles} × NdFeB, {d.h_m * 1e3:.1f} mm thick, arc ratio {d.alpha_p_actual:.3f}, Br {d.Br20:.2f} T (20 °C) → {d.Br():.3f} T ({d.T_mag:.0f} °C), mass {sm['mass_magnet_t'] * 1e3:.1f} kg |
| Stator | {d.Qs} slots, double layer, pitch {d.coil_pitch}/{d.Qs // d.poles}, {d.series_turns:.0f} series turns, skew {d.skew_slots:g} slot |

![cross-section]({rel(files['dimensions'])})
![mesh]({rel(files['geometry'])})

## 3. Open circuit (magnets only)

| Result | Value |
|---|---|
| E0 line, {d.T_mag:.0f} °C / 20 °C | {em['E0_line_hot']:.0f} V ({em['E0_pu_hot']:.3f} pu) / {em['E0_line_cold']:.0f} V ({em['E0_pu_cold']:.3f} pu) |
| Air-gap B<sub>1</sub> | {r.gap_field['B1']:.3f} T |
| EMF THD line-to-line / phase | {100 * w['thd_line']:.2f} % / {100 * w['thd']:.2f} % |
| Peak cogging torque | {np.abs(w['cogging']).max():.1f} Nm |

Open-circuit voltage after disconnection can reach **{em['E0_pu_cold']:.2f} pu** (cold magnets).

![no load]({rel(files['noload'])})
![air gap]({rel(files['gap'])})
![emf]({rel(files['wave'])})

## 4. Reactances, short circuit and demagnetisation

| Item | Value |
|---|---|
| X<sub>d</sub> / X<sub>q</sub> (incl. end leakage {d.X_end_pu:.2f} pu) | {p['Xd_pu']:.3f} / {p['Xq_pu']:.3f} pu |
| Steady 3-phase short-circuit current | {fl['I_sc']:.0f} A ({fl['I_sc_pu']:.2f} pu) |
| Max static torque at rated current | {r.torque_angle['T_max_pu']:.2f} pu |
| Min B along magnetisation: no load / rated / short circuit | {fl['Bd_min_noload']:.2f} {ok(fl['Bd_min_noload'] > d.B_demag_limit)} / {fl['Bd_min_rated']:.2f} {ok(fl['demag_ok_rated'])} / {fl['Bd_min_sc']:.2f} {ok(fl['demag_ok_sc'])} T (limit {d.B_demag_limit:.2f} T at {d.T_mag:.0f} °C) |

![rated]({rel(files['rated'])})
![short circuit]({rel(files['sc'])})
![demag]({rel(files['demag'])})

## 5. Losses (estimate)

| Component | Value |
|---|---|
| Stator core | {L['P_iron_kW']:.2f} kW |
| Armature copper (R<sub>ph</sub> = {L['R_phase_mohm']:.2f} mΩ, J = {L['J_arm_A_mm2']:.2f} A/mm²) | {L['P_arm_cu_kW']:.2f} kW |
| Field copper | 0 (no field winding) |
| **Total electromagnetic** | **{L['P_total_kW']:.2f} kW ({L['loss_pct']:.2f} %)** |

Magnet eddy-current, friction/windage, bearing and stray losses are not included.

## 6. Modelling assumptions

* 2-D magnetostatic; end-winding leakage as a fixed reactance; stator skew applied analytically.
* Magnets: linear recoil line, radial magnetisation, Br temperature coefficient {d.alpha_Br * 100:.2f} %/K.
  Demagnetisation is judged by the minimum flux density along the magnetisation versus a knee limit.
* Steady-state short circuit only; the sub-transient peak current is higher (transient analysis needed).
* Purely reactive operation (current on the d-axis); resistance neglected for the voltage balance.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="pm300", choices=DESIGN_PRESETS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mesh", default="normal", choices=["coarse", "normal", "fine", "ultra"])
    ap.add_argument("--solver", default="auto", choices=["auto", "gpu", "pardiso", "superlu"],
                    help="linear solver backend (auto: GPU if CuPy + CUDA device available)")
    ap.add_argument("--hm", type=float, help="magnet thickness [m]")
    ap.add_argument("--size-magnets", action="store_true", help="size h_m for rated kVAr first")
    ap.add_argument("--Q", type=float, help="rated reactive power kVAr (used with --size-magnets)")
    a = ap.parse_args()

    backend = solver.set_backend(a.solver)
    if solver.gpu_help():
        print(solver.gpu_help())
    d = apply_mesh_preset(design_preset(a.design), a.mesh)
    a.out = a.out or f"results_{a.design}"
    if a.Q:
        d.S_rated = a.Q * 1e3
    if a.hm:
        d.h_m = a.hm
    print(f"design: {a.design} | solver: {backend} | mesh: {a.mesh}")
    t0 = time.time()
    if a.size_magnets:
        h, _ = size_magnets(d)
        d.h_m = math.ceil(h * 2000) / 2000             # round UP to 0.5 mm steps
        print(f"    -> magnet thickness {h * 1e3:.2f} mm, using {d.h_m * 1e3:.1f} mm")
    an = PMAnalysis(d)
    an.run_all(quick=a.quick)
    files = plots.make_all(an, a.out)
    runtime = time.time() - t0
    write_report(an, files, os.path.join(a.out, "report.md"), runtime, backend)
    r = an.res
    data = dict(design=d.__dict__, mesh=r.mesh_stats, solver=solver.solver_stats(), axes=r.axes,
                emf=r.emf, params=r.params, rated=r.rated, qv=r.qv, fault=r.fault, losses=r.losses,
                waveform={k: v for k, v in r.waveform.items() if k != "Bmax_stator"},
                torque_angle=r.torque_angle, summary=r.summary)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(to_jsonable(data), fh, indent=1)
    st = solver.solver_stats()
    print(f"\nDone in {runtime:.0f} s ({st['backend']}, GPU fallbacks {st['gpu_fallbacks']}).  "
          f"Report: {os.path.join(a.out, 'report.md')}")


if __name__ == "__main__":
    main()
