#!/usr/bin/env python3
"""2-D electromagnetic FEM study of a synchronous condenser.

Usage:
    python run_sc_fem.py            # full study  (~ a few minutes)
    python run_sc_fem.py --quick    # coarser sweeps for a fast check
    python run_sc_fem.py --design sc100     # original 100 MVAr example
    python run_sc_fem.py --Q 500             # factory design re-rated to 500 kVAr

Outputs (in --out): PNG figures, results.json and report.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from sc_fem2d.analysis import SCAnalysis
from sc_fem2d.machine import DESIGN_PRESETS, apply_mesh_preset, design_preset, fmt_q, fmt_v
from sc_fem2d.solver import backend_name
from sc_fem2d import plots


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def write_report(an, files, path, runtime):
    d, r = an.d, an.res
    o, p, ra, L, w = r.occ, r.params, r.rated, r.losses, r.waveform
    rel = lambda f: os.path.basename(f)
    B1 = r.gap_field["B1"]
    lines = f"""# Synchronous Condenser — 2-D Electromagnetic FEM Study

Machine: **{fmt_q(d.S_rated / 1e3)}, {fmt_v(d.V_line)}, {d.f:.0f} Hz, {d.poles}-pole, {d.rpm:.0f} rpm**, round (cylindrical) rotor with distributed field winding.
Nonlinear magnetostatic A<sub>z</sub> FEM, first-order triangles, Newton–Raphson, sliding air-gap interface.
Mesh: {r.mesh_stats['elements']} elements / {r.mesh_stats['nodes']} nodes. Runtime {runtime:.0f} s.

![design summary]({rel(files['summary'])})

## 1. Main dimensions

| Item | Value |
|---|---|
| Rotor diameter | {2 * d.R_rotor * 1e3:.0f} mm |
| Air gap (mechanical) | {d.gap * 1e3:.0f} mm |
| Stator bore / outer diameter | {2 * d.R_bore * 1e3:.0f} / {2 * d.R_stator_out * 1e3:.0f} mm |
| Active length | {d.L_stack:.2f} m |
| Stator slots / coil pitch | {d.Qs} / {d.coil_pitch} slots (full pitch {d.Qs // d.poles}) |
| Armature winding | double layer, {d.parallel_paths} parallel paths, {d.series_turns:.0f} series turns/phase, k<sub>w1</sub> = {d.winding_factor():.4f} |
| Rotor slots | {d.wound_rotor_slots} wound of {d.Qr_pitches} pitches, {d.Nf_per_slot} turns/slot |
| Rotor peripheral speed | {d.omega / (d.poles / 2) * d.R_rotor:.0f} m/s |

![cross-section dimensions]({rel(files['dimensions'])})
![geometry]({rel(files['geometry'])})

## 2. No-load / open circuit

| Result | Value |
|---|---|
| Field current for rated voltage, I<sub>f0</sub> | **{o['If_noload']:.1f} A** |
| Field current on the air-gap line | {o['If_airgap']:.1f} A |
| Saturation factor I<sub>f0</sub>/I<sub>f,ag</sub> | {o['saturation_factor']:.3f} |
| Air-gap B<sub>r</sub> fundamental | {B1:.3f} T |
| EMF fundamental (rotation sweep) | {fmt_v(w['E_line_rms_fund'])} line |
| EMF THD line-to-line / phase | {100 * w['thd_line']:.2f} % / {100 * w['thd']:.2f} % (triplen harmonics cancel line-to-line) |
| Peak no-load cogging torque | {np.abs(w['cogging']).max():.3g} Nm |

![no load]({rel(files['noload'])})
![air gap]({rel(files['gap'])})

## 3. Reactances and short-circuit ratio

| Parameter | Value |
|---|---|
| X<sub>d</sub> (unsaturated, incl. end leakage {d.X_end_pu:.2f} pu) | **{p['Xd_pu']:.3f} pu** |
| X<sub>q</sub> (unsaturated) | {p['Xq_pu']:.3f} pu |
| Field current for rated armature current on short circuit, I<sub>f,sc</sub> | {p['If_sc']:.1f} A |
| Short-circuit ratio SCR = I<sub>f0</sub>/I<sub>f,sc</sub> | **{p['SCR']:.3f}** |

## 4. Reactive capability

| Operating point | Value |
|---|---|
| Rated over-excited Q | +{fmt_q(ra['Q_over_kVAr'])} at I = {d.I_rated:.0f} A |
| Rated field current (over-excited) | **{ra['If_rated_over']:.1f} A** ({ra['If_rated_over'] / o['If_noload']:.2f} × I<sub>f0</sub>) |
| Under-excited capability at I<sub>f</sub> = 0 | {fmt_q(ra['Q_under_If0_kVAr'])} (I = {ra['I_under_If0']:.0f} A) |
| Air-gap torque at rated over-excited point | {ra['torque_over']:.3g} Nm (≈ 0: no active power) |
| Max static torque at rated I, rated I<sub>f</sub> | {r.torque_angle['T_max_pu']:.2f} pu (T<sub>base</sub> = {r.torque_angle['T_base']:,.0f} Nm) |

![rated]({rel(files['rated'])})
![under]({rel(files['under'])})
![occ]({rel(files['occ'])})
![emf]({rel(files['wave'])})

## 5. Losses (estimate)

| Loss component | Value |
|---|---|
| Stator core (Steinmetz, build factor {d.build_factor}) | {L['P_iron_kW']:.3g} kW (core mass {L['stator_core_mass_t'] * 1e3:,.0f} kg) |
| Armature copper, R<sub>ph</sub> = {L['R_phase_mohm']:.2f} mΩ | {L['P_arm_cu_kW']:.3g} kW (J = {L['J_arm_A_mm2']:.1f} A/mm²) |
| Field copper, R<sub>f</sub> = {L['R_field_ohm']:.3g} Ω, U<sub>f</sub> = {L['V_field_rated']:.0f} V | {L['P_field_cu_kW']:.3g} kW (J = {L['J_field_A_mm2']:.1f} A/mm²) |
| **Total electromagnetic loss** | **{L['P_total_kW']:.3g} kW ({L['loss_pct']:.2f} % of rating)** |

Friction, windage, bearing, and stray-load losses are not included.

## 6. Modelling assumptions

* 2-D magnetostatic analysis per unit length × active length; end-winding leakage is added as a fixed reactance (X<sub>end</sub> = {d.X_end_pu:.2f} pu); end-region fields, axial ducts, and stacking factor are neglected in the field solution.
* Slots are modelled as annular sectors (radial sides) on a structured polar mesh; slot openings and wedges are non-magnetic.
* Rotor body: solid steel B–H curve. Stator: M330-50A-class lamination. No eddy currents (magnetostatic).
* Purely reactive operation: armature MMF on the d-axis (−d over-excited, +d under-excited); armature resistance is neglected for voltage.
* Losses: Steinmetz iron loss using the peak element flux density from the rotation sweep; copper at 75 °C.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="factory300", choices=DESIGN_PRESETS,
                    help="design preset: factory300 (300 kVAr 0.4 kV 4-pole, default) or sc100 (100 MVAr 13.8 kV 2-pole)")
    ap.add_argument("--out", default=None, help="output folder (default results_<design>)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--Q", type=float, help="rated reactive power kVAr")
    ap.add_argument("--S", type=float, help="rated reactive power MVAr")
    ap.add_argument("--V", type=float, help="rated line voltage kV")
    ap.add_argument("--f", type=float, help="frequency Hz")
    ap.add_argument("--length", type=float, help="active core length m")
    ap.add_argument("--gap", type=float, help="air gap m (stator bore kept fixed)")
    ap.add_argument("--mesh", default="normal", choices=["coarse", "normal", "fine", "ultra"],
                    help="mesh density preset (default normal)")
    args = ap.parse_args()

    d = apply_mesh_preset(design_preset(args.design), args.mesh)
    args.out = args.out or ("results" if args.design == "sc100" else f"results_{args.design}")
    if args.Q: d.S_rated = args.Q * 1e3
    if args.S: d.S_rated = args.S * 1e6
    if args.V: d.V_line = args.V * 1e3
    if args.f: d.f = args.f
    if args.length: d.L_stack = args.length
    if args.gap:
        bore = d.R_bore
        d.gap = args.gap
        d.R_rotor = bore - d.gap
    print(f"design: {args.design} | linear solver: {backend_name()} | mesh preset: {args.mesh}")

    t0 = time.time()
    an = SCAnalysis(d)
    an.run_all(quick=args.quick)
    files = plots.make_all(an, args.out)
    runtime = time.time() - t0
    write_report(an, files, os.path.join(args.out, "report.md"), runtime)
    r = an.res
    data = dict(design=d.__dict__, mesh=r.mesh_stats, axes=r.axes, occ=r.occ, params=r.params,
                rated=r.rated, vcurve=r.vcurve, losses=r.losses,
                waveform={k: v for k, v in r.waveform.items() if k != "Bmax_stator"},
                torque_angle=r.torque_angle, summary=r.summary)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(to_jsonable(data), fh, indent=1)
    print(f"\nDone in {runtime:.0f} s.  Report: {os.path.join(args.out, 'report.md')}")


if __name__ == "__main__":
    main()
