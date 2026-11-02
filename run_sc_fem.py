#!/usr/bin/env python3
"""2-D electromagnetic FEM study of a synchronous condenser.

Usage:
    python run_sc_fem.py            # full study  (~ a few minutes)
    python run_sc_fem.py --quick    # coarser sweeps for a fast check
    python run_sc_fem.py --out results --S 160 --V 15.75

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
from sc_fem2d.machine import SCDesign
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

Example machine: **{d.S_rated / 1e6:.0f} MVAr, {d.V_line / 1e3:.1f} kV, {d.f:.0f} Hz, {d.poles}-pole, {d.rpm:.0f} rpm**, round (cylindrical) rotor.
Nonlinear magnetostatic A<sub>z</sub> FEM, first-order triangles, Newton–Raphson, sliding air-gap interface.
Mesh: {r.mesh_stats['elements']} elements / {r.mesh_stats['nodes']} nodes. Runtime {runtime:.0f} s.

## 1. Main dimensions

| Item | Value |
|---|---|
| Rotor diameter | {2 * d.R_rotor * 1e3:.0f} mm |
| Air gap (mechanical) | {d.gap * 1e3:.0f} mm |
| Stator bore / outer diameter | {2 * d.R_bore * 1e3:.0f} / {2 * d.R_stator_out * 1e3:.0f} mm |
| Active length | {d.L_stack:.2f} m |
| Stator slots / coil pitch | {d.Qs} / {d.coil_pitch} slots (full pitch {d.Qs // d.poles}) |
| Armature winding | double layer, {d.parallel_paths} parallel paths, {d.series_turns:.0f} series turns/phase, k<sub>w1</sub> = {d.winding_factor():.4f} |
| Rotor slots | {d.Qr_pitches - 2 * d.Qr_omit_per_pole} wound of {d.Qr_pitches} pitches, {d.Nf_per_slot} turns/slot |
| Rotor peripheral speed | {d.omega / (d.poles / 2) * d.R_rotor:.0f} m/s |

![geometry]({rel(files['geometry'])})

## 2. No-load / open circuit

| Result | Value |
|---|---|
| Field current for rated voltage, I<sub>f0</sub> | **{o['If_noload']:.0f} A** |
| Field current on the air-gap line | {o['If_airgap']:.0f} A |
| Saturation factor I<sub>f0</sub>/I<sub>f,ag</sub> | {o['saturation_factor']:.3f} |
| Air-gap B<sub>r</sub> fundamental | {B1:.3f} T |
| EMF fundamental (rotation sweep) | {w['E_line_rms_fund'] / 1e3:.2f} kV line |
| EMF THD | {100 * w['thd']:.2f} % |
| Peak no-load cogging torque | {np.abs(w['cogging']).max() / 1e3:.1f} kNm |

![no load]({rel(files['noload'])})
![air gap]({rel(files['gap'])})

## 3. Reactances and short-circuit ratio

| Parameter | Value |
|---|---|
| X<sub>d</sub> (unsaturated, incl. end leakage {d.X_end_pu:.2f} pu) | **{p['Xd_pu']:.3f} pu** |
| X<sub>q</sub> (unsaturated) | {p['Xq_pu']:.3f} pu |
| Field current for rated armature current on short circuit, I<sub>f,sc</sub> | {p['If_sc']:.0f} A |
| Short-circuit ratio SCR = I<sub>f0</sub>/I<sub>f,sc</sub> | **{p['SCR']:.3f}** |

## 4. Reactive capability

| Operating point | Value |
|---|---|
| Rated over-excited Q | +{ra['Q_over_MVAr']:.0f} MVAr at I = {d.I_rated:.0f} A |
| Rated field current (over-excited) | **{ra['If_rated_over']:.0f} A** ({ra['If_rated_over'] / o['If_noload']:.2f} × I<sub>f0</sub>) |
| Under-excited capability at I<sub>f</sub> = 0 | {ra['Q_under_If0_MVAr']:.1f} MVAr (I = {ra['I_under_If0']:.0f} A) |
| Air-gap torque at rated over-excited point | {ra['torque_over'] / 1e3:.2f} kNm (≈ 0: no active power) |
| Max static torque at rated I, rated I<sub>f</sub> | {r.torque_angle['T_max_pu']:.2f} pu (T<sub>base</sub> = {r.torque_angle['T_base'] / 1e3:.0f} kNm) |

![rated]({rel(files['rated'])})
![under]({rel(files['under'])})
![occ]({rel(files['occ'])})
![emf]({rel(files['wave'])})

## 5. Losses (estimate)

| Loss component | Value |
|---|---|
| Stator core (Steinmetz, build factor {d.build_factor}) | {L['P_iron_kW']:.0f} kW (core mass {L['stator_core_mass_t']:.1f} t) |
| Armature copper, R<sub>ph</sub> = {L['R_phase_mohm']:.2f} mΩ | {L['P_arm_cu_kW']:.0f} kW (J = {L['J_arm_A_mm2']:.1f} A/mm²) |
| Field copper, R<sub>f</sub> = {L['R_field_ohm']:.3f} Ω, U<sub>f</sub> = {L['V_field_rated']:.0f} V | {L['P_field_cu_kW']:.0f} kW (J = {L['J_field_A_mm2']:.1f} A/mm²) |
| **Total electromagnetic loss** | **{L['P_total_kW']:.0f} kW ({L['loss_pct']:.2f} % of rating)** |

Friction, windage, bearing, and stray-load losses are not included.

## 6. Modelling assumptions

* 2-D magnetostatic analysis per unit length × active length; end-winding leakage is added as a fixed reactance (X<sub>end</sub> = {d.X_end_pu:.2f} pu); end-region fields, axial ducts, and stacking factor are neglected in the field solution.
* Slots are modelled as annular sectors (radial sides) on a structured polar mesh; slot openings and wedges are non-magnetic.
* Rotor body: solid forging B–H curve. Stator: M330-50A-class lamination. No eddy currents (magnetostatic).
* Purely reactive operation: armature MMF on the d-axis (−d over-excited, +d under-excited); armature resistance is neglected for voltage.
* Losses: Steinmetz iron loss using the peak element flux density from the rotation sweep; copper at 75 °C.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--S", type=float, help="rated MVAr")
    ap.add_argument("--V", type=float, help="rated line voltage kV")
    ap.add_argument("--f", type=float, help="frequency Hz")
    ap.add_argument("--length", type=float, help="active core length m")
    args = ap.parse_args()

    d = SCDesign()
    if args.S: d.S_rated = args.S * 1e6
    if args.V: d.V_line = args.V * 1e3
    if args.f: d.f = args.f
    if args.length: d.L_stack = args.length

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
                torque_angle=r.torque_angle)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(to_jsonable(data), fh, indent=1)
    print(f"\nDone in {runtime:.0f} s.  Report: {os.path.join(args.out, 'report.md')}")


if __name__ == "__main__":
    main()
