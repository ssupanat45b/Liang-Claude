#!/usr/bin/env python3
"""Cross-check the Python FEM against FEMM 4.2 (via pyfemm) on the same machine.

Requirements (Windows):  FEMM 4.2 from https://www.femm.info  +  pip install pyfemm

    python femm_compare.py                # build model, solve 4 cases in FEMM and in Python, compare
    python femm_compare.py --build-only   # only write femm_model/sc_model.fem (open it in the FEMM GUI)
    python femm_compare.py --show         # keep the FEMM window visible while solving

The FEMM model is drawn from the same SCDesign data (sc_fem2d/machine.py): identical
annular-sector slots, winding layout, B-H tables and stack length.  Flux linkages
are read from FEMM circuits, torque from the weighted Maxwell stress tensor on the
rotor blocks, and air-gap B from point samples.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from sc_fem2d.analysis import SCAnalysis, fundamental_axis
from sc_fem2d.dimensions import geometry, rotor_wound
from sc_fem2d.machine import SCDesign, field_sign, winding_layout
from sc_fem2d.materials import ROTOR_FORGING, STATOR_LAMINATION

ROTOR_GROUP = 1
PHASES = ("A", "B", "C")


# ----------------------------------------------------------------- geometry
class FemmDrawer:
    def __init__(self, femm):
        self.f = femm
        self.nodes = set()

    @staticmethod
    def xy(r, deg):
        a = math.radians(deg)
        return r * math.cos(a), r * math.sin(a)

    def node(self, r, deg):
        x, y = self.xy(r, deg)
        key = (round(x, 7), round(y, 7))
        if key not in self.nodes:
            self.f.mi_addnode(x, y)
            self.nodes.add(key)
        return x, y

    def seg(self, r1, r2, deg):
        x1, y1 = self.node(r1, deg)
        x2, y2 = self.node(r2, deg)
        self.f.mi_addsegment(x1, y1, x2, y2)

    def arc(self, r, a0, a1, maxseg):
        span = (a1 - a0) % 360.0
        if span < 1e-9:
            return
        x1, y1 = self.node(r, a0)
        x2, y2 = self.node(r, a1)
        self.f.mi_addarc(x1, y1, x2, y2, span, maxseg)

    def circle(self, r, angles, maxseg):
        """Full circle split at the given angles (at least two pieces)."""
        a = sorted({round(v % 360.0, 9) for v in angles})
        if len(a) < 2:
            a = sorted({0.0, 180.0} | set(a))
        for a0, a1 in zip(a, a[1:] + [a[0] + 360.0]):
            self.arc(r, a0, a1, maxseg)

    def label(self, r, deg, material, mesh, circuit="<None>", group=0, turns=0):
        x, y = self.xy(r, deg)
        self.f.mi_addblocklabel(x, y)
        self.f.mi_selectlabel(x, y)
        self.f.mi_setblockprop(material, 0, mesh, circuit, 0, group, turns)
        self.f.mi_clearselected()


def slot_angles(d: SCDesign):
    g = geometry(d)
    st = [(k * g["s_pitch_deg"] + g["s_slot_start"],
           k * g["s_pitch_deg"] + g["s_slot_start"] + g["s_slot_deg"]) for k in range(d.Qs)]
    rt = [(i, i * g["r_pitch_deg"] + g["r_slot_start"],
           i * g["r_pitch_deg"] + g["r_slot_start"] + g["r_slot_deg"])
          for i in range(d.Qr_pitches) if rotor_wound(d, i)]
    return st, rt


def build_femm_model(femm, d: SCDesign, path: str, linear: bool = False):
    g = geometry(d)
    femm.newdocument(0)
    femm.mi_probdef(0, "meters", "planar", 1e-8, d.L_stack, 30)
    # materials -----------------------------------------------------------
    femm.mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    femm.mi_addmaterial("Copper", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    for name, mat in (("StatorLam", STATOR_LAMINATION), ("RotorForging", ROTOR_FORGING)):
        if linear:      # diagnostic / --linear mode: constant mu_r = 1000
            femm.mi_addmaterial(name, 1000, 1000, 0, 0, 0, 0, 0, 1, 0, 0, 0)
            continue
        femm.mi_addmaterial(name, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
        for B, H in zip(mat.B_table, mat.H_table):
            femm.mi_addbhpoint(name, float(B), float(H))
    femm.mi_addboundprop("A0", 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for c in PHASES + ("F",):
        femm.mi_addcircprop(c, 0, 1)                       # series circuits
    # drawing ---------------------------------------------------------------
    dr = FemmDrawer(femm)
    st, rt = slot_angles(d)
    fine, coarse = 0.5, 2.0                                 # max degrees per arc segment
    dr.circle(g["R_out"], [0, 180], coarse)
    dr.circle(g["R_bore"], [a for s in st for a in s], fine)
    for a0, a1 in st:
        for r in (g["s1"], g["s2"], g["s3"]):
            dr.arc(r, a0, a1, fine)
        for a in (a0, a1):
            dr.seg(g["R_bore"], g["s1"], a); dr.seg(g["s1"], g["s2"], a); dr.seg(g["s2"], g["s3"], a)
    dr.circle(g["R_rotor"], [a for _, a0, a1 in rt for a in (a0, a1)], fine)
    for _, a0, a1 in rt:
        for r in (g["r_slot_bottom"], g["r_wedge_bottom"]):
            dr.arc(r, a0, a1, fine)
        for a in (a0, a1):
            dr.seg(g["r_slot_bottom"], g["r_wedge_bottom"], a); dr.seg(g["r_wedge_bottom"], g["R_rotor"], a)
    # outer boundary A = 0
    for deg in (90, 270):
        x, y = dr.xy(g["R_out"], deg)
        femm.mi_selectarcsegment(x, y)
    femm.mi_setarcsegmentprop(coarse, "A0", 0, 0)
    femm.mi_clearselected()
    # block labels ----------------------------------------------------------
    m_gap, m_slot, m_iron = d.gap / 6, 0.010, 0.025
    dr.label(0.5 * (g["s3"] + g["R_out"]), 0.1, "StatorLam", m_iron)
    dr.label(0.5 * g["r_slot_bottom"], 0.1, "RotorForging", m_iron, group=ROTOR_GROUP)
    dr.label(0.5 * (g["R_rotor"] + g["R_bore"]), 0.1, "Air", m_gap)
    lay = winding_layout(d)
    for k, (a0, a1) in enumerate(st):
        am = 0.5 * (a0 + a1)
        dr.label(0.5 * (g["R_bore"] + g["s1"]), am, "Air", m_slot)
        for layer, (r0, r1) in enumerate(((g["s1"], g["s2"]), (g["s2"], g["s3"]))):
            ph, sg = lay[layer][k]
            dr.label(0.5 * (r0 + r1), am, "Copper", m_slot, PHASES[ph], 0,
                     sg * d.turns_per_coil / d.parallel_paths)
    fs = field_sign(d, np.arange(d.Qr_pitches))
    for i, a0, a1 in rt:
        am = 0.5 * (a0 + a1)
        dr.label(0.5 * (g["r_wedge_bottom"] + g["R_rotor"]), am, "Air", m_slot, group=ROTOR_GROUP)
        dr.label(0.5 * (g["r_slot_bottom"] + g["r_wedge_bottom"]), am, "Copper", m_slot, "F",
                 ROTOR_GROUP, float(fs[i] * d.Nf_per_slot))
    femm.mi_zoomnatural()
    try:
        femm.mi_smartmesh(0)          # use the explicit block mesh sizes only
    except Exception:
        pass
    femm.mi_saveas(path)


def mesh_check(femm):
    """Run the mesher only; returns the number of mesh nodes (None if unavailable)."""
    try:
        n = femm.mi_createmesh()
        femm.mi_purgemesh()
        return n
    except Exception as e:                          # noqa: BLE001
        print("   mesher error:", e)
        return None


def diagnose(femm, d: SCDesign, out_dir: str, show: bool):
    """Find out which step fails: meshing, the linear solve or the nonlinear iteration."""
    print("\n=== FEMM diagnostics ===")
    nl = os.path.abspath(os.path.join(out_dir, "sc_model.fem"))
    femm.opendocument(nl)
    n = mesh_check(femm)
    print(f"1) mesh of nonlinear model: {n} nodes" if n else "1) MESH FAILED (geometry problem)")
    lin = os.path.abspath(os.path.join(out_dir, "sc_model_linear.fem"))
    build_femm_model(femm, d, lin, linear=True)
    femm.mi_modifycircprop("F", 1, 100.0)
    femm.mi_saveas(lin)
    try:
        femm.mi_analyze(0 if show else 1)
        femm.mi_loadsolution()
        psi = femm.mo_getcircuitproperties("F")[2]
        femm.mo_close()
        print(f"2) linear model (mu_r = 1000) SOLVED, field flux linkage = {psi:.4f} Wb")
        print("   -> geometry and mesh are fine; the NONLINEAR iteration is what fails.")
        print("      Run the comparison with linear iron instead:  python femm_compare.py --linear")
    except Exception as e:                          # noqa: BLE001
        print(f"2) linear model ALSO FAILED: {e}")
        print("   -> meshing/solver problem. Open femm_model\\sc_model_linear.fem in the FEMM window,")
        print("      press 'Mesh' (yellow grid) and then 'Run' (crank): FEMM shows the detailed message.")
    print("========================")


def femm_solve(femm, d: SCDesign, i_abc, i_f, path=None, show=False):
    for c, i in zip(PHASES, i_abc):
        femm.mi_modifycircprop(c, 1, float(i))
    femm.mi_modifycircprop("F", 1, float(i_f))
    if path:
        femm.mi_saveas(path)             # make sure FEMM solves the current state
    femm.mi_analyze(0 if show else 1)
    femm.mi_loadsolution()
    psi = [femm.mo_getcircuitproperties(c)[2] for c in PHASES]
    femm.mo_groupselectblock(ROTOR_GROUP)
    torque = femm.mo_blockintegral(22)
    femm.mo_clearblock()
    r = d.R_rotor + 0.5 * d.gap
    th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    Br = []
    for t in th:
        bx, by = femm.mo_getb(r * math.cos(t), r * math.sin(t))
        Br.append(bx * math.cos(t) + by * math.sin(t))
    femm.mo_close()
    B1 = fundamental_axis(th, np.array(Br))[0]
    return np.array(psi, float), float(np.real(torque)), B1


# ----------------------------------------------------------------- compare
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="femm_model")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--linear", action="store_true",
                    help="use linear iron (mu_r = 1000) in both FEMM and Python")
    a = ap.parse_args()
    try:
        import femm
    except ImportError:
        raise SystemExit("pyfemm not found: install FEMM 4.2 (femm.info) and run  pip install pyfemm")

    os.makedirs(a.out, exist_ok=True)
    fem_path = os.path.abspath(os.path.join(a.out, "sc_model.fem"))
    d = SCDesign()
    femm.openfemm(0 if a.show else 1)
    build_femm_model(femm, d, fem_path, linear=a.linear)
    print("FEMM model written:", fem_path)
    n = mesh_check(femm)
    print(f"FEMM mesh check: {n} nodes" if n else "FEMM mesh check: no node count returned")
    if a.build_only:
        femm.closefemm()
        return

    print("Python FEM: axes, OCC, rated point ...")
    an = SCAnalysis(d, verbose=False)
    an.find_axes()
    an.occ_auto(quick=True)
    If0 = an.res.occ["If_noload"]
    If_r, _ = an.field_for_point(d.I_rated, d.V_phase, If_hi=6 * If0)
    Ipk = math.sqrt(2) * d.I_rated
    td = an.theta_d()
    cases = [
        ("No-load, If0", [0, 0, 0], If0),
        ("Rated over-excited (-d), If rated", an.currents(Ipk, td + math.pi), If_r),
        ("Rated I on q-axis, If rated (torque)", an.currents(Ipk, td + math.pi / 2), If_r),
        ("5% I on +d, If = 0 (Xd unsat.)", an.currents(0.05 * Ipk, td), 0.0),
    ]

    def dq(psi):
        th = an.res.axes["phase"]
        pd = 2 / 3 * sum(p * math.cos(td - t) for p, t in zip(psi, th))
        pq = 2 / 3 * sum(p * math.cos(td + math.pi / 2 - t) for p, t in zip(psi, th))
        return pd, pq

    rows = []
    for name, cur, If in cases:
        t0 = time.time()
        sp = an.solve(i_abc=cur, i_f=If, **({"linear_mu_r": 1000.0} if a.linear else {}))
        th, Br, _ = an.model.gap_flux_density(sp, radius=d.R_rotor + 0.5 * d.gap)
        py = dict(psi=dq(sp.psi_abc), T=sp.torque, B1=fundamental_axis(th, Br)[0])
        try:
            psi_f, T_f, B1_f = femm_solve(femm, d, cur, If, fem_path, a.show)
        except Exception as e:                      # noqa: BLE001
            print(f"\nFEMM failed on case '{name}': {e}")
            diagnose(femm, d, a.out, a.show)
            femm.closefemm()
            raise SystemExit(1)
        fm = dict(psi=dq(psi_f), T=T_f, B1=B1_f)
        rows.append((name, py, fm))
        print(f"  {name:40s} done ({time.time() - t0:.0f} s)")
    femm.closefemm()

    def pct(a_, b_):
        return "—" if abs(b_) < 1e-9 * max(1.0, abs(a_)) else f"{100 * (a_ - b_) / b_:+.2f} %"

    lines = ["| Case | Quantity | Python FEM | FEMM | Python vs FEMM |", "|---|---|---|---|---|"]
    for name, py, fm in rows:
        E_py = math.sqrt(3) * an.E_phase(math.hypot(*py["psi"])) / 1e3
        E_fm = math.sqrt(3) * an.E_phase(math.hypot(*fm["psi"])) / 1e3
        lines += [f"| {name} | ψd [Wb] | {py['psi'][0]:.3f} | {fm['psi'][0]:.3f} | {pct(py['psi'][0], fm['psi'][0])} |",
                  f"|  | ψq [Wb] | {py['psi'][1]:.3f} | {fm['psi'][1]:.3f} | |",
                  f"|  | E line [kV] | {E_py:.3f} | {E_fm:.3f} | {pct(E_py, E_fm)} |",
                  f"|  | Air-gap B₁ [T] | {py['B1']:.4f} | {fm['B1']:.4f} | {pct(py['B1'], fm['B1'])} |",
                  f"|  | Torque [kNm] | {py['T'] / 1e3:.2f} | {fm['T'] / 1e3:.2f} | {pct(py['T'], fm['T']) if abs(fm['T']) > 1e3 else '—'} |"]
    xd_py = d.omega * rows[3][1]["psi"][0] / (0.05 * Ipk) / d.Z_base + d.X_end_pu
    xd_fm = d.omega * rows[3][2]["psi"][0] / (0.05 * Ipk) / d.Z_base + d.X_end_pu
    lines.append(f"| Derived | Xd [pu] | {xd_py:.3f} | {xd_fm:.3f} | {pct(xd_py, xd_fm)} |")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(a.out, "femm_comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("# Python FEM vs FEMM 4.2\n\n"
                 f"If0 = {If0:.0f} A, If rated = {If_r:.0f} A, I rated = {d.I_rated:.0f} A.\n\n"
                 + table + "\n\nDifferences of 1–3 % are expected (different meshes; FEMM uses its own "
                 "B–H interpolation; torque from the weighted stress tensor vs Arkkio).\n")
    print("\nSaved", os.path.join(a.out, "femm_comparison.md"))


if __name__ == "__main__":
    main()
