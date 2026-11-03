#!/usr/bin/env python3
"""Cross-check the IPM Python FEM against FEMM 4.2 (via pyfemm) on the same machine.

Requirements (Windows):  FEMM 4.2 from https://www.femm.info  +  pip install pyfemm

    python femm_compare_ipm.py                # build model, solve 4 cases in FEMM and in Python, compare
    python femm_compare_ipm.py --build-only   # only write femm_model_ipm300/ipm_model.fem (open in FEMM GUI)
    python femm_compare_ipm.py --show         # keep the FEMM window visible while solving

The FEMM model is drawn from the same PMDesign data (ipm_fem2d/machine.py): identical
annular-sector slots, winding layout, B-H tables, stack length, buried arc magnets,
flux barriers and bridges
(radial magnetisation "theta" / "theta+180", Hc = Br/(mu0*mu_r) at the operating temperature).  Flux linkages
are read from FEMM circuits, torque from the weighted Maxwell stress tensor on the
rotor blocks, and air-gap B from point samples.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from ipm_fem2d.analysis import PMAnalysis, fundamental_axis
from ipm_fem2d.dimensions import geometry
from ipm_fem2d.machine import DESIGN_PRESETS, PMDesign, design_preset, winding_layout
from ipm_fem2d.materials import MU0, ROTOR_FORGING, STATOR_LAMINATION

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

    def node_xy(self, x, y):
        key = (round(x, 7), round(y, 7))
        if key not in self.nodes:
            self.f.mi_addnode(x, y)
            self.nodes.add(key)
        return x, y

    def polyline(self, pts, closed=True):
        m = len(pts)
        for j in range(m if closed else m - 1):
            (x1, y1), (x2, y2) = pts[j], pts[(j + 1) % m]
            self.node_xy(x1, y1); self.node_xy(x2, y2)
            self.f.mi_addsegment(x1, y1, x2, y2)

    def label_xy(self, x, y, material, mesh, group=0, magdir=0):
        self.f.mi_addblocklabel(x, y)
        self.f.mi_selectlabel(x, y)
        self.f.mi_setblockprop(material, 0, mesh, "<None>", magdir, group, 0)
        self.f.mi_clearselected()

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

    def label(self, r, deg, material, mesh, circuit="<None>", group=0, turns=0, magdir=0):
        x, y = self.xy(r, deg)
        self.f.mi_addblocklabel(x, y)
        self.f.mi_selectlabel(x, y)
        self.f.mi_setblockprop(material, 0, mesh, circuit, magdir, group, turns)
        self.f.mi_clearselected()


def slot_angles(d: PMDesign):
    g = geometry(d)
    st = [(k * g["s_pitch_deg"] + g["s_slot_start"],
           k * g["s_pitch_deg"] + g["s_slot_start"] + g["s_slot_deg"]) for k in range(d.Qs)]
    mags = [(kp, kp * g["pole_deg"] - g["mag_deg"] / 2, kp * g["pole_deg"] + g["mag_deg"] / 2)
            for kp in range(d.poles)]
    return st, mags


def build_femm_model(femm, d: PMDesign, path: str, linear: bool = False):
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
    Hc = d.Br() / (MU0 * d.mu_rm)                           # magnet coercivity at T_mag [A/m]
    femm.mi_addmaterial("NdFeB", d.mu_rm, d.mu_rm, Hc, 0, 0, 0, 0, 1, 0, 0, 0)
    for c in PHASES:
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
    if d.magnet_shape == "rect":
        from ipm_fem2d.rotor_rect import interior_point, pole_polygons
        dr.circle(g["R_rotor"], [0, 180], fine)
        for k in range(d.poles):
            mag, bu, bl = pole_polygons(d, k)
            dr.polyline(mag)
            for bar in (bu, bl):
                dr.polyline(bar[1:] + [bar[0]], closed=False)          # A-B is the magnet end
        rt = []
    # IPM rotor: outer surface is a plain circle; per pole a closed pocket made of
    #   barrier | magnet | barrier  between R_core and R_bridge, magnet top at R_mag
    if d.magnet_shape != "rect":
        dr.circle(g["R_rotor"], [0, 180], fine)
    for _, m0, m1 in rt:
        b0, b1 = m0 - g["bar_deg"], m1 + g["bar_deg"]
        for a0_, a1_ in ((b0, m0), (m0, m1), (m1, b1)):
            dr.arc(g["R_core"], a0_, a1_, fine)                 # pocket floor
        dr.arc(g["R_bridge"], b0, m0, fine)                     # barrier roofs (under the bridges)
        dr.arc(g["R_bridge"], m1, b1, fine)
        dr.arc(g["R_mag"], m0, m1, fine)                        # magnet top / pole-shoe underside
        for a in (b0, b1):
            dr.seg(g["R_core"], g["R_bridge"], a)               # outer barrier walls
        for a in (m0, m1):
            dr.seg(g["R_core"], g["R_mag"], a)                  # magnet ends
            dr.seg(g["R_mag"], g["R_bridge"], a)                # pole-shoe sides
    # outer boundary A = 0
    for deg in (90, 270):
        x, y = dr.xy(g["R_out"], deg)
        femm.mi_selectarcsegment(x, y)
    femm.mi_setarcsegmentprop(coarse, "A0", 0, 0)
    femm.mi_clearselected()
    # block labels ----------------------------------------------------------
    sc = d.R_stator_out / 1.12                     # mesh sizes scale with machine size
    m_gap, m_slot, m_iron = d.gap / 6, 0.010 * sc, 0.025 * sc
    dr.label(0.5 * (g["s3"] + g["R_out"]), 0.1, "StatorLam", m_iron)
    dr.label(0.5 * g["R_core"], 0.1, "RotorForging", m_iron, group=ROTOR_GROUP)
    dr.label(0.5 * (g["R_rotor"] + g["R_bore"]), 0.1, "Air", m_gap)
    lay = winding_layout(d)
    for k, (a0, a1) in enumerate(st):
        am = 0.5 * (a0 + a1)
        dr.label(0.5 * (g["R_bore"] + g["s1"]), am, "Air", m_slot)
        for layer, (r0, r1) in enumerate(((g["s1"], g["s2"]), (g["s2"], g["s3"]))):
            ph, sg = lay[layer][k]
            dr.label(0.5 * (r0 + r1), am, "Copper", m_slot, PHASES[ph], 0,
                     sg * d.turns_per_coil / d.parallel_paths)
    r_mid = 0.5 * (g["R_core"] + g["R_mag"])
    m_mag = min(m_slot, d.h_m / 3, d.t_shoe / 2)
    if d.magnet_shape == "rect":
        m_mag = min(m_slot, d.h_m / 4)
        for k in range(d.poles):
            mag, bu, bl = pole_polygons(d, k)
            cx, cy = np.mean(np.array(mag), axis=0)
            dr.label_xy(cx, cy, "NdFeB", m_mag, ROTOR_GROUP,
                        magdir=float((360.0 * k / d.poles + (0 if k % 2 == 0 else 180)) % 360))
            for bar in (bu, bl):
                bx, by = interior_point(bar)
                dr.label_xy(float(bx), float(by), "Air", m_mag / 2, ROTOR_GROUP)
    for kp, a0, a1 in rt:
        dr.label(r_mid, 0.5 * (a0 + a1), "NdFeB", m_mag, group=ROTOR_GROUP,
                 magdir="theta" if kp % 2 == 0 else "theta+180")       # radial N / S
        rb = 0.5 * (g["R_core"] + g["R_bridge"])
        dr.label(rb, a0 - 0.5 * g["bar_deg"], "Air", m_mag, group=ROTOR_GROUP)    # flux barriers
        dr.label(rb, a1 + 0.5 * g["bar_deg"], "Air", m_mag, group=ROTOR_GROUP)
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


def diagnose(femm, d: PMDesign, out_dir: str, show: bool):
    """Find out which step fails: meshing, the linear solve or the nonlinear iteration."""
    print("\n=== FEMM diagnostics ===")
    nl = os.path.abspath(os.path.join(out_dir, "ipm_model.fem"))
    femm.opendocument(nl)
    n = mesh_check(femm)
    print(f"1) mesh of nonlinear model: {n} nodes" if n else "1) MESH FAILED (geometry problem)")
    lin = os.path.abspath(os.path.join(out_dir, "ipm_model_linear.fem"))
    build_femm_model(femm, d, lin, linear=True)
    femm.mi_modifycircprop("A", 1, 100.0)
    femm.mi_saveas(lin)
    try:
        femm.mi_analyze(0 if show else 1)
        femm.mi_loadsolution()
        psi = femm.mo_getcircuitproperties("A")[2]
        femm.mo_close()
        print(f"2) linear model (mu_r = 1000) SOLVED, phase-A flux linkage = {psi:.4f} Wb")
        print("   -> geometry and mesh are fine; the NONLINEAR iteration is what fails.")
        print("      Run the comparison with linear iron instead:  python femm_compare_ipm.py --linear")
    except Exception as e:                          # noqa: BLE001
        print(f"2) linear model ALSO FAILED: {e}")
        print("   -> meshing/solver problem. Open femm_model_ipm300\\ipm_model_linear.fem in the FEMM window,")
        print("      press 'Mesh' (yellow grid) and then 'Run' (crank): FEMM shows the detailed message.")
    print("========================")


def femm_solve(femm, d: PMDesign, i_abc, path=None, show=False):
    for c, i in zip(PHASES, i_abc):
        femm.mi_modifycircprop(c, 1, float(i))
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
    B1 = fundamental_axis(th, np.array(Br), d.pp)[0]
    return np.array(psi, float), float(np.real(torque)), B1


# ----------------------------------------------------------------- compare
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="ipm300", choices=DESIGN_PRESETS)
    ap.add_argument("--out", default=None, help="default femm_model_<design>")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--linear", action="store_true",
                    help="use linear iron (mu_r = 1000) in both FEMM and Python")
    a = ap.parse_args()
    try:
        import femm
    except ImportError:
        raise SystemExit("pyfemm not found: install FEMM 4.2 (femm.info) and run  pip install pyfemm")

    d = design_preset(a.design)
    a.out = a.out or f"femm_model_{a.design}"
    os.makedirs(a.out, exist_ok=True)
    fem_path = os.path.abspath(os.path.join(a.out, "ipm_model.fem"))
    femm.openfemm(0 if a.show else 1)
    build_femm_model(femm, d, fem_path, linear=a.linear)
    print("FEMM model written:", fem_path)
    n = mesh_check(femm)
    print(f"FEMM mesh check: {n} nodes" if n else "FEMM mesh check: no node count returned")
    if a.build_only:
        femm.closefemm()
        return

    print("Python FEM: axes, rated point, short circuit ...")
    an = PMAnalysis(d, verbose=False)
    an.find_axes()
    I_r, _ = an.current_for_voltage(d.V_phase)
    I_sc, _ = an.current_for_voltage(0.0)
    Ipk = math.sqrt(2) * d.I_rated
    td = an.theta_d()
    cases = [
        ("Open circuit (magnets only)", [0, 0, 0]),
        (f"Rated voltage: I = {I_r:.0f} A on -d", an.reactive_currents(I_r)),
        (f"Short circuit: I = {I_sc:.0f} A on -d", an.reactive_currents(I_sc)),
        ("Rated I on q-axis (torque)", an.currents(Ipk, td + math.pi / 2)),
    ]

    def dq(psi):
        th = an.res.axes["phase"]
        pd = 2 / 3 * sum(p * math.cos(td - t) for p, t in zip(psi, th))
        pq = 2 / 3 * sum(p * math.cos(td + math.pi / 2 - t) for p, t in zip(psi, th))
        return pd, pq

    rows = []
    for name, cur in cases:
        t0 = time.time()
        sp = an.solve(i_abc=cur, pm=1.0, **({"linear_mu_r": 1000.0} if a.linear else {}))
        th, Br, _ = an.model.gap_flux_density(sp, radius=d.R_rotor + 0.5 * d.gap)
        py = dict(psi=dq(sp.psi_abc), T=sp.torque, B1=fundamental_axis(th, Br, d.pp)[0])
        try:
            psi_f, T_f, B1_f = femm_solve(femm, d, cur, fem_path, a.show)
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
        E_py = math.sqrt(3) * an.E_phase(math.hypot(*py["psi"]))
        E_fm = math.sqrt(3) * an.E_phase(math.hypot(*fm["psi"]))
        lines += [f"| {name} | ψd [Wb] | {py['psi'][0]:.3f} | {fm['psi'][0]:.3f} | {pct(py['psi'][0], fm['psi'][0])} |",
                  f"|  | ψq [Wb] | {py['psi'][1]:.3f} | {fm['psi'][1]:.3f} | |",
                  f"|  | E air-gap line [V] | {E_py:.1f} | {E_fm:.1f} | {pct(E_py, E_fm)} |",
                  f"|  | Air-gap B₁ [T] | {py['B1']:.4f} | {fm['B1']:.4f} | {pct(py['B1'], fm['B1'])} |",
                  f"|  | Torque [Nm] | {py['T']:.1f} | {fm['T']:.1f} | {pct(py['T'], fm['T']) if abs(fm['T']) > 50 else '—'} |"]
    # derived: saturated Xd from the open-circuit -> rated-voltage linkage change
    xd = [d.omega * (rows[0][k]["psi"][0] - rows[1][k]["psi"][0]) / (math.sqrt(2) * I_r) / d.Z_base
          + d.X_end_pu for k in (1, 2)]
    lines.append(f"| Derived | Xd (open circuit → rated) [pu] | {xd[0]:.3f} | {xd[1]:.3f} | {pct(*xd)} |")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(a.out, "femm_comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("# PM condenser: Python FEM vs FEMM 4.2\n\n"
                 f"Magnets {d.h_m * 1e3:.1f} mm, Br {d.Br():.3f} T at {d.T_mag:.0f} °C; "
                 f"I rated = {d.I_rated:.0f} A, I at rated V = {I_r:.0f} A, I short circuit = {I_sc:.0f} A.\n\n"
                 + table + "\n\nDifferences of 1–3 % are expected (different meshes; FEMM uses its own "
                 "B–H interpolation; torque from the weighted stress tensor vs Arkkio).\n")
    print("\nSaved", os.path.join(a.out, "femm_comparison.md"))


if __name__ == "__main__":
    main()
