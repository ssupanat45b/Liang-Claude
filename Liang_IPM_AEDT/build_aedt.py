#!/usr/bin/env python3
"""Build the 300 kVAr IPM synchronous condenser (flat magnets) in ANSYS Electronics Desktop.

Creates a Maxwell 2D *Transient (XY)* design with PyAEDT from exactly the same design
data as the Python FEM (ipm_fem2d): same stator slots, double-layer winding, flat
magnets with flux barriers and bridges, B-H curves, stack length and air gap.

    pip install pyaedt                       # = ansys-aedt-core (PyAEDT >= 0.10)
    python build_aedt.py                     # best design of Liang_IPM_MagGap (else BO, else ipm300)
    python build_aedt.py --design ipm300
    python build_aedt.py --design bo --version 2025.1 --solve
    python build_aedt.py --load noload       # magnets only: EMF and cogging

What is created
---------------
* materials: M330-50A (stator) and rotor steel with nonlinear B-H, NdFeB N38SH at 80 C
  (Br 1.141 T, mu_r 1.05 -> Hc = Br/(mu0 mu_r)), copper, vacuum
* stator lamination, 2 x Qs coil sides, rotor core, magnets (each with its own
  coordinate system for the parallel magnetisation, N/S alternating), flux barriers (air)
* band (rotating region), rotation 1500 rpm, vector potential A = 0 on the outer boundary
* windings PhaseA/B/C (current-driven, Ipk*cos(w t + phase)), coils with turns and polarity,
  parallel branches; the current is placed on the -d axis = rated over-excited operation,
  using the phase axes computed by the Python FEM (identical winding/geometry)
* mesh: length-based on magnets, bridges/rotor, coils, band
* transient setup: 2 electrical periods; torque, flux-linkage and induced-voltage reports

The Python FEM results of the same design are written to aedt_expected.json for comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ----------------------------------------------------------------------------- design
def _knee(rows, keys):
    F = np.array([[float(r[k]) for k in keys] for r in rows])
    lo, hi = F.min(0), F.max(0)
    N = (F - lo) / np.where(hi > lo, hi - lo, 1)
    return rows[int(np.argmin(np.linalg.norm(N, axis=1)))]


def load_design(which):
    """Return (PMDesign, description)."""
    from design_space import make_design
    from ipm_fem2d.machine import design_preset
    from variants import load_base, make_variant
    mg = os.path.join(HERE, "..", "Liang_IPM_MagGap", "results", "pareto.csv")
    bo = os.path.join(HERE, "..", "Liang_IPM_BO", "results", "pareto.csv")
    if which in ("auto", "maggap") and os.path.exists(mg):
        with open(mg) as fh:
            pf = list(csv.DictReader(fh))
        if pf:
            r = _knee(pf, ["thd_line_pct", "P_loss_kW", "cogging_Nm", "I_cold_pu"])
            base, _ = load_base(bo)
            v = {k: float(r[k]) for k in ("h_m", "w_m", "gap")}
            return make_variant(base, v), (f"Liang_IPM_MagGap best compromise: magnet {v['w_m'] * 1e3:.0f} x "
                                           f"{v['h_m'] * 1e3:.1f} mm, gap {v['gap'] * 1e3:.1f} mm")
    if which in ("auto", "maggap", "bo") and os.path.exists(bo):
        base, src = load_base(bo)
        d, _ = make_design(base)
        return d, f"Liang_IPM_BO best compromise ({src})"
    if which in ("auto", "maggap", "bo"):
        print("note: no optimisation results found next to this folder - using the built-in BO compromise")
        base, src = load_base(bo)
        d, _ = make_design(base)
        return d, src
    return design_preset("ipm300"), "ipm300 preset (4-pole, 400 mm bore, 213 x 41 mm magnets)"


# ----------------------------------------------------------------------------- geometry (mm)
def sector(r_in, r_out, a0, a1, step_deg=0.5):
    """Closed annular-sector polygon (mm, degrees)."""
    n = max(2, int(math.ceil(abs(a1 - a0) / step_deg)) + 1)
    t = np.radians(np.linspace(a0, a1, n))
    inner = [(r_in * math.cos(x), r_in * math.sin(x)) for x in t]
    outer = [(r_out * math.cos(x), r_out * math.sin(x)) for x in t[::-1]]
    return inner + outer


def geometry_mm(d):
    """All polygons of the machine in mm (pure Python - also used for the preview)."""
    from ipm_fem2d.dimensions import geometry
    from ipm_fem2d.machine import winding_layout
    from ipm_fem2d.rotor_rect import pole_polygons
    g = geometry(d)
    mm = 1e3
    lay = winding_layout(d)
    slots, coils = [], []
    for k in range(d.Qs):
        a0 = k * g["s_pitch_deg"] + g["s_slot_start"]
        a1 = a0 + g["s_slot_deg"]
        slots.append(sector(g["R_bore"] * mm, g["s3"] * mm, a0, a1))
        for layer, (r0, r1) in enumerate(((g["s1"], g["s2"]), (g["s2"], g["s3"]))):
            ph, sg = lay[layer][k]
            coils.append(dict(name=f"Coil_{'ABC'[ph]}_{k + 1:02d}_{'top' if layer == 0 else 'bot'}",
                              phase="ABC"[ph], sign=sg, poly=sector(r0 * mm, r1 * mm, a0, a1)))
    magnets, barriers = [], []
    for k in range(d.poles):
        mag, bu, bl = pole_polygons(d, k)
        magnets.append(dict(name=f"Magnet_{k + 1}", pole=k, poly=[(x * mm, y * mm) for x, y in mag],
                            axis_deg=360.0 * k / d.poles, north=(k % 2 == 0)))
        for j, b in enumerate((bu, bl)):
            barriers.append(dict(name=f"Barrier_{k + 1}{'ab'[j]}", poly=[(x * mm, y * mm) for x, y in b]))
    return dict(R_out=d.R_stator_out * mm, R_bore=d.R_bore * mm, R_rotor=d.R_rotor * mm,
                R_band=(d.R_rotor + 0.5 * d.gap) * mm, slots=slots, coils=coils,
                magnets=magnets, barriers=barriers)


def preview(d, G, path):
    """PNG of the geometry that will be sent to AEDT (to check before opening AEDT)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_patch(Circle((0, 0), G["R_out"], fc="#b9c2cf", ec="k"))
    ax.add_patch(Circle((0, 0), G["R_bore"], fc="w", ec="k"))
    for s in G["slots"]:
        ax.add_patch(Polygon(s, fc="w", ec="k", lw=0.3))
    col = dict(A="#c0392b", B="#f1c40f", C="#2980b9")
    for c in G["coils"]:
        ax.add_patch(Polygon(c["poly"], fc=col[c["phase"]], ec="k", lw=0.2, alpha=0.9 if c["sign"] > 0 else 0.5))
    ax.add_patch(Circle((0, 0), G["R_band"], fill=False, ec="#7f8c8d", ls="--", lw=0.6))
    ax.add_patch(Circle((0, 0), G["R_rotor"], fc="#8c96a6", ec="k"))
    for b in G["barriers"]:
        ax.add_patch(Polygon(b["poly"], fc="w", ec="k", lw=0.4))
    for m in G["magnets"]:
        ax.add_patch(Polygon(m["poly"], fc="#c2185b" if m["north"] else "#1565c0", ec="k", lw=0.4))
        a = math.radians(m["axis_deg"]); s = 1 if m["north"] else -1
        cx, cy = np.mean(np.array(m["poly"]), axis=0)
        ax.annotate("", (cx + s * 25 * math.cos(a), cy + s * 25 * math.sin(a)), (cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color="w", lw=1.5))
    R = G["R_out"] * 1.03
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
    ax.set_title(f"AEDT model preview — {d.poles}-pole, Qs = {d.Qs}, L = {d.L_stack * 1e3:.0f} mm "
                 "(arrows = magnetisation; red/yellow/blue = phase A/B/C, pale = return side)", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- python reference
def python_reference(d):
    from ipm_fem2d.analysis import PMAnalysis
    an = PMAnalysis(d, verbose=False)
    an.find_axes(); an.no_load_emf()
    I, sol = an.current_for_voltage(d.V_phase)
    pd, pq = an.psi_dq(sol)
    ax = an.res.axes
    return an, dict(I_rms=I, Q_kVAr=3 * d.V_phase * I / 1e3, theta_d_deg=math.degrees(ax["d"]),
                    phase_axes_deg=[math.degrees(t) for t in ax["phase"]],
                    psi_d_rated=pd, E0_line_hot=an.res.emf["E0_line_hot"],
                    psi_pm=an.res.emf["psi_pm_hot"], torque_rated_Nm=sol.torque)


# ----------------------------------------------------------------------------- AEDT
def build(d, G, ref, args):
    try:
        from ansys.aedt.core import Maxwell2d
    except ImportError:
        try:
            from pyaedt import Maxwell2d                    # older PyAEDT
        except ImportError:
            raise SystemExit("PyAEDT not found:  pip install pyaedt")
    from ipm_fem2d.materials import ROTOR_FORGING, STATOR_LAMINATION, MU0

    load = args.load
    proj = os.path.abspath(os.path.join(args.out, f"IPM_SynCon_300kVAr_{load}.aedt"))
    try:
        m2d = Maxwell2d(project=proj, design=f"IPM_{d.poles}p_{load}", solution_type="TransientXY",
                        version=args.version, non_graphical=args.non_graphical, new_desktop=True)
    except Exception as e:                                  # noqa: BLE001
        raise SystemExit(
            f"PyAEDT could not start/connect to AEDT ({type(e).__name__}: {e}).\n"
            "If AEDT showed 'File not found: ...\\<number>', this AEDT version does not accept the gRPC\n"
            "start-up argument of this PyAEDT. Use the native script instead (no PyAEDT needed):\n"
            f"   AEDT > Tools > Run Script... > {os.path.abspath(args.out)}\\IPM_SynCon_aedt_script_{load}.py")
    m2d.modeler.model_units = "mm"
    m2d.model_depth = f"{d.L_stack * 1e3:.3f}mm"

    # ---- design variables -------------------------------------------------------------
    f = d.f
    Ipk = math.sqrt(2) * abs(ref["I_rms"]) if load == "rated" else 0.0
    m2d["Ipk"] = f"{Ipk:.3f}A"
    m2d["Freq"] = f"{f}Hz"
    m2d["Speed_rpm"] = f"{d.rpm:.1f}"
    m2d["Theta_d0"] = f"{ref['theta_d_deg']:.4f}deg"

    # ---- materials -----------------------------------------------------------------------
    def bh_material(name, mat):
        m = m2d.materials.add_material(name)
        m.permeability.value = [[float(H), float(B)] for B, H in zip(mat.B_table, mat.H_table)]
        m.conductivity = 0
        return m
    bh_material("M330_50A_pyFEM", STATOR_LAMINATION)
    bh_material("RotorSteel_pyFEM", ROTOR_FORGING)
    mag = m2d.materials.add_material("NdFeB_N38SH_80C")
    Br = d.Br()
    mag.permeability = d.mu_rm
    mag.set_magnetic_coercivity(value=-Br / (MU0 * d.mu_rm), x=1, y=0, z=0)
    mag.conductivity = 0                                    # magnetostatic-equivalent (no magnet eddy)

    # ---- stator ----------------------------------------------------------------------------
    md = m2d.modeler
    stator = md.create_circle([0, 0, 0], G["R_out"], name="Stator", material="M330_50A_pyFEM")
    bore = md.create_circle([0, 0, 0], G["R_bore"], name="Bore_tmp")
    md.subtract(stator, bore, keep_originals=False)
    slot_objs = [md.create_polyline([[x, y, 0] for x, y in s], close_surface=True, cover_surface=True,
                                    name=f"Slot_{k + 1:02d}") for k, s in enumerate(G["slots"])]
    md.subtract(stator, slot_objs, keep_originals=False)
    coil_objs = {}
    for c in G["coils"]:
        o = md.create_polyline([[x, y, 0] for x, y in c["poly"]], close_surface=True, cover_surface=True,
                               name=c["name"], material="copper")
        coil_objs[c["name"]] = (o, c)

    # ---- rotor -----------------------------------------------------------------------------
    rotor = md.create_circle([0, 0, 0], G["R_rotor"], name="Rotor", material="RotorSteel_pyFEM")
    tools = [md.create_polyline([[x, y, 0] for x, y in b["poly"]], close_surface=True, cover_surface=True,
                                name=b["name"]) for b in G["barriers"]]
    mag_objs = []
    for m in G["magnets"]:
        o = md.create_polyline([[x, y, 0] for x, y in m["poly"]], close_surface=True, cover_surface=True,
                               name=m["name"], material="NdFeB_N38SH_80C")
        # parallel magnetisation along the pole axis: local x of a rotated CS (S poles: +180 deg)
        ang = (m["axis_deg"] + (0.0 if m["north"] else 180.0)) % 360.0
        cs = md.create_coordinate_system(origin=[0, 0, 0], name=f"CS_{m['name']}", mode="zyz",
                                         phi=f"{ang}deg")
        o.part_coordinate_system = cs.name
        mag_objs.append(o)
        md.set_working_coordinate_system("Global")
    # barriers and magnet pockets are removed from the rotor core (barriers stay air)
    md.subtract(rotor, tools + [md.create_polyline([[x, y, 0] for x, y in m["poly"]], close_surface=True,
                                                    cover_surface=True, name=f"Pocket_{m['name']}")
                                for m in G["magnets"]], keep_originals=False)

    # ---- band, region, boundary, motion -----------------------------------------------------
    band = md.create_circle([0, 0, 0], G["R_band"], name="Band", material="vacuum")
    region = md.create_circle([0, 0, 0], G["R_out"] * 1.02, name="Region", material="vacuum")
    m2d.assign_vector_potential(assignment=region.edges, vector_value=0, boundary="VectorPotential_A0")
    m2d.assign_rotate_motion(assignment=band.name, coordinate_system="Global", axis="Z", positive_movement=True,
                             start_position="0deg", angular_velocity=f"{d.rpm:.1f}rpm")

    # ---- windings: Ipk*cos(w t + Theta_d0 + 180 deg - theta_phase) -> current on the -d axis -------
    for ph, th in zip("ABC", ref["phase_axes_deg"]):
        phase = f"Theta_d0 + 180deg - ({th:.4f}deg)" if load == "rated" else "0deg"
        expr = f"Ipk*cos(2*pi*Freq*Time + {phase})"
        w = m2d.assign_winding(assignment=None, winding_type="Current", is_solid=False, current=expr,
                               parallel_branches=d.parallel_paths, name=f"Phase{ph}")
        names = []
        for nm, (o, c) in coil_objs.items():
            if c["phase"] != ph:
                continue
            m2d.assign_coil(assignment=[o.name], conductors_number=d.turns_per_coil,
                            polarity="Positive" if c["sign"] > 0 else "Negative", name=f"C_{nm}")
            names.append(f"C_{nm}")
        m2d.add_winding_coils(assignment=w.name, coils=names)

    # ---- torque on the rotor ----------------------------------------------------------------
    try:
        m2d.assign_torque(assignment=[rotor.name] + [o.name for o in mag_objs], is_positive=True,
                          is_virtual=True, torque_name="Torque_rotor")
    except Exception as e:                                  # noqa: BLE001  (motion also reports torque)
        print("   torque parameter skipped:", e)

    # ---- mesh ---------------------------------------------------------------------------------
    m2d.mesh.assign_length_mesh([o.name for o in mag_objs], maximum_length=2, maximum_elements=None, name="Mesh_magnets")
    m2d.mesh.assign_length_mesh([rotor.name], maximum_length=4, maximum_elements=None, name="Mesh_rotor")
    m2d.mesh.assign_length_mesh([o.name for o, _ in coil_objs.values()], maximum_length=3,
                                maximum_elements=None, name="Mesh_coils")
    m2d.mesh.assign_length_mesh([stator.name], maximum_length=5, maximum_elements=None, name="Mesh_stator")
    m2d.mesh.assign_length_mesh([band.name], maximum_length=1.5, maximum_elements=None, name="Mesh_band")

    # ---- transient setup ------------------------------------------------------------------------
    period = 1.0 / f
    setup = m2d.create_setup(name="Transient")
    setup.props["StopTime"] = f"{2 * period * 1e3:.3f}ms"
    setup.props["TimeStep"] = f"{period / 200 * 1e3:.4f}ms"
    setup.props["SaveFieldsType"] = "Every N Steps"
    setup.props["N Steps"] = "20"
    setup.props["Steps From"] = "0ms"
    setup.props["Steps To"] = f"{2 * period * 1e3:.3f}ms"
    setup.update()

    # ---- reports ----------------------------------------------------------------------------------
    for title, exprs in (("Torque", ["Moving1.Torque"]),
                         ("Flux linkage", [f"FluxLinkage(Phase{p})" for p in "ABC"]),
                         ("Induced voltage", [f"InducedVoltage(Phase{p})" for p in "ABC"]),
                         ("Winding current", [f"Current(Phase{p})" for p in "ABC"])):
        try:
            m2d.post.create_report(expressions=exprs, setup_sweep_name="Transient : Transient",
                                   primary_sweep_variable="Time", plot_name=title)
        except Exception as e:                              # noqa: BLE001
            print(f"   report '{title}' skipped:", e)

    m2d.save_project()
    if args.solve:
        print("solving ... (this can take several minutes)")
        m2d.analyze_setup("Transient")
        m2d.save_project()
    print("AEDT project saved:", proj)
    if not args.keep_open:
        m2d.release_desktop(close_projects=False, close_desktop=False)
    return proj


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="auto", choices=["auto", "maggap", "bo", "ipm300"])
    ap.add_argument("--load", default="rated", choices=["rated", "noload"],
                    help="rated: rated over-excited current on the -d axis; noload: magnets only (EMF, cogging)")
    ap.add_argument("--version", default=None, help="AEDT version, e.g. 2024.2 or 2025.1 (default: latest installed)")
    ap.add_argument("--non-graphical", action="store_true")
    ap.add_argument("--solve", action="store_true", help="also run the transient solution")
    ap.add_argument("--keep-open", action="store_true")
    ap.add_argument("--preview-only", action="store_true", help="only write the geometry preview + expected values")
    ap.add_argument("--script-only", action="store_true",
                    help="only write the native AEDT script (use when PyAEDT cannot start AEDT)")
    ap.add_argument("--out", default="aedt_model")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d, src = load_design(a.design)
    print(f"Design: {src}\n  {d.poles}-pole, Qs {d.Qs}, bore {d.R_bore * 2e3:.1f} mm, rotor {d.R_rotor * 2e3:.1f} mm, "
          f"L {d.L_stack * 1e3:.1f} mm, magnets {d.h_m * 1e3:.1f} mm, {d.turns_per_coil} turns/coil, "
          f"{d.parallel_paths} parallel path(s)")
    G = geometry_mm(d)
    preview(d, G, os.path.join(a.out, "aedt_geometry_preview.png"))
    print("Python FEM reference of the same design ...")
    _, ref = python_reference(d)
    ref.update(design=src, load=a.load, Ipk_A=math.sqrt(2) * abs(ref["I_rms"]) if a.load == "rated" else 0.0,
               expected=dict(
                   flux_linkage_peak_Wb=abs(ref["psi_d_rated"]) if a.load == "rated" else abs(ref["psi_pm"]),
                   induced_voltage_line_rms_V=(math.sqrt(3) * d.omega * abs(ref["psi_d_rated"]) / math.sqrt(2)
                                               if a.load == "rated" else ref["E0_line_hot"]),
                   torque_mean_Nm=0.0,
                   note="AEDT InducedVoltage is per phase (peak = omega*psi). End-winding leakage is not in the 2-D "
                        "model of either code; the stator skew (1 slot) is not modelled in Maxwell 2D."))
    with open(os.path.join(a.out, "aedt_expected.json"), "w") as fh:
        json.dump(ref, fh, indent=1)
    print(json.dumps(ref["expected"], indent=1))
    import aedt_native
    sp = aedt_native.export(d, G, ref, a.out, a.load)
    print("native AEDT script (no PyAEDT needed, run in AEDT: Tools > Run Script):", os.path.abspath(sp))
    if a.preview_only or a.script_only:
        return
    build(d, G, ref, a)


if __name__ == "__main__":
    main()
