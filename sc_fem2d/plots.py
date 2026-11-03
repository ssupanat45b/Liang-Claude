"""Figures for the synchronous-condenser FEM study."""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from . import machine as mc
from .machine import fmt_q, fmt_v


def _vscale(d):
    return (1e3, "kV") if d.V_line >= 1000 else (1.0, "V")

REGION_COLORS = {mc.AIR: "#ffffff", mc.ROTOR_IRON: "#8c96a6", mc.STATOR_IRON: "#b9c2cf",
                 mc.FIELD_COIL: "#d9822b", mc.ARM_COIL: "#c0392b", mc.ROTOR_WEDGE: "#d6d6d6",
                 mc.STATOR_WEDGE: "#ececec"}


def _tri(an, k):
    s = an.model.element_data(k)
    P = an.mesh.node_coords(k)
    return Triangulation(P[:, 0], P[:, 1], s["tri"]), s


def plot_geometry(an, path):
    tri, s = _tri(an, 0)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))
    cols = np.array([REGION_COLORS[r] for r in s["reg"]])
    for ax, zoom in zip(axs, (False, True)):
        from matplotlib.collections import PolyCollection
        P = np.c_[tri.x, tri.y]
        polys = P[s["tri"]]
        ax.add_collection(PolyCollection(polys, facecolors=cols,
                                         edgecolors="k" if zoom else "none", linewidths=0.15))
        ax.set_aspect("equal")
        R = an.d.R_stator_out
        if zoom:
            Rb = an.d.R_bore
            ax.set_xlim(0.45 * Rb, 1.52 * Rb); ax.set_ylim(-0.09 * Rb, 0.80 * Rb)
            ax.set_title("Mesh detail: rotor slots, air gap, stator slots")
        else:
            ax.set_xlim(-R * 1.02, R * 1.02); ax.set_ylim(-R * 1.02, R * 1.02)
            ax.set_title(f"Cross-section ({an.res.mesh_stats['elements']} elements, "
                         f"{an.res.mesh_stats['nodes']} nodes)")
    # phase labels on stator slots (top layer)
    lay = mc.winding_layout(an.d)
    for q in range(an.d.Qs):
        ph, sg = lay[0][q]
        a = (q + 0.5) * 2 * math.pi / an.d.Qs
        r = an.d.R_bore + an.d.s_wedge + 0.5 * an.d.s_layer
        axs[0].text(r * math.cos(a), r * math.sin(a), "ABC"[ph] + ("+" if sg > 0 else "-"),
                    fontsize=5, ha="center", va="center", color="w")
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color=c, label=mc.REGION_NAMES[r]) for r, c in REGION_COLORS.items()
                           if r != mc.AIR], loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_field(an, sol, title, path):
    tri, s = _tri(an, sol.k_shift)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    tpc = ax.tripcolor(tri, facecolors=sol.Bmag, cmap="turbo", vmin=0, vmax=2.4, shading="flat")
    lev = np.linspace(sol.A.min(), sol.A.max(), 32)[1:-1]
    ax.tricontour(tri, sol.A, levels=lev, colors="k", linewidths=0.5)
    fig.colorbar(tpc, ax=ax, label="|B|  [T]", shrink=0.85)
    ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_gap(an, path):
    g = an.res.gap_field
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[2, 1]))
    axs[0].plot(np.degrees(g["theta"]), g["Br"], lw=1, label="B_r")
    axs[0].plot(np.degrees(g["theta"]), g["Bt"], lw=0.8, label="B_t")
    axs[0].plot(np.degrees(g["theta"]), g["B1"] * np.cos(an.d.pp * g["theta"] - an.res.axes["d"]), "--",
                label=f"fundamental {g['B1']:.3f} T")
    axs[0].set_xlabel("stator angle [deg]"); axs[0].set_ylabel("B [T]"); axs[0].grid(alpha=.3)
    axs[0].legend(); axs[0].set_title("No-load air-gap flux density (rated voltage)")
    h = g["harm"]
    axs[1].bar(np.arange(1, len(h)), h[1:] / h[1] * 100)
    axs[1].set_yscale("log"); axs[1].set_xlabel("space harmonic order (electrical)")
    axs[1].set_ylabel("% of fundamental"); axs[1].set_title("Br harmonic spectrum")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_occ_vcurve(an, path):
    d, o, v = an.d, an.res.occ, an.res.vcurve
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    If = np.r_[0, o["If"]]
    vs, vu = _vscale(d)
    axs[0].plot(If, np.r_[0, o["E_line"]] / vs, "o-", label="OCC (FEM, saturated)")
    axs[0].plot(If, o["airgap_slope"] * If / vs, "--", label="air-gap line")
    axs[0].axhline(d.V_line / vs, color="gray", lw=0.8)
    axs[0].axvline(o["If_noload"], color="gray", lw=0.8)
    axs[0].set_xlabel("field current If [A]"); axs[0].set_ylabel(f"E line [{vu} rms]")
    axs[0].set_title(f"Open-circuit characteristic  (If0 = {o['If_noload']:.1f} A)")
    axs[0].grid(alpha=.3); axs[0].legend()
    axs[1].plot(v["If"], v["Q_kVAr"], "o-")
    axs[1].axhline(0, color="gray", lw=0.8)
    axs[1].set_xlabel("field current If [A]"); axs[1].set_ylabel("Q [kVAr]  (+ over-excited)")
    axs[1].set_title("Reactive capability / V-curve at rated voltage")
    axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_waveform(an, path):
    w, ta = an.res.waveform, an.res.torque_angle
    vs, vu = _vscale(an.d)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    axs[0].plot(np.degrees(w["angle"]), w["e"] / vs)
    axs[0].set_xlabel("rotor angle [electrical deg]"); axs[0].set_ylabel(f"phase EMF [{vu}]")
    axs[0].set_title(f"No-load phase-A EMF  (THD phase {100 * w['thd']:.2f} %, line {100 * w['thd_line']:.2f} %)"); axs[0].grid(alpha=.3)
    h = w["e_harm"]
    axs[1].bar(np.arange(1, len(h)), h[1:] / h[1] * 100)
    axs[1].set_yscale("log"); axs[1].set_xlabel("time harmonic order")
    axs[1].set_ylabel("% of fundamental"); axs[1].set_title("EMF harmonics")
    axs[2].plot(np.degrees(ta["gamma"]), ta["T"] / 1e3, "o-")
    axs[2].set_xlabel("current angle from +d axis [electrical deg]"); axs[2].set_ylabel("torque [kNm]")
    axs[2].set_title("Torque vs. current angle (rated I, rated If)"); axs[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_all(an, outdir):
    os.makedirs(outdir, exist_ok=True)
    files = {}
    from .dimensions import plot_cross_section
    files["dimensions"] = plot_cross_section(an.d, os.path.join(outdir, "00_cross_section_dimensions.png"))
    files["geometry"] = os.path.join(outdir, "01_geometry_mesh.png"); plot_geometry(an, files["geometry"])
    files["noload"] = os.path.join(outdir, "02_field_no_load.png")
    plot_field(an, an.res.solutions["no_load"], "No-load flux lines and |B| at rated voltage", files["noload"])
    files["rated"] = os.path.join(outdir, "03_field_rated_overexcited.png")
    plot_field(an, an.res.solutions["rated_over"],
               f"Rated over-excited (+{fmt_q(an.d.S_rated / 1e3)}) flux lines and |B|", files["rated"])
    files["under"] = os.path.join(outdir, "04_field_under_excited.png")
    plot_field(an, an.res.solutions["under_If0"],
               "Under-excited limit at If = 0 (armature-only excitation)", files["under"])
    files["gap"] = os.path.join(outdir, "05_airgap_flux.png"); plot_gap(an, files["gap"])
    files["occ"] = os.path.join(outdir, "06_occ_vcurve.png"); plot_occ_vcurve(an, files["occ"])
    files["wave"] = os.path.join(outdir, "07_emf_torque.png"); plot_waveform(an, files["wave"])
    if an.res.summary:
        files["summary"] = plot_design_sheet(an, os.path.join(outdir, "08_design_summary.png"))
    return files


# ---------------------------------------------------------------------------
# Design summary sheet: cross-section + key parameters with indicative checks
# ---------------------------------------------------------------------------
# (lo, hi) indicative ranges for an indirectly-cooled 2-pole machine of this size.
CHECKS = {
    "surface_speed": (None, 200.0), "B_gap1": (0.7, 1.05), "electric_loading_kA_m": (50.0, 130.0),
    "B_stator_tooth_max": (None, 2.05), "B_stator_yoke_max": (None, 1.65),
    "B_rotor_tooth_max": (None, 2.2), "B_rotor_body_max": (None, 2.0),
    "J_arm": (1.5, 4.5), "J_field": (None, 8.0), "thd": (None, 5.0),
}
# overrides for small (< 5 MVA) low/medium-voltage machines
CHECKS_SMALL = {"electric_loading_kA_m": (20.0, 50.0), "B_gap1": (0.6, 1.0), "J_arm": (2.5, 6.0)}


def _status(key, v, small=False):
    lim = CHECKS_SMALL.get(key, CHECKS.get(key)) if small else CHECKS.get(key)
    if lim is None or v is None or (isinstance(v, float) and math.isnan(v)):
        return "", "#444444"
    lo, hi = lim
    ok = (lo is None or v >= lo) and (hi is None or v <= hi)
    rng = (f"≤ {hi:g}" if lo is None else f"{lo:g} – {hi:g}")
    return ("✓ " if ok else "⚠ ") + rng, ("#1e8449" if ok else "#c0392b")


def _mass(t):
    return f"{t:.1f} t" if t >= 1 else f"{t * 1e3:.0f} kg"


def plot_design_sheet(an, path):
    from .dimensions import draw_section, geometry
    d, r = an.d, an.res
    sm, o, p, ra, L, w = r.summary, r.occ, r.params, r.rated, r.losses, r.waveform
    g = geometry(d)
    fig = plt.figure(figsize=(21, 11.5))
    fig.suptitle(f"Synchronous Condenser — Design Summary   |   {fmt_q(d.S_rated / 1e3)}  ·  "
                 f"{fmt_v(d.V_line)}  ·  {d.f:.0f} Hz  ·  {d.rpm:.0f} rpm  ·  {d.poles}-pole round rotor",
                 fontsize=15, fontweight="bold", x=0.5, y=0.985)
    ax = fig.add_axes([0.005, 0.06, 0.30, 0.86])
    draw_section(ax, d)
    R = g["R_out"]
    ax.text(-R * 0.98, -R * 1.10, f"Ø rotor {2 * d.R_rotor * 1e3:.0f}  ·  Ø bore {2 * d.R_bore * 1e3:.0f}  ·  "
            f"Ø outer {2 * R * 1e3:.0f} mm\nair gap {d.gap * 1e3:.0f} mm  ·  core length {d.L_stack * 1e3:.0f} mm",
            fontsize=10, color="#1f4e9c")
    ax.set_xlim(-R * 1.03, R * 1.03); ax.set_ylim(-R * 1.18, R * 1.03); ax.set_aspect("equal"); ax.axis("off")

    f0 = lambda v, n=0: f"{v:,.{n}f}"
    cols = [
        [("RATING", None),
         ("Rated reactive power", f"+{fmt_q(d.S_rated / 1e3)} / {fmt_q(ra['Q_under_If0_kVAr'])}", None),
         ("Rated voltage / current", f"{fmt_v(d.V_line)} / {f0(d.I_rated)} A", None),
         ("Frequency / speed", f"{d.f:.0f} Hz / {d.rpm:.0f} rpm", None),
         ("Base impedance", f"{d.Z_base:.3f} Ω", None),
         ("MAIN DIMENSIONS", None),
         ("Stator bore / outer Ø", f"{sm['D_bore'] * 1e3:.0f} / {sm['D_out'] * 1e3:.0f} mm", None),
         ("Rotor Ø / air gap", f"{sm['D_rotor'] * 1e3:.0f} / {d.gap * 1e3:.0f} mm", None),
         ("Active length", f"{d.L_stack * 1e3:.0f} mm", None),
         ("Rotor surface speed", f"{sm['surface_speed']:.0f} m/s", "surface_speed"),
         ("Output coeff. S/(D²·L·n)", f"{sm['esson_C_kVAmin_m3']:.2f} kVA·min/m³", None),
         ("WINDINGS", None),
         ("Stator slots / pitch", f"{d.Qs} / {d.coil_pitch}:{d.Qs // d.poles}", None),
         ("Series turns / paths", f"{d.series_turns:.0f} / {d.parallel_paths}", None),
         ("Winding factor kw1", f"{d.winding_factor():.4f}", None),
         ("Copper area per turn", f"{sm['A_bar_mm2']:.0f} mm²", None),
         ("Rotor slots / turns per slot", f"{d.wound_rotor_slots} / {d.Nf_per_slot}", None),
         ("Field turns (total)", f"{d.wound_rotor_slots * d.Nf_per_slot // 2}", None)],
        [("ELECTROMAGNETIC LOADING", None),
         ("Air-gap B₁ (no-load, rated V)", f"{sm['B_gap1']:.3f} T", "B_gap1"),
         ("Flux per pole", f"{sm['flux_per_pole_Wb']:.3g} Wb", None),
         ("Electric loading A", f"{sm['electric_loading_kA_m']:.0f} kA/m", "electric_loading_kA_m"),
         ("FLUX DENSITY (no-load, 98th pct)", None),
         ("Stator teeth", f"{sm['B_stator_tooth_max']:.2f} T", "B_stator_tooth_max"),
         ("Stator yoke", f"{sm['B_stator_yoke_max']:.2f} T", "B_stator_yoke_max"),
         ("Rotor teeth", f"{sm['B_rotor_tooth_max']:.2f} T", "B_rotor_tooth_max"),
         ("Rotor body (pole)", f"{sm['B_rotor_body_max']:.2f} T", "B_rotor_body_max"),
         ("CURRENT DENSITY (rated)", None),
         ("Armature", f"{sm['J_arm']:.2f} A/mm²", "J_arm"),
         ("Field (rated over-excited)", f"{sm['J_field']:.2f} A/mm²", "J_field"),
         ("EMF QUALITY", None),
         ("EMF THD line-to-line", f"{100 * w.get('thd_line', float('nan')):.2f} %", "thd"),
         ("EMF THD phase (incl. 3rd)", f"{100 * w.get('thd', float('nan')):.2f} %", None),
         ("Saturation factor (OCC)", f"{o['saturation_factor']:.3f}", None)],
        [("REACTANCES / EXCITATION", None),
         ("Xd / Xq (unsaturated)", f"{p['Xd_pu']:.3f} / {p['Xq_pu']:.3f} pu", None),
         ("Short-circuit ratio SCR", f"{p['SCR']:.3f}", None),
         ("If no-load (rated V)", f"{f0(o['If_noload'], 1)} A", None),
         ("If short-circuit (rated I)", f"{f0(p['If_sc'], 1)} A", None),
         ("If rated over-excited", f"{f0(ra['If_rated_over'], 1)} A  ({ra['If_rated_over'] / o['If_noload']:.2f}×)", None),
         ("Field voltage / resistance", f"{L['V_field_rated']:.0f} V / {L['R_field_ohm']:.3g} Ω", None),
         ("Rated field ampere-turns / pole", f"{sm['ampere_turns_pole'] / 1e3:.1f} kAt", None),
         ("Max static torque", f"{r.torque_angle.get('T_max_pu', float('nan')):.2f} pu", None),
         ("LOSSES (estimate)", None),
         ("Stator core", f"{L['P_iron_kW']:.3g} kW", None),
         ("Armature copper", f"{L['P_arm_cu_kW']:.3g} kW", None),
         ("Field copper", f"{L['P_field_cu_kW']:.3g} kW", None),
         ("Total EM loss", f"{L['P_total_kW']:.3g} kW  ({L['loss_pct']:.2f} %)", None),
         ("MASS / INERTIA (active part)", None),
         ("Stator core / rotor iron", f"{_mass(sm['mass_stator_core_t'])} / {_mass(sm['mass_rotor_iron_active_t'])}", None),
         ("Armature / field copper", f"{_mass(sm['mass_arm_cu_t'])} / {_mass(sm['mass_field_cu_t'])}", None),
         ("Rotor inertia J / H", f"{sm['J_rotor_kgm2']:,.4g} kg·m² / {sm['H_inertia_s']:.2f} s", None)],
    ]
    values = {"thd": 100 * w.get("thd_line", float("nan"))}
    for ci, col in enumerate(cols):
        axc = fig.add_axes([0.315 + ci * 0.228, 0.06, 0.218, 0.86]); axc.axis("off")
        y = 0.985
        for item in col:
            if item[1] is None:
                y -= 0.012
                axc.add_patch(plt.Rectangle((0, y - 0.014), 1, 0.036, fc="#1f4e9c", ec="none",
                                            transform=axc.transAxes))
                axc.text(0.02, y + 0.004, item[0], fontsize=10, color="w", fontweight="bold",
                         va="center", transform=axc.transAxes)
                y -= 0.046
                continue
            name, val, key = item
            axc.text(0.02, y, name, fontsize=9.5, va="center", transform=axc.transAxes)
            axc.text(0.98, y, val, fontsize=9.5, ha="right", va="center", family="monospace",
                     transform=axc.transAxes)
            if key:
                v = values.get(key, sm.get(key))
                txt, colr = _status(key, v, small=d.S_rated < 5e6)
                axc.text(0.98, y - 0.021, txt, fontsize=7.5, ha="right", va="center", color=colr,
                         transform=axc.transAxes)
                y -= 0.012
            axc.plot([0.02, 0.98], [y - 0.017, y - 0.017], color="#e2e2e2", lw=0.6,
                     transform=axc.transAxes)
            y -= 0.040
    scope = ("small LV/MV machines" if d.S_rated < 5e6 else "large indirectly-cooled machines")
    fig.text(0.315, 0.025, f"✓/⚠ ranges are indicative guidance for {scope}, "
             "not a code requirement. Flux densities: 98th percentile of element |B| at no-load, rated voltage. "
             "Iron masses over the active length; copper masses include end windings.",
             fontsize=8, style="italic", color="#555555")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
