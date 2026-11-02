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
            ax.set_xlim(0.25, 0.85); ax.set_ylim(-0.05, 0.45)
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
    axs[0].plot(np.degrees(g["theta"]), g["B1"] * np.cos(g["theta"] - an.res.axes["d"]), "--",
                label=f"fundamental {g['B1']:.3f} T")
    axs[0].set_xlabel("stator angle [deg]"); axs[0].set_ylabel("B [T]"); axs[0].grid(alpha=.3)
    axs[0].legend(); axs[0].set_title("No-load air-gap flux density (rated voltage)")
    h = g["harm"]
    axs[1].bar(np.arange(1, len(h)), h[1:] / h[1] * 100)
    axs[1].set_yscale("log"); axs[1].set_xlabel("space harmonic order")
    axs[1].set_ylabel("% of fundamental"); axs[1].set_title("Br harmonic spectrum")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_occ_vcurve(an, path):
    d, o, v = an.d, an.res.occ, an.res.vcurve
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    If = np.r_[0, o["If"]]
    axs[0].plot(If, np.r_[0, o["E_line"]] / 1e3, "o-", label="OCC (FEM, saturated)")
    axs[0].plot(If, o["airgap_slope"] * If / 1e3, "--", label="air-gap line")
    axs[0].axhline(d.V_line / 1e3, color="gray", lw=0.8)
    axs[0].axvline(o["If_noload"], color="gray", lw=0.8)
    axs[0].set_xlabel("field current If [A]"); axs[0].set_ylabel("E line [kV rms]")
    axs[0].set_title(f"Open-circuit characteristic  (If0 = {o['If_noload']:.0f} A)")
    axs[0].grid(alpha=.3); axs[0].legend()
    axs[1].plot(v["If"], v["Q_MVAr"], "o-")
    axs[1].axhline(0, color="gray", lw=0.8)
    axs[1].set_xlabel("field current If [A]"); axs[1].set_ylabel("Q [MVAr]  (+ over-excited)")
    axs[1].set_title("Reactive capability / V-curve at rated voltage")
    axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_waveform(an, path):
    w, ta = an.res.waveform, an.res.torque_angle
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    axs[0].plot(np.degrees(w["angle"]), w["e"] / 1e3)
    axs[0].set_xlabel("rotor angle [deg]"); axs[0].set_ylabel("phase EMF [kV]")
    axs[0].set_title(f"No-load phase-A EMF  (THD {100 * w['thd']:.2f} %)"); axs[0].grid(alpha=.3)
    h = w["e_harm"]
    axs[1].bar(np.arange(1, len(h)), h[1:] / h[1] * 100)
    axs[1].set_yscale("log"); axs[1].set_xlabel("time harmonic order")
    axs[1].set_ylabel("% of fundamental"); axs[1].set_title("EMF harmonics")
    axs[2].plot(np.degrees(ta["gamma"]), ta["T"] / 1e3, "o-")
    axs[2].set_xlabel("current angle from +d axis [deg]"); axs[2].set_ylabel("torque [kNm]")
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
               f"Rated over-excited (+{an.d.S_rated / 1e6:.0f} MVAr) flux lines and |B|", files["rated"])
    files["under"] = os.path.join(outdir, "04_field_under_excited.png")
    plot_field(an, an.res.solutions["under_If0"],
               "Under-excited limit at If = 0 (armature-only excitation)", files["under"])
    files["gap"] = os.path.join(outdir, "05_airgap_flux.png"); plot_gap(an, files["gap"])
    files["occ"] = os.path.join(outdir, "06_occ_vcurve.png"); plot_occ_vcurve(an, files["occ"])
    files["wave"] = os.path.join(outdir, "07_emf_torque.png"); plot_waveform(an, files["wave"])
    return files
