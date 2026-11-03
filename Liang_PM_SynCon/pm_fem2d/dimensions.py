"""Dimensioned cross-section drawing of the stator and the SPM rotor.

Drawn directly from the design data (no FEM needed):

    python draw_pm_cross_section.py   -> results_pm300/00_cross_section_dimensions.png
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

from .machine import PMDesign, fmt_q, fmt_v, winding_layout

C_STATOR = "#b9c2cf"
C_ROTOR = "#8c96a6"
C_ARM = "#c0392b"
C_MAG_N = "#c2185b"
C_MAG_S = "#1565c0"
C_SLEEVE = "#f3e5ab"
C_WEDGE = "#e3e3e3"
C_DIM = "#1f4e9c"


def geometry(d: PMDesign) -> dict:
    """All main dimensions in metres / degrees."""
    n = d.n_theta
    dth = 360.0 / n
    g = dict(
        R_rotor=d.R_rotor, R_bore=d.R_bore, R_out=d.R_stator_out,
        R_core=d.R_core, R_sleeve=d.R_rotor + d.t_sleeve,
        s1=d.R_bore + d.s_wedge, s2=d.R_bore + d.s_wedge + d.s_layer,
        s3=d.R_bore + d.s_wedge + 2 * d.s_layer,
        s_pitch_deg=360.0 / d.Qs, s_slot_deg=d.slot_divs * dth,
        s_slot_start=((d.div_per_slot - d.slot_divs) // 2) * dth,
        pole_deg=360.0 / d.poles, mag_deg=d.magnet_divs * dth,
    )
    g["yoke"] = g["R_out"] - g["s3"]
    return g


def draw_section(ax, d: PMDesign, rot: float = 0.0, labels: bool = False):
    """Draw stator, rotor, slots and windings; ``rot`` rotates the drawing [deg]."""
    g = geometry(d)
    ax.add_patch(Wedge((0, 0), g["R_out"], 0, 360, width=g["R_out"] - g["R_bore"],
                       fc=C_STATOR, ec="k", lw=0.6))
    ax.add_patch(Circle((0, 0), g["R_sleeve"], fc=C_SLEEVE, ec="k", lw=0.4))      # retaining sleeve
    ax.add_patch(Circle((0, 0), g["R_rotor"], fc="white", ec="k", lw=0.4))        # inter-pole filler
    ax.add_patch(Circle((0, 0), g["R_core"], fc=C_ROTOR, ec="k", lw=0.6))
    lay = winding_layout(d)
    for k in range(d.Qs):
        a0 = k * g["s_pitch_deg"] + g["s_slot_start"] + rot
        a1 = a0 + g["s_slot_deg"]
        ax.add_patch(Wedge((0, 0), g["s1"], a0, a1, width=d.s_wedge, fc=C_WEDGE, ec="k", lw=0.3))
        ax.add_patch(Wedge((0, 0), g["s2"], a0, a1, width=d.s_layer, fc=C_ARM, ec="k", lw=0.3))
        ax.add_patch(Wedge((0, 0), g["s3"], a0, a1, width=d.s_layer, fc=C_ARM, ec="k", lw=0.3,
                           alpha=0.8))
        if labels:
            am = math.radians(0.5 * (a0 + a1))
            for layer, r in ((0, 0.5 * (g["s1"] + g["s2"])), (1, 0.5 * (g["s2"] + g["s3"]))):
                ph, sg = lay[layer][k]
                ax.text(r * math.cos(am), r * math.sin(am), "ABC"[ph] + ("+" if sg > 0 else "−"),
                        fontsize=4.5, color="w", ha="center", va="center",
                        rotation=math.degrees(am) - 90)
    for kp in range(d.poles):
        c = kp * g["pole_deg"] + rot
        ax.add_patch(Wedge((0, 0), g["R_rotor"], c - g["mag_deg"] / 2, c + g["mag_deg"] / 2,
                           width=d.h_m, fc=C_MAG_N if kp % 2 == 0 else C_MAG_S, ec="k", lw=0.4))


def _dim(ax, p1, p2, text, off=(0, 0), fs=8, rot=0, color=C_DIM):
    ax.annotate("", p1, p2, arrowprops=dict(arrowstyle="<|-|>", color=color, lw=0.8,
                                            shrinkA=0, shrinkB=0, mutation_scale=7))
    xm, ym = 0.5 * (p1[0] + p2[0]) + off[0], 0.5 * (p1[1] + p2[1]) + off[1]
    ax.text(xm, ym, text, fontsize=fs, color=color, ha="center", va="center", rotation=rot,
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def plot_cross_section(d: PMDesign, path: str):
    g = geometry(d)
    mm = lambda v: f"{v * 1e3:.0f}"
    fig = plt.figure(figsize=(20, 9.5))
    axA = fig.add_axes([0.01, 0.04, 0.40, 0.88])
    axB = fig.add_axes([0.42, 0.04, 0.30, 0.88])
    axC = fig.add_axes([0.74, 0.04, 0.25, 0.88]); axC.axis("off")

    # ---------------- A: full section with diameters ------------------
    draw_section(axA, d, labels=True)
    R = g["R_out"]
    for r, name, y in ((g["R_rotor"], "Ø rotor", -R * 1.09), (g["R_bore"], "Ø stator bore", -R * 1.18),
                       (g["s3"], "Ø slot bottom", -R * 1.27), (R, "Ø stator outer", -R * 1.36)):
        for sx in (-1, 1):
            axA.plot([sx * r, sx * r], [0, y - 0.018 * R], color=C_DIM, lw=0.4, ls=(0, (4, 3)))
        _dim(axA, (-r, y), (r, y), f"{name} {mm(2 * r)} mm")
    # pole (d) axes with N/S polarity
    Rr = g["R_rotor"]
    for kp in range(d.poles):
        a = math.radians(kp * 360.0 / d.poles)
        axA.plot([0, 1.05 * Rr * math.cos(a)], [0, 1.05 * Rr * math.sin(a)], "k-.", lw=0.6)
        rm = Rr - d.h_m / 2
        axA.text(rm * math.cos(a), rm * math.sin(a), "NS"[kp % 2], fontsize=11,
                 ha="center", va="center", fontweight="bold", color="w")
    axA.text(0.30 * Rr, 0.05 * Rr, "d-axis", fontsize=7)
    axA.set_xlim(-R * 1.08, R * 1.08); axA.set_ylim(-R * 1.42, R * 1.08)
    axA.set_aspect("equal"); axA.axis("off")
    axA.set_title(f"Cross-section — {fmt_q(d.S_rated / 1e3)}, {fmt_v(d.V_line)}, "
                  f"{d.f:.0f} Hz, {d.poles}-pole {d.rpm:.0f} rpm, {d.Qs} stator slots / "
                  f"SPM rotor {d.poles} NdFeB magnets", fontsize=11)

    # ---------------- B: radial detail -----------------------------------
    k = d.Qs // 4                               # stator slot near the top
    s_centre = k * g["s_pitch_deg"] + g["s_slot_start"] + 0.5 * g["s_slot_deg"]
    rot = 90.0 - s_centre                       # put that slot vertical
    draw_section(axB, d, rot=rot)
    r_lo = g["R_core"] - 0.06 * (g["R_out"] - g["R_core"])
    sc = (g["R_out"] - r_lo) / 0.77                     # layout scale
    x0 = -0.20 * sc
    radii = [g["R_core"], g["R_rotor"], g["R_sleeve"], g["R_bore"], g["s1"], g["s2"], g["s3"], g["R_out"]]
    names = ["magnet", "sleeve", "mech. gap", "stator wedge", "top layer", "bottom layer", "stator yoke"]
    for r in radii:
        axB.plot([x0 - 0.01 * sc, 0.13 * sc], [r, r], color=C_DIM, lw=0.35, ls=(0, (4, 3)))
    mm1 = lambda v: f"{v * 1e3:.1f}"
    for i, name in enumerate(names):
        r0, r1 = radii[i], radii[i + 1]
        side = -0.075 * sc if i % 2 else 0.0
        _dim(axB, (x0 + side, r0), (x0 + side, r1), f"{name}\n{mm1(r1 - r0)} mm",
             off=((-0.045 if i % 2 else 0.045) * sc, 0), fs=7)
    _dim(axB, (0.16 * sc, g["R_rotor"]), (0.16 * sc, g["R_bore"]),
         f"magnetic gap\n{mm1(d.gap)} mm", off=(0.05 * sc, 0), fs=7)
    _dim(axB, (0.16 * sc, g["R_bore"]), (0.16 * sc, g["s3"]),
         f"stator slot depth\n{mm(g['s3'] - g['R_bore'])} mm", off=(0.05 * sc, 0), fs=7)
    # widths (chords) of the vertical stator slot at the bore and at the bottom
    for r, lab, dy in ((g["s1"], "slot width at bore", 0.004), (g["s3"], "slot width at bottom", 0.012)):
        half = r * math.sin(math.radians(0.5 * g["s_slot_deg"]))      # chord at radius r
        _dim(axB, (-half, r + dy * sc), (half, r + dy * sc), f"{lab} {mm(2 * half)} mm",
             off=(0, 0.018 * sc), fs=7)
    tooth_bore = 2 * g["R_bore"] * math.sin(math.radians(0.5 * (g["s_pitch_deg"] - g["s_slot_deg"])))
    axB.set_xlim(-0.36 * sc, 0.30 * sc)
    axB.set_ylim(r_lo, g["R_out"] + 0.04 * sc)
    axB.set_aspect("equal"); axB.axis("off")
    axB.set_title("Radial build-up (detail at a stator slot)", fontsize=11)

    # ---------------- C: dimension table --------------------------------
    st_bot = 2 * g["s3"] * math.sin(math.radians(0.5 * (g["s_pitch_deg"] - g["s_slot_deg"])))
    rows = [
        ("MAIN DIMENSIONS", ""),
        ("Stator outer diameter", f"{mm(2 * g['R_out'])} mm"),
        ("Stator bore diameter", f"{mm(2 * g['R_bore'])} mm"),
        ("Rotor diameter", f"{mm(2 * g['R_rotor'])} mm"),
        ("Gap: mechanical + sleeve", f"{d.g_mech * 1e3:.1f} + {d.t_sleeve * 1e3:.1f} mm"),
        ("Active core length", f"{d.L_stack * 1e3:.0f} mm"),
        ("Rotor surface speed", f"{d.omega / (d.poles / 2) * d.R_rotor:.0f} m/s"),
        ("STATOR", ""),
        ("Slots / slot pitch", f"{d.Qs} / {g['s_pitch_deg']:.2f}°"),
        ("Slot depth (wedge + 2 layers)", f"{mm(g['s3'] - g['R_bore'])} mm"),
        ("Slot width bore / bottom", f"{mm(2 * g['s1'] * math.sin(math.radians(g['s_slot_deg'] / 2)))}"
                                     f" / {mm(2 * g['s3'] * math.sin(math.radians(g['s_slot_deg'] / 2)))} mm"),
        ("Tooth width bore / root", f"{mm(tooth_bore)} / {mm(st_bot)} mm"),
        ("Yoke depth", f"{mm(g['yoke'])} mm"),
        ("Winding", f"2-layer, pitch {d.coil_pitch}/{d.Qs // d.poles}, {d.parallel_paths} path(s)"),
        ("Series turns / phase", f"{d.series_turns:.0f}"),
        ("ROTOR", ""),
        ("Rotor core Ø (magnet seat)", f"{mm(2 * g['R_core'])} mm"),
        ("Magnet thickness", f"{d.h_m * 1e3:.1f} mm"),
        ("Magnet arc / pole pitch", f"{g['mag_deg']:.1f}° / {g['pole_deg']:.0f}° ({d.alpha_p_actual:.3f})"),
        ("Magnet arc length (at OD)", f"{mm(math.radians(g['mag_deg']) * d.R_rotor)} mm"),
        ("Magnet grade", f"NdFeB Br {d.Br20:.2f} T @20°C, μr {d.mu_rm}"),
        ("Magnetisation", "radial, alternating N/S"),
    ]
    y = 0.97
    for name, val in rows:
        if not val:
            axC.text(0.0, y, name, fontsize=10, fontweight="bold", color=C_DIM, transform=axC.transAxes)
            y -= 0.040
            continue
        axC.text(0.02, y, name, fontsize=9, transform=axC.transAxes)
        axC.text(0.98, y, val, fontsize=9, ha="right", transform=axC.transAxes, family="monospace")
        axC.plot([0.02, 0.98], [y - 0.012, y - 0.012], color="#dddddd", lw=0.5, transform=axC.transAxes)
        y -= 0.033
    # legend
    y -= 0.02
    for c, lab in ((C_STATOR, "stator core (M330-50A)"), (C_ROTOR, "rotor steel"),
                   (C_ARM, "armature winding (A/B/C)"), (C_MAG_N, "magnet N (outward)"),
                   (C_MAG_S, "magnet S (inward)"), (C_SLEEVE, "retaining sleeve"), (C_WEDGE, "wedges")):
        axC.add_patch(plt.Rectangle((0.02, y - 0.008), 0.05, 0.022, fc=c, ec="k", lw=0.4,
                                    transform=axC.transAxes))
        axC.text(0.10, y, lab, fontsize=9, va="center", transform=axC.transAxes)
        y -= 0.029
    axC.text(0.02, y - 0.01, "Note: slots are annular sectors (radial sides), as in the FEM mesh.",
             fontsize=7.5, style="italic", transform=axC.transAxes)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
