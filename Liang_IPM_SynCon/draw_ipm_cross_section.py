#!/usr/bin/env python3
"""Draw the dimensioned stator / IPM-rotor cross-section only (no FEM, ~2 s).

    python draw_ipm_cross_section.py     -> results_ipm300/00_cross_section_dimensions.png
"""
import argparse
import os

from ipm_fem2d.dimensions import plot_cross_section
from ipm_fem2d.machine import DESIGN_PRESETS, design_preset

ap = argparse.ArgumentParser()
ap.add_argument("--design", default="ipm300", choices=DESIGN_PRESETS)
ap.add_argument("--hm", type=float, help="magnet thickness [m]")
ap.add_argument("--out", default=None)
a = ap.parse_args()
a.out = a.out or f"results_{a.design}"
os.makedirs(a.out, exist_ok=True)
d = design_preset(a.design)
if a.hm:
    d.h_m = a.hm
print("saved", plot_cross_section(d, os.path.join(a.out, "00_cross_section_dimensions.png")))
