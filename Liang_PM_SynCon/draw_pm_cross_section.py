#!/usr/bin/env python3
"""Draw the dimensioned stator / SPM-rotor cross-section only (no FEM, ~2 s).

    python draw_pm_cross_section.py            -> results_pm300/00_cross_section_dimensions.png
"""
import argparse
import os

from pm_fem2d.dimensions import plot_cross_section
from pm_fem2d.machine import DESIGN_PRESETS, design_preset

ap = argparse.ArgumentParser()
ap.add_argument("--design", default="pm300", choices=DESIGN_PRESETS)
ap.add_argument("--out", default=None)
args = ap.parse_args()
args.out = args.out or f"results_{args.design}"
os.makedirs(args.out, exist_ok=True)
print("saved", plot_cross_section(design_preset(args.design),
                                  os.path.join(args.out, "00_cross_section_dimensions.png")))
