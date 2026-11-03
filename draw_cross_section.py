#!/usr/bin/env python3
"""Draw the dimensioned stator/rotor cross-section only (no FEM, takes ~2 s).

Usage:
    python draw_cross_section.py                 # -> results_factory300/00_cross_section_dimensions.png
    python draw_cross_section.py --design sc100  # -> results/...
    python draw_cross_section.py --out my_dir
"""
import argparse
import os

from sc_fem2d.dimensions import plot_cross_section
from sc_fem2d.machine import DESIGN_PRESETS, design_preset

ap = argparse.ArgumentParser()
ap.add_argument("--design", default="factory300", choices=DESIGN_PRESETS)
ap.add_argument("--out", default=None)
args = ap.parse_args()
args.out = args.out or ("results" if args.design == "sc100" else f"results_{args.design}")
os.makedirs(args.out, exist_ok=True)
path = plot_cross_section(design_preset(args.design), os.path.join(args.out, "00_cross_section_dimensions.png"))
print("saved", path)
