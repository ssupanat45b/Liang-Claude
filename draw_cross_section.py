#!/usr/bin/env python3
"""Draw the dimensioned stator/rotor cross-section only (no FEM, takes ~2 s).

Usage:
    python draw_cross_section.py                 # -> results/00_cross_section_dimensions.png
    python draw_cross_section.py --out my_dir
"""
import argparse
import os

from sc_fem2d.dimensions import plot_cross_section
from sc_fem2d.machine import SCDesign

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
path = plot_cross_section(SCDesign(), os.path.join(args.out, "00_cross_section_dimensions.png"))
print("saved", path)
