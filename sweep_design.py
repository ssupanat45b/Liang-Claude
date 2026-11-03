#!/usr/bin/env python3
"""Parameter study (design sweep) for the synchronous condenser FEM model.

Each case is an independent FEM study, so cases run in parallel processes.

Examples
    python sweep_design.py                                   # factory300: air gap 2.5/3/3.5/4 mm
    python sweep_design.py --design sc100                    # 100 MVAr: air gap 50/60/70 mm
    python sweep_design.py --param gap --values 0.045 0.05 0.055 0.06 0.065 0.07 --jobs 6
    python sweep_design.py --param L_stack --values 2.6 2.9 3.2
    python sweep_design.py --param Nf_per_slot --values 8 10 12 --mesh fine

For --param gap the stator bore is kept fixed (same stator punching) and the
rotor diameter changes; use --keep-rotor to keep the rotor fixed instead.
Any numeric field of SCDesign (sc_fem2d/machine.py) can be swept.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time


def run_case(args):
    param, value, mesh, keep_rotor, design = args
    import numpy as np
    from sc_fem2d.analysis import SCAnalysis
    from sc_fem2d.machine import apply_mesh_preset, design_preset

    d = apply_mesh_preset(design_preset(design), mesh)
    base_bore = d.R_bore
    cur = getattr(d, param)
    setattr(d, param, type(cur)(value))
    if param == "gap" and not keep_rotor:
        d.R_rotor = base_bore - d.gap
    t0 = time.time()
    an = SCAnalysis(d, verbose=False)
    an.find_axes()
    an.occ_auto(quick=True)
    an.reactances()
    an.rated_points()
    sol0 = an.solve(i_f=an.res.occ["If_noload"])
    an.res.solutions["no_load"] = sol0
    g = an.gap_field()
    L = an.losses()
    summ = an.design_summary()
    r, o, p = an.res.rated, an.res.occ, an.res.params
    return dict(
        param=param, value=value, R_rotor_mm=d.R_rotor * 1e3, gap_mm=d.gap * 1e3,
        If0_A=o["If_noload"], sat_factor=o["saturation_factor"], B_gap1_T=g["B1"],
        Xd_pu=p["Xd_pu"], Xq_pu=p["Xq_pu"], SCR=p["SCR"], If_sc_A=p["If_sc"],
        If_rated_A=r["If_rated_over"], Q_under_If0_kVAr=r["Q_under_If0_kVAr"],
        Q_under_pct=-r["Q_under_If0_kVAr"] / (d.S_rated / 1e3) * 100,
        P_field_kW=L["P_field_cu_kW"], P_total_kW=L["P_total_kW"], J_arm_A_mm2=L["J_arm_A_mm2"], U_field_V=L["V_field_rated"], J_field_A_mm2=L["J_field_A_mm2"],
        B_tooth_max_T=summ["B_stator_tooth_max"], B_yoke_max_T=summ["B_stator_yoke_max"],
        B_rotor_body_max_T=summ["B_rotor_body_max"],
        runtime_s=time.time() - t0)


COLUMNS = [("value", "{:g}"), ("R_rotor_mm", "{:.1f}"), ("gap_mm", "{:.1f}"), ("If0_A", "{:.1f}"),
           ("If_sc_A", "{:.1f}"), ("If_rated_A", "{:.1f}"), ("Xd_pu", "{:.3f}"), ("Xq_pu", "{:.3f}"),
           ("SCR", "{:.3f}"), ("Q_under_If0_kVAr", "{:.1f}"), ("Q_under_pct", "{:.0f}"),
           ("B_gap1_T", "{:.3f}"), ("sat_factor", "{:.3f}"), ("P_field_kW", "{:.3g}"), ("P_total_kW", "{:.3g}"), ("J_arm_A_mm2", "{:.2f}"),
           ("U_field_V", "{:.0f}"), ("J_field_A_mm2", "{:.2f}"), ("B_tooth_max_T", "{:.2f}"),
           ("B_rotor_body_max_T", "{:.2f}")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="factory300", choices=["factory300", "sc100"])
    ap.add_argument("--param", default="gap")
    ap.add_argument("--values", type=float, nargs="+", default=None,
                    help="default for gap: factory300 2.5-4 mm, sc100 50/60/70 mm")
    ap.add_argument("--mesh", default="normal", choices=["coarse", "normal", "fine", "ultra"])
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel processes (0 = auto: min(cases, CPU cores // 2))")
    ap.add_argument("--keep-rotor", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.values is None:
        if a.param != "gap":
            ap.error("--values is required for this parameter")
        a.values = [0.0025, 0.003, 0.0035, 0.004] if a.design == "factory300" else [0.050, 0.060, 0.070]
    a.out = a.out or ("results_sweep" if a.design == "sc100" else f"results_sweep_{a.design}")

    # one BLAS/solver thread per worker so parallel cases don't oversubscribe the CPU
    jobs = a.jobs or max(1, min(len(a.values), (os.cpu_count() or 2) // 2))
    if jobs > 1:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")
    from concurrent.futures import ProcessPoolExecutor
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out, exist_ok=True)
    cases = [(a.param, v, a.mesh, a.keep_rotor, a.design) for v in a.values]
    print(f"Design {a.design}: sweeping {a.param} = {a.values}  | mesh '{a.mesh}' | {jobs} parallel job(s)")
    t0 = time.time()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            rows = list(ex.map(run_case, cases))
    else:
        rows = [run_case(c) for c in cases]
    rows.sort(key=lambda r: r["value"])

    with open(os.path.join(a.out, f"sweep_{a.param}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    head = "| " + " | ".join(c for c, _ in COLUMNS) + " |"
    sep = "|" + "---|" * len(COLUMNS)
    body = ["| " + " | ".join(f.format(r[c]) for c, f in COLUMNS) + " |" for r in rows]
    table = "\n".join([head, sep] + body)
    print("\n" + table)

    keys = [("Xd_pu", "Xd [pu]"), ("SCR", "SCR [-]"), ("If_rated_A", "If rated (+Q) [A]"),
            ("If0_A", "If0 no-load [A]"), ("Q_under_If0_kVAr", "Q under-excited @If=0 [kVAr]"),
            ("P_field_kW", "Field Cu loss [kW]")]
    x = [r["value"] * (1e3 if a.param == "gap" else 1) for r in rows]
    xl = "air gap [mm]" if a.param == "gap" else a.param
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, lab) in zip(axs.ravel(), keys):
        ax.plot(x, [r[k] for r in rows], "o-", lw=2)
        for xi, r in zip(x, rows):
            ax.annotate(f"{r[k]:.3g}", (xi, r[k]), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8)
        ax.set_xlabel(xl); ax.set_title(lab); ax.grid(alpha=.3)
    fig.suptitle(f"Design sweep: {a.param}", fontsize=13)
    fig.tight_layout()
    png = os.path.join(a.out, f"sweep_{a.param}.png")
    fig.savefig(png, dpi=150); plt.close(fig)

    with open(os.path.join(a.out, f"sweep_{a.param}.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Design sweep ({a.design}): {a.param}\n\nMesh preset: {a.mesh}. "
                 + ("Stator bore kept fixed; rotor diameter = bore − 2·gap.\n\n"
                    if a.param == "gap" and not a.keep_rotor else "\n\n")
                 + table + f"\n\n![sweep](sweep_{a.param}.png)\n")
    print(f"\nDone in {time.time() - t0:.0f} s -> {a.out}/sweep_{a.param}.(csv|md|png)")


if __name__ == "__main__":
    sys.exit(main())
