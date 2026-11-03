#!/usr/bin/env python3
"""Parameter study for the PM (SPM) synchronous condenser.

    python sweep_pm.py                                    # magnet thickness 16-28 mm
    python sweep_pm.py --param alpha_p --values 0.70 0.75 0.80 0.85 0.90
    python sweep_pm.py --param gap --values 0.003 0.0035 0.004 --jobs 6
    python sweep_pm.py --solver gpu --jobs 2              # GPU: few jobs share one card

Any numeric field of PMDesign can be swept (h_m, alpha_p, gap, L_stack, Br20, T_mag, ...).
For --param gap the stator bore is kept fixed and the rotor (magnet OD) shrinks.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time


def run_case(args):
    param, value, mesh, solver_name = args
    from pm_fem2d import solver
    from pm_fem2d.analysis import PMAnalysis
    from pm_fem2d.machine import apply_mesh_preset, design_preset
    solver.set_backend(solver_name)
    d = apply_mesh_preset(design_preset(), mesh)
    bore = d.R_bore
    setattr(d, param, type(getattr(d, param))(value))
    if param == "gap":
        d.R_rotor = bore - d.gap
    t0 = time.time()
    an = PMAnalysis(d, verbose=False)
    an.find_axes(); an.no_load_emf(); an.reactances(); an.rated_point()
    an.q_v_characteristic((0.95, 1.0, 1.05)); an.faults_and_demag(); an.gap_field()
    an.losses(); sm = an.design_summary()
    r, f, e = an.res.rated, an.res.fault, an.res.emf
    return dict(param=param, value=value, h_m_mm=d.h_m * 1e3, alpha_p=d.alpha_p_actual,
                gap_mm=d.gap * 1e3, E0_pu=e["E0_pu_hot"], E0_cold_pu=e["E0_pu_cold"],
                Xd_pu=an.res.params["Xd_pu"], Q_kVAr=r["Q_kVAr"], Q_cold_kVAr=r["Q_cold_kVAr"],
                I_pu=r["I_pu"], I_cold_pu=r["I_cold_pu"], dQ_per_pct=an.res.qv["dQdV_kVAr_per_pu"] / 100,
                I_sc_pu=f["I_sc_pu"], Bd_min_rated=f["Bd_min_rated"], Bd_min_sc=f["Bd_min_sc"],
                magnet_kg=sm["mass_magnet_t"] * 1e3, B_tooth_T=sm["B_stator_tooth_max"],
                P_loss_kW=an.res.losses["P_total_kW"], backend=solver.backend_name(),
                runtime_s=time.time() - t0)


COLS = [("value", "{:g}"), ("h_m_mm", "{:.1f}"), ("alpha_p", "{:.3f}"), ("gap_mm", "{:.1f}"),
        ("E0_pu", "{:.3f}"), ("Xd_pu", "{:.3f}"), ("Q_kVAr", "{:.0f}"), ("Q_cold_kVAr", "{:.0f}"),
        ("I_cold_pu", "{:.2f}"), ("dQ_per_pct", "{:.1f}"), ("I_sc_pu", "{:.2f}"),
        ("Bd_min_sc", "{:.2f}"), ("magnet_kg", "{:.1f}"), ("B_tooth_T", "{:.2f}"), ("P_loss_kW", "{:.2f}")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param", default="h_m")
    ap.add_argument("--values", type=float, nargs="+", default=None)
    ap.add_argument("--mesh", default="normal", choices=["coarse", "normal", "fine", "ultra"])
    ap.add_argument("--solver", default="auto", choices=["auto", "gpu", "pardiso", "superlu"])
    ap.add_argument("--jobs", type=int, default=0, help="0 = auto (GPU: 2, CPU: cores/2)")
    ap.add_argument("--out", default="results_sweep_pm300")
    a = ap.parse_args()
    if a.values is None:
        if a.param != "h_m":
            ap.error("--values is required for this parameter")
        a.values = [0.016, 0.019, 0.022, 0.025, 0.028]
    from pm_fem2d import solver
    use_gpu = solver.set_backend(a.solver).startswith("GPU")
    jobs = a.jobs or (min(2, len(a.values)) if use_gpu else max(1, min(len(a.values), (os.cpu_count() or 2) // 2)))
    if jobs > 1 and not use_gpu:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(a.out, exist_ok=True)
    cases = [(a.param, v, a.mesh, a.solver) for v in a.values]
    print(f"PM sweep {a.param} = {a.values} | mesh {a.mesh} | solver {solver.backend_name()} | {jobs} job(s)")
    t0 = time.time()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            rows = list(ex.map(run_case, cases))
    else:
        rows = [run_case(c) for c in cases]
    rows.sort(key=lambda r: r["value"])
    with open(os.path.join(a.out, f"sweep_{a.param}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    table = "\n".join(["| " + " | ".join(c for c, _ in COLS) + " |", "|" + "---|" * len(COLS)]
                      + ["| " + " | ".join(f.format(r[c]) for c, f in COLS) + " |" for r in rows])
    print("\n" + table)
    scale = 1e3 if a.param in ("h_m", "gap", "L_stack") else 1
    x = [r["value"] * scale for r in rows]
    xl = f"{a.param} [mm]" if scale == 1e3 else a.param
    keys = [("Q_kVAr", "Q at rated V, hot [kVAr]"), ("I_cold_pu", "current with cold magnets [pu]"),
            ("Bd_min_sc", "min magnet B at short circuit [T]"), ("magnet_kg", "magnet mass [kg]"),
            ("Xd_pu", "Xd [pu]"), ("E0_cold_pu", "open-circuit E0 cold [pu]")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, lab) in zip(axs.ravel(), keys):
        ax.plot(x, [r[k] for r in rows], "o-", lw=2)
        if k == "Bd_min_sc":
            from pm_fem2d.machine import design_preset
            ax.axhline(design_preset().B_demag_limit, color="#c0392b", ls=":", label="demag limit")
            ax.legend()
        if k == "Q_kVAr":
            ax.axhline(300, color="#1e8449", ls=":")
        if k == "I_cold_pu":
            ax.axhline(1.0, color="#c0392b", ls=":")
        ax.set_xlabel(xl); ax.set_title(lab); ax.grid(alpha=.3)
    fig.suptitle(f"PM condenser design sweep: {a.param}", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"sweep_{a.param}.png"), dpi=150); plt.close(fig)
    with open(os.path.join(a.out, f"sweep_{a.param}.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# PM condenser sweep: {a.param}\n\n{table}\n\n![sweep](sweep_{a.param}.png)\n")
    print(f"\nDone in {time.time() - t0:.0f} s -> {a.out}/sweep_{a.param}.(csv|md|png)")


if __name__ == "__main__":
    sys.exit(main())
