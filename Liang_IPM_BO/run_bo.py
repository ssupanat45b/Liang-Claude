#!/usr/bin/env python3
"""Multi-objective Bayesian optimisation of a 300 kVAr / 400 V IPM synchronous condenser
with flat (rectangular) magnets.

    minimise  (1) active mass [kg]        (2) electromagnetic losses [kW]
    subject to 300 <= Q(400 V) <= 330 kVAr, magnet B at short circuit >= 0.10 T,
               stator tooth B (rated) <= 2.0 T, 0.3 <= L/D <= 3.0, valid rotor geometry
    over       poles {2,4,6,8}, bore D, length L, magnet thickness, depth and width

After every iteration (one parallel batch of FEM runs) a live figure is updated and saved
(results/iterations/iter_NN.png): objective space with the Pareto front, hypervolume
convergence, the current best-compromise design and the explored design space.

    python run_bo.py --jobs 8                       # 32 initial + 12 x 8 BO evaluations
    python run_bo.py --jobs 8 --iters 20 --batch 8
    python run_bo.py --resume                       # continue from results/all_evals.csv
    python run_bo.py --final-only                   # re-select and re-analyse the best designs
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

from design_space import BOUNDS, CONT, LIMITS, POLES, cheap, fem_evaluate, from_unit, is_feasible, to_unit

REF = np.array([1800.0, 8.0])          # hypervolume reference point: (mass kg, losses kW)
KW_PV = 7066.0                         # USD per kW of continuous loss (10 y, 6 %, 0.12 USD/kWh, 8000 h/y)
POLE_COL = {2: "#1f77b4", 4: "#2ca02c", 6: "#ff7f0e", 8: "#9467bd"}


# ----------------------------------------------------------------------------- sampling
def sobol_designs(n_per_pole, seed, poles=POLES, oversample=40):
    from scipy.stats import qmc
    out = []
    for k, p in enumerate(poles):
        sob = qmc.Sobol(d=len(CONT), scramble=True, seed=seed + 17 * k)
        U = sob.random(2 ** int(math.ceil(math.log2(n_per_pole * oversample))))
        got = 0
        for u in U:
            x = from_unit(u, p)
            if cheap(x)["valid"]:
                out.append(x); got += 1
                if got >= n_per_pole:
                    break
    return out


def candidates(rows, n_per_pole, seed, rng):
    """Random + local candidates (validity and the exact mass computed analytically)."""
    from scipy.stats import qmc
    X = []
    for k, p in enumerate(POLES):
        U = qmc.Sobol(d=len(CONT), scramble=True, seed=seed + 101 * k).random(n_per_pole)
        X += [from_unit(u, p) for u in U]
    feas = [r for r in rows if is_feasible(r)]
    # seeds for local search: feasible designs + those whose Q is closest to the target band
    ok = [r for r in rows if r.get("fem_ok")]
    near = sorted(ok, key=lambda r: abs(r["Q_kVAr"] - 0.5 * (LIMITS["Q_min"] + LIMITS["Q_max"])))[:8]
    seeds = feas + [r for r in near if r not in feas]
    for r in seeds:                                      # local refinement around good designs
        u0 = to_unit(r)[:len(CONT)]
        for _ in range(40):
            X.append(from_unit(np.clip(u0 + rng.normal(0, 0.06, len(CONT)), 0, 1), r["poles"]))
            if rng.random() < 0.15:                      # also try a neighbouring pole count
                p2 = int(rng.choice(POLES))
                X.append(from_unit(np.clip(u0 + rng.normal(0, 0.06, len(CONT)), 0, 1), p2))
    C = [cheap(x) for x in X]
    C = [c for c in C if c["valid"]]
    return C


# ----------------------------------------------------------------------------- selection
def front_of(rows):
    feas = [r for r in rows if is_feasible(r)]
    if not feas:
        return [], np.zeros((0, 2))
    F = np.array([[r["m_active"], r["P_loss_kW"]] for r in feas])
    m = pareto_mask(F)
    pf = [r for r, k in zip(feas, m) if k]
    pf.sort(key=lambda r: r["m_active"])
    return pf, np.array([[r["m_active"], r["P_loss_kW"]] for r in pf])


def pick(pf):
    """Named designs on the Pareto front."""
    if not pf:
        return {}
    F = np.array([[r["m_active"], r["P_loss_kW"]] for r in pf])
    lo, hi = F.min(0), F.max(0)
    N = (F - lo) / np.where(hi > lo, hi - lo, 1)
    knee = pf[int(np.argmin(np.hypot(N[:, 0], N[:, 1])))]
    life = min(pf, key=lambda r: r["cost"] + r["P_loss_kW"] * KW_PV)
    return dict(lightest=pf[0], lowest_loss=pf[-1], compromise=knee, lifetime_cost=life)


from mobo import GP, hypervolume_2d, pareto_mask, select_batch  # noqa: E402


# ----------------------------------------------------------------------------- live plot
class Live:
    def __init__(self, outdir, window):
        import matplotlib
        if not window:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.interactive = window and matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps")
        if self.interactive:
            plt.ion()
        self.dir = os.path.join(outdir, "iterations")
        os.makedirs(self.dir, exist_ok=True)
        self.fig = plt.figure(figsize=(19, 10.5))

    def update(self, it, rows, hv_hist, batch_ids):
        from design_space import make_design
        from ipm_fem2d.dimensions import draw_section
        plt, fig = self.plt, self.fig
        fig.clf()
        pf, F = front_of(rows)
        sel = pick(pf)
        nfe = sum(1 for r in rows if is_feasible(r))
        fig.suptitle(f"MOBO — 300 kVAr / 400 V IPM condenser, flat magnets  |  iteration {it}, "
                     f"{len(rows)} FEM evaluations, {nfe} feasible, Pareto size {len(pf)}",
                     fontsize=13, fontweight="bold")
        # (a) objective space
        ax = fig.add_axes([0.04, 0.55, 0.40, 0.37])
        for r in rows:
            if not r.get("fem_ok"):
                continue
            if is_feasible(r):
                ax.plot(r["m_active"], r["P_loss_kW"], "o", ms=6, color=POLE_COL[r["poles"]], alpha=0.85)
            else:
                ax.plot(r["m_active"], r["P_loss_kW"], "x", ms=5, color="#aaaaaa")
        for i in batch_ids:
            r = rows[i]
            if r.get("fem_ok"):
                ax.plot(r["m_active"], r["P_loss_kW"], "o", ms=12, mfc="none", mec="#e53935", mew=1.6)
        if len(F):
            xs = np.r_[F[:, 0], REF[0]]
            ys = np.r_[F[:, 1], F[-1, 1]]
            ax.step(np.r_[F[0, 0], xs], np.r_[REF[1], ys], where="post", color="k", lw=1.4)
            for name, r in sel.items():
                ax.annotate(name.replace("_", " "), (r["m_active"], r["P_loss_kW"]), fontsize=8,
                            xytext=(6, 6), textcoords="offset points")
        for p, c in POLE_COL.items():
            ax.plot([], [], "o", color=c, label=f"{p}-pole feasible")
        ax.plot([], [], "x", color="#aaaaaa", label="infeasible"); ax.plot([], [], "o", mfc="none", mec="#e53935", label="this batch")
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.set_xlabel("active mass [kg]"); ax.set_ylabel("EM losses [kW]"); ax.grid(alpha=.3)
        ax.set_title("Objective space (black: Pareto front)")
        good = [r for r in rows if r.get("fem_ok")]
        if good:
            ms = [r["m_active"] for r in good]; ls = [r["P_loss_kW"] for r in good]
            ax.set_xlim(0.9 * min(ms), min(max(ms) * 1.05, 3500)); ax.set_ylim(0.9 * min(ls), min(max(ls) * 1.05, 15))
        # (b) hypervolume
        axh = fig.add_axes([0.50, 0.55, 0.20, 0.37])
        if hv_hist:
            n, hv = zip(*hv_hist)
            axh.plot(n, hv, "o-")
        axh.set_xlabel("FEM evaluations"); axh.set_title("Hypervolume (higher = better)"); axh.grid(alpha=.3)
        # (c) best compromise cross-section + table
        best = sel.get("compromise")
        axs = fig.add_axes([0.73, 0.50, 0.26, 0.44])
        axs.axis("off")
        if best:
            d, _ = make_design(best)
            draw_section(axs, d)
            R = d.R_stator_out
            axs.set_xlim(-R * 1.05, R * 1.05); axs.set_ylim(-R * 1.05, R * 1.05); axs.set_aspect("equal")
            axs.set_title(f"best compromise: {best['poles']}-pole, Ø{best['D'] * 1e3:.0f} × L{best['L'] * 1e3:.0f} mm",
                          fontsize=10)
        axt = fig.add_axes([0.73, 0.05, 0.26, 0.40]); axt.axis("off")
        if best:
            rows_t = [("poles / speed", f"{best['poles']} / {best['rpm']:.0f} rpm"),
                      ("bore Ø / outer Ø", f"{best['D'] * 1e3:.0f} / {best['D_out'] * 1e3:.0f} mm"),
                      ("core length (L/D)", f"{best['L'] * 1e3:.0f} mm ({best['L_over_D']:.2f})"),
                      ("magnet w × h", f"{best['w_m'] * 1e3:.0f} × {best['h_m'] * 1e3:.1f} mm"),
                      ("magnet depth", f"{best['d_m'] * 1e3:.1f} mm"),
                      ("active / magnet mass", f"{best['m_active']:.0f} / {best['m_magnet']:.0f} kg"),
                      ("losses", f"{best['P_loss_kW']:.2f} kW ({100 * best['P_loss_kW'] / 300:.2f} %)"),
                      ("Q at 400 V", f"{best['Q_kVAr']:.0f} kVAr"),
                      ("magnet B at s.c.", f"{best['Bd_min_sc']:.2f} T"),
                      ("tooth B rated", f"{best['B_tooth_rated']:.2f} T"),
                      ("cost index", f"{best['cost']:,.0f} USD")]
            y = 0.97
            for k, v in rows_t:
                axt.text(0.0, y, k, fontsize=10, transform=axt.transAxes)
                axt.text(1.0, y, v, fontsize=10, ha="right", family="monospace", transform=axt.transAxes)
                y -= 0.085
        # (d) design space D vs L, (e) magnet h vs mass
        axd = fig.add_axes([0.04, 0.06, 0.30, 0.38])
        axm = fig.add_axes([0.40, 0.06, 0.30, 0.38])
        for r in rows:
            if not r.get("valid"):
                continue
            f = is_feasible(r)
            c = POLE_COL[r["poles"]]
            axd.plot(r["D"] * 1e3, r["L"] * 1e3, "o" if f else ".", color=c, ms=6 if f else 4, alpha=0.9 if f else 0.4)
            if r.get("fem_ok"):
                axm.plot(r["D"] * 1e3, min(r.get("Q_max_kVAr", 0), 600), "o" if f else ".", color=c,
                         ms=6 if f else 4, alpha=0.9 if f else 0.4)
        axm.axhline(LIMITS["Q_min"], color="#1e8449", ls="--")
        axd.set_xlabel("bore Ø [mm]"); axd.set_ylabel("core length [mm]"); axd.grid(alpha=.3)
        axd.set_title("Explored design space (large = feasible)")
        axm.set_xlabel("bore Ø [mm]"); axm.set_ylabel("max achievable Q at 400 V [kVAr]"); axm.grid(alpha=.3)
        axm.set_title("Achievable Q with the best magnet thickness (green = 300 kVAr)")
        axm.set_ylim(0, 600)
        path = os.path.join(self.dir, f"iter_{it:02d}.png")
        fig.savefig(path, dpi=110)
        fig.savefig(os.path.join(os.path.dirname(self.dir), "live_progress.png"), dpi=110)
        if self.interactive:
            fig.canvas.draw_idle(); plt.pause(0.05)
        return path


# ----------------------------------------------------------------------------- io
def save_rows(rows, path):
    keys = sorted({k for r in rows for k in r})
    first = ["it", "poles", "D", "L", "h_m", "d_frac", "w_frac", "valid", "fem_ok", "reason"]
    keys = first + [k for k in keys if k not in first]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def load_rows(path):
    def num(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    with open(path) as fh:
        rows = [{k: num(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    for r in rows:
        r["poles"] = int(r["poles"])
    return rows


# ----------------------------------------------------------------------------- final
def final_report(rows, outdir, solver_name, analyse=True):
    import matplotlib.pyplot as plt
    from design_space import make_design
    pf, F = front_of(rows)
    sel = pick(pf)
    save_rows(pf, os.path.join(outdir, "pareto.csv"))
    # Pareto figure
    fig, ax = plt.subplots(figsize=(11, 7))
    for r in rows:
        if r.get("fem_ok") and not is_feasible(r):
            ax.plot(r["m_active"], r["P_loss_kW"], "x", color="#bbbbbb", ms=5)
    for r in rows:
        if is_feasible(r):
            ax.plot(r["m_active"], r["P_loss_kW"], "o", color=POLE_COL[r["poles"]], ms=6, alpha=0.6)
    if len(F):
        ax.plot(F[:, 0], F[:, 1], "k-", lw=1.5)
        for r in pf:
            ax.plot(r["m_active"], r["P_loss_kW"], "o", ms=9, mfc=POLE_COL[r["poles"]], mec="k")
        for name, r in sel.items():
            ax.annotate(f"{name.replace('_', ' ')}\n{r['poles']}p Ø{r['D'] * 1e3:.0f} L{r['L'] * 1e3:.0f}",
                        (r["m_active"], r["P_loss_kW"]), fontsize=8, xytext=(8, 8), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.6))
    for p, c in POLE_COL.items():
        ax.plot([], [], "o", color=c, label=f"{p}-pole")
    ax.legend(); ax.grid(alpha=.3)
    ax.set_xlabel("active mass [kg]"); ax.set_ylabel("EM losses [kW]")
    ax.set_title("Pareto front — 300 kVAr IPM condenser (flat magnets)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "pareto_front.png"), dpi=150); plt.close(fig)
    # report
    L = ["# MOBO — 300 kVAr / 400 V IPM synchronous condenser (flat magnets)\n",
         f"{len(rows)} FEM evaluations, {sum(1 for r in rows if is_feasible(r))} feasible, "
         f"{len(pf)} Pareto-optimal designs.\n",
         "Objectives: active mass and EM losses (both minimised). Constraints: "
         f"{LIMITS['Q_min']:.0f} ≤ Q ≤ {LIMITS['Q_max']:.0f} kVAr at 400 V, magnet B at short circuit ≥ "
         f"{LIMITS['Bd_min']} T, tooth B ≤ {LIMITS['B_tooth_max']} T, {LIMITS['LD_min']} ≤ L/D ≤ {LIMITS['LD_max']}.\n",
         "![pareto](pareto_front.png)\n", "## Selected designs\n",
         "| design | poles | Ø bore | Ø out | L | magnet w×h×depth | active kg | magnet kg | losses kW | Q kVAr | B_sc T | cost USD | 10-y cost USD |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in sel.items():
        L.append(f"| {name.replace('_', ' ')} | {r['poles']} | {r['D'] * 1e3:.0f} | {r['D_out'] * 1e3:.0f} | {r['L'] * 1e3:.0f} | "
                 f"{r['w_m'] * 1e3:.0f}×{r['h_m'] * 1e3:.1f}×{r['d_m'] * 1e3:.1f} | {r['m_active']:.0f} | {r['m_magnet']:.0f} | "
                 f"{r['P_loss_kW']:.2f} | {r['Q_kVAr']:.0f} | {r['Bd_min_sc']:.2f} | {r['cost']:,.0f} | "
                 f"{r['cost'] + r['P_loss_kW'] * KW_PV:,.0f} |")
    L += ["\n## Pareto front\n", "| poles | Ø bore | L | h_m | active kg | losses kW | Q | B_sc |", "|---|---|---|---|---|---|---|---|"]
    for r in pf:
        L.append(f"| {r['poles']} | {r['D'] * 1e3:.0f} | {r['L'] * 1e3:.0f} | {r['h_m'] * 1e3:.1f} | {r['m_active']:.0f} | "
                 f"{r['P_loss_kW']:.2f} | {r['Q_kVAr']:.0f} | {r['Bd_min_sc']:.2f} |")
    with open(os.path.join(outdir, "bo_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    # full FEM study + design sheet of the recommended (best-compromise) and lightest design
    if analyse and sel:
        from ipm_fem2d import plots, solver
        from ipm_fem2d.analysis import PMAnalysis
        solver.set_backend(solver_name)
        done = set()
        for name in ("compromise", "lightest"):
            r = sel[name]
            key = (r["poles"], round(r["D"], 5), round(r["L"], 5), round(r["h_m"], 6))
            if key in done:
                continue
            done.add(key)
            d, _ = make_design(r)
            print(f"\nFull analysis of the '{name}' design ...")
            an = PMAnalysis(d)
            an.run_all(quick=True)
            out = os.path.join(outdir, f"design_{name}")
            plots.make_all(an, out)
    return sel


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-init", type=int, default=32, help="initial Sobol designs (split over the pole counts)")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8, help="FEM evaluations per iteration (run in parallel)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel processes (0 = min(batch, CPU/2))")
    ap.add_argument("--solver", default="auto", choices=["auto", "gpu", "pardiso", "superlu"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip-full", action="store_true", help="skip the full FEM study of the selected designs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "all_evals.csv")
    rng = np.random.default_rng(a.seed)
    jobs = a.jobs or max(1, min(a.batch, (os.cpu_count() or 2) // 2))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    if a.final_only:
        rows = load_rows(csv_path)
        sel = final_report(rows, a.out, a.solver, analyse=not a.skip_full)
        _print_sel(sel)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows, hv_hist, it0 = [], [], 0
    if a.resume and os.path.exists(csv_path):
        rows = load_rows(csv_path)
        it0 = int(max(r.get("it", 0) for r in rows)) + 1
        print(f"resumed {len(rows)} evaluations")
    live = Live(a.out, window=not a.no_window)
    t0 = time.time()

    def run_batch(designs, it):
        ids = []
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(fem_evaluate, (x, a.solver)): x for x in designs}
            for fu in as_completed(futs):
                r = fu.result(); r["it"] = it
                rows.append(r); ids.append(len(rows) - 1)
                f = is_feasible(r)
                print(f"  [{len(rows):3d}] {r['poles']}p Ø{r['D'] * 1e3:4.0f} L{r['L'] * 1e3:5.0f} "
                      f"w{r['w_frac']:.2f} d{r['d_frac']:.3f} -> magnet {r['h_m'] * 1e3:5.1f} mm  "
                      + (f"Q {r['Q_kVAr']:6.1f} (max {r.get('Q_max_kVAr', float('nan')):5.0f})  mass {r['m_active']:6.0f} kg  loss {r['P_loss_kW']:5.2f} kW  "
                         f"Bsc {r['Bd_min_sc']:5.2f}  {'FEASIBLE' if f else ''}" if r.get("fem_ok") else r.get("reason", ""))
                      + f"  ({r['runtime_s']:.0f} s)", flush=True)
        save_rows(rows, csv_path)
        return ids

    if not rows:
        init = sobol_designs(max(1, a.n_init // len(POLES)), a.seed)
        print(f"Initial design: {len(init)} Sobol points, {jobs} parallel FEM jobs")
        ids = run_batch(init, 0)
        _, F = front_of(rows)
        hv_hist.append((len(rows), hypervolume_2d(F, REF)))
        live.update(0, rows, hv_hist, ids)
        it0 = 1
    for it in range(it0, it0 + a.iters):
        ok = [r for r in rows if r.get("fem_ok")]
        X = np.array([to_unit(r) for r in ok])
        models = dict(M=GP(log=True).fit(X, [r["m_active"] for r in ok]),
                      P=GP(log=True).fit(X, [r["P_loss_kW"] for r in ok]),
                      Qmax=GP().fit(X, [min(r["Q_max_kVAr"], 600.0) for r in ok]),
                      Bd=GP().fit(X, [r["Bd_min_sc"] for r in ok]),
                      Bt=GP().fit(X, [r["B_tooth_rated"] for r in ok]))
        C = candidates(rows, 1024, a.seed * 1000 + it, rng)
        U = np.array([to_unit(c) for c in C])
        _, F = front_of(rows)
        chosen = select_batch(U, models, F, REF, LIMITS, a.batch, rng=rng)
        designs = [{k: C[j][k] for k in ("poles",) + CONT} for j in chosen]
        print(f"\nIteration {it}: {len(C)} candidates, evaluating {len(designs)}")
        ids = run_batch(designs, it)
        _, F = front_of(rows)
        hv_hist.append((len(rows), hypervolume_2d(F, REF)))
        print(f"  hypervolume {hv_hist[-1][1]:.1f}, feasible {sum(1 for r in rows if is_feasible(r))}, "
              f"elapsed {(time.time() - t0) / 60:.1f} min")
        live.update(it, rows, hv_hist, ids)
    sel = final_report(rows, a.out, a.solver, analyse=not a.skip_full)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.  Report: {os.path.join(a.out, 'bo_report.md')}")
    _print_sel(sel)
    if live.interactive:
        live.plt.ioff(); print("Close the live window to exit."); live.plt.show()


def _print_sel(sel):
    for name, r in sel.items():
        print(f"  {name:14s}: {r['poles']}-pole Ø{r['D'] * 1e3:.0f} L{r['L'] * 1e3:.0f} mm, magnet "
              f"{r['w_m'] * 1e3:.0f}x{r['h_m'] * 1e3:.1f} mm, mass {r['m_active']:.0f} kg "
              f"(magnets {r['m_magnet']:.0f} kg), losses {r['P_loss_kW']:.2f} kW, Q {r['Q_kVAr']:.0f} kVAr")


if __name__ == "__main__":
    sys.exit(main())
