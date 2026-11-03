#!/usr/bin/env python3
"""Magnet-size and air-gap optimisation of the 300 kVAr IPM condenser (main dimensions fixed).

Variables (mm resolution): magnet thickness h_m, magnet width w_m, air gap g.
Constraints : 300 <= Q(400 V, hot magnets) <= 330 kVAr  (rating met, current <= 1.1 pu),
              magnet B at short circuit >= 0.10 T, stator tooth B <= 2.0 T, magnet fits.
Objectives  : minimise  (1) EMF THD line-line  (2) EM losses (heat)
                        (3) cogging torque     (4) current with cold magnets (start-up overload)

Note: at a fixed 400 V the delivered kVAr IS the current (Q = 3 V I), so "more kVAr" and
"less current" cannot both be maximised; the rating is a constraint and the current that
still varies with the magnet design - the cold-magnet (start-up) current - is minimised.

Method: ParEGO-type multi-objective Bayesian optimisation for 4 objectives.  Each batch
member gets a random weight vector; a GP is fitted to the augmented-Chebyshev scalarisation
of the normalised objectives and the candidate maximising expected improvement x probability
of feasibility (GP constraints) is evaluated.  Batches run in parallel; a live figure is
updated after every iteration.

    python run_opt.py --jobs 8                  # 20 initial + 10 x 8 = 100 FEM evaluations
    python run_opt.py --jobs 8 --iters 15
    python run_opt.py --resume --iters 5
    python run_opt.py --final-only
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from scipy.stats import norm

from mobo import GP, pareto_mask
from variants import (BOUNDS, LIMITS, OBJECTIVES, VARS, base_values, evaluate, from_unit, is_feasible,
                      load_base, make_variant, snap, to_unit, valid)

OBJ_KEYS = [k for k, _, _ in OBJECTIVES]


# ----------------------------------------------------------------------------- helpers
def F_of(rows):
    return np.array([[r[k] for k in OBJ_KEYS] for r in rows], float)


def pareto(rows):
    feas = [r for r in rows if is_feasible(r)]
    if not feas:
        return []
    m = pareto_mask(F_of(feas))
    return [r for r, k in zip(feas, m) if k]


def norm_bounds(rows):
    ok = [r for r in rows if r.get("fem_ok")]
    F = F_of(ok)
    lo, hi = F.min(0), F.max(0)
    return lo, np.where(hi > lo, hi - lo, 1.0)


def knee(pf, lo, span):
    if not pf:
        return None
    N = (F_of(pf) - lo) / span
    return pf[int(np.argmin(np.linalg.norm(N, axis=1)))]


def pick(pf, rows):
    if not pf:
        return {}
    lo, span = norm_bounds(rows)
    out = dict(compromise=knee(pf, lo, span))
    names = dict(thd_line_pct="best_THD", P_loss_kW="best_losses", cogging_Nm="best_cogging",
                 I_cold_pu="best_current")
    for k, lab, _ in OBJECTIVES:
        out[names[k]] = min(pf, key=lambda r: r[k])
    return out


def random_weights(rng, m):
    w = rng.dirichlet(np.ones(m))
    return w


def chebyshev(Fn, w, rho=0.05):
    return np.max(w * Fn, axis=1) + rho * np.sum(w * Fn, axis=1)


def candidates(rows, n, rng):
    U = rng.random((n, len(VARS)))
    X = [from_unit(u) for u in U]
    seeds = pareto(rows) or sorted([r for r in rows if r.get("fem_ok")],
                                   key=lambda r: abs(r["Q_kVAr"] - 315))[:6]
    for r in seeds:
        u0 = to_unit(r)
        for _ in range(60):
            X.append(from_unit(np.clip(u0 + rng.normal(0, 0.05, len(VARS)), 0, 1)))
    seen, out = set(), []
    for x in X:
        key = tuple(x[k] for k in VARS)
        if key not in seen:
            seen.add(key); out.append(x)
    return out


# ----------------------------------------------------------------------------- live plot
class Live:
    def __init__(self, outdir, window, base, ref):
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
        self.base, self.ref = base, ref

    def update(self, it, rows, batch_ids, hist):
        from ipm_fem2d.dimensions import draw_section
        plt, fig = self.plt, self.fig
        fig.clf()
        pf = pareto(rows)
        sel = pick(pf, rows)
        best = sel.get("compromise")
        fig.suptitle(f"Magnet & air-gap optimisation (ParEGO MOBO) — iteration {it}: {len(rows)} FEM evaluations, "
                     f"{sum(1 for r in rows if is_feasible(r))} feasible, {len(pf)} Pareto-optimal",
                     fontsize=13, fontweight="bold")
        ok = [r for r in rows if r.get("fem_ok")]
        # three projections of the 4-D objective space against losses
        for n_, (ky, lab) in enumerate((("thd_line_pct", "THD line-line [%]"), ("cogging_Nm", "cogging [Nm]"),
                                        ("I_cold_pu", "cold-magnet current [pu]"))):
            ax = fig.add_axes([0.04 + 0.205 * n_, 0.57, 0.17, 0.34])
            for r in ok:
                f = is_feasible(r)
                ax.plot(r["P_loss_kW"], r[ky], "o" if f else "x", ms=5 if f else 4,
                        color="#1f77b4" if f else "#bbbbbb", alpha=0.7)
            for r in pf:
                ax.plot(r["P_loss_kW"], r[ky], "o", ms=8, mfc="#2ca02c", mec="k")
            for i in batch_ids:
                r = rows[i]
                if r.get("fem_ok"):
                    ax.plot(r["P_loss_kW"], r[ky], "o", ms=12, mfc="none", mec="#e53935", mew=1.5)
            if self.ref:
                ax.plot(self.ref["P_loss_kW"], self.ref[ky], "*", ms=15, mfc="#ffca28", mec="k")
            if best:
                ax.plot(best["P_loss_kW"], best[ky], "D", ms=10, mfc="#e53935", mec="k")
            ax.set_xlabel("losses [kW]"); ax.set_ylabel(lab); ax.grid(alpha=.3)
        fig.text(0.04, 0.93, "green = Pareto, red ring = this batch, ★ = base design, ◆ = best compromise, x = infeasible",
                 fontsize=9)
        # parallel coordinates of the feasible designs
        axp = fig.add_axes([0.04, 0.07, 0.52, 0.38])
        keys = list(VARS) + ["Q_kVAr"] + OBJ_KEYS
        labs = ["h_m [mm]", "w_m [mm]", "gap [mm]", "Q [kVAr]", "THD [%]", "loss [kW]", "cogging [Nm]", "I_cold [pu]"]
        feas = [r for r in rows if is_feasible(r)]
        if feas:
            M = np.array([[r[k] for k in keys] for r in feas + ([self.ref] if self.ref else [])], float)
            lo, hi = M.min(0), M.max(0)
            Mn = (M - lo) / np.where(hi > lo, hi - lo, 1)
            pfset = {id(r) for r in pf}
            for i, r in enumerate(feas):
                axp.plot(range(len(keys)), Mn[i], "-", lw=2.2 if id(r) in pfset else 0.6,
                         color="#2ca02c" if id(r) in pfset else "#9e9e9e", alpha=0.9 if id(r) in pfset else 0.4)
            if self.ref:
                axp.plot(range(len(keys)), Mn[-1], "--", color="#ff8f00", lw=2, label="base")
            if best:
                axp.plot(range(len(keys)), (np.array([best[k] for k in keys]) - lo) / np.where(hi > lo, hi - lo, 1),
                         "-", color="#e53935", lw=3, label="best compromise")
            axp.legend(fontsize=8, loc="upper right")
            for j in range(len(keys)):
                s = 1e3 if keys[j] in VARS else 1
                axp.text(j, -0.08, f"{lo[j] * s:.3g}", ha="center", fontsize=7)
                axp.text(j, 1.04, f"{hi[j] * s:.3g}", ha="center", fontsize=7)
        axp.set_xticks(range(len(keys)), labs, fontsize=9); axp.set_yticks([]); axp.set_ylim(-0.12, 1.1)
        axp.set_title("Feasible designs (parallel coordinates; green = Pareto)")
        # rotor of the best compromise + table
        axs = fig.add_axes([0.62, 0.30, 0.20, 0.30]); axs.axis("off")
        axt = fig.add_axes([0.82, 0.07, 0.17, 0.85]); axt.axis("off")
        if best:
            d = make_variant(self.base, best)
            draw_section(axs, d)
            R = d.R_rotor * 1.12
            axs.set_xlim(-R, R); axs.set_ylim(-R, R); axs.set_aspect("equal")
            axs.set_title("best compromise rotor", fontsize=10)
            ref = self.ref or {}
            rows_t = [("", "best   /  base"),
                      ("magnet thickness", f"{best['h_m'] * 1e3:5.1f} / {ref.get('h_m', np.nan) * 1e3:5.1f} mm"),
                      ("magnet width", f"{best['w_m'] * 1e3:5.0f} / {ref.get('w_m', np.nan) * 1e3:5.0f} mm"),
                      ("air gap", f"{best['gap'] * 1e3:5.1f} / {ref.get('gap', np.nan) * 1e3:5.1f} mm"),
                      ("Q at 400 V", f"{best['Q_kVAr']:5.0f} / {ref.get('Q_kVAr', np.nan):5.0f} kVAr"),
                      ("THD line", f"{best['thd_line_pct']:5.2f} / {ref.get('thd_line_pct', np.nan):5.2f} %"),
                      ("losses", f"{best['P_loss_kW']:5.2f} / {ref.get('P_loss_kW', np.nan):5.2f} kW"),
                      ("cogging", f"{best['cogging_Nm']:5.0f} / {ref.get('cogging_Nm', np.nan):5.0f} Nm"),
                      ("current cold", f"{best['I_cold_pu']:5.2f} / {ref.get('I_cold_pu', np.nan):5.2f} pu"),
                      ("current rated", f"{best['I_pu']:5.2f} / {ref.get('I_pu', np.nan):5.2f} pu"),
                      ("magnet B s.c.", f"{best['Bd_min_sc']:5.2f} / {ref.get('Bd_min_sc', np.nan):5.2f} T"),
                      ("magnet mass", f"{best['m_magnet_kg']:5.0f} / {ref.get('m_magnet_kg', np.nan):5.0f} kg")]
            y = 0.98
            for k, v in rows_t:
                axt.text(0.0, y, k, fontsize=9.5, transform=axt.transAxes)
                axt.text(1.0, y, v, fontsize=9.5, ha="right", family="monospace", transform=axt.transAxes)
                y -= 0.05
        # progress
        axh = fig.add_axes([0.62, 0.07, 0.17, 0.18])
        if hist:
            n, nf, npf = zip(*hist)
            axh.plot(n, nf, "o-", label="feasible"); axh.plot(n, npf, "s-", label="Pareto")
            axh.legend(fontsize=8)
        axh.set_xlabel("evaluations", fontsize=9); axh.grid(alpha=.3); axh.tick_params(labelsize=8)
        path = os.path.join(self.dir, f"iter_{it:02d}.png")
        fig.savefig(path, dpi=110)
        fig.savefig(os.path.join(os.path.dirname(self.dir), "live_progress.png"), dpi=110)
        if self.interactive:
            fig.canvas.draw_idle(); plt.pause(0.05)


# ----------------------------------------------------------------------------- io
def save_rows(rows, path):
    keys = sorted({k for r in rows for k in r})
    first = ["it"] + list(VARS) + ["valid", "fem_ok", "reason"]
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
        return [{k: num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def final(rows, ref, base, outdir, solver_name, analyse=True):
    import matplotlib.pyplot as plt
    pf = pareto(rows)
    sel = pick(pf, rows)
    save_rows(pf, os.path.join(outdir, "pareto.csv"))
    # radar chart: selected designs vs base (normalised, lower = better)
    lo, span = norm_bounds(rows + ([ref] if ref else []))
    fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, polar=True)
    ang = np.linspace(0, 2 * np.pi, len(OBJ_KEYS), endpoint=False)
    for name, r, c in [("base", ref, "#ff8f00")] + [(k, v, None) for k, v in sel.items()]:
        if not r:
            continue
        vals = (np.array([r[k] for k in OBJ_KEYS]) - lo) / span
        ax.plot(np.r_[ang, ang[0]], np.r_[vals, vals[0]], "-o", lw=2 if name in ("base", "compromise") else 1,
                label=name.replace("_", " "), color=c)
    ax.set_xticks(ang, [lab for _, lab, _ in OBJECTIVES], fontsize=9)
    ax.set_title("Normalised objectives (closer to centre = better)"); ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(outdir, "radar_selected.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    L = ["# Magnet & air-gap optimisation — 300 kVAr IPM condenser (fixed main dimensions)\n",
         f"{len(rows)} FEM evaluations, {sum(1 for r in rows if is_feasible(r))} feasible, {len(pf)} Pareto-optimal.\n",
         "Objectives (minimised): EMF THD line-line, EM losses, cogging torque (unskewed 2-D peak), "
         "current with cold magnets. Constraints: 300 ≤ Q ≤ 330 kVAr at 400 V, magnet B at short circuit ≥ 0.10 T, "
         "tooth B ≤ 2.0 T.\n", "![radar](radar_selected.png)\n",
         "| design | h_m mm | w_m mm | gap mm | Q kVAr | I rated pu | I cold pu | THD % | losses kW | cogging Nm | B_sc T | magnet kg |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in [("base", ref)] + list(sel.items()):
        if not r:
            continue
        L.append(f"| {name.replace('_', ' ')} | {r['h_m'] * 1e3:.1f} | {r['w_m'] * 1e3:.0f} | {r['gap'] * 1e3:.1f} | "
                 f"{r['Q_kVAr']:.0f} | {r['I_pu']:.2f} | {r['I_cold_pu']:.2f} | {r['thd_line_pct']:.2f} | "
                 f"{r['P_loss_kW']:.2f} | {r['cogging_Nm']:.0f} | {r['Bd_min_sc']:.2f} | {r['m_magnet_kg']:.0f} |")
    L += ["\n## Pareto set\n", "| h_m | w_m | gap | Q | THD % | losses kW | cogging Nm | I cold pu |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(pf, key=lambda r: r["P_loss_kW"]):
        L.append(f"| {r['h_m'] * 1e3:.1f} | {r['w_m'] * 1e3:.0f} | {r['gap'] * 1e3:.1f} | {r['Q_kVAr']:.0f} | "
                 f"{r['thd_line_pct']:.2f} | {r['P_loss_kW']:.2f} | {r['cogging_Nm']:.0f} | {r['I_cold_pu']:.2f} |")
    L.append("\nCogging is the 2-D value without skew; the one-slot stator skew of the design reduces it "
             "strongly in the real machine. THD includes the skew.")
    with open(os.path.join(outdir, "opt_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    if analyse and sel.get("compromise"):
        from ipm_fem2d import plots, solver
        from ipm_fem2d.analysis import PMAnalysis
        solver.set_backend(solver_name)
        print("\nFull analysis of the best-compromise design ...")
        an = PMAnalysis(make_variant(base, sel["compromise"]))
        an.run_all(quick=True)
        plots.make_all(an, os.path.join(outdir, "design_compromise"))
    return sel


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--solver", default="auto", choices=["auto", "gpu", "pardiso", "superlu"])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--bo-results", default="../Liang_IPM_BO/results/pareto.csv",
                    help="Pareto CSV of the size optimisation; its best compromise is the base design")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    rng = np.random.default_rng(a.seed)
    jobs = a.jobs or max(1, min(a.batch, (os.cpu_count() or 2) // 2))
    base, src = load_base(a.bo_results)
    print(f"Base design: {src}: {base['poles']}-pole, Ø{base['D'] * 1e3:.0f} mm, L {base['L'] * 1e3:.0f} mm")
    csv_path = os.path.join(a.out, "all_evals.csv")
    ref_path = os.path.join(a.out, "base_design.csv")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows, hist, it0 = [], [], 1
    if (a.resume or a.final_only) and os.path.exists(csv_path):
        rows = load_rows(csv_path)
        it0 = int(max(r.get("it", 0) for r in rows)) + 1
    ref = load_rows(ref_path)[0] if os.path.exists(ref_path) else None
    if a.final_only:
        sel = final(rows, ref, base, a.out, a.solver, analyse=not a.skip_full)
        _print(sel, ref); return
    live = Live(a.out, not a.no_window, base, ref)
    t0 = time.time()

    def run(designs, it):
        ids = []
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(evaluate, (base, x, a.solver)) for x in designs]
            for fu in as_completed(futs):
                r = fu.result(); r["it"] = it
                rows.append(r); ids.append(len(rows) - 1)
                print(f"  [{len(rows):3d}] h {r['h_m'] * 1e3:4.1f}  w {r['w_m'] * 1e3:4.0f}  g {r['gap'] * 1e3:3.1f} mm -> "
                      + (f"Q {r['Q_kVAr']:5.0f}  THD {r['thd_line_pct']:4.2f} %  loss {r['P_loss_kW']:4.2f} kW  "
                         f"cog {r['cogging_Nm']:4.0f} Nm  Icold {r['I_cold_pu']:4.2f}  Bsc {r['Bd_min_sc']:4.2f}  "
                         f"{'FEASIBLE' if is_feasible(r) else ''}" if r.get("fem_ok") else r.get("reason", ""))
                      + f"  ({r['runtime_s']:.0f} s)", flush=True)
        save_rows(rows, csv_path)
        return ids

    if ref is None:
        print("Evaluating the base design ...")
        ref = evaluate((base, snap(base_values(base)), a.solver)); ref["it"] = 0
        save_rows([ref], ref_path); live.ref = ref
    if not rows:
        rows.append(dict(ref, it=0))                       # the base design competes too
        from scipy.stats import qmc
        U = qmc.Sobol(d=len(VARS), scramble=True, seed=a.seed).random(256)
        init = []
        for u in U:
            x = from_unit(u)
            if valid(base, x)[0]:
                init.append(x)
            if len(init) >= a.n_init:
                break
        print(f"Initial design: {len(init)} Sobol points, {jobs} parallel FEM jobs")
        ids = run(init, 0)
        hist.append((len(rows), sum(1 for r in rows if is_feasible(r)), len(pareto(rows))))
        live.update(0, rows, ids, hist)
    for it in range(it0, it0 + a.iters):
        ok = [r for r in rows if r.get("fem_ok")]
        X = np.array([to_unit(r) for r in ok])
        cons = dict(Q=GP().fit(X, [r["Q_kVAr"] for r in ok]), Bd=GP().fit(X, [r["Bd_min_sc"] for r in ok]),
                    Bt=GP().fit(X, [r["B_tooth_rated"] for r in ok]))
        lo, span = norm_bounds(rows)
        Fn = (F_of(ok) - lo) / span
        feas_ok = np.array([is_feasible(r) for r in ok])
        C = [c for c in candidates(rows, 3000, rng) if valid(base, c)[0]]
        U = np.array([to_unit(c) for c in C])
        mq, sq = cons["Q"].predict(U); mb, sb = cons["Bd"].predict(U); mt, st = cons["Bt"].predict(U)
        pof = ((norm.cdf((mq - LIMITS["Q_min"]) / sq) - norm.cdf((mq - LIMITS["Q_max"]) / sq))
               * norm.cdf((mb - LIMITS["Bd_min"]) / sb) * norm.cdf((LIMITS["B_tooth_max"] - mt) / st))
        chosen = []
        for _ in range(a.batch):
            w = random_weights(rng, len(OBJ_KEYS))
            s = chebyshev(Fn, w)
            s_pen = np.where(feas_ok, s, s.max() + 0.5)  # infeasible points: worst scalar value
            gp = GP().fit(X, s_pen)
            mu, sd = gp.predict(U)
            best = s[feas_ok].min() if feas_ok.any() else s_pen.min()
            z = (best - mu) / sd
            ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
            acq = ei * pof if feas_ok.any() else pof
            for j in chosen:
                acq[np.linalg.norm(U - U[j], axis=1) < 0.06] = -1
            j = int(np.argmax(acq))
            if acq[j] < 0:
                break
            chosen.append(j)
        designs = [C[j] for j in chosen]
        print(f"\nIteration {it}: {len(C)} candidates, evaluating {len(designs)}")
        ids = run(designs, it)
        hist.append((len(rows), sum(1 for r in rows if is_feasible(r)), len(pareto(rows))))
        print(f"  feasible {hist[-1][1]}, Pareto {hist[-1][2]}, elapsed {(time.time() - t0) / 60:.1f} min")
        live.update(it, rows, ids, hist)
    sel = final(rows, ref, base, a.out, a.solver, analyse=not a.skip_full)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.  Report: {os.path.join(a.out, 'opt_report.md')}")
    _print(sel, ref)
    if live.interactive:
        live.plt.ioff(); print("Close the live window to exit."); live.plt.show()


def _print(sel, ref):
    for name, r in [("base", ref)] + list(sel.items()):
        if r:
            print(f"  {name:16s}: h {r['h_m'] * 1e3:4.1f} w {r['w_m'] * 1e3:4.0f} g {r['gap'] * 1e3:3.1f} mm | "
                  f"Q {r['Q_kVAr']:4.0f} kVAr, THD {r['thd_line_pct']:4.2f} %, loss {r['P_loss_kW']:4.2f} kW, "
                  f"cogging {r['cogging_Nm']:4.0f} Nm, I cold {r['I_cold_pu']:4.2f} pu, B_sc {r['Bd_min_sc']:4.2f} T")


if __name__ == "__main__":
    sys.exit(main())
