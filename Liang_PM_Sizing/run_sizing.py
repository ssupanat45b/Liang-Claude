#!/usr/bin/env python3
"""Sizing study of a 300 kVAr / 400 V SPM synchronous condenser:
stator bore 250-550 mm (largest -> smallest) x 2, 4, 6, 8 poles.

During the run a live figure is updated after every finished case (and saved as
iterations/iter_NN_pX_DYYY.png): cross-section of that design, its key
parameters, the heatmap filling in, and trends vs. bore diameter.
At the end, heatmaps of all key quantities are written to results/.

    python run_sizing.py                       # 28 cases, live window, parallel
    python run_sizing.py --jobs 8              # parallel worker processes
    python run_sizing.py --no-window           # no live window (figures are still saved)
    python run_sizing.py --demag-limit -0.3    # e.g. N38UH grade (knee below -0.3 T at 80 C)
    python run_sizing.py --poles 4 6 --D 0.35 0.40 0.45 --quick
    python run_sizing.py --replot              # redraw heatmaps from results/sizing_results.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

METRICS = [  # key, title, format, colormap, "lower is better"
    ("L_over_D", "Core length / bore  L/D", "{:.2f}", "viridis", None),
    ("D_out", "Stator outer Ø [mm]", "{:.0f}", "viridis", True),
    ("h_m_mm", "Magnet thickness [mm]", "{:.1f}", "magma_r", True),
    ("m_magnet_kg", "Magnet mass [kg]", "{:.0f}", "magma_r", True),
    ("m_active_kg", "Active mass [kg]", "{:.0f}", "magma_r", True),
    ("cost_usd", "Material cost index [USD]", "{:,.0f}", "magma_r", True),
    ("loss_pct", "EM losses [% of rating]", "{:.2f}", "magma_r", True),
    ("Xd_pu", "Xd [pu]", "{:.3f}", "viridis", None),
    ("I_sc_pu", "Steady short-circuit current [pu]", "{:.2f}", "magma_r", True),
    ("Bd_min_sc", "Min magnet B at short circuit [T]", "{:.2f}", "RdYlGn", False),
    ("I_cold_pu", "Current with cold magnets [pu]", "{:.2f}", "magma_r", True),
    ("E0_cold_pu", "Open-circuit EMF, cold [pu]", "{:.2f}", "magma_r", True),
    ("dQ_per_pct", "dQ/dV [kVAr per 1 %]", "{:.1f}", "viridis", None),
    ("B_tooth_rated", "Stator tooth B, rated [T]", "{:.2f}", "magma_r", True),
    ("H_s", "Inertia constant H [s]", "{:.2f}", "viridis", False),
    ("thd_line_pct", "EMF THD line-line [%]", "{:.2f}", "magma_r", True),
]

LIMITS = dict(L_over_D=(0.3, 3.0), B_tooth_rated=(None, 2.0), I_cold_pu=(None, 1.6))


def feasible(r, demag_limit):
    """(geometric/electrical feasibility, demag ok, reasons)"""
    why = []
    if r.get("status") != "ok":
        why.append(r.get("reason") or r.get("status"))
    for k, (lo, hi) in LIMITS.items():
        v = r.get(k, float("nan"))
        if isinstance(v, float) and math.isnan(v):
            continue
        if lo is not None and v < lo:
            why.append(f"{k} < {lo}")
        if hi is not None and v > hi:
            why.append(f"{k} > {hi}")
    dm = r.get("Bd_min_sc", float("nan"))
    demag_ok = (not math.isnan(dm)) and dm > demag_limit
    return len(why) == 0, demag_ok, why


# ----------------------------------------------------------------------------- live plot
class LivePlot:
    def __init__(self, poles, Ds, outdir, window):
        import matplotlib
        if not window:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.interactive = window and matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps")
        if self.interactive:
            plt.ion()
        self.poles, self.Ds = poles, Ds
        self.outdir = os.path.join(outdir, "iterations")
        os.makedirs(self.outdir, exist_ok=True)
        self.fig = plt.figure(figsize=(19, 10.5))
        self.n = 0

    def update(self, r, design, done, total, demag_limit):
        from pm_fem2d.dimensions import draw_section
        plt, fig = self.plt, self.fig
        fig.clf()
        self.n += 1
        ok, dm_ok, why = feasible(r, demag_limit)
        fig.suptitle(f"Sizing study 300 kVAr / 400 V SPM condenser  —  case {done}/{total}:  "
                     f"{r['poles']}-pole, Ø bore {r['D_bore'] * 1e3:.0f} mm   "
                     f"[{'feasible' if ok else 'not feasible'}{'' if dm_ok else ', demag risk'}]",
                     fontsize=13, fontweight="bold", color="#1e8449" if ok and dm_ok else "#c0392b")
        # 1) cross-section of this case
        ax = fig.add_axes([0.01, 0.40, 0.25, 0.52])
        try:
            draw_section(ax, design)
        except Exception:                                   # noqa: BLE001
            pass
        R = design.R_stator_out
        ax.set_xlim(-R * 1.05, R * 1.05); ax.set_ylim(-R * 1.05, R * 1.05)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"Ø {r['D_bore'] * 1e3:.0f} / {r['D_out'] * 1e3:.0f} mm, L = {r['L'] * 1e3:.0f} mm, "
                     f"Qs = {r['Qs']}", fontsize=10)
        # 2) key parameters of this case
        axt = fig.add_axes([0.27, 0.40, 0.22, 0.52]); axt.axis("off")
        g = lambda k, f="{:.2f}": (f.format(r[k]) if k in r and not (isinstance(r[k], float) and math.isnan(r[k])) else "—")
        rows = [("speed", f"{r['rpm']:.0f} rpm"), ("core length L", f"{r['L'] * 1e3:.0f} mm  (L/D {r['L_over_D']:.2f})"),
                ("turns/coil × paths", f"{r['turns_per_coil']} × {r['a']}  (N = {r['N_series']:.0f})"),
                ("magnet thickness", g("h_m_mm", "{:.1f}") + " mm"), ("magnet mass", g("m_magnet_kg", "{:.1f}") + " kg"),
                ("Q at 400 V", g("Q_kVAr", "{:.0f}") + " kVAr"), ("Xd", g("Xd_pu", "{:.3f}") + " pu"),
                ("E0 hot / cold", g("E0_pu") + " / " + g("E0_cold_pu") + " pu"),
                ("short-circuit current", g("I_sc_pu") + " pu"),
                ("min magnet B (s.c.)", g("Bd_min_sc") + " T"), ("tooth B (rated)", g("B_tooth_rated") + " T"),
                ("EM losses", g("P_loss_kW") + " kW  (" + g("loss_pct") + " %)"),
                ("active mass", g("m_active_kg", "{:.0f}") + " kg"), ("cost index", g("cost_usd", "{:,.0f}") + " USD"),
                ("inertia H", g("H_s") + " s"), ("runtime", f"{r.get('runtime_s', 0):.0f} s")]
        y = 0.97
        for k, v in rows:
            axt.text(0.0, y, k, fontsize=10, transform=axt.transAxes)
            axt.text(1.0, y, v, fontsize=10, ha="right", family="monospace", transform=axt.transAxes)
            y -= 0.058
        if why:
            axt.text(0.0, y - 0.01, "⚠ " + "; ".join(why), fontsize=8.5, color="#c0392b", wrap=True,
                     transform=axt.transAxes)
        # 3) heatmap filling in (cost index)
        axh = fig.add_axes([0.55, 0.42, 0.42, 0.48])
        M = np.full((len(self.poles), len(self.Ds)), np.nan)
        for rr in self.results:
            if rr["poles"] in self.poles:
                i, j = self.poles.index(rr["poles"]), _nearest(self.Ds, rr["D_bore"])
                M[i, j] = rr.get("cost_usd", np.nan)
        im = axh.imshow(M, cmap="magma_r", aspect="auto", origin="lower")
        for rr in self.results:
            i, j = self.poles.index(rr["poles"]), _nearest(self.Ds, rr["D_bore"])
            v = rr.get("cost_usd", np.nan)
            okk, dmk, _ = feasible(rr, demag_limit)
            axh.text(j, i, ("—" if np.isnan(v) else f"{v:,.0f}") + ("" if okk else "\n✗") + ("" if dmk else " D"),
                     ha="center", va="center", fontsize=8, color="k" if np.isnan(v) or v < np.nanmax(M) * 0.7 else "w")
        axh.add_patch(plt.Rectangle((_nearest(self.Ds, r["D_bore"]) - 0.5, self.poles.index(r["poles"]) - 0.5),
                                    1, 1, fill=False, ec="#00bcd4", lw=3))
        axh.set_xticks(range(len(self.Ds)), [f"{x * 1e3:.0f}" for x in self.Ds])
        axh.set_yticks(range(len(self.poles)), [f"{p}-pole" for p in self.poles])
        axh.set_xlabel("stator bore Ø [mm]")
        axh.set_title(f"Material cost index [USD] — {done}/{total} cases done (✗ not feasible, D demag risk)")
        fig.colorbar(im, ax=axh, shrink=0.85)
        # 4) trends vs bore diameter per pole count
        for n_, (key, title) in enumerate((("m_magnet_kg", "magnet mass [kg]"), ("loss_pct", "EM losses [%]"),
                                           ("Bd_min_sc", "min magnet B at s.c. [T]"), ("L_over_D", "L / D"))):
            axp = fig.add_axes([0.04 + 0.245 * n_, 0.06, 0.20, 0.26])
            for p in self.poles:
                pts = sorted((rr["D_bore"] * 1e3, rr.get(key, np.nan)) for rr in self.results if rr["poles"] == p)
                if pts:
                    xs, ys = zip(*pts)
                    axp.plot(xs, ys, "o-", ms=4, label=f"{p}p")
            if key == "Bd_min_sc":
                axp.axhline(demag_limit, color="#c0392b", ls=":", lw=1)
            if key == "L_over_D":
                for lim in LIMITS["L_over_D"]:
                    axp.axhline(lim, color="#c0392b", ls=":", lw=1)
            axp.set_title(title, fontsize=10); axp.set_xlabel("Ø bore [mm]", fontsize=9); axp.grid(alpha=.3)
            axp.tick_params(labelsize=8)
            if n_ == 0:
                axp.legend(fontsize=8)
        path = os.path.join(self.outdir, f"iter_{self.n:02d}_p{r['poles']}_D{r['D_bore'] * 1e3:.0f}.png")
        fig.savefig(path, dpi=110)
        fig.savefig(os.path.join(os.path.dirname(self.outdir), "live_progress.png"), dpi=110)
        if self.interactive:
            fig.canvas.draw_idle()
            plt.pause(0.05)
        return path


def _nearest(arr, v):
    return int(np.argmin(np.abs(np.asarray(arr) - v)))


# ----------------------------------------------------------------------------- heatmaps
def plot_heatmaps(rows, poles, Ds, outdir, demag_limit):
    import matplotlib.pyplot as plt
    idx = {(r["poles"], _nearest(Ds, r["D_bore"])): r for r in rows}
    ncol = 4
    nrow = math.ceil(len(METRICS) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow))
    best = best_design(rows, demag_limit)
    for ax, (key, title, fmt, cmap, lower_better) in zip(axs.ravel(), METRICS):
        M = np.full((len(poles), len(Ds)), np.nan)
        for (p, j), r in idx.items():
            if p in poles:
                v = r.get(key, np.nan)
                M[poles.index(p), j] = v if isinstance(v, (int, float)) else np.nan
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower")
        vmax, vmin = np.nanmax(M) if np.isfinite(M).any() else 1, np.nanmin(M) if np.isfinite(M).any() else 0
        for (p, j), r in idx.items():
            i = poles.index(p)
            v = M[i, j]
            ok, dm_ok, _ = feasible(r, demag_limit)
            txt = "—" if np.isnan(v) else fmt.format(v)
            frac = 0.5 if vmax == vmin or np.isnan(v) else (v - vmin) / (vmax - vmin)
            dark = (frac > 0.6) if not cmap.endswith("_r") else (frac > 0.6)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="w" if dark and cmap != "RdYlGn" else "k")
            if not ok:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec="#555555", lw=0))
            if not dm_ok:
                ax.text(j + 0.42, i + 0.32, "D", fontsize=6, color="#c0392b", ha="right", fontweight="bold")
        if best is not None:
            ax.plot(_nearest(Ds, best["D_bore"]), poles.index(best["poles"]), marker="*", ms=16,
                    mec="k", mfc="#00e5ff")
        ax.set_xticks(range(len(Ds)), [f"{x * 1e3:.0f}" for x in Ds], fontsize=8)
        ax.set_yticks(range(len(poles)), [f"{p}p" for p in poles], fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
    for ax in axs.ravel()[len(METRICS):]:
        ax.axis("off")
    fig.suptitle("300 kVAr / 400 V SPM synchronous condenser — sizing heatmaps  (x: stator bore Ø [mm], "
                 "y: poles;  /// not feasible,  D = demagnetisation risk at short circuit,  ★ lowest-cost feasible)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(outdir, "heatmaps_all.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    # a focused 2x2 decision chart
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for ax, key in zip(axs.ravel(), ("cost_usd", "loss_pct", "m_active_kg", "Bd_min_sc")):
        key_, title, fmt, cmap, _ = next(m for m in METRICS if m[0] == key)
        M = np.full((len(poles), len(Ds)), np.nan)
        for (p, j), r in idx.items():
            M[poles.index(p), j] = r.get(key, np.nan)
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower")
        for (p, j), r in idx.items():
            i = poles.index(p); ok, dm_ok, _ = feasible(r, demag_limit)
            ax.text(j, i, ("—" if np.isnan(M[i, j]) else fmt.format(M[i, j])) + ("" if dm_ok else "\nD"),
                    ha="center", va="center", fontsize=9)
            if not ok:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec="#555555", lw=0))
        if best is not None:
            ax.plot(_nearest(Ds, best["D_bore"]), poles.index(best["poles"]), "*", ms=20, mec="k", mfc="#00e5ff")
        ax.set_xticks(range(len(Ds)), [f"{x * 1e3:.0f}" for x in Ds]); ax.set_yticks(range(len(poles)), [f"{p}-pole" for p in poles])
        ax.set_xlabel("stator bore Ø [mm]"); ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Decision heatmaps (/// not feasible, D demag risk, ★ lowest-cost feasible)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path2 = os.path.join(outdir, "heatmaps_decision.png")
    fig.savefig(path2, dpi=140)
    plt.close(fig)
    return path, path2


def best_design(rows, demag_limit, require_demag=False):
    cand = []
    for r in rows:
        ok, dm_ok, _ = feasible(r, demag_limit)
        if ok and (dm_ok or not require_demag) and not math.isnan(r.get("cost_usd", float("nan"))):
            cand.append(r)
    if not cand:
        return None
    dm = [r for r in cand if feasible(r, demag_limit)[1]]
    return min(dm or cand, key=lambda r: r["cost_usd"])


def write_report(rows, poles, Ds, outdir, demag_limit, files, runtime):
    best = best_design(rows, demag_limit)
    feas = [r for r in rows if feasible(r, demag_limit)[0]]
    lines = [f"# Sizing study — 300 kVAr / 400 V SPM synchronous condenser\n",
             f"{len(rows)} cases: poles {poles}, stator bore {', '.join(f'{d * 1e3:.0f}' for d in Ds)} mm. "
             f"Runtime {runtime / 60:.1f} min. Demagnetisation limit {demag_limit:.2f} T.\n",
             "Feasibility: " + ", ".join(f"{k} in {v}" for k, v in LIMITS.items()) + "; magnets sized for rated kVAr.\n"]
    if best:
        lines.append(f"**Lowest material cost feasible design:** {best['poles']}-pole, Ø bore {best['D_bore'] * 1e3:.0f} mm, "
                     f"L = {best['L'] * 1e3:.0f} mm, OD {best['D_out'] * 1e3:.0f} mm, magnets {best['h_m_mm']:.1f} mm "
                     f"({best['m_magnet_kg']:.0f} kg), losses {best['loss_pct']:.2f} %, cost index {best['cost_usd']:,.0f} USD, "
                     f"demag {'OK' if feasible(best, demag_limit)[1] else 'RISK'}.\n")
    lines += [f"![decision]({os.path.basename(files[1])})\n", f"![all]({os.path.basename(files[0])})\n",
              "| poles | Ø bore | L | L/D | OD | h_m | magnet kg | loss % | Xd | I_sc pu | B_sc T | cost USD | feasible | demag |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["poles"], -r["D_bore"])):
        ok, dm_ok, why = feasible(r, demag_limit)
        f = lambda k, s="{:.2f}": s.format(r[k]) if k in r and not (isinstance(r[k], float) and math.isnan(r[k])) else "—"
        lines.append(f"| {r['poles']} | {r['D_bore'] * 1e3:.0f} | {r['L'] * 1e3:.0f} | {r['L_over_D']:.2f} | "
                     f"{r['D_out'] * 1e3:.0f} | {f('h_m_mm', '{:.1f}')} | {f('m_magnet_kg', '{:.0f}')} | {f('loss_pct')} | "
                     f"{f('Xd_pu', '{:.3f}')} | {f('I_sc_pu')} | {f('Bd_min_sc')} | {f('cost_usd', '{:,.0f}')} | "
                     f"{'✓' if ok else '✗ ' + '; '.join(why)} | {'✓' if dm_ok else '⚠'} |")
    with open(os.path.join(outdir, "sizing_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return best


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poles", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--D", type=float, nargs="+", default=list(np.linspace(0.25, 0.55, 7)),
                    help="stator bore diameters [m]")
    ap.add_argument("--jobs", type=int, default=0, help="parallel processes (0 = CPU cores / 2)")
    ap.add_argument("--solver", default="auto", choices=["auto", "gpu", "pardiso", "superlu"])
    ap.add_argument("--quick", action="store_true", help="skip the EMF/cogging rotation sweep")
    ap.add_argument("--no-window", action="store_true", help="no live window (PNG files only)")
    ap.add_argument("--demag-limit", type=float, default=0.10,
                    help="min B along magnetisation [T]: N38SH 0.10 (default), UH/EH grades e.g. -0.3")
    ap.add_argument("--out", default="results")
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args()
    poles = sorted(a.poles)
    Ds = sorted(a.D, reverse=True)                           # largest -> smallest
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "sizing_results.csv")

    if a.replot:
        with open(csv_path) as fh:
            rows = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]
        Ds = sorted({r["D_bore"] for r in rows}); poles = sorted({int(r["poles"]) for r in rows})
        for r in rows:
            r["poles"] = int(r["poles"])
        files = plot_heatmaps(rows, poles, Ds, a.out, a.demag_limit)
        write_report(rows, poles, Ds, a.out, a.demag_limit, files, 0)
        print("heatmaps:", *files)
        return

    jobs = a.jobs or max(1, (os.cpu_count() or 2) // 2)
    if jobs > 1:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cases = [(p, D, a.solver, not a.quick) for D in Ds for p in poles]
    print(f"Sizing study: {len(cases)} cases (poles {poles}, Ø bore {Ds[0] * 1e3:.0f} -> {Ds[-1] * 1e3:.0f} mm), "
          f"{jobs} parallel job(s)")
    live = LivePlot(poles, sorted(Ds), a.out, window=not a.no_window)
    live.results = []
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_run_case, c): c for c in cases}
        for k, fu in enumerate(as_completed(futs), 1):
            r, design = fu.result()
            rows.append(r); live.results.append(r)
            ok, dm_ok, why = feasible(r, a.demag_limit)
            print(f"[{k:2d}/{len(cases)}] {r['poles']}-pole Ø{r['D_bore'] * 1e3:4.0f}: L {r['L'] * 1e3:5.0f} mm, "
                  f"h_m {r.get('h_m_mm', float('nan')):5.1f} mm, magnets {r.get('m_magnet_kg', float('nan')):5.1f} kg, "
                  f"loss {r.get('loss_pct', float('nan')):4.2f} %, B_sc {r.get('Bd_min_sc', float('nan')):5.2f} T, "
                  f"cost {r.get('cost_usd', float('nan')):7,.0f}  {'OK' if ok else 'x'}{'' if dm_ok else ' D'}  "
                  f"({r.get('runtime_s', 0):.0f} s)", flush=True)
            try:
                live.update(r, design, k, len(cases), a.demag_limit)
            except Exception as e:                           # noqa: BLE001  (plotting must not stop the study)
                print("   (live plot skipped:", e, ")")
            _write_csv(rows, csv_path)
    runtime = time.time() - t0
    Ds_sorted = sorted(Ds)
    files = plot_heatmaps(rows, poles, Ds_sorted, a.out, a.demag_limit)
    best = write_report(rows, poles, Ds_sorted, a.out, a.demag_limit, files, runtime)
    print(f"\nDone in {runtime / 60:.1f} min.  Heatmaps: {files[0]}, {files[1]}")
    if best:
        print(f"Lowest-cost feasible design: {best['poles']}-pole, Ø bore {best['D_bore'] * 1e3:.0f} mm, "
              f"L {best['L'] * 1e3:.0f} mm, magnets {best['h_m_mm']:.1f} mm, cost index {best['cost_usd']:,.0f} USD")
    if live.interactive:
        live.plt.ioff()
        print("Close the live window to exit.")
        live.plt.show()


def _run_case(c):
    from sizing import evaluate
    return evaluate(c)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _write_csv(rows, path):
    keys = sorted({k for r in rows for k in r})
    first = ["poles", "D_bore", "status", "reason"]
    keys = first + [k for k in keys if k not in first]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
