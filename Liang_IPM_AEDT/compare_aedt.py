#!/usr/bin/env python3
"""Compare ANSYS Maxwell 2D transient results (exported CSV) with the Python FEM.

In AEDT: right-click each report (Torque, Flux linkage, Induced voltage, Winding current)
> Export...  > save as .csv into  aedt_model\\   then run

    python compare_aedt.py                     # all aedt_model/*.csv, rated case
    python compare_aedt.py --load noload       # EMF / THD / cogging of the no-load model
    python compare_aedt.py a.csv b.csv         # explicit files

Only the last full electrical period is evaluated (the first one contains the start-up
transient).  Phasors are fundamentals from an FFT over that period.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np

SCALE = {"": 1, "s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
         "wb": 1, "mwb": 1e-3, "uwb": 1e-6, "v": 1, "mv": 1e-3, "kv": 1e3,
         "a": 1, "ma": 1e-3, "ka": 1e3, "newtonmeter": 1, "nm": 1, "n-m": 1, "mnewtonmeter": 1e-3,
         "knewtonmeter": 1e3, "w": 1, "kw": 1e3, "mw": 1e-3}


def parse_header(h):
    """'FluxLinkage(PhaseA) [mWb] - Ipk=...' -> ('FluxLinkage(PhaseA)', 1e-3)."""
    m = re.match(r'\s*"?([^\[\]"]+?)\s*\[([^\]]*)\]', h)
    if not m:
        return h.strip().strip('"'), 1.0
    unit = m.group(2).strip().lower().replace(" ", "")
    if unit not in SCALE:
        print(f"   note: unknown unit '{m.group(2)}' in '{h.strip()}' - taken as SI")
    return m.group(1).strip(), SCALE.get(unit, 1.0)


def read_csvs(files):
    data = {}
    for f in files:
        with open(f, newline="", encoding="utf-8-sig", errors="replace") as fh:
            rows = list(csv.reader(fh))
        if len(rows) < 3:
            continue
        heads = [parse_header(h) for h in rows[0]]
        arr = np.array([[float(x) for x in r[:len(heads)]] for r in rows[1:] if r and r[0].strip()], float)
        t = arr[:, 0] * heads[0][1]
        for j, (name, sc) in enumerate(heads[1:], start=1):
            data[name] = (t, arr[:, j] * sc)
        print(f"   {os.path.basename(f)}: {', '.join(n for n, _ in heads[1:])}")
    return data


def last_period(t, y, f):
    T = 1.0 / f
    if t[-1] - t[0] < T * 0.999:
        raise ValueError("less than one electrical period of data")
    t0 = t[-1] - T
    n = 360
    tu = np.linspace(t0, t[-1], n, endpoint=False)
    return tu, np.interp(tu, t, y)


def harmonics(y):
    c = np.fft.rfft(y) / len(y) * 2
    c[0] /= 2
    return c                                           # c[1] = fundamental complex amplitude


def thd(c, nmax=40):
    h = np.abs(c[2:nmax + 1])
    return float(np.sqrt(np.sum(h**2)) / abs(c[1])) if abs(c[1]) > 0 else float("nan")


def find(data, key, ph):
    for name in data:
        n = name.replace(" ", "").lower()
        if n.startswith(key.lower()) and f"phase{ph.lower()}" in n:
            return data[name]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--load", default="rated", choices=["rated", "noload"])
    ap.add_argument("--dir", default="aedt_model")
    a = ap.parse_args()
    files = a.files or sorted(glob.glob(os.path.join(a.dir, "*.csv")))
    if not files:
        raise SystemExit(f"no CSV files - export the AEDT reports into {a.dir}\\ first")
    exp_path = os.path.join(a.dir, "aedt_expected.json")
    ref = json.load(open(exp_path)) if os.path.exists(exp_path) else {}
    if ref and ref.get("load") != a.load:
        print(f"   note: {exp_path} was written for load='{ref.get('load')}' - rerun "
              f"'python build_aedt.py --script-only --load {a.load}' to refresh it")
    f = 50.0
    print("reading:")
    data = read_csvs(files)

    out = {}
    psi = {p: find(data, "FluxLinkage", p) for p in "ABC"}
    emf = {p: find(data, "InducedVoltage", p) for p in "ABC"}
    cur = {p: find(data, "Current", p) for p in "ABC"}
    trq = next((v for k, v in data.items() if "torque" in k.lower()), None)

    if psi["A"] is not None:
        c = harmonics(last_period(*psi["A"], f)[1])
        out["flux_linkage_peak_Wb"] = abs(c[1])
    if emf["A"] is not None:
        y = last_period(*emf["A"], f)[1]
        c = harmonics(y)
        out["induced_voltage_line_rms_V"] = math.sqrt(3) * abs(c[1]) / math.sqrt(2)
        out["thd_phase_pct"] = 100 * thd(c)
        if emf["B"] is not None:                      # line-line THD (triplen harmonics cancel)
            yb = last_period(*emf["B"], f)[1]
            out["thd_line_pct"] = 100 * thd(harmonics(y - yb))
    if cur["A"] is not None and abs(harmonics(last_period(*cur["A"], f)[1])[1]) > 1e-3 and emf["A"] is not None:
        P = Q = 0.0
        I = 0.0
        for p in "ABC":
            if emf[p] is None or cur[p] is None:
                continue
            v = harmonics(last_period(*emf[p], f)[1])[1]
            i = harmonics(last_period(*cur[p], f)[1])[1]
            s = 0.5 * v * np.conj(i)                  # motor (load) convention of AEDT
            P += s.real
            Q += s.imag
            I = abs(i) / math.sqrt(2)
        out["I_rms_A"] = I
        out["P_kW_2D"] = P / 1e3
        out["Q_delivered_kVAr_2D"] = -Q / 1e3         # Q < 0 absorbed = Q delivered to the grid
    if trq is not None:
        tq = last_period(*trq, f)[1]
        out["torque_mean_Nm"] = float(tq.mean())
        out["torque_pkpk_Nm"] = float(tq.max() - tq.min())
        if a.load == "noload":
            out["cogging_peak_Nm"] = 0.5 * out["torque_pkpk_Nm"]

    e = dict(ref.get("expected", {}))
    if "I_rms" in ref:
        e.setdefault("I_rms_A", abs(ref["I_rms"]))
        if a.load == "rated" and "torque_rated_Nm" in ref:
            e["torque_mean_Nm"] = ref["torque_rated_Nm"]
        if a.load == "rated" and "induced_voltage_line_rms_V" in e:
            e["Q_delivered_kVAr_2D"] = math.sqrt(3) * e["induced_voltage_line_rms_V"] * abs(ref["I_rms"]) / 1e3
    print(f"\n{'quantity':32s} {'AEDT':>12s} {'Python FEM':>12s} {'diff':>8s}")
    for k, v in out.items():
        r = e.get(k)
        if isinstance(r, (int, float)) and abs(r) > 1e-9:
            print(f"{k:32s} {v:12.4g} {r:12.4g} {100 * (v - r) / r:+7.1f}%")
        else:
            print(f"{k:32s} {v:12.4g} {'-':>12s}")
    print("\nnotes: '_2D' = terminal quantities of the 2-D model (no end-winding leakage, no resistance);"
          "\n       the rated 305 kVAr of the Python design includes the end-winding reactance."
          "\n       Python THD/cogging values are in ../Liang_IPM_MagGap results (unskewed cogging).")
    with open(os.path.join(a.dir, f"aedt_compare_{a.load}.json"), "w") as fh:
        json.dump(dict(aedt=out, python=e), fh, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = [("Flux linkage [Wb]", psi), ("Induced voltage [V]", emf), ("Current [A]", cur)]
        groups = [g for g in groups if g[1]["A"] is not None]
        n = len(groups) + (trq is not None)
        fig, ax = plt.subplots(n, 1, figsize=(9, 2.6 * n), sharex=True, squeeze=False)
        for i, (lab, g) in enumerate(groups):
            for p, col in zip("ABC", ("tab:red", "goldenrod", "tab:blue")):
                if g[p] is not None:
                    ax[i, 0].plot(g[p][0] * 1e3, g[p][1], color=col, label=f"Phase{p}")
            ax[i, 0].set_ylabel(lab)
            ax[i, 0].legend(loc="upper right", fontsize=8)
            ax[i, 0].grid(alpha=0.3)
        if "flux_linkage_peak_Wb" in e and groups and groups[0][0].startswith("Flux"):
            for s in (1, -1):
                ax[0, 0].axhline(s * e["flux_linkage_peak_Wb"], ls="--", color="k", lw=0.8)
        if trq is not None:
            ax[-1, 0].plot(trq[0] * 1e3, trq[1], color="k")
            ax[-1, 0].set_ylabel("Torque [Nm]")
            ax[-1, 0].grid(alpha=0.3)
        ax[-1, 0].set_xlabel("time [ms]")
        fig.suptitle(f"AEDT Maxwell 2D results ({a.load}) - dashed: Python FEM flux-linkage amplitude")
        fig.tight_layout()
        pth = os.path.join(a.dir, f"aedt_compare_{a.load}.png")
        fig.savefig(pth, dpi=130)
        print("figure:", pth)
    except Exception as ex:                            # noqa: BLE001
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
