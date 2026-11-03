#!/usr/bin/env python3
"""Benchmark the linear-solver backends (GPU / MKL PARDISO / SuperLU) on this computer.

    python benchmark_solver.py                 # meshes coarse, normal, fine, ultra
    python benchmark_solver.py --mesh fine

For every mesh preset one nonlinear rated-current solve (magnets + armature) is
timed with each available backend; the fastest is recommended for run_pm_fem.py.
"""
import argparse
import time

import numpy as np

from pm_fem2d import solver
from pm_fem2d.analysis import PMAnalysis
from pm_fem2d.machine import apply_mesh_preset, design_preset


def available():
    out = ["superlu"]
    if solver._try_pardiso():
        out.insert(0, "pardiso")
    if solver._try_gpu():
        out.insert(0, "gpu")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", nargs="+", default=["coarse", "normal", "fine", "ultra"])
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    backs = available()
    print("available backends:", backs)
    rows = []
    for mesh in a.mesh:
        d = apply_mesh_preset(design_preset(), mesh)
        res = {}
        for b in backs:
            solver._state.update(backend=None, fallbacks=0, cg_iters=0)
            name = solver.set_backend(b)
            an = PMAnalysis(d, verbose=False)
            an.find_axes()
            cur = an.reactive_currents(d.I_rated)
            an.model.solve(cur, 1.0)                            # warm-up (GPU init, caches)
            t = []
            for _ in range(a.repeat):
                t0 = time.perf_counter()
                sol = an.model.solve(cur, 1.0)
                t.append(time.perf_counter() - t0)
            st = solver.solver_stats()
            res[b] = (min(t), sol.psi_abc, st["gpu_fallbacks"])
            print(f"  mesh {mesh:6s} {an.res.mesh_stats['nodes']:7d} nodes  {name:32s} "
                  f"{min(t):7.2f} s / nonlinear solve  ({sol.iterations} NR it, fallbacks {st['gpu_fallbacks']})")
        ref = res["superlu"][1]
        for b, (t, psi, fb) in res.items():
            assert np.allclose(psi, ref, rtol=1e-6, atol=1e-9), f"{b} result differs"
        best = min(res, key=lambda k: res[k][0])
        rows.append((mesh, best, {k: v[0] for k, v in res.items()}))
    print("\nRecommendation (fastest backend per mesh):")
    for mesh, best, t in rows:
        print(f"  --mesh {mesh:6s} -> --solver {best:8s}  " + "  ".join(f"{k}: {v:.2f}s" for k, v in t.items()))


if __name__ == "__main__":
    main()
