# PM Synchronous Condenser (SPMSM) — 2-D Electromagnetic FEM

A separate project from the wound-field model in the parent folder (`sc_fem2d/`).
The rotor carries **surface-mounted, radially magnetised NdFeB magnets** instead of
a field winding.

**Design `pm300`:** 300 kVAr, 400 V, 50 Hz, 4-pole, 1500 rpm. The stator is the same
as the wound-field `factory300` design (Ø 400/600 mm, 450 mm core, 48 skewed slots).
The magnets are 23 mm thick, sized so the machine delivers 300 kVAr at 400 V.

## Quick start

```bash
pip install -r requirements.txt
python benchmark_solver.py          # compare solver speed on this PC
python run_pm_fem.py                # full study -> results_pm300/
python run_pm_fem.py --mesh fine    # final values (about 1 % from FEMM)
python run_pm_fem.py --size-magnets # re-size the magnet thickness first
python draw_pm_cross_section.py     # dimensioned drawing only (~2 s)
python sweep_pm.py                  # magnet thickness 16–28 mm
python femm_compare_pm.py           # cross-check with FEMM 4.2 (pyfemm)
python -m unittest discover -s tests -v
```

## Solver backends (`--solver`)

| Backend | Option | Notes |
|---|---|---|
| SciPy SuperLU | `--solver superlu` | always available; the default (`auto`) unless pypardiso is installed |
| Intel MKL PARDISO | `--solver pardiso` | `pip install pypardiso`; multithreaded CPU; picked by `auto` when installed |
| GPU (NVIDIA) | `--solver gpu` | `pip install "cupy-cuda12x[ctk]"`; Jacobi-preconditioned CG on the GPU; used only on request |

**Measured on Core Ultra 7 265K + RTX 4060** (one nonlinear solve):

| Mesh | Nodes | CPU SuperLU | GPU (CuPy PCG) |
|---|---|---|---|
| coarse | 10.7k | 0.25 s | 1.65 s |
| normal | 13.0k | 0.27 s | 2.29 s |
| fine | 47.6k | 1.47 s | 4.49 s |
| ultra | 99.1k | 4.68 s | 8.06 s |

For 2-D models of this size the sparse direct CPU solver wins: the GPU has to run
about 1000 CG iterations per Newton step on an ill-conditioned magnetic matrix. Use the
CPU and parallel sweeps (`--jobs 6–8`). The GPU option is kept for experiments.

**Accuracy:** with the 3–3.5 mm air gap of the 300 kVAr designs, use `--mesh fine` for
final values. Against FEMM 4.2 the flux linkage differs by −4.7 % (normal), −2.1 % (fine)
and −1.3 % (ultra).

## Outputs (`results_pm300/`)

| File | Content |
|---|---|
| 00_cross_section_dimensions.png | dimensioned stator / SPM rotor drawing |
| 01–04 | mesh and flux plots: open circuit, rated, short circuit |
| 05_airgap_flux.png | air-gap flux density and its harmonics |
| 06_q_v_characteristic.png | reactive power and current vs terminal voltage |
| 07_emf_torque.png | EMF waveform, harmonics, torque vs angle, cogging |
| 08_design_summary.png | all key parameters on one sheet |
| 09_magnet_demag.png | magnet working point (demagnetisation check) |
| report.md / results.json | report and all numbers |

## Important differences from a wound-field condenser

* **Q cannot be controlled by excitation.** It follows the voltage: Q ≈ 3V(E−V)/Xs,
  giving about −6 kVAr for each +1 % of voltage.
* **The EMF is fixed:** E0 ≈ 1.32 pu with hot magnets and 1.41 pu cold. Expect this
  overvoltage after disconnection.
* **Cold magnets give more kVAr and more current** (about 1.3 pu at a cold start).
* **Short circuit:** the steady current is about 4 pu, and the magnet working point
  drops near the knee. Use a high-coercivity grade (UH/EH), or thicker magnets, plus
  fast protection.
