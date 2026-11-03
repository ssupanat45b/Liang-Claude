# Magnet-size and air-gap optimisation — 300 kVAr IPM condenser (main dimensions fixed)

A separate folder. The main dimensions stay those of the **best-compromise design of the
size optimisation** (`Liang_IPM_BO`): 4 poles, Ø bore 376 mm, core 546 mm, the same
stator, and the magnet depth below the rotor surface. If
`../Liang_IPM_BO/results/pareto.csv` exists, the exact compromise is read from it.

Only three quantities are optimised, at mm resolution:

| Variable | Range | Step |
|---|---|---|
| magnet thickness h_m | 20–55 mm | 0.5 mm |
| magnet width w_m | 150–230 mm (must fit under the bridges and leave the q-axis rib) | 1 mm |
| air gap | 1.5–4.0 mm (bore fixed, rotor OD changes) | 0.1 mm |

## Objectives and constraints

| | |
|---|---|
| Minimise | ① EMF **THD** (line-line) ② **losses** (heat: iron + copper) ③ **cogging torque** ④ **current with cold magnets** (start-up overload) |
| Constraints | **300 ≤ Q ≤ 330 kVAr at 400 V** (hot magnets): the rating is met and the current is ≤ 1.1 pu · magnet B at short circuit ≥ 0.10 T · tooth B ≤ 2.0 T |

> At a fixed 400 V the delivered reactive power **is** the current (Q = 3·V·I). "Most kVAr"
> and "least current" therefore cannot both be optimised. The rating is a constraint, and
> the current that still depends on the magnet design (the cold-magnet start-up current)
> is minimised.

Cogging is the unskewed 2-D value, sampled over one cogging period at every mesh step.
The one-slot stator skew reduces it strongly in the real machine. THD includes the skew.

## Method: ParEGO multi-objective Bayesian optimisation (4 objectives)

* Sobol initial designs plus the base design.
* Each batch member draws a random weight vector. A GP is fitted to the augmented-Chebyshev
  scalarisation of the normalised objectives. The candidate maximising expected improvement
  × probability of feasibility is evaluated.
* The constraint GPs model Q, magnet B at short circuit and tooth B.
* Batches run in parallel, and a live figure is updated after every iteration.

## Run

```bash
pip install numpy scipy matplotlib triangle scikit-learn pypardiso
python run_opt.py --jobs 8             # base + 20 initial + 10 x 8 evaluations
python run_opt.py --resume --iters 5   # continue
python run_opt.py --final-only         # re-select / re-analyse
```

## Outputs (`results/`)

* `live_progress.png` and `iterations/iter_NN.png`: live figures
* `opt_report.md`: base vs. best designs (best compromise, best THD, best losses, best
  cogging, best current) and the whole Pareto set
* `radar_selected.png`: normalised objectives of the selected designs against the base
* `design_compromise/`: full FEM study and design sheet of the recommended variant
* `all_evals.csv`, `pareto.csv`

---

### สรุป (ภาษาไทย)

* คงขนาดเครื่องตามแบบ best compromise (4 ขั้ว, Ø 376 mm, L 546 mm)
* ปรับเฉพาะความหนาแม่เหล็ก (ขั้นละ 0.5 mm) ความกว้างแม่เหล็ก (ขั้นละ 1 mm) และ air gap (ขั้นละ 0.1 mm)
* เป้าหมาย: THD ต่ำ ความร้อนต่ำ cogging ต่ำ และกระแสตอนแม่เหล็กเย็นต่ำ
* เงื่อนไข: Q ต้องได้ 300–330 kVAr ที่ 400 V
* วิธีรัน: `python run_opt.py --jobs 8`
