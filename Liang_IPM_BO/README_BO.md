# Multi-objective Bayesian optimisation — 300 kVAr IPM synchronous condenser (flat magnets)

A separate folder that finds the **lightest and most suitable** 300 kVAr / 400 V IPM
condenser with **rectangular (flat-bar) buried magnets**. It uses constrained
multi-objective Bayesian optimisation (MOBO) on top of the validated nonlinear FEM
(`ipm_fem2d/`, copied from `Liang_IPM_SynCon`).

## Problem

| | |
|---|---|
| Variables | poles **{2, 4, 6, 8}**, bore **Ø 250–550 mm**, core length **100–1200 mm**, magnet **thickness 8–70 mm**, magnet **depth** 6–20 % of the pole pitch, magnet **width** 60–100 % of the widest that fits |
| Objective 1 | **active mass** (stator + rotor iron + magnets + copper), minimised (analytic, exact) |
| Objective 2 | **EM losses** (iron from FEM + armature copper), minimised |
| Constraints | 300 ≤ Q(400 V) ≤ 330 kVAr · magnet B at a 3-phase short circuit ≥ 0.10 T · stator tooth B (rated) ≤ 2.0 T · 0.3 ≤ L/D ≤ 3.0 · valid rotor geometry (bridges, q-axis rib) |
| Derived (sizing rules) | q = 3 winding with 7/9 pitch, turns from 400 V, slot depth from 3.7 A/mm², yoke at 1.4 T, gap 0.625 % of D, bridges 0.3 % of D |

## Method (`mobo.py`)

* **Sobol initial design**, spread evenly over the 4 pole counts. Only geometrically
  valid designs are sampled (cheap analytic check).
* **Gaussian processes** (Matern 5/2, ARD, noise) model losses (log scale), Q, magnet B at
  short circuit and tooth B. The GP input is the 5 continuous variables plus a one-hot
  pole number.
* **Acquisition:** expected hypervolume improvement (Monte-Carlo) × probability of
  feasibility. The mass objective is exact, so only the loss objective carries uncertainty.
* **Batches of q designs per iteration** (greedy, Kriging-believer front update, distance
  penalty), evaluated by FEM **in parallel**.
* Candidates per iteration: Sobol points over the whole space plus local perturbations
  around feasible designs and around the designs closest to the Q target (including
  neighbouring pole counts).

## Run

```bash
pip install numpy scipy matplotlib triangle scikit-learn pypardiso
python run_bo.py --jobs 8                       # 32 initial + 12 x 8 = 128 FEM evaluations
python run_bo.py --jobs 8 --iters 20            # longer search
python run_bo.py --resume --iters 5             # continue a previous run
python run_bo.py --final-only                   # re-select and re-analyse from results/all_evals.csv
```

A live window is updated after every iteration and also saved as
`results/iterations/iter_NN.png`. It shows:
* the objective space with the Pareto front and the latest batch
* hypervolume convergence
* the cross-section and key data of the current best compromise
* the explored design space
* Q against magnet thickness

## Outputs (`results/`)

| File | Content |
|---|---|
| `pareto_front.png`, `pareto.csv` | Pareto-optimal designs |
| `bo_report.md` | named designs (lightest, lowest loss, best compromise = knee, lowest 10-year cost) and the full front |
| `design_compromise/`, `design_lightest/` | full FEM study (cross-section drawing, flux plots, Q–V, demagnetisation, design sheet) |
| `all_evals.csv` | every FEM evaluation |

The **best compromise** is the knee of the front: the design closest to the ideal
point after normalising both objectives. The **lifetime-cost** pick adds material cost to
10 years of loss energy (0.12 USD/kWh, 8000 h/y, 6 %).

---

### สรุป (ภาษาไทย)

**ตัวแปรออกแบบ:**
* จำนวนขั้ว 2/4/6/8
* Ø bore
* ความยาวแกน
* ขนาดแม่เหล็กสี่เหลี่ยม (หนา ลึก กว้าง)

**เป้าหมาย:**
* ① น้ำหนักเบาที่สุด
* ② กำลังสูญเสียต่ำที่สุด

**เงื่อนไข:**
* Q 300–330 kVAr ที่ 400 V
* แม่เหล็กไม่เสื่อมตอนลัดวงจร
* B ฟัน ≤ 2 T
* L/D อยู่ในช่วง 0.3–3

**วิธีรัน:** `python run_bo.py --jobs 8` กราฟ live จะอัปเดตทุก iteration

**ผลลัพธ์:**
* Pareto front
* แบบที่เบาที่สุด
* แบบที่สมดุลที่สุด (knee)
* แบบที่ต้นทุน 10 ปีต่ำที่สุด
* การวิเคราะห์ FEM เต็มรูปแบบของแบบที่แนะนำ
