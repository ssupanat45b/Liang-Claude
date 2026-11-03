# Sizing study — 300 kVAr / 400 V SPM synchronous condenser

A separate folder that sweeps the **stator/rotor size** (stator bore Ø 550 → 250 mm,
7 steps) and the **pole number** (2, 4, 6, 8) and compares the 28 designs on heatmaps.
It uses its own copy of the validated PM FEM package (`pm_fem2d/`, checked within
≤ 1 % of FEMM).

## Run

```bash
pip install numpy scipy matplotlib pypardiso
python run_sizing.py --jobs 8                 # full study; a live window updates after every case
python run_sizing.py --jobs 8 --quick         # skip the EMF/cogging rotation sweep (about 2x faster)
python run_sizing.py --demag-limit -0.3       # judge demagnetisation for a UH/EH magnet grade
python run_sizing.py --poles 4 6 --D 0.35 0.40 0.45
python run_sizing.py --replot                 # redraw heatmaps from results/sizing_results.csv
```

## What happens in each case (`sizing.py`)

1. The geometry follows from (poles, Ø bore):
   * q = 3 slots/pole/phase (Qs = 9p), coil pitch 7/9
   * core length from the output coefficient (C0 = 2.78 kVA·min/m³)
   * turns from 400 V, then the core length is corrected for the integer turns
   * slot depth from J = 3.7 A/mm² and a fill factor of 0.45
   * stator yoke at 1.4 T
   * gap = 0.625 % of D plus a 1 mm sleeve
   * end-winding leakage scaled with the end-winding length
2. Magnet thickness by FEM root-finding, so the machine delivers **300 kVAr at 400 V**
   (magnets at 80 °C).
3. Performance: EMF, Xd/Xq, Q–V slope, steady short circuit, demagnetisation, THD,
   cogging, losses, masses, inertia and a material cost index (NdFeB 60, Cu 10,
   steel 2 USD/kg).

## Outputs (`results/`)

| File | Content |
|---|---|
| `live_progress.png`, `iterations/iter_NN_pX_DYYY.png` | live figure saved after every case |
| `heatmaps_decision.png` | cost, losses, active mass, demagnetisation margin |
| `heatmaps_all.png` | 16 quantities on a poles × bore grid |
| `sizing_results.csv`, `sizing_report.md` | all numbers, a ranking and the best design |

Feasibility (`///` hatched otherwise):
* 0.3 ≤ L/D ≤ 3.0
* stator tooth B at rated ≤ 2.0 T
* current with cold magnets ≤ 1.6 pu
* magnets able to reach 300 kVAr

**D** marks a demagnetisation risk at short circuit. **★** marks the lowest-cost feasible
design.

---

### สรุป (ภาษาไทย)

* จำลอง 28 กรณี: Ø bore 550 → 250 mm (7 ขนาด) × 2/4/6/8 ขั้ว ทุกกรณีให้ 300 kVAr ที่ 400 V
* ระหว่างรัน หน้าต่างกราฟจะอัปเดตทุกครั้งที่จำลองเสร็จหนึ่งกรณี แสดงภาพตัดขวาง พารามิเตอร์สำคัญ heatmap ที่ค่อยๆ เติม และกราฟแนวโน้ม
* ภาพของทุกรอบบันทึกไว้ใน `results/iterations/`
* เมื่อรันเสร็จจะได้ heatmap เปรียบเทียบใน `results/heatmaps_decision.png` และ `results/heatmaps_all.png`
