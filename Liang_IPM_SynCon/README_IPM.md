# IPM Synchronous Condenser — 2-D Electromagnetic FEM + Sizing Study

A separate project from the SPM folders. The rotor carries **buried (interior) NdFeB
magnets**. Two shapes are available (`magnet_shape`):

| Shape | Description | Mesh |
|---|---|---|
| `"rect"` (default) | **flat rectangular bar**, one per pole, **parallel** magnetised; pentagon flux barriers at both ends, saturable bridges under the surface | rotor disk: unstructured constrained Delaunay (`triangle`), joined to the structured air-gap rings |
| `"arc"` | buried arc magnet, radially magnetised, pole shoe + annular-sector barriers | structured polar mesh |

The iron between the poles makes the rotor salient (Lq > Ld). No retaining sleeve is needed.

**Design `ipm300` (flat magnets):** 300 kVAr, 400 V, 4-pole, 1500 rpm, same stator as
`pm300`/`factory300` (Ø 400/600 mm, 450 mm core, 48 skewed slots), 2.5 mm air gap.
Each magnet is **213 × 41 mm** (width × thickness), its outer face 36 mm below the rotor
surface on the pole axis, with 1.2 mm bridges and 4° barrier openings.

The magnet width is the largest that still
* keeps the outer corner 3 mm under the bridge,
* leaves a q-axis iron rib ≥ 3 mm to the neighbouring pole at the inner corner,
* leaves the same rib after the barrier opening.

The thickness was found by FEM for 300 kVAr. Q is *not* monotonic in thickness, because a
thicker bar has to get narrower.

## Run

```bash
pip install numpy scipy matplotlib triangle pypardiso
python run_ipm_fem.py                     # full study -> results_ipm300/
python run_ipm_fem.py --size-magnets      # re-size the magnet thickness first
python draw_ipm_cross_section.py          # dimensioned drawing only
python sweep_ipm.py --jobs 8              # magnet thickness 18–34 mm
python run_sizing.py --jobs 8             # 28 cases: Ø bore 550->250 mm x 2/4/6/8 poles, live plots + heatmaps
python femm_compare_ipm.py                # cross-check with FEMM 4.2 (pyfemm)
python -m unittest discover -s tests -v   # 19 verification tests (flat + arc)
```

## SPM vs IPM-arc vs IPM-flat (same stator, 300 kVAr at 400 V)

| | SPM (pm300) | IPM, arc magnets | **IPM, flat magnets (ipm300)** |
|---|---|---|---|
| Magnet size / mass | 23 mm arc / 71 kg | 29.5 mm arc / 88 kg | **213 × 41 mm bar / 118 kg** |
| Xd / Xq | 0.30 / 0.32 pu | 0.37 / 1.13 pu | 0.33 / 1.51 pu |
| Min magnet B at short circuit | 0.07 T ⚠ | 0.27 T ✓ | 0.10 T (at the limit) |
| Sleeve | needed | not needed | not needed |
| Magnet manufacture | curved segments | curved segments | **simple rectangular blocks** |
| EM losses | 5.2 kW | 5.2 kW | 5.2 kW |

Flat bars are the cheapest magnets to make, but on a 4-pole, 400 mm rotor one bar per pole
is narrower (213 mm) than an arc magnet (about 240 mm of arc). The flat design therefore
needs about 35 % more NdFeB than the arc IPM. With more poles the flat bar approaches the
arc shape; see the sizing heatmaps.

## Notes

* Nonlinear iron is essential for IPM: with linear iron the unsaturated bridges
  short-circuit the magnets (this is checked by a test).
* Solver backends and GPU: see `benchmark_solver.py`. MKL PARDISO is the fastest on CPU.

---

### สรุป (ภาษาไทย)

โฟลเดอร์ใหม่สำหรับ synchronous condenser แบบแม่เหล็กฝังใน (IPMSM) ขนาด 300 kVAr / 400 V / 4 ขั้ว

* แม่เหล็กแบบแท่งสี่เหลี่ยม (ค่าเริ่มต้น) ขนาด 213 × 41 mm ต่อขั้ว magnetize แบบขนาน ผลิตง่ายที่สุด แต่ใช้แม่เหล็ก 118 kg
* แม่เหล็กแบบโค้งฝัง (`magnet_shape="arc"`) หนา 29.5 mm ใช้ 88 kg และทนการลัดวงจรได้ดีที่สุด (0.27 T)
* ไม่ต้องใช้ sleeve
* มีการ sizing 28 กรณีพร้อมกราฟ live และ heatmap เหมือนโฟลเดอร์ `Liang_PM_Sizing`
