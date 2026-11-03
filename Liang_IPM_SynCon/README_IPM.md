# IPM Synchronous Condenser — 2-D Electromagnetic FEM + Sizing Study

A separate project from the SPM folders. The rotor carries **buried (interior) NdFeB
arc magnets**: one radially magnetised magnet per pole, under an iron pole shoe, with an
air flux barrier at each magnet end and a thin iron bridge at the rotor surface. The
iron between the poles makes the rotor salient (Lq > Ld). No retaining sleeve is needed.

> The structured polar mesh cannot model V-shaped magnets, so this is the single-layer
> buried-arc IPM topology.

**Design `ipm300`:** 300 kVAr, 400 V, 4-pole, 1500 rpm. It uses the same stator as
`pm300`/`factory300` (Ø 400/600 mm, 450 mm core, 48 skewed slots), a 2.5 mm air gap,
5 mm pole shoe, 1.2 mm bridges, an 80 % magnet arc and **29.5 mm magnets** (sized for
300 kVAr at 400 V).

## Run

```bash
pip install numpy scipy matplotlib pypardiso
python run_ipm_fem.py                     # full study -> results_ipm300/
python run_ipm_fem.py --size-magnets      # re-size the magnet thickness first
python draw_ipm_cross_section.py          # dimensioned drawing only
python sweep_ipm.py --jobs 8              # magnet thickness 18–34 mm
python run_sizing.py --jobs 8             # 28 cases: Ø bore 550->250 mm x 2/4/6/8 poles, live plots + heatmaps
python femm_compare_ipm.py                # cross-check with FEMM 4.2 (pyfemm)
python -m unittest discover -s tests -v   # 10 verification tests
```

## IPM vs SPM (same stator, 300 kVAr at 400 V)

| | SPM (pm300) | IPM (ipm300) |
|---|---|---|
| Magnet thickness / mass | 23 mm / 71 kg | 29.5 mm / 88 kg |
| Xd / Xq | 0.30 / 0.32 pu | 0.37 / 1.13 pu (saliency 3.1) |
| Min magnet B at short circuit | 0.07 T ⚠ | **0.27 T ✓** |
| Retaining sleeve | needed | not needed (bridges hold the magnets) |
| EM losses | 5.2 kW | 5.2 kW |

The pole shoe carries the armature d-axis flux around the magnets, so they are much
better protected against demagnetisation. The price is a higher Xd (more EMF needed for
the same kVAr) and bridge leakage, which together mean about 25 % more magnet material.
The bridges saturate by design (about 2.3 T); the rotor-yoke flux check excludes them.

## Notes

* Nonlinear iron is essential for IPM: with linear iron the unsaturated bridges
  short-circuit the magnets (this is checked by a test).
* Solver backends and GPU: see `benchmark_solver.py`. MKL PARDISO is the fastest on CPU.

---

### สรุป (ภาษาไทย)

โฟลเดอร์ใหม่สำหรับ synchronous condenser แบบแม่เหล็กฝังใน (IPMSM) ขนาด 300 kVAr / 400 V / 4 ขั้ว

* แม่เหล็กหนา 29.5 mm และมี pole shoe ปกป้อง จึงทนการลัดวงจรได้ดีกว่าแบบ SPM มาก (B ต่ำสุดในแม่เหล็ก 0.27 T เทียบกับ 0.07 T)
* ไม่ต้องใช้ sleeve
* แลกกับการใช้แม่เหล็กมากขึ้นประมาณ 25 %
* มีการ sizing 28 กรณีพร้อมกราฟ live และ heatmap เหมือนโฟลเดอร์ `Liang_PM_Sizing`
