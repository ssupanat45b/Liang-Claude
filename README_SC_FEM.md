# Synchronous Condenser — 2-D Electromagnetic FEM

A self-contained Python FEM model of the cross-section of a round-rotor
synchronous condenser. It needs only numpy, scipy, and matplotlib.

**Example rating:** 100 MVAr, 13.8 kV, 50 Hz, 2-pole, 3000 rpm.

## Quick start

```bash
pip install -r requirements.txt
python run_sc_fem.py              # full study, about 1–2 min
python run_sc_fem.py --quick      # coarser sweeps
python run_sc_fem.py --S 160 --V 15.75 --length 3.4 --out results_160
python -m unittest discover -s tests -v
python draw_cross_section.py      # dimensioned stator/rotor drawing only, about 2 s
```

The run writes `results/report.md`, `results/results.json`, and seven figures.

## What it computes

| Analysis | Method |
|---|---|
| Field solution | Nonlinear magnetostatic A<sub>z</sub>, linear triangles, Newton–Raphson with line search, B–H curves for the lamination and the solid rotor forging |
| Open-circuit characteristic, I<sub>f0</sub>, saturation factor | Field-current sweep, flux linkage → EMF |
| X<sub>d</sub>, X<sub>q</sub>, SCR | dq flux linkages from FEM, plus end-winding leakage |
| Rated over- and under-excited operation, V-curve | Root-finding for the field current at rated V and I (armature MMF on the ±d axis) |
| EMF waveform, THD, cogging torque | Rotor rotation using a sliding air-gap interface, with no re-meshing |
| Air-gap flux density and its harmonics | Gap B<sub>r</sub> and B<sub>t</sub> |
| Torque vs. current angle | Arkkio Maxwell-stress torque |
| Losses | Steinmetz iron loss from peak B per element; armature and field I²R |

## Files

* `sc_fem2d/machine.py`: design data (`SCDesign`), polar mesh generator, winding layout
* `sc_fem2d/materials.py`: nonlinear B–H materials
* `sc_fem2d/solver.py`: FEM assembly, Newton–Raphson, flux linkage, torque
* `sc_fem2d/analysis.py`: condenser performance analyses
* `sc_fem2d/plots.py`, `run_sc_fem.py`: figures and report
* `sc_fem2d/dimensions.py`, `draw_cross_section.py`: dimensioned cross-section drawing
* `tests/test_sc_fem2d.py`: verification tests (reciprocity, EMF formula, virtual work against Arkkio torque, linearity, mesh and winding checks)

To change the machine, edit the defaults of `SCDesign` in `sc_fem2d/machine.py`.
These cover dimensions, slots, turns, air gap, and the end-winding reactance.

## Limitations

The model is 2-D and magnetostatic: it has no eddy currents and no end-region field.
Slots are drawn as annular sectors, and the layout generator handles 2-pole
machines only. The phase EMF contains a 3rd harmonic (about 3.7%), which cancels in
the line-to-line voltage.

---

### สรุป (ภาษาไทย)

โปรแกรม FEM แม่เหล็กไฟฟ้า 2 มิติ (nonlinear magnetostatic) สำหรับ Synchronous Condenser
แบบโรเตอร์ทรงกระบอก 2 ขั้ว ใช้พิกัดตัวอย่าง 100 MVAr, 13.8 kV, 50 Hz
รันด้วยคำสั่ง `python run_sc_fem.py` แล้วจะได้รายงาน `results/report.md` พร้อมรูปต่อไปนี้

* เส้นแรงแม่เหล็กและค่า |B|
* กราฟ OCC และ V-curve
* ค่า Xd, Xq และ SCR
* กระแสฟีลด์ที่พิกัด
* ขีดความสามารถ Q ด้าน under-excited
* รูปคลื่น EMF และค่า THD
* แรงบิดเทียบกับมุมกระแส
* แบบภาพตัดขวาง stator/rotor พร้อมขนาด (รันเฉพาะรูปนี้ได้ด้วย `python draw_cross_section.py`)
* การประมาณค่ากำลังสูญเสีย

หากต้องการเปลี่ยนขนาดเครื่อง ให้แก้ค่าใน `SCDesign` (ไฟล์ `sc_fem2d/machine.py`)
