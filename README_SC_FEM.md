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
python sweep_design.py            # design sweep (default: air gap 50/60/70 mm)
python femm_compare.py            # cross-check against FEMM 4.2 (Windows, needs pyfemm)
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
* `sweep_design.py`: parallel parameter study of any `SCDesign` field
* `femm_compare.py`: builds the same model in FEMM 4.2 and compares ψ, E, B₁, torque and Xd
* `tests/test_sc_fem2d.py`: verification tests (reciprocity, EMF formula, virtual work against Arkkio torque, linearity, mesh and winding checks)

To change the machine, edit the defaults of `SCDesign` in `sc_fem2d/machine.py`.
These cover dimensions, slots, turns, air gap, and the end-winding reactance.

## Run settings (example: Intel Core Ultra 7 265K, 32 GB RAM, RTX 4060)

| Setting | Option | Recommendation |
|---|---|---|
| Mesh density | `--mesh coarse/normal/fine/ultra` | `normal` for design work; `fine` (74k elements) for final values; `ultra` (162k) for a mesh-convergence check |
| Parallel sweep | `sweep_design.py --jobs N` | `--jobs 6` to `8` on 20 cores; each case uses one core and under 1 GB RAM |
| Faster solver | `pip install pypardiso` | Intel MKL PARDISO is multithreaded and usually 2–5× faster than SuperLU on Intel CPUs; picked up automatically (force with `set SC_FEM_SOLVER=superlu`) |
| GPU | — | Not used. The 2-D problem is too small to benefit, and SciPy's sparse direct solver runs on the CPU |

Example commands:

```bash
pip install pypardiso
python run_sc_fem.py --mesh fine --out results_fine
python sweep_design.py --param gap --values 0.045 0.05 0.055 0.06 0.065 0.07 0.075 --jobs 7
```

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
* ตารางสรุปพารามิเตอร์สำคัญ (`results/08_design_summary.png`)

**ตั้งค่าสำหรับเครื่อง Core Ultra 7 265K / RAM 32 GB / RTX 4060:**

1. ติดตั้ง `pip install pypardiso` เพื่อให้ตัวแก้สมการทำงานหลายเธรดและเร็วขึ้น
2. ใช้ `--mesh fine` เมื่อต้องการผลละเอียด
3. ใช้ `--jobs 6` ถึง `8` เมื่อรัน sweep
4. GPU ไม่ถูกใช้ เพราะปัญหา 2 มิติขนาดนี้ไม่ได้ประโยชน์จาก GPU

หากต้องการเปลี่ยนขนาดเครื่อง ให้แก้ค่าใน `SCDesign` (ไฟล์ `sc_fem2d/machine.py`)
