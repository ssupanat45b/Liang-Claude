# Liang_IPM_AEDT — IPM Synchronous Condenser 300 kVAr ใน ANSYS Electronics Desktop

สคริปต์ `build_aedt.py` ใช้ **PyAEDT** สร้างโมเดล **Maxwell 2D – Transient (XY)** ของเครื่อง IPM
(แม่เหล็กสี่เหลี่ยม) โดยอัตโนมัติ ด้วยข้อมูลออกแบบชุดเดียวกับ Python FEM (`ipm_fem2d`) ได้แก่ slot,
ขดลวด double layer, แม่เหล็ก/flux barrier/bridge, B-H curve, ความยาวแกน และ air gap

## ต้องมี
* ANSYS Electronics Desktop 2022 R2 ขึ้นไป พร้อม license Maxwell (Windows)
* Python 3.10–3.13: `pip install -r requirements.txt`  (PyAEDT = `ansys-aedt-core`)

## วิธีรัน (ใน VS Code terminal)
```
cd C:\Users\Liang\Desktop\Liang_Syn_Con\Liang_IPM_AEDT
python build_aedt.py --preview-only        # ตรวจรูปตัด + ค่าอ้างอิง (ไม่เปิด AEDT)
python build_aedt.py                        # สร้างโมเดล rated load (เปิด AEDT ให้)
python build_aedt.py --load noload          # แม่เหล็กอย่างเดียว: EMF, cogging
python build_aedt.py --version 2024.2 --solve   # ระบุเวอร์ชัน + solve เลย
```
ตัวเลือก `--design auto|maggap|bo|ipm300`
* `auto` (default): ใช้ knee ของ `../Liang_IPM_MagGap/results/pareto.csv` ถ้ามี
  ไม่มีก็ใช้ BO compromise (`../Liang_IPM_BO/results/pareto.csv`) และสุดท้ายคือค่าตั้งต้นในโค้ด
* ให้วางโฟลเดอร์นี้ข้าง ๆ `Liang_IPM_MagGap` / `Liang_IPM_BO` เพื่อให้หาผล optimisation เจอ

## ⚠ ถ้า AEDT ขึ้น "File not found: ...\Liang_IPM_AEDT\51032" (ตัวเลขใด ๆ)
PyAEDT 1.x เปิด AEDT ด้วยคำสั่ง `ansysedt.exe -grpcsrv <port>` ถ้า AEDT เวอร์ชันนั้นไม่รู้จัก
argument นี้ (เวอร์ชันก่อน 2022 R2, Student บางรุ่น หรือ service pack ไม่ตรงกับ PyAEDT) จะมองตัวเลข
port เป็นชื่อไฟล์ → ให้ใช้ **native script** แทน (ไม่ต้องใช้ PyAEDT, ใช้ได้ทุกเวอร์ชัน)

1. ปิด dialog / ปิด AEDT ที่ค้าง แล้วรัน
   ```
   python build_aedt.py --script-only                 # rated load
   python build_aedt.py --script-only --load noload   # แม่เหล็กอย่างเดียว
   ```
   ได้ไฟล์ `aedt_model\IPM_SynCon_aedt_script_rated.py` (และ `_noload.py`)
2. เปิด ANSYS Electronics Desktop เอง → **Tools ▸ Run Script...** → เลือกไฟล์ข้างบน
   (ถ้าถามชนิด script เลือก IronPython/Python) ดูความคืบหน้าในหน้าต่าง Message Manager
3. สคริปต์บันทึกโปรเจกต์ `aedt_model\IPM_SynCon_300kVAr_<load>_native.aedt`
   แล้วกด **Validate (F7)** → **Analyze All**

native script สร้างทุกอย่างเหมือน PyAEDT (วัสดุ B-H, สเตเตอร์/ขดลวด 72 ชิ้น, โรเตอร์ + barrier,
แม่เหล็ก 4 ก้อนพร้อม CS, band 1500 rpm, A=0, winding/coil, mesh, transient setup, report)
ถ้าขั้นใดใน script error ให้ส่งข้อความใน Message Manager มา

## สิ่งที่สร้างในโปรเจกต์ `aedt_model/IPM_SynCon_300kVAr_<load>.aedt`
| ส่วน | รายละเอียด |
|---|---|
| วัสดุ | M330_50A_pyFEM / RotorSteel_pyFEM (B-H nonlinear เดียวกับ Python), NdFeB_N38SH_80C (Br 1.141 T, μr 1.05), copper |
| เรขาคณิต | Stator, Coil_X_nn_top/bot (2×Qs), Rotor, Magnet_n (มี CS ของตัวเอง ทิศ N/S สลับ), barrier = air |
| Motion | Band หมุน 1500 rpm, A = 0 ที่ขอบ Region |
| Winding | PhaseA/B/C แบบ Current, `Ipk*cos(2*pi*Freq*Time + Theta_d0 + 180deg - θph)` → กระแสอยู่แกน −d (over-excited, ส่ง kVAr) |
| Mesh | length-based: magnet 2 mm, band 1.5 mm, coil 3 mm, rotor 4 mm, stator 5 mm |
| Setup | Transient 2 คาบ, step = T/200 |
| Report | Torque, FluxLinkage, InducedVoltage, Current |

ตัวแปร design (`Ipk`, `Freq`, `Speed_rpm`, `Theta_d0`) แก้ได้ใน Project ▸ Design Properties
เช่นตั้ง `Ipk = 0A` เพื่อดู no-load EMF/cogging

## การเปรียบเทียบผล
`aedt_model/aedt_expected.json` เก็บค่าจาก Python FEM ของ design เดียวกัน
* `flux_linkage_peak_Wb` ↔ peak ของ FluxLinkage(PhaseA)
* `induced_voltage_line_rms_V` ↔ √3 × rms ของ InducedVoltage(PhaseA) (AEDT ให้ค่าต่อเฟส)
* `torque_mean_Nm` ≈ 0 (condenser ไม่ส่งกำลังจริง) — cogging/ripple ดูจาก `--load noload`
คาดว่าต่างกัน ~2–5 % (mesh, AEDT ไม่มี skew 1 slot ใน 2D, ขดลวดปลายไม่ถูกรวมทั้งคู่)

## ข้อควรระวัง
* สคริปต์ทดสอบเฉพาะการเรียก API กับ signature ของ PyAEDT 1.x (mock) (ไม่มี AEDT บน cloud)
  ถ้า PyAEDT เวอร์ชันของคุณ error ในขั้นตอนใด ให้ส่ง traceback มา
* ขั้น report/torque อยู่ใน try — ถ้าข้าม ให้สร้าง report เองใน Results ▸ Create Transient Report
* ถ้าค่า Theta_d0 / แกนเฟสไม่ตรง (เช่น Q ติดลบ) ให้รัน `--load noload` ดูเฟสของ EMF ก่อน
