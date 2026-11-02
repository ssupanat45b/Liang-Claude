# Synchronous Condenser — 2-D Electromagnetic FEM Study

Example machine: **100 MVAr, 13.8 kV, 50 Hz, 2-pole, 3000 rpm**, round (cylindrical) rotor.
Nonlinear magnetostatic A<sub>z</sub> FEM, first-order triangles, Newton–Raphson, sliding air-gap interface.
Mesh: 21840 elements / 11113 nodes. Runtime 75 s.

## 1. Main dimensions

| Item | Value |
|---|---|
| Rotor diameter | 1000 mm |
| Air gap (mechanical) | 60 mm |
| Stator bore / outer diameter | 1120 / 2240 mm |
| Active length | 2.90 m |
| Stator slots / coil pitch | 48 / 20 slots (full pitch 24) |
| Armature winding | double layer, 1 parallel paths, 16 series turns/phase, k<sub>w1</sub> = 0.9231 |
| Rotor slots | 24 wound of 32 pitches, 10 turns/slot |
| Rotor peripheral speed | 157 m/s |

![geometry](01_geometry_mesh.png)

## 2. No-load / open circuit

| Result | Value |
|---|---|
| Field current for rated voltage, I<sub>f0</sub> | **714 A** |
| Field current on the air-gap line | 708 A |
| Saturation factor I<sub>f0</sub>/I<sub>f,ag</sub> | 1.009 |
| Air-gap B<sub>r</sub> fundamental | 0.771 T |
| EMF fundamental (rotation sweep) | 13.80 kV line |
| EMF THD | 3.71 % |
| Peak no-load cogging torque | 3.8 kNm |

![no load](02_field_no_load.png)
![air gap](05_airgap_flux.png)

## 3. Reactances and short-circuit ratio

| Parameter | Value |
|---|---|
| X<sub>d</sub> (unsaturated, incl. end leakage 0.06 pu) | **2.103 pu** |
| X<sub>q</sub> (unsaturated) | 2.033 pu |
| Field current for rated armature current on short circuit, I<sub>f,sc</sub> | 1488 A |
| Short-circuit ratio SCR = I<sub>f0</sub>/I<sub>f,sc</sub> | **0.480** |

## 4. Reactive capability

| Operating point | Value |
|---|---|
| Rated over-excited Q | +100 MVAr at I = 4184 A |
| Rated field current (over-excited) | **2251 A** (3.15 × I<sub>f0</sub>) |
| Under-excited capability at I<sub>f</sub> = 0 | -47.7 MVAr (I = 1996 A) |
| Air-gap torque at rated over-excited point | 6.81 kNm (≈ 0: no active power) |
| Max static torque at rated I, rated I<sub>f</sub> | 1.71 pu (T<sub>base</sub> = 318 kNm) |

![rated](03_field_rated_overexcited.png)
![under](04_field_under_excited.png)
![occ](06_occ_vcurve.png)
![emf](07_emf_torque.png)

## 5. Losses (estimate)

| Loss component | Value |
|---|---|
| Stator core (Steinmetz, build factor 1.5) | 144 kW (core mass 55.0 t) |
| Armature copper, R<sub>ph</sub> = 1.82 mΩ | 96 kW (J = 2.4 A/mm²) |
| Field copper, R<sub>f</sub> = 0.063 Ω, U<sub>f</sub> = 143 V | 321 kW (J = 7.1 A/mm²) |
| **Total electromagnetic loss** | **561 kW (0.56 % of rating)** |

Friction, windage, bearing, and stray-load losses are not included.

## 6. Modelling assumptions

* 2-D magnetostatic analysis per unit length × active length; end-winding leakage is added as a fixed reactance (X<sub>end</sub> = 0.06 pu); end-region fields, axial ducts, and stacking factor are neglected in the field solution.
* Slots are modelled as annular sectors (radial sides) on a structured polar mesh; slot openings and wedges are non-magnetic.
* Rotor body: solid forging B–H curve. Stator: M330-50A-class lamination. No eddy currents (magnetostatic).
* Purely reactive operation: armature MMF on the d-axis (−d over-excited, +d under-excited); armature resistance is neglected for voltage.
* Losses: Steinmetz iron loss using the peak element flux density from the rotation sweep; copper at 75 °C.
