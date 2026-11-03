"""Nonlinear B-H materials for the 2-D magnetostatic solver.

The reluctivity nu = H/B is tabulated against B^2 so that the Newton-Raphson
Jacobian term d(nu)/d(B^2) is readily available.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

MU0 = 4e-7 * np.pi
NU0 = 1.0 / MU0


class BHMaterial:
    def __init__(self, name: str, B: list[float], H: list[float], b_max: float = 3.0):
        self.name = name
        B, H = np.asarray(B, float), np.asarray(H, float)
        self.B_table, self.H_table = B.copy(), H.copy()     # raw data (used for FEMM export)
        # smooth monotone (PCHIP) H(B); beyond the table the slope tends to mu0
        b2 = np.linspace(0.0, b_max**2, 6001)
        b = np.sqrt(b2)
        Hb = PchipInterpolator(B, H)(np.minimum(b, B[-1]))  # smooth, monotone
        over = b > B[-1]
        Hb[over] = H[-1] + (b[over] - B[-1]) * NU0
        nu = np.empty_like(b)
        nu[1:] = Hb[1:] / b[1:]
        nu[0] = H[1] / B[1]
        self._b2, self._nu = b2, nu
        self._dnu = np.gradient(nu, b2)
        self._Hend, self._Bend = Hb[-1], b[-1]

    def nu(self, b2: np.ndarray):
        """Return (nu, d nu / d B^2) for an array of B^2 values."""
        b2 = np.atleast_1d(np.asarray(b2, float))
        nu = np.interp(b2, self._b2, self._nu)
        dnu = np.interp(b2, self._b2, self._dnu)
        hi = b2 > self._b2[-1]
        if np.any(hi):
            b = np.sqrt(b2[hi])
            H = self._Hend + (b - self._Bend) * NU0
            nu[hi] = H / b
            dnu[hi] = (NU0 - H / b) / b / (2 * b)   # d(H/B)/dB / (2B)
        return nu, dnu

    def mu_r(self, B):
        nu, _ = self.nu(np.asarray(B, float) ** 2)
        return NU0 / nu


# Non-oriented electrical steel lamination (M330-50A class) - stator core
STATOR_LAMINATION = BHMaterial(
    "M330-50A lamination",
    B=[0.0, 0.5, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
    H=[0.0, 60, 115, 160, 270, 420, 850, 2000, 4800, 9500, 17000, 30000, 55000],
)

# Solid rotor forging (26NiCrMoV14-5 class) - rotor body and teeth
ROTOR_FORGING = BHMaterial(
    "Rotor forging 26NiCrMoV",
    B=[0.0, 0.5, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2],
    H=[0.0, 300, 600, 900, 1500, 2200, 3300, 5000, 8500, 15000, 26000, 45000, 75000],
)
