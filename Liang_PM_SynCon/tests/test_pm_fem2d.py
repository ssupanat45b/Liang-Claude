"""Verification tests for the PM (SPM) synchronous-condenser FEM.

Run:  python -m unittest discover -s tests -v
"""
import math
import unittest

import numpy as np

from pm_fem2d import solver
from pm_fem2d.analysis import PMAnalysis, fundamental_axis
from pm_fem2d.machine import MAGNET, build_mesh, design_preset, magnet_sign


class TestMeshMagnets(unittest.TestCase):
    def test_magnet_layout(self):
        d = design_preset()
        m = build_mesh(d)
        mag = m.r_reg == MAGNET
        counts = np.bincount(m.r_tag[mag], minlength=d.poles)
        self.assertEqual(len(counts), d.poles)
        self.assertTrue(np.all(counts == counts[0]), "all magnets must have the same mesh")
        np.testing.assert_array_equal(magnet_sign(d, np.arange(4)), [1, -1, 1, -1])
        tri, xy, reg, part, nn, outer = m.connectivity(5)
        x, y = xy[..., 0], xy[..., 1]
        area = 0.5 * ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
        self.assertTrue(np.all(area > 0))
        # magnet area = arc fraction * annulus area (polygonal mesh, small chord error)
        ring = math.pi * (d.R_rotor**2 - d.R_core**2) * d.alpha_p_actual
        nr = len(m.r_tri)
        self.assertAlmostEqual(area[:nr][mag].sum() / ring, 1.0, delta=0.002)


class TestPMPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solver.set_backend("superlu")
        cls.d = design_preset()
        cls.d.skew_slots = 0.0
        cls.an = PMAnalysis(cls.d, verbose=False)
        cls.an.find_axes()

    def test_axes(self):
        ax = self.an.res.axes
        self.assertLess(abs(math.degrees(ax["d"])), 0.5)                  # N pole centred at 0
        th = ax["phase"]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            diff = (th[b] - th[a]) % (2 * math.pi)
            self.assertTrue(min(abs(diff - 2 * math.pi / 3), abs(diff - 4 * math.pi / 3)) < math.radians(1))

    def test_linear_superposition(self):
        """Linear iron: field(magnets + current) = field(magnets) + field(current)."""
        m = self.an.model
        cur = self.an.reactive_currents(200.0)
        a = m.solve(pm=1.0, linear_mu_r=1000.0)
        b = m.solve(i_abc=cur, pm=0.0, linear_mu_r=1000.0)
        c = m.solve(i_abc=cur, pm=1.0, linear_mu_r=1000.0)
        np.testing.assert_allclose(c.psi_abc, a.psi_abc + b.psi_abc, rtol=1e-8, atol=1e-10)

    def test_gap_field_vs_analytic(self):
        """Radial 1-D analytic SPM field  B(r) = C/r  with Carter's factor on the whole magnetic
        path (gap + h_m/mu_r), fundamental of a rectangular pole wave, vs FEM (linear iron)."""
        d = self.d
        sol = self.an.model.solve(pm=1.0, linear_mu_r=1e5)
        r = d.R_bore - 0.25 * d.gap
        th, Br, _ = self.an.model.gap_flux_density(sol, radius=r)
        B1, _ = fundamental_axis(th, Br, d.pp)
        tau_s = 2 * math.pi * d.R_bore / d.Qs
        b0 = tau_s * d.slot_divs / d.div_per_slot
        g_eff = d.gap + d.h_m / d.mu_rm
        u = b0 / (2 * g_eff)
        gam = 4 / math.pi * (u * math.atan(u) - math.log(math.sqrt(1 + u * u)))
        kc = tau_s / (tau_s - gam * g_eff)
        Rc, Rr, Rb = d.R_core, d.R_rotor, d.R_bore
        C = d.Br() * (Rr - Rc) / d.mu_rm / (kc * (math.log(Rr / Rc) / d.mu_rm + math.log(Rb / Rr)))
        B1_an = 4 / math.pi * (C / r) * math.sin(d.alpha_p_actual * math.pi / 2)
        self.assertAlmostEqual(B1 / B1_an, 1.0, delta=0.04)

    def test_emf_scales_with_remanence(self):
        m = self.an.model
        a = m.solve(pm=1.0, linear_mu_r=1000.0)
        b = m.solve(pm=0.5, linear_mu_r=1000.0)
        np.testing.assert_allclose(b.psi_abc, 0.5 * a.psi_abc, rtol=1e-8, atol=1e-12)

    def test_operating_point_balance(self):
        """At the solved point the air-gap EMF equals V + X_end*I, and Q decreases with V."""
        an, d = self.an, self.d
        I1, s1 = an.current_for_voltage(d.V_phase)
        I2, _ = an.current_for_voltage(1.05 * d.V_phase)
        E = an.E_phase(an.psi_dq(s1)[0])
        self.assertAlmostEqual(E, d.V_phase + an.X_end * I1, delta=1e-3 * d.V_phase)
        self.assertGreater(I1, I2)

    def test_torque_symmetry(self):
        """Pure d-axis current gives (near) zero torque; q-axis current gives torque."""
        an, d = self.an, self.d
        Ipk = math.sqrt(2) * d.I_rated
        Td = an.model.solve(i_abc=an.currents(Ipk, an.theta_d() + math.pi)).torque
        Tq = an.model.solve(i_abc=an.currents(Ipk, an.theta_d() + math.pi / 2)).torque
        self.assertLess(abs(Td), 0.1 * abs(Tq))


if __name__ == "__main__":
    unittest.main()
