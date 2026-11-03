"""Verification tests for the IPM synchronous-condenser FEM.

Run:  python -m unittest discover -s tests -v
"""
import math
import unittest

import numpy as np

from ipm_fem2d import solver
from ipm_fem2d.analysis import PMAnalysis, fundamental_axis
from ipm_fem2d.machine import AIR, MAGNET, ROTOR_IRON, build_mesh, design_preset, magnet_sign


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
        ring = math.pi * (d.R_mag**2 - d.R_core**2) * d.alpha_p_actual
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

    def test_rotor_topology(self):
        """Magnets buried: iron above every magnet (pole shoe) and flux barriers at both ends."""
        d, m = self.d, self.an.mesh
        c = m.r_xy[m.r_tri].mean(axis=1)
        r = np.hypot(c[:, 0], c[:, 1])
        th = np.degrees(np.arctan2(c[:, 1], c[:, 0]))
        on_axis = np.abs(th) < 2.0                                  # around the N-pole axis
        shoe = on_axis & (r > d.R_mag) & (r < d.R_rotor)
        self.assertTrue(np.all(m.r_reg[shoe] == ROTOR_IRON))
        mag = on_axis & (r > d.R_core) & (r < d.R_mag)
        self.assertTrue(np.all(m.r_reg[mag] == MAGNET))
        half = np.degrees(math.pi / d.poles * d.alpha_p_actual)
        bdeg = 360.0 / d.n_theta * d.barrier_divs
        bar = (th > half + 0.1) & (th < half + bdeg - 0.1) & (r > d.R_core) & (r < d.R_bridge)
        self.assertTrue(bar.any() and np.all(m.r_reg[bar] == AIR))
        bridge = (th > half + 0.1) & (th < half + bdeg - 0.1) & (r > d.R_bridge) & (r < d.R_rotor)
        self.assertTrue(bridge.any() and np.all(m.r_reg[bridge] == ROTOR_IRON))

    def test_saliency(self):
        """IPM: q-axis iron between the poles gives Lq > Ld (and the SPM limit Lq ~ Ld is not met)."""
        p = self.an.reactances()
        self.assertGreater(p["Xq_pu"] / p["Xd_pu"], 1.5)

    def test_gap_field_vs_analytic_upper_bound(self):
        """Buried magnets: bridge/pole-shoe leakage means the FEM air-gap B1 lies below the
        leakage-free SPM-type analytic estimate, but still well above half of it.
        Nonlinear iron is essential: the bridges must SATURATE to limit the leakage
        (with linear high-mu iron they short-circuit the magnets - checked below)."""
        d = self.d
        lin = self.an.model.solve(pm=1.0, linear_mu_r=1e5)
        thl, Brl, _ = self.an.model.gap_flux_density(lin, radius=d.R_bore - 0.25 * d.gap)
        self.assertLess(fundamental_axis(thl, Brl, d.pp)[0], 0.1)       # shorted by unsaturated bridges
        sol = self.an.model.solve(pm=1.0)
        r = d.R_bore - 0.25 * d.gap
        th, Br, _ = self.an.model.gap_flux_density(sol, radius=r)
        B1, _ = fundamental_axis(th, Br, d.pp)
        g_tot = d.gap + d.h_m / d.mu_rm
        Bg = d.Br() * (d.h_m / d.mu_rm) / g_tot
        B1_an = 4 / math.pi * Bg * math.sin(d.alpha_p_actual * math.pi / 2)
        self.assertLess(B1, B1_an)
        self.assertGreater(B1, 0.5 * B1_an)

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

    def test_reluctance_torque(self):
        """With magnets off, an IPM rotor still produces (reluctance) torque at 45 deg current."""
        an, d = self.an, self.d
        Ipk = math.sqrt(2) * d.I_rated
        T = an.model.solve(i_abc=an.currents(Ipk, an.theta_d() + math.radians(135)), pm=0.0).torque
        self.assertGreater(abs(T), 0.05 * d.S_rated / (d.omega / d.pp))


if __name__ == "__main__":
    unittest.main()
