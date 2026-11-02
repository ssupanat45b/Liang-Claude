"""Verification tests for the 2-D synchronous-condenser FEM.

Run:  python -m unittest discover -s tests -v
"""
import math
import unittest

import numpy as np

from sc_fem2d.analysis import SCAnalysis, fundamental_axis
from sc_fem2d.materials import MU0, STATOR_LAMINATION, ROTOR_FORGING
from sc_fem2d.machine import SCDesign, build_mesh, winding_layout


class TestMaterials(unittest.TestCase):
    def test_bh_consistency(self):
        for mat in (STATOR_LAMINATION, ROTOR_FORGING):
            B = np.array([0.3, 1.0, 1.6, 2.0, 2.8, 3.5])
            nu, dnu = mat.nu(B**2)
            H = nu * B
            self.assertTrue(np.all(np.diff(H) > 0), "H(B) must be monotone")
            # numerical derivative check of d nu / d B^2
            eps = 1e-4
            nup, _ = mat.nu(B**2 + eps)
            num, _ = mat.nu(B**2 - eps)
            np.testing.assert_allclose(dnu, (nup - num) / (2 * eps), rtol=0.05, atol=1e-3)
        # deep saturation approaches vacuum reluctivity slope
        self.assertGreater(STATOR_LAMINATION.mu_r(3.5)[0], 1.0)


class TestMeshAndWinding(unittest.TestCase):
    def test_mesh_and_balanced_winding(self):
        d = SCDesign()
        m = build_mesh(d)
        tri, xy, reg, part, nn, outer = m.connectivity(0)
        x, y = xy[..., 0], xy[..., 1]
        area = 0.5 * ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
        self.assertTrue(np.all(area > 0))
        # total area equals the stator outer disk (polygonal)
        n = d.n_theta
        poly = 0.5 * n * d.R_stator_out**2 * math.sin(2 * math.pi / n)
        self.assertAlmostEqual(area.sum(), poly, places=6)
        # rotation keeps node count and validity
        tri2, *_ , nn2, _ = m.connectivity(7)
        self.assertEqual(nn, nn2)
        lay = winding_layout(d)
        for ph in range(3):
            n_sides = sum(1 for L in (0, 1) for (p, s) in lay[L] if p == ph)
            net = sum(s for L in (0, 1) for (p, s) in lay[L] if p == ph)
            self.assertEqual(n_sides, 2 * d.Qs // 3)
            self.assertEqual(net, 0)


class TestPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an = SCAnalysis(SCDesign(), verbose=False)
        cls.an.find_axes()

    def test_phase_axes_120deg(self):
        th = self.an.res.axes["phase"]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            diff = (th[b] - th[a]) % (2 * math.pi)
            self.assertTrue(min(abs(diff - 2 * math.pi / 3), abs(diff - 4 * math.pi / 3)) < math.radians(1.0))

    def test_reciprocity_linear(self):
        """Mutual inductance field->phase A equals phase A->field (linear iron)."""
        m = self.an.model
        I = 200.0
        s1 = m.solve(i_f=I, linear_mu_r=1000.0)
        s2 = m.solve(i_abc=(I, 0, 0), linear_mu_r=1000.0)
        self.assertAlmostEqual(s1.psi_abc[0] / I, s2.psi_f / I, delta=1e-3 * abs(s1.psi_abc[0] / I) + 1e-9)

    def test_emf_formula(self):
        """FEM flux linkage vs classical psi = N kw B1 D L (2-pole) from the FEM gap field."""
        an, d = self.an, self.an.d
        sol = an.model.solve(i_f=300.0, linear_mu_r=1000.0)
        th, Br, _ = an.model.gap_flux_density(sol, radius=d.R_bore - 1e-3)
        B1, _ = fundamental_axis(th, Br)
        psi_formula = d.series_turns * d.winding_factor() * B1 * (2 * d.R_bore) * d.L_stack
        pd, pq = an.psi_dq(sol)
        self.assertAlmostEqual(math.hypot(pd, pq) / psi_formula, 1.0, delta=0.04)

    def test_torque_virtual_work(self):
        """Arkkio torque agrees with co-energy derivative (linear, rotor shifted +-1 division)."""
        an, d = self.an, self.an.d
        cur = an.currents(2000.0, an.theta_d() + math.radians(120))
        If = 400.0
        W = []
        for k in (-1, 1):
            s = an.model.solve(i_abc=cur, i_f=If, k_shift=k, linear_mu_r=1000.0)
            W.append(0.5 * (np.dot(cur, s.psi_abc) + If * s.psi_f))
        dth = 2 * math.pi / d.n_theta
        T_vw = (W[1] - W[0]) / (2 * dth)
        T_ark = an.model.solve(i_abc=cur, i_f=If, k_shift=0, linear_mu_r=1000.0).torque
        self.assertAlmostEqual(T_ark / T_vw, 1.0, delta=0.05)

    def test_ampere_law(self):
        """Line integral of H around the rotor (in the gap) is ~0 for the balanced field winding,
        and B fundamental scales linearly with If in the linear regime."""
        m = self.an.model
        a = m.solve(i_f=100.0, linear_mu_r=1000.0)
        b = m.solve(i_f=200.0, linear_mu_r=1000.0)
        th, Br1, Bt1 = m.gap_flux_density(a)
        _, Br2, _ = m.gap_flux_density(b)
        self.assertAlmostEqual(np.mean(Bt1) / MU0 * 2 * math.pi * 0.55, 0.0, delta=5.0)
        np.testing.assert_allclose(Br2, 2 * Br1, rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
