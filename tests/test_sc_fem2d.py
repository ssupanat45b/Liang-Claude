"""Verification tests for the 2-D synchronous-condenser FEM.

Run:  python -m unittest discover -s tests -v
"""
import math
import unittest

import numpy as np

from sc_fem2d.analysis import SCAnalysis, fundamental_axis
from sc_fem2d.materials import MU0, STATOR_LAMINATION, ROTOR_FORGING
from sc_fem2d.machine import SCDesign, build_mesh, design_preset, winding_layout


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
    """2-pole, 100 MVAr design.  TestPhysics4Pole repeats every test on the 4-pole design."""

    @classmethod
    def make_design(cls):
        return SCDesign()

    @classmethod
    def setUpClass(cls):
        cls.an = SCAnalysis(cls.make_design(), verbose=False)
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
        """FEM flux linkage vs classical psi = N kw Phi,  Phi = 2 B1 D L / p, from the FEM gap field."""
        an, d = self.an, self.an.d
        sol = an.model.solve(i_f=300.0, linear_mu_r=1000.0)
        th, Br, _ = an.model.gap_flux_density(sol, radius=d.R_bore - 0.25 * d.gap)
        B1, _ = fundamental_axis(th, Br, d.pp)
        psi_formula = d.series_turns * d.winding_factor() * d.skew_factor(1) \
            * 2 * B1 * (2 * d.R_bore) * d.L_stack / d.poles
        pd, pq = an.psi_dq(sol)
        self.assertAlmostEqual(math.hypot(pd, pq) / psi_formula, 1.0, delta=0.04)

    def test_torque_virtual_work(self):
        """Arkkio torque vs co-energy virtual work, both averaged over one stator slot pitch
        (removes slot ripple).  Arkkio converges to virtual work as the gap mesh is refined;
        12 gap layers (normal preset) give ~1-3 %."""
        an, d = self.an, self.an.d
        cur = an.currents(2 * d.I_rated, an.theta_d() + math.radians(110))
        If = 0.5 * d.I_rated * d.series_turns / d.Nf_per_slot     # comparable MMF
        ks = range(d.div_per_slot + 1)
        sols = [an.model.solve(i_abc=cur, i_f=If, k_shift=k, linear_mu_r=1000.0) for k in ks]
        W = [0.5 * (np.dot(cur, s.psi_abc) + If * s.psi_f) for s in sols]
        dth = 2 * math.pi / d.n_theta
        T_vw = (W[-1] - W[0]) / (d.div_per_slot * dth)
        T_ark = np.trapezoid([s.torque for s in sols], dx=dth) / (d.div_per_slot * dth)
        self.assertAlmostEqual(T_ark / T_vw, 1.0, delta=0.04)

    def test_ampere_law(self):
        """Line integral of H around the rotor (in the gap) is ~0 for the balanced field winding,
        and B fundamental scales linearly with If in the linear regime."""
        m = self.an.model
        a = m.solve(i_f=100.0, linear_mu_r=1000.0)
        b = m.solve(i_f=200.0, linear_mu_r=1000.0)
        th, Br1, Bt1 = m.gap_flux_density(a)
        _, Br2, _ = m.gap_flux_density(b)
        r = self.an.d.R_bore - 0.25 * self.an.d.gap
        self.assertAlmostEqual(np.mean(Bt1) / MU0 * 2 * math.pi * r, 0.0, delta=5.0)
        np.testing.assert_allclose(Br2, 2 * Br1, rtol=1e-6, atol=1e-9)


class TestPhysics4Pole(TestPhysics):
    """Same checks on the 4-pole 300 kVAr factory design (skew off: tests the 2-D FEM itself)."""

    @classmethod
    def make_design(cls):
        d = design_preset("factory300")
        d.skew_slots = 0.0
        return d

    def test_four_poles_in_gap_field(self):
        sol = self.an.model.solve(i_f=10.0, linear_mu_r=1000.0)
        th, Br, _ = self.an.model.gap_flux_density(sol)
        h = np.abs(np.fft.rfft(Br))
        self.assertEqual(int(np.argmax(h[1:12])) + 1, 2)      # dominant wave has 2 pole pairs


if __name__ == "__main__":
    unittest.main()
