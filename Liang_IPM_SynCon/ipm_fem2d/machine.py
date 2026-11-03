"""Interior permanent-magnet synchronous condenser (IPMSM rotor): design data and FE mesh.

Rotor: solid steel rotor with one BURIED, radially magnetised NdFeB arc magnet
per pole (alternating N/S).  Above each magnet an iron pole shoe; at both
magnet ends an air flux barrier that stops just below the rotor surface,
leaving a thin saturable iron bridge that holds the pole shoe.  The iron
between the poles (q-axis) makes the rotor salient (Lq > Ld).  No sleeve.

Geometry per pole (polar, centred on the pole axis):

    R_rotor ───────────── iron bridge (t_bridge) ───────────────
            |barrier|      pole shoe (t_shoe)      |barrier|  q-axis iron
    R_mag   |       |========== magnet h_m ========|       |
    R_core  ─────────────────── rotor core ─────────────────────

The structured polar mesh cannot represent V-shaped magnets; this is the
single-layer buried-arc ("inset/buried arc") IPM topology.

Mesh strategy
-------------
The cross-section is meshed with a structured polar grid of linear
triangles.  Slots are annular sectors aligned with the angular divisions,
which keeps the mesh conforming without an external mesher.  The rotor and
the stator are meshed separately and joined on a circle in the middle of
the air gap ("sliding interface"): rotating the rotor by k angular
divisions only re-maps the interface node numbering, so no re-meshing is
needed for rotor-position sweeps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# region codes
AIR, ROTOR_IRON, STATOR_IRON, MAGNET, ARM_COIL, STATOR_WEDGE = range(6)
REGION_NAMES = {AIR: "air gap / flux barrier", ROTOR_IRON: "rotor iron", STATOR_IRON: "stator iron",
                MAGNET: "permanent magnet", ARM_COIL: "armature winding",
                STATOR_WEDGE: "stator slot opening"}


@dataclass
class PMDesign:
    # ---- rating ---------------------------------------------------------
    S_rated: float = 300e3        # VA (rated reactive output at rated voltage)
    V_line: float = 400.0         # V rms line-line
    f: float = 50.0               # Hz
    poles: int = 4
    # ---- main dimensions ---------------------------------------------
    L_stack: float = 0.45         # m, active core length
    stacking: float = 0.95        # lamination stacking factor (loss/mass only)
    R_shaft_center: float = 0.0
    R_rotor: float = 0.1975       # rotor outer radius = magnet outer surface
    gap: float = 0.0025           # air gap rotor OD -> stator bore (no sleeve)
    R_stator_out: float = 0.300   # stator core outer radius
    # ---- stator slots / winding --------------------------------------
    Qs: int = 48                  # stator slots
    div_per_slot: int = 8         # angular divisions per stator slot pitch
    slot_divs: int = 4            # divisions occupied by the slot (=> slot/pitch = 0.5)
    s_wedge: float = 0.003        # slot-opening/wedge depth
    s_layer: float = 0.018        # depth of each winding layer (2 layers)
    coil_pitch: int = 10          # slots (full pitch = Qs/poles = 12 -> 5/6 chording)
    parallel_paths: int = 2
    turns_per_coil: int = 2
    # ---- interior permanent magnets (IPM rotor) ------------------------
    h_m: float = 0.0295           # magnet radial thickness [m] (sized for 300 kVAr at 400 V)
    alpha_p: float = 0.80         # magnet arc / pole pitch (rounded to the mesh divisions)
    t_shoe: float = 0.005         # iron pole shoe above the magnet [m]
    t_bridge: float = 0.0012      # iron bridge between flux barrier and rotor surface [m]
    barrier_frac: float = 0.06    # angular width of EACH flux barrier / pole pitch
    t_sleeve: float = 0.0         # no retaining sleeve on an IPM rotor
    Br20: float = 1.23            # remanence at 20 degC [T]  (NdFeB N38SH class)
    mu_rm: float = 1.05           # magnet recoil permeability
    alpha_Br: float = -0.0012     # remanence temperature coefficient [1/K]
    T_mag: float = 80.0           # magnet operating temperature [degC]
    B_demag_limit: float = 0.10   # min. flux density along magnetisation (N38SH @ 80 C + margin) [T]
    rho_mag: float = 7500.0       # magnet density [kg/m^3]
    # ---- extra (3-D) effects not captured by the 2-D model -----------
    X_end_pu: float = 0.06        # end-winding leakage reactance [pu]
    skew_slots: float = 1.0       # stator skew in stator slot pitches (applied analytically)
    # ---- loss data --------------------------------------------------
    rho_cu: float = 2.1e-8        # ohm m at ~75 C
    fill_arm: float = 0.45        # copper fill of armature layer
    end_arm: float = 0.30         # m, end-winding length per half-turn
    kh: float = 0.020             # W/kg/(T^2 Hz)   hysteresis   (M330-50A class)
    ke: float = 1.9e-4            # W/kg/(T^2 Hz^2) eddy current
    build_factor: float = 1.5
    rho_fe: float = 7650.0
    # ---- mesh density (see MESH_PRESETS / apply_mesh_preset) ----------
    gap_layers: int = 12
    radial_refine: int = 1        # multiplies the number of radial rings in slots/yoke

    # derived ---------------------------------------------------------
    @property
    def pp(self) -> int:
        """Pole pairs."""
        return self.poles // 2

    @property
    def R_mag(self) -> float:
        """Outer radius of the buried magnet (= underside of the pole shoe)."""
        return self.R_rotor - self.t_shoe

    @property
    def R_core(self) -> float:
        """Magnet seat radius (inner radius of magnet and flux barriers)."""
        return self.R_mag - self.h_m

    @property
    def R_bridge(self) -> float:
        """Outer radius of the flux barriers (bridge underside)."""
        return self.R_rotor - self.t_bridge

    @property
    def barrier_divs(self) -> int:
        per_pole = self.n_theta // self.poles
        return max(1, int(round(self.barrier_frac * per_pole)))

    @property
    def magnet_divs(self) -> int:
        """Angular mesh divisions per magnet (even, centred on the pole)."""
        per_pole = self.n_theta // self.poles
        return max(2, min(per_pole - 2, 2 * int(round(self.alpha_p * per_pole / 2))))

    @property
    def alpha_p_actual(self) -> float:
        return self.magnet_divs / (self.n_theta // self.poles)

    def Br(self, T=None) -> float:
        """Remanence at temperature T [degC] (default: operating temperature)."""
        T = self.T_mag if T is None else T
        return self.Br20 * (1 + self.alpha_Br * (T - 20.0))

    @property
    def g_mech(self) -> float:
        return self.gap - self.t_sleeve

    @property
    def n_theta(self) -> int:
        return self.Qs * self.div_per_slot

    @property
    def R_bore(self) -> float:
        return self.R_rotor + self.gap

    @property
    def R_slide(self) -> float:
        return self.R_rotor + 0.5 * self.gap

    @property
    def omega(self) -> float:
        return 2 * math.pi * self.f

    @property
    def rpm(self) -> float:
        return 120 * self.f / self.poles

    @property
    def V_phase(self) -> float:
        return self.V_line / math.sqrt(3)

    @property
    def I_rated(self) -> float:
        return self.S_rated / (math.sqrt(3) * self.V_line)

    @property
    def Z_base(self) -> float:
        return self.V_line**2 / self.S_rated

    @property
    def series_turns(self) -> float:
        # double layer: Qs coils, Qs/3 coils per phase
        return self.Qs / 3 * self.turns_per_coil / self.parallel_paths

    def skew_factor(self, nu=1):
        """Skew factor for EMF harmonic ``nu`` (electrical order); 1 for an unskewed core."""
        a = self.skew_slots * 2 * math.pi * self.pp / self.Qs * np.asarray(nu, float) / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(np.abs(a) < 1e-12, 1.0, np.sin(a) / np.where(a == 0, 1, a))
        return k if np.ndim(k) else float(k)

    def winding_factor(self) -> float:
        q = self.Qs / (3 * self.poles)
        alpha = math.pi * self.poles / self.Qs
        kd = math.sin(q * alpha / 2) / (q * math.sin(alpha / 2))
        kp = math.sin(math.pi / 2 * self.coil_pitch / (self.Qs / self.poles))
        return kd * kp

    def check(self):
        if self.poles % 2 or self.poles < 2:
            raise ValueError("poles must be an even number")
        if self.Qs % (3 * self.poles):
            raise ValueError("Qs must be a multiple of 3*poles (integer slots per pole and phase)")
        if self.n_theta % (2 * self.poles):
            raise ValueError("n_theta must be a multiple of 2*poles")
        if not 0 < self.h_m < 0.6 * self.R_rotor:
            raise ValueError("magnet thickness out of range")
        if not 0 < self.t_bridge < self.t_shoe:
            raise ValueError("bridge must be thinner than the pole shoe")
        per_pole = self.n_theta // self.poles
        if self.magnet_divs + 2 * self.barrier_divs >= per_pole:
            raise ValueError("magnet + flux barriers wider than the pole pitch")


# Mesh presets: angular divisions per stator slot pitch / rotor slot share, gap layers,
# radial refinement.  Geometry (slot/pitch ratios) is identical for every preset.
MESH_PRESETS = {
    #           div_per_slot slot_divs gap_layers radial_refine
    "coarse": (8, 4, 6, 1),
    "normal": (8, 4, 12, 1),
    "fine":   (16, 8, 24, 2),
    "ultra":  (24, 12, 32, 3),
}


def apply_mesh_preset(d: "PMDesign", name: str) -> "PMDesign":
    d.div_per_slot, d.slot_divs, d.gap_layers, d.radial_refine = MESH_PRESETS[name]
    return d


# ---------------------------------------------------------------------------
@dataclass
class Mesh:
    """Rotor + stator triangle mesh joined at the sliding circle."""
    design: PMDesign
    # rotor part
    r_xy: np.ndarray = field(default=None)        # rotor node coords (rotor frame)
    r_tri: np.ndarray = field(default=None)
    r_reg: np.ndarray = field(default=None)
    r_tag: np.ndarray = field(default=None)       # magnet pole index or -1
    r_ring: np.ndarray = field(default=None)      # rotor node ids on the sliding circle
    # stator part
    s_xy: np.ndarray = field(default=None)
    s_tri: np.ndarray = field(default=None)
    s_reg: np.ndarray = field(default=None)
    s_tag: np.ndarray = field(default=None)       # (slot, layer) packed: 2*slot+layer or -1
    s_ring: np.ndarray = field(default=None)      # stator node ids on the sliding circle
    s_outer: np.ndarray = field(default=None)     # stator nodes on the outer boundary

    # ------------------------------------------------------------------
    def connectivity(self, k_shift: int = 0):
        """Global element arrays for a rotor rotated by ``k_shift`` divisions (CCW).

        Returns tri (ne,3) global node ids, xy (ne,3,2) element-local coordinates
        (rotor elements in the rotor frame), region, part (0 rotor/1 stator),
        number of global nodes and the outer-boundary node ids."""
        n = self.design.n_theta
        nr = len(self.r_xy)
        # stator node -> global id
        smap = np.arange(len(self.s_xy)) + nr
        # stator interface node j (angle j*dth) coincides with rotor node at angle j - k
        smap[self.s_ring] = self.r_ring[(np.arange(n) - k_shift) % n]
        # renumber stator-only nodes compactly
        only = np.ones(len(self.s_xy), bool); only[self.s_ring] = False
        smap[only] = nr + np.arange(only.sum())
        nnodes = nr + only.sum()
        tri = np.vstack([self.r_tri, smap[self.s_tri]])
        xy = np.concatenate([self.r_xy[self.r_tri], self.s_xy[self.s_tri]])
        reg = np.concatenate([self.r_reg, self.s_reg])
        part = np.concatenate([np.zeros(len(self.r_tri), int), np.ones(len(self.s_tri), int)])
        return tri, xy, reg, part, nnodes, smap[self.s_outer]

    def node_coords(self, k_shift: int = 0) -> np.ndarray:
        """Physical (stator-frame) coordinates of the global nodes for plotting."""
        n = self.design.n_theta
        tri, xy, reg, part, nn, outer = self.connectivity(k_shift)
        P = np.zeros((nn, 2))
        a = 2 * np.pi * k_shift / n
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        P[:len(self.r_xy)] = self.r_xy @ R.T
        nr = len(self.r_xy)
        only = np.ones(len(self.s_xy), bool); only[self.s_ring] = False
        P[nr:] = self.s_xy[only]
        return P


def _ring_nodes(r, n, start_id, xy_list):
    th = 2 * np.pi * np.arange(n) / n
    xy_list.append(np.c_[r * np.cos(th), r * np.sin(th)])
    return start_id + np.arange(n)


def _connect(inner, outer):
    """Triangles between two rings; len(outer) == len(inner) or 2*len(inner)."""
    ni, no = len(inner), len(outer)
    tris = []
    if no == ni:
        for j in range(ni):
            a, b = inner[j], inner[(j + 1) % ni]
            c, d = outer[j], outer[(j + 1) % no]
            tris += [(a, c, d), (a, d, b)]
    elif no == 2 * ni:
        for j in range(ni):
            a, b = inner[j], inner[(j + 1) % ni]
            c, m, d = outer[2 * j], outer[2 * j + 1], outer[(2 * j + 2) % no]
            tris += [(a, c, m), (a, m, b), (b, m, d)]
    else:
        raise ValueError("ring sizes must be equal or doubling")
    return tris


def _build_part(radii, sizes, center: bool):
    xy_list, tris, rings = [], [], []
    nid = 0
    if center:
        xy_list.append(np.zeros((1, 2))); nid = 1
        prev = None
    for k, (r, n) in enumerate(zip(radii, sizes)):
        ids = _ring_nodes(r, n, nid, xy_list); nid += n
        if k == 0 and center:
            for j in range(n):
                tris.append((0, ids[j], ids[(j + 1) % n]))
        elif k > 0:
            tris += _connect(rings[-1], ids)
        rings.append(ids)
    return np.vstack(xy_list), np.array(tris), rings


def build_mesh(d: PMDesign) -> Mesh:
    d.check()
    n = d.n_theta
    # ---------------- rotor radii -------------------------------------
    r_c = d.R_core
    inner = []
    nn = n
    while nn % 2 == 0 and nn > 24:
        nn //= 2
        inner.append(nn)
    inner = inner[::-1]                        # e.g. 24, 48, 96, 192
    r_inner = list(np.linspace(0, r_c * 0.92, len(inner) + 2)[1:-1])
    rf = max(1, int(d.radial_refine))
    rotor_r = r_inner + [r_c * 0.92, r_c * 0.97] + list(np.linspace(r_c, d.R_mag, 4 * rf + 1)) \
        + list(np.linspace(d.R_mag, d.R_bridge, 3 * rf + 1)[1:]) \
        + list(np.linspace(d.R_bridge, d.R_rotor, 2 * rf + 1)[1:])
    gl = d.gap_layers // 2
    rotor_r += list(np.linspace(d.R_rotor, d.R_slide, gl + 1)[1:])
    rotor_n = inner + [n] * (len(rotor_r) - len(inner))
    r_xy, r_tri, r_rings = _build_part(rotor_r, rotor_n, center=True)

    # ---------------- stator radii ------------------------------------
    s1 = d.R_bore + d.s_wedge
    s2 = s1 + d.s_layer
    s3 = s2 + d.s_layer
    stator_r = list(np.linspace(d.R_slide, d.R_bore, gl + 1)) + [d.R_bore + 0.5 * d.s_wedge] \
        + list(np.linspace(s1, s2, 3 * rf + 1)) + list(np.linspace(s2, s3, 3 * rf + 1)[1:]) \
        + list(np.geomspace(s3, d.R_stator_out, 6 * rf + 1)[1:])
    stator_r = sorted(set(np.round(stator_r, 9)))
    s_xy, s_tri, s_rings = _build_part(stator_r, [n] * len(stator_r), center=False)

    # ---------------- regions (by element centroid) --------------------
    def polar(xy, tri):
        c = xy[tri].mean(axis=1)
        return np.hypot(c[:, 0], c[:, 1]), np.mod(np.arctan2(c[:, 1], c[:, 0]), 2 * np.pi)

    dth = 2 * np.pi / n
    # rotor: iron + one buried arc magnet per pole (centred on k * pole pitch) + flux barriers
    rr, tt = polar(r_xy, r_tri)
    r_reg = np.full(len(r_tri), ROTOR_IRON)
    r_tag = np.full(len(r_tri), -1)
    per_pole = n // d.poles
    jdiv = np.floor(np.mod(tt + np.pi / d.poles, 2 * np.pi) / dth + 1e-9).astype(int)
    pole = jdiv // per_pole                    # pole index, pole k centred at k*2pi/poles
    pos = jdiv % per_pole
    lo = (per_pole - d.magnet_divs) // 2
    in_mag = (pos >= lo) & (pos < lo + d.magnet_divs)
    bd = d.barrier_divs
    in_bar = ((pos >= lo - bd) & (pos < lo)) | ((pos >= lo + d.magnet_divs) & (pos < lo + d.magnet_divs + bd))
    mag = in_mag & (rr > d.R_core) & (rr < d.R_mag)
    bar = in_bar & (rr > d.R_core) & (rr < d.R_bridge)
    r_reg[bar] = AIR                           # flux barriers
    r_reg[mag] = MAGNET
    r_tag[mag] = pole[mag] % d.poles
    r_reg[rr > d.R_rotor] = AIR

    # stator
    rs, ts = polar(s_xy, s_tri)
    s_reg = np.full(len(s_tri), STATOR_IRON)
    s_tag = np.full(len(s_tri), -1)
    jdiv = np.floor(ts / dth + 1e-9).astype(int)
    slot = jdiv // d.div_per_slot
    pos = jdiv % d.div_per_slot
    lo = (d.div_per_slot - d.slot_divs) // 2
    in_slot = (pos >= lo) & (pos < lo + d.slot_divs)
    s_reg[rs < d.R_bore] = AIR
    s_reg[in_slot & (rs > d.R_bore) & (rs < s1)] = STATOR_WEDGE
    top = in_slot & (rs > s1) & (rs < s2)
    bot = in_slot & (rs > s2) & (rs < s3)
    s_reg[top | bot] = ARM_COIL
    s_tag[top] = 2 * slot[top]
    s_tag[bot] = 2 * slot[bot] + 1

    return Mesh(d, r_xy, r_tri, r_reg, r_tag, r_rings[-1], s_xy, s_tri, s_reg, s_tag,
                s_rings[0], s_rings[-1])


# ---------------------------------------------------------------------------
def winding_layout(d: PMDesign):
    """Double-layer, 60-degree phase-belt, chorded armature winding.

    Returns dict layer -> array (Qs,) of (phase index 0/1/2, sign) for
    layer 0 (top, next to bore) and layer 1 (bottom)."""
    q = d.Qs // (3 * d.poles)
    belts = [(0, +1), (2, -1), (1, +1), (0, -1), (2, +1), (1, -1)]
    top = [belts[(k // q) % 6] for k in range(d.Qs)]
    bot = [None] * d.Qs
    for k in range(d.Qs):
        ph, sg = top[k]
        bot[(k + d.coil_pitch) % d.Qs] = (ph, -sg)
    return {0: top, 1: bot}


def magnet_sign(d: PMDesign, pole_idx) -> np.ndarray:
    """+1 (N, magnetised outward) for even poles, -1 (S) for odd poles."""
    return np.where(np.asarray(pole_idx) % 2 == 0, 1.0, -1.0)


# ---------------------------------------------------------------------------
def design_preset(name: str = "ipm300") -> PMDesign:
    """ipm300: 300 kVAr, 0.4 kV, 4-pole 1500 rpm IPM synchronous condenser.

    Same stator as pm300 / factory300 (400 mm bore, 600 mm OD, 450 mm core, 48 skewed
    slots); 2.5 mm air gap without sleeve; the magnet thickness is sized by
    ``size_magnets`` so that rated kVAr is delivered at rated voltage."""
    if name == "ipm300":
        return PMDesign()          # defaults = ipm300
    raise KeyError(f"unknown design preset '{name}' (use ipm300)")


DESIGN_PRESETS = ("ipm300",)


def fmt_q(kvar: float) -> str:
    """Format reactive power given in kVAr."""
    return f"{kvar / 1e3:.1f} MVAr" if abs(kvar) >= 1000 else f"{kvar:.0f} kVAr"


def fmt_v(volts: float) -> str:
    return f"{volts / 1e3:g} kV" if volts >= 1000 else f"{volts:.0f} V"
