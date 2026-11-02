"""Synchronous-condenser design data and polar finite-element mesh.

Example machine: 2-pole, cylindrical (round) rotor, air/hydrogen-cooled
synchronous condenser, 100 MVA, 13.8 kV, 50 Hz, 3000 rpm.

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
AIR, ROTOR_IRON, STATOR_IRON, FIELD_COIL, ARM_COIL, ROTOR_WEDGE, STATOR_WEDGE = range(7)
REGION_NAMES = {AIR: "air gap", ROTOR_IRON: "rotor iron", STATOR_IRON: "stator iron",
                FIELD_COIL: "field winding", ARM_COIL: "armature winding",
                ROTOR_WEDGE: "rotor wedge", STATOR_WEDGE: "stator slot opening"}


@dataclass
class SCDesign:
    # ---- rating ---------------------------------------------------------
    S_rated: float = 100e6        # VA (rated over-excited reactive output, MVAr)
    V_line: float = 13.8e3        # V rms line-line
    f: float = 50.0               # Hz
    poles: int = 2
    # ---- main dimensions ---------------------------------------------
    L_stack: float = 2.9          # m, active core length
    stacking: float = 0.95        # lamination stacking factor (loss/mass only)
    R_shaft_center: float = 0.0
    R_rotor: float = 0.50         # rotor outer radius
    gap: float = 0.060            # mechanical air gap
    R_stator_out: float = 1.12    # stator core outer radius
    # ---- stator slots / winding --------------------------------------
    Qs: int = 48                  # stator slots
    div_per_slot: int = 8         # angular divisions per stator slot pitch
    slot_divs: int = 4            # divisions occupied by the slot (=> slot/pitch = 0.5)
    s_wedge: float = 0.020        # slot-opening/wedge depth
    s_layer: float = 0.075        # depth of each winding layer (2 layers)
    coil_pitch: int = 20          # slots (full pitch = Qs/poles = 24 -> 5/6 chording)
    parallel_paths: int = 1
    turns_per_coil: int = 1       # Roebel bars: 1 turn per coil
    # ---- rotor slots / field winding ---------------------------------
    Qr_pitches: int = 32          # rotor slot pitches around the periphery
    Qr_omit_per_pole: int = 4     # unslotted pitches forming each pole face
    r_wedge: float = 0.020
    r_slot_depth: float = 0.150   # incl. wedge
    rotor_slot_divs: int = 6      # of 12 divisions per rotor slot pitch
    Nf_per_slot: int = 10         # field turns per rotor slot
    # ---- extra (3-D) effects not captured by the 2-D model -----------
    X_end_pu: float = 0.06        # end-winding leakage reactance [pu]
    # ---- loss data --------------------------------------------------
    rho_cu: float = 2.1e-8        # ohm m at ~75 C
    fill_arm: float = 0.55        # copper fill of armature layer
    fill_field: float = 0.60
    end_arm: float = 1.9          # m, end-winding length per half-turn
    end_field: float = 1.1
    kh: float = 0.020             # W/kg/(T^2 Hz)   hysteresis   (M330-50A class)
    ke: float = 1.9e-4            # W/kg/(T^2 Hz^2) eddy current
    build_factor: float = 1.5
    rho_fe: float = 7650.0
    # ---- mesh density -----------------------------------------------
    gap_layers: int = 6

    # derived ---------------------------------------------------------
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

    def winding_factor(self) -> float:
        q = self.Qs / (3 * self.poles)
        alpha = math.pi * self.poles / self.Qs
        kd = math.sin(q * alpha / 2) / (q * math.sin(alpha / 2))
        kp = math.sin(math.pi / 2 * self.coil_pitch / (self.Qs / self.poles))
        return kd * kp

    def check(self):
        if self.poles != 2:
            raise NotImplementedError("the example layout generator handles 2-pole machines")
        if self.n_theta % self.Qr_pitches:
            raise ValueError("n_theta must be a multiple of the rotor slot pitches")
        if (self.n_theta // self.Qr_pitches) % 2:
            raise ValueError("divisions per rotor slot pitch must be even")


# ---------------------------------------------------------------------------
@dataclass
class Mesh:
    """Rotor + stator triangle mesh joined at the sliding circle."""
    design: SCDesign
    # rotor part
    r_xy: np.ndarray = field(default=None)        # rotor node coords (rotor frame)
    r_tri: np.ndarray = field(default=None)
    r_reg: np.ndarray = field(default=None)
    r_tag: np.ndarray = field(default=None)       # rotor slot index or -1
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


def build_mesh(d: SCDesign) -> Mesh:
    d.check()
    n = d.n_theta
    # ---------------- rotor radii -------------------------------------
    r_sb = d.R_rotor - d.r_slot_depth          # slot bottom
    r_wb = d.R_rotor - d.r_wedge               # wedge bottom / conductor top
    inner = []
    nn = n
    while nn % 2 == 0 and nn > 24:
        nn //= 2
        inner.append(nn)
    inner = inner[::-1]                        # e.g. 24, 48, 96, 192
    r_inner = list(np.linspace(0, r_sb, len(inner) + 2)[1:-1])
    rotor_r = r_inner + list(np.linspace(r_sb, r_wb, 6)) + [r_wb + 0.5 * d.r_wedge, d.R_rotor]
    gl = d.gap_layers // 2
    rotor_r += list(np.linspace(d.R_rotor, d.R_slide, gl + 1)[1:])
    rotor_n = inner + [n] * (len(rotor_r) - len(inner))
    r_xy, r_tri, r_rings = _build_part(rotor_r, rotor_n, center=True)

    # ---------------- stator radii ------------------------------------
    s1 = d.R_bore + d.s_wedge
    s2 = s1 + d.s_layer
    s3 = s2 + d.s_layer
    stator_r = list(np.linspace(d.R_slide, d.R_bore, gl + 1)) + [d.R_bore + 0.5 * d.s_wedge] \
        + list(np.linspace(s1, s2, 4)) + list(np.linspace(s2, s3, 4)[1:]) \
        + list(np.geomspace(s3, d.R_stator_out, 7)[1:])
    stator_r = sorted(set(np.round(stator_r, 9)))
    s_xy, s_tri, s_rings = _build_part(stator_r, [n] * len(stator_r), center=False)

    # ---------------- regions (by element centroid) --------------------
    def polar(xy, tri):
        c = xy[tri].mean(axis=1)
        return np.hypot(c[:, 0], c[:, 1]), np.mod(np.arctan2(c[:, 1], c[:, 0]), 2 * np.pi)

    dth = 2 * np.pi / n
    # rotor
    rr, tt = polar(r_xy, r_tri)
    r_reg = np.full(len(r_tri), ROTOR_IRON)
    r_tag = np.full(len(r_tri), -1)
    dpr = n // d.Qr_pitches
    jdiv = np.floor(tt / dth + 1e-9).astype(int)
    pitch = jdiv // dpr
    pos = jdiv % dpr
    lo = (dpr - d.rotor_slot_divs) // 2
    in_slot = (pos >= lo) & (pos < lo + d.rotor_slot_divs)
    centre = (pitch + 0.5) * 2 * np.pi / d.Qr_pitches
    half_pole = d.Qr_omit_per_pole / 2 * 2 * np.pi / d.Qr_pitches
    ang_to_pole = np.minimum(np.abs((centre + np.pi) % (2 * np.pi) - np.pi),
                             np.abs(((centre - np.pi) + np.pi) % (2 * np.pi) - np.pi))
    wound = ang_to_pole > half_pole
    slot_zone = in_slot & wound & (rr > r_sb) & (rr < d.R_rotor)
    r_reg[slot_zone & (rr < r_wb)] = FIELD_COIL
    r_reg[slot_zone & (rr >= r_wb)] = ROTOR_WEDGE
    r_tag[r_reg == FIELD_COIL] = pitch[r_reg == FIELD_COIL]
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
def winding_layout(d: SCDesign):
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


def field_sign(d: SCDesign, pitch_idx: np.ndarray) -> np.ndarray:
    """+1 for rotor slots in the upper half (0..pi), -1 in the lower half."""
    centre = (pitch_idx + 0.5) * 2 * np.pi / d.Qr_pitches
    return np.where(centre < np.pi, 1.0, -1.0)
