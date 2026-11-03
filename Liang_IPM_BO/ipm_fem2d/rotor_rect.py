"""Rotor with RECTANGULAR (flat-bar) interior magnets, meshed with Triangle.

Per pole (pole-local coordinates: x along the pole axis, y tangential):

            rotor surface R_rotor
         bridge t_bridge  ________
               barrier   /  C   D \\        D, C on the circle R_bridge
                        /B        E/
      magnet  (x_in..x_out, |y| <= w/2)    B = (x_out, w/2), A = (x_in, w/2)
                        \\A________/

* magnet: rectangle x in [x_in, x_out], y in [-w/2, w/2], x_out = R_rotor - d_m,
  magnetised PARALLEL to the pole axis (N: +x, S: -x);
* flux barriers: polygon A-B-C-(arc)-D-E at both magnet ends, stopping at
  R_bridge = R_rotor - t_bridge;
* everything else up to R_rotor is solid rotor steel (q-axis iron between poles).

The disk r <= R_rotor is meshed with a constrained Delaunay triangulation whose
outer boundary nodes are exactly the n_theta equally spaced nodes used by the
structured air-gap rings, so the sliding-interface machinery is unchanged.
"""
from __future__ import annotations

import math

import numpy as np

KIND_IRON, KIND_BARRIER, KIND_MAGNET = 1, 2, 10      # magnet attribute = 10 + pole index


def layout(d):
    """Pole-0 geometry in pole-local coordinates (metres / radians).

    The magnet width is the smaller of
      * the width whose OUTER corner stays ``corner_margin`` below the bridge circle, and
      * the width whose INNER corner leaves a q-axis iron rib >= rib_min to the next pole
        (plus the barrier wall) -- otherwise neighbouring magnets/barriers would collide.
    """
    R, Rb = d.R_rotor, d.R_rotor - d.t_bridge
    x_out = R - d.d_m
    x_in = x_out - d.h_m
    half = math.pi / d.poles
    rib_min = max(0.003, 2 * d.t_bridge)
    dE = math.radians(1.5)                             # barrier wall beyond the inner corner
    r_corner = Rb - d.corner_margin
    if r_corner <= x_out:
        raise ValueError("magnet too close to the rotor surface (increase d_m or reduce t_bridge)")
    w_corner = 2 * math.sqrt(r_corner**2 - x_out**2)
    th_A_max = half - rib_min / max(x_in, 1e-6) - dE
    w_rib = 2 * x_in * math.tan(th_A_max) if th_A_max > 0 else 0.0
    # the barrier opening along the bridge must also end before the q-axis rib
    th_B_max = half - rib_min / Rb - math.radians(d.barrier_gap_deg + d.barrier_deg)
    w_bar = 2 * x_out * math.tan(th_B_max) if th_B_max > 0 else 0.0
    w = d.w_frac * min(w_corner, w_rib, w_bar)
    A = (x_in, w / 2)
    B = (x_out, w / 2)
    th_A = math.atan2(A[1], A[0])
    th_B = math.atan2(B[1], B[0])
    th_C = th_B + math.radians(d.barrier_gap_deg)
    th_D = th_C + math.radians(d.barrier_deg)
    th_E = max(th_D, th_A + dE)
    rA = math.hypot(*A)
    E = (rA * math.cos(th_E), rA * math.sin(th_E))
    arc = [(Rb * math.cos(t), Rb * math.sin(t)) for t in np.linspace(th_C, th_D, 6)]
    barrier = [A, B] + arc + [E]                      # upper barrier (y > 0)
    return dict(x_in=x_in, x_out=x_out, w=w, w_corner=w_corner, w_rib=w_rib, w_bar=w_bar, A=A, B=B, E=E,
                th_A=th_A, th_B=th_B, th_C=th_C, th_D=th_D, th_E=th_E, rib_min=rib_min,
                R_bridge=Rb, barrier_upper=barrier,
                barrier_lower=[(x, -y) for (x, y) in barrier],
                magnet=[(x_in, -w / 2), (x_out, -w / 2), (x_out, w / 2), (x_in, w / 2)])


def check(d):
    g = layout(d)
    half = math.pi / d.poles
    if g["x_in"] < 0.35 * d.R_rotor:
        raise ValueError("magnet too thick / too deep for the rotor")
    if g["w"] < 0.01:
        raise ValueError("no room for a flat magnet in this pole pitch")
    # q-axis rib between this pole's barrier and the next pole's barrier
    if (half - g["th_D"]) * g["R_bridge"] < 0.5 * g["rib_min"] or \
            (half - g["th_E"]) * math.hypot(*g["A"]) < 0.5 * g["rib_min"]:
        raise ValueError("flux barriers of adjacent poles overlap (q-axis rib too thin)")
    from matplotlib.path import Path
    P = np.array(g["barrier_upper"])
    # simple-polygon sanity check: barrier must not contain the magnet centre or the rotor centre
    if Path(P).contains_point(((g["x_in"] + g["x_out"]) / 2, 0.0)) or Path(P).contains_point((0.0, 0.0)):
        raise ValueError("flux-barrier polygon is degenerate")
    return g


def _rot(pts, a):
    c, s = math.cos(a), math.sin(a)
    return [(c * x - s * y, s * x + c * y) for (x, y) in pts]


def pole_polygons(d, k, rot=0.0):
    """Magnet and the two barrier polygons of pole k in rotor coordinates."""
    g = layout(d)
    a = 2 * math.pi * k / d.poles + rot
    return _rot(g["magnet"], a), _rot(g["barrier_upper"], a), _rot(g["barrier_lower"], a)


def interior_point(poly):
    """A point strictly inside a (possibly concave) simple polygon, for Triangle region seeds."""
    from matplotlib.path import Path
    P = np.array(poly)
    path = Path(P)
    c = P.mean(axis=0)
    if path.contains_point(c):
        return c
    # try midpoints between non-adjacent vertices, then small shifts from each edge midpoint
    m = len(P)
    for i in range(m):
        for j in range(i + 2, m):
            q = 0.5 * (P[i] + P[j])
            if path.contains_point(q):
                return q
    raise RuntimeError("could not find an interior point of a flux-barrier polygon")


def build(d, n):
    """Triangulate the rotor disk.  Returns xy, tri, region kind attribute, boundary ids (0..n-1)."""
    import triangle as tr
    R = d.R_rotor
    g = check(d)
    th = 2 * np.pi * np.arange(n) / n
    verts = [(R * math.cos(t), R * math.sin(t)) for t in th]
    segs = [(i, (i + 1) % n) for i in range(n)]
    regions = []
    h_s = 2 * math.pi * R / n                          # boundary node spacing
    a_iron = 0.9 * h_s**2 * 4
    a_fine = 0.45 * min(h_s, d.h_m / 3) ** 2

    def add_poly(pts):
        i0 = len(verts)
        verts.extend(pts)
        m = len(pts)
        segs.extend((i0 + j, i0 + (j + 1) % m) for j in range(m))
        return list(range(i0, i0 + m))

    for k in range(d.poles):
        mag, bu, bl = pole_polygons(d, k)
        # shared vertices: magnet corners A/B are also barrier vertices -> add magnet once,
        # then barriers re-using its corner indices
        im = add_poly(mag)                             # (x_in,-w/2),(x_out,-w/2),(x_out,w/2),(x_in,w/2)
        for bar, (ia, ib) in ((bu, (im[3], im[2])), (bl, (im[0], im[1]))):
            rest = bar[2:]
            i0 = len(verts)
            verts.extend(rest)
            ids = [ia, ib] + list(range(i0, i0 + len(rest)))
            segs.extend((ids[j], ids[(j + 1) % len(ids)]) for j in range(len(ids)) if j != 0)
            # (segment A-B is the magnet end, already present)
        cx, cy = np.mean(np.array(mag), axis=0)
        regions.append([cx, cy, KIND_MAGNET + k, a_fine])
        for bar in (bu, bl):
            bx, by = interior_point(bar)
            regions.append([bx, by, KIND_BARRIER, a_fine])
    regions.append([0.0, 0.0, KIND_IRON, a_iron])
    A = dict(vertices=np.array(verts), segments=np.array(segs), regions=np.array(regions))
    T = tr.triangulate(A, "pq28YAaQ")
    xy = T["vertices"]
    tri = T["triangles"].astype(int)
    kind = T["triangle_attributes"][:, 0].astype(int)
    # orientation check (Triangle returns CCW)
    p = xy[tri]
    ar = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    flip = ar < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    if not np.allclose(xy[:n], np.array(verts[:n])):
        raise RuntimeError("Triangle moved the rotor boundary nodes")
    return xy, tri, kind, np.arange(n)


def magnet_direction(d, pole_idx):
    """Parallel magnetisation unit vector of pole k (N outward along its axis, S inward)."""
    a = 2 * np.pi * np.asarray(pole_idx) / d.poles
    s = np.where(np.asarray(pole_idx) % 2 == 0, 1.0, -1.0)
    return np.c_[s * np.cos(a), s * np.sin(a)]
