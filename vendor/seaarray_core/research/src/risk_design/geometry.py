"""Synthetic three-beacon geometry and path primitives.

Design horizontal distances and depths are the historical planning values in
experiment_config. They are a synthetic truth for this study, not a field solution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BEACONS = (2, 5, 6)
# Historical planning values. Not a posterior and not a measured baseline.
DESIGN_HORIZONTAL_M = {(5, 2): 3012.0, (5, 6): 3220.0, (2, 6): 4293.0}
DESIGN_DEPTH_M = {2: 3150.0, 5: 3265.0, 6: 3180.0}
# Fixed transducer depth for synthetic tracks. Not a logged draft.
SHIP_U_M = -10.0


def design_positions() -> dict[int, np.ndarray]:
    """Place beacon 5 at the horizontal origin and match the design horizontal ranges."""
    d52 = DESIGN_HORIZONTAL_M[(5, 2)]
    d56 = DESIGN_HORIZONTAL_M[(5, 6)]
    d26 = DESIGN_HORIZONTAL_M[(2, 6)]
    x6 = (d56 ** 2 + d52 ** 2 - d26 ** 2) / (2.0 * d52)
    y2 = d56 ** 2 - x6 ** 2
    if y2 <= 0.0:
        raise ValueError("design ranges do not form a triangle")
    y6 = math.sqrt(y2)
    pos = {
        5: np.array([0.0, 0.0, -DESIGN_DEPTH_M[5]], dtype=np.float64),
        2: np.array([d52, 0.0, -DESIGN_DEPTH_M[2]], dtype=np.float64),
        6: np.array([x6, y6, -DESIGN_DEPTH_M[6]], dtype=np.float64),
    }
    return pos


def horizontal_ranges(pos: dict[int, np.ndarray]) -> dict[str, float]:
    out = {}
    for a, b in ((5, 2), (5, 6), (2, 6)):
        d = pos[a][:2] - pos[b][:2]
        out[f"{a}-{b}"] = float(np.hypot(d[0], d[1]))
    return out


@dataclass
class Segment:
    kind: str  # line or arc
    label: str
    length: float
    x0: float
    y0: float
    h0: float
    x1: float
    y1: float
    h1: float
    # arc extras
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    a0: float = 0.0
    sweep: float = 0.0

    def point(self, s: float) -> tuple[float, float, float]:
        s = float(np.clip(s, 0.0, self.length))
        if self.kind == "line":
            if self.length < 1e-9:
                return self.x0, self.y0, self.h0
            f = s / self.length
            return (
                self.x0 + f * (self.x1 - self.x0),
                self.y0 + f * (self.y1 - self.y0),
                self.h0,
            )
        ang = self.a0 + self.sweep * (s / self.length)
        x = self.cx + self.radius * math.cos(ang)
        y = self.cy + self.radius * math.sin(ang)
        # CCW tangent heading is ang + pi/2; CW (negative sweep) is ang - pi/2.
        heading = ang + math.copysign(math.pi / 2.0, self.sweep if self.sweep != 0 else 1.0)
        return x, y, _wrap(heading)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def line_segment(x0, y0, x1, y1, h0=None, label="survey") -> Segment:
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    heading = math.atan2(dy, dx) if length > 1e-9 else (0.0 if h0 is None else h0)
    if h0 is not None and length > 1e-9 and abs(_wrap(heading - h0)) > 1e-3:
        raise ValueError("line does not continue the incoming heading")
    return Segment("line", label, length, x0, y0, heading, x1, y1, heading)


def arc_segment(x, y, heading, radius, sweep, label="turn") -> Segment:
    """Turn by sweep radians at the given radius. Positive sweep is left/CCW."""
    if radius <= 0.0:
        raise ValueError("arc radius must be positive")
    if abs(sweep) < 1e-8:
        return line_segment(x, y, x, y, heading)
    side = 1.0 if sweep > 0.0 else -1.0
    nx, ny = -math.sin(heading), math.cos(heading)  # left normal
    cx = x + radius * nx * side
    cy = y + radius * ny * side
    a0 = math.atan2(y - cy, x - cx)
    # Positive geometric angle follows CCW. A right turn decreases the center angle.
    geo = sweep
    a1 = a0 + geo
    x1 = cx + radius * math.cos(a1)
    y1 = cy + radius * math.sin(a1)
    h1 = _wrap(heading + sweep)
    return Segment(
        "arc", label, abs(sweep) * radius, x, y, heading, x1, y1, h1,
        cx=cx, cy=cy, radius=radius, a0=a0, sweep=geo,
    )


def concat(parts: list[Segment]) -> list[Segment]:
    out = [p for p in parts if p.length > 1e-6]
    for a, b in zip(out, out[1:]):
        if math.hypot(a.x1 - b.x0, a.y1 - b.y0) > 1e-4:
            raise ValueError("segments are not connected")
        if abs(_wrap(a.h1 - b.h0)) > 1e-3:
            raise ValueError("heading jumps between segments")
    return out


def path_length(parts: list[Segment]) -> float:
    return float(sum(p.length for p in parts))


def pose_at(parts: list[Segment], s: float) -> tuple[float, float, float]:
    remain = s
    if remain <= 0.0:
        p = parts[0]
        return p.x0, p.y0, p.h0
    for p in parts:
        if remain <= p.length or p is parts[-1]:
            return p.point(min(remain, p.length))
        remain -= p.length
    p = parts[-1]
    return p.x1, p.y1, p.h1


def sample_path(parts: list[Segment], speed: float, dt: float, t0: float = 0.0):
    """Constant-speed samples. Returns t, E, N, heading, segment index."""
    total = path_length(parts)
    n = int(math.floor(total / speed / dt)) + 1
    t = t0 + np.arange(n, dtype=np.float64) * dt
    # Drop a sample that would pass the path, then insert the true endpoint.
    s = (t - t0) * speed
    keep = s <= total + 1e-8
    t, s = t[keep], np.minimum(s[keep], total)
    if t.size == 0 or total - float(s[-1]) > 1e-6:
        t = np.append(t, t0 + total / speed)
        s = np.append(s, total)
    e = np.empty(t.size)
    ncoord = np.empty(t.size)
    hdg = np.empty(t.size)
    kind = []
    # Walk once.
    edges = np.cumsum([p.length for p in parts])
    for i, dist in enumerate(s):
        k = int(np.searchsorted(edges, dist, side="right"))
        k = min(k, len(parts) - 1)
        prev = 0.0 if k == 0 else edges[k - 1]
        x, y, h = parts[k].point(dist - prev)
        e[i], ncoord[i], hdg[i] = x, y, h
        kind.append(parts[k].label)
    return t, e, ncoord, hdg, kind


def _turn_mismatch(phi: float, x: float, y: float, heading: float, radius: float) -> float:
    arc = arc_segment(x, y, heading, radius, phi)
    bearing = math.atan2(-arc.y1, -arc.x1)
    return _wrap(bearing - arc.h1)


def return_to_home(x, y, heading, radius) -> list[Segment]:
    """One min-radius turn, then a straight line to the origin. Final heading is free."""
    if math.hypot(x, y) < 1.0:
        return []
    grid = np.linspace(-2.0 * math.pi + 1e-3, 2.0 * math.pi - 1e-3, 361)
    vals = [_turn_mismatch(float(phi), x, y, heading, radius) for phi in grid]
    best = None
    for i in range(len(grid) - 1):
        a, b = float(grid[i]), float(grid[i + 1])
        fa, fb = vals[i], vals[i + 1]
        if abs(fa) < 1e-8:
            roots = [a]
        elif fa * fb < 0.0 and abs(fa - fb) < math.pi:
            # Sign change that is not a wrap jump.
            lo, hi = a, b
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                fm = _turn_mismatch(mid, x, y, heading, radius)
                if fa * fm <= 0.0:
                    hi, fb = mid, fm
                else:
                    lo, fa = mid, fm
            roots = [0.5 * (lo + hi)]
        else:
            roots = []
        for phi in roots:
            arc = arc_segment(x, y, heading, radius, phi)
            if math.hypot(arc.x1, arc.y1) < 1.0:
                parts = [arc]
            else:
                parts = concat([arc, line_segment(arc.x1, arc.y1, 0.0, 0.0)])
            length = path_length(parts)
            end = pose_at(parts, length)
            if math.hypot(end[0], end[1]) > 0.5:
                continue
            if best is None or length < best[0]:
                best = (length, parts)
    if best is None:
        raise RuntimeError("no feasible return arc")
    return best[1]
