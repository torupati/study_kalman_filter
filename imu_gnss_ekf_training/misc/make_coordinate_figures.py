"""Regenerate the coordinate-frame figures used by doc/coordinate_frames.md.

Usage: uv run python imu_gnss_ekf_training/misc/make_coordinate_figures.py (writes to imu_gnss_ekf_training/figures/)
"""
import argparse
import logging
import math
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"

logger = logging.getLogger(__name__)

NAV = "#333333"
XC = "#d62728"
YC = "#2ca02c"
ZC = "#1f77b4"
YAW = "#ff7f0e"
CAR_FILL = "#dfe7f2"
CAR_EDGE = "#4a5a70"

YAW_DEG = 35.0


def defs(colors):
    parts = ["<defs>"]
    for name, c in colors.items():
        parts.append(
            f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        )
    parts.append("</defs>")
    return "\n".join(parts)


def arrow(p, q, color, name, width=2.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" stroke="{color}" '
            f'stroke-width="{width}"{d} marker-end="url(#ah-{name})"/>')


def text(p, s, color="#222", size=18, anchor="middle", italic=True, weight="normal"):
    style = "italic" if italic else "normal"
    return (f'<text x="{p[0]:.1f}" y="{p[1]:.1f}" fill="{color}" font-size="{size}" font-family="serif" '
            f'font-style="{style}" font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="middle">{s}</text>')


def sub(base, s, rest=""):
    tail = f'<tspan dy="-5">{rest}</tspan>' if rest else ""
    return f'{base}<tspan dy="5" font-size="13">{s}</tspan>{tail}'


COLORS = {"nav": NAV, "x": XC, "y": YC, "z": ZC, "yaw": YAW}


# ---------------------------------------------------------------- top view
def top_view(out_dir: Path) -> None:
    W, H = 560, 440
    o = (70, 380)
    car = (290, 230)
    psi = math.radians(YAW_DEG)
    fwd = (math.cos(psi), -math.sin(psi))       # screen coords (y down)
    left = (-math.sin(psi), -math.cos(psi))
    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    s.append(defs(COLORS))
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(text((W / 2, 24), "Top view (looking down the z axes)", size=17, italic=False, weight="bold"))

    # navigation frame
    s.append(arrow(o, (510, o[1]), NAV, "nav"))
    s.append(arrow(o, (o[0], 60), NAV, "nav"))
    s.append(text((522, o[1] + 22), sub("x", "n"), NAV))
    s.append(text((o[0] - 22, 62), sub("y", "n"), NAV))
    s.append(f'<circle cx="{o[0]}" cy="{o[1]}" r="9" fill="white" stroke="{NAV}" stroke-width="2"/>')
    s.append(f'<circle cx="{o[0]}" cy="{o[1]}" r="3" fill="{NAV}"/>')
    s.append(text((o[0] - 40, o[1] + 22), sub("z", "n", " ⊙"), NAV, size=16, anchor="start"))
    s.append(text((o[0] + 4, o[1] + 44), "O (local origin)", NAV, size=14, italic=False))

    # position projections
    s.append(f'<line x1="{car[0]}" y1="{car[1]}" x2="{car[0]}" y2="{o[1]}" stroke="#999" stroke-dasharray="4 4"/>')
    s.append(f'<line x1="{car[0]}" y1="{car[1]}" x2="{o[0]}" y2="{car[1]}" stroke="#999" stroke-dasharray="4 4"/>')
    s.append(text((car[0], o[1] + 20), "x", "#555"))
    s.append(text((o[0] - 16, car[1]), "y", "#555"))

    # car body (drawn in body coordinates, then rotated)
    s.append(f'<g transform="translate({car[0]} {car[1]}) rotate({-YAW_DEG})">')
    for wx, wy in [(-38, -31), (-38, 25), (28, -31), (28, 25)]:
        s.append(f'<rect x="{wx}" y="{wy}" width="22" height="7" rx="2" fill="#555"/>')
    s.append(f'<path d="M-60,-27 L42,-27 Q62,-27 62,0 Q62,27 42,27 L-60,27 Q-64,27 -64,23 L-64,-23 Q-64,-27 -60,-27 z" '
             f'fill="{CAR_FILL}" stroke="{CAR_EDGE}" stroke-width="2"/>')
    s.append(f'<path d="M22,-20 Q34,0 22,20" fill="none" stroke="{CAR_EDGE}" stroke-width="2"/>')
    s.append("</g>")

    # yaw reference and arc
    s.append(f'<line x1="{car[0]}" y1="{car[1]}" x2="{car[0] + 170}" y2="{car[1]}" stroke="{NAV}" stroke-dasharray="6 4"/>')
    s.append(text((car[0] + 176, car[1] + 2), "∥ " + sub("x", "n"), NAV, size=15, anchor="start"))
    r = 115
    end = (car[0] + r * math.cos(psi), car[1] - r * math.sin(psi))
    s.append(f'<path d="M{car[0] + r},{car[1]} A{r},{r} 0 0 0 {end[0]:.1f},{end[1]:.1f}" fill="none" '
             f'stroke="{YAW}" stroke-width="2.5" marker-end="url(#ah-yaw)"/>')
    mid = psi / 2
    s.append(text((car[0] + (r + 18) * math.cos(mid), car[1] - (r + 18) * math.sin(mid)), "ψ", YAW, size=22))

    # body frame
    fl, ll = 150, 105
    xb = (car[0] + fl * fwd[0], car[1] + fl * fwd[1])
    yb = (car[0] + ll * left[0], car[1] + ll * left[1])
    s.append(arrow(car, xb, XC, "x", width=3))
    s.append(arrow(car, yb, YC, "y", width=3))
    s.append(text((xb[0] + 14, xb[1] - 12), sub("x", "b", " (forward)"), XC, anchor="start"))
    s.append(text((yb[0] - 40, yb[1] - 16), sub("y", "b", " (left)"), YC, anchor="start"))
    s.append(f'<circle cx="{car[0]}" cy="{car[1]}" r="8" fill="white" stroke="{ZC}" stroke-width="2"/>')
    s.append(f'<circle cx="{car[0]}" cy="{car[1]}" r="3" fill="{ZC}"/>')
    s.append(text((car[0] + 70, car[1] + 30), sub("z", "b", " ⊙ (up, at IMU)"), ZC, size=15, anchor="start"))

    # gyro rotation sense around z_b
    s.append(f'<path d="M{car[0] - 30},{car[1] + 58} A26,10 0 1 0 {car[0] + 6},{car[1] + 64}" fill="none" '
             f'stroke="{ZC}" stroke-width="1.8" marker-end="url(#ah-z)"/>')
    s.append(text((car[0] + 30, car[1] + 76), "ω (CCW +)", ZC, size=14, anchor="start"))

    s.append(text((W - 12, H - 14), "⊙ = axis points out of the page (up)", "#555", size=13, anchor="end", italic=False))
    s.append("</svg>")
    path = out_dir / "coordinate_frames_top.svg"
    path.write_text("\n".join(s) + "\n")
    logger.info("wrote %s", path)


# ---------------------------------------------------------------- 3D view
def make_projector(az_deg, el_deg, scale, origin):
    az, el = math.radians(az_deg), math.radians(el_deg)
    c = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    f = (-c[0], -c[1], -c[2])
    r = (f[1], -f[0], 0.0)
    n = math.hypot(r[0], r[1])
    r = (r[0] / n, r[1] / n, 0.0)
    u = (r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0])

    def proj(p):
        sx = sum(a * b for a, b in zip(p, r))
        sy = sum(a * b for a, b in zip(p, u))
        return (origin[0] + scale * sx, origin[1] - scale * sy)

    def depth(p):  # larger = closer to camera
        return sum(a * b for a, b in zip(p, c))

    return proj, depth


def three_d_view(out_dir: Path) -> None:
    W, H = 620, 400
    proj, depth = make_projector(-74, 30, 60, (80, 300))
    psi = math.radians(YAW_DEG)
    cp, sp = math.cos(psi), math.sin(psi)
    center = (4.0, 2.6)

    def body(bx, by, bz):
        return (center[0] + cp * bx - sp * by, center[1] + sp * bx + cp * by, bz)

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    s.append(defs(COLORS))
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(text((W / 2, 24), "3D view (both frames right-handed, z up)", size=17, italic=False, weight="bold"))

    # ground grid (z = 0 plane)
    for i in range(0, 7):
        a, b = proj((i, 0, 0)), proj((i, 5, 0))
        s.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#e3e3e3"/>')
    for j in range(0, 6):
        a, b = proj((0, j, 0)), proj((6, j, 0))
        s.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#e3e3e3"/>')
    s.append(text(proj((6.3, 2.2, 0)), "horizontal plane (z = 0)", "#999", size=13, italic=False))

    # nav axes
    origin = proj((0, 0, 0))
    s.append(arrow(origin, proj((6.6, 0, 0)), NAV, "nav"))
    s.append(arrow(origin, proj((0, 5.6, 0)), NAV, "nav"))
    s.append(arrow(origin, proj((0, 0, 3.0)), NAV, "nav"))
    s.append(text(proj((6.9, -0.25, 0)), sub("x", "n"), NAV))
    s.append(text(proj((-0.3, 5.9, 0)), sub("y", "n"), NAV))
    s.append(text(proj((-0.35, 0, 3.0)), sub("z", "n"), NAV))
    s.append(text((origin[0] - 14, origin[1] + 14), "O", NAV, size=15))

    # car box, painter's algorithm
    L, Wd, Hh = 1.2, 0.6, 0.55
    v = {}
    for ix, bx in enumerate((-L, L)):
        for iy, by in enumerate((-Wd, Wd)):
            for iz, bz in enumerate((0.0, Hh)):
                v[(ix, iy, iz)] = body(bx, by, bz)
    faces = [
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
        [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    ]
    shade = {1: "#c9d6e8", 5: "#aebfd8"}
    order = sorted(range(6), key=lambda k: sum(depth(v[i]) for i in faces[k]))
    for k in order:
        pts = " ".join(f"{proj(v[i])[0]:.1f},{proj(v[i])[1]:.1f}" for i in faces[k])
        s.append(f'<polygon points="{pts}" fill="{shade.get(k, CAR_FILL)}" fill-opacity="0.9" stroke="{CAR_EDGE}" stroke-width="1.5"/>')

    # yaw reference and arc, drawn on the car-top plane
    ob = body(0, 0, Hh)
    s.append(f'<line x1="{proj(ob)[0]:.1f}" y1="{proj(ob)[1]:.1f}" x2="{proj((ob[0] + 2.3, ob[1], Hh))[0]:.1f}" '
             f'y2="{proj((ob[0] + 2.3, ob[1], Hh))[1]:.1f}" stroke="{NAV}" stroke-dasharray="6 4"/>')
    s.append(text(proj((ob[0] + 2.4, ob[1], Hh)), "∥ " + sub("x", "n"), NAV, size=15, anchor="start"))
    arc_r = 1.9
    arc = [proj((ob[0] + arc_r * math.cos(t), ob[1] + arc_r * math.sin(t), Hh))
           for t in [psi * k / 20 for k in range(21)]]
    s.append(f'<polyline points="{" ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in arc)}" fill="none" stroke="{YAW}" '
             f'stroke-width="2.5" marker-end="url(#ah-yaw)"/>')
    s.append(text(proj((ob[0] + (arc_r + 0.35) * math.cos(psi / 2), ob[1] + (arc_r + 0.35) * math.sin(psi / 2), Hh)),
                  "ψ", YAW, size=22))

    # body axes from car top center (IMU)
    s.append(arrow(proj(ob), proj(body(2.6, 0, Hh)), XC, "x", width=3))
    s.append(arrow(proj(ob), proj(body(0, 1.9, Hh)), YC, "y", width=3))
    s.append(arrow(proj(ob), proj(body(0, 0, Hh + 2.0)), ZC, "z", width=3))
    s.append(text(proj(body(2.95, 0, Hh)), sub("x", "b"), XC))
    s.append(text(proj(body(0, 2.2, Hh)), sub("y", "b"), YC))
    zt = proj(body(0, 0, Hh + 2.0))
    s.append(text((zt[0] + 14, zt[1] - 6), sub("z", "b"), ZC, anchor="start"))
    s.append(f'<circle cx="{proj(ob)[0]:.1f}" cy="{proj(ob)[1]:.1f}" r="4" fill="#222"/>')
    s.append(text((proj(ob)[0] - 10, proj(ob)[1] + 16), "IMU", "#222", size=12, anchor="end", italic=False))

    # gyro rotation sense around z_b
    ring = [proj((ob[0] + 0.45 * math.cos(t), ob[1] + 0.45 * math.sin(t), Hh + 1.45))
            for t in [math.radians(-40 + 12 * k) for k in range(26)]]
    s.append(f'<polyline points="{" ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in ring)}" fill="none" stroke="{ZC}" '
             f'stroke-width="1.8" marker-end="url(#ah-z)"/>')
    s.append(text((ring[0][0] + 14, ring[0][1] + 2), "ω", ZC, size=16, anchor="start"))

    s.append("</svg>")
    path = out_dir / "coordinate_frames_3d.svg"
    path.write_text("\n".join(s) + "\n")
    logger.info("wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the navigation/body coordinate-frame figures as SVG.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_view(args.output_dir)
    three_d_view(args.output_dir)


if __name__ == "__main__":
    main()
