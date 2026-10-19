"""
Procedural spider specimen generator for Blender (4.2+ / 5.x).

Builds a real-scale specimen (1 Blender unit = 1 mm, scene unit scale 0.001):
  * one connected all-quad mesh (prosoma + abdomen + 8 legs + palps + chelicerae) + 8 eyes,
  * Subdivision Surface, UVs, vertex groups named for rigging (leg1.femur.L ...),
  * procedural Cycles material (resolution independent, good for macro close-ups),
  * optional particle hair (setae) coloured from a baked copy of the body colour.

Morphology follows published descriptions (Dankittipakul et al. 2010-2012, Zhang & Zhang
on M. sphaerica / M. pluma): elongate-oval dark carapace with a black median furrow, compact
eye group (AME largest), dark abdomen with 3 pairs of white dorso-lateral patches + a posterior
transverse patch, reddish-brown shiny dorsal scutum in males, venter grey-black with 3 white
stripes, pale spinnerets, long legs with formula 4123, yellowish coxae, darker femora.

Usage
  blender -b -P mallinella_generator.py -- --preset sphaerica_m --seed 3 --out specimens/m3.blend
  blender -b -P mallinella_generator.py -- --preset random --seed 12 --hair 1.0 --out x.blend
Presets: sphaerica_f, sphaerica_m, pluma_m, ctenus_generic,
macroctenus_generic, amicactenus_generic, random
"""
import bpy, bmesh, math, sys, json, argparse, random, os
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

# ------------------------------------------------------------------ arguments
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--preset', default='random')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--hair', type=float, default=1.0, help='hair density multiplier (0 = no hair)')
ap.add_argument('--bake-res', type=int, default=1024, help='resolution of the colour map used by the hair')
ap.add_argument('--out', default='mallinella.blend')
args = ap.parse_args(argv)
R = random.Random(args.seed)
# ------------------------------------------------------------------ random jitter
# This function applies a small random jitter to a value, within a specified relative range 
# in order to introduce variability in the generated spider model. The 'rel' parameter defines the maximum relative change allowed.
def jit(v, rel=0.06):
    return v * (1 + R.uniform(-rel, rel))

# ------------------------------------------------------------------ presets (mm, linear-RGB colours)
PRESETS = {
    # M. sphaerica female (paratype: total 8.26, prosoma 3.57x2.55, opisthosoma 4.28x2.75)
    'sphaerica_f': dict(sex='F', Lp=3.57, Wp=2.55, La=4.28, Wa=2.75, Ha=2.5,
                        legs=[8.68, 8.47, 8.00, 10.34], scutum=0.12,
                        carapace=(0.022, 0.012, 0.008), legs_col=(0.075, 0.045, 0.022),
                        femur_col=(0.030, 0.017, 0.010), coxa_col=(0.20, 0.13, 0.06),
                        abd_col=(0.010, 0.008, 0.008), patch_col=(0.62, 0.60, 0.55),
                        scutum_col=(0.10, 0.030, 0.014), hair=1.0, bands=0.0),
    # M. sphaerica male (holotype: total 6.27, prosoma 3.11x2.14, opisthosoma 2.96x1.99)
    'sphaerica_m': dict(sex='M', Lp=3.11, Wp=2.14, La=2.96, Wa=1.99, Ha=1.75,
                        legs=[10.36, 9.90, 9.29, 12.09], scutum=1.0,
                        carapace=(0.020, 0.011, 0.007), legs_col=(0.08, 0.048, 0.024),
                        femur_col=(0.030, 0.017, 0.010), coxa_col=(0.18, 0.12, 0.055),
                        abd_col=(0.010, 0.008, 0.008), patch_col=(0.62, 0.60, 0.55),
                        scutum_col=(0.11, 0.032, 0.014), hair=0.8, bands=0.0),
    # M. pluma male (holotype: total 5.30, prosoma 2.86x2.04, opisthosoma 2.24x1.84), hairy black abdomen
    'pluma_m': dict(sex='M', Lp=2.86, Wp=2.04, La=2.24, Wa=1.84, Ha=1.6,
                    legs=[9.51, 8.83, 8.57, 11.22], scutum=0.2,
                    carapace=(0.035, 0.018, 0.009), legs_col=(0.25, 0.165, 0.075),
                    femur_col=(0.12, 0.072, 0.034), coxa_col=(0.30, 0.21, 0.10),
                    abd_col=(0.008, 0.007, 0.007), patch_col=(0.60, 0.58, 0.52),
                    scutum_col=(0.03, 0.015, 0.010), hair=1.6, bands=0.0),
    # Generic Ctenidae morphotypes. These are synthetic genus-level profiles,
    # not species-specific reconstructions.
    'ctenus_generic': dict(family='Ctenidae', genus='Ctenus', sex='F',
                           Lp=5.1, Wp=3.7, La=6.4, Wa=4.1, Ha=3.1,
                           legs=[14.5, 16.0, 17.5, 18.5], scutum=0.0,
                           carapace=(0.045, 0.025, 0.012), legs_col=(0.12, 0.075, 0.035),
                           femur_col=(0.055, 0.030, 0.016), coxa_col=(0.22, 0.14, 0.06),
                           abd_col=(0.018, 0.014, 0.012), patch_col=(0.45, 0.40, 0.30),
                           scutum_col=(0.02, 0.012, 0.008), hair=1.25, bands=0.25),
    'macroctenus_generic': dict(family='Ctenidae', genus='Macroctenus', sex='F',
                                Lp=7.0, Wp=5.0, La=8.5, Wa=5.8, Ha=4.2,
                                legs=[20.0, 22.0, 24.0, 25.0], scutum=0.0,
                                carapace=(0.035, 0.020, 0.012), legs_col=(0.10, 0.060, 0.030),
                                femur_col=(0.045, 0.025, 0.014), coxa_col=(0.18, 0.11, 0.05),
                                abd_col=(0.012, 0.011, 0.010), patch_col=(0.52, 0.48, 0.38),
                                scutum_col=(0.02, 0.012, 0.008), hair=1.0, bands=0.15),
    'amicactenus_generic': dict(family='Ctenidae', genus='Amicactenus', sex='M',
                                Lp=4.4, Wp=3.2, La=5.0, Wa=3.5, Ha=2.7,
                                legs=[13.0, 14.5, 15.5, 16.5], scutum=0.35,
                                carapace=(0.028, 0.016, 0.010), legs_col=(0.16, 0.09, 0.040),
                                femur_col=(0.070, 0.038, 0.018), coxa_col=(0.25, 0.16, 0.07),
                                abd_col=(0.014, 0.012, 0.011), patch_col=(0.58, 0.53, 0.42),
                                scutum_col=(0.08, 0.025, 0.012), hair=1.4, bands=0.35),
}
for key in ('sphaerica_f', 'sphaerica_m', 'pluma_m'):
    PRESETS[key].setdefault('family', 'Zodariidae')
    PRESETS[key].setdefault('genus', 'Mallinella')
if args.preset == 'random':
    base = dict(R.choice([PRESETS[name] for name in ('sphaerica_f', 'sphaerica_m', 'pluma_m')]))
    base['sex'] = R.choice('MF')
    s = R.uniform(0.85, 1.1)
    for k in ('Lp', 'Wp', 'La', 'Wa', 'Ha'):
        base[k] *= s
    if base['sex'] == 'F':
        base['La'] *= 1.15; base['Wa'] *= 1.1; base['Ha'] *= 1.15; base['scutum'] = R.uniform(0, 0.3)
    else:
        base['scutum'] = R.uniform(0.4, 1.0)
    base['legs'] = [l * s * R.uniform(0.9, 1.1) for l in base['legs']]
    # colour variants seen in the genus: wine-red carapace, banded legs, cream patches
    if R.random() < 0.35:
        base['carapace'] = (0.05, 0.012, 0.008)
    if R.random() < 0.3:
        base['bands'] = R.uniform(0.4, 0.9)
    f_leg = R.uniform(0.6, 2.4)                       # brightness only, keeps the brown hue
    base['legs_col'] = tuple(c * f_leg * R.uniform(0.92, 1.08) for c in base['legs_col'])
    base['patch_col'] = (R.uniform(0.5, 0.7), R.uniform(0.45, 0.65), R.uniform(0.3, 0.55))
    base['hair'] = R.uniform(0.6, 1.6)
    P = base
else:
    P = dict(PRESETS[args.preset])
for k in ('Lp', 'Wp', 'La', 'Wa', 'Ha'):
    P[k] = jit(P[k], 0.04)
P['legs'] = [jit(l, 0.03) for l in P['legs']]
# pattern jitter
P['patch_u'] = [jit(0.36, 0.06), jit(0.53, 0.05), jit(0.70, 0.04)]
P['patch_a'] = [jit(0.62, 0.1), jit(0.58, 0.1), jit(0.48, 0.12)]
P['patch_s'] = jit(1.0, 0.25)
P['bridge'] = R.uniform(0.3, 1.0)       # how strongly the 3rd pair is connected across the midline
P['post_patch'] = R.uniform(0.6, 1.0)
P['femur_el'] = R.uniform(42, 62)
P['seed'] = args.seed; P['preset'] = args.preset

Lp, Wp, La, Wa, Ha = P['Lp'], P['Wp'], P['La'], P['Wa'], P['Ha']
Wh = Wp / 2
Ht, Hb = 0.55 * Ha, 0.45 * Ha
Zb = max(0.45 * Lp, Hb - 0.12 * Ha + 0.30 * Lp / 3)
Za = Zb + 0.12 * Ha
LS = Lp / 2.1           # scale factor relative to the earlier 2.1-unit prosoma template

# ------------------------------------------------------------------ helpers
def pchip(xs, ys):
    """Monotone cubic interpolation (Fritsch-Carlson), pure python."""
    n = len(xs); h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0: m[i] = 0
        else:
            w1 = 2 * h[i] + h[i - 1]; w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    def f(x):
        x = min(max(x, xs[0]), xs[-1])
        for i in range(n - 1):
            if x <= xs[i + 1]: break
        t = (x - xs[i]) / h[i]
        h00 = 2 * t**3 - 3 * t**2 + 1; h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2; h11 = t**3 - t**2
        return h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]
    return f

def smooth(e0, e1, x):
    t = min(1, max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t)

bpy.ops.wm.read_factory_settings(use_empty=True)
bm = bmesh.new()
DL = bm.verts.layers.deform.verify()
A_PART = bm.verts.layers.float.new('part')
A_SEG = bm.verts.layers.float.new('seg')
A_SEGT = bm.verts.layers.float.new('segt')
VG = []
def vg(name):
    if name not in VG: VG.append(name)
    return VG.index(name)
def nv(co, part, weights, seg=0.0, segt=0.0):
    v = bm.verts.new(co)
    v[A_PART] = part; v[A_SEG] = seg; v[A_SEGT] = segt
    for name, w in weights.items():
        if w > 1e-4: v[DL][vg(name)] = w
    return v
def quad(a, b, c, d):
    try: return bm.faces.new((a, b, c, d))
    except ValueError: return None
SEAMS = []

def cap(ring, bulge_dir, bulge, part, weights_fn):
    n = len(ring); m = n // 4
    G = [[None] * (m + 1) for _ in range(m + 1)]
    for k in range(m + 1):
        G[k][0] = ring[k]; G[m][k] = ring[m + k]
        G[m - k][m] = ring[2 * m + k]; G[0][m - k] = ring[(3 * m + k) % n]
    B = lambda i, j: G[i][j].co
    for i in range(1, m):
        for j in range(1, m):
            u = i / m; w = j / m
            c = ((1 - w) * B(i, 0) + w * B(i, m) + (1 - u) * B(0, j) + u * B(m, j)
                 - ((1 - u) * (1 - w) * B(0, 0) + u * (1 - w) * B(m, 0) + (1 - u) * w * B(0, m) + u * w * B(m, m)))
            c = c + bulge_dir * bulge * math.sin(math.pi * u) * math.sin(math.pi * w)
            G[i][j] = nv(c, part, weights_fn(c), ring[0][A_SEG], ring[0][A_SEGT])
    faces = []
    for i in range(m):
        for j in range(m):
            f = quad(G[i][j], G[i + 1][j], G[i + 1][j + 1], G[i][j + 1])
            if f: faces.append(f)
    return faces

# ------------------------------------------------------------------ body
N = 40
def spow(s, e): return math.copysign(abs(s) ** e, s)
def ring_pts(y, w, cz, ht, hb, p):
    pts = []
    for j in range(N):
        a = 2 * math.pi * j / N; c, s = math.cos(a), math.sin(a)
        pts.append(Vector((w * spow(c, 2 / p), y, cz + (ht if s > 0 else hb) * spow(s, 2 / p))))
    return pts

y_a0 = 0.13 * Lp
def body_weights(co):
    t = smooth(0.0, y_a0 + 0.05 * La, co.y)
    return {'prosoma': 1 - t, 'abdomen': t}

# prosoma profile (u = 0 front .. 1 rear), relative to half width
pu = [0, 0.15, 0.35, 0.55, 0.72, 0.88, 1.0]
fw = pchip(pu, [0.60, 0.72, 0.88, 0.98, 1.00, 0.82, 0.45])
fht = pchip(pu, [0.55, 0.58, 0.54, 0.47, 0.40, 0.30, 0.20])
fhb = pchip(pu, [0.38, 0.40, 0.40, 0.37, 0.34, 0.27, 0.19])
rows = []         # (verts, y)
PRO = {}          # prosoma station index -> row index
def add_row(pts, part, y):
    rows.append(([nv(p, part, body_weights(Vector((0, y, 0)))) for p in pts], y))

y0 = -0.95 * Lp
def pros_ring(u, scale=1.0, yoff=0.0):
    y = y0 + 0.95 * Lp * u
    return ring_pts(y + yoff, fw(u) * Wh * scale, Zb + 0.04 * Lp * (1 - u),
                    fht(u) * Wh * scale, fhb(u) * Wh * scale, 2.5)
add_row(pros_ring(0, 0.86, -0.052 * Lp), 0, y0 - 0.052 * Lp)
add_row(pros_ring(0, 0.97, -0.024 * Lp), 0, y0 - 0.024 * Lp)
for k in range(15):
    PRO[k] = len(rows)
    add_row(pros_ring(k / 14), 0, y0 + 0.95 * Lp * k / 14)
# pedicel
rp = 0.075 * Lp
for yy in (0.05 * Lp, 0.10 * Lp):
    add_row(ring_pts(yy, rp, Zb + 0.02 * Lp, rp * 0.9, rp * 0.9, 2.0), 1, yy)
# abdomen (egg shaped, cosine spaced stations)
NS = 18
for i in range(1, NS):
    v = (1 - math.cos(math.pi * i / NS)) / 2
    vv = v ** 0.92
    r = max((4 * vv * (1 - vv)) ** 0.5, 0.0)
    y = y_a0 + La * v
    w = max(Wa / 2 * r, rp * 1.1)
    cz = Zb + 0.02 * Lp + (Za - Zb - 0.02 * Lp) * smooth(0.0, 0.3, v)
    add_row(ring_pts(y, w, cz, max(Ht * r, rp), max(Hb * r, rp), 2.0), 1, y)
lastw = rows[-1][0][0].co.x
add_row([Vector((p.x * 0.6, p.y + 0.012 * La, Za + (p.z - Za) * 0.6)) for p in [v.co for v in rows[-1][0]]], 1,
        rows[-1][1] + 0.012 * La)
for r in range(len(rows) - 1):
    A, Bv = rows[r][0], rows[r + 1][0]
    for j in range(N):
        quad(A[j], A[(j + 1) % N], Bv[(j + 1) % N], Bv[j])
    SEAMS.append((A[30], Bv[30]))
cap(rows[0][0][::-1], Vector((0, -1, 0)), 0.05 * Lp, 0, body_weights)
back_cap = cap(rows[-1][0], Vector((0, 1, 0)), 0.02 * La, 1, body_weights)

def make_hole(k, j):
    r = PRO[k]
    Rv = lambda rr, jj: rows[rr][0][jj % N]
    loop = [Rv(r - 1, j - 1), Rv(r - 1, j), Rv(r - 1, j + 1), Rv(r, j + 1),
            Rv(r + 1, j + 1), Rv(r + 1, j), Rv(r + 1, j - 1), Rv(r, j - 1)]
    c = Rv(r, j).co.copy()
    bmesh.ops.delete(bm, geom=[Rv(r, j)], context='VERTS')
    for i in range(8): SEAMS.append((loop[i], loop[(i + 1) % 8]))
    return loop, c

bm.normal_update()
bvh = BVHTree.FromBMesh(bm)

# ------------------------------------------------------------------ limbs
def perp(t):
    a = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
    n = t.cross(a); n.normalize(); return n

def bridge(loop, ring, axis):
    def orient(vs):
        c = sum((v.co for v in vs), Vector()) / len(vs)
        return sum((vs[i].co - c).cross(vs[(i + 1) % len(vs)].co - c).dot(axis) for i in range(len(vs)))
    if (orient(loop) > 0) != (orient(ring) > 0): loop = loop[::-1]
    n = len(loop)
    off = min(range(n), key=lambda o: sum((loop[(i + o) % n].co - ring[i].co).length for i in range(n)))
    L2 = [loop[(i + off) % n] for i in range(n)]
    for i in range(n): quad(L2[i], L2[(i + 1) % n], ring[(i + 1) % n], ring[i])

def build_limb(prefix, side, start, segs, part, sides=8, hole=None, tip_bulge=0.4):
    S = '.L' if side > 0 else '.R'
    pts = [start.copy()]
    for s in segs: pts.append(pts[-1] + s['dir'].normalized() * s['length'])
    rd = []
    for k, s in enumerate(segs):
        d = s['dir'].normalized(); name = f"{prefix}.{s['name']}{S}"
        prev = f"{prefix}.{segs[k - 1]['name']}{S}" if k > 0 else name
        if k > 0:
            ta = (segs[k - 1]['dir'].normalized() + d).normalized()
            rd.append((pts[k], ta, min(segs[k - 1]['r1'], s['r0']) * 0.8, k, 0.0, {prev: 0.5, name: 0.5}))
        for t in s['stations']:
            r = (s['r0'] + (s['r1'] - s['r0']) * t) * (1 + s.get('bulge', 0.1) * math.sin(math.pi * t))
            w = {name: 1.0}
            if t < 0.2 and k > 0:
                a = 0.5 + 0.5 * t / 0.2; w = {name: a, prev: 1 - a}
            if t > 0.8 and k < len(segs) - 1:
                a = 0.5 + 0.5 * (1 - t) / 0.2; w = {name: a, f"{prefix}.{segs[k + 1]['name']}{S}": 1 - a}
            rd.append((pts[k] + d * s['length'] * t, d, r, k, t, w))
    last = segs[-1]; dl = last['dir'].normalized(); lname = f"{prefix}.{last['name']}{S}"
    rd.append((pts[-1], dl, last['r1'] * 0.75, len(segs) - 1, 1.0, {lname: 1.0}))
    rings = []; n = perp(rd[0][1]); pt = rd[0][1]
    for (c, t, r, k, st, w) in rd:
        ax = pt.cross(t)
        if ax.length > 1e-6: n = Matrix.Rotation(pt.angle(t), 3, ax.normalized()) @ n
        n = (n - t * n.dot(t)).normalized(); b = t.cross(n)
        rings.append(([nv(c + (n * math.cos(2 * math.pi * i / sides) + b * math.sin(2 * math.pi * i / sides)) * r,
                          part, w, k, st) for i in range(sides)], t, k, st))
        pt = t
    bands = []
    for i in range(len(rings) - 1):
        A, Bv = rings[i][0], rings[i + 1][0]
        bands.append(([quad(A[j], A[(j + 1) % sides], Bv[(j + 1) % sides], Bv[j]) for j in range(sides)],
                      rings[i][1], rings[i][2], rings[i][3]))
    c0 = sum((v.co for v in rings[0][0]), Vector()) / sides
    col = max(range(sides), key=lambda i: -(rings[0][0][i].co - c0).normalized().z)
    for i in range(len(rings) - 1): SEAMS.append((rings[i][0][col], rings[i + 1][0][col]))
    cap(rings[-1][0], dl, last['r1'] * tip_bulge, part, lambda co: {lname: 1.0})
    if hole is not None: bridge(hole, rings[0][0], rings[0][1])
    return rings, bands

def hdir(az_deg, el_deg, side):
    az = math.radians(az_deg); el = math.radians(el_deg)
    return Vector((side * math.cos(az) * math.cos(el), -math.sin(az) * math.cos(el), math.sin(el)))

PROP = [('coxa', .06), ('trochanter', .035), ('femur', .27), ('patella', .10),
        ('tibia', .215), ('metatarsus', .245), ('tarsus', .14)]
RADR = [(.050, .048), (.043, .042), (.048, .040), (.037, .035), (.035, .030), (.025, .020), (.019, .012)]
STN_LONG = [0.1, 0.3, 0.5, 0.7, 0.9]; STN_SHORT = [0.2, 0.5, 0.8]

def solve_leg(L, start_z, base, femur_el):
    def tipz(k):
        z = start_z
        for i, ((nm, f), e) in enumerate(zip(PROP, base)):
            e = femur_el if i == 2 else (max(-89, e * k) if i >= 3 else e)
            z += L * f * math.sin(math.radians(e))
        return z
    lo, hi = 0.1, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if tipz(mid) > 0.004 * Lp: lo = mid
        else: hi = mid
    k = (lo + hi) / 2
    return [femur_el if i == 2 else (max(-89, e * k) if i >= 3 else e) for i, e in enumerate(base)]

LEGS = [(3, 55, 10), (6, 20, 6), (9, -18, -6), (12, -52, -10)]
SPINE_JOBS = []
for li, (k, az, drift) in enumerate(LEGS):
    T = P['legs'][li]
    for side in (1, -1):
        hole, hc = make_hole(k, -2 if side > 0 else 22)
        nrm = hc - Vector((0, hc.y, Zb)); nrm.y = 0; nrm.normalize()
        start = sum((v.co for v in hole), Vector()) / 8 + nrm * 0.02 * Lp
        a_j = az + R.uniform(-7, 7)
        els = solve_leg(T, start.z, [-15, 15, 0, -10, -38, -55, -28], P['femur_el'] - li * 3 + R.uniform(-5, 5))
        segs = []
        for s_i, ((nm, f), (r0, r1), e) in enumerate(zip(PROP, RADR, els)):
            d = hdir(a_j + (drift + R.uniform(-4, 4)) * s_i / 6, e, side)
            if s_i == 0: d = (d + nrm * 0.8).normalized()
            segs.append(dict(name=nm, dir=d, length=T * f, r0=r0 * Lp, r1=r1 * Lp,
                             stations=STN_LONG if f > 0.15 else STN_SHORT,
                             bulge=0.10 if nm in ('femur', 'tibia') else 0.05))
        rings, bands = build_limb(f'leg{li + 1}', side, start, segs, 2, hole=hole)
        SPINE_JOBS.append((bands, 'leg'))

for side in (1, -1):
    hole, hc = make_hole(1, -5 if side > 0 else 25)
    nrm = hc - Vector((0, hc.y, Zb)); nrm.y = 0; nrm.normalize()
    start = sum((v.co for v in hole), Vector()) / 8 + nrm * 0.01 * Lp
    PP = [('coxa', .11, -25, 70, .040, .038), ('trochanter', .07, 0, 70, .033, .033),
          ('femur', .30, 38, 75, .034, .030), ('patella', .12, -10, 78, .028, .026),
          ('tibia', .18, -45, 80, .026, .025), ('tarsus', .22, -66, 82, .030 if P['sex'] == 'M' else .026, .014)]
    Lpp = 0.78 * Lp
    segs = []
    for s_i, (nm, f, e, a, r0, r1) in enumerate(PP):
        d = hdir(a + R.uniform(-5, 5), e + R.uniform(-6, 6), side)
        if s_i == 0: d = (d + nrm * 0.6 + Vector((0, 0, -0.4))).normalized()
        segs.append(dict(name=nm, dir=d, length=Lpp * f, r0=r0 * Lp, r1=r1 * Lp,
                         stations=STN_LONG if f > 0.15 else STN_SHORT, bulge=0.08))
    build_limb('palp', side, start, segs, 2, hole=hole, tip_bulge=0.8)

for side in (1, -1):
    hole, hc = make_hole(0, -8 if side > 0 else 28)
    start = sum((v.co for v in hole), Vector()) / 8 + Vector((0, -0.01, -0.015)) * Lp
    c = 0.85 * LS
    segs = [dict(name='base', dir=Vector((side * 0.05, -0.25, -1)), length=0.50 * c, r0=0.10 * c, r1=0.105 * c,
                 stations=[0.1, 0.3, 0.5, 0.7, 0.9], bulge=0.35),
            dict(name='fang', dir=Vector((-side * 0.7, -0.15, -0.55)), length=0.15 * c, r0=0.045 * c, r1=0.032 * c,
                 stations=[0.3, 0.7], bulge=0.0),
            dict(name='fangtip', dir=Vector((-side * 0.8, 0.45, -0.1)), length=0.12 * c, r0=0.028 * c, r1=0.005 * c,
                 stations=[0.4, 0.8], bulge=0.0)]
    rings, _ = build_limb('chelicera', side, start, segs, 4, hole=hole, tip_bulge=0.5)
    for ring, t, k, st in rings:
        if k >= 1:
            for v in ring: v[A_PART] = 5

# ------------------------------------------------------------------ few short spines (zodariids are weakly spined)
def spike(face, direction, length, steps=3, inset=0.25, shrink=0.45, part=6):
    bmesh.ops.inset_individual(bm, faces=[face], thickness=inset * math.sqrt(face.calc_area()), depth=0)
    f = face
    for s in range(steps):
        f = bmesh.ops.extrude_discrete_faces(bm, faces=[f])['faces'][0]
        c = f.calc_center_median(); nc = c + direction * (length / steps)
        kk = shrink if s < steps - 1 else 0.15
        for v in f.verts:
            v.co = nc + (v.co - c) * kk; v[A_PART] = part
bm.normal_update()
names = [p[0] for p in PROP]
spec = {'tibia': (Vector((0, 0, -1)), 0.045, [0.3, 0.7]), 'metatarsus': (Vector((0, 0, -1)), 0.04, [0.1, 0.5])}
for bands, _ in SPINE_JOBS:
    for fs, t, k, st in bands:
        nm = names[k]
        if nm not in spec: continue
        pref, length, tl = spec[nm]
        if not any(abs(st - x) < 0.06 for x in tl) or R.random() < 0.3: continue
        bm.normal_update()
        cands = sorted([f for f in fs if f and f.is_valid], key=lambda f: -f.normal.dot(pref))[:2]
        for f in cands:
            spike(f, (f.normal * 0.5 + t * 0.9).normalized(), length * Lp * R.uniform(0.8, 1.2))
bm.normal_update()
for sx in (0.025, -0.025):
    target = Vector((sx * Lp, y_a0 + La * 1.02, Za - 0.05 * Ha))
    f = min([f for f in back_cap if f.is_valid], key=lambda f: (f.calc_center_median() - target).length)
    spike(f, Vector((0, 0.8, -0.4)).normalized(), 0.05 * Lp, steps=2, inset=0.12, shrink=0.75, part=7)

# ------------------------------------------------------------------ eyes: compact group, two procurved rows
def eye(x, y, Rr, kind):
    hit = bvh.ray_cast(Vector((x, y, Zb + 3 * Lp)), Vector((0, 0, -1)))
    if hit[0] is None:
        print('EYE MISS'); return
    t = hit[1].normalized(); c = hit[0] - t * Rr * 0.35
    n = perp(t); b = t.cross(n); S = 12; rings = []
    for phi in (40, 70, 90, 110, 140):
        ph = math.radians(phi); cc = c - t * Rr * math.cos(ph); rr = Rr * math.sin(ph)
        rings.append([nv(cc + (n * math.cos(2 * math.pi * i / S) + b * math.sin(2 * math.pi * i / S)) * rr,
                         3, {'prosoma': 1}, kind) for i in range(S)])
    for i in range(len(rings) - 1):
        for jj in range(S):
            quad(rings[i][jj], rings[i][(jj + 1) % S], rings[i + 1][(jj + 1) % S], rings[i + 1][jj])
        SEAMS.append((rings[i][0], rings[i + 1][0]))
    cap(rings[0][::-1], -t, Rr * 0.25, 3, lambda co: {'prosoma': 1})
    cap(rings[-1], t, Rr * 0.25, 3, lambda co: {'prosoma': 1})
for sx in (1, -1):
    eye(sx * 0.050 * Lp, -0.875 * Lp, 0.032 * Lp, 0)     # AME (largest, dark)
    eye(sx * 0.140 * Lp, -0.895 * Lp, 0.024 * Lp, 1)     # ALE
    eye(sx * 0.065 * Lp, -0.790 * Lp, 0.024 * Lp, 1)     # PME
    eye(sx * 0.160 * Lp, -0.805 * Lp, 0.025 * Lp, 1)     # PLE

# ------------------------------------------------------------------ hair density
HD = vg('hair_density')
for v in bm.verts:
    p = round(v[A_PART])
    w = {0: 0.45, 1: 1.0, 2: 0.75, 3: 0.0, 4: 0.5, 5: 0.0, 6: 0.0, 7: 0.3}[p]
    if p == 1 and v.co.y < y_a0 + 0.5 * La and v.co.z > Za + 0.25 * Ht:
        w *= 1 - 0.85 * P['scutum']
    v[DL][HD] = w

# ------------------------------------------------------------------ mesh object
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
for a, b in SEAMS:
    if a.is_valid and b.is_valid:
        e = bm.edges.get((a, b))
        if e: e.seam = True
for f in bm.faces: f.smooth = True
name = f"{P['genus']}_{args.preset}_{args.seed:03d}"
me = bpy.data.meshes.new(name); bm.to_mesh(me)
ob = bpy.data.objects.new(name, me)
coll = bpy.data.collections.new('Specimen'); bpy.context.scene.collection.children.link(coll)
coll.objects.link(ob)
for n_ in VG: ob.vertex_groups.new(name=n_)
ob.pass_index = 1
ob['spider_specimen'] = 1
ob['mallinella'] = int(P.get('genus') == 'Mallinella')
ob['family'] = P['family']
ob['genus'] = P['genus']
ob['morphotype'] = P['preset']
ob['params'] = json.dumps(P)
sc = bpy.context.scene
sc.unit_settings.system = 'METRIC'; sc.unit_settings.scale_length = 0.001; sc.unit_settings.length_unit = 'MILLIMETERS'
bpy.context.view_layer.objects.active = ob; ob.select_set(True)
me.uv_layers.new(name='UVTex')
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.unwrap(method='CONFORMAL', margin=0.002)
bpy.ops.uv.average_islands_scale(); bpy.ops.uv.pack_islands(margin=0.004, rotate=True)
bpy.ops.object.mode_set(mode='OBJECT')
sub = ob.modifiers.new('Subsurf', 'SUBSURF'); sub.levels = 1; sub.render_levels = 2; sub.quality = 2

# ------------------------------------------------------------------ procedural material
mat = bpy.data.materials.new(name + '_body')
try: mat.use_nodes = True
except Exception: pass
nt = mat.node_tree; Nn = nt.nodes; Lk = nt.links; Nn.clear(); CNT = [0]
def node(t, **kw):
    n = Nn.new(t); CNT[0] += 1; n.location = (-220 * (CNT[0] % 14), -200 * (CNT[0] // 14))
    for k, v in kw.items(): setattr(n, k, v)
    return n
def lnk(a, b): Lk.new(a, b)
def isnum(v): return isinstance(v, (int, float))
def M(op, a, b=None, c=None):
    n = node('ShaderNodeMath', operation=op)
    for i, val in enumerate((a, b, c)):
        if val is None: continue
        if isnum(val): n.inputs[i].default_value = val
        else: lnk(val, n.inputs[i])
    return n.outputs[0]
add = lambda a, b: M('ADD', a, b); mul = lambda a, b: M('MULTIPLY', a, b)
sub_ = lambda a, b: M('SUBTRACT', a, b); inv = lambda a: M('SUBTRACT', 1.0, a)
def ss(e0, e1, x):
    n = node('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP')
    for sock, val in (('From Min', e0), ('From Max', e1)):
        if isnum(val): n.inputs[sock].default_value = val
        else: lnk(val, n.inputs[sock])
    lnk(x, n.inputs['Value']); return n.outputs[0]
def mixc(fac, a, b):
    n = node('ShaderNodeMix', data_type='RGBA', clamp_factor=True)
    if isnum(fac): n.inputs['Factor'].default_value = fac
    else: lnk(fac, n.inputs['Factor'])
    for sock, val in ((n.inputs[6], a), (n.inputs[7], b)):
        if isinstance(val, tuple): sock.default_value = tuple(val) + (1,)
        else: lnk(val, sock)
    return n.outputs[2]
def mulc(cc, f):
    n = node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', clamp_factor=True)
    n.inputs['Factor'].default_value = 1; lnk(cc, n.inputs[6])
    cmb = node('ShaderNodeCombineXYZ'); [lnk(f, cmb.inputs[i]) for i in range(3)]
    lnk(cmb.outputs[0], n.inputs[7]); return n.outputs[2]
def attr(nm): return node('ShaderNodeAttribute', attribute_name=nm).outputs['Fac']
def noise(vec, scale, detail=4):
    n = node('ShaderNodeTexNoise'); lnk(vec, n.inputs['Vector'])
    n.inputs['Scale'].default_value = scale; n.inputs['Detail'].default_value = detail
    return n.outputs['Fac']
def voro(vec, scale):
    n = node('ShaderNodeTexVoronoi'); lnk(vec, n.inputs['Vector'])
    n.inputs['Scale'].default_value = scale; return n.outputs['Distance']
def ispart(p, k): return M('COMPARE', p, float(k), 0.5)

tc = node('ShaderNodeTexCoord'); Pc = tc.outputs['Object']
sp = node('ShaderNodeSeparateXYZ'); lnk(Pc, sp.inputs[0]); x, y, z = sp.outputs[0], sp.outputs[1], sp.outputs[2]
ax = M('ABSOLUTE', x)
geo = node('ShaderNodeNewGeometry'); ns = node('ShaderNodeSeparateXYZ'); lnk(geo.outputs['Normal'], ns.inputs[0])
nz = ns.outputs[2]
part = attr('part'); seg = attr('seg'); segt = attr('segt')
dorsal = ss(-0.35, 0.25, nz)
n_big = noise(Pc, 0.9 / LS); n_mid = noise(Pc, 4.0 / LS)

# carapace
car = mixc(mul(n_big, 0.5), P['carapace'], tuple(c * 0.6 for c in P['carapace']))
furrow = mul(inv(ss(0.0, 0.018 * Lp, ax)), mul(ss(-0.62 * Lp, -0.55 * Lp, y), ss(-0.22 * Lp, -0.3 * Lp, y)))
eyereg = mul(ss(-0.72 * Lp, -0.8 * Lp, y), ss(0.25, 0.6, nz))
ang = M('ARCTAN2', add(y, 0.42 * Lp), add(ax, 0.001))
striae = mul(ss(0.6, 0.95, M('ABSOLUTE', M('SINE', mul(ang, 8.0)))), ss(0.15 * Lp, 0.3 * Lp, ax))
car = mixc(mul(striae, 0.5), car, (0.006, 0.004, 0.003))
car = mixc(furrow, car, (0.004, 0.003, 0.003))
car = mixc(mul(eyereg, 0.85), car, (0.005, 0.004, 0.004))
car = mixc(inv(dorsal), car, (0.06, 0.035, 0.02))          # sternum brown
# abdomen: normalised coordinates
u = M('DIVIDE', sub_(y, y_a0), La)
aa = M('DIVIDE', ax, Wa / 2)
zr = M('DIVIDE', sub_(z, Za), Ht)
def blob(uc, ac, su, sa, nfac=0.25):
    du = M('DIVIDE', sub_(u, uc), su); da = M('DIVIDE', sub_(aa, ac), sa)
    d = M('SQRT', add(mul(du, du), mul(da, da)))
    return ss(1.05, 0.75, add(d, mul(sub_(n_mid, 0.5), nfac)))
ps = P['patch_s']
w1 = blob(P['patch_u'][0], P['patch_a'][0], 0.07 * ps, 0.22 * ps)
w2 = blob(P['patch_u'][1], P['patch_a'][1], 0.07 * ps, 0.24 * ps)
w3 = blob(P['patch_u'][2], P['patch_a'][2], 0.065 * ps, 0.26 * ps)
w3b = mul(blob(P['patch_u'][2], 0.0, 0.05 * ps, 0.55 * ps), P['bridge'])
w4 = mul(blob(0.86, 0.0, 0.05 * ps, 0.42 * ps), P['post_patch'])
white = M('MINIMUM', add(add(add(w1, w2), add(w3, w3b)), w4), 1.0)
white = mul(white, ss(-0.25, 0.05, zr))
scut = mul(mul(ss(0.52, 0.44, add(u, mul(sub_(n_mid, 0.5), 0.06))), ss(0.05, 0.3, zr)), P['scutum'])
abd = mixc(mul(n_big, 0.4), P['abd_col'], tuple(c * 2.2 for c in P['abd_col']))
abd = mixc(scut, abd, P['scutum_col'])
abd = mixc(white, abd, P['patch_col'])
# venter: grey-black with three white longitudinal stripes
ven = ss(-0.25, -0.5, zr)
stripes = add(ss(0.09, 0.05, aa), ss(0.07, 0.03, M('ABSOLUTE', sub_(aa, 0.42))))
stripes = mul(mul(stripes, ss(0.2, 0.3, u)), ss(0.88, 0.78, u))
venc = mixc(M('MINIMUM', stripes, 1.0), (0.025, 0.023, 0.022), (0.45, 0.43, 0.40))
abd = mixc(ven, abd, venc)
# legs
legc = mixc(mul(n_big, 0.5), P['legs_col'], tuple(c * 0.75 for c in P['legs_col']))
legc = mixc(M('COMPARE', seg, 2.0, 0.5), legc, P['femur_col'])
legc = mixc(mul(M('COMPARE', seg, 0.0, 0.5), 1.0), legc, P['coxa_col'])
band = mul(ss(0.5, 0.85, add(M('SINE', add(mul(segt, 6.0), 1.0)), mul(sub_(n_mid, 0.5), 0.5))), P['bands'])
band = mul(band, M('MINIMUM', add(M('COMPARE', seg, 4.0, 0.5), M('COMPARE', seg, 5.0, 0.5)), 1.0))
legc = mixc(band, legc, P['femur_col'])
col = car
col = mixc(ispart(part, 1), col, abd)
col = mixc(ispart(part, 2), col, legc)
col = mixc(ispart(part, 4), col, (0.03, 0.012, 0.007))
col = mixc(ispart(part, 5), col, (0.006, 0.003, 0.002))
col = mixc(ispart(part, 6), col, (0.010, 0.007, 0.005))
col = mixc(ispart(part, 7), col, (0.50, 0.40, 0.17))     # pale spinnerets
eyec = mixc(M('COMPARE', seg, 0.0, 0.5), (0.35, 0.36, 0.38), (0.003, 0.003, 0.004))
col = mixc(ispart(part, 3), col, eyec)
noteye = inv(ispart(part, 3))
col = mulc(col, add(mul(noise(Pc, 30.0 / LS, 2), mul(noteye, 0.5)), add(mul(noteye, 0.75), ispart(part, 3))))
rough = add(0.55, mul(sub_(n_mid, 0.5), 0.15))
rough = sub_(rough, mul(mul(scut, ispart(part, 1)), 0.3))        # shiny scutum
rough = add(mul(rough, noteye), mul(ispart(part, 3), 0.03))
gran = mul(ss(0.0, 0.35, voro(Pc, 22.0 / LS)), M('MAXIMUM', ispart(part, 0), mul(scut, 0.4)))
hgt = add(mul(noise(Pc, 60.0 / LS, 3), 0.4), mul(gran, 0.6))
hgt = mul(hgt, noteye)
bump = node('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.3
bump.inputs['Distance'].default_value = 0.01 * LS
lnk(hgt, bump.inputs['Height'])
bsdf = node('ShaderNodeBsdfPrincipled')
lnk(col, bsdf.inputs['Base Color']); lnk(rough, bsdf.inputs['Roughness'])
lnk(bump.outputs['Normal'], bsdf.inputs['Normal'])
bsdf.inputs['Sheen Weight'].default_value = 0.25
bsdf.inputs['Coat Weight'].default_value = 0.0
out = node('ShaderNodeOutputMaterial'); lnk(bsdf.outputs[0], out.inputs['Surface'])
ob.data.materials.append(mat)

# ------------------------------------------------------------------ hair (setae)
if args.hair > 0:
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 2
    sc.render.bake.margin = 8
    im = bpy.data.images.new(name + '_colour', args.bake_res, args.bake_res)
    tn = Nn.new('ShaderNodeTexImage'); tn.image = im; Nn.active = tn
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
    Nn.remove(tn); im.pack()
    hm = bpy.data.materials.new(name + '_setae')
    try: hm.use_nodes = True
    except Exception: pass
    hn = hm.node_tree.nodes; hl = hm.node_tree.links; hn.clear()
    uvn = hn.new('ShaderNodeUVMap'); uvn.uv_map = 'UVTex'; uvn.from_instancer = False
    ti = hn.new('ShaderNodeTexImage'); ti.image = im
    hl.new(uvn.outputs[0], ti.inputs[0])
    hi = hn.new('ShaderNodeHairInfo')
    mx = hn.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'MULTIPLY'
    hl.new(ti.outputs[0], mx.inputs[6]); mx.inputs[7].default_value = (1.25, 1.2, 1.15, 1)
    mx.inputs['Factor'].default_value = 1
    pb = hn.new('ShaderNodeBsdfPrincipled')
    hl.new(mx.outputs[2], pb.inputs['Base Color']); pb.inputs['Roughness'].default_value = 0.35
    o2 = hn.new('ShaderNodeOutputMaterial'); hl.new(pb.outputs[0], o2.inputs['Surface'])
    ob.data.materials.append(hm)
    ps_mod = ob.modifiers.new('setae', 'PARTICLE_SYSTEM')
    psys = ps_mod.particle_system; st = psys.settings
    st.type = 'HAIR'; st.use_advanced_hair = True
    area = sum(f.calc_area() for f in bm.faces)
    st.count = int(min(60000, area * 350 * args.hair * P['hair']))
    st.emit_from = 'FACE'; st.use_emit_random = True; st.use_even_distribution = True
    st.normal_factor = 1.0; st.factor_random = 0.6
    st.object_align_factor = (0.0, 0.7, -0.15)        # combed backwards and slightly down
    st.length_random = 0.5
    st.hair_length = 0.05 * Lp                          # rescales normal_factor only...
    nf = st.normal_factor                               # ...so scale the combing factor with it
    st.object_align_factor = (0.0, 0.8 * nf, -0.2 * nf)
    st.factor_random = 0.45 * nf
    st.material = 2
    st.child_type = 'NONE'
    st.root_radius = 1.0; st.tip_radius = 0.1; st.radius_scale = 0.0035 * LS
    st.display_step = 3; st.render_step = 3
    st.use_hair_bspline = True
    psys.vertex_group_density = 'hair_density'
    psys.seed = args.seed
bm.free()
sc.render.engine = 'CYCLES'
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out))
print('SPECIMEN', args.out, json.dumps({k: P[k] for k in ('sex', 'Lp', 'La', 'legs')}))
