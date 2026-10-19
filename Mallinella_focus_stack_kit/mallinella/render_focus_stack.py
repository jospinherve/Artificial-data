"""
Synthetic macro focus-stacking + ground-truth depth for a spider specimen (Blender 4.2+ / 5.x, Cycles).

  blender -b specimens/mallinella_sphaerica_m_001.blend -P render_focus_stack.py -- --config config.json
  (optional: --out DIR  --views dorsal,lateral_L  --quick)

For every view listed in the config this writes
  <out>/<specimen>/<view>/slices/slice_000.png ...   defocused images (the focus stack)
  <out>/<specimen>/<view>/all_in_focus.png           pinhole render from the reference camera
  <out>/<specimen>/<view>/depth.exr / depth.npy      planar depth (mm) along the optical axis of the
                                                     reference camera; background = substrate depth,
                                                     or +inf where nothing is hit
  <out>/<specimen>/<view>/depth_vis.png              16-bit visualisation (near = bright)
  <out>/<specimen>/<view>/mask.png                   specimen (incl. setae) = 255
  <out>/<specimen>/<view>/normals.npy                camera-space normals (optional)
  <out>/<specimen>/<view>/meta.json                  intrinsics, extrinsics (OpenCV convention),
                                                     optics, per-slice focus depths and camera poses

Units: 1 Blender unit = 1 mm everywhere.

Optics model (thin lens, matches real macro rigs):
  "lens" mode       : objective focal f, magnification m, f-number N
                      working distance s = f (1 + 1/m), aperture D = f / N, effective N_eff = N (1 + m)
  "objective" mode  : infinity microscope objective: magnification m, NA, working distance WD
                      s = WD, D = 2 WD tan(asin NA)
  Blender is a pinhole + thin-lens DOF model: lens = m*s gives the right field of view, and the aperture
  diameter is set through f-stop = lens[mm]*1e-3 / D[mm] because Cycles converts f-stop to an aperture
  assuming 1 unit = 1 m (verified empirically, see tests/aperture_test.py).
Stacking modes:
  "rail"  : the whole camera moves along its axis (StackShot-like); magnification fixed, slight perspective change
  "focus" : camera fixed, focus distance changes
"""
import bpy, sys, os, json, math, argparse, random, zlib
import numpy as np
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--blend', default=None, help='specimen .blend to open (when not given on the blender command line)')
ap.add_argument('--out', default=None)
ap.add_argument('--views', default=None, help='comma separated subset of view names')
ap.add_argument('--quick', action='store_true', help='low resolution / samples for testing')
args = ap.parse_args(argv)
CFG = json.load(open(args.config))
if args.blend:
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))

def cfg(path, default=None):
    d = CFG
    for k in path.split('.'):
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d

scene = bpy.context.scene
spec = [o for o in scene.objects if o.get('spider_specimen') or o.get('mallinella')]
if not spec:
    raise SystemExit('No object with the "spider_specimen" custom property in this file')
spec = spec[0]
spec.pass_index = 1
SPEC_NAME = spec.name
OUT = os.path.abspath(args.out or cfg('output_dir', 'renders'))
OUT = os.path.join(OUT, SPEC_NAME)

# ------------------------------------------------------------------ scene / render settings
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 0.001
scene.render.engine = 'CYCLES'
res = cfg('resolution', [2048, 1536])
if args.quick: res = [res[0] // 4, res[1] // 4]
scene.render.resolution_x, scene.render.resolution_y = res
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.view_settings.view_transform = cfg('color.view_transform', 'Standard')
scene.view_settings.look = 'None'
scene.view_settings.exposure = cfg('color.exposure', 0.0)
SAMPLES = cfg('samples', 128) if not args.quick else 16
DENOISE = cfg('denoise', True)
device = cfg('device', 'CPU')
if device != 'CPU':
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = device          # 'CUDA', 'OPTIX', 'HIP', 'METAL', 'ONEAPI'
        prefs.get_devices()
        for d in prefs.devices: d.use = True
        scene.cycles.device = 'GPU'
    except Exception as e:
        print('GPU setup failed, using CPU:', e); scene.cycles.device = 'CPU'
scene.cycles.max_bounces = 6
scene.cycles.transparent_max_bounces = 16

# ------------------------------------------------------------------ world + lights
world = bpy.data.worlds.new('macro_world'); scene.world = world
try: world.use_nodes = True
except Exception: pass
wn = world.node_tree.nodes; bg = wn.get('Background') or wn.new('ShaderNodeBackground')
wout = wn.get('World Output') or wn.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs[0], wout.inputs[0])
bgcol = cfg('background.color', [0.75, 0.75, 0.75])
def set_world(strength, col):
    bg.inputs[0].default_value = tuple(col) + (1,)
    bg.inputs[1].default_value = strength

cam_data = bpy.data.cameras.new('macro_cam')
cam = bpy.data.objects.new('macro_cam', cam_data); scene.collection.objects.link(cam); scene.camera = cam
cam_data.sensor_fit = 'HORIZONTAL'
cam_data.sensor_width = cfg('optics.sensor_width_mm', 36.0)
cam_data.sensor_height = cam_data.sensor_width * res[1] / res[0]
cam_data.clip_start = 0.05; cam_data.clip_end = 1e5

LIGHTS = []
def make_lights():
    """Diffuse macro lighting fixed to the camera: dome (world) + two large soft boxes."""
    for i, (ax, ay) in enumerate(((-1, 1), (1, 0.6))):
        L = bpy.data.lights.new(f'softbox{i}', 'AREA'); L.shape = 'DISK'
        o = bpy.data.objects.new(f'softbox{i}', L); scene.collection.objects.link(o)
        LIGHTS.append((o, ax, ay))
make_lights()

def place_lights(cam_pos, forward, right, up, dist, rng):
    jitter = cfg('lighting.jitter', 0.2)
    power = cfg('lighting.softbox_power', 1.0)
    for o, ax, ay in LIGHTS:
        a = ax * (1 + rng.uniform(-jitter, jitter)); b = ay * (1 + rng.uniform(-jitter, jitter))
        p = cam_pos + right * a * dist * 0.6 + up * b * dist * 0.6 + forward * dist * 0.25
        o.location = p
        o.rotation_euler = (cam_pos + forward * dist - p).to_track_quat('-Z', 'Y').to_euler()
        o.data.size = dist * 0.5
        # watts scale with distance^2 so the irradiance stays constant whatever the working distance
        o.data.energy = power * 2.5 * dist ** 2 * (1 + rng.uniform(-jitter, jitter))

# ------------------------------------------------------------------ substrate
sub_type = cfg('background.substrate', 'paper')      # 'paper', 'sand', 'plain' or 'none'
substrate = None
if sub_type != 'none':
    me = bpy.data.meshes.new('substrate')
    S = 400.0
    me.from_pydata([(-S, -S, 0), (S, -S, 0), (S, S, 0), (-S, S, 0)], [], [(0, 1, 2, 3)])
    substrate = bpy.data.objects.new('substrate', me); scene.collection.objects.link(substrate)
    substrate.pass_index = 2
    m = bpy.data.materials.new('substrate')
    try: m.use_nodes = True
    except Exception: pass
    nt = m.node_tree; n = nt.nodes; l = nt.links; n.clear()
    tc = n.new('ShaderNodeTexCoord'); bs = n.new('ShaderNodeBsdfPrincipled'); o = n.new('ShaderNodeOutputMaterial')
    l.new(bs.outputs[0], o.inputs[0])
    base = cfg('background.substrate_color', [0.55, 0.52, 0.47])
    if sub_type == 'plain':
        bs.inputs['Base Color'].default_value = tuple(base) + (1,)
    else:
        nz = n.new('ShaderNodeTexNoise'); l.new(tc.outputs['Object'], nz.inputs['Vector'])
        nz.inputs['Scale'].default_value = 1.5 if sub_type == 'paper' else 4.0
        nz.inputs['Detail'].default_value = 8
        vo = n.new('ShaderNodeTexVoronoi'); l.new(tc.outputs['Object'], vo.inputs['Vector'])
        vo.inputs['Scale'].default_value = 6.0 if sub_type == 'sand' else 25.0
        ramp = n.new('ShaderNodeValToRGB'); l.new(nz.outputs['Fac'], ramp.inputs[0])
        ramp.color_ramp.elements[0].color = tuple(c * 0.75 for c in base) + (1,)
        ramp.color_ramp.elements[1].color = tuple(min(1, c * 1.2) for c in base) + (1,)
        l.new(ramp.outputs[0], bs.inputs['Base Color'])
        bump = n.new('ShaderNodeBump'); bump.inputs['Distance'].default_value = 0.05
        bump.inputs['Strength'].default_value = 0.6 if sub_type == 'sand' else 0.25
        l.new(vo.outputs['Distance'], bump.inputs['Height']); l.new(bump.outputs[0], bs.inputs['Normal'])
    bs.inputs['Roughness'].default_value = 0.8
    me.materials.append(m)

# ------------------------------------------------------------------ ground-truth override material
gt_mat = bpy.data.materials.new('GT_override')
try: gt_mat.use_nodes = True
except Exception: pass
gnt = gt_mat.node_tree; gn = gnt.nodes; gl = gnt.links; gn.clear()
camd = gn.new('ShaderNodeCameraData'); oi = gn.new('ShaderNodeObjectInfo')
cmb = gn.new('ShaderNodeCombineColor'); em = gn.new('ShaderNodeEmission'); go = gn.new('ShaderNodeOutputMaterial')
gl.new(camd.outputs['View Z Depth'], cmb.inputs[0]); gl.new(oi.outputs['Object Index'], cmb.inputs[1])
cmb.inputs[2].default_value = 1.0
gl.new(cmb.outputs[0], em.inputs['Color']); gl.new(em.outputs[0], go.inputs['Surface'])
nrm_mat = bpy.data.materials.new('GT_normals')
try: nrm_mat.use_nodes = True
except Exception: pass
nnt = nrm_mat.node_tree; nn = nnt.nodes; nl = nnt.links; nn.clear()
geo = nn.new('ShaderNodeNewGeometry'); vt = nn.new('ShaderNodeVectorTransform')
vt.vector_type = 'NORMAL'; vt.convert_from = 'WORLD'; vt.convert_to = 'CAMERA'
em2 = nn.new('ShaderNodeEmission'); no2 = nn.new('ShaderNodeOutputMaterial')
nl.new(geo.outputs['Normal'], vt.inputs[0]); nl.new(vt.outputs[0], em2.inputs['Color']); nl.new(em2.outputs[0], no2.inputs['Surface'])

# ------------------------------------------------------------------ helpers
def eval_points():
    """World-space vertices of the evaluated specimen (subdivided mesh + hair keys)."""
    dg = bpy.context.evaluated_depsgraph_get()
    oe = spec.evaluated_get(dg); me = oe.to_mesh()
    co = np.empty(len(me.vertices) * 3, dtype=np.float32); me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3); oe.to_mesh_clear()
    M = np.array(spec.matrix_world)
    pts = co @ M[:3, :3].T + M[:3, 3]
    extra = []
    for ps in oe.particle_systems:
        for p in ps.particles:
            for k in p.hair_keys:
                extra.append(tuple(k.co))
    if extra:
        pts = np.vstack([pts, np.array(extra, dtype=np.float32) @ M[:3, :3].T + M[:3, 3]])
    return pts

def look_basis(view_dir, roll_deg):
    """view_dir: unit vector from target to camera. Returns forward, right, up (image up)."""
    fwd = -view_dir
    ref = Vector((0, 0, 1)) if abs(fwd.z) < 0.98 else Vector((0, -1, 0))   # dorsal/ventral: head up
    right = fwd.cross(ref).normalized(); up = right.cross(fwd).normalized()
    r = math.radians(roll_deg)
    right, up = right * math.cos(r) + up * math.sin(r), -right * math.sin(r) + up * math.cos(r)
    return fwd, right, up

def set_cam(pos, fwd, right, up):
    R = Matrix((right, up, -fwd)).transposed()   # columns: camera X, Y, Z(back)
    cam.matrix_world = Matrix.Translation(pos) @ R.to_4x4()

def opencv_extrinsic(pos, fwd, right, up):
    """4x4 world->camera, OpenCV convention (x right, y down, z forward), mm."""
    Rwc = np.array([list(right), [-c for c in up], list(fwd)])
    t = -Rwc @ np.array(pos)
    E = np.eye(4); E[:3, :3] = Rwc; E[:3, 3] = t
    return E.tolist()

def render_to(path, fmt='PNG', depth='8', samples=None, denoise=None, aa=True):
    scene.cycles.samples = samples or SAMPLES
    scene.cycles.use_denoising = DENOISE if denoise is None else denoise
    scene.cycles.filter_width = 1.5 if aa else 0.01
    s = scene.render.image_settings
    s.file_format = fmt
    if fmt == 'PNG':
        s.color_mode = 'RGB'; s.color_depth = depth; s.compression = 15
    else:
        s.color_mode = 'RGBA'; s.color_depth = '32'; s.exr_codec = 'ZIP'
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)

def load_exr(path):
    im = bpy.data.images.load(path, check_existing=False)
    w, h = im.size
    a = np.empty(w * h * 4, dtype=np.float32); im.pixels.foreach_get(a)
    bpy.data.images.remove(im)
    return a.reshape(h, w, 4)[::-1]           # top row first

def save_png16(path, arr):
    """Write a single-channel 16-bit PNG (values 0..1) with zlib only (no Blender colour management)."""
    import zlib, struct
    a = (np.clip(arr, 0, 1) * 65535 + 0.5).astype('>u2')
    h, w = a.shape
    raw = b''.join(b'\x00' + a[y].tobytes() for y in range(h))
    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 16, 0, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))

# ------------------------------------------------------------------ optics
opt = cfg('optics', {})
mode = opt.get('mode', 'lens')
pitch = cam_data.sensor_width / res[0]
coc = opt.get('coc_mm') or 2 * pitch

def optics_for(m):
    if mode == 'objective':
        s = opt.get('working_distance_mm', 34.0); NA = opt.get('na', 0.14)
        D = 2 * s * math.tan(math.asin(NA))
        lam = 0.00055
        dof = lam / NA ** 2 + pitch / (m * NA)
        neff = 1 / (2 * NA) * m
    else:
        f = opt.get('focal_length_mm', 100.0); N = opt.get('f_number', 8.0)
        s = f * (1 + 1 / m); D = f / N
        neff = N * (1 + m)
        dof = 2 * N * coc * (1 + m) / m ** 2
    lens = m * s
    return dict(s=s, D=D, lens=lens, fstop_blender=lens * 1e-3 / D, dof=dof, n_eff=neff, magnification=m)

def auto_mag(pts, centre, right, up):
    rel = pts - np.array(centre)
    xs = rel @ np.array(right); ys = rel @ np.array(up)
    w = 2 * max(abs(xs.min()), abs(xs.max())); h = 2 * max(abs(ys.min()), abs(ys.max()))
    margin = 1 + cfg('optics.auto_margin', 0.08)
    return min(cam_data.sensor_width / (w * margin), cam_data.sensor_height / (h * margin))

# ------------------------------------------------------------------ main loop
views = CFG['views']
if args.views:
    keep = set(args.views.split(',')); views = [v for v in views if v['name'] in keep]
base_matrix = spec.matrix_world.copy()
os.makedirs(OUT, exist_ok=True)
summary = []
for vi, view in enumerate(views):
    rng = random.Random(zlib.crc32((SPEC_NAME + view['name']).encode()))
    vdir_name = os.path.join(OUT, view['name']); os.makedirs(os.path.join(vdir_name, 'slices'), exist_ok=True)
    # optional rotation of the specimen itself (about its bbox centre)
    spec.matrix_world = base_matrix
    bpy.context.view_layer.update()
    pts0 = eval_points(); c0 = Vector(((pts0.min(0) + pts0.max(0)) / 2).tolist())
    rot = view.get('specimen_rotation_deg')
    if rot:
        Rm = Matrix.Rotation(math.radians(rot[2]), 4, 'Z') @ Matrix.Rotation(math.radians(rot[1]), 4, 'Y') @ \
             Matrix.Rotation(math.radians(rot[0]), 4, 'X')
        spec.matrix_world = Matrix.Translation(c0) @ Rm @ Matrix.Translation(-c0) @ base_matrix
        bpy.context.view_layer.update()
    pts = eval_points()
    target = view.get('target')
    if target is None:
        # centre on the body (prosoma+abdomen), legs are allowed to leave the frame
        params = json.loads(spec.get('params', '{}'))
        Lp = params.get('Lp', 3.0); La = params.get('La', 3.0)
        target_local = Vector((0, (0.13 * Lp + La - 0.95 * Lp) / 2, max(0.45 * Lp, 1.0)))
        target = (spec.matrix_world @ target_local) if view.get('center', 'body') == 'body' else c0
    target = Vector(target) + Vector(view.get('target_offset_mm', [0, 0, 0]))
    az = math.radians(view.get('azimuth', 0)); el = math.radians(view.get('elevation', 90))
    vdir = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))).normalized()
    fwd, right, up = look_basis(vdir, view.get('roll', 0))
    m = view.get('magnification', opt.get('magnification', 'auto'))
    if m == 'auto':
        sel = pts if view.get('fit', 'all') == 'all' else pts[np.linalg.norm(pts - np.array(target), axis=1) < 6]
        m = auto_mag(sel, target, right, up)
    O = optics_for(m)
    P0 = target - fwd * O['s']
    # depth range of the specimen seen from P0 (points inside the frustum only)
    rel = pts - np.array(P0)
    d = rel @ np.array(fwd)
    half_w = cam_data.sensor_width / 2 / O['lens']; half_h = cam_data.sensor_height / 2 / O['lens']
    inside = (np.abs(rel @ np.array(right)) < half_w * d) & (np.abs(rel @ np.array(up)) < half_h * d)
    dmin, dmax = float(d[inside].min()), float(d[inside].max())
    st = cfg('stack', {})
    step = st.get('step_mm') or O['dof'] * st.get('overlap', 0.7)
    margin = st.get('margin_mm', 0.2)
    n = int(math.ceil((dmax - dmin + 2 * margin) / step)) + 1
    nmax = st.get('max_slices', 150) if not args.quick else 12
    if n > nmax:
        print(f'[{view["name"]}] {n} slices needed, capped to {nmax} (step increased)')
        n = nmax; step = (dmax - dmin + 2 * margin) / (n - 1)
    focus = [dmin - margin + i * step for i in range(n)]
    smode = st.get('mode', 'rail')
    # substrate only when the camera is above it
    if substrate:
        hide = el < math.radians(2)
        substrate.hide_render = hide
    # lighting
    set_world(cfg('lighting.dome_strength', 0.6), bgcol)
    place_lights(P0, fwd, right, up, O['s'], rng)
    cam_data.lens = O['lens']
    meta = dict(specimen=SPEC_NAME, view=view, units='mm', resolution=res,
                sensor_width_mm=cam_data.sensor_width, sensor_height_mm=cam_data.sensor_height,
                pixel_pitch_mm=pitch, object_pixel_size_mm=pitch / m,
                intrinsics=dict(fx=O['lens'] / pitch, fy=O['lens'] / pitch, cx=res[0] / 2, cy=res[1] / 2),
                optics={**opt, **O, 'mode': mode, 'coc_mm': coc},
                reference_camera=dict(position=list(P0), world_to_cam_opencv=opencv_extrinsic(P0, fwd, right, up)),
                depth_convention='planar distance along the reference optical axis, mm',
                stack=dict(mode=smode, step_mm=step, n=n, specimen_depth_range_mm=[dmin, dmax]),
                slices=[])
    print(f'[{view["name"]}] m={m:.3f}x  s={O["s"]:.1f}mm  DOF={O["dof"]*1000:.0f}um  '
          f'depth {dmin:.2f}-{dmax:.2f}mm  -> {n} slices, step {step*1000:.0f}um ({smode})')
    # ---- focus stack
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = O['fstop_blender']
    cam_data.dof.aperture_blades = cfg('optics.aperture_blades', 0)
    for i, fd in enumerate(focus):
        if smode == 'rail':
            pos = P0 + fwd * (fd - O['s']); cam_data.dof.focus_distance = O['s']
        else:
            pos = P0; cam_data.dof.focus_distance = fd
        set_cam(pos, fwd, right, up)
        fn = os.path.join(vdir_name, 'slices', f'slice_{i:03d}.png')
        render_to(fn, depth=str(cfg('outputs.bit_depth', 8)))
        meta['slices'].append(dict(file=os.path.relpath(fn, vdir_name), focus_depth_mm=fd,
                                   camera_position=list(pos), focus_distance_mm=cam_data.dof.focus_distance,
                                   world_to_cam_opencv=opencv_extrinsic(pos, fwd, right, up)))
    # ---- all in focus (reference camera)
    set_cam(P0, fwd, right, up); cam_data.dof.use_dof = False
    if cfg('outputs.all_in_focus', True):
        render_to(os.path.join(vdir_name, 'all_in_focus.png'), depth=str(cfg('outputs.bit_depth', 8)))
    # ---- ground truth: depth (R), object index (G), coverage (B) in one emission pass
    vl = bpy.context.view_layer
    saved_world = (tuple(bg.inputs[0].default_value), bg.inputs[1].default_value)
    set_world(0.0, [0, 0, 0])
    for o, *_ in LIGHTS: o.hide_render = True
    vl.material_override = gt_mat
    gt_path = os.path.join(vdir_name, 'gt_raw.exr')
    render_to(gt_path, fmt='OPEN_EXR', samples=1, denoise=False, aa=False)
    gt = load_exr(gt_path)
    cover = gt[..., 2] > 0.5
    depth = np.where(cover, gt[..., 0], np.inf).astype(np.float32)
    mask = (np.abs(gt[..., 1] - 1) < 0.25) & cover
    np.save(os.path.join(vdir_name, 'depth.npy'), depth)
    os.replace(gt_path, os.path.join(vdir_name, 'depth.exr'))
    fin = np.isfinite(depth)
    dv = np.zeros_like(depth)
    lo = float(depth[fin].min()) if fin.any() else dmin
    hi = min(float(depth[fin].max()), dmax + 2 * (dmax - dmin)) if fin.any() else dmax
    dv[fin] = 1 - np.clip((depth[fin] - lo) / max(hi - lo, 1e-6), 0, 1)
    save_png16(os.path.join(vdir_name, 'depth_vis.png'), dv)
    save_png16(os.path.join(vdir_name, 'mask.png'), mask.astype(np.float32))
    if cfg('outputs.normals', True):
        vl.material_override = nrm_mat
        npath = os.path.join(vdir_name, 'normals_raw.exr')
        render_to(npath, fmt='OPEN_EXR', samples=1, denoise=False, aa=False)
        nr = load_exr(npath)[..., :3]
        nr[~cover] = 0
        np.save(os.path.join(vdir_name, 'normals.npy'), nr.astype(np.float16))
        os.remove(npath)
    vl.material_override = None
    for o, *_ in LIGHTS: o.hide_render = False
    set_world(saved_world[1], saved_world[0][:3])
    meta['depth_files'] = dict(npy='depth.npy', exr='depth.exr (R = depth mm, G = object index, B = coverage)',
                               vis='depth_vis.png', mask='mask.png')
    json.dump(meta, open(os.path.join(vdir_name, 'meta.json'), 'w'), indent=1, default=float)
    summary.append(dict(view=view['name'], slices=n, magnification=m, dof_um=O['dof'] * 1000))
spec.matrix_world = base_matrix
json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=float)
print('DONE', OUT)
