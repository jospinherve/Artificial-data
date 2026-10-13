"""
Minimal loader for the rendered focus stacks (numpy + Pillow only; wrap it in a torch Dataset if needed).

    from load_dataset import load_view, iter_views
    for v in iter_views('renders'):
        stack, focus_mm, depth_mm, mask, meta = load_view(v)
        # stack: (N, H, W, 3) float32 in [0,1], focus_mm: (N,), depth_mm: (H, W) (inf = no hit)

Typical depth-from-focus targets:
  * metric depth (mm)            -> depth_mm
  * slice index of best focus    -> np.argmin(|depth_mm[...,None] - focus_mm|, axis=-1)
  * normalised depth in [0,1]    -> (depth_mm - focus_mm[0]) / (focus_mm[-1] - focus_mm[0])
Note: in "rail" mode each slice is taken from a slightly different camera position (like a real
focus rail); meta['slices'][i]['world_to_cam_opencv'] gives every pose, and depth_mm is expressed in
the reference camera frame (the one of all_in_focus.png).
"""
import os, json, glob
import numpy as np
from PIL import Image

def iter_views(root):
    for meta in sorted(glob.glob(os.path.join(root, '*', '*', 'meta.json'))):
        yield os.path.dirname(meta)

def load_view(view_dir, max_slices=None):
    meta = json.load(open(os.path.join(view_dir, 'meta.json')))
    sl = meta['slices'][:max_slices] if max_slices else meta['slices']
    stack = np.stack([np.asarray(Image.open(os.path.join(view_dir, s['file'])).convert('RGB'), np.float32) / 255
                      for s in sl])
    focus = np.array([s['focus_depth_mm'] for s in sl], np.float32)
    depth = np.load(os.path.join(view_dir, 'depth.npy'))
    mask = np.asarray(Image.open(os.path.join(view_dir, 'mask.png'))) > 0
    return stack, focus, depth, mask, meta

def best_focus_index(depth, focus):
    d = np.where(np.isfinite(depth), depth, focus[-1])
    return np.argmin(np.abs(d[..., None] - focus[None, None, :]), axis=-1)

if __name__ == '__main__':
    import sys
    for v in iter_views(sys.argv[1] if len(sys.argv) > 1 else 'renders'):
        stack, focus, depth, mask, meta = load_view(v)
        fin = np.isfinite(depth) & mask
        print(f"{v}: {stack.shape[0]} slices {stack.shape[1:3]}, specimen depth "
              f"{depth[fin].min():.2f}-{depth[fin].max():.2f} mm, step {meta['stack']['step_mm']*1000:.0f} um, "
              f"{meta['object_pixel_size_mm']*1000:.1f} um/px")
