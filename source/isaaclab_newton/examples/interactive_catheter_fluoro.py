"""Interactive X-ray fluoroscopy catheter simulator — Gradio web UI.

Opens a browser-accessible interface at http://localhost:7860

Controls
--------
Advance      : push catheter forward (3 physics substeps per click)
Retract      : pull catheter back    (3 physics substeps per click)
Rotate CW    : clockwise torque on proximal end
Rotate CCW   : counter-clockwise torque
Idle step    : physics step with zero control — gravity deforms rod
Reset        : restore catheter to initial straight position
Projection   : dropdown — AP / LAO-45 / Lateral / RAO-30
Advance speed: slider — insertion velocity in mm/s
Show DSA Frame: 3-dispatch composite — background + fat-catheter + catheter

Architecture
------------
The script lives inside the Isaac Lab extension (isaaclab_newton/examples/) and
imports two components:

  - Physics : isaaclab_newton.solvers.XCathRodSolver  (this repo)
  - Renderer: fluorosim.rendering.SlangDiffDRRRenderer (fluoro-simulator repo)

Both packages are resolved in the following order:
  1. If already pip-installed (pip install -e .) — used directly, no path work.
  2. If not installed, paths are inferred relative to this file's location and
     a known sibling directory for fluoro-simulator.

The Slang shader (.slang) is resolved relative to the renderer module itself,
not relative to cwd, so no special working directory is required.

Recommended one-time setup (run once per environment):

    pip install -e /path/to/IsaacLab/source/isaaclab_newton
    pip install -e /path/to/i4h-sensor-simulation-internal/fluoro-simulator[all]

Usage
-----
    conda activate isaaclab
    python3 examples/interactive_catheter_fluoro.py [--ct-dir /tmp/patient_001]

Then open http://localhost:7860 in your browser.
If running on a remote server, forward the port:
    ssh -L 7860:localhost:7860 <server>
"""

import sys, os, math, json, time, argparse, importlib.util
import numpy as np
from pathlib import Path
from PIL import Image
import gradio as gr

# ── package resolution ────────────────────────────────────────────────────────
# Both isaaclab_newton and fluorosim are proper Python packages (setup.py /
# pyproject.toml).  If they are pip-installed nothing below is needed.
# When running from a bare checkout, we locate them relative to this file.

def _ensure_importable(package: str, *candidate_dirs: str) -> None:
    """Add the first existing candidate directory to sys.path if *package* is
    not already importable.  Raises ImportError if none of the candidates work.
    """
    if importlib.util.find_spec(package) is not None:
        return
    for d in candidate_dirs:
        if Path(d).is_dir():
            sys.path.insert(0, d)
            if importlib.util.find_spec(package) is not None:
                return
    raise ImportError(
        f"Cannot import '{package}'.  Install it with:\n"
        f"  pip install -e <repo_root>\n"
        f"Searched: {list(candidate_dirs)}"
    )

_SCRIPT_DIR  = Path(__file__).resolve().parent          # …/examples/
_ISAAC_ROOT  = str(_SCRIPT_DIR.parent)                  # …/isaaclab_newton/
_ISAACLAB_ROOT = str(_SCRIPT_DIR.parent.parent.parent)  # …/IsaacLab/

# fluoro-simulator lives in a sibling repo next to IsaacLab
_FLUORO_SIBLING = str(Path(_ISAACLAB_ROOT).parent /
                       'i4h-sensor-simulation-internal' / 'fluoro-simulator')

_ensure_importable('isaaclab_newton', _ISAAC_ROOT)
_ensure_importable('fluorosim',       _FLUORO_SIBLING)

# ── CLI arguments ─────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(
    description='Interactive X-ray fluoroscopy catheter simulator (Gradio UI)')
_parser.add_argument(
    '--ct-dir', default='/tmp/patient_001',
    metavar='PATH',
    help='Directory containing mu_volume.npy and metadata.json '
         '(default: /tmp/patient_001)')
_parser.add_argument(
    '--port', type=int, default=7860,
    help='Gradio server port (default: 7860)')
_parser.add_argument(
    '--share', action='store_true', default=False,
    help='Create a public Gradio tunnel link (gradio.live)')
_args, _ = _parser.parse_known_args()

CT_DIR = _args.ct_dir

import warp as wp
from fluorosim.vasculature import extract_vessel_mesh
from fluorosim.rendering.diffdrr_slang_renderer import (
    SlangDiffDRRRenderer, SlangDiffDRRConfig, CatheterSegmentData)
from isaaclab_newton.solvers import XCathRodSolver
from isaaclab_newton.solvers.rod_data import RodConfig
import torch

# ── constants ─────────────────────────────────────────────────────────────────
DET_SIZE    = 256
CATHETER_R  = 1.8   # mm  — actual wire radius for Beer-Lambert compositing
CATHETER_MU = 0.50  # mm⁻¹ — NiTi shaft linear attenuation coefficient
DSA_FAT_R   = 2.5   # mm  — fat-catheter radius for vessel-lumen DSA highlight
DSA_FAT_MU  = 0.80  # mm⁻¹ — fat-catheter attenuation (must dominate CT bone noise)
PHYSICS_FPS = 30
DT          = 1.0 / PHYSICS_FPS

PROJECTIONS = {
    'AP (0°)':       np.zeros((1, 3), dtype=np.float32),
    'LAO-45':        np.array([[0., math.radians(45),   0.]], dtype=np.float32),
    'Lateral (90°)': np.array([[0., math.radians(90),   0.]], dtype=np.float32),
    'RAO-30':        np.array([[0., math.radians(-30),  0.]], dtype=np.float32),
}

# ── pre-allocated hot-path constants ──────────────────────────────────────────
# Zero-translation array: shared across every render call, never mutated.
_TRANS_ZERO: np.ndarray = np.zeros((1, 3), dtype=np.float32)

# CUDA availability checked once at import; queried every physics step otherwise.
_CUDA_AVAILABLE: bool = torch.cuda.is_available()

# DSA panel header bars: allocated once, reused on every do_dsa() call.
_DSA_BAR    = np.zeros((14, DET_SIZE, 3), dtype=np.uint8); _DSA_BAR[:, :, 1]    = 140
_FLUORO_BAR = np.zeros((14, DET_SIZE, 3), dtype=np.uint8); _FLUORO_BAR[:, :, 2] = 180
_CL_BAR     = np.zeros((14, DET_SIZE, 3), dtype=np.uint8); _CL_BAR[:, :, 0]     = 0; _CL_BAR[:, :, 1] = 200; _CL_BAR[:, :, 2] = 160

# ── global simulation state (initialised once at startup) ─────────────────────
_sim = {}


# ── vessel mask helpers ───────────────────────────────────────────────────────

def _build_vessel_mask_downsampled(mu_zyx, meta):
    """Build a downsampled (4×) vessel mask used to generate the physics collision mesh."""
    sz_mm, sy_mm, sx_mm = meta['spacing_zyx_mm']
    ox, oy, oz = meta['origin_xyz_mm']
    nz, ny, nx = mu_zyx.shape
    cx_mm = ox + (nx / 2) * sx_mm
    cy_mm = oy + (ny / 2) * sy_mm
    cz_mm = oz + (nz * 0.45) * sz_mm
    DS = 4
    nzd, nyd, nxd = nz // DS, ny // DS, nx // DS
    mask = np.zeros((nzd, nyd, nxd), dtype=np.uint8)
    vcy  = nyd // 2
    vcz  = int(round((cz_mm - oz) / (sz_mm * DS)))
    r_y  = int(round(8.0 / (sy_mm * DS)))
    r_z  = int(round(8.0 / (sz_mm * DS)))
    for xi in range(nxd):
        frac   = xi / nxd
        dz_off = int(round(6 * math.sin(math.pi * frac)))
        for yi in range(max(0, vcy - r_y - 2), min(nyd, vcy + r_y + 2)):
            for zi in range(max(0, vcz + dz_off - r_z - 2),
                            min(nzd, vcz + dz_off + r_z + 2)):
                dy = (yi - vcy)              / r_y
                dz = (zi - (vcz + dz_off))   / r_z
                if dy * dy + dz * dz <= 1.0:
                    mask[zi, yi, xi] = 1
    spacing_ds = (sz_mm * DS, sy_mm * DS, sx_mm * DS)
    return mask, spacing_ds, (ox, oy, oz), (cx_mm, cy_mm, cz_mm)


# ── coordinate conversion ─────────────────────────────────────────────────────

def _pos_to_vol_mm(pos_m: np.ndarray):
    """Convert physics-world positions (metres) to CT volume coordinates (mm)."""
    pos_ct_mm  = (pos_m - _sim['local_z0_m'] + _sim['ct_offset_m']) * 1000.0
    pos_vol_mm = pos_ct_mm - _sim['ct_origin_mm']
    return pos_vol_mm, pos_ct_mm


def _catheter_segment_data(pos_vol_mm: np.ndarray,
                            radius: float, mu: float) -> CatheterSegmentData:
    """Pack rod particle positions into a CatheterSegmentData buffer.

    Scalar radius/mu are passed directly — the renderer's to_structured_array()
    uses np.broadcast_to internally, so no per-call np.full allocation is needed.
    positions is passed without astype: the renderer copies it inside
    to_structured_array(), so a second copy here would be redundant.
    """
    return CatheterSegmentData(positions=pos_vol_mm, radii=radius, mu_values=mu)


# ── physics diagnostics ───────────────────────────────────────────────────────

def _catheter_bend_mm(pos_vol_mm: np.ndarray) -> float:
    """Max perpendicular distance from the straight line between the two endpoints.

    A straight, unconstrained catheter returns ~0 mm.
    When vessel collision is actively deflecting the rod this grows to 5–30 mm,
    providing a live confirmation that containment constraints are firing.
    """
    if len(pos_vol_mm) < 3:
        return 0.0
    start, end = pos_vol_mm[0], pos_vol_mm[-1]
    axis = end - start
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-6:
        return float(np.max(np.linalg.norm(pos_vol_mm - start, axis=1)))
    axis_unit = axis / axis_len
    vecs      = pos_vol_mm - start
    proj      = np.outer(np.dot(vecs, axis_unit), axis_unit)
    perp_dist = np.linalg.norm(vecs - proj, axis=1)
    return float(np.max(perp_dist))


# ── rendering helpers ─────────────────────────────────────────────────────────

def _render(proj_name: str) -> Image.Image:
    """Single Beer-Lambert render — anatomy + actual catheter wire."""
    pos_vol_mm, pos_ct_mm = _pos_to_vol_mm(
        _sim['solver'].positions.cpu().numpy())
    cat = _catheter_segment_data(pos_vol_mm, CATHETER_R, CATHETER_MU)

    t0  = time.perf_counter()
    img = _sim['renderer'].render_batch_with_catheter(
        PROJECTIONS[proj_name], _TRANS_ZERO, [cat])[0]
    _sim['t_render_ms'] = (time.perf_counter() - t0) * 1000.0
    _sim['tip_ct_mm']   = pos_ct_mm[-1]

    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    # Directly build RGB array — avoids PIL's internal grayscale→RGB pass.
    return Image.fromarray(np.stack([arr, arr, arr], axis=-1), mode='RGB')


def _render_dsa(proj_name: str) -> Image.Image:
    """Three-dispatch DSA composite: [vessel lumen (green) | fluoro + catheter].

    Dispatch 1 — background DRR (anatomy, no catheter) → bg_drr
    Dispatch 2 — fat-catheter DRR (radius=DSA_FAT_R)   → fat_drr
    Dispatch 3 — actual catheter DRR (radius=CATHETER_R) → fluoro

    Vessel lumen signal = fat_drr − bg_drr  (positive where fat catheter projected).
    bg_drr is reused as the DSA overlay background (identical to a 4th dispatch).

    All three dispatches share the same C-arm rotation matrix, guaranteeing
    spatial registration between the DSA panel and the fluoro panel.
    """
    renderer   = _sim['renderer']
    rot        = PROJECTIONS[proj_name]
    pos_vol_mm, _ = _pos_to_vol_mm(_sim['solver'].positions.cpu().numpy())

    # Dispatch 1: background
    bg_drr = renderer.render_batch(rot, _TRANS_ZERO)[0]

    # Dispatch 2: fat catheter at vessel-lumen radius
    fat_drr = renderer.render_batch_with_catheter(
        rot, _TRANS_ZERO,
        [_catheter_segment_data(pos_vol_mm, DSA_FAT_R, DSA_FAT_MU)])[0]

    # Dispatch 3: actual catheter wire
    fluoro = renderer.render_batch_with_catheter(
        rot, _TRANS_ZERO,
        [_catheter_segment_data(pos_vol_mm, CATHETER_R, CATHETER_MU)])[0]

    # ── vessel lumen signal ────────────────────────────────────────────────────
    # fat_drr > bg_drr where the fat catheter added attenuation (inverted DRR:
    # high-μ regions are bright, so fat catheter brightens its footprint).
    signal  = np.clip(fat_drr.astype(np.float32) - bg_drr.astype(np.float32), 0, None)
    s_max   = float(signal.max())
    dsa_raw = np.sqrt(signal / s_max) if s_max > 1e-8 else np.zeros_like(signal)

    # ── composite DSA panel: anatomy (grey) + vessel lumen (green tint) ───────
    bg_u8   = (np.clip(bg_drr, 0, 1) * 255).astype(np.uint8)
    bg_i16  = bg_u8.astype(np.int16)   # cast once; reused for all three channels
    boost   = dsa_raw * 160
    rb_adj  = boost * 0.5               # r and b channels dampened equally
    r_ch = np.clip(bg_i16 - rb_adj, 0, 255).astype(np.uint8)
    g_ch = np.clip(bg_i16 + boost,  0, 255).astype(np.uint8)
    b_ch = np.clip(bg_i16 - rb_adj, 0, 255).astype(np.uint8)
    dsa_panel_img = np.stack([r_ch, g_ch, b_ch], axis=-1)

    fluoro_u8  = (np.clip(fluoro, 0, 1) * 255).astype(np.uint8)
    fluoro_rgb = np.stack([fluoro_u8, fluoro_u8, fluoro_u8], axis=-1)

    # _DSA_BAR / _FLUORO_BAR are pre-allocated module-level constants.
    return Image.fromarray(
        np.hstack([np.vstack([_DSA_BAR,    dsa_panel_img]),
                   np.vstack([_FLUORO_BAR, fluoro_rgb])]),
        mode='RGB',
    )


# ── centerline projection helpers ────────────────────────────────────────────

def _euler_zxy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Python port of the Slang ZXY euler convention: R = Rz @ Rx @ Ry."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [cz * cy - sz * sx * sy, -sz * cx, cz * sy + sz * sx * cy],
        [sz * cy + cz * sx * sy,  cz * cx, sz * sy - cz * sx * cy],
        [-cx * sy,                sx,      cx * cy                ],
    ], dtype=np.float64)


def _project_rod_to_pixels(proj_name: str) -> np.ndarray:
    """Project rod particle positions (vol_mm) to 2D pixel coordinates.

    Replicates the Slang shader geometry exactly:
      source_local  = (0, 0, -SID)
      detector_z    = SDD - SID
      pixel (u, v)  = (det_x / ps + W/2,  det_y / ps + H/2)

    Returns (N, 2) float32 array of (u, v) pixel coordinates.
    """
    euler = PROJECTIONS[proj_name][0]
    R = _euler_zxy_to_matrix(float(euler[0]), float(euler[1]), float(euler[2]))

    SID = 510.0    # source-to-isocenter mm  (SlangDiffDRRConfig default)
    SDD = 1020.0   # source-to-detector mm   (SlangDiffDRRConfig default)
    ps  = 0.5      # pixel spacing mm/px     (SlangDiffDRRConfig default)
    W = H = float(DET_SIZE)

    nx, ny, nz = _sim['vol_shape_xyz']
    sx, sy, sz = _sim['spacing_xyz_mm']
    iso = np.array([nx * sx * 0.5, ny * sy * 0.5, nz * sz * 0.5], dtype=np.float64)

    pos_vol_mm, _ = _pos_to_vol_mm(_sim['solver'].positions.cpu().numpy())
    pts = pos_vol_mm.astype(np.float64)

    # Transform to C-arm local frame: P_local = R^T @ (P - iso)
    pts_local = (pts - iso) @ R   # (N, 3), row-vector convention

    denom = pts_local[:, 2] + SID
    det_x = SDD * pts_local[:, 0] / denom
    det_y = SDD * pts_local[:, 1] / denom

    u = det_x / ps + W * 0.5
    v = det_y / ps + H * 0.5
    return np.stack([u, v], axis=-1).astype(np.float32)


def _draw_centerline_overlay(base_img: Image.Image, uv: np.ndarray) -> Image.Image:
    """Draw projected rod centerline on *base_img* as a cyan polyline + dots."""
    from PIL import ImageDraw
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    r = 3
    pts = [(int(round(float(u))), int(round(float(v)))) for u, v in uv
           if 0 <= float(u) < W and 0 <= float(v) < H]
    if len(pts) >= 2:
        draw.line(pts, fill=(0, 255, 180), width=2)
    for (u, v) in pts:
        draw.ellipse([u - r, v - r, u + r, v + r], fill=(0, 255, 180))
    return img


def _render_side_by_side_centerline(proj_name: str) -> Image.Image:
    """Left panel: standard fluoro. Right panel: fluoro + projected centerline."""
    base   = _render(proj_name)
    uv     = _project_rod_to_pixels(proj_name)
    overlay = _draw_centerline_overlay(base, uv)

    arr_left  = np.vstack([_FLUORO_BAR, np.array(base)])
    arr_right = np.vstack([_CL_BAR,     np.array(overlay)])
    return Image.fromarray(np.hstack([arr_left, arr_right]), mode='RGB')


# ── simulation initialisation ─────────────────────────────────────────────────

def init_simulation():
    print('Loading CT data...')
    mu_zyx = np.load(os.path.join(CT_DIR, 'mu_volume.npy'))
    meta   = json.load(open(os.path.join(CT_DIR, 'metadata.json')))
    sz_mm, sy_mm, sx_mm = meta['spacing_zyx_mm']
    ox, oy, oz = meta['origin_xyz_mm']
    nz, ny, nx = mu_zyx.shape

    print('Building vessel mesh...')
    mask, sp_ds, origin_xyz, (cx_mm, cy_mm, cz_mm) = \
        _build_vessel_mask_downsampled(mu_zyx, meta)
    mesh = extract_vessel_mesh(mask, spacing_zyx_mm=sp_ds,
                               origin_xyz_mm=origin_xyz, device='cuda')
    print(f'  Mesh: {mesh.points.numpy().shape[0]} verts')

    x_start_mm    = ox + nx * sx_mm * 0.15
    x_end_mm      = ox + nx * sx_mm * 0.80
    rod_len_m     = (x_end_mm - x_start_mm) / 1000.0
    seg_len       = rod_len_m / 20
    track_start_m = np.array([x_start_mm / 1000, cy_mm / 1000, cz_mm / 1000],
                              dtype=np.float32)

    print('Initialising physics solver...')
    rod_cfg = RodConfig()
    rod_cfg.device                    = 'cuda'
    rod_cfg.geometry.num_segments     = 20
    rod_cfg.geometry.rest_length      = rod_len_m
    rod_cfg.geometry.segment_length   = seg_len
    rod_cfg.solver.num_substeps       = 8

    # track_start must be in solver-space (rod lives at x=0..rod_len_m, y=0, z=initial_height)
    track_start_solver = np.array([0.0, 0.0, float(track_start_m[2])], dtype=np.float32)

    solver = XCathRodSolver(
        rod_cfg,
        collision_mesh=mesh,
        track_start=track_start_solver,
        track_dir=np.array([1., 0., 0.], dtype=np.float32),
        track_length=rod_len_m,
        tip_num_edges=6,
        particle_radius=0.001,
        segment_length=seg_len,
        collision_iterations=2,
        sign_scale=1.0, target_phi=-0.001, max_dist=0.025,
        initial_height=float(track_start_m[2]),
    )

    print('Initialising renderer...')
    renderer = SlangDiffDRRRenderer(
        mu_zyx,
        spacing_zyx_mm=(sz_mm, sy_mm, sx_mm),
        cfg=SlangDiffDRRConfig(det_width_px=DET_SIZE, det_height_px=DET_SIZE,
                               step_mm=1.0, i0=1.0),
        num_envs=1,
    )

    _sim['solver']       = solver
    _sim['renderer']     = renderer
    _sim['ct_offset_m']  = track_start_m.copy()
    _sim['local_z0_m']   = np.array([0., 0., float(track_start_m[2])],
                                     dtype=np.float32)
    _sim['ct_origin_mm'] = np.array([ox, oy, oz], dtype=np.float32)
    _sim['vol_shape_xyz']  = (int(nx), int(ny), int(nz))
    _sim['spacing_xyz_mm'] = (float(sx_mm), float(sy_mm), float(sz_mm))
    # Store as NumPy — solver.positions/.orientations are properties returning
    # wp.to_torch(...).clone(), so reset must write via the underlying Warp buffer.
    _sim['initial_pos']  = solver.positions.cpu().numpy().copy()
    _sim['initial_ori']  = solver.orientations.cpu().numpy().copy()
    _sim['tip_ct_mm']    = np.zeros(3, dtype=np.float32)
    _sim['t_render_ms']  = 0.0
    _sim['render_count'] = 0

    print('Warm-up render...')
    _render('LAO-45')
    print('Ready.')


# ── Gradio action callbacks ───────────────────────────────────────────────────

def _pick_render(proj_name: str, show_cl: bool) -> Image.Image:
    """Return the appropriate image based on the centerline toggle."""
    if show_cl:
        return _render_side_by_side_centerline(proj_name)
    return _render(proj_name)


def _step_and_render(velocity: float, torque: float,
                     proj_name: str, show_cl: bool, steps: int = 1) -> tuple:
    solver = _sim['solver']

    t_phys = time.perf_counter()
    for _ in range(steps):
        solver.apply_proximal_control(velocity, torque, DT)
        solver.step(DT)
    if _CUDA_AVAILABLE:
        torch.cuda.synchronize()
    t_phys_ms = (time.perf_counter() - t_phys) * 1000.0

    img         = _pick_render(proj_name, show_cl)
    t_render_ms = _sim['t_render_ms']
    t_loop_ms   = t_phys_ms + t_render_ms
    _sim['render_count'] += 1
    tip  = _sim['tip_ct_mm']

    pos_vol_mm, _ = _pos_to_vol_mm(solver.positions.cpu().numpy())
    bend_mm = _catheter_bend_mm(pos_vol_mm)
    bend_flag = '  ← vessel wall deflecting rod' if bend_mm > 2.0 else '  (straight / unconstrained)'

    info = (f"Projection   : {proj_name}\n"
            f"Tip (CT mm)  : X={tip[0]:.1f}  Y={tip[1]:.1f}  Z={tip[2]:.1f}\n"
            f"Catheter bend: {bend_mm:.1f} mm{bend_flag}\n"
            f"Physics step : {t_phys_ms:.1f} ms  ({steps} substep(s))\n"
            f"Render (GPU) : {t_render_ms:.1f} ms\n"
            f"Sim loop     : {t_loop_ms:.1f} ms  (~{1000/max(t_loop_ms, 1):.0f} fps)\n"
            f"Frame #      : {_sim['render_count']}")
    return img, info


def do_advance(proj, speed, show_cl):
    return _step_and_render(float(speed) / 1000, 0.0, proj, show_cl, steps=3)

def do_retract(proj, speed, show_cl):
    return _step_and_render(-float(speed) / 1000, 0.0, proj, show_cl, steps=3)

def do_rotate_cw(proj, speed, show_cl):
    return _step_and_render(0.0,  0.015, proj, show_cl, steps=2)

def do_rotate_ccw(proj, speed, show_cl):
    return _step_and_render(0.0, -0.015, proj, show_cl, steps=2)

def do_idle(proj, speed, show_cl):
    return _step_and_render(0.0, 0.0, proj, show_cl, steps=1)

def do_reset(proj, speed, show_cl):
    solver = _sim['solver']
    ws     = solver._ws

    # Write directly into Warp buffers — the solver properties return clones,
    # so .copy_() on them is a no-op against the actual GPU allocation.
    init_pos = torch.from_numpy(_sim['initial_pos'])
    init_ori = torch.from_numpy(_sim['initial_ori'])

    wp.to_torch(ws.positions).copy_(init_pos)
    wp.to_torch(ws.predicted_positions).copy_(init_pos)
    if hasattr(ws, 'prev_positions'):
        wp.to_torch(ws.prev_positions).copy_(init_pos)

    wp.to_torch(ws.velocities).zero_()
    if hasattr(ws, 'forces'):
        wp.to_torch(ws.forces).zero_()

    wp.to_torch(ws.orientations).copy_(init_ori)
    if hasattr(ws, 'predicted_orientations'):
        wp.to_torch(ws.predicted_orientations).copy_(init_ori)
    if hasattr(ws, 'prev_orientations'):
        wp.to_torch(ws.prev_orientations).copy_(init_ori)

    # Invalidate the CUDA graph — next step() re-captures from the restored state.
    solver.reset_cuda_graph()
    torch.cuda.synchronize()

    _sim['render_count'] = 0
    return _pick_render(proj, show_cl), "Reset to initial position."

def do_change_view(proj, speed, show_cl):
    img = _pick_render(proj, show_cl)
    tip = _sim['tip_ct_mm']
    return img, (f"Projection : {proj}\n"
                 f"Tip (CT mm): X={tip[0]:.1f}  Y={tip[1]:.1f}  Z={tip[2]:.1f}")

def do_dsa(proj, speed, bolus_t, show_cl):
    t0        = time.perf_counter()
    composite = _render_dsa(proj)
    elapsed   = (time.perf_counter() - t0) * 1000.0
    tip = _sim['tip_ct_mm']
    info = (f"DSA roadmap + Fluoro — bolus t={bolus_t:.1f}s\n"
            f"LEFT  (green bar) : DSA roadmap — vessel lumen highlighted in green\n"
            f"RIGHT (blue bar)  : Live fluoro — skull anatomy + catheter wire\n"
            f"Projection : {proj}\n"
            f"Tip (CT mm): X={tip[0]:.1f}  Y={tip[1]:.1f}  Z={tip[2]:.1f}\n"
            f"Total render: {elapsed:.0f} ms  (3 dispatches: bg + fat-catheter + catheter)")
    return composite, info



# ── Gradio UI layout ──────────────────────────────────────────────────────────

def build_ui():
    init_simulation()
    initial_img = _render('LAO-45')

    with gr.Blocks(title='XCath Fluoroscopy Simulator') as demo:

        gr.Markdown(
            "## XCath Fluoroscopy Simulator\n"
            "Physics-based catheter simulation inside real cranial CT anatomy. "
            "Vessel-mesh collision active. "
            "GPU-rendered DRR via Slang Beer-Lambert ray marching."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                proj_dd  = gr.Dropdown(choices=list(PROJECTIONS.keys()),
                                       value='LAO-45', label='C-arm projection')
                speed_sl = gr.Slider(minimum=1, maximum=20, value=5, step=1,
                                     label='Advance speed (mm/s)')
                gr.Markdown("---")
                btn_adv  = gr.Button("▶  Advance",             variant='primary')
                btn_ret  = gr.Button("◀  Retract",             variant='secondary')
                btn_rcw  = gr.Button("↻  Rotate CW",           variant='secondary')
                btn_rccw = gr.Button("↺  Rotate CCW",          variant='secondary')
                gr.Markdown("---")
                btn_idle  = gr.Button("⏸  Idle step (gravity)", variant='secondary')
                btn_reset = gr.Button("⟳  Reset",               variant='stop')
                gr.Markdown("---\n**Display**")
                cl_toggle = gr.Checkbox(value=False,
                                        label="Show centerline overlay  (fluoro | fluoro+CL)")
                gr.Markdown("---\n**DSA / Contrast Injection**")
                bolus_sl = gr.Slider(minimum=0.0, maximum=12.0, value=3.0, step=0.5,
                                     label='Bolus time (s)  — 0=injection, peak ≈3–4s')
                btn_dsa  = gr.Button("Show DSA Frame",          variant='primary')

            with gr.Column(scale=2):
                fluoro_img = gr.Image(value=initial_img, label='Fluoroscopy (DRR)',
                                      type='pil', height=420)
                info_box   = gr.Textbox(label='Simulation info', lines=6,
                                        interactive=False, value='Ready.')

        outputs = [fluoro_img, info_box]
        inputs  = [proj_dd, speed_sl, cl_toggle]

        btn_adv.click(do_advance,     inputs=inputs,                           outputs=outputs)
        btn_ret.click(do_retract,     inputs=inputs,                           outputs=outputs)
        btn_rcw.click(do_rotate_cw,   inputs=inputs,                           outputs=outputs)
        btn_rccw.click(do_rotate_ccw, inputs=inputs,                           outputs=outputs)
        btn_idle.click(do_idle,       inputs=inputs,                           outputs=outputs)
        btn_reset.click(do_reset,     inputs=inputs,                           outputs=outputs)
        proj_dd.change(do_change_view, inputs=inputs,                          outputs=outputs)
        cl_toggle.change(do_change_view, inputs=inputs,                        outputs=outputs)
        btn_dsa.click(do_dsa, inputs=[proj_dd, speed_sl, bolus_sl, cl_toggle], outputs=outputs)

    return demo


def main():
    """Entry point — registered as the ``xcath-fluoro`` console script."""
    demo = build_ui()
    demo.launch(server_name='0.0.0.0', server_port=_args.port,
                share=_args.share, theme=gr.themes.Base())


if __name__ == '__main__':
    main()
