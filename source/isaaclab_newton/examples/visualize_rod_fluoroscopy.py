#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Fluoroscopy Visualization with Beer-Lambert Catheter Compositing

Composites the simulated rod (catheter/guidewire) onto DRR fluoroscopy images
using physically-correct Beer-Lambert transmission attenuation. The catheter
darkens the X-ray background based on material attenuation coefficients and
projected cylindrical thickness — matching real fluoroscopy appearance.

DRR images are loaded from:
    /home/cdinea/Downloads/new_CTdata/drr_output_diffdrr_slang/

Available views (auto-selected by C-arm angle or keyboard):
    AP       →  xray_diffdrr_slang_AP.png        (0°)
    LAO 30   →  xray_diffdrr_slang_LAO_30.png    (+30°)
    Lateral  →  xray_diffdrr_slang_Lateral.png   (+90°)
    RAO 30   →  xray_diffdrr_slang_RAO_30.png    (-30°)

Dependencies: torch, warp-lang, numpy, opencv-python
Optional:     pydicom, matplotlib (fallback)

Usage:
    python visualize_rod_fluoroscopy.py                        # AP view, production solver
    python visualize_rod_fluoroscopy.py --use-xpbd             # XPBD block-Thomas solver
    python visualize_rod_fluoroscopy.py --use-slang            # GPU Slang Beer-Lambert (fused DRR+catheter)
    python visualize_rod_fluoroscopy.py --ap-and-lateral       # AP + Lateral side-by-side
    python visualize_rod_fluoroscopy.py --view Lateral         # start on lateral DRR
    python visualize_rod_fluoroscopy.py --save-video out.mp4
    python visualize_rod_fluoroscopy.py --use-matplotlib       # fallback (no OpenCV)

Interactive keys (OpenCV mode):
    Q        quit
    SPACE    pause / resume
    1        AP view
    2        LAO 30° view
    3        Lateral view
    4        RAO 30° view
    +/-      rotate C-arm ±5°
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

# ── DRR image directory ─────────────────────────────────────────────────
DRR_DIR = "/home/cdinea/Downloads/new_CTdata/drr_output_diffdrr_slang"

# Maps a view name to (filename, C-arm LAO/RAO angle in degrees)
DRR_VIEWS: dict[str, tuple[str, float]] = {
    "AP":      ("xray_diffdrr_slang_AP.png",       0.0),
    "LAO_30":  ("xray_diffdrr_slang_LAO_30.png",  30.0),
    "Lateral": ("xray_diffdrr_slang_Lateral.png",  90.0),
    "RAO_30":  ("xray_diffdrr_slang_RAO_30.png", -30.0),
}

# ── Rod Solvers ─────────────────────────────────────────────────────────
from isaaclab_newton.solvers import (
    RodConfig,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolver,
    RodSolverConfig,
)
from isaaclab_newton.solvers import XPBDRodSolver


# ═══════════════════════════════════════════════════════════════════════
# C-arm geometry helpers
# ═══════════════════════════════════════════════════════════════════════

def make_carm_intrinsics(
    sid: float = 1000.0,
    pixel_spacing: float = 0.3,
    image_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Build 3×3 camera intrinsic matrix for a C-arm fluoroscope.

    The focal length in pixels is  f_px = SID / pixel_spacing.
    The principal point is placed at the detector centre.

    Args:
        sid: Source-to-Image-Distance in mm.
        pixel_spacing: Detector pixel pitch in mm/px.
        image_size: (width, height) of the detector image in pixels.

    Returns:
        K: 3×3 intrinsic matrix (numpy, float64).
    """
    f = sid / pixel_spacing  # focal length in pixels
    cx = image_size[0] / 2.0
    cy = image_size[1] / 2.0
    return np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def make_carm_extrinsics(
    sod: float = 600.0,
    carm_angle_deg: float = 0.0,
    cran_caud_deg: float = 0.0,
) -> np.ndarray:
    """Build 3×4 extrinsic [R | t] for a C-arm at a given angulation.

    Convention:
      - The iso-centre is at world origin.
      - At 0° / 0°  the X-ray source is at  (0, 0, -SOD)  looking +Z.
      - LAO/RAO rotates around the patient cranio-caudal (Y) axis.
      - CRAN/CAUD rotates around the lateral (X) axis.

    Args:
        sod: Source-to-Object-Distance in mm.
        carm_angle_deg: LAO(+) / RAO(-) angulation in degrees.
        cran_caud_deg: CRAN(+) / CAUD(-) angulation in degrees.

    Returns:
        [R | t]: 3×4 extrinsic matrix (numpy, float64).
    """
    alpha = math.radians(carm_angle_deg)
    beta = math.radians(cran_caud_deg)

    # Rotation: R = Rx(beta) @ Ry(alpha)
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)

    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    Rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    R = Rx @ Ry

    # Translation: camera is at  R^T * (0, 0, SOD)  in world → t = -R @ cam_world
    cam_world = R.T @ np.array([0.0, 0.0, -sod])
    t = -R @ cam_world

    return np.hstack([R, t.reshape(3, 1)])


def project_points_3d_to_2d(
    points_3d: np.ndarray,
    K: np.ndarray,
    Rt: np.ndarray,
) -> np.ndarray:
    """Project Nx3 world points to Nx2 pixel coordinates.

    Args:
        points_3d: (N, 3) array of 3D points in world frame.
        K: 3×3 intrinsic matrix.
        Rt: 3×4 extrinsic matrix [R | t].

    Returns:
        (N, 2) array of (u, v) pixel coordinates.
    """
    P = K @ Rt  # 3×4 full projection matrix
    ones = np.ones((points_3d.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([points_3d, ones])  # (N, 4)
    proj = (P @ pts_h.T).T  # (N, 3)
    uv = proj[:, :2] / proj[:, 2:3]
    return uv.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Fluoroscopy background helpers
# ═══════════════════════════════════════════════════════════════════════

def load_fluoroscopy_background(
    path: str | None,
    image_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Load or synthesise a fluoroscopy background image.

    Supports: PNG, JPEG, DICOM (.dcm).
    If *path* is None a synthetic dark image with vignetting is generated.

    Returns:
        BGR uint8 image of shape (H, W, 3).
    """
    import cv2

    if path is None:
        return _synthesise_fluoroscopy(image_size)

    ext = path.rsplit(".", 1)[-1].lower()

    if ext == "dcm":
        try:
            import pydicom
        except ImportError:
            raise ImportError("pydicom is required to load DICOM files: pip install pydicom")
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        # Window/level normalisation
        if hasattr(ds, "WindowCenter"):
            wc = float(ds.WindowCenter if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter[0])
            ww = float(ds.WindowWidth if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth[0])
            lo, hi = wc - ww / 2, wc + ww / 2
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        img = (arr * 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, image_size)
        return img

    # Generic image (PNG, JPEG, …)
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: could not load '{path}', using synthetic background")
        return _synthesise_fluoroscopy(image_size)
    return cv2.resize(img, image_size)


def _synthesise_fluoroscopy(
    image_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Generate a synthetic fluoroscopy-like background (dark with vignette + noise)."""
    W, H = image_size
    # Base grey level (typical dark fluoroscopy)
    base = np.full((H, W), 30, dtype=np.float32)

    # Circular vignette
    Y, X = np.ogrid[:H, :W]
    cx, cy = W / 2.0, H / 2.0
    r_max = min(cx, cy) * 0.92
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = np.clip(1.0 - (dist - r_max * 0.8) / (r_max * 0.2), 0, 1)
    base *= mask

    # Add Poisson-like noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 5, (H, W)).astype(np.float32)
    base = np.clip(base + noise, 0, 255).astype(np.uint8)

    # Some faint anatomical-ish texture (simulated spine shadow)
    for i in range(6):
        y_center = int(H * 0.25 + i * H * 0.1)
        cv2_available = True
        try:
            import cv2
        except ImportError:
            cv2_available = False
            break
        cv2.ellipse(
            base,
            (W // 2, y_center),
            (int(W * 0.08), int(H * 0.03)),
            0, 0, 360,
            45, -1,
        )

    return np.stack([base, base, base], axis=-1)  # grayscale → BGR


# ═══════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════

def _segment_attenuation_profile(num_segments: int) -> np.ndarray:
    """Per-segment linear attenuation coefficient (mu * nominal_diameter).

    Models a typical catheter construction:
      - Proximal marker band (tungsten)
      - Braided shaft (nitinol)
      - Transition zone (sparse braid + polymer)
      - Soft polymer tip
      - Distal tip marker (platinum coil)

    Returns:
        (num_segments,) array of effective mu*2r values (dimensionless).
    """
    mu = np.full(num_segments, 0.8, dtype=np.float32)
    n = num_segments

    # Proximal marker band — first 2 segments
    mu[:min(2, n)] = 3.0

    # Braided shaft — up to 60% of length
    shaft_end = int(0.6 * n)

    # Transition — 60% to 85%
    trans_start = shaft_end
    trans_end = int(0.85 * n)
    for i in range(trans_start, min(trans_end, n)):
        t = (i - trans_start) / max(trans_end - trans_start, 1)
        mu[i] = 0.8 * (1.0 - t) + 0.2 * t

    # Soft tip — 85% to 95%
    tip_start = trans_end
    tip_end = int(0.95 * n)
    mu[tip_start:min(tip_end, n)] = 0.15

    # Distal marker — last 2–3 segments (platinum)
    mu[max(0, n - 3):] = 5.0

    return mu


def composite_catheter_beer_lambert(
    background: np.ndarray,
    uv: np.ndarray,
    radii_px: np.ndarray,
    mu_profile: np.ndarray,
    *,
    scatter_sigma: float = 18.0,
    scatter_fraction: float = 0.03,
    noise_photon_count: float = 2000.0,
    detector_psf_sigma: float = 0.7,
) -> np.ndarray:
    """Composite catheter onto DRR using Beer-Lambert transmission attenuation.

    For each segment, the projected cylinder cross-section produces a chord
    thickness t(d) = 2*sqrt(r^2 - d^2) at perpendicular pixel distance d.
    The DRR background is darkened multiplicatively:

        I_final(u,v) = I_DRR(u,v) * exp(-sum_i mu_i * t_i(u,v))

    This produces physically correct darkening with smooth edges, visible
    background anatomy through the catheter, and additive opacity at
    self-crossings.

    Args:
        background: BGR uint8 DRR image (H, W, 3).
        uv: (N, 2) projected pixel coordinates of particle centres.
        radii_px: (N-1,) projected pixel radius per segment (cone-beam scaled).
        mu_profile: (N-1,) linear attenuation * nominal diameter per segment.
        scatter_sigma: Gaussian sigma (px) for veiling glare / scatter kernel.
        scatter_fraction: Fraction of blocked intensity re-added as scatter.
        noise_photon_count: Mean photon count for Poisson quantum noise.
            Set to 0 to disable noise.
        detector_psf_sigma: Gaussian sigma (px) for detector point-spread function.
            Set to 0 to disable.

    Returns:
        Composited BGR uint8 image.
    """
    import cv2

    H, W = background.shape[:2]
    N = uv.shape[0]
    if N < 2:
        return background.copy()

    bg_float = background.astype(np.float32) / 255.0

    # Build per-pixel attenuation map (single channel)
    atten_map = np.zeros((H, W), dtype=np.float32)

    for seg in range(N - 1):
        p0 = uv[seg]
        p1 = uv[seg + 1]
        r_px = float(radii_px[seg])
        mu_val = float(mu_profile[seg])

        if r_px < 0.3 or mu_val < 1e-6:
            continue

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 0.5:
            continue

        # Bounding box around this segment with padding = r_px + 1
        pad = int(r_px + 2)
        u_min = max(0, int(min(p0[0], p1[0]) - pad))
        u_max = min(W, int(max(p0[0], p1[0]) + pad + 1))
        v_min = max(0, int(min(p0[1], p1[1]) - pad))
        v_max = min(H, int(max(p0[1], p1[1]) + pad + 1))

        if u_min >= u_max or v_min >= v_max:
            continue

        # Pixel grid within bounding box
        uu = np.arange(u_min, u_max, dtype=np.float32) + 0.5
        vv = np.arange(v_min, v_max, dtype=np.float32) + 0.5
        grid_u, grid_v = np.meshgrid(uu, vv)

        # Perpendicular distance from each pixel to the segment line
        # Parameterise segment as P(t) = p0 + t * (p1 - p0), t in [0, 1]
        rel_u = grid_u - p0[0]
        rel_v = grid_v - p0[1]
        t_param = (rel_u * dx + rel_v * dy) / (seg_len * seg_len)
        t_param = np.clip(t_param, 0.0, 1.0)

        # Closest point on segment
        closest_u = p0[0] + t_param * dx
        closest_v = p0[1] + t_param * dy
        dist = np.sqrt((grid_u - closest_u) ** 2 + (grid_v - closest_v) ** 2)

        # Cylinder chord thickness: t(d) = 2 * sqrt(r^2 - d^2) for d < r
        inside = dist < r_px
        if not np.any(inside):
            continue

        chord = np.zeros_like(dist)
        chord[inside] = 2.0 * np.sqrt(r_px * r_px - dist[inside] ** 2)

        # Normalise chord by diameter so mu_val controls peak attenuation
        chord_norm = chord / (2.0 * r_px + 1e-8)

        # Accumulate attenuation (additive in exponent)
        atten_map[v_min:v_max, u_min:u_max] += mu_val * chord_norm

    # Apply Beer-Lambert: I_final = I_DRR * exp(-atten_map)
    transmission = np.exp(-atten_map)
    transmission_3ch = transmission[:, :, np.newaxis]

    composited = bg_float * transmission_3ch

    # Veiling glare / scatter: large-kernel blur of blocked intensity
    if scatter_fraction > 0 and scatter_sigma > 1.0:
        blocked = bg_float * (1.0 - transmission_3ch)
        ksize = int(scatter_sigma * 6) | 1  # ensure odd
        scatter = cv2.GaussianBlur(blocked, (ksize, ksize), scatter_sigma)
        composited += scatter_fraction * scatter

    # Detector point-spread function (small blur)
    if detector_psf_sigma > 0.2:
        ksize_psf = int(detector_psf_sigma * 6) | 1
        composited = cv2.GaussianBlur(composited, (ksize_psf, ksize_psf), detector_psf_sigma)

    # Poisson quantum noise
    if noise_photon_count > 0:
        # Scale to photon counts, apply Poisson, scale back
        rng = np.random.default_rng()
        photons = composited * noise_photon_count
        photons = np.clip(photons, 0, None)
        noisy = rng.poisson(photons.astype(np.float64)).astype(np.float32)
        composited = noisy / noise_photon_count

    composited = np.clip(composited * 255.0, 0, 255).astype(np.uint8)
    return composited


def compute_projected_radii(
    pos_3d: np.ndarray,
    K: np.ndarray,
    Rt: np.ndarray,
    physical_radius_mm: float,
) -> np.ndarray:
    """Compute per-segment projected pixel radius with cone-beam magnification.

    Args:
        pos_3d: (N, 3) world positions in mm.
        K: 3x3 intrinsic matrix.
        Rt: 3x4 extrinsic matrix.
        physical_radius_mm: Physical catheter radius in mm.

    Returns:
        (N-1,) array of projected radii in pixels.
    """
    N = pos_3d.shape[0]
    R = Rt[:, :3]
    t = Rt[:, 3]

    # Transform to camera frame
    cam_pts = (R @ pos_3d.T).T + t  # (N, 3)
    f_px = K[0, 0]

    radii = np.zeros(N - 1, dtype=np.float32)
    for seg in range(N - 1):
        midpoint_cam = 0.5 * (cam_pts[seg] + cam_pts[seg + 1])
        z_cam = midpoint_cam[2]
        if z_cam > 1.0:
            radii[seg] = physical_radius_mm * f_px / z_cam
        else:
            radii[seg] = 1.0

    return np.clip(radii, 1.0, 50.0)


def draw_rod_overlay(
    img: np.ndarray,
    uv: np.ndarray,
    rod_color: tuple[int, int, int] = (220, 220, 255),
    rod_thickness: int = 2,
    tip_color: tuple[int, int, int] = (0, 200, 255),
    fixed_color: tuple[int, int, int] = (0, 0, 255),
    draw_segments: bool = True,
) -> np.ndarray:
    """Legacy overlay (opaque polyline). Use composite_catheter_beer_lambert instead."""
    import cv2

    N = uv.shape[0]
    if N < 2:
        return img

    pts = uv.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=rod_color, thickness=rod_thickness, lineType=cv2.LINE_AA)

    glow = np.zeros_like(img)
    cv2.polylines(glow, [pts], isClosed=False, color=rod_color, thickness=rod_thickness + 4, lineType=cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=3)
    img = cv2.addWeighted(img, 1.0, glow, 0.3, 0)

    if draw_segments:
        for i in range(N):
            center = (int(uv[i, 0]), int(uv[i, 1]))
            cv2.circle(img, center, 2, rod_color, -1, cv2.LINE_AA)

    cv2.circle(img, (int(uv[0, 0]), int(uv[0, 1])), 5, fixed_color, -1, cv2.LINE_AA)
    cv2.circle(img, (int(uv[-1, 0]), int(uv[-1, 1])), 5, tip_color, -1, cv2.LINE_AA)

    return img


def draw_hud(
    img: np.ndarray,
    sim_time: float,
    tip_pos: np.ndarray,
    fps: float,
    carm_angle: float = 0.0,
) -> np.ndarray:
    """Draw heads-up display text onto the image."""
    import cv2

    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 200, 0)  # green HUD
    th = 1

    lines = [
        f"SIM  {sim_time:.2f}s",
        f"FPS  {fps:.0f}",
        f"TIP  ({tip_pos[0]:.1f}, {tip_pos[1]:.1f}, {tip_pos[2]:.1f}) mm",
        f"C-ARM  {carm_angle:.0f} deg",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (10, 20 + i * 18), font, scale, color, th, cv2.LINE_AA)

    return img


# ═══════════════════════════════════════════════════════════════════════
# Main simulation + overlay loop
# ═══════════════════════════════════════════════════════════════════════

def load_drr_backgrounds(
    image_size: tuple[int, int] = (512, 512),
    drr_dir: str = DRR_DIR,
) -> dict[str, np.ndarray]:
    """Load all available DRR images from the output directory.

    Returns:
        Dict mapping view name → BGR uint8 image.
    """
    import cv2

    backgrounds: dict[str, np.ndarray] = {}
    for view_name, (filename, _angle) in DRR_VIEWS.items():
        path = os.path.join(drr_dir, filename)
        if os.path.isfile(path):
            img = cv2.imread(path)
            if img is not None:
                backgrounds[view_name] = cv2.resize(img, image_size)
                print(f"  Loaded DRR: {filename}  ({view_name})")
            else:
                print(f"  Warning: could not decode {path}")
        else:
            print(f"  Warning: DRR not found: {path}")

    # Fallback: generate synthetic background for missing views
    for view_name in DRR_VIEWS:
        if view_name not in backgrounds:
            backgrounds[view_name] = _synthesise_fluoroscopy(image_size)

    return backgrounds


def select_drr_for_angle(
    angle_deg: float,
    backgrounds: dict[str, np.ndarray],
) -> tuple[str, np.ndarray]:
    """Pick the DRR image whose C-arm angle is closest to *angle_deg*.

    Returns:
        (view_name, background_image).
    """
    best_name = "AP"
    best_dist = float("inf")
    for view_name, (_filename, view_angle) in DRR_VIEWS.items():
        d = abs(((angle_deg - view_angle + 180) % 360) - 180)
        if d < best_dist and view_name in backgrounds:
            best_dist = d
            best_name = view_name
    return best_name, backgrounds[best_name]


def create_rod_config(
    num_segments: int = 30,
    length_mm: float = 200.0,
    radius_mm: float = 0.45,
    young_modulus: float = 1e8,
    device: str = "cuda",
) -> RodConfig:
    """Create a catheter/guidewire configuration (SI units: metres internally).

    Default length is 200 mm — a typical neuro-interventional microcatheter
    segment visible within a head/neck fluoroscopy field-of-view.
    """
    length_m = length_mm / 1000.0
    radius_m = radius_mm / 1000.0

    return RodConfig(
        material=RodMaterialConfig(
            young_modulus=young_modulus,
            density=7800.0,  # Nitinol / stainless steel
            damping=0.05,
        ),
        geometry=RodGeometryConfig(
            num_segments=num_segments,
            rest_length=length_m,
            radius=radius_m,
        ),
        solver=RodSolverConfig(
            dt=1.0 / 60.0,
            num_substeps=2,
            newton_iterations=4,
            use_direct_solver=True,
            gravity=(0.0, -9.81, 0.0),
        ),
        device=device,
    )


def _position_rod_in_anatomy(solver: RodSolver, device: str) -> None:
    """Offset the rod so it starts overlaid on the cervical spine / skull base.

    The DRR images show the head in roughly the centre of a 512×512 frame.
    We place the catheter origin near the lower cervical spine and orient it
    upward (cranially) so it follows the vertebral / carotid artery course.

    World coordinates (mm):
      X → patient-left (+)     (LAO direction)
      Y → superior (+)         (cranial direction)
      Z → anterior (+)         (AP source direction)

    The iso-centre (world origin) corresponds to the image centre.
    """
    n = solver.data.positions.shape[1]
    seg_len = solver.data.config.geometry.segment_length

    # Position catheter from lower cervical spine growing cranially (+Y).
    # Rod root at (0, -60mm, 0) — lower spine region in the DRR.
    with torch.no_grad():
        for i in range(n):
            t = i / max(n - 1, 1)
            x_mm = 8.0 * math.sin(t * math.pi)  # gentle lateral S-curve
            y_mm = -60.0 + i * seg_len * 1000.0  # cranial
            z_mm = 3.0 * math.sin(t * 0.5 * math.pi)  # slight AP offset
            solver.data.positions[0, i, 0] = x_mm / 1000.0
            solver.data.positions[0, i, 1] = y_mm / 1000.0
            solver.data.positions[0, i, 2] = z_mm / 1000.0

    # Sync updated positions to Warp arrays
    solver.data.sync_to_warp()


def _build_compositing_view(
    background: np.ndarray,
    pos_3d: np.ndarray,
    K: np.ndarray,
    Rt: np.ndarray,
    mu_profile: np.ndarray,
    radius_mm: float,
    noise_photon_count: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite catheter onto a single DRR view via Beer-Lambert.

    Returns:
        (composited_image, uv_projected_points)
    """
    uv = project_points_3d_to_2d(pos_3d, K, Rt)
    radii_px = compute_projected_radii(pos_3d, K, Rt, radius_mm)

    composited = composite_catheter_beer_lambert(
        background, uv, radii_px, mu_profile,
        scatter_sigma=18.0,
        scatter_fraction=0.03,
        noise_photon_count=noise_photon_count,
        detector_psf_sigma=0.7,
    )
    return composited, uv


def run_fluoroscopy_overlay(
    num_segments: int = 30,
    stiffness: float = 1e8,
    duration: float = 10.0,
    fluoro_image_path: str | None = None,
    save_video: str | None = None,
    dual_view: bool = False,
    initial_view: str = "AP",
    device: str = "cuda",
    image_size: tuple[int, int] = (512, 512),
    drr_dir: str = DRR_DIR,
    use_xpbd: bool = False,
    noise_photon_count: float = 2000.0,
):
    """Run the real-time fluoroscopy visualization with Beer-Lambert compositing."""
    import cv2

    # Detect whether a display is available for cv2.imshow
    has_display = os.environ.get("DISPLAY") is not None or os.environ.get("WAYLAND_DISPLAY") is not None
    if not has_display:
        print("  No display detected — running headless (snapshots + video)")
    else:
        WINDOW_NAME = "Fluoroscopy - Beer-Lambert Compositing"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, image_size[0], image_size[1])

    # ── Load DRR backgrounds ─────────────────────────────────────────
    print("Loading DRR backgrounds …")
    backgrounds = load_drr_backgrounds(image_size=image_size, drr_dir=drr_dir)

    if fluoro_image_path is not None:
        bg_custom = load_fluoroscopy_background(fluoro_image_path, image_size=image_size)
        backgrounds["AP"] = bg_custom

    # ── Build solver ─────────────────────────────────────────────────
    catheter_radius_mm = 1.0  # 6-French catheter (2mm OD)
    rod_length_mm = 120.0
    config = create_rod_config(
        num_segments=num_segments,
        length_mm=rod_length_mm,
        radius_mm=catheter_radius_mm,
        young_modulus=stiffness,
        device=device,
    )
    radius_mm = catheter_radius_mm

    if use_xpbd:
        xpbd_stiffness = min(stiffness, 1e5)
        xpbd_config = RodConfig(
            material=RodMaterialConfig(
                young_modulus=xpbd_stiffness,
                density=config.material.density,
                damping=config.material.damping,
                bend_stiffness=config.material.bend_stiffness,
                twist_stiffness=config.material.twist_stiffness,
            ),
            geometry=config.geometry,
            solver=RodSolverConfig(
                dt=config.solver.dt,
                num_substeps=config.solver.num_substeps,
                gravity=(0.0, 0.0, -9.81),
            ),
            device=device,
        )
        solver = XPBDRodSolver(xpbd_config, initial_height=0.0, floor_z=None)
        solver_name = f"XPBDRodSolver (E={xpbd_stiffness:.0e})"
    else:
        solver = RodSolver(config, num_envs=1)
        solver.data.fix_segment(0, 0)
        solver_name = "RodSolver (production)"

    # ── Per-segment attenuation profile ──────────────────────────────
    mu_profile = _segment_attenuation_profile(num_segments)

    # ── C-arm geometry ───────────────────────────────────────────────
    # pixel_spacing must match the DRR's physical field-of-view.
    # The DRR skull image spans ~250mm, so at SOD=600mm, SID=1000mm:
    #   FOV_at_object = image_width * pixel_spacing * SOD / SID
    #   250 = 512 * ps * 600/1000 → ps ≈ 0.81
    K = make_carm_intrinsics(sid=1000.0, pixel_spacing=0.81, image_size=image_size)

    # ── Video writer (optional) ──────────────────────────────────────
    writer = None
    if save_video:
        out_w = image_size[0] * 2 if dual_view else image_size[0]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, 30.0, (out_w, image_size[1]))

    # ── Sim loop ─────────────────────────────────────────────────────
    steps_per_frame = 2
    fps_clock = time.time()
    fps_val = 0.0
    frame_count = 0
    total_frames = int(duration * 30)

    SCALE = 1000.0  # m → mm

    current_view = initial_view if initial_view in DRR_VIEWS else "AP"
    current_angle = DRR_VIEWS[current_view][1]

    print("=" * 60)
    print("  FLUOROSCOPY — Beer-Lambert Catheter Compositing")
    print("=" * 60)
    print(f"  Solver:       {solver_name}")
    print(f"  DRR dir:      {drr_dir}")
    print(f"  Segments:     {num_segments}")
    print(f"  Stiffness:    {stiffness:.2e} Pa")
    print(f"  Compositing:  Beer-Lambert transmission")
    print(f"  Noise:        Poisson ({noise_photon_count:.0f} photons/px)")
    print(f"  Initial view: {current_view} ({current_angle}°)")
    print(f"  Dual view:    {dual_view}")
    print(f"  Duration:     {duration}s")
    print("=" * 60)
    print("  Keys:  [Q] quit   [SPACE] pause   [1] AP  [2] LAO30")
    print("         [3] Lateral  [4] RAO30  [+/-] angle ±5°")
    print("=" * 60)

    paused = False
    sim_time = 0.0
    dt = config.solver.dt

    while frame_count < total_frames:
        # --- Physics ---
        if not paused:
            for _ in range(steps_per_frame):
                if use_xpbd:
                    solver.step(dt)
                else:
                    solver.step()
            sim_time += steps_per_frame * dt

        # --- Extract 3D positions (m → mm) ---
        if use_xpbd:
            pos_3d = solver.positions.cpu().numpy() * SCALE
        else:
            pos_3d = solver.data.positions[0].cpu().numpy() * SCALE
        tip_pos = pos_3d[-1]

        # --- Primary view: Beer-Lambert compositing ---
        _view_name, bg_primary = select_drr_for_angle(current_angle, backgrounds)
        Rt_primary = make_carm_extrinsics(sod=600.0, carm_angle_deg=current_angle)

        frame_primary, _ = _build_compositing_view(
            bg_primary, pos_3d, K, Rt_primary, mu_profile, radius_mm, noise_photon_count,
        )

        # FPS
        now = time.time()
        elapsed = now - fps_clock
        if elapsed > 0.5:
            fps_val = frame_count / max(elapsed, 1e-6)

        frame_primary = draw_hud(frame_primary, sim_time, tip_pos, fps_val, current_angle)

        cv2.putText(frame_primary, _view_name, (image_size[0] - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        if dual_view:
            lat_angle = current_angle + 90.0
            _lat_name, bg_lat = select_drr_for_angle(lat_angle, backgrounds)
            Rt_lat = make_carm_extrinsics(sod=600.0, carm_angle_deg=lat_angle)

            frame_lat, _ = _build_compositing_view(
                bg_lat, pos_3d, K, Rt_lat, mu_profile, radius_mm, noise_photon_count,
            )
            frame_lat = draw_hud(frame_lat, sim_time, tip_pos, fps_val, lat_angle)
            cv2.putText(frame_lat, _lat_name, (image_size[0] - 100, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            canvas = np.hstack([frame_primary, frame_lat])
        else:
            canvas = frame_primary

        if writer is not None:
            writer.write(canvas)

        # Save snapshot at specific frames
        if frame_count in (0, total_frames // 4, total_frames // 2, total_frames - 1):
            snap_path = f"/tmp/fluoro_frame_{frame_count:04d}.png"
            cv2.imwrite(snap_path, canvas)
            if frame_count == 0:
                print(f"  Snapshots → /tmp/fluoro_frame_*.png")

        # --- Display (skip if no GUI available) ---
        if has_display:
            try:
                cv2.imshow(WINDOW_NAME, canvas)
            except cv2.error:
                has_display = False
                print("  Display unavailable, running headless (snapshots + video only)")

        if has_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("1"):
                current_view, current_angle = "AP", 0.0
            elif key == ord("2"):
                current_view, current_angle = "LAO_30", 30.0
            elif key == ord("3"):
                current_view, current_angle = "Lateral", 90.0
            elif key == ord("4"):
                current_view, current_angle = "RAO_30", -30.0
            elif key in (ord("+"), ord("=")):
                current_angle += 5.0
            elif key in (ord("-"), ord("_")):
                current_angle -= 5.0

        if not paused:
            frame_count += 1

        if frame_count % 30 == 0:
            print(f"  frame {frame_count}/{total_frames}  tip=({tip_pos[0]:.1f}, {tip_pos[1]:.1f}, {tip_pos[2]:.1f}) mm")

    if writer is not None:
        writer.release()
        print(f"Video saved → {save_video}")
    if has_display:
        cv2.destroyAllWindows()
    print("Done.")


# ═══════════════════════════════════════════════════════════════════════
# Matplotlib fallback (if OpenCV is unavailable)
# ═══════════════════════════════════════════════════════════════════════

def run_matplotlib_fluoroscopy(
    num_segments: int = 30,
    stiffness: float = 1e8,
    duration: float = 5.0,
    device: str = "cuda",
    image_size: tuple[int, int] = (512, 512),
    drr_dir: str = DRR_DIR,
):
    """Fallback matplotlib-based fluoroscopy overlay (no OpenCV required).

    Loads the AP DRR image as background if available.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    config = create_rod_config(
        num_segments=num_segments,
        length_mm=200.0,
        radius_mm=0.45,
        young_modulus=stiffness,
        device=device,
    )
    solver = RodSolver(config, num_envs=1)
    solver.data.fix_segment(0, 0)
    _position_rod_in_anatomy(solver, device)

    # C-arm
    K = make_carm_intrinsics(sid=1000.0, pixel_spacing=0.3, image_size=image_size)
    Rt = make_carm_extrinsics(sod=600.0, carm_angle_deg=0.0)
    SCALE = 1000.0

    # Try loading AP DRR, fall back to synthetic
    ap_path = os.path.join(drr_dir, DRR_VIEWS["AP"][0])
    if os.path.isfile(ap_path):
        import cv2
        bg_bgr = cv2.imread(ap_path)
        bg_bgr = cv2.resize(bg_bgr, image_size)
        bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
        print(f"Using DRR background: {ap_path}")
    else:
        bg = _synthesise_fluoroscopy(image_size)[:, :, 0]
        print("Using synthetic background (AP DRR not found)")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Fluoroscopy Overlay — Rod Solver + DiffDRR", fontsize=14, fontweight="bold", color="lime")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.imshow(bg, cmap="gray", vmin=0, vmax=255, extent=[0, image_size[0], image_size[1], 0])
    (rod_line,) = ax.plot([], [], "-", color="#ddddff", linewidth=2)
    (seg_dots,) = ax.plot([], [], "o", color="#ddddff", markersize=2)
    (tip_dot,) = ax.plot([], [], "o", color="yellow", markersize=6)
    (fix_dot,) = ax.plot([], [], "s", color="red", markersize=6)
    time_text = ax.text(
        10, 20, "", fontsize=10, color="lime",
        fontfamily="monospace",
    )

    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)
    ax.set_aspect("equal")
    ax.axis("off")

    steps_per_frame = 2
    total_frames = int(duration * 30)

    def animate(frame):
        for _ in range(steps_per_frame):
            solver.step()
        pos_3d = solver.data.positions[0].cpu().numpy() * SCALE
        uv = project_points_3d_to_2d(pos_3d, K, Rt)

        rod_line.set_data(uv[:, 0], uv[:, 1])
        seg_dots.set_data(uv[:, 0], uv[:, 1])
        tip_dot.set_data([uv[-1, 0]], [uv[-1, 1]])
        fix_dot.set_data([uv[0, 0]], [uv[0, 1]])
        time_text.set_text(f"t={solver.time:.2f}s  tip=({pos_3d[-1, 0]:.0f},{pos_3d[-1, 1]:.0f},{pos_3d[-1, 2]:.0f})mm")
        return rod_line, seg_dots, tip_dot, fix_dot, time_text

    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=33, blit=True)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Slang GPU unified sim loop (fused DRR + catheter Beer-Lambert)
# ═══════════════════════════════════════════════════════════════════════

MU_VOLUME_PATH = "/home/cdinea/Downloads/new_CTdata/processed/ct_mu_zyx.npy"
MU_SPACING_ZYX = (0.625, 0.429688, 0.429688)


def run_slang_unified_loop(
    num_segments: int = 30,
    stiffness: float = 1e8,
    duration: float = 10.0,
    save_video: str | None = None,
    device: str = "cuda",
    image_size: tuple[int, int] = (512, 512),
    noise_photon_count: float = 2000.0,
):
    """Unified sim loop: XPBD physics + Slang GPU Beer-Lambert (volume + catheter).

    The Slang shader ray-marches through the CT mu-volume AND the catheter
    geometry in a single fused pass, producing depth-correct compositing
    entirely on GPU with zero CPU compositing.
    """
    import cv2

    has_display = os.environ.get("DISPLAY") is not None or os.environ.get("WAYLAND_DISPLAY") is not None

    # ── Load CT mu-volume ──────────────────────────────────────────────
    print("Loading CT mu-volume …")
    mu_vol = np.load(MU_VOLUME_PATH).astype(np.float32)
    print(f"  Volume shape: {mu_vol.shape}  spacing: {MU_SPACING_ZYX}")

    # ── Init Slang renderer ────────────────────────────────────────────
    import sys
    sys.path.insert(0, "/home/cdinea/i4h-sensor-simulation-internal/fluoro-simulator")
    from fluorosim.rendering.diffdrr_slang_renderer import (
        CatheterSegmentData,
        SlangDiffDRRConfig,
        SlangDiffDRRRenderer,
    )

    cfg = SlangDiffDRRConfig(
        det_height_px=image_size[1],
        det_width_px=image_size[0],
        pixel_spacing_mm=0.81,
        source_to_detector_mm=1000.0,
        source_to_isocenter_mm=600.0,
        step_mm=0.5,
        i0=1.0,
        normalize=True,
        invert=True,
    )
    renderer = SlangDiffDRRRenderer(mu_vol, MU_SPACING_ZYX, cfg)

    # Warmup
    print("  GPU warmup (2 frames) …")
    _ = renderer.render(rotation=(0, 0, 0), translation=(0, 0, 0))
    _ = renderer.render(rotation=(0, 0, 0), translation=(0, 0, 0))

    # ── Build XPBD solver ──────────────────────────────────────────────
    catheter_radius_mm = 1.0
    rod_length_mm = 120.0
    xpbd_stiffness = min(stiffness, 1e5)

    xpbd_config = RodConfig(
        material=RodMaterialConfig(
            young_modulus=xpbd_stiffness,
            density=7800.0,
            damping=0.05,
        ),
        geometry=RodGeometryConfig(
            num_segments=num_segments,
            rest_length=rod_length_mm / 1000.0,
            radius=catheter_radius_mm / 1000.0,
        ),
        solver=RodSolverConfig(
            dt=1.0 / 60.0,
            num_substeps=2,
            gravity=(0.0, -9.81, 0.0),
        ),
        device=device,
    )
    solver = XPBDRodSolver(xpbd_config, initial_height=0.0, floor_z=None)
    dt = xpbd_config.solver.dt

    # Reposition rod inside the volume: grow along +Y with a gentle S-curve.
    # Write directly to the Warp workspace arrays (solver.positions is a clone).
    import warp as _wp
    seg_len_m = rod_length_mm / 1000.0 / num_segments
    n_pts = solver.num_points
    new_pos = np.zeros((n_pts, 3), dtype=np.float32)
    for i in range(n_pts):
        t = i / max(n_pts - 1, 1)
        new_pos[i, 0] = 0.008 * math.sin(t * math.pi)
        new_pos[i, 1] = -0.060 + i * seg_len_m
        new_pos[i, 2] = 0.003 * math.sin(t * 0.5 * math.pi)

    # Orientations: align local Z → +Y (rotation of -90° around X)
    half = math.pi * 0.25
    q_y = np.array([-math.sin(half), 0.0, 0.0, math.cos(half)], dtype=np.float32)
    new_ori = np.tile(q_y, (n_pts, 1))

    # Assign to workspace (positions, predicted_positions, orientations, etc.)
    solver._ws.positions.assign(_wp.array(new_pos, dtype=_wp.vec3, device=device))
    solver._ws.predicted_positions.assign(_wp.array(new_pos, dtype=_wp.vec3, device=device))
    solver._ws.orientations.assign(_wp.array(new_ori, dtype=_wp.quat, device=device))
    solver._ws.predicted_orientations.assign(_wp.array(new_ori, dtype=_wp.quat, device=device))
    solver._ws.prev_orientations.assign(_wp.array(new_ori, dtype=_wp.quat, device=device))

    # Zero velocities so the first step doesn't carry residual momentum
    solver._ws.velocities.zero_()
    solver._ws.angular_velocities.zero_()

    # ── Attenuation profile ────────────────────────────────────────────
    mu_profile = _segment_attenuation_profile(num_segments)

    # ── Video writer ───────────────────────────────────────────────────
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, 30.0, image_size)

    if has_display:
        WINDOW_NAME = "Slang Unified — Beer-Lambert (Volume + Catheter)"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, image_size[0], image_size[1])

    # ── Sim loop ───────────────────────────────────────────────────────
    steps_per_frame = 2
    total_frames = int(duration * 30)
    frame_count = 0
    sim_time = 0.0
    fps_clock = time.time()
    fps_val = 0.0
    carm_angle = 0.0
    SCALE = 1000.0

    # Volume centre in mm (for catheter positioning)
    vol_shape = np.array(mu_vol.shape)  # ZYX
    vol_spacing = np.array(MU_SPACING_ZYX)
    vol_size_mm = vol_shape * vol_spacing  # ZYX
    vol_center_mm_xyz = np.array([
        vol_size_mm[2] / 2.0,  # X
        vol_size_mm[1] / 2.0,  # Y
        vol_size_mm[0] / 2.0,  # Z
    ])

    print("=" * 60)
    print("  SLANG UNIFIED SIM LOOP")
    print("  Beer-Lambert: Volume + Catheter in single GPU ray march")
    print("=" * 60)
    print(f"  Solver:     XPBDRodSolver (E={xpbd_stiffness:.0e})")
    print(f"  Volume:     {mu_vol.shape}")
    print(f"  Detector:   {image_size[0]}×{image_size[1]}")
    print(f"  Vol centre: {vol_center_mm_xyz}")
    print(f"  Duration:   {duration}s  ({total_frames} frames)")
    print("=" * 60)

    while frame_count < total_frames:
        # --- Physics ---
        for _ in range(steps_per_frame):
            solver.step(dt)
        sim_time += steps_per_frame * dt

        # --- Extract positions (m → mm) & place in volume frame ---
        pos_3d = solver.positions.cpu().numpy() * SCALE  # (N, 3) mm

        # Offset catheter so it's roughly in the volume centre
        pos_world = pos_3d + vol_center_mm_xyz[np.newaxis, :]

        # --- Build catheter segments for Slang ---
        catheter = CatheterSegmentData(
            positions=pos_world.astype(np.float32),
            radii=catheter_radius_mm,
            mu_values=mu_profile,
        )

        # --- Render via Slang: fused volume + catheter Beer-Lambert ---
        rotation = (0.0, math.radians(carm_angle), 0.0)
        image = renderer.render_with_catheter(
            rotation=rotation,
            translation=(0.0, 0.0, 0.0),
            catheter=catheter,
        )

        # Convert to BGR uint8 for display/save
        img_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        canvas = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

        # FPS
        now = time.time()
        elapsed = now - fps_clock
        if elapsed > 0.5:
            fps_val = frame_count / max(elapsed, 1e-6)

        tip_pos = pos_3d[-1]
        canvas = draw_hud(canvas, sim_time, tip_pos, fps_val, carm_angle)
        cv2.putText(canvas, "SLANG GPU", (image_size[0] - 130, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if writer is not None:
            writer.write(canvas)

        if frame_count in (0, total_frames // 4, total_frames // 2, total_frames - 1):
            snap_path = f"/tmp/slang_fluoro_{frame_count:04d}.png"
            cv2.imwrite(snap_path, canvas)
            if frame_count == 0:
                print(f"  Snapshots → /tmp/slang_fluoro_*.png")

        if has_display:
            try:
                cv2.imshow(WINDOW_NAME, canvas)
            except cv2.error:
                has_display = False

        if has_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")):
                carm_angle += 5.0
            elif key in (ord("-"), ord("_")):
                carm_angle -= 5.0

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  frame {frame_count}/{total_frames}  fps={fps_val:.1f}  tip=({tip_pos[0]:.1f}, {tip_pos[1]:.1f}, {tip_pos[2]:.1f}) mm")

    if writer is not None:
        writer.release()
        print(f"Video saved → {save_video}")
    if has_display:
        cv2.destroyAllWindows()
    print("Done.")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fluoroscopy visualization with Beer-Lambert catheter compositing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
DRR images are loaded from:
  %(drr_dir)s

Available views: AP, LAO_30, Lateral, RAO_30
Interactive keys: [Q] quit  [SPACE] pause  [1] AP  [2] LAO30  [3] LAT  [4] RAO30  [+/-] angle
""" % {"drr_dir": DRR_DIR},
    )
    parser.add_argument("--num-segments", type=int, default=30)
    parser.add_argument("--stiffness", type=float, default=1e8, help="Young's modulus (Pa)")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (s)")
    parser.add_argument("--fluoro-image", type=str, default=None,
                        help="Override AP background with this image (PNG/JPEG/DICOM)")
    parser.add_argument("--drr-dir", type=str, default=DRR_DIR,
                        help="Directory containing DiffDRR output images")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Save output to MP4 video file")
    parser.add_argument("--ap-and-lateral", action="store_true", dest="dual_view",
                        help="Show AP + lateral dual-view side-by-side")
    parser.add_argument("--view", type=str, default="AP",
                        choices=["AP", "LAO_30", "Lateral", "RAO_30"],
                        help="Initial view (default: AP)")
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--use-matplotlib", action="store_true",
                        help="Use matplotlib fallback (no OpenCV required)")
    parser.add_argument("--use-xpbd", action="store_true",
                        help="Use self-contained XPBD rod solver (block-Thomas direct solve)")
    parser.add_argument("--use-slang", action="store_true",
                        help="Use Slang GPU unified loop: fused DRR + catheter Beer-Lambert")
    parser.add_argument("--noise", type=float, default=2000.0,
                        help="Poisson noise photon count (0 to disable)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    image_size = (args.image_width, args.image_height)

    if args.use_slang:
        run_slang_unified_loop(
            num_segments=args.num_segments,
            stiffness=args.stiffness,
            duration=args.duration,
            save_video=args.save_video,
            device=device,
            image_size=image_size,
            noise_photon_count=args.noise,
        )
        return

    if args.use_matplotlib:
        run_matplotlib_fluoroscopy(
            num_segments=args.num_segments,
            stiffness=args.stiffness,
            duration=args.duration,
            device=device,
            image_size=image_size,
            drr_dir=args.drr_dir,
        )
    else:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("OpenCV not found, falling back to matplotlib. Install with: pip install opencv-python")
            run_matplotlib_fluoroscopy(
                num_segments=args.num_segments,
                stiffness=args.stiffness,
                duration=args.duration,
                device=device,
                image_size=image_size,
                drr_dir=args.drr_dir,
            )
            return

        run_fluoroscopy_overlay(
            num_segments=args.num_segments,
            stiffness=args.stiffness,
            duration=args.duration,
            fluoro_image_path=args.fluoro_image,
            save_video=args.save_video,
            dual_view=args.dual_view,
            initial_view=args.view,
            device=device,
            image_size=image_size,
            drr_dir=args.drr_dir,
            use_xpbd=args.use_xpbd,
            noise_photon_count=args.noise,
        )


if __name__ == "__main__":
    main()

