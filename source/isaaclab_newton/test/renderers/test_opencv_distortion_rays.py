# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less, GPU-less tests for the OpenCV pinhole distortion ray-generation kernel.

These exercise the Warp kernel in
:mod:`isaaclab_newton.renderers.opencv_distortion_rays` on the warp CPU device, without ``newton``,
a renderer or a GPU. The kernel inverts the OpenCV forward distortion model to recover the
camera-space ray for each output pixel. Correctness is checked by re-applying the OpenCV *forward*
model (computed here in NumPy) to the recovered undistorted point and confirming it lands back on the
originating pixel (round-trip), which is the property the fixed-point inversion must satisfy.

OpenCV fisheye rays use Newton's native ``compute_camera_rays_fisheye_opencv`` helper and are covered
by the Newton camera integration test.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_REQUIRED_MODULES = ("warp",)
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    import warp as wp
    from isaaclab_newton.renderers.opencv_distortion_rays import compute_camera_rays_opencv_pinhole

WIDTH, HEIGHT = 64, 48
_CALIB = dict(fx=339.26592887, fy=338.82010626, cx=323.55809091, cy=250.27360914)
# calibrated image the intrinsics refer to (the render grid is smaller and rescaled onto it)
_IMAGE_W, _IMAGE_H = 640, 480
_PINHOLE_COEFFS = dict(
    k1=0.1,
    k2=-0.05,
    k3=0.01,
    k4=0.005,
    k5=-0.002,
    k6=0.0005,
    p1=0.001,
    p2=-0.002,
    s1=0.0005,
    s2=-0.0002,
    s3=0.0003,
    s4=-0.0001,
)


def _launch_pinhole(coeffs: dict) -> np.ndarray:
    """Launch the pinhole kernel on the warp CPU device and return the ray field as NumPy."""
    rays = wp.empty((1, HEIGHT, WIDTH, 2), dtype=wp.vec3f, device="cpu")
    wp.launch(
        compute_camera_rays_opencv_pinhole,
        dim=(1, HEIGHT, WIDTH),
        inputs=[
            WIDTH,
            HEIGHT,
            _CALIB["fx"],
            _CALIB["fy"],
            _CALIB["cx"],
            _CALIB["cy"],
            float(_IMAGE_W),
            float(_IMAGE_H),
            coeffs["k1"],
            coeffs["k2"],
            coeffs["k3"],
            coeffs["k4"],
            coeffs["k5"],
            coeffs["k6"],
            coeffs["p1"],
            coeffs["p2"],
            coeffs["s1"],
            coeffs["s2"],
            coeffs["s3"],
            coeffs["s4"],
        ],
        outputs=[rays],
        device="cpu",
    )
    return rays.numpy()


def _pixel_distorted_normalized(px: int, py: int) -> tuple[float, float]:
    """The OpenCV *distorted* normalized coordinates the kernel derives from a render pixel."""
    u = ((px + 0.5) / WIDTH) * _IMAGE_W
    v = ((py + 0.5) / HEIGHT) * _IMAGE_H
    x_d = (u - _CALIB["cx"]) / _CALIB["fx"]
    y_d = (v - _CALIB["cy"]) / _CALIB["fy"]
    return x_d, y_d


def _forward_pinhole(x: float, y: float, c: dict) -> tuple[float, float]:
    """OpenCV pinhole forward model: undistorted normalized ``(x, y)`` -> distorted normalized."""
    r2 = x * x + y * y
    r4, r6 = r2 * r2, r2 * r2 * r2
    radial = (1.0 + c["k1"] * r2 + c["k2"] * r4 + c["k3"] * r6) / (1.0 + c["k4"] * r2 + c["k5"] * r4 + c["k6"] * r6)
    x_d = x * radial + 2.0 * c["p1"] * x * y + c["p2"] * (r2 + 2.0 * x * x) + c["s1"] * r2 + c["s2"] * r4
    y_d = y * radial + c["p1"] * (r2 + 2.0 * y * y) + 2.0 * c["p2"] * x * y + c["s3"] * r2 + c["s4"] * r4
    return x_d, y_d


def _ray_to_opencv_normalized(direction: np.ndarray) -> tuple[float, float]:
    """Map a Newton OpenGL camera-space ray back to OpenCV undistorted normalized ``(x_u, y_u)``.

    The kernel emits ``normalize(vec3(x_u, -y_u, -1))``; undo the normalization and the ``y``/``z``
    sign flip that maps OpenCV camera space onto Newton's OpenGL camera space.
    """
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
    # dz corresponds to -1 before normalization, so scale by -1/dz to recover the z == 1 plane.
    x_u = dx / (-dz)
    y_u = -dy / (-dz)
    return x_u, y_u


def test_pinhole_ray_origins_are_zero_and_directions_unit():
    """Every ray has a zero origin and a unit-length direction."""
    rays = _launch_pinhole(_PINHOLE_COEFFS)
    origins = rays[..., 0, :]
    directions = rays[..., 1, :]
    assert np.allclose(origins, 0.0)
    norms = np.linalg.norm(directions, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # all rays look down -Z in Newton's OpenGL camera space
    assert np.all(directions[..., 2] < 0.0)


def test_pinhole_inversion_round_trips_to_pixel():
    """Re-applying the OpenCV forward model to the recovered ray lands back on each pixel's distorted point."""
    rays = _launch_pinhole(_PINHOLE_COEFFS)
    directions = rays[0, :, :, 1, :]
    max_err = 0.0
    for py in range(0, HEIGHT, 7):
        for px in range(0, WIDTH, 9):
            x_u, y_u = _ray_to_opencv_normalized(directions[py, px])
            x_d_fwd, y_d_fwd = _forward_pinhole(x_u, y_u, _PINHOLE_COEFFS)
            x_d, y_d = _pixel_distorted_normalized(px, py)
            max_err = max(max_err, abs(x_d_fwd - x_d), abs(y_d_fwd - y_d))
    assert max_err < 1e-5, f"pinhole inversion round-trip error {max_err:.2e} too large"


def test_pinhole_zero_coeffs_matches_ideal_projection():
    """With zero coefficients the recovered ray is the plain pinhole ray through the pixel."""
    zero = {k: 0.0 for k in _PINHOLE_COEFFS}
    rays = _launch_pinhole(zero)
    directions = rays[0, :, :, 1, :]
    for py in range(0, HEIGHT, 11):
        for px in range(0, WIDTH, 13):
            x_u, y_u = _ray_to_opencv_normalized(directions[py, px])
            x_d, y_d = _pixel_distorted_normalized(px, py)
            assert x_u == pytest.approx(x_d, abs=1e-5)
            assert y_u == pytest.approx(y_d, abs=1e-5)
