# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate that the Newton warp renderer applies the OpenCV lens-distortion camera model.

A ground plane is rendered through a camera carrying an OpenCV pinhole (or fisheye) calibration. The
Newton renderer inverts the OpenCV forward model per output pixel to trace the distorted camera-space
rays, so a per-pixel ray-hit distance (``distance_to_camera``) map warps under the distortion. The
distance map is used as the comparison signal because it is purely geometric: it does not depend on
scene textures (which Newton skips without Kit), so the distortion effect is visible across the whole
frame rather than only on sparse textured features.

With the coefficients applied vs. muted (``apply_lens_distortion=False``) the same calibrated camera
produces meaningfully different distance maps; the OpenCV fisheye projection likewise differs from an
undistorted pinhole.

Notes:
  * Runs against the Newton warp renderer (no Kit/Isaac Sim, no OVRTX). It requires ``newton`` and a
    CUDA GPU; it skips cleanly otherwise.
  * Uses Newton physics (``NewtonCfg`` + ``MJWarpSolverCfg``) so the scene is built through the
    Newton model the warp renderer traces.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

_REQUIRED_MODULES = ("isaaclab_newton", "newton", "warp", "torch")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]


def _cuda_available() -> bool:
    """Whether a CUDA device is available for the Newton warp renderer."""
    if _MISSING_MODULES:
        return False
    import torch

    return torch.cuda.is_available()


_SKIP_NO_NEWTON = pytest.mark.skipif(
    bool(_MISSING_MODULES),
    reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
)
_SKIP_NO_CUDA = pytest.mark.skipif(
    not _cuda_available(),
    reason="requires a CUDA GPU for the Newton warp renderer",
)

if not _MISSING_MODULES:
    import torch
    from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
    from isaaclab.sim.spawners.sensors.sensors_cfg import (
        OpenCvDistortionCfg,
        OpenCvFisheyeDistortionCfg,
        OpenCvPinholeDistortionCfg,
        PinholeCameraCfg,
    )
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

SIM_DT = 1.0 / 60.0
WIDTH, HEIGHT = 640, 480
WARMUP_STEPS = 4

# OpenCV calibration with non-square focal lengths and an off-center principal point.
_CALIB = dict(fx=339.26592887, fy=338.82010626, cx=323.55809091, cy=250.27360914)
# The radial map r_d = r_u * (1 + k1 * r_u**2) is globally monotonic because
# its derivative is 1 + 3 * k1 * r_u**2 > 0.
_PINHOLE_K1 = 0.1
# OpenCV fisheye (equidistant) coefficients; the base fisheye projection alone differs strongly from pinhole
_FISHEYE_COEFFS = dict(k1=0.1, k2=-0.05, k3=0.0, k4=0.0)

_CAM_EYE = (0.0, 0.0, 2.5)
_CAM_TARGET = (1.75, 0.0, 0.0)


if not _MISSING_MODULES:

    @configclass
    class _DistortionSceneCfg(InteractiveSceneCfg):
        """A ground plane, calibrated camera, and off-screen anchor body for Newton."""

        ground = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
        )
        anchor = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Anchor",
            spawn=sim_utils.CuboidCfg(
                size=(0.01, 0.01, 0.01),
                rigid_props=sim_utils.RigidBodyBaseCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
                collision_props=sim_utils.CollisionBaseCfg(),
                physics_material=RigidBodyMaterialBaseCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
        )
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=HEIGHT,
            width=WIDTH,
            data_types=["distance_to_camera"],
            spawn=PinholeCameraCfg(focal_length=13.6, clipping_range=(0.001, 20.0)),
            renderer_cfg=NewtonWarpRendererCfg(),
        )


def _pinhole_distortion(apply_lens_distortion: bool) -> OpenCvPinholeDistortionCfg:
    """Pinhole OpenCV calibration with a globally invertible synthetic radial coefficient."""
    return OpenCvPinholeDistortionCfg(
        image_size=(WIDTH, HEIGHT),
        apply_lens_distortion=apply_lens_distortion,
        k1=_PINHOLE_K1,
        **_CALIB,
    )


def _fisheye_distortion(apply_lens_distortion: bool) -> OpenCvFisheyeDistortionCfg:
    """Fisheye OpenCV calibration reusing the SO-101 intrinsics with fisheye coefficients."""
    return OpenCvFisheyeDistortionCfg(
        image_size=(WIDTH, HEIGHT),
        apply_lens_distortion=apply_lens_distortion,
        **_CALIB,
        **_FISHEYE_COEFFS,
    )


def _expected_pinhole_ground_distance(px: int, py: int) -> float:
    """Compute the expected distorted-ray distance to the ground plane [m]."""
    u = px + 0.5
    v = py + 0.5
    x_d = (u - _CALIB["cx"]) / _CALIB["fx"]
    y_d = (v - _CALIB["cy"]) / _CALIB["fy"]
    radius_d = float(np.hypot(x_d, y_d))

    if radius_d > 0.0:
        lower, upper = 0.0, radius_d
        for _ in range(64):
            radius_u = 0.5 * (lower + upper)
            if radius_u * (1.0 + _PINHOLE_K1 * radius_u**2) < radius_d:
                lower = radius_u
            else:
                upper = radius_u
        scale = (0.5 * (lower + upper)) / radius_d
        x_u, y_u = x_d * scale, y_d * scale
    else:
        x_u, y_u = 0.0, 0.0

    ray_camera = np.array((x_u, -y_u, -1.0))
    ray_camera /= np.linalg.norm(ray_camera)

    eye = np.asarray(_CAM_EYE)
    forward = np.asarray(_CAM_TARGET) - eye
    z_axis = -forward / np.linalg.norm(forward)
    x_axis = np.cross(np.array((0.0, 0.0, 1.0)), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    ray_world = np.column_stack((x_axis, y_axis, z_axis)) @ ray_camera
    return float(-eye[2] / ray_world[2])


def _render_distance(distortion: OpenCvDistortionCfg, device: str) -> np.ndarray:
    """Render the ground-plane distance map through an OpenCV-calibrated Newton camera.

    ``distance_to_camera`` (per-pixel ray-hit distance [m]) is used instead of ``rgb`` because it is
    purely geometric and does not depend on scene textures, which Newton skips without Kit.
    """
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(
        SimulationCfg(dt=SIM_DT, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1), device=device)
    )
    rot = tuple(
        quat_from_matrix(
            create_rotation_matrix_from_view(torch.tensor([_CAM_EYE]), torch.tensor([_CAM_TARGET]), up_axis="Z")
        )[0].tolist()
    )
    scene_cfg = _DistortionSceneCfg(num_envs=1, env_spacing=20.0)
    scene_cfg.camera.offset = CameraCfg.OffsetCfg(pos=_CAM_EYE, rot=rot, convention="opengl")
    scene_cfg.camera.spawn.distortion = distortion
    scene = InteractiveScene(scene_cfg)
    camera: Camera = scene["camera"]
    try:
        sim.reset()
        for _ in range(WARMUP_STEPS):
            sim.step()
            camera.update(SIM_DT, force_recompute=True)
        distance = camera.data.output["distance_to_camera"].torch[0].detach().cpu().float().numpy().copy()
        return distance
    finally:
        del camera
        del scene
        sim.stop()
        sim.clear_instance()


def _mean_abs_distance_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute per-pixel distance difference [m] over pixels that hit geometry in both maps."""
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
    assert valid.mean() > 0.5, "too few valid distance samples to compare"
    return float(np.abs(a[valid] - b[valid]).mean())


@pytest.mark.parametrize("device", ["cuda:0"])
@_SKIP_NO_NEWTON
@_SKIP_NO_CUDA
def test_opencv_distortion_changes_newton_render(device):
    """The Newton renderer must render the distorted and zero-coefficient cameras meaningfully differently."""
    distorted = _render_distance(_pinhole_distortion(True), device=device)
    reference = _render_distance(_pinhole_distortion(False), device=device)

    assert distorted.shape == (HEIGHT, WIDTH, 1)
    assert np.isfinite(distorted).mean() > 0.9
    assert np.isfinite(reference).mean() > 0.9
    mean_abs_diff = _mean_abs_distance_diff(distorted, reference)
    assert mean_abs_diff > 0.01, f"distorted vs reference distance maps differ by only {mean_abs_diff:.4f} m"
    for px, py in ((0, 0), (WIDTH // 2, HEIGHT // 2), (WIDTH - 1, HEIGHT - 1)):
        assert distorted[py, px, 0] == pytest.approx(_expected_pinhole_ground_distance(px, py), abs=2e-3)


@pytest.mark.parametrize("device", ["cuda:0"])
@_SKIP_NO_NEWTON
@_SKIP_NO_CUDA
def test_opencv_fisheye_distortion_renders_through_newton(device):
    """The Newton renderer honors the OpenCV fisheye model: its render differs meaningfully from the pinhole.

    The same calibrated camera is rendered under the OpenCV fisheye model and under an undistorted
    pinhole. The fisheye equidistant projection bends the rays, so the two distance maps must differ
    well beyond render noise.
    """
    fisheye = _render_distance(_fisheye_distortion(True), device=device)
    pinhole = _render_distance(_pinhole_distortion(False), device=device)

    assert fisheye.shape == (HEIGHT, WIDTH, 1)
    assert np.isfinite(fisheye).mean() > 0.9
    assert np.isfinite(pinhole).mean() > 0.9
    mean_abs_diff = _mean_abs_distance_diff(fisheye, pinhole)
    assert mean_abs_diff > 0.05, f"fisheye vs pinhole distance maps differ by only {mean_abs_diff:.4f} m"
