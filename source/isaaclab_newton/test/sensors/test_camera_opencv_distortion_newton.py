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
undistorted pinhole. The reconstructed ``intrinsic_matrices`` are also checked end-to-end against the
authored, non-square, off-center calibration.

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
    from isaaclab.sim.spawners.sensors.sensors_cfg import (
        OpenCvDistortionCfg,
        OpenCvFisheyeDistortionCfg,
        OpenCvPinholeDistortionCfg,
        PinholeCameraCfg,
    )
    from isaaclab.utils.configclass import configclass
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

SIM_DT = 1.0 / 60.0
WIDTH, HEIGHT = 640, 480
WARMUP_STEPS = 4

# Example real-world OpenCV pinhole calibration (fx != fy, off-center principal point).
_CALIB = dict(fx=339.26592887, fy=338.82010626, cx=323.55809091, cy=250.27360914)
_COEFFS = dict(k1=0.07702322, k2=-0.13605453, k3=0.05163219, p1=-0.00024938, p2=-0.00175006)
# scale the (mild) real coefficients so the barrel effect is unambiguous in the assertion
_K_SCALE = 15.0
# OpenCV fisheye (equidistant) coefficients; the base fisheye projection alone differs strongly from pinhole
_FISHEYE_COEFFS = dict(k1=0.1, k2=-0.05, k3=0.0, k4=0.0)

_CAM_EYE = (0.0, 0.0, 2.5)
_CAM_TARGET = (1.75, 0.0, 0.0)


if not _MISSING_MODULES:

    @configclass
    class _DistortionSceneCfg(InteractiveSceneCfg):
        """The grid-textured ground plane, a dome light and an off-screen anchor body for Newton."""

        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
        )
        anchor = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Anchor",
            spawn=sim_utils.CuboidCfg(
                size=(0.01, 0.01, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
        )


def _pinhole_distortion(apply_lens_distortion: bool) -> OpenCvPinholeDistortionCfg:
    """Pinhole OpenCV calibration with the (scaled) SO-101 radial/tangential coefficients."""
    return OpenCvPinholeDistortionCfg(
        image_size=(WIDTH, HEIGHT),
        apply_lens_distortion=apply_lens_distortion,
        **_CALIB,
        **{name: value * _K_SCALE for name, value in _COEFFS.items()},
    )


def _fisheye_distortion(apply_lens_distortion: bool) -> OpenCvFisheyeDistortionCfg:
    """Fisheye OpenCV calibration reusing the SO-101 intrinsics with fisheye coefficients."""
    return OpenCvFisheyeDistortionCfg(
        image_size=(WIDTH, HEIGHT),
        apply_lens_distortion=apply_lens_distortion,
        **_CALIB,
        **_FISHEYE_COEFFS,
    )


def _render_distance(distortion: OpenCvDistortionCfg, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Render the ground-plane distance map through an OpenCV-calibrated Newton camera; return ``(dist, K)``.

    ``distance_to_camera`` (per-pixel ray-hit distance [m]) is used instead of ``rgb`` because it is
    purely geometric and does not depend on scene textures, which Newton skips without Kit.
    """
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(
        SimulationCfg(dt=SIM_DT, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1), device=device)
    )
    scene = InteractiveScene(_DistortionSceneCfg(num_envs=1, env_spacing=20.0))

    rot = tuple(
        quat_from_matrix(
            create_rotation_matrix_from_view(torch.tensor([_CAM_EYE]), torch.tensor([_CAM_TARGET]), up_axis="Z")
        )[0].tolist()
    )
    camera = Camera(
        CameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            update_period=0.0,
            height=HEIGHT,
            width=WIDTH,
            data_types=["distance_to_camera"],
            offset=CameraCfg.OffsetCfg(pos=_CAM_EYE, rot=rot, convention="opengl"),
            spawn=PinholeCameraCfg(focal_length=13.6, clipping_range=(0.001, 20.0), distortion=distortion),
            renderer_cfg=NewtonWarpRendererCfg(),
        )
    )
    try:
        sim.reset()
        for _ in range(WARMUP_STEPS):
            sim.step()
            camera.update(SIM_DT, force_recompute=True)
        distance = camera.data.output["distance_to_camera"].torch[0].detach().cpu().float().numpy().copy()
        intrinsics = camera.data.intrinsic_matrices.torch[0].detach().cpu().numpy()
        return distance, intrinsics
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
    distorted, _ = _render_distance(_pinhole_distortion(True), device=device)
    reference, _ = _render_distance(_pinhole_distortion(False), device=device)

    assert distorted.shape == (HEIGHT, WIDTH, 1)
    # both frames render geometry (the ground plane fills the frame)
    assert np.isfinite(distorted).mean() > 0.9
    assert np.isfinite(reference).mean() > 0.9
    # the renderer applied the lens distortion: the distance maps warp well beyond render noise
    mean_abs_diff = _mean_abs_distance_diff(distorted, reference)
    assert mean_abs_diff > 0.05, f"distorted vs reference distance maps differ by only {mean_abs_diff:.4f} m"


@pytest.mark.parametrize("device", ["cuda:0"])
@_SKIP_NO_NEWTON
@_SKIP_NO_CUDA
def test_opencv_distortion_intrinsics_match_authored_newton(device):
    """The Newton camera reports intrinsics matching the authored, non-square, off-center calibration."""
    _distance, k = _render_distance(_pinhole_distortion(True), device=device)

    assert k[0, 0] == pytest.approx(_CALIB["fx"], abs=1e-2)
    assert k[1, 1] == pytest.approx(_CALIB["fy"], abs=1e-2)
    assert k[0, 2] == pytest.approx(_CALIB["cx"], abs=1e-2)
    assert k[1, 2] == pytest.approx(_CALIB["cy"], abs=1e-2)
    # not the stock fx == fy / centered-principal-point collapse
    assert k[0, 0] != k[1, 1]
    assert k[0, 2] != pytest.approx(WIDTH / 2)


@pytest.mark.parametrize("device", ["cuda:0"])
@_SKIP_NO_NEWTON
@_SKIP_NO_CUDA
def test_opencv_fisheye_distortion_renders_through_newton(device):
    """The Newton renderer honors the OpenCV fisheye model: its render differs meaningfully from the pinhole.

    The same calibrated camera is rendered under the OpenCV fisheye model and under an undistorted
    pinhole. The fisheye equidistant projection bends the rays, so the two distance maps must differ
    well beyond render noise, and the reported intrinsics must still match the authored calibration.
    """
    fisheye, k = _render_distance(_fisheye_distortion(True), device=device)
    pinhole, _ = _render_distance(_pinhole_distortion(False), device=device)

    assert fisheye.shape == (HEIGHT, WIDTH, 1)
    # both frames render geometry (the ground plane fills the frame)
    assert np.isfinite(fisheye).mean() > 0.9
    assert np.isfinite(pinhole).mean() > 0.9
    # the renderer applied the fisheye projection: the distance map differs from the pinhole beyond noise
    mean_abs_diff = _mean_abs_distance_diff(fisheye, pinhole)
    assert mean_abs_diff > 0.05, f"fisheye vs pinhole distance maps differ by only {mean_abs_diff:.4f} m"
    # the fisheye camera still reports the authored, non-square, off-center calibration
    assert k[0, 0] == pytest.approx(_CALIB["fx"], abs=1e-2)
    assert k[1, 1] == pytest.approx(_CALIB["fy"], abs=1e-2)
    assert k[0, 2] == pytest.approx(_CALIB["cx"], abs=1e-2)
    assert k[1, 2] == pytest.approx(_CALIB["cy"], abs=1e-2)
