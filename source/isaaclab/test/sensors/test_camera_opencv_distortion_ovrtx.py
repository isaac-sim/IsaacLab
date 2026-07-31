# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate that the ``ovrtx`` renderer honors the OpenCV lens-distortion camera model natively.

The grid-textured ground plane is rendered through a camera carrying an OpenCV pinhole calibration.
With the distortion coefficients applied, OVRTX (which reads ``omni:lensdistortion:opencvPinhole:*``
from the prim) bows the straight grid lines; with the coefficients muted, the same camera renders them
straight. The two renders must therefore differ meaningfully. The reconstructed ``intrinsic_matrices``
are also checked end-to-end against the authored, non-square, off-center calibration.

Notes:
  * Runs **kit-less**: this test does not call :class:`~isaaclab.app.AppLauncher`. ``ovrtx`` and Isaac
    Sim Kit ship conflicting RTX hydra libraries that cannot co-load; see
    :func:`isaaclab.app.sim_launcher.launch_simulation`.
  * Uses Newton physics because ``ovrtx`` is incompatible with Kit/Isaac Sim.
  * The distorted and reference passes each build their own :class:`SimulationContext`; a single
    OVRTX render context applies one lens model, so the two passes must not share a process context.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "isaaclab_newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
_SKIP_MISSING_OVRTX = pytest.mark.skipif(
    bool(_MISSING_MODULES),
    reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
)

if not _MISSING_MODULES:
    import torch
    from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg

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


def _render_grid(distortion: OpenCvDistortionCfg, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Render the ground-plane grid through an OpenCV-calibrated OVRTX camera; return ``(rgb, K)``."""
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
            data_types=["rgb"],
            offset=CameraCfg.OffsetCfg(pos=_CAM_EYE, rot=rot, convention="opengl"),
            spawn=PinholeCameraCfg(focal_length=13.6, clipping_range=(0.001, 20.0), distortion=distortion),
            renderer_cfg=OVRTXRendererCfg(),
        )
    )
    try:
        sim.reset()
        for _ in range(WARMUP_STEPS):
            sim.step()
            camera.update(SIM_DT, force_recompute=True)
        rgb = camera.data.output["rgb"].torch[0].detach().cpu().float().numpy()[..., :3]
        intrinsics = camera.data.intrinsic_matrices.torch[0].detach().cpu().numpy()
        return rgb, intrinsics
    finally:
        del camera
        del scene
        sim.stop()
        sim.clear_instance()


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
@_SKIP_MISSING_OVRTX
def test_opencv_distortion_changes_ovrtx_render(device):
    """OVRTX must render the distorted and zero-coefficient cameras meaningfully differently."""
    distorted, _ = _render_grid(_pinhole_distortion(True), device=device)
    reference, _ = _render_grid(_pinhole_distortion(False), device=device)

    assert distorted.shape == (HEIGHT, WIDTH, 3)
    # both frames render content (not degenerate)
    assert distorted.std() > 1.0
    assert reference.std() > 1.0
    # OVRTX applied the lens distortion: the two renders differ well beyond render noise
    mean_abs_diff = np.abs(distorted.astype(np.float32) - reference.astype(np.float32)).mean()
    assert mean_abs_diff > 2.0, f"distorted vs reference differ by only {mean_abs_diff:.3f}/255"


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
@_SKIP_MISSING_OVRTX
def test_opencv_distortion_intrinsics_match_authored_ovrtx(device):
    """The OVRTX camera reports intrinsics matching the authored, non-square, off-center calibration."""
    _rgb, k = _render_grid(_pinhole_distortion(True), device=device)

    assert k[0, 0] == pytest.approx(_CALIB["fx"], abs=1e-2)
    assert k[1, 1] == pytest.approx(_CALIB["fy"], abs=1e-2)
    assert k[0, 2] == pytest.approx(_CALIB["cx"], abs=1e-2)
    assert k[1, 2] == pytest.approx(_CALIB["cy"], abs=1e-2)
    # not the stock fx == fy / centered-principal-point collapse
    assert k[0, 0] != k[1, 1]
    assert k[0, 2] != pytest.approx(WIDTH / 2)


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
@_SKIP_MISSING_OVRTX
def test_opencv_fisheye_distortion_renders_through_ovrtx(device):
    """OVRTX honors the OpenCV fisheye schema: its render differs meaningfully from the pinhole projection.

    The same calibrated camera is rendered under the OpenCV fisheye model and under an undistorted
    pinhole. The fisheye equidistant projection bends the straight grid, so the two frames must differ
    well beyond render noise, and the reported intrinsics must still match the authored calibration.
    """
    fisheye, k = _render_grid(_fisheye_distortion(True), device=device)
    pinhole, _ = _render_grid(_pinhole_distortion(False), device=device)

    assert fisheye.shape == (HEIGHT, WIDTH, 3)
    # both frames render content (not degenerate)
    assert fisheye.std() > 1.0
    assert pinhole.std() > 1.0
    # OVRTX applied the fisheye projection: it differs from the pinhole render well beyond render noise
    mean_abs_diff = np.abs(fisheye.astype(np.float32) - pinhole.astype(np.float32)).mean()
    assert mean_abs_diff > 2.0, f"fisheye vs pinhole differ by only {mean_abs_diff:.3f}/255"
    # the fisheye camera still reports the authored, non-square, off-center calibration
    assert k[0, 0] == pytest.approx(_CALIB["fx"], abs=1e-2)
    assert k[1, 1] == pytest.approx(_CALIB["fy"], abs=1e-2)
    assert k[0, 2] == pytest.approx(_CALIB["cx"], abs=1e-2)
    assert k[1, 2] == pytest.approx(_CALIB["cy"], abs=1e-2)
