# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera pose writes must take effect on every physics backend.

:meth:`Camera.set_world_poses` writes through the sensor's :class:`FrameView`. Under PhysX that view is
Fabric-backed (:class:`FabricFrameView`), so the RTX renderer -- which reads the USD/Fabric camera prim --
follows the write. Under Newton the view is a :class:`NewtonSiteFrameView`, whose writes land in Warp
state and are mirrored onto the camera prim's Fabric transforms when the writer scope exits.

Both tests move a downward-looking camera straight up over a ground plane, which multiplies the distance
to every visible surface, and check the two observable consequences of the write:

- ``_moves_reported_pose_*`` reads ``camera.data.pos_w`` (deterministic, no renderer involved).
- ``_moves_render_*`` compares the rendered depth before and after the move.

The render check is not redundant with the Fabric-level coverage in
``isaaclab_newton/test/physics/test_newton_fabric_body_sync.py``. Writing through a read-only Fabric
selection still lands the new matrix in Fabric -- a test that reads the matrix back passes -- while the
renderer is never notified and keeps drawing the old pose. Only rendered output distinguishes the two.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

BACKEND_CFGS = [PhysxCfg(), NewtonCfg(solver_cfg=MJWarpSolverCfg())]
BACKEND_IDS = ["physx", "newton"]


@configclass
class _SceneCfg(InteractiveSceneCfg):
    """A single rigid body under the camera; Newton cannot build a model from an empty scene."""

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )


def _capture_at_heights(physics_cfg, heights_m: tuple[float, ...]) -> list[tuple[torch.Tensor, float]]:
    """Move the camera to each height in turn, returning its ``(pos_w, mean visible depth [m])`` at each.

    The camera looks straight down at the ground plane, so the depth it reports is dominated by its height.
    It spawns at the first height rather than at the origin, so that a dropped pose write leaves a valid
    (but unchanged) depth image rather than an empty one.

    Args:
        physics_cfg: The physics backend configuration to build the simulation with.
        heights_m: Camera heights [m] above the ground plane to capture at, in order.
    """
    device = "cuda:0"
    # Physics steps taken after each pose write so the renderer produces a frame at the new pose.
    steps_per_pose = 2
    max_range_m = 1.0e5
    camera_cfg = CameraCfg(
        prim_path="/World/Camera",
        height=128,
        width=256,
        update_period=0,
        update_latest_camera_pose=True,
        data_types=["distance_to_camera"],
        # OpenGL convention: the camera looks along its -Z axis, so identity orientation looks straight down.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, heights_m[0]), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, max_range_m)
        ),
    )

    sim_cfg = SimulationCfg(physics=physics_cfg, device=device)
    captures = []
    with build_simulation_context(device=device, sim_cfg=sim_cfg, add_ground_plane=True, add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.0))
        camera = Camera(camera_cfg)
        sim.reset()

        orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
        for height in heights_m:
            positions = torch.tensor([[0.0, 0.0, height]], device=device)
            camera.set_world_poses(positions, orientations, convention="opengl")
            for _ in range(steps_per_pose):
                sim.step()
            camera.update(sim.get_physics_dt())
            depth = camera.data.output["distance_to_camera"].torch.detach().float().cpu()
            # The sky renders at the far clipping range; average only over pixels that hit geometry.
            visible = depth[torch.isfinite(depth) & (depth < max_range_m)]
            assert visible.numel() > 0, "No valid depth pixels; the camera sees no geometry."
            captures.append((camera.data.pos_w.torch.detach().float().cpu().clone(), visible.mean().item()))

        del camera

    return captures


@pytest.mark.parametrize("physics_cfg", BACKEND_CFGS, ids=BACKEND_IDS)
def test_camera_pose_write_moves_reported_pose(physics_cfg):
    """``camera.data.pos_w`` follows a ``set_world_poses`` write on every backend."""
    # Camera heights [m] above the ground plane, and the reported shift below which the write was dropped.
    close_m, far_m = 2.0, 8.0
    pose_shift_threshold_m = 0.5 * (far_m - close_m)

    (pos_close, _), (pos_far, _) = _capture_at_heights(physics_cfg, (close_m, far_m))

    np.testing.assert_allclose(pos_close.numpy(), [[0.0, 0.0, close_m]], atol=1e-3)
    np.testing.assert_allclose(pos_far.numpy(), [[0.0, 0.0, far_m]], atol=1e-3)
    shift = (pos_far - pos_close).norm(dim=-1).max().item()
    assert shift > pose_shift_threshold_m, (
        f"Expected camera.data.pos_w to follow the pose write (> {pose_shift_threshold_m} m); got {shift:.4f} m."
    )


@pytest.mark.parametrize("physics_cfg", BACKEND_CFGS, ids=BACKEND_IDS)
def test_camera_pose_write_moves_render(physics_cfg):
    """The rendered depth follows a ``set_world_poses`` write on every backend."""
    # Camera heights [m] above the ground plane. The true far-to-close depth ratio is ~4x; 1.5 leaves room
    # for the ground plane filling different fractions of the frame while still failing hard if the render
    # does not move at all.
    close_m, far_m = 2.0, 8.0
    depth_ratio_threshold = 1.5

    (_, depth_close_m), (_, depth_far_m) = _capture_at_heights(physics_cfg, (close_m, far_m))

    ratio = depth_far_m / depth_close_m
    assert ratio > depth_ratio_threshold, (
        f"Far depth ({depth_far_m:.2f} m) should be > {depth_ratio_threshold}x close depth"
        f" ({depth_close_m:.2f} m); got {ratio:.2f}x. The camera pose write is not reaching the renderer."
    )
