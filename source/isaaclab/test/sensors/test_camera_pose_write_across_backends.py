# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera pose writes must take effect on every physics backend.

:meth:`Camera.set_world_poses` writes through the sensor's :class:`FrameView`, whose implementation is
backend-specific. The Newton view used to update Warp state only, leaving ``camera.data.pos_w`` correct
while the RTX renderer -- which reads the camera prim -- kept drawing the old pose.

Both tests raise a downward-looking camera over the ground plane, which scales the distance to every
visible surface, and check one observable consequence each: the reported pose, and the rendered depth.
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

# Camera heights [m] above the ground plane. Far/close is a 4x depth change, far past any framing noise.
CLOSE_M, FAR_M = 2.0, 8.0


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
    """Move the camera to each height in turn, returning its ``(pos_w, mean depth [m])`` at each.

    The camera looks straight down at the ground plane, so the depth it reports is dominated by its
    height. It spawns at the first height rather than at the origin, so a dropped pose write leaves a
    valid (but unchanged) depth image rather than an empty one.
    """
    device = "cuda:0"
    camera_cfg = CameraCfg(
        prim_path="/World/Camera",
        height=128,
        width=256,
        update_period=0,
        # Re-read the pose from the backend view on update, so pos_w reports what the write achieved
        # rather than echoing the commanded value.
        update_latest_camera_pose=True,
        data_types=["distance_to_camera"],
        # OpenGL convention: the camera looks along its -Z axis, so identity orientation looks straight down.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, heights_m[0]), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
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
            # Two steps so the renderer produces a frame at the new pose.
            for _ in range(2):
                sim.step()
            camera.update(sim.get_physics_dt())
            depth = camera.data.output["distance_to_camera"].torch.detach().float().cpu()
            visible = depth[torch.isfinite(depth)]
            assert visible.numel() > 0, "No valid depth pixels; the camera sees no geometry."
            captures.append((camera.data.pos_w.torch.detach().float().cpu().clone(), visible.mean().item()))

        del camera

    return captures


@pytest.mark.parametrize("physics_cfg", BACKEND_CFGS, ids=BACKEND_IDS)
def test_camera_pose_write_moves_reported_pose(physics_cfg):
    """``camera.data.pos_w`` follows a ``set_world_poses`` write on every backend."""
    (pos_close, _), (pos_far, _) = _capture_at_heights(physics_cfg, (CLOSE_M, FAR_M))

    np.testing.assert_allclose(pos_close.numpy(), [[0.0, 0.0, CLOSE_M]], atol=1e-3)
    np.testing.assert_allclose(pos_far.numpy(), [[0.0, 0.0, FAR_M]], atol=1e-3)


@pytest.mark.parametrize("physics_cfg", BACKEND_CFGS, ids=BACKEND_IDS)
def test_camera_pose_write_moves_render(physics_cfg):
    """The rendered depth follows a ``set_world_poses`` write on every backend.

    A write that never reaches the camera prim the RTX renderer reads leaves the image at the old pose,
    collapsing the ratio to ~1.
    """
    # The true ratio is ~4x; 1.5 leaves room for the ground plane filling different fractions of the
    # frame while still failing hard if the render does not move at all.
    depth_ratio_threshold = 1.5

    (_, depth_close_m), (_, depth_far_m) = _capture_at_heights(physics_cfg, (CLOSE_M, FAR_M))

    ratio = depth_far_m / depth_close_m
    assert ratio > depth_ratio_threshold, (
        f"Far depth ({depth_far_m:.2f} m) should be > {depth_ratio_threshold}x close depth"
        f" ({depth_close_m:.2f} m); got {ratio:.2f}x. The camera pose write is not reaching the renderer."
    )
