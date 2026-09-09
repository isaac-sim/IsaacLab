# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera pose writes must take effect on every physics backend.

A downward-looking camera moves up over a ground plane; both consequences are checked --
``camera.data.pos_w`` and the rendered depth. The render half is not covered by
``isaaclab_newton/test/physics/test_newton_fabric_body_sync.py``: a write can land in Fabric, and
read back fine, while the renderer is never notified.
"""

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
    """Move the camera to each height [m] in turn, returning its ``(pos_w, mean visible depth [m])``.

    The camera spawns at the first height rather than the origin, so a dropped write leaves an
    unchanged image, not an empty one.
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
def test_camera_pose_write_moves_reported_pose_and_render(physics_cfg):
    """``camera.data.pos_w`` and the rendered depth both follow a ``set_world_poses`` write."""
    close_m, far_m = 2.0, 8.0
    # True far-to-close depth ratio is ~4x; 1.5 tolerates framing differences but not a frozen render.
    depth_ratio_threshold = 1.5

    (pos_close, depth_close_m), (pos_far, depth_far_m) = _capture_at_heights(physics_cfg, (close_m, far_m))

    np.testing.assert_allclose(pos_close.numpy(), [[0.0, 0.0, close_m]], atol=1e-3)
    np.testing.assert_allclose(pos_far.numpy(), [[0.0, 0.0, far_m]], atol=1e-3)
    ratio = depth_far_m / depth_close_m
    assert ratio > depth_ratio_threshold, (
        f"Far depth ({depth_far_m:.2f} m) should be > {depth_ratio_threshold}x close depth"
        f" ({depth_close_m:.2f} m); got {ratio:.2f}x. The camera pose write is not reaching the renderer."
    )
