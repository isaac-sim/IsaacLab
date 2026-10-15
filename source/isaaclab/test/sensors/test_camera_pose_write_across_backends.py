# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera pose writes must take effect on every physics backend.

:meth:`Camera.set_world_poses` writes through the sensor's :class:`FrameView`, whose implementation is
backend-specific. The Newton view used to update Warp state only, leaving ``camera.data.pos_w`` correct
while the RTX renderer -- which reads the camera prim -- kept drawing the old pose.

The test raises a downward-looking camera over the ground plane, which scales the distance to every
visible surface, and checks both observable consequences: the reported pose and the rendered depth.

A camera attached to a moving body must also follow it. The PhysX view reads the camera's world matrix
from Fabric, which PhysX updates continuously only while something renders continuously, so a camera on
a falling PhysX body is checked with the Newton Warp renderer.
"""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.configclass import configclass, replace
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

BACKEND_CFGS = [PhysxCfg(), NewtonCfg(solver_cfg=MJWarpSolverCfg())]
BACKEND_IDS = ["physx", "newton"]

# Camera heights [m] above the ground plane. Far/close is a 4x depth change, far past any framing noise.
CLOSE_M, FAR_M = 2.0, 8.0

# Height [m] the cube falls from, and the attached camera's position [m] in the cube frame.
DROP_HEIGHT_M = 3.0
CAMERA_OFFSET_M = (0.0, 0.5, 0.0)


@configclass
class _SceneCfg(InteractiveSceneCfg):
    """A single rigid body under the camera; Newton cannot build a model from an empty scene."""

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )


@configclass
class _CameraOnCubeSceneCfg(_SceneCfg):
    """The cube with downward-looking cameras attached beside it, over a ground plane."""

    terrain: TerrainImporterCfg = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Cube/Camera",
        height=8,
        width=8,
        update_period=0,
        update_latest_camera_pose=True,
        data_types=["distance_to_image_plane"],
        # OpenGL identity looks straight down; beside the cube, the camera sees only the ground plane.
        offset=CameraCfg.OffsetCfg(pos=CAMERA_OFFSET_M, rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
        # Unlike Isaac RTX, Newton Warp does not keep the PhysX poses in Fabric updated.
        renderer_cfg=NewtonWarpRendererCfg(),
    )
    # Its data keeps cached poses, but get_world_poses() reads the live pose.
    cached_pose_camera: CameraCfg = replace(
        camera, prim_path="{ENV_REGEX_NS}/Cube/CachedPoseCamera", update_latest_camera_pose=False
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
def test_camera_pose_write_moves_pose_and_render(physics_cfg):
    """``camera.data.pos_w`` and the rendered depth follow a ``set_world_poses`` write on every backend.

    A write that never reaches the camera prim the RTX renderer reads leaves the image at the old pose,
    collapsing the depth ratio to ~1.
    """
    # The true ratio is ~4x; 1.5 leaves room for the ground plane filling different fractions of the
    # frame while still failing hard if the render does not move at all.
    depth_ratio_threshold = 1.5

    (pos_close, depth_close_m), (pos_far, depth_far_m) = _capture_at_heights(physics_cfg, (CLOSE_M, FAR_M))

    np.testing.assert_allclose(pos_close.numpy(), [[0.0, 0.0, CLOSE_M]], atol=1e-3)
    np.testing.assert_allclose(pos_far.numpy(), [[0.0, 0.0, FAR_M]], atol=1e-3)

    ratio = depth_far_m / depth_close_m
    assert ratio > depth_ratio_threshold, (
        f"Far depth ({depth_far_m:.2f} m) should be > {depth_ratio_threshold}x close depth"
        f" ({depth_close_m:.2f} m); got {ratio:.2f}x. The camera pose write is not reaching the renderer."
    )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_physx_camera_on_moving_body_follows_it_without_rtx(device):
    """A camera on a falling PhysX body follows it and keeps a pose written while the body moves.

    Without an RTX camera or a viewer rendering continuously, PhysX does not update Fabric on its own, so a
    camera on a moving body used to keep its spawn pose in ``camera.data.pos_w``, the rendered image and
    :meth:`Camera.get_world_poses`, which reads the live pose whatever ``update_latest_camera_pose`` is.
    """
    sim_cfg = SimulationCfg(physics=PhysxCfg(), device=device)
    with build_simulation_context(device=device, sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_CameraOnCubeSceneCfg(num_envs=2, env_spacing=2.0))
        sim.reset()
        cube, camera, cached_pose_camera = scene["cube"], scene["camera"], scene["cached_pose_camera"]

        def step(num_steps: int) -> torch.Tensor:
            """Step physics and return the cube poses."""
            for _ in range(num_steps):
                sim.step()
                scene.update(sim.get_physics_dt())
            return cube.data.root_link_pose_w.torch.clone()

        drop_pose = torch.zeros(scene.num_envs, 7, device=device)
        drop_pose[:, :3] = scene.env_origins
        drop_pose[:, 2] += DROP_HEIGHT_M
        drop_pose[:, 6] = 1.0
        cube.write_root_pose_to_sim_index(root_pose=drop_pose)

        pose = step(10)
        offset = torch.tensor([CAMERA_OFFSET_M] * scene.num_envs, device=device)
        expected, _ = combine_frame_transforms(pose[:, :3], pose[:, 3:], offset)
        # Read before any capture, which would refresh Fabric for every camera.
        torch.testing.assert_close(cached_pose_camera.get_world_poses()[0], expected, rtol=0.0, atol=1e-3)
        torch.testing.assert_close(camera.data.pos_w.torch, expected, rtol=0.0, atol=1e-3)
        # Looking straight down at the ground plane, every pixel's planar depth is the camera height.
        depth = camera.data.output["distance_to_image_plane"].torch
        torch.testing.assert_close(depth.mean(dim=(1, 2, 3)), expected[:, 2], rtol=0.0, atol=1e-3)

        # The cube keeps falling while the camera is not read; a pose written now must hold and then
        # move with the cube.
        pose = step(5)
        target = pose[:, :3] + torch.tensor([0.3, -0.4, 0.5], device=device)
        camera.set_world_poses(target, pose[:, 3:], convention="opengl")
        torch.testing.assert_close(camera.data.pos_w.torch, target, rtol=0.0, atol=1e-3)
        written_offset, _ = subtract_frame_transforms(pose[:, :3], pose[:, 3:], target)
        pose = step(5)
        expected, _ = combine_frame_transforms(pose[:, :3], pose[:, 3:], written_offset)
        torch.testing.assert_close(camera.data.pos_w.torch, expected, rtol=0.0, atol=1e-3)
