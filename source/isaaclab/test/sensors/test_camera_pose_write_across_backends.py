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
"""

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

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
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
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
def test_physx_mounted_warp_camera_tracks_motion_and_state_writes(device):
    """A Warp-rendered camera follows native PhysX transforms without requiring RTX rendering."""
    sim_cfg = SimulationCfg(physics=PhysxCfg(), device=device, gravity=(0.0, 0.0, 0.0))
    with build_simulation_context(device=device, sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = _SceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
        scene_cfg.cube.init_state.pos = (0.0, 0.0, CLOSE_M)
        scene_cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Cube/Camera",
            height=32,
            width=32,
            data_types=["distance_to_image_plane"],
            renderer_cfg=NewtonWarpRendererCfg(),
            update_latest_camera_pose=True,
            offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.5), convention="opengl"),
            spawn=sim_utils.PinholeCameraCfg(),
        )
        scene = InteractiveScene(scene_cfg)
        camera = scene["camera"]
        sim.set_setting("/physics/fabricUpdateTransformations", False)
        sim.reset()
        camera.reset()
        cube = scene["cube"]
        cube.write_root_velocity_to_sim_index(
            root_velocity=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=device)
        )
        for step in range(1, 3):
            sim.step(render=False)
            camera.update(sim.get_physics_dt())
            expected_z = CLOSE_M + 0.5 + step * sim.get_physics_dt()
            data = camera.data
            torch.testing.assert_close(
                data.pos_w.torch, torch.tensor([[1.0, 0.0, expected_z]], device=device), atol=1e-4, rtol=0
            )
            assert data.output["distance_to_image_plane"].torch[0, 16, 16, 0].item() == pytest.approx(
                expected_z, abs=1e-3
            )

        # Match reset-event ordering: clear sensor state, then teleport the body, then read without a step.
        camera.reset()
        pose = torch.tensor([[0.0, 0.0, FAR_M, 0.0, 0.0, 0.0, 1.0]], device=device)
        cube.write_root_pose_to_sim_index(root_pose=pose)
        data = camera.data
        torch.testing.assert_close(data.pos_w.torch, torch.tensor([[1.0, 0.0, FAR_M + 0.5]], device=device))
        assert data.output["distance_to_image_plane"].torch[0, 16, 16, 0].item() == pytest.approx(FAR_M + 0.5, abs=1e-3)

        # Direct pose queries stay current even when measurement poses are configured to remain cached.
        camera.cfg.update_latest_camera_pose = False
        pose[:, 2] = 3.0
        cube.write_root_pose_to_sim_index(root_pose=pose)
        frame = camera.frame.torch.clone()
        torch.testing.assert_close(camera.get_world_poses()[0], torch.tensor([[1.0, 0.0, 3.5]], device=device))
        torch.testing.assert_close(camera.data.pos_w.torch, torch.tensor([[1.0, 0.0, FAR_M + 0.5]], device=device))
        torch.testing.assert_close(camera.frame.torch, frame)
        if device.startswith("cuda"):
            assert not sim.get_setting("/physics/fabricUpdateTransformations")
