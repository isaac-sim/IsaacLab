# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fabric world transforms and camera observations after environment resets on PhysX.

RTX renders rigid bodies from ``omni:fabric:worldMatrix``. A reset writes poses through the PhysX tensor API
without a physics step, so it must publish them to Fabric itself. The cube test also checks the depth
observations that the reset returns.
"""

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

launch_test_simulation(enable_cameras=True)

import pytest
import torch
from isaaclab_physx.physics import PhysxCfg

import usdrt

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim.utils.stage import get_current_stage_id
from isaaclab.test.env_cfgs import make_empty_manager_based_env_cfg
from isaaclab.test.integration_scene_cfgs import CartpoleTestSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import matrix_from_quat

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

# resolution
HEIGHT = 256
WIDTH = 256

# number of sim steps before capturing a stabilised reference frame
STABILISATION_STEPS = 5

# far camera clipping distance [m]
DEPTH_FAR = 10.0

# cube reset pose relative to its spawn pose along x [m]
CUBE_RESET_X_OFFSET = 0.15


@configclass
class _DepthObservationsCfg(ObservationGroupCfg):
    """Camera geometry returned directly by environment reset."""

    concatenate_terms = False
    depth = ObservationTermCfg(
        func=mdp.image_depth,
        params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane", "normalize": False},
    )


def _make_reset_env_cfg(device: str) -> ManagerBasedEnvCfg:
    """Return a gravity-free Fabric scene with one top-down RTX depth camera.

    The camera switches on RTX sensor rendering and continuous Fabric transform updates. In this configuration,
    the renderer reads the reset poses from Fabric.

    Args:
        device: Simulation device.
    """
    cfg = make_empty_manager_based_env_cfg(device=device)
    cfg.seed = 0
    cfg.decimation = 1
    cfg.sim.render_interval = 1
    cfg.sim.physics = PhysxCfg()
    cfg.sim.use_fabric = True
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.sim.visualizer_cfgs = []
    cfg.num_rerenders_on_reset = 2
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=HEIGHT,
        width=WIDTH,
        data_types=["distance_to_image_plane"],
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), convention="opengl"),
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1, DEPTH_FAR)),
    )
    return cfg


def _add_reset_cube(cfg: ManagerBasedEnvCfg) -> None:
    """Add a cube whose reset event moves it :data:`CUBE_RESET_X_OFFSET` away from its spawn pose."""
    cfg.events.reset_cube_pose = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {"x": (CUBE_RESET_X_OFFSET, CUBE_RESET_X_OFFSET)},
            "velocity_range": {},
        },
    )
    cfg.scene.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.15, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, 0.0, 0.2)),
    )


def _cube_reset_pose(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the world pose that the reset event writes for the cube."""
    pose = env.scene["cube"].data.default_root_pose.torch.clone()
    pose[:, :3] += env.scene.env_origins
    pose[:, 0] += CUBE_RESET_X_OFFSET
    return pose


def _moved_cube_pose(reset_pose: torch.Tensor) -> torch.Tensor:
    """Return a cube pose that differs from the reset pose in position and orientation."""
    pose = reset_pose.clone()
    pose[:, 0] += 0.7
    pose[:, 3:] = torch.tensor((0.0, 0.0, 2**-0.5, 2**-0.5), device=pose.device)
    return pose


def _step(env: ManagerBasedEnv, num_steps: int = 1) -> dict:
    """Step physics with zero actions and return the last observations."""
    action = torch.zeros_like(env.action_manager.action)
    for _ in range(num_steps):
        obs, _ = env.step(action)
    return obs


def _silhouette(depth: torch.Tensor) -> torch.Tensor:
    """Return the pixels whose ray hits geometry. RTX writes ``+inf`` where the ray hits nothing."""
    return torch.isfinite(depth)


def _silhouette_iou(mask: torch.Tensor, reference_mask: torch.Tensor) -> float:
    """Return the intersection over union of two silhouettes."""
    return ((mask & reference_mask).sum() / (mask | reference_mask).sum()).item()


def _fabric_world_transforms(prim_paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Fabric world positions [m] and orientation matrices of prims.

    The read is passive. Fabric frame views are not used, because they seed Fabric from USD.
    """
    stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
    matrices = torch.tensor(
        [
            [list(row) for row in stage.GetPrimAtPath(path).GetAttribute("omni:fabric:worldMatrix").Get()]
            for path in prim_paths
        ],
        dtype=torch.float32,
    )
    # USD matrices use row vectors: the rows of the upper 3x3 block are the scaled axes of the prim frame.
    orientation_matrices = torch.nn.functional.normalize(matrices[:, :3, :3], dim=-1).transpose(1, 2)
    positions = matrices[:, 3, :3]
    return positions, orientation_matrices


def _assert_transforms_match(transforms: tuple[torch.Tensor, torch.Tensor], poses: torch.Tensor) -> None:
    """Assert that world transforms equal poses given as position [m] and quaternion (x, y, z, w)."""
    positions, orientation_matrices = transforms
    torch.testing.assert_close(positions, poses[:, :3].cpu())
    torch.testing.assert_close(orientation_matrices, matrix_from_quat(poses[:, 3:]).cpu())


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.isaacsim_ci
def test_env_reset_shows_reset_pose_in_fabric_and_camera_observation(device: str):
    """The first and subsequent resets must show the reset pose in Fabric and in the camera observation.

    Neither reset may step physics.
    """
    sim_utils.create_new_stage()
    cfg = _make_reset_env_cfg(device)
    cfg.observations = {"camera": _DepthObservationsCfg()}
    _add_reset_cube(cfg)
    env = ManagerBasedEnv(cfg)
    try:
        cube = env.scene["cube"]
        cube_paths = cube.root_view.prim_paths
        reset_pose = _cube_reset_pose(env)
        first_reset_obs, _ = env.reset()
        first_reset_depth = first_reset_obs["camera"]["depth"].clone()
        assert env.sim.get_physics_step_count() == 0
        torch.testing.assert_close(cube.data.root_link_pose_w.torch, reset_pose)
        _assert_transforms_match(_fabric_world_transforms(cube_paths), reset_pose)

        # Establish rendered references at both poses before testing the reset boundary.
        reference_depth = _step(env, STABILISATION_STEPS)["camera"]["depth"].clone()
        torch.testing.assert_close(cube.data.root_link_pose_w.torch, reset_pose)
        # Fabric must show a different pose before the later reset, so a stale Fabric cannot match the reset pose.
        moved_pose = _moved_cube_pose(reset_pose)
        cube.write_root_pose_to_sim_index(root_pose=moved_pose)
        moved_depth = _step(env, STABILISATION_STEPS)["camera"]["depth"].clone()
        _assert_transforms_match(_fabric_world_transforms(cube_paths), moved_pose)

        step_before_reset = env.sim.get_physics_step_count()
        reset_obs, _ = env.reset()
        assert env.sim.get_physics_step_count() == step_before_reset
        torch.testing.assert_close(cube.data.root_link_pose_w.torch, reset_pose)
        _assert_transforms_match(_fabric_world_transforms(cube_paths), reset_pose)

        # The renderer must turn the published poses into the observations that both resets return.
        reference_mask, moved_mask = _silhouette(reference_depth), _silhouette(moved_depth)
        assert reference_mask.any() and moved_mask.any(), "The cube must be visible at both reference poses."
        assert not (reference_mask & moved_mask).any(), "Reference silhouettes must be spatially separated."
        for phase, depth in (("First", first_reset_depth), ("Later", reset_obs["camera"]["depth"])):
            overlap = _silhouette_iou(_silhouette(depth), reference_mask)
            assert overlap > 0.9, f"{phase} reset camera missed the reset silhouette: IoU={overlap:.3f}"
    finally:
        env.close()


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.isaacsim_ci
def test_env_first_reset_publishes_articulation_link_poses_to_fabric(device: str):
    """The first reset must publish the link poses of its joint positions to Fabric without stepping physics."""
    sim_utils.create_new_stage()
    cfg = _make_reset_env_cfg(device)
    cfg.scene.robot = CartpoleTestSceneCfg().robot
    # Link poses follow from joint positions through forward kinematics, not from root pose writes.
    joint_offset = 0.8
    cfg.events.reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (joint_offset, joint_offset),
            "velocity_range": (0.0, 0.0),
        },
    )
    env = ManagerBasedEnv(cfg)
    try:
        robot = env.scene["robot"]
        paths_by_name = dict(zip(robot.backend_body_names, robot.root_view.link_paths[0]))
        link_paths = [paths_by_name[name] for name in robot.body_names]
        env.reset()
        assert env.sim.get_physics_step_count() == 0
        torch.testing.assert_close(robot.data.joint_pos.torch, robot.data.default_joint_pos.torch + joint_offset)
        reset_link_poses = robot.data.body_link_pose_w.torch[0].clone()
        reset_fabric_transforms = _fabric_world_transforms(link_paths)

        # A physics step runs forward kinematics, so PhysX then holds the link poses of the reset joint positions.
        # Without gravity and drive stiffness, the joints hold their positions during the step.
        _step(env)
        link_poses = robot.data.body_link_pose_w.torch[0]
        torch.testing.assert_close(reset_link_poses, link_poses)
        _assert_transforms_match(reset_fabric_transforms, link_poses)
    finally:
        env.close()


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.isaacsim_ci
@pytest.mark.xfail(
    strict=True,
    raises=pytest.xfail.Exception,
    reason=(
        "PhysX Fabric limitation (#8428): after a timeline resume, force_update() ignores GPU tensor pose writes"
        " until the next simulate() call."
    ),
)
def test_env_reset_after_resume_publishes_reset_pose_to_fabric(device: str):
    """A reset directly after pause and resume must publish the reset pose to Fabric without stepping physics."""
    sim_utils.create_new_stage()
    cfg = _make_reset_env_cfg(device)
    _add_reset_cube(cfg)
    env = ManagerBasedEnv(cfg)
    try:
        cube = env.scene["cube"]
        cube_paths = cube.root_view.prim_paths
        reset_pose = _cube_reset_pose(env)
        env.reset()
        # Fabric must show a different pose before the pause, so a stale Fabric cannot match the reset pose.
        moved_pose = _moved_cube_pose(reset_pose)
        cube.write_root_pose_to_sim_index(root_pose=moved_pose)
        _step(env)
        _assert_transforms_match(_fabric_world_transforms(cube_paths), moved_pose)

        env.sim.pause()
        env.sim.play()
        step_before_reset = env.sim.get_physics_step_count()
        env.reset()
        assert env.sim.get_physics_step_count() == step_before_reset
        torch.testing.assert_close(cube.data.root_link_pose_w.torch, reset_pose)
        try:
            _assert_transforms_match(_fabric_world_transforms(cube_paths), reset_pose)
        except AssertionError as error:
            # pytest.xfail() does not set sys.last_exc, which makes the Kit launcher exit with status 1.
            pytest.xfail(f"Fabric holds a stale pose: {error}")
    finally:
        env.close()
