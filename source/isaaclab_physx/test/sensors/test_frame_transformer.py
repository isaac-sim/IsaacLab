# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math
from types import SimpleNamespace

import pytest
import scipy.spatial.transform as tf
import torch
import warp as wp
from isaaclab_physx.sensors.frame_transformer import frame_transformer as frame_transformer_module
from isaaclab_physx.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab_physx.sensors.frame_transformer.frame_transformer_data import FrameTransformerData
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.sensors.frame_transformer import BaseFrameTransformer
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


def quat_from_euler_rpy(roll, pitch, yaw, degrees=False):
    """Converts Euler XYZ to Quaternion (x, y, z, w)."""
    quat = tf.Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees).as_quat()
    return tuple(quat.tolist())  # scipy already returns xyzw


def euler_rpy_apply(rpy, xyz, degrees=False):
    """Applies rotation from Euler XYZ on position vector."""
    rot = tf.Rotation.from_euler("xyz", rpy, degrees=degrees)
    return tuple(rot.apply(xyz).tolist())


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # terrain - flat terrain plane
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    # articulation - robot
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors - frame transformer (filled inside unit test)
    frame_transformer: FrameTransformerCfg = None

    # block
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=PhysxRigidBodyCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 5)),
    )


@pytest.fixture
def sim(request):
    """Create a simulation context, on CPU unless a device is passed via indirect parametrization."""
    device = getattr(request, "param", "cpu")
    sim_cfg = sim_utils.SimulationCfg(device=device, dt=0.005)
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        # Set main camera
        sim.set_camera_view(eye=(5.0, 5.0, 5.0), target=(0.0, 0.0, 0.0))
        yield sim
    # Cleanup is handled by build_simulation_context


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "sim",
    [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    ],
    indirect=True,
)
def test_frame_transformer_feet_wrt_base(sim):
    """Test frame transformers with different source and target frames in one scene.

    The sensors share one robot and cube scene:

    * ``frame_transformer``: feet (rotated offsets) w.r.t. the robot base.
    * ``ft_thigh``: feet w.r.t. a non-root thigh source, preserving the target name order.
    * ``ft_cube``: a target on another asset (the cube) w.r.t. the robot base.
    * ``ft_cube_offsets``: source equal to target with pure-translation offsets on one body.
    * ``ft_all``: all robot bodies through a regex target.
    """
    # Spawn things into stage
    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="LF_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/LF_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(0.08795, 0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, -math.pi / 2),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="RF_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/RF_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, math.pi / 2), xyz=(0.08795, -0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, math.pi / 2),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="LH_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/LH_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(-0.08795, 0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, -math.pi / 2),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="RH_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/RH_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, math.pi / 2), xyz=(-0.08795, -0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, math.pi / 2),
                ),
            ),
        ],
    )
    scene_cfg.ft_thigh = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_THIGH",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="LF_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/LF_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, -math.pi / 2), xyz=(0.08795, 0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, -math.pi / 2),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="RF_FOOT_USER",
                prim_path="{ENV_REGEX_NS}/Robot/RF_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, math.pi / 2), xyz=(0.08795, -0.01305, -0.33797)),
                    rot=quat_from_euler_rpy(0, 0, math.pi / 2),
                ),
            ),
        ],
    )
    scene_cfg.ft_cube = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="CUBE_USER",
                prim_path="{ENV_REGEX_NS}/cube",
            ),
        ],
    )
    scene_cfg.ft_cube_offsets = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="CUBE_CENTER",
                prim_path="{ENV_REGEX_NS}/cube",
            ),
            FrameTransformerCfg.FrameCfg(
                name="CUBE_TOP",
                prim_path="{ENV_REGEX_NS}/cube",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1),
                    rot=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="CUBE_BOTTOM",
                prim_path="{ENV_REGEX_NS}/cube",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, -0.1),
                    rot=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ],
    )
    scene_cfg.ft_all = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/[^/]*",
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    robot = scene.articulations["robot"]
    cube = scene.rigid_objects["cube"]
    ft = scene.sensors["frame_transformer"]
    ft_thigh = scene.sensors["ft_thigh"]
    ft_cube = scene.sensors["ft_cube"]
    ft_cube_offsets = scene.sensors["ft_cube_offsets"]
    ft_all = scene.sensors["ft_all"]
    assert "FrameTransformer @" in str(ft_all)

    # Acquire the index of ground truth bodies
    feet_indices, feet_names = robot.find_bodies(["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])

    # Reorder the feet indices to match the order of the target frames with _USER suffix removed
    target_frame_names = [name.split("_USER")[0] for name in ft.data.target_frame_names]
    reordering_indices = [feet_names.index(name) for name in target_frame_names]
    feet_indices = [feet_indices[i] for i in reordering_indices]

    # A non-root source frame keeps the target frames in body order
    thigh_index = robot.find_bodies("LF_THIGH")[0][0]
    thigh_feet_indices, thigh_feet_names = robot.find_bodies(["LF_FOOT", "RF_FOOT"])
    assert ft_thigh.data.target_frame_names == [f"{name}_USER" for name in thigh_feet_names]

    # Offset frames on the cube
    offset_frame_names = ft_cube_offsets.data.target_frame_names
    cube_center_idx = offset_frame_names.index("CUBE_CENTER")
    cube_bottom_idx = offset_frame_names.index("CUBE_BOTTOM")
    cube_top_idx = offset_frame_names.index("CUBE_TOP")

    # All bodies, reordered to the articulation body order
    articulation_body_names = robot.data.body_names
    all_reordering_indices = [ft_all.data.target_frame_names.index(name) for name in articulation_body_names]

    # default joint targets
    default_actions = robot.data.default_joint_pos.torch.clone()
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulate physics
    for count in range(100):
        # # reset
        if count % 25 == 0:
            # reset root state
            root_state = torch.cat((robot.data.default_root_pose.torch, robot.data.default_root_vel.torch), dim=-1)
            root_state = root_state.clone()
            root_state[:, :3] += scene.env_origins
            # -- robot
            robot.write_root_pose_to_sim_index(root_pose=root_state[:, :7])
            robot.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:])
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch)
            # -- cube
            cube_state = torch.cat((cube.data.default_root_pose.torch, cube.data.default_root_vel.torch), dim=-1)
            cube_state = cube_state.clone()
            cube_state[:, :3] += scene.env_origins
            cube.write_root_pose_to_sim_index(root_pose=cube_state[:, :7])
            cube.write_root_velocity_to_sim_index(root_velocity=cube_state[:, 7:])
            # reset buffers
            scene.reset()

        # set joint targets
        robot_actions = default_actions + 0.5 * torch.randn_like(default_actions)
        robot.set_joint_position_target_index(target=robot_actions)
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim_dt)

        # -- ground-truth
        root_pose_w = robot.data.root_pose_w.torch
        feet_pos_w_gt = robot.data.body_pos_w.torch[:, feet_indices]
        feet_quat_w_gt = robot.data.body_quat_w.torch[:, feet_indices]
        cube_pos_w_gt = cube.data.root_pos_w.torch
        cube_quat_w_gt = cube.data.root_quat_w.torch

        # feet w.r.t. base: world frame transforms
        feet_pos_w_tf = ft.data.target_pos_w.torch
        feet_quat_w_tf = ft.data.target_quat_w.torch
        torch.testing.assert_close(root_pose_w[:, :3], ft.data.source_pos_w.torch)
        torch.testing.assert_close(root_pose_w[:, 3:], ft.data.source_quat_w.torch)
        torch.testing.assert_close(feet_pos_w_gt, feet_pos_w_tf)
        torch.testing.assert_close(feet_quat_w_gt, feet_quat_w_tf)
        # feet w.r.t. base: relative transforms
        for index in range(len(feet_indices)):
            foot_pos_b, foot_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:], feet_pos_w_tf[:, index], feet_quat_w_tf[:, index]
            )
            torch.testing.assert_close(ft.data.target_pos_source.torch[:, index], foot_pos_b)
            torch.testing.assert_close(ft.data.target_quat_source.torch[:, index], foot_quat_b)

        # feet w.r.t. thigh
        thigh_pose_w_gt = robot.data.body_state_w.torch[:, thigh_index, :7]
        thigh_feet_pos_w_tf = ft_thigh.data.target_pos_w.torch
        thigh_feet_quat_w_tf = ft_thigh.data.target_quat_w.torch
        torch.testing.assert_close(thigh_pose_w_gt[:, :3], ft_thigh.data.source_pos_w.torch)
        torch.testing.assert_close(thigh_pose_w_gt[:, 3:], ft_thigh.data.source_quat_w.torch)
        torch.testing.assert_close(robot.data.body_pos_w.torch[:, thigh_feet_indices], thigh_feet_pos_w_tf)
        torch.testing.assert_close(robot.data.body_quat_w.torch[:, thigh_feet_indices], thigh_feet_quat_w_tf)
        for index in range(len(thigh_feet_indices)):
            foot_pos_b, foot_quat_b = math_utils.subtract_frame_transforms(
                thigh_pose_w_gt[:, :3],
                thigh_pose_w_gt[:, 3:],
                thigh_feet_pos_w_tf[:, index],
                thigh_feet_quat_w_tf[:, index],
            )
            torch.testing.assert_close(ft_thigh.data.target_pos_source.torch[:, index], foot_pos_b)
            torch.testing.assert_close(ft_thigh.data.target_quat_source.torch[:, index], foot_quat_b)

        # cube w.r.t. base
        cube_pos_w_tf = ft_cube.data.target_pos_w.torch.squeeze()
        cube_quat_w_tf = ft_cube.data.target_quat_w.torch.squeeze()
        torch.testing.assert_close(root_pose_w[:, :3], ft_cube.data.source_pos_w.torch)
        torch.testing.assert_close(root_pose_w[:, 3:], ft_cube.data.source_quat_w.torch)
        torch.testing.assert_close(cube_pos_w_gt, cube_pos_w_tf)
        torch.testing.assert_close(cube_quat_w_gt, cube_quat_w_tf)
        cube_pos_b, cube_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:], cube_pos_w_tf, cube_quat_w_tf
        )
        torch.testing.assert_close(ft_cube.data.target_pos_source.torch[:, 0], cube_pos_b)
        torch.testing.assert_close(ft_cube.data.target_quat_source.torch[:, 0], cube_quat_b)

        # cube offset frames
        target_pos_w_tf = ft_cube_offsets.data.target_pos_w.torch.squeeze()
        target_quat_w_tf = ft_cube_offsets.data.target_quat_w.torch.squeeze()
        torch.testing.assert_close(cube_pos_w_gt, ft_cube_offsets.data.source_pos_w.torch)
        torch.testing.assert_close(cube_quat_w_gt, ft_cube_offsets.data.source_quat_w.torch)
        torch.testing.assert_close(cube_pos_w_gt, target_pos_w_tf[:, cube_center_idx])
        torch.testing.assert_close(cube_quat_w_gt, target_quat_w_tf[:, cube_center_idx])
        torch.testing.assert_close(
            target_pos_w_tf[:, cube_top_idx], cube_pos_w_gt + torch.tensor([0.0, 0.0, 0.1], device=sim.device)
        )
        torch.testing.assert_close(target_quat_w_tf[:, cube_top_idx], cube_quat_w_gt)
        torch.testing.assert_close(
            target_pos_w_tf[:, cube_bottom_idx], cube_pos_w_gt + torch.tensor([0.0, 0.0, -0.1], device=sim.device)
        )
        torch.testing.assert_close(target_quat_w_tf[:, cube_bottom_idx], cube_quat_w_gt)

        # all bodies w.r.t. base
        bodies_pos_w_tf = ft_all.data.target_pos_w.torch
        bodies_quat_w_tf = ft_all.data.target_quat_w.torch
        torch.testing.assert_close(root_pose_w[:, :3], ft_all.data.source_pos_w.torch)
        torch.testing.assert_close(root_pose_w[:, 3:], ft_all.data.source_quat_w.torch)
        torch.testing.assert_close(robot.data.body_pos_w.torch, bodies_pos_w_tf[:, all_reordering_indices])
        torch.testing.assert_close(robot.data.body_quat_w.torch, bodies_quat_w_tf[:, all_reordering_indices])
        for index in range(len(articulation_body_names)):
            body_pos_b, body_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:], bodies_pos_w_tf[:, index], bodies_quat_w_tf[:, index]
            )
            torch.testing.assert_close(ft_all.data.target_pos_source.torch[:, index], body_pos_b)
            torch.testing.assert_close(ft_all.data.target_quat_source.torch[:, index], body_quat_b)

    # the recorded-launch optimization must be active on CUDA; a recording failure would only
    # warn and silently fall back to eager launches, defeating the optimization.
    if "cuda" in str(sim.device):
        sensors = (ft, ft_thigh, ft_cube, ft_cube_offsets, ft_all)
        for sensor in sensors:
            assert sensor._update_cmd is not None
        # The recorded launch is reused across updates rather than re-recorded every step.
        recorded = [sensor._update_cmd for sensor in sensors]
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        for sensor, update_cmd in zip(sensors, recorded):
            assert sensor._update_cmd is update_cmd


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(("source_robot", "path_prefix"), [("Robot_1", "{ENV_REGEX_NS}"), ("Robot", "/World")])
def test_frame_transformer_duplicate_body_names(sim, source_robot, path_prefix):
    """Test tracking bodies with same leaf name at different hierarchy levels.

    This test verifies that bodies with the same leaf name but different paths
    (e.g., Robot/LF_SHANK vs Robot_1/LF_SHANK, or arm/link vs leg/link) are tracked
    separately using their full relative paths internally.

    The test uses 4 target frames to cover both scenarios:

    Explicit frame names (recommended when bodies share the same leaf name):
        User provides unique names like "Robot_LF_SHANK" and "Robot_1_LF_SHANK" to
        distinguish between bodies at different hierarchy levels. This makes it
        easy to identify which transform belongs to which body.

    Implicit frame names (backward compatibility):
        When no name is provided, it defaults to the leaf body name (e.g., "RF_SHANK").
        This preserves backward compatibility for users who may have existing code like
        `idx = target_frame_names.index("RF_SHANK")`. However, when multiple bodies share
        the same leaf name, this results in duplicate frame names. The transforms are
        still distinct because internal body tracking uses full relative paths.

    Args:
        source_robot: The robot to use as the source frame ("Robot" or "Robot_1").
                      This tests that both source frames work correctly when there are
                      duplicate body names.
        path_prefix: The path prefix to use ("{ENV_REGEX_NS}" for env patterns or "/World" for direct paths).
    """

    # Create a custom scene config with two robots
    @configclass
    class MultiRobotSceneCfg(InteractiveSceneCfg):
        """Scene with two robots having bodies with same names."""

        terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

        # Frame transformer will be set after config creation (needs source_robot parameter)
        frame_transformer: FrameTransformerCfg = None  # type: ignore

    # Use multiple envs for env patterns, single env for direct paths
    num_envs = 2 if path_prefix == "{ENV_REGEX_NS}" else 1
    env_spacing = 10.0 if path_prefix == "{ENV_REGEX_NS}" else 0.0

    # Create scene config with appropriate prim paths
    scene_cfg = MultiRobotSceneCfg(num_envs=num_envs, env_spacing=env_spacing, lazy_sensor_update=False)
    scene_cfg.robot = ANYMAL_C_CFG.replace(prim_path=f"{path_prefix}/Robot")
    scene_cfg.robot_1 = ANYMAL_C_CFG.replace(
        prim_path=f"{path_prefix}/Robot_1",
        init_state=ANYMAL_C_CFG.init_state.replace(pos=(2.0, 0.0, 0.6)),
    )

    # Frame transformer tracking same-named bodies from both robots
    # Source frame is parametrized to test both Robot/base and Robot_1/base
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path=f"{path_prefix}/{source_robot}/base",
        target_frames=[
            # Explicit frame names (recommended when bodies share the same leaf name)
            FrameTransformerCfg.FrameCfg(
                name="Robot_LF_SHANK",
                prim_path=f"{path_prefix}/Robot/LF_SHANK",
            ),
            FrameTransformerCfg.FrameCfg(
                name="Robot_1_LF_SHANK",
                prim_path=f"{path_prefix}/Robot_1/LF_SHANK",
            ),
            # Implicit frame names (backward compatibility)
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{path_prefix}/Robot/RF_SHANK",
                offset=OffsetCfg(pos=(0.1, 0.0, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{path_prefix}/Robot_1/RF_SHANK",
                offset=OffsetCfg(pos=(0.0, 0.2, 0.0)),
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Get target frame names
    target_frame_names = scene.sensors["frame_transformer"].data.target_frame_names

    # Verify explicit frame names are present
    assert "Robot_LF_SHANK" in target_frame_names, f"Expected 'Robot_LF_SHANK', got {target_frame_names}"
    assert "Robot_1_LF_SHANK" in target_frame_names, f"Expected 'Robot_1_LF_SHANK', got {target_frame_names}"

    # Without explicit names, both RF_SHANK frames default to same name "RF_SHANK"
    # This results in duplicate frame names (expected behavior for backwards compatibility)
    rf_shank_count = target_frame_names.count("RF_SHANK")
    assert rf_shank_count == 2, f"Expected 2 'RF_SHANK' entries (name collision), got {rf_shank_count}"

    # Get indices for explicit named frames
    robot_lf_idx = target_frame_names.index("Robot_LF_SHANK")
    robot_1_lf_idx = target_frame_names.index("Robot_1_LF_SHANK")

    # Get indices for implicit named frames (both named "RF_SHANK")
    rf_shank_indices = [i for i, name in enumerate(target_frame_names) if name == "RF_SHANK"]
    assert len(rf_shank_indices) == 2, f"Expected 2 RF_SHANK indices, got {rf_shank_indices}"

    # Acquire ground truth body indices
    robot_base_body_idx = scene.articulations["robot"].find_bodies("base")[0][0]
    robot_1_base_body_idx = scene.articulations["robot_1"].find_bodies("base")[0][0]
    robot_lf_shank_body_idx = scene.articulations["robot"].find_bodies("LF_SHANK")[0][0]
    robot_1_lf_shank_body_idx = scene.articulations["robot_1"].find_bodies("LF_SHANK")[0][0]
    robot_rf_shank_body_idx = scene.articulations["robot"].find_bodies("RF_SHANK")[0][0]
    robot_1_rf_shank_body_idx = scene.articulations["robot_1"].find_bodies("RF_SHANK")[0][0]

    # Determine expected source frame based on parameter
    expected_source_robot = "robot" if source_robot == "Robot" else "robot_1"
    expected_source_base_body_idx = robot_base_body_idx if source_robot == "Robot" else robot_1_base_body_idx

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Simulate physics
    for count in range(20):
        # Reset periodically
        if count % 10 == 0:
            # Reset robot
            root_state = torch.cat(
                (
                    scene.articulations["robot"].data.default_root_pose.torch,
                    scene.articulations["robot"].data.default_root_vel.torch,
                ),
                dim=-1,
            ).clone()
            root_state[:, :3] += scene.env_origins
            scene.articulations["robot"].write_root_pose_to_sim_index(root_pose=root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:])
            scene.articulations["robot"].write_joint_position_to_sim_index(
                position=scene.articulations["robot"].data.default_joint_pos.torch
            )
            scene.articulations["robot"].write_joint_velocity_to_sim_index(
                velocity=scene.articulations["robot"].data.default_joint_vel.torch
            )
            # Reset robot_1
            root_state_1 = torch.cat(
                (
                    scene.articulations["robot_1"].data.default_root_pose.torch,
                    scene.articulations["robot_1"].data.default_root_vel.torch,
                ),
                dim=-1,
            ).clone()
            root_state_1[:, :3] += scene.env_origins
            scene.articulations["robot_1"].write_root_pose_to_sim_index(root_pose=root_state_1[:, :7])
            scene.articulations["robot_1"].write_root_velocity_to_sim_index(root_velocity=root_state_1[:, 7:])
            scene.articulations["robot_1"].write_joint_position_to_sim_index(
                position=scene.articulations["robot_1"].data.default_joint_pos.torch
            )
            scene.articulations["robot_1"].write_joint_velocity_to_sim_index(
                velocity=scene.articulations["robot_1"].data.default_joint_vel.torch
            )
            scene.reset()

        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Read data from sim
        scene.update(sim_dt)

        # Get frame transformer data
        frame_transformer_data = scene.sensors["frame_transformer"].data
        source_pos_w = frame_transformer_data.source_pos_w.torch
        source_quat_w = frame_transformer_data.source_quat_w.torch
        target_pos_w = frame_transformer_data.target_pos_w.torch

        # Get ground truth positions and orientations (after scene.update() so they're current)
        robot_lf_pos_w = scene.articulations["robot"].data.body_pos_w.torch[:, robot_lf_shank_body_idx]
        robot_1_lf_pos_w = scene.articulations["robot_1"].data.body_pos_w.torch[:, robot_1_lf_shank_body_idx]
        robot_rf_pos_w = scene.articulations["robot"].data.body_pos_w.torch[:, robot_rf_shank_body_idx]
        robot_1_rf_pos_w = scene.articulations["robot_1"].data.body_pos_w.torch[:, robot_1_rf_shank_body_idx]
        robot_rf_quat_w = scene.articulations["robot"].data.body_quat_w.torch[:, robot_rf_shank_body_idx]
        robot_1_rf_quat_w = scene.articulations["robot_1"].data.body_quat_w.torch[:, robot_1_rf_shank_body_idx]

        # Get expected source frame positions and orientations (after scene.update() so they're current)
        expected_source_base_pos_w = scene.articulations[expected_source_robot].data.body_pos_w.torch[
            :, expected_source_base_body_idx
        ]
        expected_source_base_quat_w = scene.articulations[expected_source_robot].data.body_quat_w.torch[
            :, expected_source_base_body_idx
        ]

        # TEST 1: Verify source frame is correctly resolved
        # The source_pos_w should match the expected source robot's base world position
        torch.testing.assert_close(source_pos_w, expected_source_base_pos_w, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(source_quat_w, expected_source_base_quat_w, rtol=1e-5, atol=1e-5)

        # TEST 2: Explicit named frames (LF_SHANK) should have DIFFERENT world positions
        lf_pos_difference = torch.linalg.norm(target_pos_w[:, robot_lf_idx] - target_pos_w[:, robot_1_lf_idx], dim=-1)
        assert torch.all(lf_pos_difference > 1.0), (
            f"Robot_LF_SHANK and Robot_1_LF_SHANK should have different positions (got diff={lf_pos_difference}). "
            "This indicates body name collision bug."
        )

        # Verify explicit named frames match correct robot bodies
        torch.testing.assert_close(target_pos_w[:, robot_lf_idx], robot_lf_pos_w)
        torch.testing.assert_close(target_pos_w[:, robot_1_lf_idx], robot_1_lf_pos_w)

        # TEST 3: Implicit named frames (RF_SHANK) should also have DIFFERENT world positions
        # Even though they have the same frame name, internal body tracking uses full paths
        rf_pos_difference = torch.linalg.norm(
            target_pos_w[:, rf_shank_indices[0]] - target_pos_w[:, rf_shank_indices[1]], dim=-1
        )
        assert torch.all(rf_pos_difference > 1.0), (
            f"The two RF_SHANK frames should have different positions (got diff={rf_pos_difference}). "
            "This indicates body name collision bug in internal body tracking."
        )

        # Verify implicit named frames preserve the offset configured for each body.
        robot_rf_expected_pos_w, _ = math_utils.combine_frame_transforms(
            robot_rf_pos_w,
            robot_rf_quat_w,
            torch.tensor((0.1, 0.0, 0.0), device=robot_rf_pos_w.device).expand_as(robot_rf_pos_w),
        )
        robot_1_rf_expected_pos_w, _ = math_utils.combine_frame_transforms(
            robot_1_rf_pos_w,
            robot_1_rf_quat_w,
            torch.tensor((0.0, 0.2, 0.0), device=robot_1_rf_pos_w.device).expand_as(robot_1_rf_pos_w),
        )
        expected_rf_positions = [robot_rf_expected_pos_w, robot_1_rf_expected_pos_w]

        # Note: Order depends on internal processing, so we check both match one of the expected transforms.
        rf_positions = [target_pos_w[:, rf_shank_indices[0]], target_pos_w[:, rf_shank_indices[1]]]
        for rf_pos in rf_positions:
            assert any(torch.allclose(rf_pos, expected, atol=1e-5) for expected in expected_rf_positions), (
                f"RF_SHANK position {rf_pos} doesn't match either configured body offset"
            )


class _FakeTransformView:
    """Return one stable PhysX-like transform buffer while counting typed-view construction."""

    def __init__(self, transforms: wp.array):
        self.transforms = transforms
        self.get_count = 0
        self.view_count = 0

    def get_transforms(self):
        self.get_count += 1
        return self

    def view(self, dtype):
        assert dtype == wp.transformf
        self.view_count += 1
        return self.transforms

    @property
    def ptr(self):
        return self.transforms.ptr


def _make_frame_transformer(use_recorded_launch: bool = True):
    """Create a one-environment FrameTransformer without a USD scene."""
    device = "cuda:0"
    transforms_torch = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    transforms = wp.from_torch(transforms_torch.contiguous()).view(wp.transformf)
    transform_view = _FakeTransformView(transforms)

    sensor = FrameTransformer.__new__(FrameTransformer)
    sensor.cfg = SimpleNamespace(prim_path="/World/Source")
    sensor._device = device
    sensor._num_envs = 1
    sensor._num_target_frames = 1
    sensor._frame_physx_view = transform_view
    sensor._source_raw_indices = wp.array([0], dtype=wp.int32, device=device)
    sensor._target_raw_indices = wp.array([[1]], dtype=wp.int32, device=device)
    sensor._source_offset_pos_wp = wp.zeros(1, dtype=wp.vec3f, device=device)
    sensor._source_offset_quat_wp = wp.array([wp.quatf(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=device)
    sensor._target_offset_pos_wp = wp.zeros(1, dtype=wp.vec3f, device=device)
    sensor._target_offset_quat_wp = wp.array([wp.quatf(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=device)
    sensor._data = FrameTransformerData()
    sensor._data.create_buffers(num_envs=1, num_target_frames=1, target_frame_names=["target"], device=device)
    sensor._raw_transforms = None
    sensor._update_cmd = None
    sensor._use_recorded_launch = use_recorded_launch
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None

    env_mask = wp.ones(1, dtype=wp.bool, device=device)
    return sensor, transform_view, transforms_torch, env_mask


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_frame_transformer_falls_back_when_recording_fails(monkeypatch):
    """A recording failure should disable recording and execute the current update eagerly.

    Later eager updates refresh the PhysX buffer but reuse one typed view over it.
    """
    sensor, transform_view, _, env_mask = _make_frame_transformer()
    original_launch = frame_transformer_module.wp.launch

    def launch_with_recording_failure(*args, record_cmd=False, **kwargs):
        if record_cmd:
            raise RuntimeError("recording failed")
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(frame_transformer_module.wp, "launch", launch_with_recording_failure)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert not sensor._use_recorded_launch
    torch.testing.assert_close(
        wp.to_torch(sensor._data._target_pos_source)[0, 0],
        torch.tensor([1.0, 0.0, 0.0], device=sensor.device),
    )

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(sensor.device)

    assert transform_view.get_count == 2
    assert transform_view.view_count == 1


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_frame_transformer_invalidation_drops_cached_launch_state(monkeypatch):
    """Physics invalidation should release the cached PhysX view and recorded command."""
    sensor, _, _, _ = _make_frame_transformer()
    sensor._raw_transforms = object()
    sensor._update_cmd = object()
    monkeypatch.setattr(BaseFrameTransformer, "_invalidate_initialize_callback", lambda self, event: None)

    sensor._invalidate_initialize_callback(None)

    assert sensor._frame_physx_view is None
    assert sensor._raw_transforms is None
    assert sensor._update_cmd is None
