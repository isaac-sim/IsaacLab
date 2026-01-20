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

import pytest
import scipy.spatial.transform as tf
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


def quat_from_euler_rpy(roll, pitch, yaw, degrees=False):
    """Converts Euler XYZ to Quaternion (w, x, y, z)."""
    quat = tf.Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees).as_quat()
    return tuple(quat[[3, 0, 1, 2]].tolist())


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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 5)),
    )


@pytest.fixture
def sim():
    """Create a simulation context."""
    # Create a new stage
    sim_utils.create_new_stage()
    # Load kit helper
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu"))
    # Set main camera
    sim.set_camera_view(eye=(5.0, 5.0, 5.0), target=(0.0, 0.0, 0.0))
    yield sim
    # Cleanup
    sim.clear_all_callbacks()
    sim.clear_instance()


def test_frame_transformer_feet_wrt_base(sim):
    """Test feet transformations w.r.t. base source frame.

    In this test, the source frame is the robot base.
    """
    # Spawn things into stage
    scene_cfg = MySceneCfg(num_envs=32, env_spacing=5.0, lazy_sensor_update=False)
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
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Acquire the index of ground truth bodies
    feet_indices, feet_names = scene.articulations["robot"].find_bodies(["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])

    target_frame_names = scene.sensors["frame_transformer"].data.target_frame_names

    # Reorder the feet indices to match the order of the target frames with _USER suffix removed
    target_frame_names = [name.split("_USER")[0] for name in target_frame_names]

    # Find the indices of the feet in the order of the target frames
    reordering_indices = [feet_names.index(name) for name in target_frame_names]
    feet_indices = [feet_indices[i] for i in reordering_indices]

    # default joint targets
    default_actions = scene.articulations["robot"].data.default_joint_pos.clone()
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulate physics
    for count in range(100):
        # # reset
        if count % 25 == 0:
            # reset root state
            root_state = scene.articulations["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot"].data.default_joint_pos
            joint_vel = scene.articulations["robot"].data.default_joint_vel
            # -- set root state
            # -- robot
            scene.articulations["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # reset buffers
            scene.reset()

        # set joint targets
        robot_actions = default_actions + 0.5 * torch.randn_like(default_actions)
        scene.articulations["robot"].set_joint_position_target(robot_actions)
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim_dt)

        # check absolute frame transforms in world frame
        # -- ground-truth
        root_pose_w = scene.articulations["robot"].data.root_pose_w
        feet_pos_w_gt = scene.articulations["robot"].data.body_pos_w[:, feet_indices]
        feet_quat_w_gt = scene.articulations["robot"].data.body_quat_w[:, feet_indices]
        # -- frame transformer
        source_pos_w_tf = scene.sensors["frame_transformer"].data.source_pos_w
        source_quat_w_tf = scene.sensors["frame_transformer"].data.source_quat_w
        feet_pos_w_tf = scene.sensors["frame_transformer"].data.target_pos_w
        feet_quat_w_tf = scene.sensors["frame_transformer"].data.target_quat_w

        # check if they are same
        torch.testing.assert_close(root_pose_w[:, :3], source_pos_w_tf)
        torch.testing.assert_close(root_pose_w[:, 3:], source_quat_w_tf)
        torch.testing.assert_close(feet_pos_w_gt, feet_pos_w_tf)
        torch.testing.assert_close(feet_quat_w_gt, feet_quat_w_tf)

        # check if relative transforms are same
        feet_pos_source_tf = scene.sensors["frame_transformer"].data.target_pos_source
        feet_quat_source_tf = scene.sensors["frame_transformer"].data.target_quat_source
        for index in range(len(feet_indices)):
            # ground-truth
            foot_pos_b, foot_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:], feet_pos_w_tf[:, index], feet_quat_w_tf[:, index]
            )
            # check if they are same
            torch.testing.assert_close(feet_pos_source_tf[:, index], foot_pos_b)
            torch.testing.assert_close(feet_quat_source_tf[:, index], foot_quat_b)


def test_frame_transformer_feet_wrt_thigh(sim):
    """Test feet transformation w.r.t. thigh source frame."""
    # Spawn things into stage
    scene_cfg = MySceneCfg(num_envs=32, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
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
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Acquire the index of ground truth bodies
    source_frame_index = scene.articulations["robot"].find_bodies("LF_THIGH")[0][0]
    feet_indices, feet_names = scene.articulations["robot"].find_bodies(["LF_FOOT", "RF_FOOT"])
    # Check names are parsed the same order
    user_feet_names = [f"{name}_USER" for name in feet_names]
    assert scene.sensors["frame_transformer"].data.target_frame_names == user_feet_names

    # default joint targets
    default_actions = scene.articulations["robot"].data.default_joint_pos.clone()
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulate physics
    for count in range(100):
        # # reset
        if count % 25 == 0:
            # reset root state
            root_state = scene.articulations["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot"].data.default_joint_pos
            joint_vel = scene.articulations["robot"].data.default_joint_vel
            # -- set root state
            # -- robot
            scene.articulations["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # reset buffers
            scene.reset()

        # set joint targets
        robot_actions = default_actions + 0.5 * torch.randn_like(default_actions)
        scene.articulations["robot"].set_joint_position_target(robot_actions)
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim_dt)

        # check absolute frame transforms in world frame
        # -- ground-truth
        source_pose_w_gt = scene.articulations["robot"].data.body_state_w[:, source_frame_index, :7]
        feet_pos_w_gt = scene.articulations["robot"].data.body_pos_w[:, feet_indices]
        feet_quat_w_gt = scene.articulations["robot"].data.body_quat_w[:, feet_indices]
        # -- frame transformer
        source_pos_w_tf = scene.sensors["frame_transformer"].data.source_pos_w
        source_quat_w_tf = scene.sensors["frame_transformer"].data.source_quat_w
        feet_pos_w_tf = scene.sensors["frame_transformer"].data.target_pos_w
        feet_quat_w_tf = scene.sensors["frame_transformer"].data.target_quat_w
        # check if they are same
        torch.testing.assert_close(source_pose_w_gt[:, :3], source_pos_w_tf)
        torch.testing.assert_close(source_pose_w_gt[:, 3:], source_quat_w_tf)
        torch.testing.assert_close(feet_pos_w_gt, feet_pos_w_tf)
        torch.testing.assert_close(feet_quat_w_gt, feet_quat_w_tf)

        # check if relative transforms are same
        feet_pos_source_tf = scene.sensors["frame_transformer"].data.target_pos_source
        feet_quat_source_tf = scene.sensors["frame_transformer"].data.target_quat_source
        for index in range(len(feet_indices)):
            # ground-truth
            foot_pos_b, foot_quat_b = math_utils.subtract_frame_transforms(
                source_pose_w_gt[:, :3], source_pose_w_gt[:, 3:], feet_pos_w_tf[:, index], feet_quat_w_tf[:, index]
            )
            # check if they are same
            torch.testing.assert_close(feet_pos_source_tf[:, index], foot_pos_b)
            torch.testing.assert_close(feet_quat_source_tf[:, index], foot_quat_b)


def test_frame_transformer_robot_body_to_external_cube(sim):
    """Test transformation from robot body to a cube in the scene."""
    # Spawn things into stage
    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="CUBE_USER",
                prim_path="{ENV_REGEX_NS}/cube",
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # default joint targets
    default_actions = scene.articulations["robot"].data.default_joint_pos.clone()
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulate physics
    for count in range(100):
        # # reset
        if count % 25 == 0:
            # reset root state
            root_state = scene.articulations["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot"].data.default_joint_pos
            joint_vel = scene.articulations["robot"].data.default_joint_vel
            # -- set root state
            # -- robot
            scene.articulations["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # reset buffers
            scene.reset()

        # set joint targets
        robot_actions = default_actions + 0.5 * torch.randn_like(default_actions)
        scene.articulations["robot"].set_joint_position_target(robot_actions)
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim_dt)

        # check absolute frame transforms in world frame
        # -- ground-truth
        root_pose_w = scene.articulations["robot"].data.root_pose_w
        cube_pos_w_gt = scene.rigid_objects["cube"].data.root_pos_w
        cube_quat_w_gt = scene.rigid_objects["cube"].data.root_quat_w
        # -- frame transformer
        source_pos_w_tf = scene.sensors["frame_transformer"].data.source_pos_w
        source_quat_w_tf = scene.sensors["frame_transformer"].data.source_quat_w
        cube_pos_w_tf = scene.sensors["frame_transformer"].data.target_pos_w.squeeze()
        cube_quat_w_tf = scene.sensors["frame_transformer"].data.target_quat_w.squeeze()

        # check if they are same
        torch.testing.assert_close(root_pose_w[:, :3], source_pos_w_tf)
        torch.testing.assert_close(root_pose_w[:, 3:], source_quat_w_tf)
        torch.testing.assert_close(cube_pos_w_gt, cube_pos_w_tf)
        torch.testing.assert_close(cube_quat_w_gt, cube_quat_w_tf)

        # check if relative transforms are same
        cube_pos_source_tf = scene.sensors["frame_transformer"].data.target_pos_source
        cube_quat_source_tf = scene.sensors["frame_transformer"].data.target_quat_source
        # ground-truth
        cube_pos_b, cube_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:], cube_pos_w_tf, cube_quat_w_tf
        )
        # check if they are same
        torch.testing.assert_close(cube_pos_source_tf[:, 0], cube_pos_b)
        torch.testing.assert_close(cube_quat_source_tf[:, 0], cube_quat_b)


@pytest.mark.isaacsim_ci
def test_frame_transformer_offset_frames(sim):
    """Test body transformation w.r.t. base source frame.

    In this test, the source frame is the cube frame.
    """
    # Spawn things into stage
    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
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
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                name="CUBE_BOTTOM",
                prim_path="{ENV_REGEX_NS}/cube",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, -0.1),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulate physics
    for count in range(100):
        # # reset
        if count % 25 == 0:
            # reset root state
            root_state = scene["cube"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            # -- set root state
            # -- cube
            scene["cube"].write_root_pose_to_sim(root_state[:, :7])
            scene["cube"].write_root_velocity_to_sim(root_state[:, 7:])
            # reset buffers
            scene.reset()

        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim_dt)

        # check absolute frame transforms in world frame
        # -- ground-truth
        cube_pos_w_gt = scene["cube"].data.root_pos_w
        cube_quat_w_gt = scene["cube"].data.root_quat_w
        # -- frame transformer
        source_pos_w_tf = scene.sensors["frame_transformer"].data.source_pos_w
        source_quat_w_tf = scene.sensors["frame_transformer"].data.source_quat_w
        target_pos_w_tf = scene.sensors["frame_transformer"].data.target_pos_w.squeeze()
        target_quat_w_tf = scene.sensors["frame_transformer"].data.target_quat_w.squeeze()
        target_frame_names = scene.sensors["frame_transformer"].data.target_frame_names

        cube_center_idx = target_frame_names.index("CUBE_CENTER")
        cube_bottom_idx = target_frame_names.index("CUBE_BOTTOM")
        cube_top_idx = target_frame_names.index("CUBE_TOP")

        # check if they are same
        torch.testing.assert_close(cube_pos_w_gt, source_pos_w_tf)
        torch.testing.assert_close(cube_quat_w_gt, source_quat_w_tf)
        torch.testing.assert_close(cube_pos_w_gt, target_pos_w_tf[:, cube_center_idx])
        torch.testing.assert_close(cube_quat_w_gt, target_quat_w_tf[:, cube_center_idx])

        # test offsets are applied correctly
        # -- cube top
        cube_pos_top = target_pos_w_tf[:, cube_top_idx]
        cube_quat_top = target_quat_w_tf[:, cube_top_idx]
        torch.testing.assert_close(cube_pos_top, cube_pos_w_gt + torch.tensor([0.0, 0.0, 0.1]))
        torch.testing.assert_close(cube_quat_top, cube_quat_w_gt)

        # -- cube bottom
        cube_pos_bottom = target_pos_w_tf[:, cube_bottom_idx]
        cube_quat_bottom = target_quat_w_tf[:, cube_bottom_idx]
        torch.testing.assert_close(cube_pos_bottom, cube_pos_w_gt + torch.tensor([0.0, 0.0, -0.1]))
        torch.testing.assert_close(cube_quat_bottom, cube_quat_w_gt)


@pytest.mark.isaacsim_ci
def test_frame_transformer_all_bodies(sim):
    """Test transformation of all bodies w.r.t. base source frame.

    In this test, the source frame is the robot base.

    The target_frames are all bodies in the robot, implemented using .* pattern.
    """
    # Spawn things into stage
    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    target_frame_names = scene.sensors["frame_transformer"].data.target_frame_names
    articulation_body_names = scene.articulations["robot"].data.body_names

    reordering_indices = [target_frame_names.index(name) for name in articulation_body_names]

    # default joint targets
    default_actions = scene.articulations["robot"].data.default_joint_pos.clone()
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulate physics
    for count in range(100):
        # # reset
        if count % 25 == 0:
            # reset root state
            root_state = scene.articulations["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            joint_pos = scene.articulations["robot"].data.default_joint_pos
            joint_vel = scene.articulations["robot"].data.default_joint_vel
            # -- set root state
            # -- robot
            scene.articulations["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # reset buffers
            scene.reset()

        # set joint targets
        robot_actions = default_actions + 0.5 * torch.randn_like(default_actions)
        scene.articulations["robot"].set_joint_position_target(robot_actions)
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim_dt)

        # check absolute frame transforms in world frame
        # -- ground-truth
        root_pose_w = scene.articulations["robot"].data.root_pose_w
        bodies_pos_w_gt = scene.articulations["robot"].data.body_pos_w
        bodies_quat_w_gt = scene.articulations["robot"].data.body_quat_w

        # -- frame transformer
        source_pos_w_tf = scene.sensors["frame_transformer"].data.source_pos_w
        source_quat_w_tf = scene.sensors["frame_transformer"].data.source_quat_w
        bodies_pos_w_tf = scene.sensors["frame_transformer"].data.target_pos_w
        bodies_quat_w_tf = scene.sensors["frame_transformer"].data.target_quat_w

        # check if they are same
        torch.testing.assert_close(root_pose_w[:, :3], source_pos_w_tf)
        torch.testing.assert_close(root_pose_w[:, 3:], source_quat_w_tf)
        torch.testing.assert_close(bodies_pos_w_gt, bodies_pos_w_tf[:, reordering_indices])
        torch.testing.assert_close(bodies_quat_w_gt, bodies_quat_w_tf[:, reordering_indices])

        bodies_pos_source_tf = scene.sensors["frame_transformer"].data.target_pos_source
        bodies_quat_source_tf = scene.sensors["frame_transformer"].data.target_quat_source

        # Go through each body and check if relative transforms are same
        for index in range(len(articulation_body_names)):
            body_pos_b, body_quat_b = math_utils.subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:], bodies_pos_w_tf[:, index], bodies_quat_w_tf[:, index]
            )

            torch.testing.assert_close(bodies_pos_source_tf[:, index], body_pos_b)
            torch.testing.assert_close(bodies_quat_source_tf[:, index], body_quat_b)


@pytest.mark.isaacsim_ci
def test_sensor_print(sim):
    """Test sensor print is working correctly."""
    # Spawn things into stage
    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # print info
    print(scene.sensors["frame_transformer"])


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("source_robot", ["Robot", "Robot_1"])
@pytest.mark.parametrize("path_prefix", ["{ENV_REGEX_NS}", "/World"])
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
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{path_prefix}/Robot_1/RF_SHANK",
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
            root_state = scene.articulations["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene.articulations["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene.articulations["robot"].write_joint_state_to_sim(
                scene.articulations["robot"].data.default_joint_pos,
                scene.articulations["robot"].data.default_joint_vel,
            )
            # Reset robot_1
            root_state_1 = scene.articulations["robot_1"].data.default_root_state.clone()
            root_state_1[:, :3] += scene.env_origins
            scene.articulations["robot_1"].write_root_pose_to_sim(root_state_1[:, :7])
            scene.articulations["robot_1"].write_root_velocity_to_sim(root_state_1[:, 7:])
            scene.articulations["robot_1"].write_joint_state_to_sim(
                scene.articulations["robot_1"].data.default_joint_pos,
                scene.articulations["robot_1"].data.default_joint_vel,
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
        source_pos_w = frame_transformer_data.source_pos_w
        source_quat_w = frame_transformer_data.source_quat_w
        target_pos_w = frame_transformer_data.target_pos_w

        # Get ground truth positions and orientations (after scene.update() so they're current)
        robot_lf_pos_w = scene.articulations["robot"].data.body_pos_w[:, robot_lf_shank_body_idx]
        robot_1_lf_pos_w = scene.articulations["robot_1"].data.body_pos_w[:, robot_1_lf_shank_body_idx]
        robot_rf_pos_w = scene.articulations["robot"].data.body_pos_w[:, robot_rf_shank_body_idx]
        robot_1_rf_pos_w = scene.articulations["robot_1"].data.body_pos_w[:, robot_1_rf_shank_body_idx]

        # Get expected source frame positions and orientations (after scene.update() so they're current)
        expected_source_base_pos_w = scene.articulations[expected_source_robot].data.body_pos_w[
            :, expected_source_base_body_idx
        ]
        expected_source_base_quat_w = scene.articulations[expected_source_robot].data.body_quat_w[
            :, expected_source_base_body_idx
        ]

        # TEST 1: Verify source frame is correctly resolved
        # The source_pos_w should match the expected source robot's base world position
        torch.testing.assert_close(source_pos_w, expected_source_base_pos_w, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(source_quat_w, expected_source_base_quat_w, rtol=1e-5, atol=1e-5)

        # TEST 2: Explicit named frames (LF_SHANK) should have DIFFERENT world positions
        lf_pos_difference = torch.norm(target_pos_w[:, robot_lf_idx] - target_pos_w[:, robot_1_lf_idx], dim=-1)
        assert torch.all(lf_pos_difference > 1.0), (
            f"Robot_LF_SHANK and Robot_1_LF_SHANK should have different positions (got diff={lf_pos_difference}). "
            "This indicates body name collision bug."
        )

        # Verify explicit named frames match correct robot bodies
        torch.testing.assert_close(target_pos_w[:, robot_lf_idx], robot_lf_pos_w)
        torch.testing.assert_close(target_pos_w[:, robot_1_lf_idx], robot_1_lf_pos_w)

        # TEST 3: Implicit named frames (RF_SHANK) should also have DIFFERENT world positions
        # Even though they have the same frame name, internal body tracking uses full paths
        rf_pos_difference = torch.norm(
            target_pos_w[:, rf_shank_indices[0]] - target_pos_w[:, rf_shank_indices[1]], dim=-1
        )
        assert torch.all(rf_pos_difference > 1.0), (
            f"The two RF_SHANK frames should have different positions (got diff={rf_pos_difference}). "
            "This indicates body name collision bug in internal body tracking."
        )

        # Verify implicit named frames match correct robot bodies
        # Note: Order depends on internal processing, so we check both match one of the robots
        rf_positions = [target_pos_w[:, rf_shank_indices[0]], target_pos_w[:, rf_shank_indices[1]]]

        # Each tracked position should match one of the ground truth positions
        for rf_pos in rf_positions:
            matches_robot = torch.allclose(rf_pos, robot_rf_pos_w, atol=1e-5)
            matches_robot_1 = torch.allclose(rf_pos, robot_1_rf_pos_w, atol=1e-5)
            assert matches_robot or matches_robot_1, (
                f"RF_SHANK position {rf_pos} doesn't match either robot's RF_SHANK position"
            )
