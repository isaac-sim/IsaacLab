# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import scipy.spatial.transform as tf
import torch
import unittest

import isaacsim.core.utils.stage as stage_utils

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


class TestFrameTransformer(unittest.TestCase):
    """Test for frame transformer sensor."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Load kit helper
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu"))
        # Set main camera
        self.sim.set_camera_view(eye=[5, 5, 5], target=[0.0, 0.0, 0.0])

    def tearDown(self):
        """Stops simulator after each test."""
        # stop simulation
        # self.sim.stop()
        # clear the stage
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Tests
    """

    def test_frame_transformer_feet_wrt_base(self):
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
        self.sim.reset()

        # Acquire the index of ground truth bodies
        feet_indices, feet_names = scene.articulations["robot"].find_bodies(
            ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        )

        target_frame_names = scene.sensors["frame_transformer"].data.target_frame_names

        # Reorder the feet indices to match the order of the target frames with _USER suffix removed
        target_frame_names = [name.split("_USER")[0] for name in target_frame_names]

        # Find the indices of the feet in the order of the target frames
        reordering_indices = [feet_names.index(name) for name in target_frame_names]
        feet_indices = [feet_indices[i] for i in reordering_indices]

        # default joint targets
        default_actions = scene.articulations["robot"].data.default_joint_pos.clone()
        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
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
            self.sim.step()
            # read data from sim
            scene.update(sim_dt)

            # check absolute frame transforms in world frame
            # -- ground-truth
            root_pose_w = scene.articulations["robot"].data.root_state_w[:, :7]
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

    def test_frame_transformer_feet_wrt_thigh(self):
        """Test feet transformation w.r.t. thigh source frame.

        In this test, the source frame is the LF leg's thigh frame. This frame is not at index 0,
        when the frame bodies are sorted in the order of the regex matching in the frame transformer.
        """
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
        self.sim.reset()

        # Acquire the index of ground truth bodies
        source_frame_index = scene.articulations["robot"].find_bodies("LF_THIGH")[0][0]
        feet_indices, feet_names = scene.articulations["robot"].find_bodies(["LF_FOOT", "RF_FOOT"])
        # Check names are parsed the same order
        user_feet_names = [f"{name}_USER" for name in feet_names]
        self.assertListEqual(scene.sensors["frame_transformer"].data.target_frame_names, user_feet_names)

        # default joint targets
        default_actions = scene.articulations["robot"].data.default_joint_pos.clone()
        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
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
            self.sim.step()
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

    def test_frame_transformer_robot_body_to_external_cube(self):
        """Test transformation from robot body to a cube in the scene.

        In this test, the source frame is the robot base.

        The target_frame is a cube in the scene, external to the robot.
        """
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
        self.sim.reset()

        # default joint targets
        default_actions = scene.articulations["robot"].data.default_joint_pos.clone()
        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
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
            self.sim.step()
            # read data from sim
            scene.update(sim_dt)

            # check absolute frame transforms in world frame
            # -- ground-truth
            root_pose_w = scene.articulations["robot"].data.root_state_w[:, :7]
            cube_pos_w_gt = scene.rigid_objects["cube"].data.root_state_w[:, :3]
            cube_quat_w_gt = scene.rigid_objects["cube"].data.root_state_w[:, 3:7]
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

    def test_frame_transformer_offset_frames(self):
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
        self.sim.reset()

        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
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
            self.sim.step()
            # read data from sim
            scene.update(sim_dt)

            # check absolute frame transforms in world frame
            # -- ground-truth
            cube_pos_w_gt = scene["cube"].data.root_state_w[:, :3]
            cube_quat_w_gt = scene["cube"].data.root_state_w[:, 3:7]
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

    def test_frame_transformer_all_bodies(self):
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
        self.sim.reset()

        target_frame_names = scene.sensors["frame_transformer"].data.target_frame_names
        articulation_body_names = scene.articulations["robot"].data.body_names

        reordering_indices = [target_frame_names.index(name) for name in articulation_body_names]

        # default joint targets
        default_actions = scene.articulations["robot"].data.default_joint_pos.clone()
        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
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
            self.sim.step()
            # read data from sim
            scene.update(sim_dt)

            # check absolute frame transforms in world frame
            # -- ground-truth
            root_pose_w = scene.articulations["robot"].data.root_state_w[:, :7]
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

    def test_sensor_print(self):
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
        self.sim.reset()
        # print info
        print(scene.sensors["frame_transformer"])


if __name__ == "__main__":
    run_tests()
