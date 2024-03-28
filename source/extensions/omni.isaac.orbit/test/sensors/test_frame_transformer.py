# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script checks the FrameTransformer sensor by visualizing the frames that it creates.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import scipy.spatial.transform as tf
import torch
import unittest

import omni.isaac.core.utils.stage as stage_utils

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.sensors import FrameTransformerCfg, OffsetCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.orbit_assets.anymal import ANYMAL_C_CFG  # isort:skip


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


class TestFrameTransformer(unittest.TestCase):
    """Test for frame transformer sensor."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Load kit helper
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
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

        In this test, the source frame is the robot base. This frame is at index 0, when
        the frame bodies are sorted in the order of the regex matching in the frame transformer.
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
                scene.articulations["robot"].write_root_state_to_sim(root_state)
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
            torch.testing.assert_close(root_pose_w[:, :3], source_pos_w_tf, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(root_pose_w[:, 3:], source_quat_w_tf, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(feet_pos_w_gt, feet_pos_w_tf, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(feet_quat_w_gt, feet_quat_w_tf, rtol=1e-3, atol=1e-3)

            # check if relative transforms are same
            feet_pos_source_tf = scene.sensors["frame_transformer"].data.target_pos_source
            feet_quat_source_tf = scene.sensors["frame_transformer"].data.target_quat_source
            for index in range(len(feet_indices)):
                # ground-truth
                foot_pos_b, foot_quat_b = math_utils.subtract_frame_transforms(
                    root_pose_w[:, :3], root_pose_w[:, 3:], feet_pos_w_tf[:, index], feet_quat_w_tf[:, index]
                )
                # check if they are same
                torch.testing.assert_close(feet_pos_source_tf[:, index], foot_pos_b, rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(feet_quat_source_tf[:, index], foot_quat_b, rtol=1e-3, atol=1e-3)

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
                scene.articulations["robot"].write_root_state_to_sim(root_state)
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
            torch.testing.assert_close(source_pose_w_gt[:, :3], source_pos_w_tf, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(source_pose_w_gt[:, 3:], source_quat_w_tf, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(feet_pos_w_gt, feet_pos_w_tf, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(feet_quat_w_gt, feet_quat_w_tf, rtol=1e-3, atol=1e-3)

            # check if relative transforms are same
            feet_pos_source_tf = scene.sensors["frame_transformer"].data.target_pos_source
            feet_quat_source_tf = scene.sensors["frame_transformer"].data.target_quat_source
            for index in range(len(feet_indices)):
                # ground-truth
                foot_pos_b, foot_quat_b = math_utils.subtract_frame_transforms(
                    source_pose_w_gt[:, :3], source_pose_w_gt[:, 3:], feet_pos_w_tf[:, index], feet_quat_w_tf[:, index]
                )
                # check if they are same
                torch.testing.assert_close(feet_pos_source_tf[:, index], foot_pos_b, rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(feet_quat_source_tf[:, index], foot_quat_b, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    run_tests()
