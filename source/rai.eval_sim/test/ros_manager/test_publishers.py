# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import unittest

import numpy as np
import rclpy
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from rai.eval_sim.ros_manager import ObservationPublisherBase, publishers_cfg
from rai.eval_sim.ros_manager.ros_manager import QOS_PROFILE
from rai.eval_sim.tasks.testing.anymal_d_testing_env_cfg import AnymalDTestAllEnvCfg
from rclpy.node import Node


class TestPublishers(unittest.TestCase):
    """Test class for ROS2 publishers. These tests verify the correct population
    and transmission of each publisher message type.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize ROS 2 and create test environment."""
        rclpy.init()
        cls._pub_node = Node("PubTestNode")
        cls._sub_node = Node("SubTestNode")
        cls.env = ManagerBasedEnv(cfg=AnymalDTestAllEnvCfg())

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down ROS 2 and test environment"""
        rclpy.shutdown()
        cls._pub_node.destroy_node()
        cls._sub_node.destroy_node()
        cls.env.close()

    def tearDown(self):
        self._cleanup_pub_sub()

    """
    Testing fixture helpers
    """

    def _sub_callback(self, msg):
        """Helper function to attached to subscriber callback."""
        self.msg = msg

    def _setup_pub_sub(self, pub_cfg: publishers_cfg.PublisherBaseTermCfg):
        """Helper function to setup publisher and subscribers."""
        # reset environment
        self.env.reset()

        use_sim_time = True
        self.pub_term = pub_cfg.class_type(pub_cfg, self._pub_node, self.env, QOS_PROFILE, use_sim_time)
        self.msg = pub_cfg.msg_type()

        self.sub_term = self._sub_node.create_subscription(
            msg_type=pub_cfg.msg_type, topic=pub_cfg.topic, callback=self._sub_callback, qos_profile=QOS_PROFILE
        )
        self.pub_cfg = pub_cfg

    def _step(self):
        """Helper function to step environment and spin nodes."""
        # create actions of correct size
        act = torch.randn_like(self.env.action_manager.action)
        obs_dict, _ = self.env.step(action=act)
        self.assertTrue(isinstance(obs_dict, dict))
        # spin and publish messages
        if isinstance(self.pub_term, ObservationPublisherBase):
            self.pub_term.publish(obs_dict)
        else:
            self.pub_term.publish()

        rclpy.spin_once(self._pub_node, timeout_sec=0.0)
        # spin to subscribe
        rclpy.spin_once(self._sub_node, timeout_sec=0.0)

        # check message type
        self.assertTrue(isinstance(self.msg, self.pub_cfg.msg_type))

        return obs_dict

    def _cleanup_pub_sub(self):
        """Helper function to clean up publisher and subscriber."""
        self.pub_term.close()
        self._sub_node.destroy_subscription(self.sub_term)

    """
    Publisher Tests
    """

    def test_clock_pub(self) -> None:
        """Tests ClockPublisher"""
        # setup publisher
        pub_cfg = publishers_cfg.ClockPublisherCfg()
        self._setup_pub_sub(pub_cfg)

        # the ground truth dt in ms
        gt_dt = int(self.env.step_dt * 1e3)
        prev_ms = 0
        curr_ms = 0
        # run a few steps of the environment
        for i in range(10):
            _ = self._step()

            # tests
            self.assertTrue(isinstance(self.msg, pub_cfg.msg_type))
            self.assertTrue(self.msg.clock.sec is not None)
            self.assertTrue(self.msg.clock.nanosec is not None)
            # test proper values are being published
            # extract nanosec and sec parts of msg.clock and convert to milliseconds
            curr_ms = math.floor(1e-6 * self.msg.clock.nanosec) + math.floor(self.msg.clock.sec * 1e3)
            if i > 0:
                check_dt = curr_ms - prev_ms
                self.assertTrue(
                    abs(check_dt - gt_dt) <= 1, msg=f"Check_dt = {check_dt} and gt_dt = {gt_dt} at iteration i = {i}"
                )

            prev_ms = curr_ms

    def test_flatten_obs_pub(self):
        """Tests the FlattenedObsPublisherCfg version of the ObservationTermPublisher with AnymalD environment."""

        # setup publisher
        pub_cfg = publishers_cfg.FlattenedObsPublisherCfg(
            topic="/height_scan",
            obs_group="policy",
            obs_term_name="height_scan",
        )

        self._setup_pub_sub(pub_cfg)

        for _ in range(10):
            obs_dict = self._step()

            height_scan_obs = obs_dict[pub_cfg.obs_group][pub_cfg.obs_term_name]

            # test observation values
            self.assertEqual(self.msg.data.tolist(), height_scan_obs.squeeze().tolist())

    def test_vector3stamped_obs_pub(self):
        """Tests the Vector3StampedCfg version of the ObservationTermPublisher."""

        # setup publisher
        pub_cfg = publishers_cfg.Vector3StampedObsPublisherCfg(
            topic="/projected_gravity",
            obs_group="policy",
            obs_term_name="projected_gravity",
        )

        self._setup_pub_sub(pub_cfg)

        for _ in range(10):
            obs_dict = self._step()

            data = obs_dict[pub_cfg.obs_group][pub_cfg.obs_term_name].squeeze().tolist()

            # test observation values
            self.assertEqual(self.msg.vector.x, data[0])
            self.assertEqual(self.msg.vector.y, data[1])
            self.assertEqual(self.msg.vector.z, data[2])

    def test_joint_state_obs_pub(self):
        """Tests the JointStateObsPublisher with AnymalD environment."""
        # setup publisher
        pub_cfg = publishers_cfg.JointStateObsPublisherCfg(
            topic="/joint_state",
            obs_group="policy",
            position_obs="joint_pos",
            velocity_obs="joint_vel",
            effort_obs="joint_effort",
            asset_cfg=SceneEntityCfg("robot"),
        )

        self._setup_pub_sub(pub_cfg)

        # extract articulation
        robot = self.env.scene["robot"]

        for i in range(10):
            _ = self._step()

            # test joint state size
            self.assertEqual(len(self.msg.name), robot.num_joints)
            self.assertEqual(len(self.msg.position), robot.num_joints)
            self.assertEqual(len(self.msg.velocity), robot.num_joints)
            self.assertEqual(len(self.msg.effort), robot.num_joints)

            # test joint state values
            self.assertEqual(self.msg.name, robot.joint_names)
            self.assertEqual(self.msg.position.tolist(), robot.data.joint_pos.squeeze().tolist())
            self.assertEqual(self.msg.velocity.tolist(), robot.data.joint_vel.squeeze().tolist())
            self.assertEqual(self.msg.effort.tolist(), robot.data.applied_torque.squeeze().tolist())

    def test_link_pose_obs_pub(self):
        """Tests the LinkPoseObsPublisher with AnymalD environment."""
        # setup publisher
        pub_cfg = publishers_cfg.LinkPoseObsPublisherCfg(
            topic="/link_pose",
            obs_group="policy",
            link_pose_obs="base_link_pose",
        )

        self._setup_pub_sub(pub_cfg)

        # extract articulation
        robot = self.env.scene["robot"]
        # grab the link id of the base_link
        link_id = robot.body_names.index(robot.body_names[0])

        for i in range(10):
            _ = self._step()

            pose = robot.data.body_state_w[self.pub_term._env_idx, link_id, :7]
            pose[:3] = pose[:3] - self.env.scene.env_origins

            self.assertEqual(self.msg.pose.position.x, pose[0].item())
            self.assertEqual(self.msg.pose.position.y, pose[1].item())
            self.assertEqual(self.msg.pose.position.z, pose[2].item())
            self.assertEqual(self.msg.pose.orientation.w, pose[3].item())
            self.assertEqual(self.msg.pose.orientation.x, pose[4].item())
            self.assertEqual(self.msg.pose.orientation.y, pose[5].item())
            self.assertEqual(self.msg.pose.orientation.z, pose[6].item())

    def test_twist_obs_pub(self):
        """Tests the TwistObsPublisher with AnymalD environment."""
        self.env.reset()

        # setup publisher
        pub_cfg = publishers_cfg.TwistObsPublisherCfg(
            topic="/twist",
            obs_group="policy",
            lin_vel_obs="base_lin_vel",
            ang_vel_obs="base_ang_vel",
            frame_id="world",
        )

        self._setup_pub_sub(pub_cfg)

        # extract articulation
        robot = self.env.scene["robot"]

        for i in range(10):
            _ = self._step()

            lin_vel = robot.data.root_lin_vel_b[self.pub_term._env_idx].tolist()
            ang_vel = robot.data.root_ang_vel_b[self.pub_term._env_idx].tolist()

            self.assertEqual(self.msg.twist.linear.x, lin_vel[0])
            self.assertEqual(self.msg.twist.linear.y, lin_vel[1])
            self.assertEqual(self.msg.twist.linear.z, lin_vel[2])
            self.assertEqual(self.msg.twist.angular.x, ang_vel[0])
            self.assertEqual(self.msg.twist.angular.y, ang_vel[1])
            self.assertEqual(self.msg.twist.angular.z, ang_vel[2])

    def test_imu_obs_pub(self):
        """Test IMU observation publisher with no defined observations."""
        # setup publisher
        pub_cfg = publishers_cfg.ImuObsPublisherCfg(
            topic="/imu",
            obs_group="policy",
            imu_quat_obs="imu_quat",
            imu_ang_vel_obs="imu_ang_vel",
            imu_lin_acc_obs="imu_lin_acc",
        )

        self._setup_pub_sub(pub_cfg)

        # extract
        imu = self.env.scene["imu"]

        for i in range(10):
            _ = self._step()

            quat = imu.data.quat_w[self.pub_term._env_idx].tolist()
            ang_vel = imu.data.ang_vel_b[self.pub_term._env_idx].tolist()
            lin_acc = imu.data.lin_acc_b[self.pub_term._env_idx].tolist()

            self.assertEqual(self.msg.orientation.w, quat[0])
            self.assertEqual(self.msg.orientation.x, quat[1])
            self.assertEqual(self.msg.orientation.y, quat[2])
            self.assertEqual(self.msg.orientation.z, quat[3])
            self.assertEqual(self.msg.angular_velocity.x, ang_vel[0])
            self.assertEqual(self.msg.angular_velocity.y, ang_vel[1])
            self.assertEqual(self.msg.angular_velocity.z, ang_vel[2])
            self.assertEqual(self.msg.linear_acceleration.x, lin_acc[0])
            self.assertEqual(self.msg.linear_acceleration.y, lin_acc[1])
            self.assertEqual(self.msg.linear_acceleration.z, lin_acc[2])

    def test_imu_obs_empty_pub(self):
        """Test IMU observation publisher."""
        # setup publisher
        pub_cfg = publishers_cfg.ImuObsPublisherCfg(
            topic="/imu_empty",
            obs_group="policy",
        )

        self._setup_pub_sub(pub_cfg)

        for i in range(10):
            _ = self._step()

            self.assertEqual(self.msg.orientation.w, 0.0)
            self.assertEqual(self.msg.orientation.x, 0.0)
            self.assertEqual(self.msg.orientation.y, 0.0)
            self.assertEqual(self.msg.orientation.z, 0.0)
            self.assertEqual(self.msg.angular_velocity.x, 0.0)
            self.assertEqual(self.msg.angular_velocity.y, 0.0)
            self.assertEqual(self.msg.angular_velocity.z, 0.0)
            self.assertEqual(self.msg.linear_acceleration.x, 0.0)
            self.assertEqual(self.msg.linear_acceleration.y, 0.0)
            self.assertEqual(self.msg.linear_acceleration.z, 0.0)

    def test_height_scan_pub(self):
        """Test height scan publisher with no defined observations."""
        # setup publisher

        pub_cfg = publishers_cfg.HeightMapPublisherCfg(
            asset_cfg=SceneEntityCfg(name="grid_map"),
            topic="/elevation_map",
            layer="elevation",
        )

        # set up publisher and advance one step
        self._setup_pub_sub(pub_cfg)
        self._step()

        # extract height scan
        grid_map = self.env.scene["grid_map"]
        height_scan = grid_map.data.ray_hits_w[:, :, 2]

        # check size
        self.assertEqual(height_scan.shape[1], len(self.msg.data[0].data))
        # check one element (order is reversed)
        self.assertEqual(height_scan[0, 0], self.msg.data[0].data[-1])
        # check norm (tis is independen of the ordering)
        self.assertEqual(torch.norm(height_scan), np.linalg.norm(self.msg.data[0].data, ord=2))
        # check layer and frame id
        self.assertEqual(self.msg.layers[0], "elevation")
        self.assertEqual(self.msg.header.frame_id, "world")

    def test_wrench_stamped_pub(self):
        """Test the Wrench observation publisher."""

        pub_cfg = publishers_cfg.WrenchStampedObsPublisherCfg(
            topic="/wrench/root", obs_group="policy", obs_term_name="RF_foot_reaction"
        )

        # set up publisher and advance one step
        self._setup_pub_sub(pub_cfg)
        self._step()

        # extract articulation
        robot = self.env.scene["robot"]

        body_id, _ = robot.find_bodies(name_keys="RF_FOOT", preserve_order=True)

        for i in range(10):
            _ = self._step()

            wrench = robot.data.body_joint_reaction_wrench_b[0, body_id, :].view(-1).tolist()

            self.assertEqual(self.msg.wrench.force.x, wrench[0])
            self.assertEqual(self.msg.wrench.force.y, wrench[1])
            self.assertEqual(self.msg.wrench.force.z, wrench[2])
            self.assertEqual(self.msg.wrench.torque.x, wrench[3])
            self.assertEqual(self.msg.wrench.torque.y, wrench[4])
            self.assertEqual(self.msg.wrench.torque.z, wrench[5])

    def test_joint_reaction_wrench_pub(self):
        """Test the JointReactionWrenchObsPublisherCfg."""

        pub_cfg = publishers_cfg.JointReactionWrenchObsPublisherCfg(
            topic="/wrench/all", obs_group="policy", obs_term_name="joint_reactions"
        )

        # set up publisher and advance one step
        self._setup_pub_sub(pub_cfg)
        self._step()

        # extract articulation
        robot = self.env.scene[pub_cfg.asset_cfg.name]

        for i in range(10):
            _ = self._step()

            wrenches = robot.data.body_joint_reaction_wrench_b.view(-1).tolist()
            self.assertEqual(wrenches, self.msg.data.tolist()[1:])

    def test_contact_force_publisher_all(self):
        """Test ContactForcePublisher with AnymalD with all collision bodies."""
        # Setup publisher
        pub_cfg = publishers_cfg.ContactForcePublisherCfg(
            topic="/contact_forces",
            asset_cfg=SceneEntityCfg("contact_forces"),
        )

        self._setup_pub_sub(pub_cfg)

        sensor = self.env.scene["contact_forces"]
        for i in range(10):
            _ = self._step()

            # check msg data
            self.assertEqual(len(self.msg.data), 3 * len(sensor.body_names))
            self.assertEqual(
                self.msg.data.tolist(),
                sensor.data.net_forces_w[:, self.pub_term.cfg.asset_cfg.body_ids, :].flatten().tolist(),
            )

    def test_contact_force_publisher_subset(self):
        """Test ContactForcePublisher with AnymalD for a subset of collision bodies."""
        # Setup publisher
        pub_cfg = publishers_cfg.ContactForcePublisherCfg(
            topic="/contact_forces",
            asset_cfg=SceneEntityCfg(
                name="contact_forces",
                body_names=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
            ),
        )

        self._setup_pub_sub(pub_cfg)

        sensor = self.env.scene["contact_forces"]

        for i in range(10):
            _ = self._step()

            # check msg data
            self.assertEqual(len(self.msg.data), 3 * len(self.pub_term.cfg.asset_cfg.body_names))
            self.assertEqual(
                self.msg.data.tolist(),
                sensor.data.net_forces_w[:, self.pub_term.cfg.asset_cfg.body_ids, :].flatten().tolist(),
            )


if __name__ == "__main__":
    run_tests()
