# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Unit tests for NullSpacePostureTask with simplified robot configuration using Pink library directly."""

import numpy as np
import pinocchio as pin
import pytest
from pink import solve_ik
from pink.configuration import Configuration
from pink.tasks import FrameTask
from pinocchio.robot_wrapper import RobotWrapper

from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
from isaaclab.controllers.pink_ik.pink_task_cfg import NullSpacePostureTaskCfg


class TestNullSpacePostureTaskSimplifiedRobot:
    """Test cases for NullSpacePostureTask with simplified robot configuration."""

    @pytest.fixture
    def num_joints(self):
        """Number of joints in the simplified robot."""
        return 20

    @pytest.fixture
    def joint_configurations(self):
        """Pre-generated joint configurations for testing."""
        # Set random seed for reproducible tests
        np.random.seed(42)

        return {
            "random": np.random.uniform(-0.5, 0.5, 20),
            "controlled_only": np.array([0.5] * 5 + [0.0] * 15),  # Non-zero for controlled joints only
            "sequential": np.linspace(0.1, 2.0, 20),
        }

    @pytest.fixture
    def robot_urdf(self):
        """Load the simplified test robot URDF file."""
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(current_dir, "simplified_test_robot.urdf")
        return urdf_path

    @pytest.fixture
    def robot_configuration(self, robot_urdf):
        """Simplified robot wrapper."""
        wrapper = RobotWrapper.BuildFromURDF(robot_urdf, None, root_joint=None)
        return Configuration(wrapper.model, wrapper.data, wrapper.q0)

    @pytest.fixture
    def tasks(self):
        """pink tasks."""
        return [
            FrameTask("left_hand_pitch_link", position_cost=1.0, orientation_cost=1.0),
            NullSpacePostureTask(
                NullSpacePostureTaskCfg(
                    cost=1.0,
                    controlled_frames=["left_hand_pitch_link"],
                    controlled_joints=[
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                    ],
                )
            ),
        ]

    def test_null_space_jacobian_zero_end_effector_velocity(
        self, robot_configuration, tasks, joint_configurations, num_joints
    ):
        """Test that velocities projected through null space Jacobian result in zero end-effector velocity."""
        # Set specific joint configuration
        robot_configuration.q = joint_configurations["random"]

        # Set frame task target to a specific position in workspace
        frame_task = tasks[0]
        # Create pin.SE3 from position and quaternion
        position = np.array([0.5, 0.3, 0.8])  # x, y, z
        quaternion = pin.Quaternion(0.0, 0.0, 0.0, 1.0)  # x, y, z, w (identity quaternion)
        target_pose = pin.SE3(quaternion, position)
        frame_task.set_target(target_pose)

        # Set null space posture task target
        null_space_task = tasks[1]
        target_posture = np.zeros(num_joints)
        null_space_task.set_target(target_posture)

        # Get the null space Jacobian
        null_space_jacobian = null_space_task.compute_jacobian(robot_configuration)

        # Get the end-effector Jacobian
        frame_task_jacobian = frame_task.compute_jacobian(robot_configuration)

        # Test multiple random velocities in null space
        for _ in range(10):
            # Generate random joint velocity
            random_velocity = np.random.randn(num_joints) * 0.1

            # Project through null space Jacobian
            null_space_velocity = null_space_jacobian @ random_velocity

            # Compute resulting end-effector velocity
            ee_velocity = frame_task_jacobian @ null_space_velocity

            # The end-effector velocity should be approximately zero
            assert np.allclose(ee_velocity, np.zeros(6), atol=1e-7), f"End-effector velocity not zero: {ee_velocity}"

    def test_null_space_jacobian_properties(self, robot_configuration, tasks, joint_configurations, num_joints):
        """Test mathematical properties of the null space Jacobian."""
        # Set specific joint configuration
        robot_configuration.q = joint_configurations["random"]

        # Set frame task target
        frame_task = tasks[0]
        # Create pin.SE3 from position and quaternion
        position = np.array([0.3, 0.4, 0.6])
        quaternion = pin.Quaternion(0.707, 0.0, 0.0, 0.707)  # w, x, y, z (90-degree rotation around X)
        target_pose = pin.SE3(quaternion, position)
        frame_task.set_target(target_pose)

        # Set null space posture task target
        null_space_task = tasks[1]
        target_posture = np.zeros(num_joints)
        target_posture[0:5] = [0.1, -0.1, 0.2, -0.2, 0.0]  # Set first 5 joints (controlled joints)
        null_space_task.set_target(target_posture)

        # Get Jacobians
        null_space_jacobian = null_space_task.compute_jacobian(robot_configuration)
        ee_jacobian = robot_configuration.get_frame_jacobian("left_hand_pitch_link")

        # Test: N * J^T should be approximately zero (null space property)
        # where N is the null space projector and J is the end-effector Jacobian
        null_space_projection = null_space_jacobian @ ee_jacobian.T
        assert np.allclose(null_space_projection, np.zeros_like(null_space_projection), atol=1e-7), (
            f"Null space projection of end-effector Jacobian not zero: {null_space_projection}"
        )

    def test_null_space_jacobian_identity_when_no_frame_tasks(
        self, robot_configuration, joint_configurations, num_joints
    ):
        """Test that null space Jacobian is identity when no frame tasks are defined."""
        # Create null space task without frame task controlled joints
        null_space_task = NullSpacePostureTask(
            NullSpacePostureTaskCfg(cost=1.0, controlled_frames=[], controlled_joints=[])
        )

        # Set specific joint configuration
        robot_configuration.q = joint_configurations["sequential"]

        # Set target
        target_posture = np.zeros(num_joints)
        null_space_task.set_target(target_posture)

        # Get the null space Jacobian
        null_space_jacobian = null_space_task.compute_jacobian(robot_configuration)

        # Should be identity matrix
        expected_identity = np.eye(num_joints)
        assert np.allclose(null_space_jacobian, expected_identity), (
            f"Null space Jacobian should be identity when no frame tasks defined: {null_space_jacobian}"
        )

    def test_null_space_jacobian_consistency_across_configurations(
        self, robot_configuration, tasks, joint_configurations, num_joints
    ):
        """Test that null space Jacobian is consistent across different joint configurations."""
        # Test multiple joint configurations
        test_configs = [
            np.zeros(num_joints),  # Zero configuration
            joint_configurations["controlled_only"],  # Non-zero for controlled joints
            joint_configurations["random"],  # Random configuration
        ]

        # Set frame task target
        frame_task = tasks[0]
        # Create pin.SE3 from position and quaternion
        position = np.array([0.3, 0.3, 0.5])
        quaternion = pin.Quaternion(0.0, 0.0, 0.0, 1.0)  # x, y, z, w (identity quaternion)
        target_pose = pin.SE3(quaternion, position)
        frame_task.set_target(target_pose)

        # Set null space posture task target
        null_space_task = tasks[1]
        target_posture = np.zeros(num_joints)
        null_space_task.set_target(target_posture)

        jacobians = []
        for config in test_configs:
            robot_configuration.q = config
            jacobian = null_space_task.compute_jacobian(robot_configuration)
            jacobians.append(jacobian)

            # Verify that velocities through this Jacobian result in zero end-effector velocity
            ee_jacobian = robot_configuration.get_frame_jacobian("left_hand_pitch_link")

            # Test with random velocity
            random_velocity = np.random.randn(num_joints) * 0.1
            null_space_velocity = jacobian @ random_velocity
            ee_velocity = ee_jacobian @ null_space_velocity

            assert np.allclose(ee_velocity, np.zeros(6), atol=1e-7), (
                f"End-effector velocity not zero for configuration {config}: {ee_velocity}"
            )

    def test_compute_error_without_target(self, robot_configuration, joint_configurations):
        """Test that compute_error raises ValueError when no target is set."""
        null_space_task = NullSpacePostureTask(
            NullSpacePostureTaskCfg(
                cost=1.0,
                controlled_frames=["left_hand_pitch_link"],
                controlled_joints=["waist_yaw_joint", "waist_pitch_joint"],
            )
        )

        robot_configuration.q = joint_configurations["sequential"]

        # Should raise ValueError when no target is set
        with pytest.raises(ValueError, match="No posture target has been set"):
            null_space_task.compute_error(robot_configuration)

    def test_joint_masking(self, robot_configuration, joint_configurations, num_joints):
        """Test that joint mask correctly filters only controlled joints."""

        controlled_joint_names = ["waist_pitch_joint", "left_shoulder_pitch_joint", "left_elbow_pitch_joint"]

        # Create task with specific controlled joints
        null_space_task = NullSpacePostureTask(
            NullSpacePostureTaskCfg(
                cost=1.0, controlled_frames=["left_hand_pitch_link"], controlled_joints=controlled_joint_names
            )
        )

        # Find the joint indexes in robot_configuration.model.names.tolist()[1:]
        joint_names = robot_configuration.model.names.tolist()[1:]
        joint_indexes = [joint_names.index(jn) for jn in controlled_joint_names]

        # Set configurations
        current_config = joint_configurations["sequential"]
        target_config = np.zeros(num_joints)

        robot_configuration.q = current_config
        null_space_task.set_target(target_config)

        # Compute error
        error = null_space_task.compute_error(robot_configuration)

        # Only controlled joints should have non-zero error
        expected_error = np.zeros(num_joints)
        for i in joint_indexes:
            expected_error[i] = current_config[i]

        assert np.allclose(error, expected_error, atol=1e-7), (
            f"Joint mask not working correctly: expected {expected_error}, got {error}"
        )

    def test_empty_controlled_joints(self, robot_configuration, joint_configurations, num_joints):
        """Test behavior when controlled_joints is empty."""
        null_space_task = NullSpacePostureTask(
            NullSpacePostureTaskCfg(cost=1.0, controlled_frames=["left_hand_pitch_link"], controlled_joints=[])
        )

        current_config = joint_configurations["sequential"]
        target_config = np.zeros(num_joints)

        robot_configuration.q = current_config
        null_space_task.set_target(target_config)

        # Error should be all zeros
        error = null_space_task.compute_error(robot_configuration)
        expected_error = np.zeros(num_joints)
        assert np.allclose(error, expected_error), f"Error should be zero when no joints controlled: {error}"

    def test_set_target_from_configuration(self, robot_configuration, joint_configurations):
        """Test set_target_from_configuration method."""
        null_space_task = NullSpacePostureTask(
            NullSpacePostureTaskCfg(
                cost=1.0,
                controlled_frames=["left_hand_pitch_link"],
                controlled_joints=["waist_yaw_joint", "waist_pitch_joint"],
            )
        )

        # Set a specific configuration
        test_config = joint_configurations["sequential"]
        robot_configuration.q = test_config

        # Set target from configuration
        null_space_task.set_target_from_configuration(robot_configuration)

        # Verify target was set correctly
        assert null_space_task.target_q is not None
        assert np.allclose(null_space_task.target_q, test_config)

    def test_multiple_frame_tasks(self, robot_configuration, joint_configurations, num_joints):
        """Test null space projection with multiple frame tasks."""
        # Create task with multiple controlled frames
        null_space_task = NullSpacePostureTask(
            NullSpacePostureTaskCfg(
                cost=1.0,
                controlled_frames=["left_hand_pitch_link", "right_hand_pitch_link"],
                controlled_joints=["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
            )
        )

        current_config = joint_configurations["sequential"]
        robot_configuration.q = current_config

        # Get null space Jacobian
        null_space_jacobian = null_space_task.compute_jacobian(robot_configuration)

        # Get Jacobians for both frames
        jacobian_left_hand = robot_configuration.get_frame_jacobian("left_hand_pitch_link")
        jacobian_right_hand = robot_configuration.get_frame_jacobian("right_hand_pitch_link")

        # Test that null space velocities result in zero velocity for both frames
        for _ in range(5):
            random_velocity = np.random.randn(num_joints) * 0.1
            null_space_velocity = null_space_jacobian @ random_velocity

            # Check both frames
            ee_velocity_left = jacobian_left_hand @ null_space_velocity
            ee_velocity_right = jacobian_right_hand @ null_space_velocity

            assert np.allclose(ee_velocity_left, np.zeros(6), atol=1e-7), (
                "Left hand velocity not zero:\n",
                f"{ee_velocity_left}",
            )
            assert np.allclose(ee_velocity_right, np.zeros(6), atol=1e-7), (
                "Right hand velocity not zero:\n",
                f"{ee_velocity_right}",
            )

    def test_solve_ik_with_null_space_posture_task(self, robot_configuration, num_joints):
        """Test that solve_ik solution projected through frame Jacobians is close to zero.

        This test sets up tasks similar to the G1 upper body IK controller configuration:
        - FrameTasks for left and right hands at the robot's initial pose
        - NullSpacePostureTask with target posture offset by 1 radian from current configuration

        The IK solver should produce joint velocities that move toward the posture target
        while keeping the end-effector velocities close to zero (since frame tasks are already
        at their target poses).
        """
        # Define frame names for left and right hands
        left_hand_frame = "left_hand_pitch_link"
        right_hand_frame = "right_hand_pitch_link"

        # Define controlled joints similar to pink_controller_cfg.py
        controlled_joints = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "waist_yaw_joint",
            "waist_pitch_joint",
            "waist_roll_joint",
        ]

        # Start from zero configuration (robot's initial pose)
        robot_configuration.q = np.zeros(num_joints)

        # Create FrameTasks at the initial pose (similar to LocalFrameTask in pink_controller_cfg.py)
        left_hand_task = FrameTask(
            left_hand_frame,
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10.0,
            gain=0.5,
        )
        right_hand_task = FrameTask(
            right_hand_frame,
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10.0,
            gain=0.5,
        )

        # Set frame task targets to current pose (so they should have zero error)
        left_hand_task.set_target_from_configuration(robot_configuration)
        right_hand_task.set_target_from_configuration(robot_configuration)

        # Create NullSpacePostureTask with target posture offset by 1 radian
        null_space_task = NullSpacePostureTask(
            cost=0.5,
            lm_damping=1.0,
            gain=0.1,
            controlled_frames=[left_hand_frame, right_hand_frame],
            controlled_joints=controlled_joints,
        )

        # Set target posture to 1 radian for each joint (offset from current zero configuration)
        target_posture = np.ones(num_joints) * 1.0
        null_space_task.set_target(target_posture)

        # Collect all tasks
        tasks = [left_hand_task, right_hand_task, null_space_task]

        # Solve IK - should produce velocities that move toward posture target
        # while keeping frame positions stable
        dt = 0.01  # 10ms timestep
        velocity = solve_ik(robot_configuration, tasks, dt, solver="daqp")

        # Get Jacobians for both frame tasks
        jacobian_left_hand = robot_configuration.get_frame_jacobian(left_hand_frame)
        jacobian_right_hand = robot_configuration.get_frame_jacobian(right_hand_frame)

        # Compute end-effector velocities from the solution
        ee_velocity_left = jacobian_left_hand @ velocity
        ee_velocity_right = jacobian_right_hand @ velocity

        # The end-effector velocities should be close to zero since frame tasks
        # are already at their target poses
        assert np.allclose(ee_velocity_left, np.zeros(6), atol=1e-5), (
            "Left hand end-effector velocity not close to zero:\n",
            f"{ee_velocity_left}\nMax component: {np.max(np.abs(ee_velocity_left))}",
        )
        assert np.allclose(ee_velocity_right, np.zeros(6), atol=1e-5), (
            "Right hand end-effector velocity not close to zero:\n",
            f"{ee_velocity_right}\nMax component: {np.max(np.abs(ee_velocity_right))}",
        )

    def test_solve_ik_with_different_posture_offsets(self, robot_configuration, num_joints):
        """Test solve_ik with various posture offsets to verify consistent null space behavior.

        This test verifies that regardless of the posture target offset magnitude,
        the IK solution always keeps end-effector velocities close to zero when
        frame tasks are at their target poses.
        """
        left_hand_frame = "left_hand_pitch_link"
        right_hand_frame = "right_hand_pitch_link"

        controlled_joints = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "waist_yaw_joint",
            "waist_pitch_joint",
            "waist_roll_joint",
        ]

        # Test with different posture offsets
        posture_offsets = [0.1, 0.5, 1.0]

        for offset in posture_offsets:
            # Reset to zero configuration
            robot_configuration.q = np.zeros(num_joints)

            # Create FrameTasks at current pose
            left_hand_task = FrameTask(
                left_hand_frame,
                position_cost=8.0,
                orientation_cost=2.0,
                lm_damping=10.0,
                gain=0.5,
            )
            right_hand_task = FrameTask(
                right_hand_frame,
                position_cost=8.0,
                orientation_cost=2.0,
                lm_damping=10.0,
                gain=0.5,
            )

            left_hand_task.set_target_from_configuration(robot_configuration)
            right_hand_task.set_target_from_configuration(robot_configuration)

            # Create NullSpacePostureTask with varying offset
            null_space_task = NullSpacePostureTask(
                cost=0.5,
                lm_damping=1.0,
                gain=0.1,
                controlled_frames=[left_hand_frame, right_hand_frame],
                controlled_joints=controlled_joints,
            )
            target_posture = np.ones(num_joints) * offset
            null_space_task.set_target(target_posture)

            tasks = [left_hand_task, right_hand_task, null_space_task]

            # Solve IK
            dt = 0.01
            velocity = solve_ik(robot_configuration, tasks, dt, solver="daqp")

            # Get Jacobians and compute end-effector velocities
            jacobian_left_hand = robot_configuration.get_frame_jacobian(left_hand_frame)
            jacobian_right_hand = robot_configuration.get_frame_jacobian(right_hand_frame)

            ee_velocity_left = jacobian_left_hand @ velocity
            ee_velocity_right = jacobian_right_hand @ velocity

            # End-effector velocities should remain close to zero
            assert np.allclose(ee_velocity_left, np.zeros(6), atol=1e-5), (
                f"Offset {offset}: Left hand velocity not zero: {ee_velocity_left}"
            )
            assert np.allclose(ee_velocity_right, np.zeros(6), atol=1e-5), (
                f"Offset {offset}: Right hand velocity not zero: {ee_velocity_right}"
            )
