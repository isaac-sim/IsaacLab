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

import unittest

import numpy as np
import torch
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    PointStamped,
    Pose,
    PoseStamped,
    Quaternion,
    QuaternionStamped,
    TransformStamped,
    Twist,
    TwistStamped,
    Vector3,
    Vector3Stamped,
)
from rai.eval_sim.utils import ros_conversions
from sensor_msgs.msg import Image, Imu, JointState
from std_msgs.msg import Float32MultiArray, Header


def create_test_stamp() -> Time:
    test_stamp = Time()
    test_stamp.sec = 10
    test_stamp.nanosec = 2000
    return test_stamp


def create_test_header() -> Header:
    test_stamp = create_test_stamp()
    test_header = Header()
    test_header.stamp = test_stamp
    test_header.frame_id = "test"
    return test_header


class TestRosConversions(unittest.TestCase):
    """Test class for ROS2/Torch tensor conversions."""

    """
    ROS To Torch
    """

    def test_vector3_to_torch(self):
        """Test vector3_to_torch function"""
        # test nonstamped version
        msg_in = Vector3()
        msg_in.x = 1.0
        msg_in.y = 2.0
        msg_in.z = 3.0
        tensor_out_expected = torch.tensor([1.0, 2.0, 3.0])
        tensor_out = torch.empty(3)
        ros_conversions.vector3_to_torch(tensor_out, msg_in)

        torch.testing.assert_close(tensor_out_expected, tensor_out)

        # test stamped version
        msg_in = Vector3Stamped()
        msg_in.vector.x = 1.0
        msg_in.vector.y = 2.0
        msg_in.vector.z = 3.0
        msg_in.header = create_test_header()

        tensor_out = torch.empty(3)
        ros_conversions.vector3_to_torch(tensor_out, msg_in)

        torch.testing.assert_close(tensor_out_expected, tensor_out)

    def test_point_to_torch(self):
        """Test point_to_torch function"""
        # test nonstamped version
        msg_in = Point()
        msg_in.x = 1.0
        msg_in.y = 2.0
        msg_in.z = 3.0
        tensor_out_expected = torch.tensor([1.0, 2.0, 3.0])
        tensor_out = torch.empty(3)
        ros_conversions.point_to_torch(tensor_out, msg_in)

        torch.testing.assert_close(tensor_out_expected, tensor_out)

        # test stamped version
        msg_in = PointStamped()
        msg_in.point.x = 1.0
        msg_in.point.y = 2.0
        msg_in.point.z = 3.0
        msg_in.header = create_test_header()

        tensor_out = torch.empty(3)
        ros_conversions.point_to_torch(tensor_out, msg_in)

        torch.testing.assert_close(tensor_out_expected, tensor_out)

    def test_joint_state_to_torch(self):
        """Test joint_state_to_torch function."""
        msg_in = JointState()
        msg_in.name = ["joint1", "joint2", "joint3"]
        msg_in.position = [0.1, 0.2, 0.3]
        msg_in.velocity = [1.0, 2.0, 3.0]
        msg_in.effort = [0.01, 0.02, 0.03]
        msg_in.header = create_test_header()

        tensor_out_expected = torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [0.01, 0.02, 0.03]])
        tensor_out = torch.empty((3, 3))
        ros_conversions.joint_state_to_torch(tensor_out, msg_in)

        torch.testing.assert_close(tensor_out_expected, tensor_out)

    def test_quaternion_to_torch(self):
        """Test quaternion_to_torch function."""
        # test unstamped version
        msg_in = Quaternion()
        msg_in.w = 0.9479013
        msg_in.x = 0.0769743
        msg_in.y = 0.0805592
        msg_in.z = 0.2984431

        tensor_out_expected = torch.tensor([0.9479013, 0.0769743, 0.0805592, 0.2984431])
        tensor_out = torch.empty(4)
        ros_conversions.quaternion_to_torch(tensor_out, msg_in)
        torch.testing.assert_close(tensor_out_expected, tensor_out)

        # test stamped version
        msg_in_stamped = QuaternionStamped()
        msg_in_stamped.quaternion = msg_in
        msg_in_stamped.header = create_test_header()

        tensor_out = torch.empty(4)
        ros_conversions.quaternion_to_torch(tensor_out, msg_in_stamped)
        torch.testing.assert_close(tensor_out_expected, tensor_out)

    def test_pose_to_torch(self):
        """Test pose_to_torch function."""
        # test unstamped version
        msg_in = Pose()
        msg_in.position.x = 1.0
        msg_in.position.y = 2.0
        msg_in.position.z = 3.0
        msg_in.orientation.w = 0.9479013
        msg_in.orientation.x = 0.0769743
        msg_in.orientation.y = 0.0805592
        msg_in.orientation.z = 0.2984431

        tensor_out_expected = torch.tensor([1.0, 2.0, 3.0, 0.9479013, 0.0769743, 0.0805592, 0.2984431])
        tensor_out = torch.empty(7)
        ros_conversions.pose_to_torch(tensor_out, msg_in)
        torch.testing.assert_close(tensor_out_expected, tensor_out)

        # test stamped version
        msg_in_stamped = PoseStamped()
        msg_in_stamped.pose = msg_in
        tensor_out = torch.empty(7)
        ros_conversions.pose_to_torch(tensor_out, msg_in)
        torch.testing.assert_close(tensor_out_expected, tensor_out)

    def test_twist_to_torch(self):
        """Test twist_to_torch function."""
        # test unstamped version
        msg_in = Twist()
        msg_in.linear.x = 1.0
        msg_in.linear.y = 2.0
        msg_in.linear.z = 3.0
        msg_in.angular.x = 4.0
        msg_in.angular.y = 5.0
        msg_in.angular.z = 6.0

        tensor_out_expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        tensor_out = torch.empty(6)
        ros_conversions.twist_to_torch(tensor_out, msg_in)
        torch.testing.assert_close(tensor_out_expected, tensor_out)

        # test stamped version
        msg_in_stamped = TwistStamped()
        msg_in_stamped.twist = msg_in
        tensor_out = torch.empty(6)
        ros_conversions.twist_to_torch(tensor_out, msg_in)
        torch.testing.assert_close(tensor_out_expected, tensor_out)

    """
    Torch to ROS
    """

    def test_torch_to_vector3(self):
        """Test torch_to_vector3 function."""
        # test unstamped version
        tensor_in = torch.tensor([1.0, 2.0, 3.0])
        msg_out_expected = Vector3()
        msg_out_expected.x = 1.0
        msg_out_expected.y = 2.0
        msg_out_expected.z = 3.0
        msg_out = Vector3()
        ros_conversions.torch_to_vector3(msg_out, tensor_in)
        self.assertTrue(msg_out == msg_out_expected)

        # test stamped version
        msg_out_expected_stamped = Vector3Stamped()
        msg_out_expected_stamped.vector = msg_out_expected
        msg_out_expected_stamped.header = create_test_header()
        msg_out = Vector3Stamped()
        ros_conversions.torch_to_vector3(msg_out, tensor_in, time=create_test_stamp(), frame_id="test")
        self.assertTrue(msg_out == msg_out_expected_stamped)

    def test_torch_to_point(self):
        """Test torch_to_point function."""
        # test unstamped version
        tensor_in = torch.tensor([1.0, 2.0, 3.0])
        msg_out_expected = Point()
        msg_out_expected.x = 1.0
        msg_out_expected.y = 2.0
        msg_out_expected.z = 3.0
        msg_out = Point()
        ros_conversions.torch_to_point(msg_out, tensor_in)
        self.assertTrue(msg_out == msg_out_expected)

        # test stamped version
        msg_out_expected_stamped = PointStamped()
        msg_out_expected_stamped.point = msg_out_expected
        msg_out_expected_stamped.header = create_test_header()
        msg_out = PointStamped()
        ros_conversions.torch_to_point(msg_out, tensor_in, time=create_test_stamp(), frame_id="test")
        self.assertTrue(msg_out == msg_out_expected_stamped)

    def test_torch_to_quat(self):
        """Test torch_to_quat function."""
        # test unstamped version
        tensor_in = torch.tensor([1.0, 2.0, 3.0, 4.0])
        msg_out_expected = Quaternion()
        msg_out_expected.w = 1.0
        msg_out_expected.x = 2.0
        msg_out_expected.y = 3.0
        msg_out_expected.z = 4.0
        msg_out = Quaternion()
        ros_conversions.torch_to_quat(msg_out, tensor_in)
        self.assertTrue(msg_out == msg_out_expected)

        # test stamped version
        msg_out_expected_stamped = QuaternionStamped()
        msg_out_expected_stamped.quaternion = msg_out_expected
        msg_out_expected_stamped.header = create_test_header()
        msg_out = QuaternionStamped()
        ros_conversions.torch_to_quat(msg_out, tensor_in, time=create_test_stamp(), frame_id="test")
        self.assertTrue(msg_out == msg_out_expected_stamped)

    def test_torch_to_float32_multi_array(self):
        """Test torch_to_float32_multi_array function."""
        tensor_in = torch.tensor([0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 0.01, 0.02, 0.03])
        msg_out_expected = Float32MultiArray()
        msg_out_expected.data = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 0.01, 0.02, 0.03]
        msg_out = Float32MultiArray()
        ros_conversions.torch_to_float32_multi_array(msg_out, tensor_in)
        self.assertTrue(msg_out == msg_out_expected)

    def test_torch_to_imu(self):
        """Test torch_to_imu function."""
        tensor_in = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        msg_out_expected = Imu()
        msg_out_expected.orientation.w = 1.0
        msg_out_expected.orientation.x = 2.0
        msg_out_expected.orientation.y = 3.0
        msg_out_expected.orientation.z = 4.0
        msg_out_expected.angular_velocity.x = 5.0
        msg_out_expected.angular_velocity.y = 6.0
        msg_out_expected.angular_velocity.z = 7.0
        msg_out_expected.linear_acceleration.x = 8.0
        msg_out_expected.linear_acceleration.y = 9.0
        msg_out_expected.linear_acceleration.z = 10.0
        msg_out_expected.header = create_test_header()
        msg_out = Imu()
        ros_conversions.torch_to_imu(msg_out, tensor_in, time=create_test_stamp(), frame_id="test")
        self.assertTrue(msg_out == msg_out_expected)

    def test_torch_to_joint_state(self):
        """test torch_to_joint_state function."""
        tensor_in = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        msg_out_expected = JointState()
        msg_out_expected.position = [1.0, 2.0, 3.0]
        msg_out_expected.velocity = [4.0, 5.0, 6.0]
        msg_out_expected.effort = [7.0, 8.0, 9.0]
        msg_out_expected.header = create_test_header()
        msg_out = JointState()
        ros_conversions.torch_to_joint_state(msg_out, tensor_in, time=create_test_stamp(), frame_id="test")
        self.assertTrue(msg_out == msg_out_expected)

    def test_torch_to_pose(self):
        """Test torch_to_pose function."""
        # test unstamped version
        tensor_in = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        msg_out_expected = Pose()
        msg_out_expected.position.x = 1.0
        msg_out_expected.position.y = 2.0
        msg_out_expected.position.z = 3.0
        msg_out_expected.orientation.w = 4.0
        msg_out_expected.orientation.x = 5.0
        msg_out_expected.orientation.y = 6.0
        msg_out_expected.orientation.z = 7.0
        msg_out = Pose()
        ros_conversions.torch_to_pose(msg_out, tensor_in)
        self.assertTrue(msg_out == msg_out_expected)

        # test stamped version
        msg_out_expected_stamped = PoseStamped()
        msg_out_expected_stamped.pose = msg_out_expected
        msg_out_expected_stamped.header = create_test_header()
        msg_out = PoseStamped()
        ros_conversions.torch_to_pose(msg_out, tensor_in, time=create_test_stamp(), frame_id="test")
        self.assertTrue(msg_out == msg_out_expected_stamped)

    def test_torch_to_transform(self):
        """Test torch_to_pose function."""
        # test unstamped version
        tensor_in = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        msg_out_expected = TransformStamped()
        msg_out_expected.transform.translation.x = 1.0
        msg_out_expected.transform.translation.y = 2.0
        msg_out_expected.transform.translation.z = 3.0
        msg_out_expected.transform.rotation.w = 4.0
        msg_out_expected.transform.rotation.x = 5.0
        msg_out_expected.transform.rotation.y = 6.0
        msg_out_expected.transform.rotation.z = 7.0
        msg_out_expected.header = create_test_header()
        msg_out_expected.child_frame_id = "child"
        msg_out = TransformStamped()
        ros_conversions.torch_to_transform(
            msg_out, tensor_in, time=create_test_stamp(), frame_id="test", child_frame="child"
        )
        self.assertTrue(msg_out == msg_out_expected)

    def test_torch_to_twist(self):
        """Test torch_to_twist function."""
        # test unstamped version
        tensor_in = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        msg_out_expected = Twist()
        msg_out_expected.linear.x = 1.0
        msg_out_expected.linear.y = 2.0
        msg_out_expected.linear.z = 3.0
        msg_out_expected.angular.x = 4.0
        msg_out_expected.angular.y = 5.0
        msg_out_expected.angular.z = 6.0
        msg_out = Twist()
        ros_conversions.torch_to_twist(msg_out, tensor_in)
        self.assertTrue(msg_out == msg_out_expected)

        # test stamped version
        msg_out_expected_stamped = TwistStamped()
        msg_out_expected_stamped.twist = msg_out_expected
        msg_out_expected_stamped.header = create_test_header()
        msg_out = TwistStamped()
        ros_conversions.torch_to_twist(msg_out, tensor_in, time=create_test_stamp(), frame_id="test")
        self.assertTrue(msg_out == msg_out_expected_stamped)

    def test_torch_to_rgb(self):
        """Test torch_to_rgb function."""
        tensor_in = torch.empty((6, 6, 3))
        tensor_in[..., 0] = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.uint8).repeat(6, 1)
        tensor_in[..., 1] = torch.tensor([7, 8, 9, 10, 11, 12], dtype=torch.uint8).repeat(6, 1)
        tensor_in[..., 2] = torch.tensor(
            [
                13,
                14,
                15,
                16,
                17,
                18,
            ],
            dtype=torch.uint8,
        ).repeat(6, 1)

        # test rgb8 encoding
        tensor_in_rgb = tensor_in.clone()
        msg_out_expected = Image()
        msg_out_expected.header = create_test_header()
        msg_out_expected.height = 6
        msg_out_expected.width = 6
        msg_out_expected.encoding = "rgb8"
        msg_out_expected.step = 18
        msg_out_expected.is_bigendian = 0
        rgb_data = np.uint8(tensor_in_rgb.view(-1).numpy()).tolist()
        msg_out_expected.data = rgb_data
        msg_out = Image()
        ros_conversions.torch_to_rgb(msg_out, tensor_in, time=create_test_stamp(), frame_id="test")
        self.assertTrue(msg_out == msg_out_expected)

        # test bgr8 encoding
        tensor_in_bgr = tensor_in.clone()
        msg_out_expected.encoding = "bgr8"
        bgr_data = np.uint8(torch.flip(tensor_in_bgr, dims=(-1,)).view(-1).numpy()).tolist()
        msg_out_expected.data = bgr_data
        msg_out = Image()
        ros_conversions.torch_to_rgb(msg_out, tensor_in, time=create_test_stamp(), frame_id="test", encoding="bgr8")
        self.assertTrue(msg_out == msg_out_expected)

    def test_torch_to_depth(self):
        """Test torch_to_depth function."""
        tensor_in = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.uint8).repeat(6, 1)
        msg_out_expected = Image()
        msg_out_expected.header = create_test_header()
        msg_out_expected.height = 6
        msg_out_expected.width = 6
        msg_out_expected.encoding = "mono8"
        msg_out_expected.step = 6
        msg_out_expected.is_bigendian = 0
        tensor_in_mono8 = tensor_in.clone()
        threshold = (2, 4)
        scale = 2.0
        mono_data = np.uint8(torch.clamp(tensor_in_mono8, min=2, max=4).view(-1).numpy() * scale).tolist()
        msg_out_expected.data = mono_data
        msg_out = Image()
        ros_conversions.torch_to_depth(
            msg_out,
            tensor_in,
            time=create_test_stamp(),
            frame_id="test",
            encoding="mono8",
            threshold=threshold,
            scale=scale,
        )
        self.assertTrue(msg_out == msg_out_expected)


if __name__ == "__main__":
    run_tests()
