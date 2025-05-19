# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from functools import wraps

import cv2
import numpy as np
import torch
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
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
    Wrench,
    WrenchStamped,
)
from sensor_msgs.msg import Image, Imu, JointState
from std_msgs.msg import Float32MultiArray


##
# Timestamp decorator wrapper
#
def apply_header_data(func):
    """
    Decorator function to apply a timestamp to a ROS2 message Header.
    """

    @wraps(func)
    def wrapper(msg, obs, time=None, frame_id="", **kwargs):
        # time is passed to the inner function for message conversion
        func(msg, obs, time=time, **kwargs)
        # update message header stamp with the given time
        if time:
            msg.header.stamp = time
        if frame_id:
            msg.header.frame_id = frame_id

    return wrapper


##
# From Ros Message Conversions
##


def vector3_to_torch(msg_converted: torch.tensor, msg: Vector3 | Vector3Stamped) -> None:
    """Convert ROS2 Vector3 message to torch tensor.

    Args:
        msg_converted: the tensor to convert the incoming ROS message into.
        msg: the incoming ROS message to convert into a tensor.
    """
    if type(msg) is Vector3Stamped:
        msg_converted[0] = msg.vector.x
        msg_converted[1] = msg.vector.y
        msg_converted[2] = msg.vector.z
    else:
        msg_converted[0] = msg.x
        msg_converted[1] = msg.y
        msg_converted[2] = msg.z


def point_to_torch(msg_converted: torch.tensor, msg: Point | PointStamped) -> None:
    """Convert ROS2 Point message to torch tensor

    Args:
        msg_converted: the tensor to convert the incoming ROS message into.
        msg: the incoming ROS message to convert into a tensor.
    """
    if type(msg) is PointStamped:
        msg_converted[0] = msg.point.x
        msg_converted[1] = msg.point.y
        msg_converted[2] = msg.point.z
    else:
        msg_converted[0] = msg.x
        msg_converted[1] = msg.y
        msg_converted[2] = msg.z


def joint_state_to_torch(msg_converted: torch.tensor, msg: JointState) -> None:
    """Convert ROS2 JointState message to torch tensor

    Args:
        msg_converted: the tensor to convert the incoming ROS message into.
        msg: the incoming ROS message to convert into a tensor.
    """
    msg_converted[0] = torch.tensor(msg.position.tolist(), dtype=torch.float32)
    msg_converted[1] = torch.tensor(msg.velocity.tolist(), dtype=torch.float32)
    msg_converted[2] = torch.tensor(msg.effort.tolist(), dtype=torch.float32)


def quaternion_to_torch(msg_converted: torch.tensor, msg: Quaternion | QuaternionStamped) -> None:
    """Convert ROS2 Quaternion message to torch tensor

    Args:
        msg_converted: the tensor to convert the incoming ROS message into.
        msg: the incoming ROS message to convert into a tensor.
    """
    if type(msg) is QuaternionStamped:
        msg_converted[0] = msg.quaternion.w
        msg_converted[1] = msg.quaternion.x
        msg_converted[2] = msg.quaternion.y
        msg_converted[3] = msg.quaternion.z
    else:
        msg_converted[0] = msg.w
        msg_converted[1] = msg.x
        msg_converted[2] = msg.y
        msg_converted[3] = msg.z


def pose_to_torch(msg_converted: torch.tensor, msg: Pose | PoseStamped) -> None:
    """Convert ROS2 Pose or PoseStamped message to torch tensor.

    Args:
        msg_converted: the tensor to convert the incoming ROS message into.
        msg: the incoming ROS message to convert into a tensor.
    """
    if type(msg) is PoseStamped:
        position_msg = msg.pose.position
        quaternion_msg = msg.pose.orientation
    else:
        position_msg = msg.position
        quaternion_msg = msg.orientation

    vector3_to_torch(msg_converted=msg_converted[0:3], msg=position_msg)
    quaternion_to_torch(msg_converted=msg_converted[3:], msg=quaternion_msg)


def twist_to_torch(msg_converted: torch.tensor, msg: Twist | TwistStamped) -> None:
    """Convert ROS2 Twist message types to torch tensor

    Args:
        msg_converted: the tensor to convert the incoming ROS message into.
        msg: the incoming ROS message to convert into a tensor.
    """
    if type(msg) is TwistStamped:
        linear_msg = msg.twist.linear
        angular_msg = msg.twist.angular
    else:
        linear_msg = msg.linear
        angular_msg = msg.angular
    vector3_to_torch(msg_converted=msg_converted[0:3], msg=linear_msg)
    vector3_to_torch(msg_converted=msg_converted[3:], msg=angular_msg)


FROM_ROS_MSG = {
    Vector3: vector3_to_torch,
    Vector3Stamped: vector3_to_torch,
    Point: point_to_torch,
    PointStamped: point_to_torch,
    Quaternion: quaternion_to_torch,
    QuaternionStamped: quaternion_to_torch,
    JointState: joint_state_to_torch,
    Pose: pose_to_torch,
    PoseStamped: pose_to_torch,
    Twist: twist_to_torch,
    TwistStamped: twist_to_torch,
}

##
# To Ros Message Conversions
##


@apply_header_data
def torch_to_vector3(
    msg: Vector3 | Vector3Stamped, obs: torch.tensor, time: Time | None = None, frame_id: str | None = None
):
    """Convert Isaac Lab 3D vector to Vector3 ROS2 message.

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
    """
    if time is not None:
        msg.vector.x = obs[0].item()
        msg.vector.y = obs[1].item()
        msg.vector.z = obs[2].item()
    else:
        msg.x = obs[0].item()
        msg.y = obs[1].item()
        msg.z = obs[2].item()


@apply_header_data
def torch_to_point(msg: Point | PointStamped, obs: torch.tensor, time: Time | None = None, frame_id: str | None = None):
    """Convert Isaac Lab 3D vector to Point or PointStamped ROS2 message.

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
    """
    if time is not None:
        msg.point.x = obs[0].item()
        msg.point.y = obs[1].item()
        msg.point.z = obs[2].item()
    else:
        msg.x = obs[0].item()
        msg.y = obs[1].item()
        msg.z = obs[2].item()


@apply_header_data
def torch_to_quat(
    msg: Quaternion | QuaternionStamped, obs: torch.tensor, time: Time | None = None, frame_id: str | None = None
):
    """Convert Isaac Lab quaternion in (w, x, y, z) format to Quaternion ROS2 message.

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
    """
    if time is not None:
        msg.quaternion.w = obs[0].item()
        msg.quaternion.x = obs[1].item()
        msg.quaternion.y = obs[2].item()
        msg.quaternion.z = obs[3].item()
    else:
        msg.w = obs[0].item()
        msg.x = obs[1].item()
        msg.y = obs[2].item()
        msg.z = obs[3].item()


def torch_to_float32_multi_array(msg: Float32MultiArray, obs: torch.tensor):
    """Convert Isaac Lab N-d tensor to Float32MultiArray ROS2 message.

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
    """
    msg.data = obs.tolist()


@apply_header_data
def torch_to_imu(
    msg: Imu,
    obs: torch.Tensor,
    time: Time | None = None,
    frame_id: str | None = None,
):
    """Convert IMU sensor torch tensors to Imu message.

    Tensor shape is assumed to be horizontally stacked [quaternion, angular_velocity, linear_acceleration].

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
    """
    # input orientation
    torch_to_quat(
        msg=msg.orientation,
        obs=obs[0:4],
    )
    # input angular velocity
    torch_to_vector3(
        msg=msg.angular_velocity,
        obs=obs[4:7],
    )
    # input angular velocity
    torch_to_vector3(
        msg=msg.linear_acceleration,
        obs=obs[7:10],
    )


@apply_header_data
def torch_to_joint_state(msg: JointState, obs: torch.tensor, time: Time | None = None, frame_id: str | None = None):
    """Convert torch tensors  to JointState ROS2 messages.

    Tensor shape is assumed to be vertically stacked in order of [position, velocity, effort].

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
    """
    msg.position = obs[0].tolist()
    msg.velocity = obs[1].tolist()
    msg.effort = obs[2].tolist()


@apply_header_data
def torch_to_pose(msg: Pose | PoseStamped, obs: torch.tensor, time: Time | None = None, frame_id: str | None = None):
    """Convert Isaac Lab pose vector to ROS2 Pose message.

    Tensor shape is assumed to be horizontally stacked [position, quaternion].

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
    """
    if time is not None:
        torch_to_point(msg=msg.pose.position, obs=obs[0:3])
        torch_to_quat(msg=msg.pose.orientation, obs=obs[3:])
    else:
        torch_to_point(msg=msg.position, obs=obs[0:3])
        torch_to_quat(msg=msg.orientation, obs=obs[3:])


@apply_header_data
def torch_to_transform(
    msg: TransformStamped,
    obs: torch.tensor,
    time: Time | None = None,
    frame_id: str | None = None,
    child_frame: str | None = None,
):
    """Convert Isaac Lab pose vector to ROS2 TransformStamped message.

    Tensor shape is assumed to be horizontally stacked [position, quaternion].

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
    """
    torch_to_vector3(msg=msg.transform.translation, obs=obs[0:3])
    torch_to_quat(msg=msg.transform.rotation, obs=obs[3:])
    msg.child_frame_id = child_frame


@apply_header_data
def torch_to_twist(msg: Twist | TwistStamped, obs: torch.tensor, time: Time | None = None, frame_id: str | None = None):
    """Convert Isaac Lab twist vector to ROS2 Twist message.

    Tensor shape is assumed to be horizontally stacked [linear_velocity, angular_velocity].

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
    """
    if time is not None:
        torch_to_vector3(msg=msg.twist.linear, obs=obs[0:3])
        torch_to_vector3(msg=msg.twist.angular, obs=obs[3:])
    else:
        torch_to_vector3(msg=msg.linear, obs=obs[0:3])
        torch_to_vector3(msg=msg.angular, obs=obs[3:])


@apply_header_data
def torch_to_wrench(
    msg: Wrench | WrenchStamped, obs: torch.tensor, time: Time | None = None, frame_id: str | None = None
):
    """Convert Isaac Lab wrench vector to ROS2 Wrench message.

    Tensor shape is assumed to be horizontally stacked [fx, fy, fz, tx, ty, tz].

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
    """
    if time is not None:
        torch_to_vector3(msg=msg.wrench.force, obs=obs[0:3])
        torch_to_vector3(msg=msg.wrench.torque, obs=obs[3:])
    else:
        torch_to_vector3(msg=msg.force, obs=obs[0:3])
        torch_to_vector3(msg=msg.torque, obs=obs[3:])


@apply_header_data
def torch_to_rgb(
    msg: Image, obs: torch.tensor, time: Time | None = None, frame_id: str | None = None, encoding: str = "rgb8"
):
    """Convert Isaac Lab image tensor to ROS2 Image message.

    Tensor shape is assumed to be [row,column,rgb].

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
        encoding: The OpenCV encoding of the message. Supported encodings: rgb8, bgr8, mono8.

    Raises:
        ValueError: When an unsupported encoding is provided.
    """
    bridge = CvBridge()
    rgb_img = np.uint8(obs.numpy()[:, :, 0:3])
    if encoding == "rgb8":
        cv_msg = bridge.cv2_to_imgmsg(rgb_img, encoding=encoding)
    elif encoding == "bgr8":
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv_msg = bridge.cv2_to_imgmsg(bgr_img, encoding=encoding)
    elif encoding == "mono8":
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        cv_msg = bridge.cv2_to_imgmsg(gray_img, encoding=encoding)
    else:
        raise RuntimeWarning(
            f"Incompatible color image encoding: '{encoding}'. Supported options are to rgb8, bgr8, mono8 (grayscale)'."
        )

    msg.data = cv_msg.data
    msg.height = cv_msg.height
    msg.width = cv_msg.width
    msg.encoding = cv_msg.encoding
    msg.is_bigendian = cv_msg.is_bigendian
    msg.step = cv_msg.step


@apply_header_data
def torch_to_depth(
    msg: Image,
    obs: torch.tensor,
    time: Time | None = None,
    frame_id: str | None = None,
    encoding: str = "mono16",
    threshold: tuple[float, float] | None = None,
    scale: float = 1.0,
):
    """Convert Isaac Lab image tensor to ROS2 depth Image.

    Tensor shape is assumed to be [row,column].

    Args:
        msg: The ROS message to convert the tensor data into.
        obs: The tensor data to convert to ROS message.
        time: The ROS time stamp to store in message header for stamped messages. Defaults to None.
        frame_id: The TF frame_id corresponding to the message to be stored in message header. Defaults to None.
        encoding: The OpenCV encoding of the message. Supported encodings: rgb8, bgr8, mono8.
        threshold: The minimum and maximum values of the depth image.
        scale: The scaling factor applied to the depth image.

    Raises:
        ValueError: When an unsupported encoding is provided.
    """
    bridge = CvBridge()

    if threshold is not None:
        obs = torch.clamp(obs, min=threshold[0], max=threshold[1])

    if encoding == "mono16":
        cv_msg = bridge.cv2_to_imgmsg(np.uint16(obs.numpy() * scale), encoding=encoding)
    elif encoding == "mono8":
        cv_msg = bridge.cv2_to_imgmsg(np.uint8(obs.numpy() * scale), encoding=encoding)
    elif encoding == "32FC1":
        cv_msg = bridge.cv2_to_imgmsg(np.float32(obs.numpy() * scale), encoding=encoding)
    else:
        raise ValueError(
            f"Incompatible depth encoding: '{encoding}'. Supported options are 'mono8', 'mono16' or '32FC1'."
        )
    msg.data = cv_msg.data
    msg.height = cv_msg.height
    msg.width = cv_msg.width
    msg.encoding = cv_msg.encoding
    msg.is_bigendian = cv_msg.is_bigendian
    msg.step = cv_msg.step


TO_ROS_MSG = {
    Vector3: torch_to_vector3,
    Vector3Stamped: torch_to_vector3,
    Point: torch_to_point,
    PointStamped: torch_to_point,
    Quaternion: torch_to_quat,
    QuaternionStamped: torch_to_quat,
    Pose: torch_to_pose,
    PoseStamped: torch_to_pose,
    TransformStamped: torch_to_transform,
    Float32MultiArray: torch_to_float32_multi_array,
    Imu: torch_to_imu,
    JointState: torch_to_joint_state,
    Twist: torch_to_twist,
    TwistStamped: torch_to_twist,
    Wrench: torch_to_wrench,
    WrenchStamped: torch_to_wrench,
}
