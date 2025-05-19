# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from geometry_msgs.msg import (
    PointStamped,
    PoseStamped,
    TransformStamped,
    TwistStamped,
    Vector3Stamped,
    WrenchStamped,
)

from grid_map_msgs.msg import GridMap
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
from std_msgs.msg import Float32MultiArray

from . import publishers


@configclass
class PublisherBaseTermCfg:
    class_type: object = MISSING
    """Publisher tied to this configuration"""
    msg_type: object = None
    """Ros message type to be used by the publisher"""
    topic: str = MISSING
    """Ros topic name to be published to"""
    substep: int | None = 1
    """Set the publish substepping n. For every n physics steps this term will publish its messages.
        substep = None will result in publishing once at beginning of simulation."""


@configclass
class ClockPublisherCfg(PublisherBaseTermCfg):
    class_type = publishers.ClockPublisher
    """Publisher tied to this configuration"""
    msg_type = Clock
    """Ros message type to be used by the publisher"""
    topic = "/clock"
    """Ros topic name to be published to"""


@configclass
class DirectPublisherBaseCfg(PublisherBaseTermCfg):
    asset_cfg: SceneEntityCfg = MISSING
    """Defines which asset (Articulation or Sensor) to be referenced."""


@configclass
class HeightMapPublisherCfg(DirectPublisherBaseCfg):
    class_type = publishers.HeightMapPublisher
    """PublisherTerm tied to this configuration"""
    msg_type = GridMap
    """Ros message type to be used by the publisher"""
    asset_cfg: SceneEntityCfg = MISSING
    """A `SceneEntityCfg` pointing to the height scanner"""
    layer: str = "elevation"
    """The name of the height layer to be published"""
    noise: NoiseCfg | None = None
    """The noise to add to the observation. Defaults to None, in which case no noise is added."""


class RGBImagePublisherCfg(DirectPublisherBaseCfg):
    class_type = publishers.RGBImagePublisher
    """PublisherBaseTerm tied to this configuration"""
    msg_type = Image
    """Ros message type to be used by the publisher"""
    encoding = "rgb8"
    """Color encoding to be used in the message: rgb8 or bgr8"""


@configclass
class DepthImagePublisherCfg(DirectPublisherBaseCfg):
    class_type = publishers.DepthImagePublisher
    """PublisherBaseTerm tied to this configuration"""
    msg_type = Image
    """Ros message type to be used by the publisher"""
    encoding = "mono8"
    """Depth encoding to be used in the message: mono8, mono16, or 32FC1 """
    threshold: tuple[float, float] | None = (0.0, 20.0)
    """(Minimum , Maximum) depth in meters to limit depth measurements to"""
    scale: float = 1.0
    """Scaling coefficient between meters and uint8(mono8) and uint16(mono16) integer ranges"""


@configclass
class CameraInfoPublisherCfg(DirectPublisherBaseCfg):
    class_type = publishers.CameraInfoPublisher
    """Publisher tied to this configuration"""
    msg_type = CameraInfo
    """Ros message type to be used by the publisher"""


@configclass
class ContactForcePublisherCfg(DirectPublisherBaseCfg):
    class_type = publishers.ContactForcePublisher
    """Publisher tied to this configuration."""
    msg_type = Float32MultiArray
    """ROS message type"""


@configclass
class TFBroadcasterCfg(DirectPublisherBaseCfg):
    class_type = publishers.TFBroadcaster
    """TF Broadcaster term tied to this configuration"""
    topic = "/tf"
    """Topic the TransformBroadcaster is publishing to (for reference only, not configurable)."""
    msg_type = TransformStamped
    """Message type used in TF broadcaster(for reference only, not configurable)."""
    asset_cfg: SceneEntityCfg | list[SceneEntityCfg] = MISSING
    """Defines a list of Articulations and Sensors."""
    additional_prim_paths: list[str] | None = None
    """List of additional prim paths to prims whose TFs that are to be broadcasted."""
    env_path: str = "/World/envs/env_0"
    """Path for additional_prim_paths to be relative to."""


@configclass
class StaticTFBroadcasterCfg(TFBroadcasterCfg):
    class_type = publishers.StaticTFBroadcaster
    """TF Broadcaster term tied to this configuration"""
    topic = "/static_tf"
    """Topic the StaticTransformBroadcaster is publishing to (for reference only, not configurable)."""
    substep = None
    """The interval upon which static transformations will be broadcast.
    If None, transformations will be published once, at the time ROS is enabled in EvalSim GUI.
    """
    frame_id_path: str | None = None
    """Path to prim (assuming relative to self.env_path) the user
     wants to publish static TF w.r.t. If None, the frame_id of the message
     will be w.r.t. the prim parent."""


##
# Observation based publisher configs
##


@configclass
class ObservationPublisherBaseCfg(PublisherBaseTermCfg):
    class_type = publishers.ObservationPublisherBase
    """Publisher tied to this configuration"""
    obs_group: str = MISSING
    """Observation group to associate with this publisher"""


@configclass
class ObservationTermPublisherCfg(ObservationPublisherBaseCfg):
    class_type = publishers.ObservationTermPublisher
    """Publisher tied to this configuration"""
    obs_term_name: str = MISSING
    """Observation term to associate with this publisher"""
    flatten: bool = True
    """Flag to flatten observation to 1D array."""


@configclass
class FlattenedObsPublisherCfg(ObservationTermPublisherCfg):
    msg_type = Float32MultiArray
    """ROS message type"""
    flatten: bool = True
    """Flag to flatten observation to 1D array."""


@configclass
class Vector3StampedObsPublisherCfg(ObservationTermPublisherCfg):
    msg_type = Vector3Stamped
    """ROS message type"""


@configclass
class PointStampedObsPublisherCfg(ObservationTermPublisherCfg):
    msg_type = PointStamped
    """ROS message type"""


@configclass
class WrenchStampedObsPublisherCfg(ObservationTermPublisherCfg):
    msg_type = WrenchStamped
    """ROS message type"""


@configclass
class JointStateObsPublisherCfg(ObservationPublisherBaseCfg):
    class_type = publishers.JointStateObsPublisher
    """Publisher tied to this configuration"""
    msg_type = JointState
    """Ros message type to be used by the publisher"""
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """Defines which Articulation asset is used. Can optionally be used to define which joints shall be published."""
    position_obs: str | None = "joint_pos"
    """Joint position observation term"""
    velocity_obs: str | None = "joint_vel"
    """Joint velocity observation term"""
    effort_obs: str | None = "joint_effort"
    """Joint effort observation term"""


@configclass
class JointReactionWrenchObsPublisherCfg(ObservationPublisherBaseCfg):
    class_type = publishers.JointReactionWrenchObsPublisher
    """PublisherTerm tied to this configuration"""
    msg_type = Float32MultiArray
    """Ros message type to be used by the publisher"""
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """Defines which asset is used. Can optionally be used to define which body's proximal joint reaction forces
    shall be published."""
    obs_term_name: str = MISSING
    """The name of the Observation term that produces a series of wrenches for bodies in an articulation to associate
    with this publisher"""


@configclass
class LinkPoseObsPublisherCfg(ObservationPublisherBaseCfg):
    class_type = publishers.LinkPoseObsPublisher
    """Publisher tied to this configuration"""
    msg_type = PoseStamped
    """Ros message type to be used by the publisher"""
    link_pose_obs: str | None = None
    """Name of the link pose observation. If None the link_pos_obs and link_quat_obs must be provided."""
    link_pos_obs: str | None = None
    """Name of the link position observation. If None, the link_pose_obs must be provided."""
    link_quat_obs: str | None = None
    """Name of the link quaternion observation. Expecting order (q,x,y,z). If None, the link_pose_obs must be provided."""
    frame_id: str = "World"
    """Name of the relative frame of the provided observations."""


@configclass
class TwistObsPublisherCfg(ObservationPublisherBaseCfg):
    class_type = publishers.TwistObsPublisher
    """Publisher tied to this configuration"""
    msg_type = TwistStamped
    """Ros message type to be used by the publisher"""
    frame_id: str = MISSING
    """Name of reference frame of the observations"""
    lin_vel_obs: str = MISSING
    """Name of linear velocity observation term"""
    ang_vel_obs: str = MISSING
    """Name of angular velocity observation term"""


@configclass
class ImuObsPublisherCfg(ObservationPublisherBaseCfg):
    class_type = publishers.ImuObsPublisher
    """Publisher tied to this configuration"""
    msg_type = Imu
    """Ros message type to be used by this publisher"""
    imu_quat_obs: str | None = None
    """Name of imu sensor orientation quaternion (wxyz) observation term.

    Defaults to None. If None zeros are published for this term in the ROS 2 Imu message."""
    imu_ang_vel_obs: str | None = None
    """Name of imu sensor angular velocity (rad/s) observation term.

    Defaults to None. If None zeros are published for this term in the ROS 2 Imu message."""
    imu_lin_acc_obs: str | None = None
    """Name of imu sensor linear acceleration (m/s) observation term.

    Defaults to None. If None zeros are published for this term in the ROS 2 Imu message."""
