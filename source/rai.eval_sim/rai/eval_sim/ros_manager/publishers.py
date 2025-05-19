# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import carb
import isaaclab.utils.math as math_utils
import isaacsim.core.utils.prims as prims_utils
import torch
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import Camera, RayCaster, SensorBase
from isaaclab.sensors.ray_caster.patterns import patterns_cfg
from isaacsim.core.prims import XFormPrim
from rai.eval_sim.utils import ros_conversions
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

if TYPE_CHECKING:
    from . import publishers_cfg


class PublisherBaseTerm(ABC):
    """Base class for ROS2 publishers

    The PublisherBaseTerm class is used to define a base class for other publishers to inherit from.
    The cfg associated with this publisher (and those inherited) will define parameters on how the
    message is created and published across the ROS network.

    Each inherited publisher definition must define a _prepare_msg function that gets called just before publishing.

    Attributes:
        cfg: Configuration for the publisher.
        env: Environment containing the scene and sensors.
        msg: The ROS message object to fill and publish.
        publisher: The ROS 2 publisher of type `msg`.
        topic_name: The name of the ROS 2 topic the `msg` is published over.
    """

    def __init__(
        self,
        cfg: publishers_cfg.PublisherBaseTermCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the ros publisher term.

        Args:
            cfg: The configuration object of type PublisherBaseTermCfg.
            node: The ros node instance to tie this publisher to.
            env: The Isaac Lab ManagerBased environment.
            qos_profile: The quality of service profile for ROS communications.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: The index of the environment, defaults to 0. Used when multiple environments are managed.
        """
        # store the inputs
        self._cfg = cfg
        self._node = node
        self._env = env
        self._use_sim_time = use_sim_time
        self._qos_profile = qos_profile
        self._env_idx = env_idx

        # set up ROS message
        self._msg = self._initialize_msg()

        # set up publisher
        self.publisher = node.create_publisher(
            msg_type=cfg.msg_type, topic=self.topic_name, qos_profile=self._qos_profile
        )

    def close(self):
        del self._env

    @property
    def cfg(self) -> publishers_cfg.PublisherBaseTermCfg:
        return self._cfg

    @property
    def msg(self) -> publishers_cfg.PublisherBaseTermCfg.msg_type:
        return self._msg

    @property
    def topic_name(self):
        return self.cfg.topic

    def _initialize_msg(self):
        """Initialize ROS message with zeros."""
        return self.cfg.msg_type()

    def get_node_clock_msg(self) -> Time:
        """Create a time stamp message for the current ROS time."""
        return self._node.get_clock().now().to_msg()

    def get_sim_timestamp(self) -> Time:
        time = self._env.sim.current_time
        sec = math.floor(time)
        nanosec = math.floor((time - math.floor(time)) * 1e9)
        return Time(sec=sec, nanosec=nanosec)

    def get_timestamp(self) -> Time:
        if self._use_sim_time:
            return self.get_sim_timestamp()
        else:
            return self.get_node_clock_msg()

    @abstractmethod
    def _prepare_msg(self, *args, **kwargs) -> None:
        """Implementation function to extract data from the simulation and format it in the chosen message type."""

    def publish(self, *args, **kwargs) -> None:
        """Publish the observation message."""
        # prepare message
        self._prepare_msg(*args, **kwargs)
        # publish message
        self.publisher.publish(self.msg)


class ClockPublisher(PublisherBaseTerm):
    """Publishes the current time of the simulation environment."""

    def _prepare_msg(self) -> None:
        """Extract simulation time and populate the clock message."""
        time = self._env.sim.current_time
        self.msg.clock.sec = math.floor(time)
        self.msg.clock.nanosec = math.floor((time - math.floor(time)) * 1e9)


class DirectPublisherBase(PublisherBaseTerm):
    """Base class for publishers that directly reference an asset in the scene for message data.

    Examples of this include directly accessing sensors or articulations for data. These publishers do not have the option
    to utilize the Noise and Modifier pipelines in the ObservationManager.

    Attributes:
        cfg: Configuration for the publisher.
        env: Environment containing the scene and sensors.
        msg: The ROS message object to fill and publish.
        publisher: The ROS 2 publisher of type `msg`.
        topic_name: The name of the ROS 2 topic the `msg` is published over.
        asset: The SceneEntity used to fill this publisher's `msg`.
    """

    def __init__(
        self,
        cfg: publishers_cfg.PublisherBaseTermCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the ros publisher term.

        Args:
            cfg: The configuration publisher.
            node: The ros node instance to tie this publisher to.
            env: The Isaac Lab ManagerBased environment.
            qos_profile: The quality of service profile for ROS communications.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: The index of the environment, defaults to 0. Used when multiple environments are managed.
        """

        # store the inputs
        self._cfg = cfg
        self._node = node
        self._env = env
        self._qos_profile = qos_profile
        self._use_sim_time = use_sim_time
        self._env_idx = env_idx

        # resolve scene entity
        self.cfg.asset_cfg.resolve(env.scene)

        # store scene entity handle handle
        self._asset = self._env.scene[self.cfg.asset_cfg.name]

        # set up ROS message
        self._msg = self._initialize_msg()

        # set up publisher
        self.publisher = node.create_publisher(
            msg_type=cfg.msg_type, topic=self.topic_name, qos_profile=self._qos_profile
        )

    @property
    def asset(self) -> Articulation | SensorBase:
        return self._asset

    @abstractmethod
    def _prepare_msg(self) -> None:
        """Implementation function to asset data and format it in the chosen message type."""


class HeightMapPublisher(DirectPublisherBase):
    """Publish the height map collected from a RayCaster sensor.

    The publisher publishes the map according to the grid map conventions. For more details, see https://github.com/ANYbotics/grid_map?tab=readme-ov-file#conventions--definitions

       Notice that grid map internally stores the map as Eigen::Matrix<float, num_rows, num_cols, ColMajor>, i.e., that the map is encoded as

       y-axis
       ^
       | +----------< start
       | |
       | +----------+
       |            |
       | <----------+
       +----------------> x-axis

       In IsaacLab, the ray casting is generated in xy-ordering, which means that the map is encoded as

       y-axis
       ^
       | +----------> end
       | |
       | +----------+
       |            |
       | -----------+
       +----------------> x-axis

       This publisher only supports:
           - xy ordering for the ray casting pattern (to be set in RayCasterCfg.pattern_cfg)
           - world as the ray alignment method (to be set in RayCasterCfg.ray_alignment)
    """

    def __init__(
        self,
        cfg: publishers_cfg.HeightMapPublisherCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the base pose publisher term.

        Args:
            cfg: The configuration publisher.
            node: The ros node instance to tie this publisher to.
            env: The Isaac Lab ManagerBased environment.
            qos_profile: The quality of service profile for ROS communications.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: The index of the environment, defaults to 0. Used when multiple environments are managed.
        """

        #         # call the base class constructor
        #         super().__init__(
        #             cfg=cfg,
        #             node=node,
        #             env=env,
        #             qos_profile=qos_profile,
        #             use_sim_time=use_sim_time,
        #             env_idx=env_idx,
        #         )

        # initialize publisher
        if isinstance(self._asset, RayCaster):
            self._grid_pattern_cfg: patterns_cfg.GridPatternCfg = self._asset.cfg.pattern_cfg
        else:
            raise RuntimeError("Expecting type RayCaster for asset, but got ", type(self._asset), ".")
        self.noise = cfg.noise
        grid_value_x = torch.arange(
            start=-self._grid_pattern_cfg.size[0] / 2,
            end=self._grid_pattern_cfg.size[0] / 2 + 1.0e-9,
            step=self._grid_pattern_cfg.resolution,
        )
        grid_values_y = torch.arange(
            start=-self._grid_pattern_cfg.size[1] / 2,
            end=self._grid_pattern_cfg.size[1] / 2 + 1.0e-9,
            step=self._grid_pattern_cfg.resolution,
        )

        # check if ordering as assumed.
        if not self._grid_pattern_cfg.ordering == "xy":
            raise RuntimeError("The publisher currently only supports xy ordering, which is the default in IsaacLab.")

        # initialize message
        self._msg.info.length_x, self._msg.info.length_y = self._grid_pattern_cfg.size
        self._msg.info.resolution = self._grid_pattern_cfg.resolution
        shape = [
            MultiArrayDimension(size=len(grid_values_y), label="column_index", stride=0),
            MultiArrayDimension(size=len(grid_value_x), label="row_index", stride=0),
        ]  # column major
        self._msg.data = [Float32MultiArray(layout=MultiArrayLayout(dim=shape, data_offset=0))]
        self._msg.data[0].data = self._asset.num_rays * [0.0]  # empty list with num_rays elements
        self._msg.layers = [self.cfg.layer]

        if self._asset.cfg.ray_alignment == "world":
            self._msg.header.frame_id = "world"
        else:
            raise RuntimeError("Only world is supported as frame id, i.e., ray_alignment must be world.")

    def _prepare_msg(self):
        "Prepares GridMap message for publishing."
        height_scan = self._asset.data.ray_hits_w[self._env_idx, :, 2]
        if self.noise:
            height_scan = self.noise.func(height_scan, self.noise)
        height_scan_list = height_scan.tolist()
        height_scan_list.reverse()  # adapt to conventions used in grid map
        self._msg.data[0].data = height_scan_list

        self._msg.header.stamp = self.get_timestamp()

        # note: we only support world frame here
        grid_map_pose = torch.cat([
            self._asset.data.pos_w[self._env_idx, :],
            torch.tensor([1, 0, 0, 0]),
        ])

        ros_conversions.torch_to_pose(
            msg=self._msg.info.pose,
            obs=grid_map_pose,
        )


class RGBImagePublisher(DirectPublisherBase):
    """Publishes rgb images generated by a Camera sensor specified in the asset_cfg.

    The Camera must have a valid 'rgb' data stream."""

    def __init__(
        self,
        cfg: publishers_cfg.RGBImagePublisherCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the rgb publisher.

        Args:
            cfg: The configuration publisher.
            node: The ros node instance to tie this publisher to.
            env: The Isaac Lab ManagerBased environment.
            qos_profile: The quality of service profile for ROS communications.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: The index of the environment, defaults to 0. Used when multiple environments are managed.
        """
        # call the base class constructor
        super().__init__(cfg, node, env, qos_profile, use_sim_time, env_idx)
        # set frame id
        self._frame_id = self.asset.cfg.prim_path.split("/")[-1]

    def _prepare_msg(self) -> None:
        """Extract rgb information from the camera sensor and fill ROS message."""
        # get camera data
        rgb_image_tensor = self.asset.data.output["rgb"][self._env_idx]
        # convert camera data
        ros_conversions.torch_to_rgb(
            msg=self.msg,
            obs=rgb_image_tensor,
            time=self.get_timestamp(),
            frame_id=self._frame_id,
            encoding=self.cfg.encoding,
        )


class DepthImagePublisher(DirectPublisherBase):
    """Publishes depth images generated by a Camera sensor specified in the asset_cfg.
    The Camera must have a valid 'distance_to_image_plane' data stream."""

    def __init__(
        self,
        cfg: publishers_cfg.DepthImagePublisherCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the depth publisher.

        Args:
            cfg: The configuration publisher.
            node: The ros node instance to tie this publisher to.
            env: The Isaac Lab ManagerBased environment.
            qos_profile: The quality of service profile for ROS communications.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: The index of the environment, defaults to 0. Used when multiple environments are managed.
        """
        # call the base class constructor
        super().__init__(cfg, node, env, qos_profile, use_sim_time, env_idx)
        # set frame id
        self._frame_id = self.asset.cfg.prim_path.split("/")[-1]

    def _prepare_msg(self) -> None:
        """Extract depth information from camera sensor and fill ROS message."""
        # get camera data
        depth_image_tensor = self.asset.data.output["distance_to_image_plane"][self._env_idx]
        # convert camera data
        ros_conversions.torch_to_depth(
            msg=self.msg,
            obs=depth_image_tensor,
            time=self.get_timestamp(),
            frame_id=self._frame_id,
            encoding=self.cfg.encoding,
            threshold=self.cfg.threshold,
            scale=self.cfg.scale,
        )


class CameraInfoPublisher(DirectPublisherBase):
    """Publishes the intrinsic properties of the camera specified in the asset_cfg."""

    def __init__(
        self,
        cfg: publishers_cfg.CameraInfoPublisherCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the camera info publisher.

        Args:
            cfg: The configuration publisher.
            node: The ros node instance to tie this publisher to.
            env: The Isaac Lab ManagerBased environment.
            qos_profile: The quality of service profile for ROS communications.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: The index of the environment, defaults to 0. Used when multiple environments are managed.
        """
        # call the base class constructor
        super().__init__(cfg, node, env, qos_profile, use_sim_time, env_idx)
        self.extract_camera_info()

    def extract_camera_info(self):
        # Calculate camera intrinsics
        # following conversions: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html
        prim_path = self.asset.cfg.prim_path
        prim_path = prim_path.replace("env_.*", "env_" + str(self._env_idx))

        focal_length = prims_utils.get_prim_property(prim_path, property_name="focalLength")
        horizontal_aperture = prims_utils.get_prim_property(prim_path, property_name="horizontalAperture")
        px_size = horizontal_aperture / self.camera.cfg.width

        fx = focal_length / px_size
        fy = fx
        cx = self.asset.cfg.width / 2
        cy = self.asset.cfg.height / 2

        # fill message
        self.msg.height = self.asset.cfg.height
        self.msg.width = self.asset.cfg.width
        # assuming no distortion
        self.msg.distortion_model = "plumb_bob"
        self.msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
        # Intrinsic camera matrix for the raw (distorted) images.
        self.msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        # Rectification matrix (stereo cameras only) assume monocular identity matrix
        self.msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # 3x3 row-major matrix
        # Projection matrix assume monocular
        self.msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        # no sub-sampling
        self.msg.binning_x = 1
        self.msg.binning_y = 1
        # header
        self.msg.header.frame_id = self.camera.cfg.prim_path.split("/")[-1]
        self.msg.header.stamp = self.get_timestamp()

    def _prepare_msg(self) -> None:
        """Camera info doesn't change so the message is only setup once."""


class TFBroadcaster(DirectPublisherBase):
    """Broadcasts transforms of articulations and sensors to TF2.

    NOTE: TFBroadcaster transforms all have frame_id="World". The child_frame for each transform is the prim name.

    NOTE: For static frames use :class:`StaticTFBroadcaster`. For increased performance, only publish the frames with dynamic
    parent transform through :class:`TFBroadcaster`, then publish frames with static transforms to parents using the
    StaticTFBroadcaster.

    NOTE: For best performance, users should use a separate robot state publisher that that listens to the joint_state message
    and utilizes a URDF of the robot to calculate link transforms in another state process.

    Attributes:
        cfg: Configuration for the publisher.
        node: ROS node to which the publisher is attached.
        env: Environment containing the scene and sensors.
        qos_profile: Quality of Service profile for the ROS publisher.
        env_idx: Index of the environment instance. Defaults to 0.
        msg: The ROS message object to fill and publish.
        tf_broadcaster: The ROS 2 TransformBroadcaster to publish to the topic /tf .
        topic_name: The name of the ROS 2 topic /tf that is only for information.
        asset: The SceneEntity used to pull transformation information.
        additional_prim_paths: The list of USD prim paths to publish transforms of.
    """

    def __init__(
        self,
        cfg: publishers_cfg.TFBroadcasterCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the TF broadcaster term. It does not create a publisher to a topic. Instead, it will
        utilize the ROS 2's built-in :class:`TransformBroadcaster`.

        Args:
            cfg: The configuration object.
            node: The ros node instance to tie this publisher to.
            env: The Isaac Lab ManagerBased environment.
            qos_profile: The quality of service profile for ROS communications.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: The index of the environment, defaults to 0. Used when multiple environments are managed.
        """
        # store the inputs
        self._cfg: publishers_cfg.TFBroadcasterCfg = cfg
        self._node = node
        self._env = env
        self._qos_profile = qos_profile
        self._use_sim_time = use_sim_time
        self._env_idx = env_idx

        # create tf broadcaster
        self.tf_broadcaster = TransformBroadcaster(self._node)

        # create list of assets to add to TransformBroadcaster
        self._asset = []
        if not isinstance(self.cfg.asset_cfg, list):
            asset_list = [self.cfg.asset_cfg]
        else:
            asset_list = self.cfg.asset_cfg

        for asset_cfg in asset_list:
            # resolve scene entities
            asset_cfg.resolve(env.scene)

            # store asset(articulation or sensor) handle
            self._asset.append(self._env.scene[asset_cfg.name])

        # add env parent path to additional prims
        self._env_path = self.cfg.env_path.replace("env_.*", f"env_{self._env_idx}")
        self.additional_prim_paths = self._additional_prim_paths()

        # create dictionary of messages
        self.tf_dict = self._initialize_msg()

    def _additional_prim_paths(self) -> list[str]:
        """Gets the list of prim paths with prepended environment path.

        Returns:
            List of prim paths prefixed with base environment path."""

        prim_paths = []
        if self.cfg.additional_prim_paths is not None:
            for prim_path in self.cfg.additional_prim_paths:
                if prim_path[0] == "/":
                    prim_paths.append(f"{self._env_path}{prim_path}")
                else:
                    prim_paths.append(f"{self._env_path}/{prim_path}")

        return prim_paths

    def _get_link_id(self, asset: Articulation, link_name: str) -> int:
        """Retrieve the index of a link name within the asset's body names.

        Args:
            asset: The Articulation object to pull link_name from.
            link_name: The name of the link to retrieve the index for.

        Returns:
            int: The index of the link name.

        Raises:
            ValueError: If the link name is not found in the asset's body names.
        """
        try:
            return asset.body_names.index(link_name)
        except ValueError:
            err_msg = f"Body name: '{link_name}' is not a valid body of asset: '{asset}'"
            carb.log_error(err_msg)
            raise ValueError(err_msg)

    def _get_prim_world_pose_at_path(self, prim_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets world pose of prim at prim_path.

        Args:
            prim_path: The prim path string.

        Returns:
            A tuple containing the position and orientation of prim w.r.t. World.
            Shape of the tensors are (1, 3) and (1, 4) respectively.

        Returns:
            RuntimeError if provided path is not a valid prim.
        """

        if prims_utils.is_prim_path_valid(prim_path.replace("env_.*", f"env_{self._env_idx}")):
            prim = XFormPrim(prim_path)
            pos, rot = prim.get_world_poses()
            return pos[self._env_idx, ...], rot[self._env_idx, ...]
        else:
            raise RuntimeError(f"Provided prim_path: {prim_path} is not a valid prim.")

    def _get_prim_local_pose_at_path(self, prim_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets local pose of prim at prim_path.

        Args:
            prim_path: The prim path string.

        Returns:
            A tuple containing the position and orientation of prim w.r.t. parent prim.
            Shape of the tensors are (1, 3) and (1, 4) respectively.

        Returns:
            RuntimeError if provided path is not a valid prim.
        """

        if prims_utils.is_prim_path_valid(prim_path.replace("env_.*", f"env_{self._env_idx}")):
            prim = XFormPrim(prim_path)
            pos, rot = prim.get_local_poses()
            return pos[self._env_idx, ...], rot[self._env_idx, ...]
        else:
            raise RuntimeError(f"Provided prim_path: {prim_path} is not a valid prim.")

    def _tf_prim_at_path_to_world(self, prim_path: str) -> TransformStamped:
        """Gets world pose of prim at prim_path.

        Args:
            prim_path: The prim path string.

        Returns:
            TransformStamped message containing position and orientation relative to World.
        """

        tf_msg = TransformStamped()

        pos_w, quat_w = self._get_prim_world_pose_at_path(prim_path=prim_path)
        transform = torch.cat((pos_w, quat_w), dim=0)
        # convert torch tensor to transform
        ros_conversions.torch_to_transform(
            msg=tf_msg,
            obs=transform,
            time=self.get_timestamp(),
            frame_id=prim_path.split("/")[1],
            child_frame=prim_path.split("/")[-1],
        )
        return tf_msg

    def _initialize_msg(self) -> dict[str, TransformStamped]:
        """Preallocates dictionary of msgs

        Returns:
            A dictionary with order with body name as key and associated message as value.

        Raises:
            TypeError when asset_cfg specifies a type other than Articulation, RigidObject, or SensorBase."""

        # initialize dictionary
        tf_dict: dict[str, TransformStamped] = dict()
        # get clock time
        msg_time = self.get_timestamp()
        # create identity transform
        zero_transform = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # add additional prim_paths to tf dictionary
        if self.additional_prim_paths is not None:
            for path in self.additional_prim_paths:
                tf_dict[path.split("/")[-1]] = self._tf_prim_at_path_to_world(path)

        # world frame id for assets
        frame_id = self._asset[0].cfg.prim_path.split("/")[1]

        for asset in self._asset:
            # if asset is an Articulation or RigidObject, add bodies to list of transforms
            if isinstance(asset, (Articulation, RigidObject)):
                for body in asset.body_names:
                    tf_msg = TransformStamped()
                    ros_conversions.torch_to_transform(
                        msg=tf_msg,
                        obs=zero_transform,
                        time=msg_time,
                        frame_id=frame_id,
                        child_frame=body,
                    )
                    tf_dict[body] = tf_msg

            # if asset is a sensor add sensor pose to list of transforms
            elif isinstance(asset, SensorBase):
                tf_msg = TransformStamped()
                body = asset.cfg.prim_path.split("/")[-1]
                # convert torch tensor to transform
                ros_conversions.torch_to_transform(
                    msg=tf_msg,
                    obs=zero_transform,
                    time=msg_time,
                    frame_id=frame_id,
                    child_frame=body,
                )
                tf_dict[body] = tf_msg
            else:
                raise TypeError(
                    f"Asset must be of type Articulation, RigidObject, or SensorBase. Provided: {type(asset)}"
                )
        return tf_dict

    def _prepare_msg(self) -> None:
        """Updates a dictionary of transform messages (self.tf_dict) to be published."""
        # get time
        msg_time = self.get_timestamp()

        # gather TF messages for any additional prim paths
        if self.additional_prim_paths is not None:
            for path in self.additional_prim_paths:
                body = path.split("/")[-1]
                pos_w, quat_w = self._get_prim_world_pose_at_path(prim_path=path)
                transform = torch.cat((pos_w, quat_w), dim=0)
                # convert torch tensor to transform
                ros_conversions.torch_to_transform(
                    msg=self.tf_dict[body],
                    obs=transform,
                    time=msg_time,
                    frame_id=path.split("/")[1],
                    child_frame=body,
                )

        # loop through assets and publish relative tfs
        frame_id = self._asset[0].cfg.prim_path.split("/")[1]
        for asset in self._asset:

            # if asset is an Articulation or RigidObject, add bodies to list of transforms
            if isinstance(asset, (Articulation, RigidObject)):
                body_states = asset.data.body_state_w
                for i, body in enumerate(asset.body_names):
                    # access link of articulation
                    transform = body_states[self._env_idx, i, 0:7]
                    # convert torch tensor to transform
                    ros_conversions.torch_to_transform(
                        msg=self.tf_dict[body],
                        obs=transform,
                        time=msg_time,
                        frame_id=frame_id,
                        child_frame=body,
                    )

            # if asset is a sensor add sensor pose to list of transforms
            elif isinstance(asset, SensorBase):
                body = asset.cfg.prim_path.split("/")[-1]
                # publish camera orientation using ros convention for camera frames
                if isinstance(asset, Camera):
                    sensor_pos_w = asset.data.pos_w.squeeze()
                    sensor_quat_w = asset.data.quat_w_ros.squeeze()
                else:
                    sensor_pos_w = asset.data.pos_w.squeeze()
                    sensor_quat_w = asset.data.quat_w.squeeze()
                transform = torch.cat((sensor_pos_w, sensor_quat_w), dim=0)
                # convert torch tensor to transform

                ros_conversions.torch_to_transform(
                    msg=self.tf_dict[body],
                    obs=transform,
                    time=msg_time,
                    frame_id=frame_id,
                    child_frame=body,
                )

    def publish(self) -> None:
        """Prepares and sends transform messages to TF2."""
        self._prepare_msg()
        self.tf_broadcaster.sendTransform(list(self.tf_dict.values()))


class StaticTFBroadcaster(TFBroadcaster):
    """Broadcasts static transforms of articulations and sensors to TF2.

    NOTE: StaticTFBroadcaster by default broadcasts the root link frame of the articulation relative to World and
    sensor frames relative to their parent prim. To change the relative source frame, set the frame_id_path parameter of
    :class:`StaticTFBroadcasterCfg`

    Static frames are meant to be published infrequently. Use :class:`StaticTFBroadcaster.substep` to set frequency.
    For dynamic frames use :class:`TFBroadcaster`. For increased performance only publish the Frames with dynamics
    parent transform through :class:`TFBroadcaster`. Then publish frames with static transforms to parents using the
    :class:`StaticTFBroadcaster`.

    """

    def __init__(
        self,
        cfg: publishers_cfg.StaticTFBroadcasterCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ) -> None:
        """Initialize the TF broadcaster term. It does not create a publisher to a topic. Instead it will
        utilize the tf_broadcaster.

        Args:
            cfg: The configuration object of type StaticTFBroadcasterCfg.
            node: The ros node instance to tie this publisher to.
            env: The Isaac Lab ManagerBased environment.
            qos_profile: The quality of service profile for ROS communications.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: The index of the environment, defaults to 0. Used when multiple environments are managed.
        """
        super().__init__(
            cfg=cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            use_sim_time=use_sim_time,
            env_idx=env_idx,
        )

        self.tf_broadcaster = StaticTransformBroadcaster(self._node)

    def _initialize_msg(self) -> dict[str, TransformStamped]:
        """Preallocates dictionary of msgs

        Returns:
            A dictionary with order with body name as key and associated message as value.

        Raises:
            TypeError when asset_cfg specifies a type other than Articulation, RigidObject, or SensorBase."""

        # initialize dictionary
        tf_dict: dict[str, TransformStamped] = dict()

        # preemptively add environment -> World transform
        tf_dict[self._env_path.split("/")[-1]] = self._tf_prim_at_path_to_world(self._env_path)

        # get pose of prim at frame_id w.r.t World
        if self.cfg.frame_id_path is not None:
            self.frame_id_path = self._env_path + "/" + self.cfg.frame_id_path
        else:
            self.frame_id_path = None

        # get clock time
        msg_time = self.get_timestamp()

        # create identity transform
        zero_transform = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # add additional prim_paths to tf dictionary
        if self.additional_prim_paths is not None:
            for path in self.additional_prim_paths:
                tf_msg = TransformStamped()
                body = path.split("/")[-1]
                frame_id_ = path.split("/")[-2] if self.frame_id_path is None else self.frame_id_path.split("/")[-1]
                ros_conversions.torch_to_transform(
                    msg=tf_msg,
                    obs=zero_transform,
                    time=msg_time,
                    frame_id=frame_id_,
                    child_frame=body,
                )
                tf_dict[body] = tf_msg

        for asset in self._asset:
            asset.cfg.prim_path = asset.cfg.prim_path.replace("env_.*", f"env_{self._env_idx}")
            # if asset is an Articulation or RigidObject, add bodies to list of transforms
            if isinstance(asset, (Articulation, RigidObject)):
                tf_msg = TransformStamped()
                body = asset.body_names[0]
                frame_id_ = (
                    asset.cfg.prim_path.split("/")[-3]
                    if self.frame_id_path is None
                    else self.frame_id_path.split("/")[-1]
                )
            # if asset is a sensor add sensor pose to list of transforms
            elif isinstance(asset, SensorBase):
                tf_msg = TransformStamped()
                body = asset.cfg.prim_path.split("/")[-1]
                frame_id_ = (
                    asset.cfg.prim_path.split("/")[-2]
                    if self.frame_id_path is None
                    else self.frame_id_path.split("/")[-1]
                )
            else:
                raise TypeError(
                    f"Asset must be of type Articulation, RigidObject, or SensorBase. Provided: {type(asset)}"
                )

            ros_conversions.torch_to_transform(
                msg=tf_msg,
                obs=zero_transform,
                time=msg_time,
                frame_id=frame_id_,
                child_frame=body,
            )
            tf_dict[body] = tf_msg

        return tf_dict

    def _prepare_asset_tfs(self) -> None:
        """Prepares TransformStamped messages for assets provided in self.cfg.asset_cfg and stores
        them in self.tf_dict."""
        # get time
        msg_time = self.get_timestamp()

        # loop through provided assets and get transform from world to asset T_wa
        for asset in self._asset:
            # if asset is an Articulation, add the root link to the static publisher
            if isinstance(asset, (Articulation, RigidObject)):
                # access root link of articulation
                child_name_ = asset.body_names[0]
                root_quat_w = asset.data.root_quat_w[self._env_idx, :]
                root_pos_w = asset.data.root_pos_w[self._env_idx, :]

                # change relative transform
                if self.frame_id_path is None:
                    transform = torch.cat((root_pos_w, root_quat_w), dim=0)
                    frame_id_ = asset.cfg.prim_path.split("/")[-3]
                else:
                    root_pos_frame_id, root_quat_frame_id = math_utils.subtract_frame_transforms(
                        t01=self.frame_id_pos_w, q01=self.frame_id_quat_w, t02=root_pos_w, q02=root_quat_w
                    )
                    transform = torch.cat((root_pos_frame_id, root_quat_frame_id), dim=0)
                    frame_id_ = self.frame_id_path.split("/")[-1]

            # if asset is a sensor add sensor pose to list of transforms
            elif isinstance(asset, SensorBase):
                s_prim_path = asset.cfg.prim_path
                s_prim_path_list = s_prim_path.split("/")
                p_prim_path = "/".join(s_prim_path_list[:-1])
                child_name_ = s_prim_path_list[-1]

                # publish camera orientation using ros convention for camera frames
                if isinstance(asset, Camera):
                    s_pos_p = torch.tensor(asset.cfg.offset.pos)
                    s_quat_p = torch.tensor(asset.cfg.offset.rot, dtype=torch.float32).unsqueeze(0)
                    s_quat_p = math_utils.convert_orientation_convention(
                        s_quat_p, origin=asset.cfg.offset.convention, target="ros"
                    )
                    s_quat_p = s_quat_p.squeeze()
                else:
                    s_pos_p = torch.tensor(asset.cfg.offset.pos)
                    s_quat_p = torch.tensor(asset.cfg.offset.rot)

                # change relative transform
                if self.frame_id_path is None:
                    transform = torch.cat((s_pos_p, s_quat_p), dim=0)
                    frame_id_ = asset.cfg.prim_path.split("/")[-2]
                else:
                    # get parent pose w.r.t world
                    p_pos_w, p_quat_w = self._get_prim_world_pose_at_path(p_prim_path)
                    # convert to parent pose w.r.t frame_id
                    p_pos_frame_id, p_quat_frame_id = math_utils.subtract_frame_transforms(
                        t01=self.frame_id_pos_w, q01=self.frame_id_quat_w, t02=p_pos_w, q02=p_quat_w
                    )
                    # convert to sensor pose w.r.t frame_id
                    s_pos_frame_id, s_quat_frame_id = math_utils.combine_frame_transforms(
                        t01=p_pos_frame_id, q01=p_quat_frame_id, t12=s_pos_p, q12=s_quat_p
                    )
                    transform = torch.cat((s_pos_frame_id, s_quat_frame_id), dim=0)
                    frame_id_ = self.frame_id_path.split("/")[-1]

            # convert torch tensor to transform message
            ros_conversions.torch_to_transform(
                msg=self.tf_dict[child_name_],
                obs=transform,
                time=msg_time,
                frame_id=frame_id_,
                child_frame=child_name_,
            )

    def _prepare_prim_tfs(self) -> None:
        """Prepares TransformStamped messages for prims provided in self.cfg.additional_prim_paths
        and stores them in self.tf_dict."""

        # look through prim_paths and publish static tfs
        if self.additional_prim_paths is not None:
            for c_prim_path in self.additional_prim_paths:
                c_prim_path_list = c_prim_path.split("/")
                c_prim_name = c_prim_path_list[-1]
                p_prim_path_list = c_prim_path_list[:-1]

                # choose between provided parent prim in prims direct parent or self.frame_id
                if self.frame_id_path is None:
                    frame_id_ = c_prim_path_list[-2]
                    p_prim_path = "/".join(p_prim_path_list)

                    # call to update parent prim before getting child to parent frame
                    # TODO: figure out why prim poses are not updating in the GUI and in
                    # the backend without a direct call.
                    XFormPrim(p_prim_path)
                    # get child pose w.r.t world
                    c_pos_p, c_quat_p = self._get_prim_local_pose_at_path(c_prim_path)
                else:
                    frame_id_ = self.frame_id_path.split("/")[-1]
                    p_prim_path = self.frame_id_path

                    # get parent pose w.r.t world
                    p_pos_w, p_quat_w = self._get_prim_world_pose_at_path(p_prim_path)

                    # get child pose w.r.t world
                    c_pos_w, c_quat_w = self._get_prim_world_pose_at_path(c_prim_path)

                    # convert to child relative to parent frame
                    c_pos_p, c_quat_p = math_utils.subtract_frame_transforms(
                        t01=p_pos_w, q01=p_quat_w, t02=c_pos_w, q02=c_quat_w
                    )

                transform = torch.cat((c_pos_p, c_quat_p), dim=0)

                # convert torch tensor to transform message
                ros_conversions.torch_to_transform(
                    msg=self.tf_dict[c_prim_name],
                    obs=transform,
                    time=self.get_timestamp(),
                    frame_id=frame_id_,
                    child_frame=c_prim_name,
                )

    def _prepare_msg(self) -> None:
        """Updates a dictionary of transform messages (self.tf_dict) to be published."""
        # calculate frame_id transform
        if self.frame_id_path is not None:
            self.frame_id_pos_w, self.frame_id_quat_w = self._get_prim_world_pose_at_path(self.frame_id_path)
        # prepare tf messages for assets from self.asset_cfg
        self._prepare_asset_tfs()
        # prepare tf messages for prims in self.prim_paths
        self._prepare_prim_tfs()


class ContactForcePublisher(DirectPublisherBase):
    """
    A class to publish contact force messages in a ROS environment.

    This class initializes and publishes contact force messages based on the configuration
    provided. It validates the configuration against the contact sensor's body names and
    publishes the net contact forces.

    Attributes:
        cfg: Configuration for the publisher.
        env: Environment containing the scene and sensors.
        msg: The ROS message object to fill and publish.
        publisher: The ROS 2 publisher of type `msg`.
        topic_name: The name of the ROS 2 topic the `msg` is published over.
        asset: The SceneEntity used to fill this publisher's `msg`.
    """

    def _initialize_msg(self):
        """
        Initialize ROS message with zeros of type Float32MultiArray.

        Returns:
            Float32MultiArray: Initialized ROS message with layout and zero data, the label includes in the name of the links.
        """
        msg = Float32MultiArray()
        dim1 = MultiArrayDimension()

        body_names = (
            self.cfg.asset_cfg.body_names if self.cfg.asset_cfg.body_names is not None else self.asset.body_names
        )

        combined_string = " ".join(body_names)
        dim1.label = "Link names: " + combined_string

        dim1.size = len(body_names)
        dim1.stride = len(body_names) * 3

        dim2 = MultiArrayDimension()
        dim2.label = "Forces"
        dim2.size = 3
        dim2.stride = 3

        layout = MultiArrayLayout()
        layout.dim = [dim1, dim2]
        layout.data_offset = 0

        msg.layout = layout
        return msg

    def _prepare_msg(self):
        """Extracts asset data and package into message of type self.cfg.msg_type."""
        net_contact_forces = self._asset.data.net_forces_w[:, self.cfg.asset_cfg.body_ids, :]
        self.msg.data = net_contact_forces.flatten().tolist()


##
# Observation Based Publishers
##


class ObservationPublisherBase(PublisherBaseTerm):
    """Publisher for an entire IsaacLab observation group.

    Utilizes a dict[str, dict[str, torch.Tensor]] format for the observation output. This allows users to pass
    in the observation group to the publisher configs to facilitate extraction of desired observation data. The user
    is expected to create accompanying observations for this publisher to access.

    NOTE: In your ManagerBasedEnvCfg, be sure to add the following in the __post_init__ method.

    # disable observation concatenation for policy group - REQUIRED for observation based publishers
    self.observations.policy.concatenate_terms = False

    Attributes:
        cfg: Configuration for the publisher.
        node: ROS node to which the publisher is attached.
        env: Environment containing the scene and sensors.
        qos_profile: Quality of Service profile for the ROS publisher.
        env_idx: Index of the environment instance. Defaults to 0.
        msg: The ROS message object to fill and publish.
        publisher: The ROS 2 publisher of type `msg`.
        topic_name: The name of the ROS 2 topic the `msg` is published over.
    """

    def __init__(
        self,
        cfg: publishers_cfg.ObservationPublisherBaseCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the observation group publisher.

        Args:
            cfg: Configuration for the contact force publisher.
            node: ROS node to which the publisher is attached.
            env: Environment containing the scene and sensors.
            qos_profile: Quality of Service profile for the ROS publisher.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: Index of the environment instance. Defaults to 0.

        Raises:
            ValueError: If the requested observation group does not exist.
            RuntimeError: If the observation group concatenate_terms == True.
        """
        # call the base class constructor
        super().__init__(
            cfg=cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            use_sim_time=use_sim_time,
            env_idx=env_idx,
        )

        # set up conversion function
        self.to_ros_msg = ros_conversions.TO_ROS_MSG[cfg.msg_type]

        # Check for observation group
        if self.cfg.obs_group not in self._env.observation_manager.group_obs_dim.keys():
            raise ValueError(
                f"Observation group name: {self.cfg.obs_group} is not a valid observation group."
                f"Valid group names include: {self._env.observation_manager.group_obs_dim.keys()}"
            )

        # check whether group data is concatenated
        if env.observation_manager.group_obs_concatenate[self.cfg.obs_group]:
            raise RuntimeError(
                f"Observation group name: {self.cfg.obs_group} cannot be concatenated. Set concatenate_terms = False in"
                " the observation group."
            )

    @abstractmethod
    def _prepare_msg(self, obs: dict[str, dict[str, torch.Tensor]]):
        """Extract observation data and format it to the self.cfg.msg_type.

        Args:
            obs: Complete observation output from environment.
        """


class ObservationTermPublisher(ObservationPublisherBase):
    """Publisher for single observation term to a single ROS message field.

    The publisher is intended to be used for multiple types of messages that have a single data field besides a Header.

    Example uses of this publisher are the FlattenedObsPublisherCfg that specifies a Float32MultiArray data type and
    specifically flattens that observation.

    """

    def __init__(
        self,
        cfg: publishers_cfg.ObservationTermPublisherCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the observation term publisher.

        Args:
            cfg: Configuration for the publisher.
            node: ROS node to which the publisher is attached.
            env: Environment containing the scene and sensors.
            qos_profile: Quality of Service profile for the ROS publisher.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: Index of the environment instance. Defaults to 0.

        Raises:
            ValueError: If the requested observation term does not exist in the provided observation group.
        """

        # Check for observation term in group
        if cfg.obs_term_name not in env.observation_manager.active_terms[cfg.obs_group]:
            raise ValueError(
                f"Observation term name: {cfg.obs_term_name} is not a valid observation term in the observation group:"
                f" {cfg.obs_group}."
            )

        # Find index of observation term in group
        self.obs_term_index = env.observation_manager.active_terms[cfg.obs_group].index(cfg.obs_term_name)

        # call the base class constructor
        super().__init__(
            cfg=cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            use_sim_time=use_sim_time,
            env_idx=env_idx,
        )

    def _initialize_msg(self):
        """Initialize ROS message.

        Returns:
            ROS message of type cfg.msg_type.
        """
        if self.cfg.msg_type == Float32MultiArray:
            obs_size = self._env.observation_manager.group_obs_term_dim[self.cfg.obs_group][self.obs_term_index]

            if self.cfg.flatten:
                dim = [math.prod(obs_size)]
            else:
                dim = obs_size

            shape = []
            for d in dim:
                shape.append(MultiArrayDimension(size=d))

            msg = Float32MultiArray(layout=MultiArrayLayout(dim=shape))
        else:
            msg = self.cfg.msg_type()
        return msg

    def _prepare_msg(self, obs: dict[str, dict[str, torch.Tensor]]):
        """Extract observation data and format it to the self.cfg.msg_type.

        Args:
            obs: Complete observation output from environment.
        """
        data = obs[self.cfg.obs_group][self.cfg.obs_term_name].to("cpu")

        if self.cfg.flatten:
            data = data.view(-1)

        if hasattr(self.msg, "header"):
            self.to_ros_msg(self.msg, data, time=self.get_sim_timestamp())
        else:
            self.to_ros_msg(self.msg, data)


class JointStateObsPublisher(ObservationPublisherBase):
    """Publishes observation data that is in the form of joint position, velocity, and effort.

    The user is expected to properly define observations associated with joint position, velocity, and effort
    separately in the same observation group. The names of the observation group and the names of the observation
    terms are used to pass information to the JointStateObsPublisher allowing the publisher to access the correct
    observation. By utilizing the JointStateObsPublisher instead of the JointStatePublisher, users can take advantage
    of the data corruption/realism features of observations (e.g. noise, bias, etc) as well as customize the
    observation calculations (i.e. relative or absolute).

    NOTE: In your ManagerBasedEnvCfg, be sure to add the following in the __post_init__ method.

    # disable observation concatenation - REQUIRED for observation based publishers
    self.observations.policy.concatenate_terms = False

    Example:
    In your ObservationGroup define the joint state observations

    from isaaclab.envs.mdp import observations as obs

    @configclass
    class ObservationsCfg:

        @configclass
        class PolicyCfg(ObsGroup):

            joint_pos = ObsTerm(func=obs.joint_pos)
            joint_vel = ObsTerm(func=obs.joint_vel)
            joint_effort = ObsTerm(func=obs.joint_effort)

        # observation groups
        policy: PolicyCfg = PolicyCfg()

    In your RosManagerCfg define the joint state publisher
    class ExampleManagerCfg(RosManagerCfg):
        joint_state = JointStateObsPublisherCfg(
            topic="/joint_state",
            asset_cfg=SceneEntityCfg("robot"),
            obs_group="policy",
            position_obs="joint_pos",
            velocity_obs="joint_vel",
            effort_obs="joint_effort",
        )
    """

    def __init__(
        self,
        cfg: publishers_cfg.JointStateObsPublisherCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the joint state publisher.

        Args:
            cfg: Configuration for the joint state observation publisher.
            node: ROS node to which the publisher is attached.
            env: Environment containing the scene and sensors.
            qos_profile: Quality of Service profile for the ROS publisher.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: Index of the environment instance. Defaults to 0.

        Raises:
            ValueError: If the requested joint position observation term does not exist in the provided observation group.
            ValueError: If the requested joint velocity observation term does not exist in the provided observation group.
            ValueError: If the requested joint effort observation term does not exist in the provided observation group.
        """
        # call the base class constructor
        super().__init__(
            cfg=cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            use_sim_time=use_sim_time,
            env_idx=env_idx,
        )

        # resolve scene entity
        self.cfg.asset_cfg.resolve(env.scene)

        # store articulation handle
        self._asset = self._env.scene[self.cfg.asset_cfg.name]

        # initialize size of joint_state
        self.joint_state = torch.zeros_like(
            torch.vstack((
                self.asset.data.joint_pos[self._env_idx, self.cfg.asset_cfg.joint_ids],
                self.asset.data.joint_vel[self._env_idx, self.cfg.asset_cfg.joint_ids],
                self.asset.data.applied_torque[self._env_idx, self.cfg.asset_cfg.joint_ids],
            ))
        )

        # Checks for observation terms in group and warn for None
        if self.cfg.position_obs is not None:
            if self.cfg.position_obs not in self._env.observation_manager.active_terms[self.cfg.obs_group]:
                raise ValueError(
                    f"Joint position observation term name: {self.cfg.position_obs} is not a valid observation term in"
                    f" the observation group: {self.cfg.obs_group}."
                )
        else:
            carb.log_warn(
                "The JointStatePublisherCfg: position_obs is set to None. Defaulting to publishing zeros for position."
            )

        if self.cfg.position_obs is not None:
            if self.cfg.velocity_obs not in self._env.observation_manager.active_terms[self.cfg.obs_group]:
                raise ValueError(
                    f"Joint velocity observation term name: {self.cfg.velocity_obs} is not a valid observation term in"
                    f" the observation group: {self.cfg.obs_group}."
                )
        else:
            carb.log_warn(
                "The JointStatePublisherCfg: velocity_obs is set to None. Defaulting to publishing zeros for velocity."
            )

        if self.cfg.effort_obs is not None:
            if self.cfg.effort_obs not in self._env.observation_manager.active_terms[self.cfg.obs_group]:
                raise ValueError(
                    f"Joint effort observation term name: {self.cfg.effort_obs} is not a valid observation term in the"
                    f" observation group: {self.cfg.obs_group}."
                )
        else:
            carb.log_warn(
                "The JointStatePublisherCfg: effort_obs is set to None. Defaulting to publishing zeros for effort."
            )

        # populate joint name information
        self.msg.name = (
            self.cfg.asset_cfg.joint_names if self.cfg.asset_cfg.joint_names else self.asset.data.joint_names
        )

    @property
    def asset(self) -> Articulation:
        return self._asset

    def _prepare_msg(self, obs: dict[str, dict[str, torch.Tensor]]):
        """Extract the joint state observation data and format it to the self.cfg.msg_type.

        Args:
            obs: Complete observation output from environment.
        """

        # extract and populate the joint position, velocity, and effort
        if self.cfg.position_obs is not None:
            self.joint_state[0] = obs[self.cfg.obs_group][self.cfg.position_obs]
        if self.cfg.velocity_obs is not None:
            self.joint_state[1] = obs[self.cfg.obs_group][self.cfg.velocity_obs]
        if self.cfg.effort_obs is not None:
            self.joint_state[2] = obs[self.cfg.obs_group][self.cfg.effort_obs]

        # convert tensor to ROS message
        self.to_ros_msg(self.msg, self.joint_state, time=self.get_timestamp())


class JointReactionWrenchObsPublisher(ObservationPublisherBase):
    """Publishes observation data for an articulation's body_joint_incoming_wrench.

    The joint wrench at body frame (w.r.t body frame) from parent frame to body frame.

    This publisher uses a Float32MultiArray msg. Float32MultiArray message format will be:

        layout
            .dim
                .size = asset.num_bodies * 6 + 1
                .stride = 6 # for [Fx, Fy, Fz, Tx, Ty, Tz]
                .label = Time (sec), body_name1, body_name2, ... # 'asset.body_names' (comma delimited)
        data = time, body1_fx, body1_fy, body1_fz, body1_Tx, body1_Ty, body1_Tz, body2_Fx, ....

    Example use:

    In your ManagerBasedEnvironment's Observation config:

        from isaaclab.utils import configclass
        from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
        import isaaclab.envs.mdp.observations as obs

        @configclass
        class MyObservationCfg():

            @configclass
            MyObservationGroupCfg(ObservationGroupCfg):

                joint_reaction_term = obs.joint_reactions = ObservationTermCfg(
                    func=obs.body_incoming_wrench, params={"asset_cfg": SceneEntityCfg("robot")}
                )

            policy = MyObservationGroupCfg()

    In your RosManagerCfg:

        from isaaclab.utils import configclass
        from rai.eval_sim.ros_manager import RosManagerCfg
        from rai.eval_sim.ros_manager import publishers_cfg as pub_cfg

        @configclass
        class MyRosManagerCfg(RosManagerCfg):
            joint_reactions = pub_cfg.JointReactionWrenchObsPublisherCfg(
                asset_cfg=SceneEntity("robot"),
                topic="/joint_reactions",
                obs_group="policy",
                obs_term_name="joint_reactions_term"
            )

    NOTE:
    - The name of the observation term 'joint_reactions_term' must be the same as the string value of 'obs_term_name'
    in the JointReactionWrenchObsPublisherCfg.
    - The asset_cfg used in the ObservationTermCfg and the JointReactionWrenchObsPublisherCfg should point to the same
    articulation. If a subset of the bodies of an articulation are desired, set the body_names in the asset_cfg.
    """

    def __init__(
        self,
        cfg: publishers_cfg.JointReactionWrenchObsPublisherCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the joint state publisher.

        Args:
            cfg: Configuration for the publisher.
            node: ROS node to which the publisher is attached.
            env: Environment containing the scene and sensors.
            qos_profile: Quality of Service profile for the ROS publisher.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: Index of the environment instance. Defaults to 0.
        """
        # resolve scene entity
        cfg.asset_cfg.resolve(env.scene)

        # store articulation handle
        self._asset = env.scene[cfg.asset_cfg.name]

        # call the base class constructor
        super().__init__(
            cfg=cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            use_sim_time=use_sim_time,
            env_idx=env_idx,
        )

    def _initialize_msg(self):
        """Initialize the ROS 2 message for a series of joint reaction wrenches."""
        msg = Float32MultiArray()
        dim1 = MultiArrayDimension()

        combined_string = ", ".join(self._asset.body_names)
        dim1.label = "Time (sec), " + combined_string

        dim1.size = len(self._asset.body_names) * 6 + 1
        dim1.stride = 6

        layout = MultiArrayLayout()
        layout.dim = [dim1]
        layout.data_offset = 0

        msg.layout = layout
        return msg

    def _prepare_msg(self, obs: dict[str, dict[str, torch.Tensor]]):
        """Extract the joint wrench observation data and format it to the self.cfg.msg_type.

        Args:
            obs: Complete observation output dictionary.
        """
        time = self._env.sim.current_time
        link_incoming_forces = obs[self.cfg.obs_group][self.cfg.obs_term_name].view(-1).tolist()
        self.msg.data = [time] + link_incoming_forces


class LinkPoseObsPublisher(ObservationPublisherBase):
    """Publishes observation data that is in the form of link position and orientation (quaternion) [x y z qw qx qy qz].

    The user is expected to properly define observations associated with pose. The names of the observation group and the
    observation terms are used to pass information to the LinkPoseObsPublisher allowing the publisher to access the correct
    observation. By utilizing the LinkPoseObsPublisher instead of the LinkPosePublisher, users can take advantage of the
    data corruption/realism features of observations (e.g. noise, bias, etc) as well as customize the observation calculations.

    NOTE: In your ManagerBasedEnvCfg, be sure to add the following in the __post_init__ method.

    # disable observation concatenation - REQUIRED for observation based publishers
    self.observations.policy.concatenate_terms = False

    Example:
    In your ObservationGroup define the link pose observations

    from isaaclab.envs.mdp import observations as obs
    @configclass
    class ObservationsCfg:

        @configclass
        class PolicyCfg(ObsGroup):

            base_link_pose = ObsTerm(func=obs.link_pose, params={"link_name": "base"})

        # observation groups
        policy: PolicyCfg = PolicyCfg()

    In your RosManagerCfg define the link pose publisher

    class ExampleManagerCfg(RosManagerCfg):
        base_pose = LinkPoseObsPublisherCfg(
            topic="/base_pose",
            obs_group="policy",
            asset_cfg=ENTITY,
            link_pose_obs="base_link_pose"
        )
    """

    def __init__(
        self,
        cfg: publishers_cfg.LinkPoseObsPublisherCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the link pose publisher term.

        Args:
            cfg: Configuration for the publisher.
            node: ROS node to which the publisher is attached.
            env: Environment containing the scene and sensors.
            qos_profile: Quality of Service profile for the ROS publisher.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: Index of the environment instance. Defaults to 0.
        """
        # call the base class constructor
        super().__init__(
            cfg=cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            use_sim_time=use_sim_time,
            env_idx=env_idx,
        )

        self.link_pose = torch.zeros(7)
        if (self.cfg.link_pose_obs is None) and ((self.cfg.link_pos_obs is None) or (self.cfg.link_quat_obs is None)):
            raise ValueError(
                "LinkPoseObsPublisherCfg must have link_pose_obs or link_pos_obs and link_quat_obs both set."
            )

        self.frame_id = self.cfg.frame_id

    def _prepare_msg(self, obs: dict[str, dict[str, torch.Tensor]]):
        """Extract the link pose observation data and format it to the self.cfg.msg_type.

        Args:
            obs: Complete observation output dictionary.
        """

        # extract link pose from observations
        if self.cfg.link_pose_obs is not None:
            self.link_pose = torch.squeeze(obs[self.cfg.obs_group][self.cfg.link_pose_obs])
        else:
            self.link_pose = torch.cat((
                torch.squeeze(obs[self.cfg.obs_group][self.cfg.link_pos_obs]),
                torch.squeeze(obs[self.cfg.obs_group][self.cfg.link_quat_obs]),
            ))
        # convert and apply to message
        self.to_ros_msg(msg=self.msg, obs=self.link_pose, time=self.get_timestamp(), frame_id=self.frame_id)


class TwistObsPublisher(ObservationPublisherBase):
    """Publishes observation data that is in the form of linear velocity and angular velocity [x y z qw qx qy qz].

    The user is expected to properly define observations associated with each velocity. The names of the
    observation group and the observation terms are used to pass information to the TwistObsPublisher
    allowing the publisher to access the correct observation. By utilizing the TwistObsPublisher
    instead of the TwistPublisher, users can take advantage of the data corruption/realism features
    of observations (e.g. noise, bias, etc) as well as customize the observation calculations.

    NOTE: In your ManagerBasedEnvCfg, be sure to add the following in the __post_init__ method.

    # disable observation concatenation - REQUIRED for observation based publishers
    self.observations.policy.concatenate_terms = False

    Example:
    In your ObservationGroup define the linear and angular velocity observations.

    from isaaclab.envs.mdp import observations as obs

    @configclass
    class ObservationsCfg:

        @configclass
        class PolicyCfg(ObsGroup):

            base_lin_vel = ObsTerm(func=obs.base_lin_vel)
            base_ang_vel = ObsTerm(func=obs.base_ang_vel)

        # observation groups
        policy: PolicyCfg = PolicyCfg()

    In your RosManagerCfg define the twist publisher.

    class ExampleManagerCfg(RosManagerCfg):
        base_twist = TwistObsPublisherCfg(
            topic="base_twist",
            asset_cfg=ENTITY,
            obs_group="policy",
            frame_id="world",
            lin_vel_obs="base_lin_vel",
            ang_vel_obs="base_ang_vel",
        )

    """

    def __init__(
        self,
        cfg: publishers_cfg.TwistObsPublisherCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the base twist publisher term.

        Args:
            cfg: Configuration for the publisher.
            node: ROS node to which the publisher is attached.
            env: Environment containing the scene and sensors.
            qos_profile: Quality of Service profile for the ROS publisher.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: Index of the environment instance. Defaults to 0.
        """
        # call the base class constructor
        super().__init__(cfg, node, env, qos_profile, use_sim_time, env_idx)
        self.twist = torch.zeros(6, device=env.device)
        self.frame_id = self.cfg.frame_id

    def _prepare_msg(self, obs: dict[str, dict[str, torch.Tensor]]):
        """Extract the twist observation data and format it to the self.cfg.msg_type.

        Args:
            obs: Complete observation output dictionary from environment.
        """

        # extract pose and quaternion from observation
        self.twist.view(-1)[:3] = torch.squeeze(obs[self.cfg.obs_group][self.cfg.lin_vel_obs])
        self.twist.view(-1)[3:] = torch.squeeze(obs[self.cfg.obs_group][self.cfg.ang_vel_obs])
        # convert and apply to message
        self.to_ros_msg(msg=self.msg, obs=self.twist, time=self.get_timestamp(), frame_id=self.frame_id)


class ImuObsPublisher(ObservationPublisherBase):
    """Publishes observation data that is in the form of a quaternion, angular velocity, and linear acceleration.

    This data typically comes from an IMU sensor observation. The user is responsible for defining a set of IMU
    observations.

    Example:

    #
    # In your InteractiveSceneCfg define the imu sensor.
    #
    from isaaclab.sensor.imu import ImuCfg
    @configclass
    class MySceneCfg(InteractiveSceneCfg)
        imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=ImuCfg.OffsetCfg(
                pos=(-0.25565, 0.00255, 0.07672),
                rot=(0.0, 0.0, 1.0, 0.0),
            ),
        )

    #
    # In your ObservationGroupCfg define the IMU observations tied to that sensor
    #

    from isaaclab.env.mdp import observations as obs

    @configclass
    class ObservationsCfg:

        @configclass
        class PolicyCfg(ObsGroup):

        imu_quat = ObsTerm(
            func=obs.imu_orientation,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        imu_ang_vel = ObsTerm(
            func=obs.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        imu_lin_acc = ObsTerm(
            func=obs.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )

        policy = PolicyCfg()

    #
    # In your RosManagerCfg define the IMU publisher.
    #

    @configclass
    class MyRosManagerCfg(RosManagerCfg):
        imu = ImuObsPublisherCfg(
            topic="/imu",
            obs_group="policy",
            imu_quat_obs="imu_quat",
            imu_ang_vel_obs="imu_ang_vel",
            imu_lin_acc_obs="imu_lin_acc",
        )

    """

    def __init__(
        self,
        cfg: publishers_cfg.ImuObsPublisherCfg,
        node: Node,
        env: ManagerBasedEnv,
        qos_profile: QoSProfile,
        use_sim_time: bool,
        env_idx: int = 0,
    ):
        """Initialize the imu publisher term.

        Args:
            cfg: Configuration for the publisher.
            node: ROS node to which the publisher is attached.
            env: Environment containing the scene and sensors.
            qos_profile: Quality of Service profile for the ROS publisher.
            use_sim_time: If true, use the env's simulation time to fill timestamps. Else, use the ros node's time.
            env_idx: Index of the environment instance. Defaults to 0.
        """
        # call the base class constructor
        super().__init__(
            cfg=cfg,
            node=node,
            env=env,
            qos_profile=qos_profile,
            use_sim_time=use_sim_time,
            env_idx=env_idx,
        )
        if self.cfg.imu_quat_obs is not None:
            if self.cfg.imu_quat_obs not in self._env.observation_manager.active_terms[self.cfg.obs_group]:
                raise ValueError(
                    f"ImuObsPublisherCfg: imu orientation observation term name: {self.cfg.imu_quat_obs} is not a valid"
                    f" observation term in the observation group: {self.cfg.obs_group}."
                )
        else:
            carb.log_warn(
                "The ImuObsPublisherCfg: imu_quat_obs is set to None. Defaulting to publishing zeros for orientation."
            )
        if self.cfg.imu_ang_vel_obs is not None:
            if self.cfg.imu_ang_vel_obs not in self._env.observation_manager.active_terms[self.cfg.obs_group]:
                raise ValueError(
                    f"ImuObsPublisherCfg: imu orientation observation term name: {self.cfg.imu_ang_vel_obs} is not a"
                    f" valid observation term in the observation group: {self.cfg.obs_group}."
                )
        else:
            carb.log_warn(
                "The ImuObsPublisherCfg: imu_ang_vel_obs is set to None. Defaulting to publishing zeros for"
                " angular_velocity."
            )
        if self.cfg.imu_lin_acc_obs is not None:
            if self.cfg.imu_lin_acc_obs not in self._env.observation_manager.active_terms[self.cfg.obs_group]:
                raise ValueError(
                    f"ImuObsPublisherCfg: imu orientation observation term name: {self.cfg.imu_lin_acc_obs} is not a"
                    f" valid observation term in the observation group: {self.cfg.obs_group}."
                )
        else:
            carb.log_warn(
                "The ImuObsPublisherCfg: imu_lin_acc_obs is set to None. Defaulting to publishing zeros for"
                " linear_acceleration."
            )

        self.quat_w = torch.zeros(4)
        self.ang_vel_b = torch.zeros(3)
        self.lin_acc_b = torch.zeros(3)

    def _prepare_msg(self, obs: dict[str, dict[str, torch.Tensor]]):
        """Extract the IMU observation data and format it to the self.cfg.msg_type.

        Args:
            obs: Complete observation output dictionary.
        """
        if self.cfg.imu_quat_obs is not None:
            self.quat_w[:] = torch.squeeze(obs[self.cfg.obs_group][self.cfg.imu_quat_obs])
        if self.cfg.imu_ang_vel_obs is not None:
            self.ang_vel_b[:] = torch.squeeze(obs[self.cfg.obs_group][self.cfg.imu_ang_vel_obs])
        if self.cfg.imu_lin_acc_obs is not None:
            self.lin_acc_b[:] = torch.squeeze(obs[self.cfg.obs_group][self.cfg.imu_lin_acc_obs])

        self.to_ros_msg(
            msg=self.msg,
            obs=torch.cat((self.quat_w, self.ang_vel_b, self.lin_acc_b), dim=-1),
            time=self.get_timestamp(),
        )
