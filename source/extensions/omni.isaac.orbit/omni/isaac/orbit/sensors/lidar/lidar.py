# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import re
import torch
from collections.abc import Sequence

from typing import TYPE_CHECKING, Literal

import omni.kit.commands
import omni.usd
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.range_sensor import _range_sensor
import omni.isaac.RangeSensorSchema as RangeSensorSchema
from omni.isaac.orbit.utils.array import convert_to_torch
from omni.isaac.orbit.utils.math import quat_from_matrix
from tensordict import TensorDict

from ..sensor_base import SensorBase
from .lidar_data import LidarData
from .utils import convert_orientation_convention, create_rotation_matrix_from_view

if TYPE_CHECKING:
    from .lidar_cfg import LidarCfg


class Lidar(SensorBase):
    r"""The lidar sensor for acquiring 2D and 3D visual depth data.

    This class wraps over the `_range_sensor` for providing a consistent API for acquiring visual lidar data.

    The following data types are supported:

    - ``"azimuth"``: The azimuth angle in radians for each column.
    - ``"depth"``: The distance from the sensor to the hit for each beam in uint16 and scaled by min and max distance.
    - ``"intensity"``: The observed specular intensity of each beam, 255 if hit, 0 if not.
    - ``"linear_depth"``: The distance from the sensor to the hit for each beam in meters.
    - ``"num_cols"``: The number of vertical scans of the sensor, 0 if error occurred.
    - ``"num_cols_ticked"``: The number of vertical scans the sensor completed in the last simulation step, 0 if error occurred. Generally only useful for lidars with a non-zero rotation speed.
    - ``"num_rows"``: The number of horizontal scans of the sensor, 0 if error occurred.
    - ``"point_cloud"``: The hit position in xyz relative to the sensor origin, not accounting for individual ray offsets.
    - ``"zenith"``: The zenith angle in radians for each row.

    Note: The class does not spawn the lidar sensor. It assumes that the lidar sensor is already present in the simulation environment. Please refer to the lidar examples below on how to set this up.

    Note: Getting segmentaion data from the lidar sensor is not yet supported.

    .. _range_sensor extension: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.range_sensor/docs/index.html
    .. Lidar Examples: https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_range_sensor_lidar.html

    """

    cfg: LidarCfg
    """The configuration parameters."""

    def __init__(self, cfg: LidarCfg):
        """Initializes the lidar sensor.

        Args:
            cfg: The configuration parameters.

        Raises:
            RuntimeError: If no lidar prim is found at the given path.
        """
        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the lidar sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )

        # initialize base class
        super().__init__(cfg)

        # Create empty variables for storing output data
        self._data = LidarData()
        self._li = _range_sensor.acquire_lidar_sensor_interface()

        self.num_beams = int(cfg.horizontal_fov / cfg.horizontal_resolution)

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""
        # unsubscribe callbacks
        super().__del__()
        # delete from replicator registry

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Lidar @ '{self.cfg.prim_path}': \n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> LidarData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame

    """
    Configuration
    """

    def set_lidar_properties(self):
        """ "If any value is set to None, the default value is used."""

        env_ids = self._ALL_INDICES

        # Iterate over environment IDs
        for i in env_ids:
            # Get corresponding lidar prim path
            lidar = RangeSensorSchema.Lidar.Define(
                omni.usd.get_context().get_stage(), self._view.prim_paths[i]
            )

            if self.cfg.horizontal_fov is not None:
                lidar.GetHorizontalFovAttr().Set(self.cfg.horizontal_fov)

            if self.cfg.horizontal_resolution is not None:
                lidar.GetHorizontalResolutionAttr().Set(self.cfg.horizontal_resolution)

            if self.cfg.max_range is not None:
                lidar.GetMaxRangeAttr().Set(self.cfg.max_range)

            if self.cfg.min_range is not None:
                lidar.GetMinRangeAttr().Set(self.cfg.min_range)

            if self.cfg.rotation_rate is not None:
                lidar.GetRotationRateAttr().Set(self.cfg.rotation_rate)

            if self.cfg.vertical_fov is not None:
                lidar.GetVerticalFovAttr().Set(self.cfg.vertical_fov)

            if self.cfg.vertical_resolution is not None:
                lidar.GetVerticalResolutionAttr().Set(self.cfg.vertical_resolution)

            if self.cfg.draw_lines is not None:
                lidar.GetDrawLinesAttr().Set(self.cfg.draw_lines)

            if self.cfg.draw_points is not None:
                lidar.GetDrawPointsAttr().Set(self.cfg.draw_points)

            if self.cfg.high_lod is not None:
                lidar.GetHighLodAttr().Set(self.cfg.high_lod)

            if self.cfg.yaw_offset is not None:
                lidar.GetYawOffsetAttr().Set(self.cfg.yaw_offset)

            # Missing enabeling of the segmentation

    """
    Operations - Set pose.
    """

    # Checked - this method does not change any behaviour
    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
        convention: Literal["opengl", "ros", "world"] = "ros",
    ):
        r"""Set the pose of the camera w.r.t. the world frame using specified convention.

        Since different fields use different conventions for camera orientations, the method allows users to
        set the camera poses in the specified convention. Possible conventions are:

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        See :meth:`omni.isaac.orbit.sensors.camera.utils.convert_orientation_convention` for more details
        on the conventions.

        Args:
            positions: The cartesian coordinates (in meters). Shape is (N, 3).
                Defaults to None, in which case the camera position in not changed.
            orientations: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the camera orientation in not changed.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
            convention: The convention in which the poses are fed. Defaults to "ros".

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # convert to backend tensor
        if positions is not None:
            if isinstance(positions, np.ndarray):
                positions = torch.from_numpy(positions).to(device=self._device)
            elif not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions, device=self._device)
        # convert rotation matrix from input convention to OpenGL
        if orientations is not None:
            if isinstance(orientations, np.ndarray):
                orientations = torch.from_numpy(orientations).to(device=self._device)
            elif not isinstance(orientations, torch.Tensor):
                orientations = torch.tensor(orientations, device=self._device)
            orientations = convert_orientation_convention(
                orientations, origin=convention, target="opengl"
            )
        # set the pose
        self._view.set_world_poses(positions, orientations, env_ids)

    def set_world_poses_from_view(
        self,
        eyes: torch.Tensor,
        targets: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes: The positions of the camera's eye. Shape is (N, 3).
            targets: The target locations to look at. Shape is (N, 3).
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # set camera poses using the view
        orientations = quat_from_matrix(
            create_rotation_matrix_from_view(eyes, targets, device=self._device)
        )
        self._view.set_world_poses(eyes, orientations, env_ids)

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        # note: cannot do smart indexing here since we do a for loop over data.
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # reset the data
        # note: this recomputation is useful if one performs randomization on the camera poses.
        self._update_poses(env_ids)
        # self._update_intrinsic_matrices(env_ids)
        # Reset the frame count
        self._frame[env_ids] = 0

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the lidar sensor handles and internal buffers.

        This function prepares the lidar sensor for data collection, ensuring it is properly configured within the simulation environment. It also initializes the internal buffers to store the lidar data.

        Raises:
            RuntimeError: If the number of lidar prims in the view does not match the expected number.
        """
        import omni.replicator.core as rep

        # Initialize the base class
        super()._initialize_impl()

        # Prepare a view for the lidar sensor based on its path
        self._view = XFormPrimView(self.cfg.prim_path, reset_xform_properties=False)
        self._view.initialize()

        # Ensure the number of detected lidar prims matches the expected number
        if self._view.count != self._num_envs:
            raise RuntimeError(
                f"Expected number of lidar prims ({self._num_envs}) does not match the found number ({self._view.count})."
            )

        # Prepare environment ID buffers
        self._ALL_INDICES = torch.arange(
            self._view.count, device=self._device, dtype=torch.long
        )

        # Initialize a frame count buffer
        self._frame = torch.zeros(
            self._view.count, device=self._device, dtype=torch.long
        )

        # Resolve device name
        if "cuda" in self._device:
            device_name = self._device.split(":")[0]
        else:
            device_name = "cpu"

        self.set_lidar_properties()

        # Create internal buffers for lidar data
        self._create_buffers()

    def _is_valid_lidar_prim(self, prim):
        # Checking if a USD prim is a valid lidar sensor in simulation environment.
        return self._li.is_lidar_sensor(prim)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # Increment frame count
        self._frame[env_ids] += 1
        # -- pose
        self._update_poses(env_ids)

        for index in env_ids:

            # Query the data from the LiDAR sensor
            azimuth = self._li.get_azimuth_data(self._view.prim_paths[index])
            depth_data = self._li.get_depth_data(self._view.prim_paths[index])
            intensity_data = self._li.get_intensity_data(self._view.prim_paths[index])
            linear_depth = self._li.get_linear_depth_data(self._view.prim_paths[index])
            num_cols = self._li.get_num_cols(self._view.prim_paths[index])
            num_cols_ticked = self._li.get_num_cols_ticked(self._view.prim_paths[index])
            num_rows = self._li.get_num_rows(self._view.prim_paths[index])
            point_cloud = self._li.get_point_cloud_data(self._view.prim_paths[index])
            zenith = self._li.get_zenith_data(self._view.prim_paths[index])

            # Convert data to torch tensors
            azimuth_tensor = convert_to_torch(azimuth, device=self.device).squeeze()
            depth_tensor = convert_to_torch(
                depth_data.astype(np.int32), device=self.device
            ).squeeze()
            intensity_tensor = convert_to_torch(
                intensity_data, device=self.device
            ).squeeze()
            linear_depth_tensor = convert_to_torch(
                linear_depth, device=self.device
            ).squeeze()
            point_cloud_tensor = convert_to_torch(
                point_cloud, device=self.device
            ).squeeze()
            zenith_tensor = convert_to_torch(zenith, device=self.device).squeeze()

            # Initialize tensors in TensorDict if they do not exist
            if "azimuth" not in self._data.output.keys():
                self._data.output["azimuth"] = torch.zeros(
                    (self._view.count, azimuth_tensor.size(0)), device=self._device
                )
            if "depth" not in self._data.output.keys():
                self._data.output["depth"] = torch.zeros(
                    (self._view.count, depth_tensor.size(0)), device=self._device
                )
            if "intensity" not in self._data.output.keys():
                self._data.output["intensity"] = torch.zeros(
                    (self._view.count, intensity_tensor.size(0)), device=self._device
                )
            if "linear_depth" not in self._data.output.keys():
                self._data.output["linear_depth"] = torch.zeros(
                    (self._view.count, linear_depth_tensor.size(0)), device=self._device
                )
            if "num_cols" not in self._data.output.keys():
                self._data.output["num_cols"] = torch.zeros(
                    (self._view.count,), device=self._device, dtype=torch.int32
                )
            if "num_cols_ticked" not in self._data.output.keys():
                self._data.output["num_cols_ticked"] = torch.zeros(
                    (self._view.count,), device=self._device, dtype=torch.int32
                )
            if "num_rows" not in self._data.output.keys():
                self._data.output["num_rows"] = torch.zeros(
                    (self._view.count,), device=self._device, dtype=torch.int32
                )
            if "point_cloud" not in self._data.output.keys():
                self._data.output["point_cloud"] = torch.zeros(
                    (
                        self._view.count,
                        point_cloud_tensor.size(0),
                        point_cloud_tensor.size(1),
                    ),
                    device=self._device,
                )
            if "zenith" not in self._data.output.keys():
                if zenith_tensor.ndimension() == 0:
                    zenith_tensor = zenith_tensor.unsqueeze(0)
                self._data.output["zenith"] = torch.zeros(
                    (self._view.count, zenith_tensor.size(0)), device=self._device
                )

            # Update the TensorDict
            self._data.output["azimuth"][index] = azimuth_tensor
            self._data.output["depth"][index] = depth_tensor
            self._data.output["intensity"][index] = intensity_tensor
            self._data.output["linear_depth"][index] = linear_depth_tensor
            self._data.output["num_cols"][index] = torch.tensor(
                num_cols, device=self.device
            )
            self._data.output["num_cols_ticked"][index] = torch.tensor(
                num_cols_ticked, device=self.device
            )
            self._data.output["num_rows"][index] = torch.tensor(
                num_rows, device=self.device
            )
            self._data.output["point_cloud"][index] = point_cloud_tensor
            self._data.output["zenith"][index] = zenith_tensor

    """
    Private Helpers
    """

    def _create_buffers(self):
        """Create buffers for storing lidar distance measurement data."""
        # Pose of the lidar sensors in the world
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros(
            (self._view.count, 4), device=self._device
        )

        # Initialize the TensorDict with zero-sized tensors as placeholders
        self._data.output = TensorDict(
            {}, batch_size=self._view.count, device=self.device
        )

    def _update_poses(self, env_ids: Sequence[int]):
        """Computes the pose of the lidar in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose.

        Returns:
            A tuple of the position (in meters) and quaternion (w, x, y, z).
        """

        # get the poses from the view
        poses, quat = self._view.get_world_poses(env_ids)
        self._data.pos_w[env_ids] = poses
        self._data.quat_w_world[env_ids] = convert_orientation_convention(
            quat, origin="opengl", target="world"
        )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
