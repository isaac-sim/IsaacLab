# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import re
import torch
from collections.abc import Sequence
from tensordict import TensorDict
from typing import TYPE_CHECKING, Any

import carb
import omni.kit.commands
import omni.usd
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.array import convert_to_torch

from ..sensor_base import SensorBase
from .rtx_lidar_data import RTX_LIDAR_INFO_FIELDS, RtxLidarData

if TYPE_CHECKING:
    from .rtx_lidar_cfg import RtxLidarCfg


class RtxLidar(SensorBase):
    """The RTX Lidar sensor to acquire render based lidar data.

    This class wraps over the `UsdGeom Camera`_ for providing a consistent API for acquiring LiDAR data.

    This implementation utilizes the "RtxSensorCpuIsaacCreateRTXLidarScanBuffer" annotator.

    RTX lidar: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_based_lidar.html
    LiDAR config files: https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.lidar/latest/lidar_extension.html
    RTX lidar Annotators: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_based_lidar/annotator_descriptions.html
    """

    cfg: RtxLidarCfg

    def __init__(self, cfg: RtxLidarCfg):
        """Initializes the RTX lidar object.

        Args:
            cfg: The configuration parameters.

        Raises:
            RuntimeError: If provided path is a regex.
            RuntimeError: If no camera prim is found at the given path.
        """
        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the rtx lidar sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )

        # Initialize base class
        super().__init__(cfg)

        # toggle rendering of rtx sensors as True
        # this flag is read by SimulationContext to determine if rtx sensors should be rendered
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/isaaclab/render/rtx_sensors", True)
        # spawn the asset
        if self.cfg.spawn is not None:
            self.cfg.spawn.func(
                self.cfg.prim_path, self.cfg.spawn, translation=self.cfg.offset.pos, orientation=self.cfg.offset.rot
            )
        # check that spawn was successful
        matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if len(matching_prims) == 0:
            raise RuntimeError(f"Could not find prim with path {self.cfg.prim_path}.")

        self._sensor_prims: list[UsdGeom.Camera] = list()
        # Create empty variables for storing output data
        self._data = RtxLidarData()

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry and clean up any custom lidar configs."""

        # delete from replicator registry
        for annotator, render_product_path in zip(self._rep_registry, self._render_product_paths):
            annotator.detach([render_product_path])
            annotator = None
        # delete custom lidar config temp files
        if self.cfg.spawn.lidar_type == "Custom":
            file_dir = self.cfg.spawn.sensor_profile_temp_dir
            if os.path.isdir(file_dir):
                for file in os.listdir(file_dir):
                    if self.cfg.spawn.sensor_profile_temp_prefix in file and os.path.isfile(
                        os.path.join(file_dir, file)
                    ):
                        os.remove(os.path.join(file_dir, file))

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return 1

    @property
    def data(self) -> RtxLidarData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def render_product_paths(self) -> list[str]:
        """The path of the render products for the cameras.

        This can be used via replicator interfaces to attach to writes or external annotator registry.
        """
        return self._render_product_paths

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        if not self._is_initialized:
            raise RuntimeError(
                "Camera could not be initialized. Please ensure --enable_cameras is used to enable rendering."
            )
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        # note: cannot do smart indexing here since we do a for loop over data.
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # reset the data
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        # self._update_poses(env_ids)
        # Reset the frame count
        self._frame[env_ids] = 0

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        This function creates handles and registers the optional output data fields with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the enable_camera flag is not set.
            RuntimeError: If the number of camera prims in the view does not match the number of environments.
            RuntimeError: If the provided USD prim is not a Camera.
        """
        carb_settings_iface = carb.settings.get_settings()
        if not carb_settings_iface.get("/isaaclab/cameras_enabled"):
            raise RuntimeError(
                "A camera was spawned without the --enable_cameras flag. Please use --enable_cameras to enable"
                " rendering."
            )

        import omni.replicator.core as rep

        super()._initialize_impl()
        # Create a view for the sensor
        self._view = XFormPrimView(self.cfg.prim_path, reset_xform_properties=False)
        self._view.initialize()
        # Check that sizes are correct
        if self._view.count != self._num_envs:
            raise RuntimeError(
                f"Number of camera prims in the view ({self._view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )

        # Create all env_ids buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)

        # lidar_prim_paths = sim_utils.find_matching_prims(self.cfg.prim_path)

        self._render_product_paths: list[str] = list()
        self._rep_registry: list[rep.annotators.Annotator] = []

        # Obtain current stage
        stage = omni.usd.get_context().get_stage()

        for lidar_prim_path in self._view.prim_paths:
            # Get lidar prim
            lidar_prim = stage.GetPrimAtPath(lidar_prim_path)
            # Check if prim is a camera
            if not lidar_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{lidar_prim_path}' is not a Camera.")
            # Add to list
            sensor_prim = UsdGeom.Camera(lidar_prim)
            self._sensor_prims.append(sensor_prim)

            init_params = {
                "outputAzimuth": False,
                "outputElevation": False,
                "outputNormal": False,
                "outputVelocity": False,
                "outputBeamId": False,
                "outputEmitterId": False,
                "outputMaterialId": False,
                "outputObjectId": False,
                "outputTimestamp": True,  # always turn on timestamp field
            }

            # create annotator node
            annotator_type = "RtxSensorCpuIsaacCreateRTXLidarScanBuffer"
            rep_annotator = rep.AnnotatorRegistry.get_annotator(annotator_type)
            # turn on any optional data type returns
            for name in self.cfg.optional_data_types:
                if name == "azimuth":
                    init_params["outputAzimuth"] = True
                elif name == "elevation":
                    init_params["outputElevation"] = True
                elif name == "normal":
                    init_params["outputNormal"] = True
                elif name == "velocity":
                    init_params["outputVelocity"] = True
                elif name == "beamId":
                    init_params["outputBeamId"] = True
                elif name == "emitterId":
                    init_params["outputEmitterId"] = True
                elif name == "materialId":
                    init_params["outputMaterialId"] = True
                elif name == "objectId":
                    init_params["outputObjectId"] = True

            # transform the data output to be relative to sensor frame
            if self.cfg.data_frame == "sensor":
                init_params["transformPoints"] = False

            rep_annotator.initialize(**init_params)

            # Get render product
            # From Isaac Sim 2023.1 onwards, render product is a HydraTexture so we need to extract the path
            render_prod_path = rep.create.render_product(lidar_prim_path, [1, 1])
            if not isinstance(render_prod_path, str):
                render_prod_path = render_prod_path.path
            self._render_product_paths.append(render_prod_path)

            rep_annotator.attach(render_prod_path)
            self._rep_registry.append(rep_annotator)

            # Debug draw
            if self.cfg.debug_vis:
                self.writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
                self.writer.attach([render_prod_path])

        # Create internal buffers
        self._create_buffers()

    def _create_buffers(self):
        """Create buffers for storing data."""
        # create the data object

        # -- output data
        # lazy allocation of data dictionary
        # since the size of the output data is not known in advance, we leave it as None
        # the memory will be allocated when the buffer() function is called for the first time.
        self._data.output = TensorDict({}, batch_size=self._view.count, device=self.device)
        self._data.info = [{name: None for name in RTX_LIDAR_INFO_FIELDS.keys()} for _ in range(self._view.count)]

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Compute and fill sensor data buffers."""
        # Increment frame count
        self._frame[env_ids] += 1
        data_all_lidar = list()
        info_data_all_lidar: dict[str, list] = {}

        # iterate over all the annotators
        for index in env_ids:
            # get the output
            output = self._rep_registry[index].get_data()
            # process the output
            data, info = self._process_annotator_output("", output)

            # add data to output
            data_all_lidar.append(data)

            # store the info
            for info_key, info_value in info.items():
                if info_key in RTX_LIDAR_INFO_FIELDS.keys():
                    if info_key == "transform":
                        self._data.info[index][info_key] = torch.tensor(info_value, device=self.device)
                    else:
                        self._data.info[index][info_key] = info_value
                else:
                    if info_key not in info_data_all_lidar:
                        info_data_all_lidar[info_key] = [torch.tensor(info_value, device=self._device)]
                    else:
                        info_data_all_lidar[info_key].append(torch.tensor(info_value, device=self._device))

            # concatenate the data along the batch dimension
            self._data.output["data"] = torch.stack(data_all_lidar, dim=0)

            for key in info_data_all_lidar:
                self._data.output[key] = torch.stack(info_data_all_lidar[key], dim=0)

    def _process_annotator_output(self, name: str, output: Any) -> tuple[torch.tensor, dict | None]:
        """Process the annotator output.

        This function is called after the data has been collected from all the cameras.
        """
        # extract info and data from the output
        if isinstance(output, dict):
            data = output["data"]
            info = output["info"]
        else:
            data = output
            info = None
        # convert data into torch tensor
        data = convert_to_torch(data, device=self.device)

        # process data for different segmentation types
        # Note: Replicator returns raw buffers of dtype int32 for segmentation types
        #   so we need to convert them to uint8 4 channel images for colorized types

        return data, info

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
