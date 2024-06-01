# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
from collections.abc import Sequence
from tensordict import TensorDict
from typing import TYPE_CHECKING, Any

import omni.usd
import warp as wp
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom

from omni.isaac.lab.utils.warp.kernels import reshape_tiled_image

from ..sensor_base import SensorBase
from .camera import Camera

if TYPE_CHECKING:
    from .camera_cfg import TiledCameraCfg


class TiledCamera(Camera):
    r"""The tiled rendering camera sensor for acquiring RGB and depth data.

    This class wraps over the `UsdGeom Camera`_ for providing a consistent API for acquiring visual data.
    It ensures that the camera follows the ROS convention for the coordinate system.

    The following sensor types are supported:

    - ``"rgb"``: A rendered color image.
    - ``"depth"``: An image containing the distance to camera optical center.

    .. _USDGeom Camera: https://graphics.pixar.com/usd/docs/api/class_usd_geom_camera.html

    """

    cfg: TiledCameraCfg
    """The configuration parameters."""

    SUPPORTED_TYPES: set[str] = {"rgb", "depth"}
    """The set of sensor types that are supported."""

    def __init__(self, cfg: TiledCameraCfg):
        """Initializes the tiled camera sensor.

        Args:
            cfg: The configuration parameters.

        Raises:
            RuntimeError: If no camera prim is found at the given path.
            ValueError: If the provided data types are not supported by the camera.
        """
        super().__init__(cfg)

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""
        SensorBase.__del__(self)
        self._annotator.detach(self.render_product_paths)

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Tiled Camera @ '{self.cfg.prim_path}': \n"
            f"\tdata types   : {self.data.output.sorted_keys} \n"
            f"\tupdate period (s): {self.cfg.update_period}\n"
            f"\tshape        : {self.image_shape}\n"
            f"\tnumber of sensors : {self._view.count}"
        )

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        if not self._is_initialized:
            raise RuntimeError(
                "TiledCamera could not be initialized. Please ensure --enable_cameras is used to enable rendering."
            )
        # reset the timestamps
        SensorBase.reset(self, env_ids)
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # Reset the frame count
        self._frame[env_ids] = 0

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        This function creates handles and registers the provided data types with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the number of camera prims in the view does not match the number of environments.
            RuntimeError: If replicator was not found.
        """
        try:
            import omni.replicator.core as rep
        except ModuleNotFoundError:
            raise RuntimeError(
                "Replicator was not found for rendering. Please use --enable_cameras to enable rendering."
            )

        # Initialize parent class
        SensorBase._initialize_impl(self)
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

        # Obtain current stage
        stage = omni.usd.get_context().get_stage()
        # Convert all encapsulated prims to Camera
        for cam_prim_path in self._view.prim_paths:
            # Get camera prim
            cam_prim = stage.GetPrimAtPath(cam_prim_path)
            # Check if prim is a camera
            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")
            # Add to list
            sensor_prim = UsdGeom.Camera(cam_prim)
            self._sensor_prims.append(sensor_prim)

        rep.orchestrator._orchestrator._is_started = True
        sensor = rep.create.tiled_sensor(
            cameras=self._view.prim_paths,
            camera_resolution=[self.image_shape[1], self.image_shape[0]],
            tiled_resolution=self._tiled_image_shape(),
            output_types=self.cfg.data_types,
        )
        render_prod_path = rep.create.render_product(camera=sensor, resolution=self._tiled_image_shape())
        if not isinstance(render_prod_path, str):
            render_prod_path = render_prod_path.path
        self._render_product_paths = [render_prod_path]
        self._annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorGpu", device=self.device, do_array_copy=False)
        self._annotator.attach(self._render_product_paths)
        # Create internal buffers
        self._create_buffers()

    def _create_annotator_data(self):
        raise RuntimeError("Annotator data is not available for the tiled camera sensor.")

    def _process_annotator_output(self, name: str, output: Any) -> tuple[torch.tensor, dict | None]:
        raise RuntimeError("Annotator data is not available for the tiled camera sensor.")

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # Increment frame count
        self._frame[env_ids] += 1
        # Extract the flattened image buffer
        tiled_data_buffer = self._annotator.get_data()
        if isinstance(tiled_data_buffer, np.ndarray):
            tiled_data_buffer = wp.array(tiled_data_buffer, device=self.device)
        else:
            tiled_data_buffer = tiled_data_buffer.to(device=self.device)
        # The offset is needed when the buffer contains rgb and depth (the buffer has RGB data first and then depth)
        offset = self._data.output["rgb"].numel() if "rgb" in self.cfg.data_types else 0
        for data_type in self.cfg.data_types:
            wp.launch(
                kernel=reshape_tiled_image,
                dim=(self._view.count, self.cfg.height, self.cfg.width),
                inputs=[
                    tiled_data_buffer,
                    wp.from_torch(self._data.output[data_type]),  # zero-copy alias
                    *list(self._data.output[data_type].shape[1:]),  # height, width, num_channels
                    self._tiling_grid_shape()[0],  # num_tiles_x
                    offset if data_type == "depth" else 0,
                ],
                device=self.device,
            )

    """
    Private Helpers
    """

    def _tiled_image_shape(self) -> tuple[int, int]:
        """A tuple containing the dimension of the tiled image."""
        cols, rows = self._tiling_grid_shape()
        return (self.cfg.width * cols, self.cfg.height * rows)

    def _tiling_grid_shape(self) -> tuple[int, int]:
        """A tuple containing the tiling grid dimension."""
        cols = round(math.sqrt(self._view.count))
        rows = math.ceil(self._view.count / cols)
        return (cols, rows)

    def _check_supported_data_types(self, cfg: TiledCameraCfg):
        """Checks if the data types are supported by the camera."""
        if not set(cfg.data_types).issubset(TiledCamera.SUPPORTED_TYPES):
            raise ValueError(
                f"The TiledCamera class only supports the following types {TiledCamera.SUPPORTED_TYPES} but the"
                f" following where provided: {cfg.data_types}"
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        # create the data object
        # -- pose of the cameras
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        self._update_poses(self._ALL_INDICES)
        # -- intrinsic matrix
        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._update_intrinsic_matrices(self._ALL_INDICES)
        self._data.image_shape = self.image_shape
        # -- output data
        data_dict = dict()
        if "rgb" in self.cfg.data_types:
            data_dict["rgb"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 3), device=self.device
            ).contiguous()
        if "depth" in self.cfg.data_types:
            data_dict["depth"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device
            ).contiguous()
        self._data.output = TensorDict(data_dict, batch_size=self._view.count, device=self.device)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
