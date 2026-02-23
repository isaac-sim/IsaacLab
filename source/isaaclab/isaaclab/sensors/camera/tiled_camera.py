# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp

from pxr import UsdGeom

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.sim.views import XformPrimView
from isaaclab.utils.warp.kernels import reshape_tiled_image

from ..sensor_base import SensorBase
from .camera import Camera

if TYPE_CHECKING:
    from isaaclab.renderers import Renderer

    from .tiled_camera_cfg import TiledCameraCfg


class TiledCamera(Camera):
    SIMPLE_SHADING_AOV: str = "SimpleShadingSD"
    r"""The tiled rendering based camera sensor for acquiring the same data as the Camera class.

    This class inherits from the :class:`Camera` class but uses the tiled-rendering API to acquire
    the visual data. Tiled-rendering concatenates the rendered images from multiple cameras into a single image.
    This allows for rendering multiple cameras in parallel and is useful for rendering large scenes with multiple
    cameras efficiently.

    The following sensor types are supported:

    - ``"rgb"``: A 3-channel rendered color image.
    - ``"rgba"``: A 4-channel rendered color image with alpha channel.
    - ``"albedo"``: A 4-channel fast diffuse-albedo only path for color image.
      Note that this path will achieve the best performance when used alone or with depth only.
    - ``"distance_to_camera"``: An image containing the distance to camera optical center.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
    - ``"depth"``: Alias for ``"distance_to_image_plane"``.
    - ``"simple_shading_constant_diffuse"``: Simple shading (constant diffuse) RGB approximation.
    - ``"simple_shading_diffuse_mdl"``: Simple shading (diffuse MDL) RGB approximation.
    - ``"simple_shading_full_mdl"``: Simple shading (full MDL) RGB approximation.
    - ``"normals"``: An image containing the local surface normal vectors at each pixel.
    - ``"motion_vectors"``: An image containing the motion vector data at each pixel.
    - ``"semantic_segmentation"``: The semantic segmentation data.
    - ``"instance_segmentation_fast"``: The instance segmentation data.
    - ``"instance_id_segmentation_fast"``: The instance id segmentation data.

    .. note::
        Currently the following sensor types are not supported in a "view" format:

        - ``"instance_segmentation"``: The instance segmentation data. Please use the fast counterparts instead.
        - ``"instance_id_segmentation"``: The instance id segmentation data. Please use the fast counterparts instead.
        - ``"bounding_box_2d_tight"``: The tight 2D bounding box data (only contains non-occluded regions).
        - ``"bounding_box_2d_tight_fast"``: The tight 2D bounding box data (only contains non-occluded regions).
        - ``"bounding_box_2d_loose"``: The loose 2D bounding box data (contains occluded regions).
        - ``"bounding_box_2d_loose_fast"``: The loose 2D bounding box data (contains occluded regions).
        - ``"bounding_box_3d"``: The 3D view space bounding box data.
        - ``"bounding_box_3d_fast"``: The 3D view space bounding box data.

    .. _replicator extension: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotator-output
    .. _USDGeom Camera: https://graphics.pixar.com/usd/docs/api/class_usd_geom_camera.html

    .. versionadded:: v1.0.0

        This feature is available starting from Isaac Sim 4.2. Before this version, the tiled rendering APIs
        were not available.

    """

    cfg: TiledCameraCfg
    """The configuration parameters."""

    def __init__(self, cfg: TiledCameraCfg, renderer: Renderer | None = None):
        """Initializes the tiled camera sensor.

        Args:
            cfg: The configuration parameters.

        Raises:
            RuntimeError: If no camera prim is found at the given path.
            ValueError: If the provided data types are not supported by the camera.
        """
        self.renderer = renderer
        self.render_data = None
        super().__init__(cfg)

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""
        # unsubscribe from callbacks
        SensorBase.__del__(self)
        # detach from the replicator registry
        for annotator in self._annotators.values():
            annotator.detach(self.render_product_paths)

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Tiled Camera @ '{self.cfg.prim_path}': \n"
            f"\tdata types   : {list(self.data.output.keys())} \n"
            f"\tsemantic filter : {self.cfg.semantic_filter}\n"
            f"\tcolorize semantic segm.   : {self.cfg.colorize_semantic_segmentation}\n"
            f"\tcolorize instance segm.   : {self.cfg.colorize_instance_segmentation}\n"
            f"\tcolorize instance id segm.: {self.cfg.colorize_instance_id_segmentation}\n"
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
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset the frame count
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
        if not get_settings_manager().get("/isaaclab/cameras_enabled"):
            raise RuntimeError(
                "A camera was spawned without the --enable_cameras flag. Please use --enable_cameras to enable"
                " rendering."
            )

        import omni.replicator.core as rep

        # Initialize parent class
        SensorBase._initialize_impl(self)
        # Create a view for the sensor
        self._view = XformPrimView(self.cfg.prim_path, device=self._device, stage=self.stage)
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

        # Convert all encapsulated prims to Camera
        cam_prim_paths = []
        for cam_prim in self._view.prims:
            # Get camera prim
            cam_prim_path = cam_prim.GetPath().pathString
            # Check if prim is a camera
            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")
            # Add to list
            self._sensor_prims.append(UsdGeom.Camera(cam_prim))
            cam_prim_paths.append(cam_prim_path)

        if self.renderer is not None:
            self.render_data = self.renderer.create_render_data(self)

        else:
            # Create replicator tiled render product
            rp = rep.create.render_product_tiled(
                cameras=cam_prim_paths, tile_resolution=(self.cfg.width, self.cfg.height)
            )
            self._render_product_paths = [rp.path]

            if any(data_type in self.SIMPLE_SHADING_MODES for data_type in self.cfg.data_types):
                rep.AnnotatorRegistry.register_annotator_from_aov(
                    aov=self.SIMPLE_SHADING_AOV, output_data_type=np.uint8, output_channels=4
                )
                # Set simple shading mode (if requested) before rendering
                simple_shading_mode = self._resolve_simple_shading_mode()
                if simple_shading_mode is not None:
                    get_settings_manager().set_int(self.SIMPLE_SHADING_MODE_SETTING, simple_shading_mode)
            # Define the annotators based on requested data types
            self._annotators = dict()
            for annotator_type in self.cfg.data_types:
                if annotator_type == "rgba" or annotator_type == "rgb":
                    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device=self.device, do_array_copy=False)
                    self._annotators["rgba"] = annotator
                elif annotator_type == "albedo":
                    # TODO: this is a temporary solution because replicator has not exposed the annotator yet
                    # once it's exposed, we can remove this
                    rep.AnnotatorRegistry.register_annotator_from_aov(
                        aov="DiffuseAlbedoSD", output_data_type=np.uint8, output_channels=4
                    )
                    annotator = rep.AnnotatorRegistry.get_annotator(
                        "DiffuseAlbedoSD", device=self.device, do_array_copy=False
                    )
                    self._annotators["albedo"] = annotator
                elif annotator_type in self.SIMPLE_SHADING_MODES:
                    annotator = rep.AnnotatorRegistry.get_annotator(
                        self.SIMPLE_SHADING_AOV, device=self.device, do_array_copy=False
                    )
                    self._annotators[annotator_type] = annotator
                elif annotator_type == "depth" or annotator_type == "distance_to_image_plane":
                    # keep depth for backwards compatibility
                    annotator = rep.AnnotatorRegistry.get_annotator(
                        "distance_to_image_plane", device=self.device, do_array_copy=False
                    )
                    self._annotators[annotator_type] = annotator
                # note: we are verbose here to make it easier to understand the code.
                #   if colorize is true, the data is mapped to colors and a uint8 4 channel image is returned.
                #   if colorize is false, the data is returned as a uint32 image with ids as values.
                else:
                    init_params = None
                    if annotator_type == "semantic_segmentation":
                        init_params = {
                            "colorize": self.cfg.colorize_semantic_segmentation,
                            "mapping": json.dumps(self.cfg.semantic_segmentation_mapping),
                        }
                    elif annotator_type == "instance_segmentation_fast":
                        init_params = {"colorize": self.cfg.colorize_instance_segmentation}
                    elif annotator_type == "instance_id_segmentation_fast":
                        init_params = {"colorize": self.cfg.colorize_instance_id_segmentation}

                    annotator = rep.AnnotatorRegistry.get_annotator(
                        annotator_type, init_params, device=self.device, do_array_copy=False
                    )
                    self._annotators[annotator_type] = annotator

            # Attach the annotator to the render product
            for annotator in self._annotators.values():
                annotator.attach(self._render_product_paths)

        # Create internal buffers
        self._create_buffers()

    def _update_poses(self, env_ids: Sequence[int]):
        super()._update_poses(env_ids)
        if self.renderer is not None:
            self.renderer.update_camera(
                self.render_data, self._data.pos_w, self._data.quat_w_world, self._data.intrinsic_matrices
            )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # Increment frame count
        self._frame[env_ids] += 1

        # update latest camera pose
        if self.cfg.update_latest_camera_pose:
            self._update_poses(env_ids)

        if self.renderer is not None:
            self.renderer.render(self.render_data)

            for output_name, output_data in self._data.output.items():
                if output_name == "rgb":
                    continue
                self.renderer.write_output(self.render_data, output_name, output_data)
            return

        # Extract the flattened image buffer
        for data_type, annotator in self._annotators.items():
            # check whether returned data is a dict (used for segmentation)
            output = annotator.get_data()
            if isinstance(output, dict):
                tiled_data_buffer = output["data"]
                self._data.info[data_type] = output["info"]
            else:
                tiled_data_buffer = output

            # convert data buffer to warp array
            if isinstance(tiled_data_buffer, np.ndarray):
                # Let warp infer the dtype from numpy array instead of hardcoding uint8
                # Different annotators return different dtypes: RGB(uint8), depth(float32), segmentation(uint32)
                tiled_data_buffer = wp.array(tiled_data_buffer, device=self.device)
            else:
                tiled_data_buffer = tiled_data_buffer.to(device=self.device)

            # process data for different segmentation types
            # Note: Replicator returns raw buffers of dtype uint32 for segmentation types
            #   so we need to convert them to uint8 4 channel images for colorized types
            if (
                (data_type == "semantic_segmentation" and self.cfg.colorize_semantic_segmentation)
                or (data_type == "instance_segmentation_fast" and self.cfg.colorize_instance_segmentation)
                or (data_type == "instance_id_segmentation_fast" and self.cfg.colorize_instance_id_segmentation)
            ):
                tiled_data_buffer = wp.array(
                    ptr=tiled_data_buffer.ptr, shape=(*tiled_data_buffer.shape, 4), dtype=wp.uint8, device=self.device
                )

            # For motion vectors, use specialized kernel that reads 4 channels but only writes 2
            # Note: Not doing this breaks the alignment of the data (check: https://github.com/isaac-sim/IsaacLab/issues/2003)
            if data_type == "motion_vectors":
                tiled_data_buffer = tiled_data_buffer[:, :, :2].contiguous()

            # For normals, we only require the first three channels of the tiled buffer
            # Note: Not doing this breaks the alignment of the data (check: https://github.com/isaac-sim/IsaacLab/issues/4239)
            if data_type == "normals":
                tiled_data_buffer = tiled_data_buffer[:, :, :3].contiguous()
            if data_type in self.SIMPLE_SHADING_MODES:
                tiled_data_buffer = tiled_data_buffer[:, :, :3].contiguous()

            wp.launch(
                kernel=reshape_tiled_image,
                dim=(self._view.count, self.cfg.height, self.cfg.width),
                inputs=[
                    tiled_data_buffer.flatten(),
                    wp.from_torch(self._data.output[data_type]),  # zero-copy alias
                    *list(self._data.output[data_type].shape[1:]),  # height, width, num_channels
                    self._tiling_grid_shape()[0],  # num_tiles_x
                ],
                device=self.device,
            )

            # alias rgb as first 3 channels of rgba
            if data_type == "rgba" and "rgb" in self.cfg.data_types:
                self._data.output["rgb"] = self._data.output["rgba"][..., :3]

            # NOTE: The `distance_to_camera` annotator returns the distance to the camera optical center. However,
            #       the replicator depth clipping is applied w.r.t. to the image plane which may result in values
            #       larger than the clipping range in the output. We apply an additional clipping to ensure values
            #       are within the clipping range for all the annotators.
            if data_type == "distance_to_camera":
                self._data.output[data_type][self._data.output[data_type] > self.cfg.spawn.clipping_range[1]] = (
                    torch.inf
                )
            # apply defined clipping behavior
            if (
                data_type == "distance_to_camera" or data_type == "distance_to_image_plane" or data_type == "depth"
            ) and self.cfg.depth_clipping_behavior != "none":
                self._data.output[data_type][torch.isinf(self._data.output[data_type])] = (
                    0.0 if self.cfg.depth_clipping_behavior == "zero" else self.cfg.spawn.clipping_range[1]
                )

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: TiledCameraCfg):
        """Checks if the data types are supported by the ray-caster camera."""
        # check if there is any intersection in unsupported types
        # reason: these use np structured data types which we can't yet convert to torch tensor
        common_elements = set(cfg.data_types) & Camera.UNSUPPORTED_TYPES
        if common_elements:
            # provide alternative fast counterparts
            fast_common_elements = []
            for item in common_elements:
                if "instance_segmentation" in item or "instance_id_segmentation" in item:
                    fast_common_elements.append(item + "_fast")
            # raise error
            raise ValueError(
                f"TiledCamera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types output numpy structured data types which"
                "can't be converted to torch tensors easily."
                "\n\tHint: If you need to work with these sensor types, we recommend using their fast counterparts."
                f"\n\t\tFast counterparts: {fast_common_elements}"
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        # create the data object
        # -- intrinsic matrix
        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._update_intrinsic_matrices(self._ALL_INDICES)
        # -- pose of the cameras
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        self._update_poses(self._ALL_INDICES)
        self._data.image_shape = self.image_shape
        # -- output data
        data_dict = dict()
        if "rgba" in self.cfg.data_types or "rgb" in self.cfg.data_types:
            data_dict["rgba"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
            ).contiguous()
        if "rgb" in self.cfg.data_types:
            # RGB is the first 3 channels of RGBA
            data_dict["rgb"] = data_dict["rgba"][..., :3]
        if "albedo" in self.cfg.data_types:
            data_dict["albedo"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
            ).contiguous()
        for data_type in self.SIMPLE_SHADING_MODES:
            if data_type in self.cfg.data_types:
                data_dict[data_type] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 3), device=self.device, dtype=torch.uint8
                ).contiguous()
        if "distance_to_image_plane" in self.cfg.data_types:
            data_dict["distance_to_image_plane"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.float32
            ).contiguous()
        if "depth" in self.cfg.data_types:
            data_dict["depth"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.float32
            ).contiguous()
        if "distance_to_camera" in self.cfg.data_types:
            data_dict["distance_to_camera"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.float32
            ).contiguous()
        if "normals" in self.cfg.data_types:
            data_dict["normals"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 3), device=self.device, dtype=torch.float32
            ).contiguous()
        if "motion_vectors" in self.cfg.data_types:
            data_dict["motion_vectors"] = torch.zeros(
                (self._view.count, self.cfg.height, self.cfg.width, 2), device=self.device, dtype=torch.float32
            ).contiguous()
        if "semantic_segmentation" in self.cfg.data_types:
            if self.cfg.colorize_semantic_segmentation:
                data_dict["semantic_segmentation"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
                ).contiguous()
            else:
                data_dict["semantic_segmentation"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.int32
                ).contiguous()
        if "instance_segmentation_fast" in self.cfg.data_types:
            if self.cfg.colorize_instance_segmentation:
                data_dict["instance_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
                ).contiguous()
            else:
                data_dict["instance_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.int32
                ).contiguous()
        if "instance_id_segmentation_fast" in self.cfg.data_types:
            if self.cfg.colorize_instance_id_segmentation:
                data_dict["instance_id_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 4), device=self.device, dtype=torch.uint8
                ).contiguous()
            else:
                data_dict["instance_id_segmentation_fast"] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 1), device=self.device, dtype=torch.int32
                ).contiguous()

        self._data.output = data_dict
        self._data.info = dict()
        if self.renderer is not None:
            self.renderer.set_outputs(self.render_data, self._data.output)

    def _tiled_image_shape(self) -> tuple[int, int]:
        """Returns a tuple containing the dimension of the tiled image."""
        cols, rows = self._tiling_grid_shape()
        return (self.cfg.width * cols, self.cfg.height * rows)

    def _tiling_grid_shape(self) -> tuple[int, int]:
        """Returns a tuple containing the tiling grid dimension."""
        cols = math.ceil(math.sqrt(self._view.count))
        rows = math.ceil(self._view.count / cols)
        return (cols, rows)

    def _create_annotator_data(self):
        # we do not need to create annotator data for the tiled camera sensor
        raise RuntimeError("This function should not be called for the tiled camera sensor.")

    def _process_annotator_output(self, name: str, output: Any) -> tuple[torch.tensor, dict | None]:
        # we do not need to process annotator output for the tiled camera sensor
        raise RuntimeError("This function should not be called for the tiled camera sensor.")

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
