# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import warp as wp
from packaging import version

from pxr import Sdf, UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.sensors as sensor_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.sim.views import XformPrimView
from isaaclab.utils import to_camel_case
from isaaclab.utils.array import convert_to_torch
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    quat_from_matrix,
)
from isaaclab.utils.version import get_isaac_sim_version

from ..sensor_base import SensorBase
from .camera_data import CameraData

if TYPE_CHECKING:
    from .camera_cfg import CameraCfg

# import logger
logger = logging.getLogger(__name__)


class Camera(SensorBase):
    r"""The camera sensor for acquiring visual data.

    This class wraps over the `UsdGeom Camera`_ for providing a consistent API for acquiring visual data.
    It ensures that the camera follows the ROS convention for the coordinate system.

    Summarizing from the `replicator extension`_, the following sensor types are supported:

    - ``"rgb"``: A 3-channel rendered color image.
    - ``"rgba"``: A 4-channel rendered color image with alpha channel.
    - ``"albedo"``: A 4-channel fast diffuse-albedo only path for color image.
      Note that this path will achieve the best performance when used alone or with depth only.
    - ``"distance_to_camera"``: An image containing the distance to camera optical center.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
    - ``"depth"``: The same as ``"distance_to_image_plane"``.
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

    """

    cfg: CameraCfg
    """The configuration parameters."""

    UNSUPPORTED_TYPES: set[str] = {
        "instance_id_segmentation",
        "instance_segmentation",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "bounding_box_3d",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d_fast",
    }
    """The set of sensor types that are not supported by the camera class."""

    SIMPLE_SHADING_MODES: dict[str, int] = {
        "simple_shading_constant_diffuse": 0,
        "simple_shading_diffuse_mdl": 1,
        "simple_shading_full_mdl": 2,
    }
    SIMPLE_SHADING_AOV: str = "SimpleShadingSD"
    SIMPLE_SHADING_MODE_SETTING: str = "/rtx/sdg/simpleShading/mode"

    def __init__(self, cfg: CameraCfg):
        """Initializes the camera sensor.

        Args:
            cfg: The configuration parameters.

        Raises:
            RuntimeError: If no camera prim is found at the given path.
            ValueError: If the provided data types are not supported by the camera.
        """
        # perform check on supported data types
        self._check_supported_data_types(cfg)
        # initialize base class
        super().__init__(cfg)

        # toggle rendering of rtx sensors as True
        # this flag is read by SimulationContext to determine if rtx sensors should be rendered
        settings = get_settings_manager()
        settings.set_bool("/isaaclab/render/rtx_sensors", True)

        # This is only introduced in isaac sim 6.0
        isaac_sim_version = get_isaac_sim_version()
        if isaac_sim_version.major >= 6:
            # Set RTX flag to enable fast path if only depth or albedo is requested
            supported_fast_types = {"distance_to_camera", "distance_to_image_plane", "depth", "albedo"}
            if all(data_type in supported_fast_types for data_type in self.cfg.data_types):
                settings.set_bool("/rtx/sdg/force/disableColorRender", True)

            # If we have GUI / viewport enabled, we turn off fast path so that the viewport is not black
            if settings.get("/isaaclab/has_gui"):
                settings.set_bool("/rtx/sdg/force/disableColorRender", False)
        else:
            if "albedo" in self.cfg.data_types:
                logger.warning(
                    "Albedo annotator is only supported in Isaac Sim 6.0+. The albedo data type will be ignored."
                )
            if any(data_type in self.SIMPLE_SHADING_MODES for data_type in self.cfg.data_types):
                logger.warning(
                    "Simple shading annotators are only supported in Isaac Sim 6.0+. The simple shading data types"
                    " will be ignored."
                )

        # Set simple shading mode (if requested) before rendering
        simple_shading_mode = self._resolve_simple_shading_mode()
        if simple_shading_mode is not None:
            settings.set_int(self.SIMPLE_SHADING_MODE_SETTING, simple_shading_mode)

        # spawn the asset
        if self.cfg.spawn is not None:
            # Use spawn_path when set (points to template location for scene-cloned sensors).
            # This allows the camera to be spawned inside the asset template (e.g. inside
            # proto_asset_0) before clone_environments replicates it to all env paths.
            spawn_target = (
                self.cfg.spawn.spawn_path
                if getattr(self.cfg.spawn, "spawn_path", None) is not None
                else self.cfg.prim_path
            )
            # compute the rotation offset
            rot = torch.tensor(self.cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
            rot_offset = convert_camera_frame_orientation_convention(
                rot, origin=self.cfg.offset.convention, target="opengl"
            )
            rot_offset = rot_offset.squeeze(0).cpu().numpy()
            # ensure vertical aperture is set, otherwise replace with default for squared pixels
            if self.cfg.spawn.vertical_aperture is None:
                self.cfg.spawn.vertical_aperture = self.cfg.spawn.horizontal_aperture * self.cfg.height / self.cfg.width
            self.cfg.spawn.func(spawn_target, self.cfg.spawn, translation=self.cfg.offset.pos, orientation=rot_offset)
        # check that spawn was successful; use spawn_path if set (template location) since env
        # paths are not yet populated at init time — they are filled in by clone_environments.
        check_path = (
            self.cfg.spawn.spawn_path
            if self.cfg.spawn is not None and getattr(self.cfg.spawn, "spawn_path", None) is not None
            else self.cfg.prim_path
        )
        matching_prims = sim_utils.find_matching_prims(check_path)
        if len(matching_prims) == 0:
            raise RuntimeError(f"Could not find prim with path {check_path}.")

        # UsdGeom Camera prim for the sensor
        self._sensor_prims: list[UsdGeom.Camera] = list()
        # Create empty variables for storing output data
        self._data = CameraData()

        # HACK: We need to disable instancing for semantic_segmentation and instance_segmentation_fast to work
        # checks for Isaac Sim v4.5 as this issue exists there
        if get_isaac_sim_version() == version.parse("4.5"):
            if "semantic_segmentation" in self.cfg.data_types or "instance_segmentation_fast" in self.cfg.data_types:
                logger.warning(
                    "Isaac Sim 4.5 introduced a bug in Camera and TiledCamera when outputting instance and semantic"
                    " segmentation outputs for instanceable assets. As a workaround, the instanceable flag on assets"
                    " will be disabled in the current workflow and may lead to longer load times and increased memory"
                    " usage."
                )
                with Sdf.ChangeBlock():
                    for prim in self.stage.Traverse():
                        prim.SetInstanceable(False)

    def __del__(self):
        """Unsubscribes from callbacks and detach from the replicator registry."""
        # unsubscribe callbacks
        super().__del__()
        # delete from replicator registry
        for _, annotators in self._rep_registry.items():
            for annotator, render_product_path in zip(annotators, self._render_product_paths):
                annotator.detach([render_product_path])
                annotator = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Camera @ '{self.cfg.prim_path}': \n"
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
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> CameraData:
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

    @property
    def image_shape(self) -> tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.height, self.cfg.width)

    """
    Configuration
    """

    def set_intrinsic_matrices(
        self, matrices: torch.Tensor, focal_length: float | None = None, env_ids: Sequence[int] | None = None
    ):
        """Set parameters of the USD camera from its intrinsic matrix.

        The intrinsic matrix is used to set the following parameters to the USD camera:

        - ``focal_length``: The focal length of the camera.
        - ``horizontal_aperture``: The horizontal aperture of the camera.
        - ``vertical_aperture``: The vertical aperture of the camera.
        - ``horizontal_aperture_offset``: The horizontal offset of the camera.
        - ``vertical_aperture_offset``: The vertical offset of the camera.

        .. warning::

            Due to limitations of Omniverse camera, we need to assume that the camera is a spherical lens,
            i.e. has square pixels, and the optical center is centered at the camera eye. If this assumption
            is not true in the input intrinsic matrix, then the camera will not set up correctly.

        Args:
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3).
            focal_length: Perspective focal length (in cm) used to calculate pixel size. Defaults to None. If None,
                focal_length will be calculated 1 / width.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # convert matrices to numpy tensors
        if isinstance(matrices, torch.Tensor):
            matrices = matrices.cpu().numpy()
        else:
            matrices = np.asarray(matrices, dtype=float)
        # iterate over env_ids
        for i, intrinsic_matrix in zip(env_ids, matrices):
            height, width = self.image_shape

            params = sensor_utils.convert_camera_intrinsics_to_usd(
                intrinsic_matrix=intrinsic_matrix.reshape(-1), height=height, width=width, focal_length=focal_length
            )

            # change data for corresponding camera index
            sensor_prim = self._sensor_prims[i]
            # set parameters for camera
            for param_name, param_value in params.items():
                # convert to camel case (CC)
                param_name = to_camel_case(param_name, to="CC")
                # get attribute from the class
                param_attr = getattr(sensor_prim, f"Get{param_name}Attr")
                # convert numpy scalar to Python float for USD compatibility (NumPy 2.0+)
                if isinstance(param_value, np.floating):
                    param_value = float(param_value)
                # set value using pure USD API
                param_attr().Set(param_value)
        # update the internal buffers
        self._update_intrinsic_matrices(env_ids)

    """
    Operations - Set pose.
    """

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

        See :meth:`isaaclab.sensors.camera.utils.convert_camera_frame_orientation_convention` for more details
        on the conventions.

        Args:
            positions: The cartesian coordinates (in meters). Shape is (N, 3).
                Defaults to None, in which case the camera position in not changed.
            orientations: The quaternion orientation in (x, y, z, w). Shape is (N, 4).
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
            orientations = convert_camera_frame_orientation_convention(orientations, origin=convention, target="opengl")
        # set the pose
        self._view.set_world_poses(positions, orientations, env_ids)

    def set_world_poses_from_view(
        self, eyes: torch.Tensor, targets: torch.Tensor, env_ids: Sequence[int] | None = None
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
        # get up axis of current stage
        up_axis = UsdGeom.GetStageUpAxis(self.stage)
        # set camera poses using the view
        orientations = quat_from_matrix(create_rotation_matrix_from_view(eyes, targets, up_axis, device=self._device))
        self._view.set_world_poses(eyes, orientations, env_ids)

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        if not self._is_initialized:
            raise RuntimeError(
                "Camera could not be initialized. Please ensure --enable_cameras is used to enable rendering."
            )
        # reset the timestamps
        super().reset(env_ids, env_mask)
        # resolve to indices for torch indexing
        if env_ids is None and env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        elif env_ids is None:
            env_ids = self._ALL_INDICES
        # reset the data
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        self._update_poses(env_ids)
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
        if not get_settings_manager().get("/isaaclab/cameras_enabled"):
            raise RuntimeError(
                "A camera was spawned without the --enable_cameras flag. Please use --enable_cameras to enable"
                " rendering."
            )

        import omni.replicator.core as rep
        from omni.syntheticdata.scripts.SyntheticData import SyntheticData

        # Initialize parent class
        super()._initialize_impl()
        # Create a view for the sensor with Fabric enabled for fast pose queries, otherwise position will be stale.
        self._view = XformPrimView(
            self.cfg.prim_path, device=self._device, stage=self.stage, sync_usd_on_fabric_write=True
        )
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

        # Attach the sensor data types to render node
        self._render_product_paths: list[str] = list()
        self._rep_registry: dict[str, list[rep.annotators.Annotator]] = {name: list() for name in self.cfg.data_types}

        # Convert all encapsulated prims to Camera
        for cam_prim in self._view.prims:
            # Obtain the prim path
            cam_prim_path = cam_prim.GetPath().pathString
            # Check if prim is a camera
            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")
            # Add to list
            sensor_prim = UsdGeom.Camera(cam_prim)
            self._sensor_prims.append(sensor_prim)

            # Get render product
            # From Isaac Sim 2023.1 onwards, render product is a HydraTexture so we need to extract the path
            render_prod_path = rep.create.render_product(cam_prim_path, resolution=(self.cfg.width, self.cfg.height))
            if not isinstance(render_prod_path, str):
                render_prod_path = render_prod_path.path
            self._render_product_paths.append(render_prod_path)

            # Check if semantic types or semantic filter predicate is provided
            if isinstance(self.cfg.semantic_filter, list):
                semantic_filter_predicate = ":*; ".join(self.cfg.semantic_filter) + ":*"
            elif isinstance(self.cfg.semantic_filter, str):
                semantic_filter_predicate = self.cfg.semantic_filter
            else:
                raise ValueError(f"Semantic types must be a list or a string. Received: {self.cfg.semantic_filter}.")
            # set the semantic filter predicate
            # copied from rep.scripts.writes_default.basic_writer.py
            SyntheticData.Get().set_instance_mapping_semantic_filter(semantic_filter_predicate)

            # Iterate over each data type and create annotator
            # TODO: This will move out of the loop once Replicator supports multiple render products within a single
            #  annotator, i.e.: rep_annotator.attach(self._render_product_paths)
            for name in self.cfg.data_types:
                # note: we are verbose here to make it easier to understand the code.
                #   if colorize is true, the data is mapped to colors and a uint8 4 channel image is returned.
                #   if colorize is false, the data is returned as a uint32 image with ids as values.
                if name == "semantic_segmentation":
                    init_params = {
                        "colorize": self.cfg.colorize_semantic_segmentation,
                        "mapping": json.dumps(self.cfg.semantic_segmentation_mapping),
                    }
                elif name == "instance_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_segmentation}
                elif name == "instance_id_segmentation_fast":
                    init_params = {"colorize": self.cfg.colorize_instance_id_segmentation}
                else:
                    init_params = None

                # Resolve device name
                if "cuda" in self._device:
                    device_name = self._device.split(":")[0]
                else:
                    device_name = "cpu"

                # TODO: this is a temporary solution because replicator has not exposed the annotator yet
                # once it's exposed, we can remove this
                if name == "albedo":
                    rep.AnnotatorRegistry.register_annotator_from_aov(
                        aov="DiffuseAlbedoSD", output_data_type=np.uint8, output_channels=4
                    )
                if name in self.SIMPLE_SHADING_MODES:
                    rep.AnnotatorRegistry.register_annotator_from_aov(
                        aov=self.SIMPLE_SHADING_AOV, output_data_type=np.uint8, output_channels=4
                    )

                # Map special cases to their corresponding annotator names
                simple_shading_cases = {key: self.SIMPLE_SHADING_AOV for key in self.SIMPLE_SHADING_MODES}
                special_cases = {
                    "rgba": "rgb",
                    "depth": "distance_to_image_plane",
                    "albedo": "DiffuseAlbedoSD",
                    **simple_shading_cases,
                }
                # Get the annotator name, falling back to the original name if not a special case
                annotator_name = special_cases.get(name, name)
                # Create the annotator node
                rep_annotator = rep.AnnotatorRegistry.get_annotator(annotator_name, init_params, device=device_name)

                # attach annotator to render product
                rep_annotator.attach(render_prod_path)
                # add to registry
                self._rep_registry[name].append(rep_annotator)

        # Create internal buffers
        self._create_buffers()
        self._update_intrinsic_matrices(self._ALL_INDICES)

    def _update_buffers_impl(self, env_mask: wp.array):
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return
        # Increment frame count
        self._frame[env_ids] += 1
        # -- pose
        if self.cfg.update_latest_camera_pose:
            self._update_poses(env_ids)
        # Ensure the RTX renderer has been pumped so annotator buffers are fresh.
        # Lazy import Isaac RTX Renderer dependency.
        # For now the Camera implementation works only with Isaac RTX Renderer.
        # Future consideration should be to move Renderer from TiledCamera up the hierarchy to Camera
        # to make the Camera backend-agnostic.
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        ensure_isaac_rtx_render_update()

        # -- read the data from annotator registry
        # check if buffer is called for the first time. If so then, allocate the memory
        if len(self._data.output) == 0:
            # this is the first time buffer is called
            # it allocates memory for all the sensors
            self._create_annotator_data()
        else:
            # iterate over all the data types
            for name, annotators in self._rep_registry.items():
                # iterate over all the annotators
                for index in env_ids:
                    # get the output
                    output = annotators[index].get_data()
                    # process the output
                    data, info = self._process_annotator_output(name, output)
                    # add data to output
                    self._data.output[name][index] = data
                    # add info to output
                    self._data.info[index][name] = info
                # NOTE: The `distance_to_camera` annotator returns the distance to the camera optical center. However,
                #       the replicator depth clipping is applied w.r.t. to the image plane which may result in values
                #       larger than the clipping range in the output. We apply an additional clipping to ensure values
                #       are within the clipping range for all the annotators.
                if name == "distance_to_camera":
                    self._data.output[name][self._data.output[name] > self.cfg.spawn.clipping_range[1]] = torch.inf
                # apply defined clipping behavior
                if (
                    name == "distance_to_camera" or name == "distance_to_image_plane"
                ) and self.cfg.depth_clipping_behavior != "none":
                    self._data.output[name][torch.isinf(self._data.output[name])] = (
                        0.0 if self.cfg.depth_clipping_behavior == "zero" else self.cfg.spawn.clipping_range[1]
                    )

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: CameraCfg):
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
                f"Camera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types output numpy structured data types which"
                "can't be converted to torch tensors easily."
                "\n\tHint: If you need to work with these sensor types, we recommend using their fast counterparts."
                f"\n\t\tFast counterparts: {fast_common_elements}"
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        # create the data object
        # -- pose of the cameras
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        # -- intrinsic matrix
        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._data.image_shape = self.image_shape
        # -- output data
        # lazy allocation of data dictionary
        # since the size of the output data is not known in advance, we leave it as None
        # the memory will be allocated when the buffer() function is called for the first time.
        self._data.output = {}
        self._data.info = [{name: None for name in self.cfg.data_types} for _ in range(self._view.count)]

    def _update_intrinsic_matrices(self, env_ids: Sequence[int]):
        """Compute camera's matrix of intrinsic parameters.

        Also called calibration matrix. This matrix works for linear depth images. We assume square pixels.

        .. note::
            The calibration matrix projects points in the 3D scene onto an imaginary screen of the camera.
            The coordinates of points on the image plane are in the homogeneous representation.
        """
        # iterate over all cameras
        for i in env_ids:
            # Get corresponding sensor prim
            sensor_prim = self._sensor_prims[i]
            # get camera parameters
            # currently rendering does not use aperture offsets or vertical aperture
            focal_length = sensor_prim.GetFocalLengthAttr().Get()
            horiz_aperture = sensor_prim.GetHorizontalApertureAttr().Get()

            # get viewport parameters
            height, width = self.image_shape
            # extract intrinsic parameters
            f_x = (width * focal_length) / horiz_aperture
            f_y = f_x
            c_x = width * 0.5
            c_y = height * 0.5
            # create intrinsic matrix for depth linear
            self._data.intrinsic_matrices[i, 0, 0] = f_x
            self._data.intrinsic_matrices[i, 0, 2] = c_x
            self._data.intrinsic_matrices[i, 1, 1] = f_y
            self._data.intrinsic_matrices[i, 1, 2] = c_y
            self._data.intrinsic_matrices[i, 2, 2] = 1

    def _update_poses(self, env_ids: Sequence[int]):
        """Computes the pose of the camera in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Returns:
            A tuple of the position (in meters) and quaternion (x, y, z, w).
        """
        # check camera prim exists
        if len(self._sensor_prims) == 0:
            raise RuntimeError("Camera prim is None. Please call 'sim.play()' first.")

        # get the poses from the view
        poses, quat = self._view.get_world_poses(env_ids)
        self._data.pos_w[env_ids] = poses
        self._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(
            quat, origin="opengl", target="world"
        )

    def _create_annotator_data(self):
        """Create the buffers to store the annotator data.

        We create a buffer for each annotator and store the data in a dictionary. Since the data
        shape is not known beforehand, we create a list of buffers and concatenate them later.

        This is an expensive operation and should be called only once.
        """
        # add data from the annotators
        for name, annotators in self._rep_registry.items():
            # create a list to store the data for each annotator
            data_all_cameras = list()
            # iterate over all the annotators
            for index in self._ALL_INDICES:
                # get the output
                output = annotators[index].get_data()
                # process the output
                data, info = self._process_annotator_output(name, output)
                # append the data
                data_all_cameras.append(data)
                # store the info
                self._data.info[index][name] = info
            # concatenate the data along the batch dimension
            self._data.output[name] = torch.stack(data_all_cameras, dim=0)
            # NOTE: `distance_to_camera` and `distance_to_image_plane` are not both clipped to the maximum defined
            #       in the clipping range. The clipping is applied only to `distance_to_image_plane` and then both
            #       outputs are only clipped where the values in `distance_to_image_plane` exceed the threshold. To
            #       have a unified behavior between all cameras, we clip both outputs to the maximum value defined.
            if name == "distance_to_camera":
                self._data.output[name][self._data.output[name] > self.cfg.spawn.clipping_range[1]] = torch.inf
            # clip the data if needed
            if (
                name == "distance_to_camera" or name == "distance_to_image_plane"
            ) and self.cfg.depth_clipping_behavior != "none":
                self._data.output[name][torch.isinf(self._data.output[name])] = (
                    0.0 if self.cfg.depth_clipping_behavior == "zero" else self.cfg.spawn.clipping_range[1]
                )

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
        height, width = self.image_shape
        if name == "semantic_segmentation":
            if self.cfg.colorize_semantic_segmentation:
                data = data.view(torch.uint8).reshape(height, width, -1)
            else:
                data = data.view(height, width, 1)
        elif name == "instance_segmentation_fast":
            if self.cfg.colorize_instance_segmentation:
                data = data.view(torch.uint8).reshape(height, width, -1)
            else:
                data = data.view(height, width, 1)
        elif name == "instance_id_segmentation_fast":
            if self.cfg.colorize_instance_id_segmentation:
                data = data.view(torch.uint8).reshape(height, width, -1)
            else:
                data = data.view(height, width, 1)
        # make sure buffer dimensions are consistent as (H, W, C)
        elif name == "distance_to_camera" or name == "distance_to_image_plane" or name == "depth":
            data = data.view(height, width, 1)
        # we only return the RGB channels from the RGBA output if rgb is required
        # normals return (x, y, z) in first 3 channels, 4th channel is unused
        elif name == "rgb" or name == "normals":
            data = data[..., :3]
        # motion vectors return (x, y) in first 2 channels, 3rd and 4th channels are unused
        elif name == "motion_vectors":
            data = data[..., :2]
        elif name in self.SIMPLE_SHADING_MODES:
            data = data[..., :3]

        # return the data and info
        return data, info

    def _resolve_simple_shading_mode(self) -> int | None:
        """Resolve the requested simple shading mode from data types."""
        requested = [data_type for data_type in self.cfg.data_types if data_type in self.SIMPLE_SHADING_MODES]
        if not requested:
            return None
        if len(requested) > 1:
            logger.warning(
                "Multiple simple shading modes requested (%s). Using '%s' only.",
                requested,
                requested[0],
            )
        return self.SIMPLE_SHADING_MODES[requested[0]]

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
