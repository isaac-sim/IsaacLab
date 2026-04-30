# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import warp as wp

from pxr import UsdGeom

import isaaclab.utils.sensors as sensor_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.renderers import BaseRenderer, Renderer
from isaaclab.sim.views import FrameView
from isaaclab.utils import to_camel_case
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    quat_from_matrix,
)

from ..sensor_base import SensorBase
from .camera_data import CameraData, RenderBufferKind

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
        self._register_renderer_scene_data_requirements()

        # TODO(follow-up PR): move this flag flip out of Camera. The cleanest path is
        # an apply_pre_reset_settings() hook on RendererCfg (default no-op) that
        # IsaacRtxRendererCfg overrides to flip /isaaclab/render/rtx_sensors. The
        # flag must be set pre-sim.reset() because SimulationContext.is_rendering
        # and several env classes read it before the renderer's __init__ runs.
        if self.cfg.renderer_cfg.renderer_type == "isaac_rtx":
            get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", True)

        # Compute camera orientation (convention conversion) and spawn
        rot = torch.tensor(self.cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
        rot_offset = convert_camera_frame_orientation_convention(
            rot, origin=self.cfg.offset.convention, target="opengl"
        )
        rot_offset = rot_offset.squeeze(0).cpu().numpy()
        if self.cfg.spawn is not None and self.cfg.spawn.vertical_aperture is None:
            self.cfg.spawn.vertical_aperture = self.cfg.spawn.horizontal_aperture * self.cfg.height / self.cfg.width
        self._resolve_and_spawn("camera", translation=self.cfg.offset.pos, orientation=rot_offset)

        # UsdGeom Camera prim for the sensor
        self._sensor_prims: list[UsdGeom.Camera] = list()
        # Allocated in :meth:`_create_buffers` once the renderer's output contract is known.
        self._data: CameraData | None = None
        # Renderer and render data — assigned in _initialize_impl.
        self._renderer: BaseRenderer | None = None
        self._render_data = None

    def _register_renderer_scene_data_requirements(self) -> None:
        """Register renderer requirements early enough for clone-time prebuilds."""
        renderer_type = getattr(getattr(self.cfg, "renderer_cfg", None), "renderer_type", None)
        if renderer_type is None:
            return

        from isaaclab.physics.scene_data_requirements import aggregate_requirements, requirement_for_renderer_type
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        if sim is None:
            logger.debug("SimulationContext not available; deferring renderer requirements registration.")
            return

        try:
            renderer_req = requirement_for_renderer_type(renderer_type)
        except ValueError:
            return

        current_req = sim.get_scene_data_requirements()
        merged_req = aggregate_requirements((current_req, renderer_req))
        if merged_req != current_req:
            sim.update_scene_data_requirements(merged_req)

    def __del__(self):
        """Unsubscribes from callbacks and cleans up renderer resources."""
        # unsubscribe callbacks
        super().__del__()
        # cleanup render resources (renderer may be None if never initialized)
        if self._renderer is not None:
            self._renderer.cleanup(self._render_data)

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Camera @ '{self.cfg.prim_path}': \n"
            f"\tdata types   : {list(self.data.output.keys())} \n"
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
        # convert torch tensors to warp arrays for the view
        pos_wp = wp.from_torch(positions.contiguous()) if positions is not None else None
        ori_wp = wp.from_torch(orientations.contiguous()) if orientations is not None else None
        if env_ids is not None:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, dtype=torch.int32, device=self._device)
            idx_wp = wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32)
        else:
            idx_wp = None
        self._view.set_world_poses(pos_wp, ori_wp, idx_wp)

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
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.int32, device=self._device)
        idx_wp = wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32)
        self._view.set_world_poses(wp.from_torch(eyes.contiguous()), wp.from_torch(orientations.contiguous()), idx_wp)

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

        This function creates a :class:`~isaaclab.renderers.Renderer` from the configured
        :attr:`~isaaclab.sensors.camera.CameraCfg.renderer_cfg` and delegates all render-product
        and annotator management to it. It also initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the number of camera prims in the view does not match the number of environments.
            RuntimeError: Propagated from the renderer constructor when the active backend's
                runtime requirements are not satisfied (e.g. the RTX backend requires the
                simulation app to be launched with ``--enable_cameras``).
        """
        # Initialize parent class
        super()._initialize_impl()

        self._renderer = Renderer(self.cfg.renderer_cfg)
        logger.info("Using renderer: %s", type(self._renderer).__name__)

        # Stage preprocessing must happen before creating the view because the view keeps
        # references to prims located in the stage.
        self._renderer.prepare_stage(self.stage, self._num_envs)

        # Create a view for the sensor with Fabric enabled for fast pose queries.
        # TODO: remove sync_usd_on_fabric_write=True once the GPU Fabric sync bug is fixed.
        self._view = FrameView(self.cfg.prim_path, device=self._device, stage=self.stage, sync_usd_on_fabric_write=True)
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
        for cam_prim in self._view.prims:
            # Obtain the prim path
            cam_prim_path = cam_prim.GetPath().pathString
            # Check if prim is a camera
            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")
            # Add to list
            self._sensor_prims.append(UsdGeom.Camera(cam_prim))

        # View needs to exist before creating render data
        self._render_data = self._renderer.create_render_data(self)

        # Create internal buffers (includes intrinsic matrix and pose init)
        self._create_buffers()

    def _update_buffers_impl(self, env_mask: wp.array):
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return
        # Increment frame count
        self._frame[env_ids] += 1
        # update latest camera pose if requested
        if self.cfg.update_latest_camera_pose:
            self._update_poses(env_ids)

        self._renderer.update_transforms()
        self._renderer.render(self._render_data)

        self._renderer.read_output(self._render_data, self._data)

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
        specs = self._renderer.supported_output_types()
        # Split requested names into known/unsupported; warn once for any the renderer can't produce.
        known: list[str] = []
        unsupported: list[str] = []
        for name in self.cfg.data_types:
            try:
                if RenderBufferKind(name) in specs:
                    known.append(name)
                else:
                    unsupported.append(name)
            except ValueError:
                unsupported.append(name)
        if unsupported:
            logger.warning(
                "Renderer %s does not support the following requested data types and will not produce them: %s",
                type(self._renderer).__name__,
                unsupported,
            )
        self._data = CameraData.allocate(
            data_types=known,
            height=self.cfg.height,
            width=self.cfg.width,
            num_views=self._view.count,
            device=self._device,
            supported_specs=specs,
        )
        # Camera-frame state (pose / intrinsics) is owned by the camera, not
        # the renderer: populate it on the freshly constructed ``CameraData``.
        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._update_intrinsic_matrices(self._ALL_INDICES)
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        self._update_poses(self._ALL_INDICES)
        self._renderer.set_outputs(self._render_data, self._data.output)

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

        # get the poses from the view (returns ProxyArray, use .torch for tensor access)
        if env_ids is not None and not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.int32, device=self._device)
        indices = wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32) if env_ids is not None else None
        pos_w, quat_w = self._view.get_world_poses(indices)
        self._data.pos_w[env_ids] = pos_w.torch
        self._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(
            quat_w.torch, origin="opengl", target="world"
        )
        # notify renderer of updated poses (guarded in case called before initialization completes)
        if self._render_data is not None:
            self._renderer.update_camera(
                self._render_data, self._data.pos_w, self._data.quat_w_world, self._data.intrinsic_matrices
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
