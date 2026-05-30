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

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.sensors as sensor_utils
from isaaclab.renderers import BaseRenderer, CameraRenderSpec
from isaaclab.sim.views import FrameView
from isaaclab.utils import to_camel_case
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    quat_from_matrix,
)
from isaaclab.utils.warp import ProxyArray

from ..sensor_base import SensorBase
from .camera_data import CameraData, RenderBufferKind

if TYPE_CHECKING:
    from .camera_cfg import CameraCfg

# import logger
logger = logging.getLogger(__name__)


@wp.kernel
def _camera_update_state_kernel(
    pos_src: wp.array(dtype=wp.vec3f),
    quat_src: wp.array(dtype=wp.quatf),
    intrinsics_src: wp.array(dtype=wp.mat33f),
    pos_dst: wp.array(dtype=wp.vec3f),
    quat_world_dst: wp.array(dtype=wp.quatf),
    intrinsics_dst: wp.array(dtype=wp.mat33f),
    frame: wp.array(dtype=wp.int64),
    env_mask: wp.array(dtype=wp.bool),
    env_ids: wp.array(dtype=wp.int32),
    use_env_ids: bool,
    use_env_mask: bool,
    update_pose: bool,
    update_intrinsics: bool,
    frame_op: int,
):
    """Update camera state for all, indexed, or masked cameras.

    ``frame_op`` uses 0 for no-op, 1 for increment, and 2 for reset.
    """
    src_id = wp.tid()
    dst_id = src_id
    if use_env_ids:
        dst_id = env_ids[src_id]
    if use_env_mask and not env_mask[dst_id]:
        return

    if update_pose:
        pos_dst[dst_id] = pos_src[src_id]
        quat_world_dst[dst_id] = quat_src[src_id] * wp.quatf(-0.5, 0.5, 0.5, 0.5)
    if update_intrinsics:
        intrinsics_dst[dst_id] = intrinsics_src[src_id]
    if frame_op == 1:
        frame[dst_id] = frame[dst_id] + wp.int64(1)
    elif frame_op == 2:
        frame[dst_id] = wp.int64(0)


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

        # Compute camera orientation (convention conversion) and spawn.
        rot = torch.tensor(self.cfg.offset.rot, dtype=torch.float32, device="cpu").unsqueeze(0)
        rot_offset = convert_camera_frame_orientation_convention(
            rot, origin=self.cfg.offset.convention, target="opengl"
        )
        rot_offset = rot_offset.squeeze(0).cpu().numpy()
        if self.cfg.spawn is not None and self.cfg.spawn.vertical_aperture is None:
            self.cfg.spawn.vertical_aperture = self.cfg.spawn.horizontal_aperture * self.cfg.height / self.cfg.width
        # Resolve the camera prim path and spawn it, redirecting to a child if prim_path is a physics body.
        spawn = self.cfg.spawn
        if spawn is not None:
            probe_path = (spawn.spawn_path or self.cfg.prim_path) if spawn is not None else self.cfg.prim_path
            probe_matches = sim_utils.resolve_matching_prims_from_source(probe_path)
            source_prim, _source_destination_expr = probe_matches[0] if probe_matches else (None, None)
            if source_prim is not None and source_prim.IsValid():
                if source_prim.HasAPI(UsdPhysics.ArticulationRootAPI) or source_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    logger.info(f" Spawning camera at '{self.cfg.prim_path}/camera'.")
                    self.cfg.prim_path = spawn.spawn_path = f"{self.cfg.prim_path}/camera"

            spawn_target = spawn.spawn_path or self.cfg.prim_path
            if sim_utils.find_first_matching_prim(spawn_target) is None:
                spawn.func(spawn_target, spawn, translation=self.cfg.offset.pos, orientation=rot_offset)
            if not sim_utils.find_matching_prims(spawn_target):
                raise RuntimeError(f"Could not find prim with path {spawn_target!r}.")

        # An ISP (any ``isp_cfg`` other than ``None``) requires the HDR AOV;
        # an explicit ``"rgb_hdr"`` in ``data_types`` also requires the
        # HDR-routing flag flipped on the RTX-bearing backends.
        require_hdr_output = "rgb_hdr" in self.cfg.data_types or self.cfg.isp_cfg is not None

        # TODO(follow-up PR): move this flag flip out of Camera. The cleanest path is
        # an apply_pre_reset_settings() hook on RendererCfg (default no-op) that
        # IsaacRtxRendererCfg overrides to flip /isaaclab/render/rtx_sensors. The
        # flag must be set pre-sim.reset() because SimulationContext.is_rendering
        # and several env classes read it before the renderer's __init__ runs.
        renderer_type = getattr(self.cfg.renderer_cfg, "renderer_type", None)
        if renderer_type == "isaac_rtx":
            from isaaclab.app.settings_manager import get_settings_manager

            settings = get_settings_manager()
            settings.set_bool("/isaaclab/render/rtx_sensors", True)
            if require_hdr_output:
                settings.set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)
        elif renderer_type == "ovrtx" and require_hdr_output:
            from isaaclab.app.settings_manager import get_settings_manager

            get_settings_manager().set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)
            # FIXME: settings set_bool is a no-op for ovrtx
            # warning only since it affects only ParticleField3DGaussianSplat scene
            logger.warning(
                "OVRTX backend with PPISP/HDR requires /rtx/rtpt/gaussian/skipTonemapping/enabled to be false."
            )

        # UsdGeom Camera prim for the sensor
        self._sensor_prims: list[UsdGeom.Camera] = list()
        # Allocated in :meth:`_create_buffers` once the renderer's output contract is known.
        self._data: CameraData | None = None
        # Renderer and render data — assigned in _initialize_impl.
        self._renderer: BaseRenderer | None = None
        self._render_data = None

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
    def frame(self) -> ProxyArray:
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
        self, matrices: torch.Tensor | wp.array, focal_length: float | None = None, env_ids: Sequence[int] | None = None
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
        if isinstance(matrices, torch.Tensor):
            if not matrices.is_contiguous():
                matrices = matrices.contiguous()
            matrices = wp.from_torch(matrices)
        elif not isinstance(matrices, wp.array):
            raise TypeError(f"Unsupported type for matrices: {type(matrices)}. Expected torch.Tensor or wp.array.")

        if env_ids is None:
            env_ids_np = np.arange(self._view.count)
        elif isinstance(env_ids, slice):
            env_ids_np = np.arange(self._view.count)[env_ids]
        else:
            env_ids_np = np.asarray(env_ids, dtype=np.int32).reshape(-1)

        matrices = matrices.numpy().astype(float, copy=False)
        if matrices.ndim == 2:
            matrices = matrices[None, ...]
        # iterate over env_ids
        for i, intrinsic_matrix in zip(env_ids_np, matrices):
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
        self._update_intrinsic_matrices(env_ids_np)

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
        pos_wp = None
        if positions is not None:
            if isinstance(positions, np.ndarray):
                positions = torch.from_numpy(positions).to(device=self._device)
            elif not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions, device=self._device)
            positions = positions.to(device=self._device, dtype=torch.float32).reshape(-1, 3)
            pos_wp = wp.from_torch(positions.contiguous(), dtype=wp.vec3f)
        ori_wp = None
        if orientations is not None:
            if isinstance(orientations, np.ndarray):
                orientations = torch.from_numpy(orientations).to(device=self._device)
            elif not isinstance(orientations, torch.Tensor):
                orientations = torch.tensor(orientations, device=self._device)
            orientations = orientations.to(device=self._device, dtype=torch.float32).reshape(-1, 4)
            orientations = convert_camera_frame_orientation_convention(orientations, origin=convention, target="opengl")
            ori_wp = wp.from_torch(orientations.contiguous(), dtype=wp.vec4f)
        idx_wp = self._resolve_env_ids_wp(env_ids)
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
            ValueError: If every eye position equals its target (look-at direction undefined for the
                whole batch). When only some rows are degenerate, those rows are skipped and the
                remaining poses are still applied; a warning is logged.
        """
        if isinstance(eyes, np.ndarray):
            eyes = torch.from_numpy(eyes).to(device=self._device)
        elif not isinstance(eyes, torch.Tensor):
            eyes = torch.tensor(eyes, device=self._device)
        eyes = eyes.to(device=self._device, dtype=torch.float32).reshape(-1, 3)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).to(device=self._device)
        elif not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self._device)
        targets = targets.to(device=self._device, dtype=torch.float32).reshape(-1, 3)
        if env_ids is None:
            env_ids_torch = torch.arange(self._view.count, dtype=torch.int32, device=self._device)
        elif isinstance(env_ids, slice):
            env_ids_torch = torch.arange(self._view.count, dtype=torch.int32, device=self._device)[env_ids]
        elif isinstance(env_ids, wp.array):
            env_ids_torch = wp.to_torch(env_ids).to(device=self._device, dtype=torch.int32).reshape(-1)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_torch = env_ids.to(device=self._device, dtype=torch.int32).reshape(-1)
        else:
            env_ids_torch = torch.tensor(env_ids, dtype=torch.int32, device=self._device).reshape(-1)
        # get up axis of current stage
        up_axis = UsdGeom.GetStageUpAxis(self.stage)
        # set camera poses using the view; degenerate rows (eye == target) come back as NaN
        rotation_matrix = create_rotation_matrix_from_view(eyes, targets, up_axis, device=self._device)
        valid_indices = (~torch.isnan(rotation_matrix).any(dim=(-2, -1))).nonzero(as_tuple=True)[0]
        n_valid = valid_indices.numel()
        n_total = rotation_matrix.shape[0]
        if n_valid == 0:
            raise ValueError("look-at is undefined: every eye position equals its target")
        if n_valid < n_total:
            logger.warning(
                "set_world_poses_from_view: skipping %d pose(s) where eye equals target",
                n_total - n_valid,
            )
            rotation_matrix = rotation_matrix.index_select(0, valid_indices)
            eyes = eyes.index_select(0, valid_indices)
            env_ids_torch = env_ids_torch.index_select(0, valid_indices)
        orientations = quat_from_matrix(rotation_matrix)
        idx_wp = wp.from_torch(env_ids_torch.contiguous(), dtype=wp.int32)
        self._view.set_world_poses(
            wp.from_torch(eyes.contiguous(), dtype=wp.vec3f),
            wp.from_torch(orientations.contiguous(), dtype=wp.vec4f),
            idx_wp,
        )

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
        # reset the data
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        if env_mask is not None:
            self._update_poses(env_mask=env_mask, frame_op=2)
        elif env_ids is None:
            self._update_poses(frame_op=2)
        else:
            env_ids_wp = self._resolve_env_ids_wp(env_ids)
            self._update_poses(env_ids_wp, frame_op=2)

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        This function obtains the simulation-scoped :class:`~isaaclab.renderers.base_renderer.BaseRenderer`
        from :attr:`~isaaclab.sim.simulation_context.SimulationContext.render_context` using the configured
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

        sim_ctx = sim_utils.SimulationContext.instance()
        if sim_ctx is None:
            raise RuntimeError("SimulationContext is not initialized.")
        self._renderer = sim_ctx.render_context.get_renderer(self.cfg.renderer_cfg)
        logger.info("Using renderer: %s", type(self._renderer).__name__)

        # Build the render spec early — both the wrapper ISP (which delegates
        # any renderer-side per-camera setup) and ``create_render_data`` consume
        # it, and the prims are already authored at this point.
        cam_paths = tuple(str(p.GetPath()) for p in sim_utils.find_matching_prims(self.cfg.prim_path, self.stage))
        env_0_prefix = "/World/envs/env_0/"
        rel_under_env0 = (
            cam_paths[0].removeprefix(env_0_prefix) if cam_paths and cam_paths[0].startswith(env_0_prefix) else ""
        )
        device_str = self._device if isinstance(self._device, str) else str(self._device)
        render_spec = CameraRenderSpec(
            cfg=self.cfg,
            device=device_str,
            num_instances=self._num_envs,
            camera_prim_paths=cam_paths,
            view_count=self._num_envs,
            camera_path_relative_to_env_0=rel_under_env0,
        )

        # Delegate per-camera USD setup to the renderer — must run **before**
        # ``ensure_prepare_stage`` so renderers that snapshot the stage
        # (ovrtx's ``stage.Export``) capture the resulting overrides in their
        # exported USD.
        self._renderer.prepare_cameras(self.stage, render_spec)

        # Stage preprocessing must happen before creating the view because the view keeps
        # references to prims located in the stage.
        sim_ctx.render_context.ensure_prepare_stage(self.stage, self._num_envs)

        self._view = FrameView(self.cfg.prim_path, device=self._device, stage=self.stage)
        # Check that sizes are correct
        if self._view.count != self._num_envs:
            raise RuntimeError(
                f"Number of camera prims in the view ({self._view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )

        # Create all env_ids buffer
        self._ALL_INDICES = wp.array(np.arange(self._view.count, dtype=np.int32), device=self._device)
        # Create frame count buffer
        self._frame = ProxyArray(wp.zeros(self._view.count, device=self._device, dtype=wp.int64))

        # Convert all encapsulated prims to Camera. Newton keeps only source USD camera prims.
        self._sensor_prims.clear()
        view_prims = list(self._view.prims)
        if not view_prims and cam_paths:
            view_prims = [self.stage.GetPrimAtPath(cam_paths[0])] * self._view.count
        for cam_prim in view_prims:
            # Obtain the prim path
            cam_prim_path = cam_prim.GetPath().pathString
            # Check if prim is a camera
            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")
            # Add to list
            self._sensor_prims.append(UsdGeom.Camera(cam_prim))

        self._render_data = self._renderer.create_render_data(render_spec)

        # Create internal buffers (includes intrinsic matrix and pose init)
        self._create_buffers()

    def _update_buffers_impl(self, env_mask: wp.array):
        if not self._env_mask_has_any(env_mask):
            return
        # Increment frame count
        if self.cfg.update_latest_camera_pose:
            self._update_poses(env_mask=env_mask, frame_op=1)
        else:
            self._update_camera_state(env_mask=env_mask, frame_op=1)

        sim_ctx = sim_utils.SimulationContext.instance()
        renderer = self._renderer
        assert renderer is not None
        if sim_ctx is not None:
            sim_ctx.render_context.render_into_camera(
                renderer,
                self._render_data,
                self._data,
                sim_ctx.get_physics_step_count(),
            )
        else:
            renderer.render(self._render_data)
            renderer.read_output(self._render_data, self._data)

    """
    Private Helpers
    """

    def _check_supported_data_types(self, cfg: CameraCfg):
        """Checks if the data types are supported by the ray-caster camera."""
        # check if there is any intersection in unsupported types
        # reason: these use np structured data types which are not compatible with the camera buffer contract
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
                "can't be stored in the camera output buffers easily."
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
        device_str = self._device if isinstance(self._device, str) else str(self._device)
        self._data = CameraData.allocate(
            data_types=known,
            height=self.cfg.height,
            width=self.cfg.width,
            num_views=self._view.count,
            device=self._device,
            supported_specs=specs,
        )
        # Camera-frame state (pose / intrinsics) is owned by the camera, not
        # the renderer: allocate warp buffers and populate them.
        self._data.create_buffers(self._view.count, device_str)
        self._update_intrinsic_matrices()
        self._update_poses()
        self._renderer.set_outputs(self._render_data, self._data.output)

    def _update_intrinsic_matrices(self, env_ids: Sequence[int] | wp.array | None = None):
        """Compute camera's matrix of intrinsic parameters.

        Also called calibration matrix. This matrix works for linear depth images. We assume square pixels.

        .. note::
            The calibration matrix projects points in the 3D scene onto an imaginary screen of the camera.
            The coordinates of points on the image plane are in the homogeneous representation.
        """
        env_ids_np = self._resolve_env_ids_np(env_ids)
        if len(env_ids_np) == 0:
            return

        intrinsic_matrices = np.zeros((len(env_ids_np), 3, 3), dtype=np.float32)
        # iterate over all cameras
        for matrix_id, i in enumerate(env_ids_np):
            # Get corresponding sensor prim
            sensor_prim = self._sensor_prims[int(i)]
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
            intrinsic_matrices[matrix_id, 0, 0] = f_x
            intrinsic_matrices[matrix_id, 0, 2] = c_x
            intrinsic_matrices[matrix_id, 1, 1] = f_y
            intrinsic_matrices[matrix_id, 1, 2] = c_y
            intrinsic_matrices[matrix_id, 2, 2] = 1.0

        intrinsic_matrices_wp = wp.array(intrinsic_matrices, dtype=wp.mat33f, device=self._device)
        self._update_camera_state(
            env_ids=None if env_ids is None else self._resolve_env_ids_wp(env_ids_np),
            intrinsics_src=intrinsic_matrices_wp,
            update_intrinsics=True,
        )

    def _update_poses(
        self, env_ids: Sequence[int] | wp.array | None = None, env_mask: wp.array | None = None, frame_op: int = 0
    ):
        """Computes the pose of the camera in the world frame with ROS convention.

        This methods uses the ROS convention to resolve the input pose. In this convention,
        we assume that the camera front-axis is +Z-axis and up-axis is -Y-axis.

        Returns:
            A tuple of the position (in meters) and quaternion (x, y, z, w).
        """
        # check camera prim exists
        if len(self._sensor_prims) == 0:
            raise RuntimeError("Camera prim is None. Please call 'sim.play()' first.")

        # get the poses from the view (returns ProxyArray)
        env_ids_wp = None if env_mask is not None else self._resolve_env_ids_wp(env_ids)
        pos_w, quat_w = self._view.get_world_poses(env_ids_wp)
        pos_w_wp = pos_w.warp
        pos_w_wp = wp.array(
            ptr=pos_w_wp.ptr,
            dtype=wp.vec3f,
            shape=(pos_w_wp.shape[0],),
            device=pos_w_wp.device,
            copy=False,
        )
        quat_w_wp = quat_w.warp
        quat_w_wp = wp.array(
            ptr=quat_w_wp.ptr,
            dtype=wp.quatf,
            shape=(quat_w_wp.shape[0],),
            device=quat_w_wp.device,
            copy=False,
        )

        self._update_camera_state(
            env_ids=env_ids_wp,
            env_mask=env_mask,
            pos_src=pos_w_wp,
            quat_src=quat_w_wp,
            update_pose=True,
            frame_op=frame_op,
        )
        # notify renderer of updated poses (guarded in case called before initialization completes)
        if self._render_data is not None:
            self._renderer.update_camera(
                self._render_data, self._data.pos_w, self._data.quat_w_world, self._data.intrinsic_matrices
            )

    def _update_camera_state(
        self,
        env_ids: wp.array | None = None,
        env_mask: wp.array | None = None,
        pos_src: wp.array | None = None,
        quat_src: wp.array | None = None,
        intrinsics_src: wp.array | None = None,
        update_pose: bool = False,
        update_intrinsics: bool = False,
        frame_op: int = 0,
    ):
        """Update camera pose, intrinsics, and frame counters through one Warp kernel."""
        count = env_ids.shape[0] if env_ids is not None else self._view.count
        if count == 0:
            return
        wp.launch(
            _camera_update_state_kernel,
            dim=count,
            inputs=[
                pos_src if pos_src is not None else self._data.pos_w.warp,
                quat_src if quat_src is not None else self._data.quat_w_world.warp,
                intrinsics_src if intrinsics_src is not None else self._data.intrinsic_matrices.warp,
                self._data.pos_w.warp,
                self._data.quat_w_world.warp,
                self._data.intrinsic_matrices.warp,
                self._frame.warp,
                env_mask if env_mask is not None else self._ALL_ENV_MASK,
                env_ids if env_ids is not None else self._ALL_INDICES,
                env_ids is not None,
                env_mask is not None,
                update_pose,
                update_intrinsics,
                frame_op,
            ],
            device=self._device,
        )

    def _resolve_env_ids_np(self, env_ids: Sequence[int] | wp.array | None) -> np.ndarray:
        """Resolve camera indices to a host ``int32`` array for USD metadata reads."""
        if env_ids is None:
            return np.arange(self._view.count, dtype=np.int32)
        if isinstance(env_ids, slice):
            return np.arange(self._view.count, dtype=np.int32)[env_ids]
        if isinstance(env_ids, wp.array):
            return env_ids.numpy().astype(np.int32, copy=False).reshape(-1)
        return np.asarray(env_ids, dtype=np.int32).reshape(-1)

    def _resolve_env_ids_wp(self, env_ids: Sequence[int] | torch.Tensor | wp.array | slice | None) -> wp.array | None:
        """Resolve camera indices to a Warp ``int32`` array."""
        if env_ids is None:
            return None
        if isinstance(env_ids, wp.array):
            if env_ids.dtype != wp.int32:
                raise TypeError(f"Unsupported wp.array dtype for env_ids: {env_ids.dtype}. Expected wp.int32.")
            if str(env_ids.device) == str(self._device):
                return env_ids
            env_ids = env_ids.numpy().astype(np.int32, copy=False).reshape(-1)
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(device=self._device, dtype=torch.int32).reshape(-1)
            if not env_ids.is_contiguous():
                env_ids = env_ids.contiguous()
            return wp.from_torch(env_ids, dtype=wp.int32)
        elif isinstance(env_ids, slice):
            env_ids = np.arange(self._view.count, dtype=np.int32)[env_ids]
        else:
            env_ids = np.asarray(env_ids, dtype=np.int32).reshape(-1)
        return wp.array(env_ids, dtype=wp.int32, device=self._device)

    @staticmethod
    def _env_mask_has_any(env_mask: wp.array) -> bool:
        """Return whether the mask selects any camera."""
        return bool(np.any(env_mask.numpy()))

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        if self._renderer is not None and self._render_data is not None:
            self._renderer.cleanup(self._render_data)
        self._render_data = None
        self._renderer = None
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._view = None
