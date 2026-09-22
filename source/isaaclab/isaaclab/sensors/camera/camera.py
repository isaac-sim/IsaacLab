# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import warp as wp

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.app.logging_utils import force_log_level
from isaaclab.renderers import BaseRenderer, CameraRenderSpec
from isaaclab.sim.views import FrameView
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

logger = logging.getLogger(__name__)


# Keep runtime calibration out of the pose module compiled during camera initialization.
@wp.kernel(module=f"{__name__}.runtime_intrinsics", enable_backward=False)
def _camera_select_intrinsics_kernel(
    matrices: wp.array3d(dtype=Any),
    env_ids: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.bool),
    width: int,
    height: int,
    rows: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
):
    """Validate a runtime device batch and select the last input row for each camera."""
    row = wp.tid()
    index = env_ids[row]
    if index < 0:
        index += rows.shape[0]
    if index < 0 or index >= rows.shape[0]:
        wp.atomic_or(status, 0, 1)
        return
    if fixed[index]:
        wp.atomic_or(status, 0, 8)
        return
    wp.atomic_max(rows, index, row)
    if wp.abs(wp.float64(matrices[row, 0, 0]) - wp.float64(matrices[row, 1, 1])) > wp.float64(1.0e-4):
        wp.atomic_or(status, 0, 2)
    if wp.abs(wp.float64(matrices[row, 0, 2]) - wp.float64(width) * wp.float64(0.5)) > wp.float64(1.0e-4) or wp.abs(
        wp.float64(matrices[row, 1, 2]) - wp.float64(height) * wp.float64(0.5)
    ) > wp.float64(1.0e-4):
        wp.atomic_or(status, 0, 4)


@wp.kernel(module=f"{__name__}.runtime_intrinsics", enable_backward=False)
def _camera_set_intrinsics_kernel(
    matrices: wp.array3d(dtype=Any),
    rows: wp.array(dtype=wp.int32),
    width: int,
    height: int,
    focal_length: wp.float64,
    use_focal_length: bool,
    current_matrices: wp.array(dtype=wp.mat33f),
    current_parameters: wp.array2d(dtype=wp.float32),
    output_matrices: wp.array(dtype=wp.mat33f),
    output_parameters: wp.array2d(dtype=wp.float32),
):
    index = wp.tid()
    row = rows[index]
    output_matrices[index] = current_matrices[index]
    for attribute in range(5):
        output_parameters[attribute, index] = current_parameters[attribute, index]
    if row < 0:
        return
    mean_focal = (wp.float64(matrices[row, 0, 0]) + wp.float64(matrices[row, 1, 1])) * wp.float64(0.5)
    pixel_size = wp.float64(1.0) / wp.float64(width)
    if use_focal_length:
        pixel_size = focal_length / mean_focal
    focal = wp.float32(pixel_size * mean_focal)
    horizontal = wp.float32(pixel_size * wp.float64(width))
    output_parameters[0, index] = focal
    output_parameters[1, index] = horizontal
    output_parameters[2, index] = wp.float32(pixel_size * wp.float64(height))
    output_parameters[3, index] = 0.0
    output_parameters[4, index] = 0.0
    fx = wp.float32(wp.float64(width) * wp.float64(focal) / wp.float64(horizontal))
    output_matrices[index] = wp.mat33f(fx, 0.0, float(width) * 0.5, 0.0, fx, float(height) * 0.5, 0.0, 0.0, 1.0)


# Register supported projection precisions together so later calls do not rebuild the module.
for _dtype in (wp.float32, wp.float64):
    wp.overload(_camera_select_intrinsics_kernel, {"matrices": wp.array3d(dtype=_dtype)})
    wp.overload(_camera_set_intrinsics_kernel, {"matrices": wp.array3d(dtype=_dtype)})


@wp.kernel
def _camera_update_state_kernel(
    pos_src: wp.array(dtype=wp.vec3f),
    quat_src: wp.array(dtype=wp.quatf),
    pos_dst: wp.array(dtype=wp.vec3f),
    quat_world_dst: wp.array(dtype=wp.quatf),
    frame: wp.array(dtype=wp.int64),
    env_mask: wp.array(dtype=wp.bool),
    env_ids: wp.array(dtype=wp.int32),
    use_env_ids: bool,
    use_env_mask: bool,
    update_pose: bool,
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
    - ``"instance_segmentation"``: The semantic instance segmentation data.
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
            ValueError: If the provided data types are not supported by the camera or active renderer.
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
            probe_path = spawn.spawn_path or self.cfg.prim_path
            probe_matches = sim_utils.resolve_matching_prims_from_source(probe_path, raise_if_no_matches=False)
            source_prim, _source_destination_expr = probe_matches[0] if probe_matches else (None, None)
            if source_prim is not None and (
                source_prim.HasAPI(UsdPhysics.ArticulationRootAPI) or source_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                logger.info(f" Spawning camera at '{self.cfg.prim_path}/camera'.")
                self.cfg.prim_path = f"{self.cfg.prim_path}/camera"
                spawn.spawn_path = f"{probe_path}/camera"

            spawn_target = spawn.spawn_path or self.cfg.prim_path
            if sim_utils.find_first_matching_prim(spawn_target) is None:
                spawn.func(spawn_target, spawn, translation=self.cfg.offset.pos, orientation=rot_offset)
            if not sim_utils.find_matching_prims(spawn_target):
                raise RuntimeError(f"Could not find prim with path {spawn_target!r}.")

        # Every renderer backend draws the visual-only geometry, so it must survive cloning even
        # when the run is otherwise headless. This must happen before plan dispatch, which is why
        # it is here rather than in ``_initialize_impl``.
        sim_ctx = sim_utils.SimulationContext.instance()
        if sim_ctx is not None:
            sim_ctx.require_visual_shapes()

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
            settings.set_bool("/physics/fabricUpdateTransformations", True)
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
        self._sensor_prims: list[UsdGeom.Camera] = []
        # Allocated in :meth:`_create_buffers` once the renderer's output contract is known.
        self._data: CameraData | None = None
        # The backend's ``__init__`` is its pre-physics phase, so it has to exist before
        # ``sim.reset()``; sensor initialization only runs on ``PhysicsEvent.PHYSICS_READY``, which is
        # too late. Backends are shared per renderer config, so this stays cheap for many cameras.
        self._renderer: BaseRenderer | None = None
        if sim_ctx is not None:
            self._renderer = sim_ctx.get_or_create_backend(self.cfg.renderer_cfg)
            with force_log_level(logging.INFO):
                logger.info("Using renderer: %s", type(self._renderer).__name__)
        # Render data — assigned in _initialize_impl.
        self._render_data = None
        # Frame view — assigned in _initialize_impl.
        self._view: FrameView | None = None

    def __del__(self, _sys=sys):
        """Unsubscribes from callbacks and cleans up renderer resources.

        Skips cleanup during interpreter shutdown so destructor-time imports or renderer teardown
        cannot raise ``ImportError: sys.meta_path is None`` and mask the original exception.
        """
        if _sys.is_finalizing() or _sys.meta_path is None:
            return
        # unsubscribe callbacks
        super().__del__()

        # release the frame view's backend state, getattr covers partial initialization
        if getattr(self, "_view", None) is not None:
            self._view.close()
            self._view = None
        # cleanup render resources (renderer may be None if never initialized)
        if getattr(self, "_renderer", None) is not None:
            self._renderer.cleanup(getattr(self, "_render_data", None))

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
        self,
        matrices: np.ndarray | torch.Tensor | wp.array,
        focal_length: float | None = None,
        env_ids: Sequence[int] | torch.Tensor | wp.array | slice | None = None,
    ):
        """Set runtime camera calibration without authoring USD.

        Camera buffers own the active calibration. The renderer receives device arrays directly:
        Kit writes Fabric, OVRTX writes its scene, and Newton updates its persistent ray field.
        USD is read at initialization; saving USD does not save these runtime overrides. To persist
        calibration, configure :meth:`PinholeCameraCfg.from_intrinsic_matrix` when spawning cameras.

        Ordinary pinhole cameras retain square pixels and a centered principal point: ``fx/fy`` are
        averaged and ``cx/cy`` become half the image dimensions. OpenCV distortion cameras retain
        their spawn-time calibration and are skipped with a warning.

        All three backends support uniform runtime updates to ordinary pinhole cameras: all
        environments in a camera batch use the same new calibration.

        .. warning::
            Different intrinsics per environment are unsupported in the tested tiled configurations:

            * Kit/OVRTX update per-view attributes and reported intrinsic matrices, but rendered
              views still use the first camera's projection. This also occurs with direct USD writes.
            * Newton Warp shares one ray field across environments. It raises :class:`ValueError`
              if the resulting calibration differs between environments, leaving active calibration
              unchanged.

            Per-environment intrinsic randomization therefore requires additional renderer support;
            successful attribute updates alone do not establish correct independent projections.

        Initialization builds calibration in NumPy without compiling this setter's Warp kernels.
        Runtime selection and projection conversion execute on the camera device, including for host
        inputs, which are uploaded first. Matrix and index batches are never copied back to the CPU.

        Shape and batch cardinality are checked before any changes. Index validation and projection
        warnings transfer one status integer to the host, so this method is not CUDA graph capturable.
        Backend validation completes before the public calibration buffers are committed. The first
        runtime update may compile Warp kernels; repeated updates reuse the compiled kernels.

        Args:
            matrices: NumPy or Torch intrinsic matrices [pixel], shape (N, 3, 3), or a Warp array of ``wp.mat33f`` /
                ``wp.mat33d``. A single (3, 3) matrix is accepted for one selected camera.
            focal_length: Perspective focal length in the scene's camera length units. If None,
                a pixel size of 1 / width is used to derive the focal length from the matrix.
            env_ids: Sensor indices in matrix order. Defaults to all cameras. Repeated indices
                retain the last matrix. Newton requires the resulting calibration to be shared
                across environments because its native ray field has no world dimension.

        Raises:
            TypeError: If the matrices or indices have unsupported types.
            ValueError: If matrix shape, batch cardinality, or a backend calibration constraint fails.
            IndexError: If a selected camera index is out of range.
        """
        if isinstance(matrices, torch.Tensor):
            if matrices.ndim == 2:
                matrices = matrices.unsqueeze(0)
            matrices = wp.from_torch(matrices)
        elif isinstance(matrices, np.ndarray):
            matrices = wp.array(matrices, device=self._device)
        elif not isinstance(matrices, wp.array):
            raise TypeError(f"Unsupported matrices: {type(matrices)}. Expected np.ndarray, torch.Tensor or wp.array.")
        if matrices.dtype in (wp.mat33f, wp.mat33d):
            matrices = matrices.view(wp.float32 if matrices.dtype == wp.mat33f else wp.float64)
        if matrices.ndim == 2:
            matrices = matrices.contiguous().reshape((1, *matrices.shape))
        if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
            raise ValueError(f"Expected intrinsic matrices with shape (N, 3, 3), got {matrices.shape}.")
        indices = self._ALL_INDICES if env_ids is None else self._resolve_env_ids_wp(env_ids)
        if matrices.shape[0] != indices.shape[0]:
            raise ValueError(
                "The number of intrinsic matrices must match the number of selected cameras: "
                f"got {matrices.shape[0]} matrices for {indices.shape[0]} cameras."
            )
        if indices.shape[0] == 0:
            return
        matrices = matrices.to(self._device)
        height, width = self.image_shape
        self._intrinsic_rows.fill_(-1)
        self._intrinsic_status.zero_()
        wp.launch(
            _camera_select_intrinsics_kernel,
            dim=indices.shape[0],
            inputs=[
                matrices,
                indices,
                self._intrinsic_fixed,
                width,
                height,
                self._intrinsic_rows,
                self._intrinsic_status,
            ],
            device=self._device,
        )
        # Synchronize to raise index errors before backend writes, without downloading the input batch.
        status = int(self._intrinsic_status.numpy()[0])
        if status & 1:
            raise IndexError("Camera indices are out of range.")
        if status & 2:
            logger.warning("Camera non square pixels are not supported; the average of f_x and f_y is used.")
        if status & 4:
            logger.warning("Camera aperture offsets are not supported; c_x and c_y are half of width and height.")
        if status & 8:
            logger.warning("set_intrinsic_matrices() skipped cameras with an OpenCV lens-distortion model.")
        wp.launch(
            _camera_set_intrinsics_kernel,
            dim=self._view.count,
            inputs=[
                matrices,
                self._intrinsic_rows,
                width,
                height,
                wp.float64(focal_length or 0.0),
                focal_length is not None,
                self._data.intrinsic_matrices.warp,
                self._intrinsic_parameters,
                self._intrinsic_pending,
                self._intrinsic_parameters_pending,
            ],
            device=self._device,
        )
        # Let the backend validate its constraints before committing the public calibration buffers.
        self._renderer.update_camera_intrinsics(
            self._render_data, self._intrinsic_pending, self._intrinsic_parameters_pending
        )
        wp.copy(self._data.intrinsic_matrices.warp, self._intrinsic_pending)
        wp.copy(self._intrinsic_parameters, self._intrinsic_parameters_pending)

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
            positions = self._as_device_tensor(positions, 3)
            pos_wp = wp.from_torch(positions.contiguous(), dtype=wp.vec3f)
        ori_wp = None
        if orientations is not None:
            orientations = self._as_device_tensor(orientations, 4)
            orientations = convert_camera_frame_orientation_convention(orientations, origin=convention, target="opengl")
            ori_wp = wp.from_torch(orientations.contiguous(), dtype=wp.vec4f)
        idx_wp = self._resolve_env_ids_wp(env_ids)
        with self._view.xform_world_space_writer() as writer:
            writer.set_poses(pos_wp, ori_wp, idx_wp)
        # write through to the data buffers so explicitly set poses are never stale,
        # regardless of :attr:`CameraCfg.update_latest_camera_pose`
        self._update_poses(env_ids=idx_wp, frame_op=0)

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
        eyes = self._as_device_tensor(eyes, 3)
        targets = self._as_device_tensor(targets, 3)
        env_ids_wp = self._resolve_env_ids_wp(env_ids)
        env_ids_torch = wp.to_torch(self._ALL_INDICES if env_ids_wp is None else env_ids_wp)
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
        with self._view.xform_world_space_writer() as writer:
            writer.set_poses(
                wp.from_torch(eyes.contiguous(), dtype=wp.vec3f),
                wp.from_torch(orientations.contiguous(), dtype=wp.vec4f),
                idx_wp,
            )
        # write through to the data buffers so explicitly set poses are never stale,
        # regardless of :attr:`CameraCfg.update_latest_camera_pose`
        self._update_poses(env_ids=idx_wp, frame_op=0)

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        if not self._is_initialized:
            raise RuntimeError("Camera could not be initialized. Check the renderer and simulation logs for details.")
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

        This function delegates all render-product and annotator management to the
        :class:`~isaaclab.renderers.base_renderer.BaseRenderer` created in :meth:`__init__`. It also
        initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the number of camera prims in the view does not match the number of environments.
            RuntimeError: Propagated from the renderer constructor when the active backend's runtime requirements
                are not satisfied.
        """
        # Initialize parent class
        super()._initialize_impl()

        sim_ctx = sim_utils.SimulationContext.instance()
        if sim_ctx is None:
            raise RuntimeError("SimulationContext is not initialized.")
        # Normally created in ``__init__``; only missing when the camera was built without a simulation.
        if self._renderer is None:
            self._renderer = sim_ctx.get_or_create_backend(self.cfg.renderer_cfg)

        # Build the render spec early — both the wrapper ISP (which delegates
        # any renderer-side per-camera setup) and ``create_render_data`` consume
        # it, and the prims are already authored at this point.
        cam_paths = tuple(str(p.GetPath()) for p in sim_utils.find_matching_prims(self.cfg.prim_path, self.stage))
        env_0_prefix = "/World/envs/env_0/"
        rel_under_env0 = (
            cam_paths[0].removeprefix(env_0_prefix) if cam_paths and cam_paths[0].startswith(env_0_prefix) else ""
        )
        render_spec = CameraRenderSpec(
            cfg=self.cfg,
            device=str(self._device),
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
        if "instance_segmentation_fast" in cfg.data_types:
            raise ValueError(
                "The data type 'instance_segmentation_fast' has been renamed to 'instance_segmentation'."
                " Please update your CameraCfg.data_types and any camera.data.output / camera.data.info key"
                " lookups to use 'instance_segmentation'."
            )
        # check if there is any intersection in unsupported types
        # reason: these use np structured data types which are not compatible with the camera buffer contract
        common_elements = set(cfg.data_types) & Camera.UNSUPPORTED_TYPES
        if common_elements:
            # provide alternative fast counterparts
            fast_common_elements = []
            for item in common_elements:
                if "instance_id_segmentation" in item:
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
        # Split requested names into known, unknown, and unsupported types.
        known: list[str] = []
        unknown: list[str] = []
        unsupported: list[str] = []
        for name in self.cfg.data_types:
            try:
                if RenderBufferKind(name) in specs:
                    known.append(name)
                else:
                    unsupported.append(name)
            except ValueError:
                unknown.append(name)
        errors = []
        if unknown:
            errors.append(f"Unknown camera data types: {unknown}.")
        if unsupported:
            errors.append(
                f"Renderer {type(self._renderer).__name__} does not support the following requested data types:"
                f" {unsupported}."
                f"\n\tSupported data types: {sorted(str(kind) for kind in specs)}"
            )
        if errors:
            raise ValueError("\n".join(errors))
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
        self._data.create_buffers(self._view.count, str(self._device))
        self._initialize_intrinsics()
        self._update_poses()
        self._renderer.set_outputs(self._render_data, self._data.output)

    def _read_authored_opencv_intrinsics(
        self, prim: Usd.Prim, width: int, height: int, env_id: int
    ) -> tuple[float, float, float, float] | None:
        """Read the authored OpenCV lens-distortion intrinsics from a camera prim.

        Returns the calibrated ``(fx, fy, cx, cy)`` that the RTX/OVRTX renderer projects through when
        the prim carries a complete OpenCV lens-distortion model, otherwise ``None`` so the caller can
        fall back to the focal-length/aperture projection. Unlike that projection, these intrinsics may
        be non-square (``fx != fy``) or off-center.

        Args:
            prim: The camera prim to read the authored intrinsics from.
            width: The render width in pixels.
            height: The render height in pixels.
            env_id: The environment index, used only for warning messages.

        Returns:
            The authored ``(fx, fy, cx, cy)`` in pixels, or ``None`` when no complete model is present.
        """
        distortion_model = prim.GetAttribute("omni:lensdistortion:model").Get()
        if not distortion_model:
            return None
        prefix = f"omni:lensdistortion:{distortion_model}"
        # a prim may carry the model token without fx/fy/cx/cy
        intrinsics = tuple(prim.GetAttribute(f"{prefix}:{name}").Get() for name in ("fx", "fy", "cx", "cy"))
        if None in intrinsics:
            # a model token without intrinsics falls back to the focal-length/aperture projection
            logger.warning(
                "Camera prim '%s' declares lens-distortion model '%s' but is missing one or more"
                " fx/fy/cx/cy intrinsics; falling back to the focal-length/aperture projection.",
                prim.GetPath(),
                distortion_model,
            )
            return None
        # the intrinsics are reported in the calibrated pixel space; warn if the render resolution
        # differs, since the matrix will not match the rendered pixels.
        image_size = prim.GetAttribute(f"{prefix}:imageSize").Get()
        if image_size is not None:
            authored_size = (int(image_size[0]), int(image_size[1]))
            if authored_size != (width, height):
                logger.warning(
                    "Camera prim '%s' (env %d) lens-distortion 'imageSize' %s does not match the render"
                    " resolution (%d, %d). The reported intrinsic matrix is in the calibrated pixel space.",
                    prim.GetPath(),
                    env_id,
                    authored_size,
                    width,
                    height,
                )
        return tuple(float(value) for value in intrinsics)

    def _initialize_intrinsics(self):
        """Build initial calibration in NumPy, then upload persistent runtime buffers."""
        height, width = self.image_shape
        count = self._view.count
        matrices = np.zeros((count, 3, 3), dtype=np.float32)
        parameters = np.empty((5, count), dtype=np.float64)
        fixed = np.zeros(count, dtype=bool)
        pinhole = np.ones(count, dtype=bool)
        for i, sensor_prim in enumerate(self._sensor_prims):
            prim = sensor_prim.GetPrim()
            parameters[:, i] = (
                sensor_prim.GetFocalLengthAttr().Get(),
                sensor_prim.GetHorizontalApertureAttr().Get(),
                sensor_prim.GetVerticalApertureAttr().Get(),
                sensor_prim.GetHorizontalApertureOffsetAttr().Get(),
                sensor_prim.GetVerticalApertureOffsetAttr().Get(),
            )
            fixed[i] = bool(prim.GetAttribute("omni:lensdistortion:model").Get())
            authored = self._read_authored_opencv_intrinsics(prim, width, height, i) if fixed[i] else None
            if authored is not None:
                matrices[i, 0, 0], matrices[i, 1, 1], matrices[i, 0, 2], matrices[i, 1, 2] = authored
                pinhole[i] = False
        matrices[pinhole, 0, 0] = matrices[pinhole, 1, 1] = width * parameters[0, pinhole] / parameters[1, pinhole]
        matrices[pinhole, 0, 2] = width * 0.5
        matrices[pinhole, 1, 2] = height * 0.5
        matrices[:, 2, 2] = 1.0
        device = self._device
        wp.copy(self._data.intrinsic_matrices.warp, wp.array(matrices, dtype=wp.mat33f, device=device))
        self._intrinsic_parameters = wp.array(parameters, dtype=wp.float32, device=device)
        self._intrinsic_fixed = wp.array(fixed, dtype=wp.bool, device=device)
        self._intrinsic_rows = wp.empty(self._view.count, dtype=wp.int32, device=device)
        self._intrinsic_status = wp.zeros(1, dtype=wp.int32, device=device)
        self._intrinsic_pending = wp.empty_like(self._data.intrinsic_matrices.warp)
        self._intrinsic_parameters_pending = wp.empty_like(self._intrinsic_parameters)

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
        update_pose: bool = False,
        frame_op: int = 0,
    ):
        """Update camera pose and frame counters through one Warp kernel."""
        count = env_ids.shape[0] if env_ids is not None else self._view.count
        if count == 0:
            return
        wp.launch(
            _camera_update_state_kernel,
            dim=count,
            inputs=[
                pos_src if pos_src is not None else self._data.pos_w.warp,
                quat_src if quat_src is not None else self._data.quat_w_world.warp,
                self._data.pos_w.warp,
                self._data.quat_w_world.warp,
                self._frame.warp,
                env_mask if env_mask is not None else self._ALL_ENV_MASK,
                env_ids if env_ids is not None else self._ALL_INDICES,
                env_ids is not None,
                env_mask is not None,
                update_pose,
                frame_op,
            ],
            device=self._device,
        )

    def _as_device_tensor(self, value: np.ndarray | torch.Tensor | Sequence, num_cols: int) -> torch.Tensor:
        """Convert array-like input to a float32 tensor of shape (N, ``num_cols``) on the camera device."""
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        return value.to(device=self._device, dtype=torch.float32).reshape(-1, num_cols)

    def _resolve_env_ids_wp(self, env_ids: Sequence[int] | torch.Tensor | wp.array | slice | None) -> wp.array | None:
        """Resolve camera indices to a Warp ``int32`` array."""
        if env_ids is None:
            return None
        if isinstance(env_ids, wp.array):
            if env_ids.dtype != wp.int32:
                raise TypeError(f"Unsupported wp.array dtype for env_ids: {env_ids.dtype}. Expected wp.int32.")
            if env_ids.ndim != 1:
                raise ValueError("Warp camera indices must be a one-dimensional array.")
            if str(env_ids.device) == str(self._device):
                return env_ids
            return env_ids.to(self._device)
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
        # release backend state deterministically, then invalidate the view
        if self._view is not None:
            self._view.close()
            self._view = None
