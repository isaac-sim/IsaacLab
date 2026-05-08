# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import warp as wp

from pxr import UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.sensors as sensor_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.renderers import BaseRenderer, CameraRenderSpec
from isaaclab.sim.views import FrameView
from isaaclab.utils import to_camel_case
from isaaclab.utils.warp.proxy_array import ProxyArray

from ..sensor_base import SensorBase
from . import kernels as camera_kernels
from .camera_data import CameraData, RenderBufferKind
from .orientation_conventions import _resolve_conversion_quat, convert_quat_array

if TYPE_CHECKING:
    from .camera_cfg import CameraCfg

# import logger
logger = logging.getLogger(__name__)


def _to_wp_array(value: Any, device: str, dtype: Any | None = None) -> wp.array:
    """Normalize an array-like input to a :class:`warp.array` on ``device``.

    Accepts :class:`warp.array`, :class:`~isaaclab.utils.warp.proxy_array.ProxyArray`,
    :class:`numpy.ndarray`, lists/tuples of numbers, and torch tensors (detected
    structurally via :func:`warp.from_torch`'s interop — no torch import needed).

    The ``dtype`` argument, if provided, controls the returned ``wp.array`` dtype.
    """
    # ProxyArray and wp.array are the canonical fast paths.
    if isinstance(value, ProxyArray):
        arr = value.warp
        if dtype is not None and arr.dtype is not dtype:
            arr = wp.array(ptr=arr.ptr, dtype=dtype, shape=arr.shape, device=arr.device, copy=False)
        return arr
    if isinstance(value, wp.array):
        if dtype is not None and value.dtype is not dtype:
            value = wp.array(ptr=value.ptr, dtype=dtype, shape=value.shape, device=value.device, copy=False)
        return value
    # Torch tensors are detected by their module and the presence of ``data_ptr``,
    # so this module never imports torch itself. ``wp.from_torch`` lives in warp's
    # interop layer.
    if type(value).__module__.startswith("torch") and hasattr(value, "data_ptr"):
        contiguous = value.contiguous() if hasattr(value, "is_contiguous") and not value.is_contiguous() else value
        return wp.from_torch(contiguous, dtype=dtype) if dtype is not None else wp.from_torch(contiguous)
    # Numpy / lists / tuples — copy onto the device.
    np_value = np.asarray(value)
    return wp.array(np_value, dtype=dtype, device=device)


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

        # TODO(follow-up PR): move this flag flip out of Camera. The cleanest path is
        # an apply_pre_reset_settings() hook on RendererCfg (default no-op) that
        # IsaacRtxRendererCfg overrides to flip /isaaclab/render/rtx_sensors. The
        # flag must be set pre-sim.reset() because SimulationContext.is_rendering
        # and several env classes read it before the renderer's __init__ runs.
        renderer_type = getattr(self.cfg.renderer_cfg, "renderer_type", None)
        if renderer_type == "isaac_rtx":
            get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", True)

        # Compute the spawn orientation in OpenGL convention without involving torch:
        # the cfg-supplied quaternion is constant per cfg, so we resolve the
        # convention conversion as a single quaternion product on the CPU.
        rot_xyzw = tuple(float(c) for c in self.cfg.offset.rot)
        conv = _resolve_conversion_quat(self.cfg.offset.convention, "opengl")
        sx, sy, sz, sw = rot_xyzw
        cx, cy, cz, cw = conv
        rot_offset = np.array(
            [
                sw * cx + sx * cw + sy * cz - sz * cy,
                sw * cy - sx * cz + sy * cw + sz * cx,
                sw * cz + sx * cy - sy * cx + sz * cw,
                sw * cw - sx * cx - sy * cy - sz * cz,
            ],
            dtype=np.float32,
        )
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
        """Frame number when the measurement took place.

        Shape is (N,), dtype ``wp.int64``. Use ``.torch`` for a zero-copy
        :class:`torch.Tensor` view or ``.warp`` for the underlying
        :class:`warp.array`.
        """
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
        matrices: wp.array | ProxyArray | Any,
        focal_length: float | None = None,
        env_ids: Sequence[int] | None = None,
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
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3). A
                :class:`warp.array` is the canonical input type;
                :class:`~isaaclab.utils.warp.proxy_array.ProxyArray`,
                :class:`numpy.ndarray`, and :class:`torch.Tensor` are also accepted.
            focal_length: Perspective focal length (in cm) used to calculate pixel size. Defaults to None. If None,
                focal_length will be calculated 1 / width.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = list(range(self._view.count))
        # USD attribute writes are scalar host-side; pull a numpy view of the input.
        matrices_wp = _to_wp_array(matrices, device=self._device)
        matrices_np = matrices_wp.numpy()
        # iterate over env_ids
        for i, intrinsic_matrix in zip(env_ids, matrices_np):
            height, width = self.image_shape

            params = sensor_utils.convert_camera_intrinsics_to_usd(
                intrinsic_matrix=np.asarray(intrinsic_matrix, dtype=float).reshape(-1),
                height=height,
                width=width,
                focal_length=focal_length,
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
        positions: wp.array | ProxyArray | Any | None = None,
        orientations: wp.array | ProxyArray | Any | None = None,
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
            positions: The cartesian coordinates [m]. Shape is (N, 3). A :class:`warp.array`
                is the canonical input type; :class:`ProxyArray`, :class:`numpy.ndarray`,
                and :class:`torch.Tensor` are also accepted. Defaults to None, in which
                case the camera position is not changed.
            orientations: The quaternion orientation in (x, y, z, w). Shape is (N, 4).
                Same accepted types as ``positions``. Defaults to None, in which case
                the camera orientation is not changed.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
            convention: The convention in which the poses are fed. Defaults to "ros".

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
        """
        # Position buffer for FrameView: (N,) of wp.vec3.
        pos_wp: wp.array | None = None
        if positions is not None:
            pos_wp = _to_wp_array(positions, device=self._device, dtype=wp.vec3)

        # Orientation buffer for FrameView: (N,) of wp.quatf in the OpenGL convention.
        ori_wp: wp.array | None = None
        if orientations is not None:
            src_quat = _to_wp_array(orientations, device=self._device, dtype=wp.quatf)
            ori_wp = (
                convert_quat_array(src_quat, origin=convention, target="opengl") if convention != "opengl" else src_quat
            )

        idx_wp = self._resolve_env_indices(env_ids)
        self._view.set_world_poses(pos_wp, ori_wp, idx_wp)

    def set_world_poses_from_view(
        self,
        eyes: wp.array | ProxyArray | Any,
        targets: wp.array | ProxyArray | Any,
        env_ids: Sequence[int] | None = None,
    ):
        """Set the poses of the camera from the eye position and look-at target position.

        Args:
            eyes: The positions of the camera's eye [m]. Shape is (N, 3). A
                :class:`warp.array` is the canonical input type;
                :class:`ProxyArray`, :class:`numpy.ndarray`, and :class:`torch.Tensor`
                are also accepted.
            targets: The target locations to look at [m]. Shape is (N, 3). Same
                accepted types as ``eyes``.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.

        Raises:
            RuntimeError: If the camera prim is not set. Need to call :meth:`initialize` method first.
            NotImplementedError: If the stage up-axis is not "Y" or "Z".
        """
        eyes_wp = _to_wp_array(eyes, device=self._device, dtype=wp.vec3)
        targets_wp = _to_wp_array(targets, device=self._device, dtype=wp.vec3)
        up_axis = UsdGeom.GetStageUpAxis(self.stage)
        if up_axis == "Y":
            up_vec = wp.vec3(0.0, 1.0, 0.0)
        elif up_axis == "Z":
            up_vec = wp.vec3(0.0, 0.0, 1.0)
        else:
            raise NotImplementedError(f"Stage up axis {up_axis!r} is not supported (expected 'Y' or 'Z').")
        n = int(eyes_wp.shape[0])
        ori_wp = wp.empty(n, dtype=wp.quatf, device=self._device)
        wp.launch(
            kernel=camera_kernels.look_at_quat_kernel,
            dim=n,
            inputs=[eyes_wp, targets_wp, up_vec],
            outputs=[ori_wp],
            device=self._device,
        )
        idx_wp = self._resolve_env_indices(env_ids)
        self._view.set_world_poses(eyes_wp, ori_wp, idx_wp)

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
        # Resolve to a wp.int32 array of indices and a wp.bool mask covering them.
        env_ids_wp, env_mask_local = self._resolve_env_ids_or_mask(env_ids, env_mask)
        if env_ids_wp.shape[0] == 0:
            return
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        self._update_poses_wp(env_ids_wp)
        # Reset the frame count via a warp scatter kernel.
        wp.launch(
            camera_kernels.masked_set_int64_kernel,
            dim=self._view.count,
            inputs=[env_mask_local, wp.int64(0), self._frame.warp],
            device=self._device,
        )

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

        # ALL_INDICES: wp.int32 array (idiomatic for indexing into other wp.arrays).
        self._ALL_INDICES = wp.array(np.arange(self._view.count, dtype=np.int32), device=self._device)
        # Frame count buffer (wp.array primary, ProxyArray-published).
        self._frame = ProxyArray(wp.zeros(self._view.count, dtype=wp.int64, device=self._device))
        # Pre-allocate scratch buffer for env_mask -> indices conversion.
        self._env_indices_scratch = wp.empty(self._view.count, dtype=wp.int32, device=self._device)
        self._env_indices_count = wp.zeros(1, dtype=wp.int32, device=self._device)

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
        cam_paths = tuple(cam_prim.GetPath().pathString for cam_prim in self._view.prims)
        env_0_prefix = "/World/envs/env_0/"
        rel_under_env0 = (
            cam_paths[0].removeprefix(env_0_prefix) if cam_paths and cam_paths[0].startswith(env_0_prefix) else ""
        )
        device_str = self._device if isinstance(self._device, str) else str(self._device)
        render_spec = CameraRenderSpec(
            cfg=self.cfg,
            device=device_str,
            num_instances=self.num_instances,
            camera_prim_paths=cam_paths,
            view_count=self._view.count,
            camera_path_relative_to_env_0=rel_under_env0,
        )
        self._render_data = self._renderer.create_render_data(render_spec)

        # Create internal buffers (includes intrinsic matrix and pose init)
        self._create_buffers()

    def _update_buffers_impl(self, env_mask: wp.array):
        # Mask -> compact env_ids array via warp atomic scan (no torch involvement).
        env_ids_wp, count = self._mask_to_env_ids(env_mask)
        if count == 0:
            return
        # Increment frame count via masked warp kernel.
        wp.launch(
            camera_kernels.masked_increment_int64_kernel,
            dim=self._view.count,
            inputs=[env_mask, self._frame.warp],
            device=self._device,
        )
        # update latest camera pose if requested
        if self.cfg.update_latest_camera_pose:
            self._update_poses_wp(env_ids_wp)

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
        # reason: these use np structured data types which we can't yet convert to a warp buffer
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
                " can't be converted to a warp buffer easily."
                "\n\tHint: If you need to work with these sensor types, we recommend using their fast counterparts."
                f"\n\t\tFast counterparts: {fast_common_elements}"
            )

    def _create_buffers(self):
        """Allocate the camera-frame state buffers (pure warp).

        Each :class:`CameraData` field stores a ``wp.array`` primary buffer and is
        published as a :class:`ProxyArray`. Renderers receive the underlying
        :class:`warp.array` references via :meth:`set_outputs`.
        """
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
        device_str = str(self._device)
        self._data = CameraData.allocate(
            data_types=known,
            height=self.cfg.height,
            width=self.cfg.width,
            num_views=self._view.count,
            device=device_str,
            supported_specs=specs,
        )
        # Camera-frame state (pose / intrinsics) is owned by the camera, not
        # the renderer: populate it on the freshly constructed CameraData.
        # ``pos_w`` / ``quat_w_world`` are stored as plain float32 arrays with
        # explicit ``(N, 3)`` / ``(N, 4)`` layout so ``ProxyArray.shape``
        # matches the documented shape; warp kernels reinterpret them as
        # ``wp.vec3`` / ``wp.quatf`` views (zero-copy) when needed.
        n = self._view.count
        self._data.intrinsic_matrices = ProxyArray(wp.zeros((n, 3, 3), dtype=wp.float32, device=device_str))
        self._update_intrinsic_matrices(list(range(n)))
        self._data.pos_w = ProxyArray(wp.zeros((n, 3), dtype=wp.float32, device=device_str))
        self._data.quat_w_world = ProxyArray(wp.zeros((n, 4), dtype=wp.float32, device=device_str))
        self._update_poses_wp(self._ALL_INDICES)
        # Hand the renderer the underlying wp.array buffers; the ProxyArray
        # wrappers in CameraData stay valid since both views share memory.
        self._renderer.set_outputs(self._render_data, {name: proxy.warp for name, proxy in self._data.output.items()})

    def _update_intrinsic_matrices(self, env_ids: Sequence[int]):
        """Compute camera's matrix of intrinsic parameters via a warp scatter kernel.

        The per-camera scalars are read from the USD prims (CPU-bound) and
        scattered into the GPU buffer in a single launch.
        """
        if len(env_ids) == 0:
            return
        height, width = self.image_shape
        fx_host = np.empty(len(env_ids), dtype=np.float32)
        fy_host = np.empty(len(env_ids), dtype=np.float32)
        cx_host = np.empty(len(env_ids), dtype=np.float32)
        cy_host = np.empty(len(env_ids), dtype=np.float32)
        idx_host = np.asarray(list(env_ids), dtype=np.int32)
        for k, i in enumerate(env_ids):
            sensor_prim = self._sensor_prims[int(i)]
            # currently rendering does not use aperture offsets or vertical aperture
            focal_length = sensor_prim.GetFocalLengthAttr().Get()
            horiz_aperture = sensor_prim.GetHorizontalApertureAttr().Get()
            f_x = (width * focal_length) / horiz_aperture
            fx_host[k] = f_x
            fy_host[k] = f_x
            cx_host[k] = width * 0.5
            cy_host[k] = height * 0.5
        device_str = str(self._device)
        wp.launch(
            camera_kernels.write_intrinsic_matrices_kernel,
            dim=len(env_ids),
            inputs=[
                wp.array(idx_host, dtype=wp.int32, device=device_str),
                wp.array(fx_host, dtype=wp.float32, device=device_str),
                wp.array(fy_host, dtype=wp.float32, device=device_str),
                wp.array(cx_host, dtype=wp.float32, device=device_str),
                wp.array(cy_host, dtype=wp.float32, device=device_str),
                self._data.intrinsic_matrices.warp,
            ],
            device=device_str,
        )

    def _update_poses_wp(self, env_ids_wp: wp.array):
        """Pull camera poses from the view and write them into ``self._data`` (pure warp).

        Args:
            env_ids_wp: ``wp.int32`` array of camera indices to refresh, or
                :attr:`_ALL_INDICES` to refresh every camera.
        """
        if len(self._sensor_prims) == 0:
            raise RuntimeError("Camera prim is None. Please call 'sim.play()' first.")

        pos_w_proxy, quat_w_proxy = self._view.get_world_poses(env_ids_wp)
        # Convert quat_w (opengl) -> world via a warp kernel and scatter into CameraData.
        n_pose = int(env_ids_wp.shape[0])
        quat_w_world_local = convert_quat_array(quat_w_proxy.warp, origin="opengl", target="world")
        n_total = self._view.count
        # Reinterpret the (N, 3) / (N, 4) float32 storage as wp.vec3 / wp.quatf
        # views so the existing scatter kernels (which take typed inputs) can be
        # reused without copying.
        pos_dst = self._data.pos_w.warp
        quat_dst = self._data.quat_w_world.warp
        pos_dst_vec3 = wp.array(
            ptr=pos_dst.ptr,
            dtype=wp.vec3,
            shape=(n_total,),
            device=pos_dst.device,
            copy=False,
        )
        quat_dst_quatf = wp.array(
            ptr=quat_dst.ptr,
            dtype=wp.quatf,
            shape=(n_total,),
            device=quat_dst.device,
            copy=False,
        )
        wp.launch(
            camera_kernels.scatter_vec3f_kernel,
            dim=n_pose,
            inputs=[env_ids_wp, pos_w_proxy.warp, pos_dst_vec3],
            device=self._device,
        )
        wp.launch(
            camera_kernels.scatter_quatf_kernel,
            dim=n_pose,
            inputs=[env_ids_wp, quat_w_world_local, quat_dst_quatf],
            device=self._device,
        )
        # notify renderer of updated poses (guarded in case called before initialization completes)
        if self._render_data is not None:
            self._renderer.update_camera(
                self._render_data,
                self._data.pos_w.warp,
                self._data.quat_w_world.warp,
                self._data.intrinsic_matrices.warp,
            )

    # ---- env id / mask helpers ------------------------------------------------------------------

    def _resolve_env_indices(self, env_ids: Sequence[int] | wp.array | ProxyArray | None) -> wp.array | None:
        """Resolve user-supplied env_ids to a contiguous ``wp.int32`` array (or ``None``).

        ``None`` is passed through so callers can short-circuit to the "all" branch.
        """
        if env_ids is None:
            return None
        if isinstance(env_ids, ProxyArray):
            env_ids = env_ids.warp
        if isinstance(env_ids, wp.array):
            if env_ids.dtype is wp.int32:
                return env_ids
            return wp.array(env_ids.numpy().astype(np.int32, copy=False), dtype=wp.int32, device=self._device)
        # Sequence of ints / numpy-array.
        return wp.array(np.asarray(list(env_ids), dtype=np.int32), dtype=wp.int32, device=self._device)

    def _resolve_env_ids_or_mask(
        self,
        env_ids: Sequence[int] | wp.array | ProxyArray | None,
        env_mask: wp.array | None,
    ) -> tuple[wp.array, wp.array]:
        """Return both an int32 indices array and a bool mask covering the same envs."""
        device_str = str(self._device)
        if env_mask is not None and env_ids is None:
            ids, _ = self._mask_to_env_ids(env_mask)
            return ids, env_mask
        if env_ids is None:
            mask = wp.array(np.ones(self._view.count, dtype=np.bool_), dtype=wp.bool, device=device_str)
            return self._ALL_INDICES, mask
        ids = self._resolve_env_indices(env_ids)
        mask = wp.zeros(self._view.count, dtype=wp.bool, device=device_str)
        wp.launch(
            camera_kernels.indices_to_mask_kernel,
            dim=int(ids.shape[0]),
            inputs=[ids, mask],
            device=device_str,
        )
        return ids, mask

    def _mask_to_env_ids(self, env_mask: wp.array) -> tuple[wp.array, int]:
        """Compact a ``wp.bool`` mask into a contiguous ``wp.int32`` indices array.

        Returns ``(indices, count)`` where ``indices`` is a freshly-allocated
        slice over the populated prefix of the per-step scratch buffer.
        """
        device_str = str(self._device)
        # Reset counter; reuse the persistent scratch buffer.
        self._env_indices_count.zero_()
        wp.launch(
            camera_kernels.mask_to_indices_kernel,
            dim=int(env_mask.shape[0]),
            inputs=[env_mask, self._env_indices_scratch, self._env_indices_count],
            device=device_str,
        )
        count = int(self._env_indices_count.numpy()[0])
        if count == 0:
            return wp.empty(0, dtype=wp.int32, device=device_str), 0
        # Sub-view over the populated prefix; non-owning so the caller mustn't
        # mutate the scratch buffer between calls.
        sub = wp.array(
            ptr=self._env_indices_scratch.ptr,
            dtype=wp.int32,
            shape=(count,),
            device=self._env_indices_scratch.device,
            copy=False,
        )
        return sub, count

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
