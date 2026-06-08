# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
import warp as wp

from pxr import UsdGeom

import isaaclab.utils.math as math_utils
from isaaclab.sensors.camera import CameraData
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.kernels import raycast_mesh_masked_kernel

from ..sensor_base import SensorBase
from . import kernels as ray_caster_kernels
from .base_ray_caster import BaseRayCaster

if TYPE_CHECKING:
    from .ray_caster_camera_cfg import RayCasterCameraCfg

logger = logging.getLogger(__name__)


class BaseRayCasterCamera(BaseRayCaster):
    """A ray-casting camera sensor.

    The ray-caster camera uses a set of rays to get the distances to meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor has the same interface as the
    :class:`isaaclab.sensors.Camera` that implements the camera class through USD camera prims.
    However, this class provides a faster image generation. The sensor converts meshes from the list of
    primitive paths provided in the configuration to Warp meshes. The camera then ray-casts against these
    Warp meshes only.

    Currently, only the following annotators are supported:

    - ``"distance_to_camera"``: An image containing the distance to camera optical center.
    - ``"distance_to_image_plane"``: An image containing distances of 3D points from camera plane along camera's z-axis.
    - ``"normals"``: An image containing the local surface normal vectors at each pixel.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: RayCasterCameraCfg
    """The configuration parameters."""
    UNSUPPORTED_TYPES: ClassVar[set[str]] = {
        "rgb",
        "instance_id_segmentation",
        "instance_id_segmentation_fast",
        "instance_segmentation",
        "instance_segmentation_fast",
        "semantic_segmentation",
        "skeleton_data",
        "motion_vectors",
        "bounding_box_2d_tight",
        "bounding_box_2d_tight_fast",
        "bounding_box_2d_loose",
        "bounding_box_2d_loose_fast",
        "bounding_box_3d",
        "bounding_box_3d_fast",
    }
    """A set of sensor types that are not supported by the ray-caster camera."""

    def __init__(self, cfg: RayCasterCameraCfg):
        """Initializes the camera object.

        Args:
            cfg: The configuration parameters.

        Raises:
            ValueError: If the provided data types are not supported by the ray-caster camera.
        """
        # perform check on supported data types
        self._check_supported_data_types(cfg)
        # initialize base class
        super().__init__(cfg)
        # create empty variables for storing output data
        self._data = CameraData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-Caster-Camera @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(BaseRayCaster.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\timage shape          : {self.image_shape}"
        )

    @property
    def data(self) -> CameraData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def image_shape(self) -> tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width)

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame

    def set_intrinsic_matrices(
        self, matrices: torch.Tensor, focal_length: float = 1.0, env_ids: Sequence[int] | None = None
    ):
        """Set the intrinsic matrix of the camera.

        Args:
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3).
            focal_length: Focal length to use when computing aperture values (in cm). Defaults to 1.0.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = slice(None)
        # save new intrinsic matrices and focal length
        self._data.intrinsic_matrices.torch[env_ids] = matrices.to(self._device)
        self._focal_length = focal_length
        # recompute ray directions
        ray_starts_torch = self.ray_starts.torch if hasattr(self.ray_starts, "torch") else self.ray_starts
        ray_directions_torch = (
            self.ray_directions.torch if hasattr(self.ray_directions, "torch") else self.ray_directions
        )
        ray_starts_torch[env_ids], ray_directions_torch[env_ids] = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices.torch[env_ids], self._device
        )
        # Refresh warp views of local ray buffers; .contiguous() may produce a copy so we store
        # the contiguous tensors explicitly to prevent GC while the warp views are alive.
        if hasattr(self, "_ray_starts_local"):
            self._ray_starts_contiguous = ray_starts_torch.contiguous()
            self._ray_directions_contiguous = ray_directions_torch.contiguous()
            self._ray_starts_local = wp.from_torch(self._ray_starts_contiguous, dtype=wp.vec3f)
            self._ray_directions_local = wp.from_torch(self._ray_directions_contiguous, dtype=wp.vec3f)

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        # reset the timestamps
        SensorBase.reset(self, env_mask=env_mask)
        # reset the data through the same Warp path used by updates.
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        self._update_ray_infos(env_mask)
        self._update_frame(env_mask, frame_op=2)

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        env_ids: Sequence[int] | None = None,
        convention: Literal["opengl", "ros", "world"] = "ros",
    ):
        """Set the pose of the camera w.r.t. the world frame using specified convention.

        Since different fields use different conventions for camera orientations, the method allows users to
        set the camera poses in the specified convention. Possible conventions are:

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        See :meth:`isaaclab.utils.math.convert_camera_frame_orientation_convention` for more details
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
        # resolve env_ids for compact source arrays and env mask for output refresh
        if env_ids is None or isinstance(env_ids, slice):
            env_ids_wp = self._empty_env_ids_wp
            env_mask = self._ALL_ENV_MASK
            count = self._num_envs
            use_env_ids = False
        else:
            self._target_env_ids_torch = torch.as_tensor(env_ids, dtype=torch.int32, device=self._device).reshape(-1)
            env_ids_wp = wp.from_torch(self._target_env_ids_torch, dtype=wp.int32)
            self._reset_mask.zero_()
            self._reset_mask_torch[self._target_env_ids_torch.to(dtype=torch.long)] = True
            env_mask = self._reset_mask
            count = self._target_env_ids_torch.numel()
            use_env_ids = True

        target_positions_wp = self._offset_pos_wp
        if positions is not None:
            positions = torch.as_tensor(positions, dtype=torch.float32, device=self._device).reshape(-1, 3)
            if positions.shape[0] == 1 and count != 1:
                positions = positions.expand(count, -1)
            elif positions.shape[0] != count:
                raise ValueError(f"Expected {count} camera positions, got {positions.shape[0]}.")
            self._target_positions_torch = positions.contiguous()
            target_positions_wp = wp.from_torch(self._target_positions_torch, dtype=wp.vec3f)

        target_quats_wp = self._offset_quat_wp
        if orientations is not None:
            # convert rotation matrix from input convention to world
            quat_w_set = math_utils.convert_camera_frame_orientation_convention(
                torch.as_tensor(orientations, dtype=torch.float32, device=self._device),
                origin=convention,
                target="world",
            )
            quat_w_set = quat_w_set.reshape(-1, 4)
            if quat_w_set.shape[0] == 1 and count != 1:
                quat_w_set = quat_w_set.expand(count, -1)
            elif quat_w_set.shape[0] != count:
                raise ValueError(f"Expected {count} camera orientations, got {quat_w_set.shape[0]}.")
            self._target_quats_torch = quat_w_set.contiguous()
            target_quats_wp = wp.from_torch(self._target_quats_torch, dtype=wp.quatf)

        wp.launch(
            ray_caster_kernels.update_camera_offsets_kernel,
            dim=count,
            inputs=[
                self._get_view_transforms_wp(),
                env_ids_wp,
                target_positions_wp,
                target_quats_wp,
                use_env_ids,
                positions is not None,
                orientations is not None,
                self._offset_pos_wp,
                self._offset_quat_wp,
            ],
            device=self._device,
        )

        # update the data through the same Warp path used by normal sensor updates
        self._update_ray_infos(env_mask)

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
        # get up axis of current stage
        up_axis = UsdGeom.GetStageUpAxis(self.stage)
        # camera position and rotation in opengl convention
        orientations = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis=up_axis, device=self._device)
        )
        self.set_world_poses(eyes, orientations, env_ids, convention="opengl")

    def _create_offset_buffers(self, count: int):
        """Create Warp-owned camera offset and drift buffers."""
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device), origin=self.cfg.offset.convention, target="world"
        )
        offset_pos = [tuple(float(v) for v in self.cfg.offset.pos)] * count
        offset_quat = [tuple(float(v) for v in quat_w.squeeze(0).tolist())] * count
        self._offset_pos_wp = wp.array(offset_pos, dtype=wp.vec3f, device=self._device)
        self._offset_quat_wp = wp.array(offset_quat, dtype=wp.quatf, device=self._device)
        self.drift = ProxyArray(wp.zeros(count, dtype=wp.vec3f, device=self._device))
        self.ray_cast_drift = ProxyArray(wp.zeros(count, dtype=wp.vec3f, device=self._device))
        self._empty_env_ids_wp = wp.empty(0, dtype=wp.int32, device=self._device)

    def _initialize_rays_impl(self):
        # Frame count is updated by Warp kernels; expose a torch view for the public API.
        self._frame_wp = wp.zeros(self._view.count, dtype=wp.int64, device=self._device)
        self._frame = wp.to_torch(self._frame_wp)
        # create buffers
        self._create_buffers()
        # compute intrinsic matrices
        self._compute_intrinsic_matrices()
        # compute ray starts and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices.torch, self._device
        )
        self.num_rays = self.ray_directions.shape[1]

        # Offset/drift buffers are warp-primary so kernels always see current values without re-wrapping.
        self._create_offset_buffers(self._view.count)

        # Warp buffers for world-frame rays (used by update kernel)
        self._ray_starts_w = wp.empty((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)
        self._ray_directions_w = wp.empty((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)

        # Warp views for ray_starts and ray_directions (from torch tensors returned by pattern_cfg.func)
        # These are (num_envs, num_rays, 3) torch tensors; wrap as warp vec3f arrays.
        # Store contiguous tensors explicitly so they are not garbage-collected while the
        # warp views are alive (mirrors the pattern in RayCaster._initialize_impl).
        self._ray_starts_contiguous = self.ray_starts.contiguous()
        self._ray_directions_contiguous = self.ray_directions.contiguous()
        self._ray_starts_local = wp.from_torch(self._ray_starts_contiguous, dtype=wp.vec3f)
        self._ray_directions_local = wp.from_torch(self._ray_directions_contiguous, dtype=wp.vec3f)

        # Intermediate warp buffers for ray results (filled with inf before each raycasting step)
        self._ray_distance_wp = wp.empty((self._view.count, self.num_rays), dtype=wp.float32, device=self._device)
        if "normals" in self.cfg.data_types:
            self._ray_normal_w = wp.empty((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)
        else:
            self._ray_normal_w = wp.empty((1, 1), dtype=wp.vec3f, device=self._device)

        # Ray hit buffer used by raycasting and debug visualization.
        self.ray_hits_w = ProxyArray(wp.empty((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device))

    def _update_ray_infos(self, env_mask: wp.array):
        """Updates camera poses and world-frame ray buffers via a single warp kernel."""
        transforms = self._get_view_transforms_wp()
        wp.launch(
            ray_caster_kernels.update_ray_caster_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                transforms,
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self.drift.warp,
                self.ray_cast_drift.warp,
                self._ray_starts_local,
                self._ray_directions_local,
                int(ray_caster_kernels.ALIGNMENT_BASE),
            ],
            outputs=[
                self._data.pos_w.warp,
                self._data.quat_w_world.warp,
                self._ray_starts_w,
                self._ray_directions_w,
            ],
            device=self._device,
        )

    def _update_frame(self, env_mask: wp.array, frame_op: int):
        """Update frame counters for masked environments."""
        wp.launch(
            ray_caster_kernels.update_frame_masked_kernel,
            dim=self._num_envs,
            inputs=[env_mask, frame_op, self._frame_wp],
            device=self._device,
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        # increment frame count
        self._update_frame(env_mask, frame_op=1)

        self._update_ray_infos(env_mask)

        # Determine whether to compute normals.
        need_normal = int("normals" in self.cfg.data_types)

        # Fill ray hit, distance, and optional normal buffers with inf before raycasting.
        wp.launch(
            ray_caster_kernels.fill_ray_hits_distance_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, bool(need_normal)],
            outputs=[self.ray_hits_w.warp, self._ray_distance_wp, self._ray_normal_w],
            device=self._device,
        )

        # Ray-cast against the mesh; use a large upper-bound max_dist so depth clipping
        # can be applied per-data-type afterwards (matching the original behaviour).
        wp.launch(
            raycast_mesh_masked_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                BaseRayCaster.meshes[(self.cfg.mesh_prim_paths[0], self._device)].id,
                env_mask,
                self._ray_starts_w,
                self._ray_directions_w,
                float(ray_caster_kernels.CAMERA_RAYCAST_MAX_DIST),
                int(True),  # return_distance: always needed for depth output
                need_normal,
                self.ray_hits_w.warp,
                self._ray_distance_wp,
                self._ray_normal_w,
            ],
            device=self._device,
        )

        if "distance_to_image_plane" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.compute_distance_to_image_plane_to_image_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._data.quat_w_world.warp,
                    self._ray_distance_wp,
                    self._ray_directions_w,
                    int(self.image_shape[1]),
                    bool(self._depth_clip_enabled),
                    float(self.cfg.max_distance),
                    self._depth_clip_fill_value,
                ],
                outputs=[
                    self._data.output["distance_to_image_plane"].warp,
                ],
                device=self._device,
            )

        if "distance_to_camera" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.copy_float2d_to_image1_depth_clipped_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._ray_distance_wp,
                    int(self.image_shape[1]),
                    bool(self._depth_clip_enabled),
                    float(self.cfg.max_distance),
                    self._depth_clip_fill_value,
                ],
                outputs=[
                    self._data.output["distance_to_camera"].warp,
                ],
                device=self._device,
            )

        if "normals" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.copy_vec3_2d_to_image3_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, self._ray_normal_w, int(self.image_shape[1]), self._data.output["normals"].warp],
                device=self._device,
            )

    def _debug_vis_callback(self, event):
        # Debug visualization can be toggled before ray buffers are initialized.
        if not hasattr(self, "ray_hits_w"):
            return
        # filter out missed rays (inf values) before visualizing
        ray_hits_flat = self.ray_hits_w.torch.reshape(-1, 3)
        valid_mask = ~torch.isinf(ray_hits_flat).any(dim=-1)
        viz_points = ray_hits_flat[valid_mask]
        # if no valid hits, skip
        if viz_points.shape[0] == 0:
            return
        self.ray_visualizer.visualize(viz_points)

    def _check_supported_data_types(self, cfg: RayCasterCameraCfg):
        """Checks if the data types are supported by the ray-caster camera."""
        # check if there is any intersection in unsupported types
        # reason: we cannot obtain this data from simplified warp-based ray caster
        common_elements = set(cfg.data_types) & BaseRayCasterCamera.UNSUPPORTED_TYPES
        if common_elements:
            raise ValueError(
                f"RayCasterCamera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types cannot be obtained in a fast way using ''warp''."
                "\n\tHint: If you need to work with these sensor types, we recommend using the USD camera"
                " interface from the isaaclab.sensors.camera module."
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        self._depth_clip_enabled = True
        if self.cfg.depth_clipping_behavior == "none":
            self._depth_clip_enabled = False
            self._depth_clip_fill_value = 0.0
        elif self.cfg.depth_clipping_behavior == "max":
            self._depth_clip_fill_value = float(self.cfg.max_distance)
        elif self.cfg.depth_clipping_behavior == "zero":
            self._depth_clip_fill_value = 0.0
        else:
            raise ValueError(
                f"Unknown depth_clipping_behavior: {self.cfg.depth_clipping_behavior!r}."
                " Valid values are 'max', 'zero', and 'none'."
            )
        # create the data object
        # -- pose of the cameras
        self._data.create_buffers(self._view.count, self._device)
        # -- intrinsic matrix
        self._data.intrinsic_matrices.torch[:, 2, 2] = 1.0
        self._data.image_shape = self.image_shape
        # -- output data
        # create the buffers to store the annotator data.
        self._data._output = {}
        self._data.info = {name: None for name in self.cfg.data_types}
        for name in self.cfg.data_types:
            if name in ["distance_to_image_plane", "distance_to_camera"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width, 1)
            elif name in ["normals"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width, 3)
            else:
                raise ValueError(f"Received unknown data type: {name}. Please check the configuration.")
            # allocate tensor to store the data
            self._data.output[name] = ProxyArray(
                wp.zeros((self._view.count, *shape), dtype=wp.float32, device=self._device)
            )

    def _compute_intrinsic_matrices(self):
        """Computes the intrinsic matrices for the camera based on the config provided."""
        # get the sensor properties
        pattern_cfg = self.cfg.pattern_cfg

        # check if vertical aperture is provided
        # if not then it is auto-computed based on the aspect ratio to preserve squared pixels
        if pattern_cfg.vertical_aperture is None:
            pattern_cfg.vertical_aperture = pattern_cfg.horizontal_aperture * pattern_cfg.height / pattern_cfg.width

        # compute the intrinsic matrix
        f_x = pattern_cfg.width * pattern_cfg.focal_length / pattern_cfg.horizontal_aperture
        f_y = pattern_cfg.height * pattern_cfg.focal_length / pattern_cfg.vertical_aperture
        c_x = pattern_cfg.horizontal_aperture_offset * f_x + pattern_cfg.width / 2
        c_y = pattern_cfg.vertical_aperture_offset * f_y + pattern_cfg.height / 2
        # allocate the intrinsic matrices
        self._data.intrinsic_matrices.torch[:, 0, 0] = f_x
        self._data.intrinsic_matrices.torch[:, 0, 2] = c_x
        self._data.intrinsic_matrices.torch[:, 1, 1] = f_y
        self._data.intrinsic_matrices.torch[:, 1, 2] = c_y

        # save focal length
        self._focal_length = pattern_cfg.focal_length

    def _compute_view_world_poses(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtains the pose of the view the camera is attached to in the world frame.

        .. deprecated v2.3.1:
            This function will be removed in a future release. Call
            ``self._view.get_world_poses(indices)`` directly instead. The returned
            ProxyArray pair exposes ``.warp`` and ``.torch`` accessors.

        Returns:
            A tuple of the position (in meters) and quaternion (x, y, z, w).


        """
        logger.warning(
            "The function '_compute_view_world_poses' is deprecated."
            " Call 'self._view.get_world_poses(indices)' directly instead."
        )

        indices = wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32) if env_ids is not None else None
        pos_w, quat_w = self._view.get_world_poses(indices)
        return pos_w.torch.clone(), quat_w.torch.clone()

    def _compute_camera_world_poses(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the camera in the world frame.

        This function applies the offset pose to the pose of the view the camera is attached to.

        .. deprecated v2.3.1:
            This function will be removed in a future release. Instead, use the code block below:

            .. code-block:: python

                indices = wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32)
                pos_w, quat_w = self._view.get_world_poses(indices)
                # The returned ProxyArray pair exposes .warp and .torch accessors
                pos_w, quat_w = pos_w.torch.clone(), quat_w.torch.clone()
                pos_w, quat_w = math_utils.combine_frame_transforms(
                    pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
                )

        Returns:
            A tuple of the position (in meters) and quaternion (x, y, z, w) in "world" convention.
        """
        logger.warning(
            "The function '_compute_camera_world_poses' is deprecated."
            " Call 'self._view.get_world_poses(indices)' and 'math_utils.combine_frame_transforms' directly instead."
        )

        indices = wp.from_torch(env_ids.to(dtype=torch.int32), dtype=wp.int32) if env_ids is not None else None
        pos_w, quat_w = self._view.get_world_poses(indices)
        offset_pos = wp.to_torch(self._offset_pos_wp)
        offset_quat = wp.to_torch(self._offset_quat_wp)
        return math_utils.combine_frame_transforms(
            pos_w.torch.clone(), quat_w.torch.clone(), offset_pos[env_ids], offset_quat[env_ids]
        )
