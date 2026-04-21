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
from isaaclab.utils.warp.kernels import raycast_mesh_masked_kernel

from .kernels import (
    ALIGNMENT_BASE,
    CAMERA_RAYCAST_MAX_DIST,
    apply_depth_clipping_masked_kernel,
    compute_distance_to_image_plane_masked_kernel,
    fill_float2d_masked_kernel,
    fill_vec3_inf_kernel,
    update_ray_caster_kernel,
)
from .ray_cast_utils import obtain_world_pose_from_view
from .ray_caster import RayCaster

if TYPE_CHECKING:
    from .ray_caster_camera_cfg import RayCasterCameraCfg

# import logger
logger = logging.getLogger(__name__)


class RayCasterCamera(RayCaster):
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
            f"\tnumber of meshes     : {len(RayCaster.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}\n"
            f"\timage shape          : {self.image_shape}"
        )

    """
    Properties
    """

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

    """
    Operations.
    """

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
        self._data.intrinsic_matrices[env_ids] = matrices.to(self._device)
        self._focal_length = focal_length
        # recompute ray directions
        self.ray_starts[env_ids], self.ray_directions[env_ids] = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices[env_ids], self._device
        )
        # Refresh warp views of local ray buffers; .contiguous() may produce a copy so we store
        # the contiguous tensors explicitly to prevent GC while the warp views are alive.
        if hasattr(self, "_ray_starts_local"):
            self._ray_starts_contiguous = self.ray_starts.contiguous()
            self._ray_directions_contiguous = self.ray_directions.contiguous()
            self._ray_starts_local = wp.from_torch(self._ray_starts_contiguous, dtype=wp.vec3f)
            self._ray_directions_local = wp.from_torch(self._ray_directions_contiguous, dtype=wp.vec3f)

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        # reset the timestamps
        super().reset(env_ids, env_mask)
        # resolve to indices for torch indexing
        if env_ids is None and env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        elif env_ids is None or isinstance(env_ids, slice):
            env_ids = self._ALL_INDICES
        # reset the data
        # note: this recomputation is useful if one performs events such as randomizations on the camera poses.
        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids, clone=True)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w
        # Reset the frame count
        self._frame[env_ids] = 0

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
        # resolve env_ids
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = self._ALL_INDICES

        # get current positions
        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids)
        if positions is not None:
            # transform to camera frame
            pos_offset_world_frame = positions - pos_w
            self._offset_pos[env_ids] = math_utils.quat_apply(math_utils.quat_inv(quat_w), pos_offset_world_frame)
        if orientations is not None:
            # convert rotation matrix from input convention to world
            quat_w_set = math_utils.convert_camera_frame_orientation_convention(
                orientations, origin=convention, target="world"
            )
            self._offset_quat[env_ids] = math_utils.quat_mul(math_utils.quat_inv(quat_w), quat_w_set)

        # update the data
        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids, clone=True)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w_world[env_ids] = quat_w

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

    """
    Implementation.
    """

    def _initialize_rays_impl(self):
        # Create all indices buffer
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)
        # create buffers
        self._create_buffers()
        # compute intrinsic matrices
        self._compute_intrinsic_matrices()
        # compute ray starts and directions
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(
            self.cfg.pattern_cfg, self._data.intrinsic_matrices, self._device
        )
        self.num_rays = self.ray_directions.shape[1]

        # Offset buffers: warp-primary so the kernel always sees the current values without re-wrapping.
        # Zero-copy torch views (_offset_pos, _offset_quat) are used by set_world_poses for indexed writes.
        self._offset_pos_wp = wp.zeros(self._view.count, dtype=wp.vec3f, device=self._device)
        self._offset_quat_wp = wp.zeros(self._view.count, dtype=wp.quatf, device=self._device)
        self._offset_pos = wp.to_torch(self._offset_pos_wp)
        self._offset_quat = wp.to_torch(self._offset_quat_wp)
        # Initialize from config
        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device), origin=self.cfg.offset.convention, target="world"
        )
        self._offset_pos[:] = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        self._offset_quat[:] = quat_w

        # Warp buffers for world-frame rays (used by update kernel)
        self._ray_starts_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)
        self._ray_directions_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)

        # Warp views for ray_starts and ray_directions (from torch tensors returned by pattern_cfg.func)
        # These are (num_envs, num_rays, 3) torch tensors; wrap as warp vec3f arrays.
        # Store contiguous tensors explicitly so they are not garbage-collected while the
        # warp views are alive (mirrors the pattern in RayCaster._initialize_impl).
        self._ray_starts_contiguous = self.ray_starts.contiguous()
        self._ray_directions_contiguous = self.ray_directions.contiguous()
        self._ray_starts_local = wp.from_torch(self._ray_starts_contiguous, dtype=wp.vec3f)
        self._ray_directions_local = wp.from_torch(self._ray_directions_contiguous, dtype=wp.vec3f)

        # Wrap the torch drift buffers (created in _create_buffers) as warp arrays (zero-copy).
        # Cameras do not apply positional drift, so these remain zero.
        self._drift_contiguous = self.drift.contiguous()
        self._ray_cast_drift_contiguous = self.ray_cast_drift.contiguous()
        self._drift = wp.from_torch(self._drift_contiguous, dtype=wp.vec3f)
        self._ray_cast_drift = wp.from_torch(self._ray_cast_drift_contiguous, dtype=wp.vec3f)

        # Warp buffers for camera pose outputs
        self._pos_w_wp = wp.zeros(self._view.count, dtype=wp.vec3f, device=self._device)
        self._quat_w_wp = wp.zeros(self._view.count, dtype=wp.quatf, device=self._device)

        # Intermediate warp buffers for ray results (filled with inf before each raycasting step)
        self._ray_distance = wp.zeros((self._view.count, self.num_rays), dtype=wp.float32, device=self._device)
        if "normals" in self.cfg.data_types:
            self._ray_normal_w = wp.zeros((self._view.count, self.num_rays), dtype=wp.vec3f, device=self._device)
        else:
            self._ray_normal_w = wp.zeros((1, 1), dtype=wp.vec3f, device=self._device)

        if "distance_to_image_plane" in self.cfg.data_types:
            self._distance_to_image_plane_wp = wp.zeros(
                (self._view.count, self.num_rays), dtype=wp.float32, device=self._device
            )

        # Torch buffer for ray hits (used by debug visualizer)
        self.ray_hits_w = torch.full((self._view.count, self.num_rays, 3), float("inf"), device=self._device)
        # Warp view of ray_hits_w
        self._ray_hits_w_wp = wp.from_torch(self.ray_hits_w.contiguous(), dtype=wp.vec3f)

        # Cache zero-copy torch views of warp output buffers to avoid per-step wrapper allocation.
        self._pos_w_torch = wp.to_torch(self._pos_w_wp)
        self._quat_w_torch = wp.to_torch(self._quat_w_wp)
        self._ray_distance_torch = wp.to_torch(self._ray_distance)
        if "distance_to_image_plane" in self.cfg.data_types:
            self._distance_to_image_plane_torch = wp.to_torch(self._distance_to_image_plane_wp)
        if "normals" in self.cfg.data_types:
            self._ray_normal_w_torch = wp.to_torch(self._ray_normal_w)

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        # Convert mask to indices for torch-indexed writes
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return
        # increment frame count
        self._frame[env_ids] += 1

        # Update world-frame ray starts/directions and camera pose via warp kernel.
        # Camera always uses ALIGNMENT_BASE (full orientation) and zero drift.
        transforms = self._get_view_transforms_wp()
        wp.launch(
            update_ray_caster_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                transforms,
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self._drift,
                self._ray_cast_drift,
                self._ray_starts_local,
                self._ray_directions_local,
                int(ALIGNMENT_BASE),
            ],
            outputs=[
                self._pos_w_wp,
                self._quat_w_wp,
                self._ray_starts_w,
                self._ray_directions_w,
            ],
            device=self._device,
        )

        # Write camera pose to CameraData (torch tensors)
        self._data.pos_w[env_ids] = self._pos_w_torch[env_ids]
        self._data.quat_w_world[env_ids] = self._quat_w_torch[env_ids]

        # Fill ray hit positions with inf before raycasting
        wp.launch(
            fill_vec3_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, float("inf"), self._ray_hits_w_wp],
            device=self._device,
        )

        # Fill ray distance with inf before raycasting
        wp.launch(
            fill_float2d_masked_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, float("inf"), self._ray_distance],
            device=self._device,
        )

        # Determine whether to compute normals
        need_normal = int("normals" in self.cfg.data_types)
        if need_normal:
            # Fill normal buffer with inf before raycasting
            wp.launch(
                fill_vec3_inf_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, float("inf"), self._ray_normal_w],
                device=self._device,
            )

        # Ray-cast against the mesh; use a large upper-bound max_dist so depth clipping
        # can be applied per-data-type afterwards (matching the original behaviour).
        wp.launch(
            raycast_mesh_masked_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                RayCaster.meshes[(self.cfg.mesh_prim_paths[0], self._device)].id,
                env_mask,
                self._ray_starts_w,
                self._ray_directions_w,
                float(CAMERA_RAYCAST_MAX_DIST),
                int(True),  # return_distance: always needed for depth output
                need_normal,
                self._ray_hits_w_wp,
                self._ray_distance,
                self._ray_normal_w,
            ],
            device=self._device,
        )

        # Compute distance_to_image_plane using a warp kernel
        if "distance_to_image_plane" in self.cfg.data_types:
            wp.launch(
                compute_distance_to_image_plane_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._quat_w_wp,
                    self._ray_distance,
                    self._ray_directions_w,
                ],
                outputs=[
                    self._distance_to_image_plane_wp,
                ],
                device=self._device,
            )
            # Apply depth clipping on the intermediate buffer (leaves _ray_distance unmodified)
            self._apply_depth_clipping(env_mask, self._distance_to_image_plane_wp)
            self._data.output["distance_to_image_plane"][env_ids] = self._distance_to_image_plane_torch[env_ids].view(
                -1, *self.image_shape, 1
            )

        if "distance_to_camera" in self.cfg.data_types:
            # d2ip (if requested) was computed before this block so _ray_distance is still unclipped.
            self._apply_depth_clipping(env_mask, self._ray_distance)
            self._data.output["distance_to_camera"][env_ids] = self._ray_distance_torch[env_ids].view(
                -1, *self.image_shape, 1
            )

        if "normals" in self.cfg.data_types:
            self._data.output["normals"][env_ids] = self._ray_normal_w_torch[env_ids].view(-1, *self.image_shape, 3)

    def _debug_vis_callback(self, event):
        # in case it crashes be safe
        if not hasattr(self, "ray_hits_w"):
            return
        # filter out missed rays (inf values) before visualizing
        ray_hits_flat = self.ray_hits_w.reshape(-1, 3)
        valid_mask = ~torch.isinf(ray_hits_flat).any(dim=-1)
        viz_points = ray_hits_flat[valid_mask]
        # if no valid hits, skip
        if viz_points.shape[0] == 0:
            return
        self.ray_visualizer.visualize(viz_points)

    """
    Private Helpers
    """

    def _apply_depth_clipping(self, env_mask: wp.array, depth: wp.array) -> None:
        """Apply depth clipping in-place on a warp float32 buffer.

        Uses :attr:`cfg.depth_clipping_behavior` to determine the fill value:
        ``"max"`` replaces out-of-range and NaN values with :attr:`cfg.max_distance`;
        ``"zero"`` replaces them with 0. No-op when behavior is ``"none"``.

        Args:
            env_mask: Boolean mask selecting which environments to update. Shape is (num_envs,).
            depth: Warp 2-D float32 buffer to clip in-place. Shape is (num_envs, num_rays).
        """
        if self.cfg.depth_clipping_behavior == "max":
            wp.launch(
                apply_depth_clipping_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, float(self.cfg.max_distance), float(self.cfg.max_distance), depth],
                device=self._device,
            )
        elif self.cfg.depth_clipping_behavior == "zero":
            wp.launch(
                apply_depth_clipping_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, float(self.cfg.max_distance), float(0.0), depth],
                device=self._device,
            )
        elif self.cfg.depth_clipping_behavior == "none":
            pass  # no clipping: inf values remain as-is
        else:
            raise ValueError(
                f"Unknown depth_clipping_behavior: {self.cfg.depth_clipping_behavior!r}."
                " Valid values are 'max', 'zero', and 'none'."
            )

    def _check_supported_data_types(self, cfg: RayCasterCameraCfg):
        """Checks if the data types are supported by the ray-caster camera."""
        # check if there is any intersection in unsupported types
        # reason: we cannot obtain this data from simplified warp-based ray caster
        common_elements = set(cfg.data_types) & RayCasterCamera.UNSUPPORTED_TYPES
        if common_elements:
            raise ValueError(
                f"RayCasterCamera class does not support the following sensor types: {common_elements}."
                "\n\tThis is because these sensor types cannot be obtained in a fast way using ''warp''."
                "\n\tHint: If you need to work with these sensor types, we recommend using the USD camera"
                " interface from the isaaclab.sensors.camera module."
            )

    def _create_buffers(self):
        """Create buffers for storing data."""
        # prepare drift (kept as torch tensors so subclasses may use torch indexing)
        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)
        # create the data object
        # -- pose of the cameras
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self._device)
        # -- intrinsic matrix
        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self._device)
        self._data.intrinsic_matrices[:, 2, 2] = 1.0
        self._data.image_shape = self.image_shape
        # -- output data
        # create the buffers to store the annotator data.
        self._data.output = {}
        self._data.info = {name: None for name in self.cfg.data_types}
        for name in self.cfg.data_types:
            if name in ["distance_to_image_plane", "distance_to_camera"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width, 1)
            elif name in ["normals"]:
                shape = (self.cfg.pattern_cfg.height, self.cfg.pattern_cfg.width, 3)
            else:
                raise ValueError(f"Received unknown data type: {name}. Please check the configuration.")
            # allocate tensor to store the data
            self._data.output[name] = torch.zeros((self._view.count, *shape), device=self._device)

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
        self._data.intrinsic_matrices[:, 0, 0] = f_x
        self._data.intrinsic_matrices[:, 0, 2] = c_x
        self._data.intrinsic_matrices[:, 1, 1] = f_y
        self._data.intrinsic_matrices[:, 1, 2] = c_y

        # save focal length
        self._focal_length = pattern_cfg.focal_length

    def _compute_view_world_poses(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtains the pose of the view the camera is attached to in the world frame.

        .. deprecated v2.3.1:
            This function will be removed in a future release in favor of implementation
            :meth:`obtain_world_pose_from_view`.

        Returns:
            A tuple of the position (in meters) and quaternion (x, y, z, w).


        """
        # deprecation
        logger.warning(
            "The function '_compute_view_world_poses' will be deprecated in favor of the util method"
            " 'obtain_world_pose_from_view'. Please use 'obtain_world_pose_from_view' instead...."
        )

        return obtain_world_pose_from_view(self._view, env_ids, clone=True)

    def _compute_camera_world_poses(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the camera in the world frame.

        This function applies the offset pose to the pose of the view the camera is attached to.

        .. deprecated v2.3.1:
            This function will be removed in a future release. Instead, use the code block below:

            .. code-block:: python

                pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids, clone=True)
                pos_w, quat_w = math_utils.combine_frame_transforms(
                    pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
                )

        Returns:
            A tuple of the position (in meters) and quaternion (x, y, z, w) in "world" convention.
        """

        # deprecation
        logger.warning(
            "The function '_compute_camera_world_poses' will be deprecated in favor of the combination of methods"
            " 'obtain_world_pose_from_view' and 'math_utils.combine_frame_transforms'. Please use"
            " 'obtain_world_pose_from_view' and 'math_utils.combine_frame_transforms' instead...."
        )

        # get the pose of the view the camera is attached to
        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids, clone=True)
        return math_utils.combine_frame_transforms(pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids])
