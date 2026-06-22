# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from pxr import UsdPhysics

from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.frame_transformer import BaseFrameTransformer
from isaaclab.sim.utils.queries import resolve_matching_prims_from_source
from isaaclab.utils.math import is_identity_pose, normalize, quat_from_angle_axis

import isaaclab_ovphysx.tensor_types as TT
from isaaclab_ovphysx.physics import OvPhysxManager as SimulationManager

from .frame_transformer_data import FrameTransformerData
from .kernels import frame_transformer_update_kernel, gather_body_pose_kernel

if TYPE_CHECKING:
    from isaaclab.sensors.frame_transformer import FrameTransformerCfg

logger = logging.getLogger(__name__)


class FrameTransformer(BaseFrameTransformer):
    """An OVPhysX sensor for reporting frame transforms.

    Reports the world-frame transform of one or more target frames relative to a source frame.
    Both the source frame (:attr:`FrameTransformerCfg.prim_path`) and target frames
    (:attr:`FrameTransformerCfg.target_frames`) must attach to rigid bodies — either
    articulation links or standalone rigid bodies. The two cases are handled uniformly
    via ``TT.RIGID_BODY_POSE`` tensor bindings.

    Per-frame offsets (position + quaternion) are applied to the source and to each target.
    The relative transforms are computed on GPU by the same warp kernel the PhysX backend uses.
    """

    cfg: FrameTransformerCfg
    """The configuration parameters."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the frame transformer sensor."""

    def __init__(self, cfg: FrameTransformerCfg):
        """Initializes the frame transformer object.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)
        self._data: FrameTransformerData = FrameTransformerData()
        self._physx_instance: Any = None
        self._body_bindings: list[Any] = []
        self._body_read_bufs: list[wp.array] = []
        self._body_dst_flat_indices: list[wp.array] = []
        self._raw_transforms: wp.array | None = None
        self._source_raw_indices: wp.array | None = None
        self._target_raw_indices: wp.array | None = None
        self._source_offset_pos_wp: wp.array | None = None
        self._source_offset_quat_wp: wp.array | None = None
        self._target_offset_pos_wp: wp.array | None = None
        self._target_offset_quat_wp: wp.array | None = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"FrameTransformer @ '{self.cfg.prim_path}': \n"
            f"\ttracked body frames: {[self._source_frame_body_name] + self._target_frame_body_names} \n"
            f"\tnumber of envs: {self._num_envs}\n"
            f"\tsource body frame: {self._source_frame_body_name}\n"
            f"\ttarget frames (count: {len(self._target_frame_names)}): {self._target_frame_names}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> FrameTransformerData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_bodies(self) -> int:
        """Returns the number of target body frames being tracked."""
        warnings.warn(
            "The `num_bodies` property will be deprecated in a future release."
            " Please use `len(data.target_frame_names)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return len(self._target_frame_body_names)

    @property
    def body_names(self) -> list[str]:
        """Returns the names of the target body frames being tracked."""
        warnings.warn(
            "The `body_names` property will be deprecated in a future release."
            " Please use `data.target_frame_names` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._target_frame_body_names

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)

    """
    Implementation.
    """

    def _initialize_impl(self) -> None:
        super()._initialize_impl()

        # resolve source frame offset
        source_frame_offset_pos = torch.tensor(self.cfg.source_frame_offset.pos, device=self.device)
        source_frame_offset_quat = torch.tensor(self.cfg.source_frame_offset.rot, device=self.device)
        # Only need to perform offsetting of source frame if the position offsets is non-zero and rotation offset is
        # not the identity quaternion for efficiency in _update_buffer_impl
        self._apply_source_frame_offset = True
        # Handle source frame offsets
        if is_identity_pose(source_frame_offset_pos, source_frame_offset_quat):
            logger.debug(f"No offset application needed for source frame as it is identity: {self.cfg.prim_path}")
            self._apply_source_frame_offset = False
        else:
            logger.debug(f"Applying offset to source frame as it is not identity: {self.cfg.prim_path}")
            # Store offsets as tensors (duplicating each env's offsets for ease of multiplication later)
            self._source_frame_offset_pos = source_frame_offset_pos.unsqueeze(0).repeat(self._num_envs, 1)
            self._source_frame_offset_quat = source_frame_offset_quat.unsqueeze(0).repeat(self._num_envs, 1)

        # Keep track of mapping from the rigid body name to the desired frames and prim path,
        # as there may be multiple frames based upon the same body name and we don't want to
        # create unnecessary views.
        body_names_to_frames: dict[str, dict[str, set[str] | str]] = {}
        # The offsets associated with each target frame
        target_offsets: dict[str, dict[str, torch.Tensor]] = {}
        # The frames whose offsets are not identity (use set to avoid duplicates across envs)
        non_identity_offset_frames: set[str] = set()

        # Only need to perform offsetting of target frame if any of the position offsets are non-zero or any of the
        # rotation offsets are not the identity quaternion for efficiency in _update_buffer_impl
        self._apply_target_frame_offset = False

        # Need to keep track of whether the source frame is also a target frame
        self._source_is_also_target_frame = False

        # Collect all target frames, their associated body prim paths and their offsets so that we can extract
        # the prim, check that it has the appropriate rigid body API in a single loop.
        # First element is None because user can't specify source frame name
        frames = [None] + [target_frame.name for target_frame in self.cfg.target_frames]
        frame_prim_paths = [self.cfg.prim_path] + [target_frame.prim_path for target_frame in self.cfg.target_frames]
        # First element is None because source frame offset is handled separately
        frame_offsets = [None] + [target_frame.offset for target_frame in self.cfg.target_frames]
        frame_types = ["source"] + ["target"] * len(self.cfg.target_frames)
        for frame, prim_path, offset, frame_type in zip(frames, frame_prim_paths, frame_offsets, frame_types):
            # Resolve source-side env prims and destination expressions. This keeps discovery plan-aware when
            # the active clone plan has physics clones without authored USD prims for every environment.
            def has_rigid_body_api(prim) -> bool:
                return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))

            matches = resolve_matching_prims_from_source(
                prim_path, predicate=has_rigid_body_api, raise_if_no_matches=False
            )
            if not matches:
                raise ValueError(
                    f"Failed to create frame transformer for frame '{frame}' with path '{prim_path}'."
                    " No matching rigid-body prims were found."
                )
            for prim, matching_prim_path in matches:
                # Get the name of the body: use relative prim path for unique identification
                body_name = self._get_relative_body_path(matching_prim_path)
                # Use leaf name of prim path if frame name isn't specified by user
                frame_name = frame if frame is not None else matching_prim_path.rsplit("/", 1)[-1]

                # Keep track of which frames are associated with which bodies
                if body_name in body_names_to_frames:
                    body_names_to_frames[body_name]["frames"].add(frame_name)

                    # This is a corner case where the source frame is also a target frame
                    if body_names_to_frames[body_name]["type"] == "source" and frame_type == "target":
                        self._source_is_also_target_frame = True

                else:
                    # Store the first matching prim path and the type of frame
                    body_names_to_frames[body_name] = {
                        "frames": {frame_name},
                        "prim_path": matching_prim_path,
                        "type": frame_type,
                    }

                if offset is not None:
                    offset_pos = torch.tensor(offset.pos, device=self.device)
                    offset_quat = torch.tensor(offset.rot, device=self.device)
                    # Check if we need to apply offsets (optimized code path in _update_buffer_impl)
                    if not is_identity_pose(offset_pos, offset_quat):
                        non_identity_offset_frames.add(frame_name)
                        self._apply_target_frame_offset = True
                    target_offsets[frame_name] = {"pos": offset_pos, "quat": offset_quat}

        if not self._apply_target_frame_offset:
            logger.info(
                f"No offsets application needed from '{self.cfg.prim_path}' to target frames as all"
                f" are identity: {frames[1:]}"
            )
        else:
            logger.info(
                f"Offsets application needed from '{self.cfg.prim_path}' to the following target frames:"
                f" {sorted(non_identity_offset_frames)}"
            )

        # The names of bodies that RigidPrim will be tracking to later extract transforms from
        tracked_prim_paths = [body_names_to_frames[body_name]["prim_path"] for body_name in body_names_to_frames.keys()]
        tracked_body_names = [body_name for body_name in body_names_to_frames.keys()]

        # --- OVPhysX: create one TT.RIGID_BODY_POSE binding per unique tracked body ---
        physx_instance = SimulationManager.get_physx_instance()
        if physx_instance is None:
            raise RuntimeError(
                "OvPhysxManager has not been initialized yet."
                " Reset the simulation context before adding the FrameTransformer."
            )
        self._physx_instance = physx_instance
        self._body_bindings = []
        self._body_read_bufs = []  # one (num_envs, 7) float32 buffer per body
        self._body_dst_flat_indices = []  # (num_envs,) int32 destination slots per body

        num_unique_bodies = len(tracked_body_names)

        for body_slot, tracked_path in enumerate(tracked_prim_paths):
            pattern = self._env_wildcardify(tracked_path)
            binding = physx_instance.create_tensor_binding(pattern=pattern, tensor_type=TT.RIGID_BODY_POSE)
            if binding.count == 0:
                raise RuntimeError(
                    f"FrameTransformer: TT.RIGID_BODY_POSE binding for pattern {pattern!r} matched zero bodies."
                    " Verify the prim has UsdPhysics.RigidBodyAPI."
                )

            if binding.count != self._num_envs:
                # OVPhysX's InteractiveScene defaults to clone_usd=True on develop, so this branch is
                # unexpected in current flows. Mirror ContactSensor's clone_usd=False fallback so the
                # sensor stays correct if a future scene runs with clone_usd=False.
                logger.warning(
                    "FrameTransformer: binding.count=%d for pattern %r differs from self._num_envs=%d;"
                    " overriding env count from binding (clone_usd=False scene).",
                    binding.count,
                    pattern,
                    self._num_envs,
                )
                self._num_envs = binding.count
                self._ALL_ENV_MASK = wp.ones((self._num_envs,), dtype=wp.bool, device=self._device)
                self._reset_mask = wp.zeros((self._num_envs,), dtype=wp.bool, device=self._device)
                self._reset_mask_torch = wp.to_torch(self._reset_mask)
                self._is_outdated = wp.ones(self._num_envs, dtype=wp.bool, device=self._device)
                self._timestamp = wp.zeros(self._num_envs, dtype=wp.float32, device=self._device)
                self._timestamp_last_update = wp.zeros_like(self._timestamp)

            read_buf = wp.zeros((self._num_envs, 7), dtype=wp.float32, device=self._device)
            dst_torch = torch.tensor(
                [env_id * num_unique_bodies + body_slot for env_id in range(self._num_envs)],
                dtype=torch.int32,
                device=self._device,
            )
            self._body_bindings.append(binding)
            self._body_read_bufs.append(read_buf)
            self._body_dst_flat_indices.append(wp.from_torch(dst_torch.contiguous(), dtype=wp.int32))

        # Flat raw transforms buffer with layout slot = env_id * num_unique_bodies + body_slot.
        # Same layout PhysX produces after its _per_env_indices reorder.
        self._raw_transforms = wp.zeros(self._num_envs * num_unique_bodies, dtype=wp.transformf, device=self._device)

        # OVPhysX chooses the flat layout itself; _per_env_indices is the identity permutation.
        self._per_env_indices = list(range(self._num_envs * num_unique_bodies))

        # tracked_prim_paths is already the env-0 representative list in insertion order.
        sorted_prim_paths = tracked_prim_paths

        # -- target frames: use relative prim path for unique identification
        self._target_frame_body_names = [self._get_relative_body_path(prim_path) for prim_path in sorted_prim_paths]

        # -- source frame: use relative prim path for unique identification
        self._source_frame_body_name = self._get_relative_body_path(self.cfg.prim_path)
        source_frame_index = self._target_frame_body_names.index(self._source_frame_body_name)

        # Only remove source frame from tracked bodies if it is not also a target frame
        if not self._source_is_also_target_frame:
            self._target_frame_body_names.remove(self._source_frame_body_name)

        # Determine indices into all tracked body frames for both source and target frames
        all_ids = torch.arange(self._num_envs * len(tracked_body_names))
        self._source_frame_body_ids = torch.arange(self._num_envs) * len(tracked_body_names) + source_frame_index

        # If source frame is also a target frame, then the target frame body ids are the same as
        # the source frame body ids
        if self._source_is_also_target_frame:
            self._target_frame_body_ids = all_ids
        else:
            self._target_frame_body_ids = all_ids[~torch.isin(all_ids, self._source_frame_body_ids)]

        # The name of each of the target frame(s) - either user specified or defaulted to the body name
        self._target_frame_names: list[str] = []
        # The position and rotation components of target frame offsets
        target_frame_offset_pos = []
        target_frame_offset_quat = []
        # Stores the indices of bodies that need to be duplicated. For instance, if body "LF_SHANK" is needed
        # for 2 frames, this list enables us to duplicate the body to both frames when doing the calculations
        # when updating sensor in _update_buffers_impl
        duplicate_frame_indices = []

        # Go through each body name and determine the number of duplicates we need for that frame
        # and extract the offsets. This is all done to handle the case where multiple frames
        # reference the same body, but have different names and/or offsets
        for i, body_name in enumerate(self._target_frame_body_names):
            for frame in body_names_to_frames[body_name]["frames"]:
                # Only need to handle target frames here as source frame is handled separately
                if frame in target_offsets:
                    target_frame_offset_pos.append(target_offsets[frame]["pos"])
                    target_frame_offset_quat.append(target_offsets[frame]["quat"])
                    self._target_frame_names.append(frame)
                    duplicate_frame_indices.append(i)

        # To handle multiple environments, need to expand so [0, 1, 1, 2] with 2 environments becomes
        # [0, 1, 1, 2, 3, 4, 4, 5]. Again, this is a optimization to make _update_buffer_impl more efficient
        duplicate_frame_indices = torch.tensor(duplicate_frame_indices, device=self.device)
        if self._source_is_also_target_frame:
            num_target_body_frames = len(tracked_body_names)
        else:
            num_target_body_frames = len(tracked_body_names) - 1

        self._duplicate_frame_indices = torch.cat(
            [duplicate_frame_indices + num_target_body_frames * env_num for env_num in range(self._num_envs)]
        )

        # Target frame offsets are only applied if at least one of the offsets are non-identity
        if self._apply_target_frame_offset:
            # Stack up all the frame offsets for shape (num_envs, num_frames, 3) and (num_envs, num_frames, 4)
            self._target_frame_offset_pos = torch.stack(target_frame_offset_pos).repeat(self._num_envs, 1)
            self._target_frame_offset_quat = torch.stack(target_frame_offset_quat).repeat(self._num_envs, 1)

        # Store number of target frames for kernel launch
        self._num_target_frames = len(self._target_frame_names)

        # --- Pre-compute warp index arrays for fused kernel ---
        # Source raw indices: (N,) — direct index into raw_transforms per env
        source_raw_list = []
        for e in range(self._num_envs):
            source_raw_list.append(self._per_env_indices[self._source_frame_body_ids[e].item()])
        self._source_raw_indices = wp.from_torch(
            torch.tensor(source_raw_list, dtype=torch.int32, device=self._device), dtype=wp.int32
        )

        # Target raw indices: (N, M) — direct index into raw_transforms per (env, frame)
        M = self._num_target_frames
        target_raw = torch.zeros((self._num_envs, M), dtype=torch.int32, device=self._device)
        for e in range(self._num_envs):
            for f in range(M):
                dup_idx = self._duplicate_frame_indices[e * M + f].item()
                body_idx = self._target_frame_body_ids[dup_idx].item()
                target_raw[e, f] = self._per_env_indices[body_idx]
        self._target_raw_indices = wp.from_torch(target_raw.contiguous(), dtype=wp.int32)

        # --- Pre-compute warp offset arrays (always created; identity when not configured) ---
        # Source offsets: (N,)
        if self._apply_source_frame_offset:
            self._source_offset_pos_wp = wp.from_torch(self._source_frame_offset_pos.contiguous(), dtype=wp.vec3f)
            self._source_offset_quat_wp = wp.from_torch(self._source_frame_offset_quat.contiguous(), dtype=wp.quatf)
        else:
            self._source_offset_pos_wp = wp.zeros(self._num_envs, dtype=wp.vec3f, device=self._device)
            self._source_offset_quat_wp = wp.zeros(self._num_envs, dtype=wp.quatf, device=self._device)
            # Identity quaternion: (0, 0, 0, 1)
            wp.to_torch(self._source_offset_quat_wp)[:, 3] = 1.0

        # Target offsets: (M,)
        if self._apply_target_frame_offset:
            # Only need per-frame offsets (not per-env*frame), take first M entries
            tgt_off_pos = torch.stack(target_frame_offset_pos)  # (M, 3)
            tgt_off_quat = torch.stack(target_frame_offset_quat)  # (M, 4)
            self._target_offset_pos_wp = wp.from_torch(tgt_off_pos.contiguous(), dtype=wp.vec3f)
            self._target_offset_quat_wp = wp.from_torch(tgt_off_quat.contiguous(), dtype=wp.quatf)
        else:
            self._target_offset_pos_wp = wp.zeros(M, dtype=wp.vec3f, device=self._device)
            self._target_offset_quat_wp = wp.zeros(M, dtype=wp.quatf, device=self._device)
            # Identity quaternion: (0, 0, 0, 1)
            wp.to_torch(self._target_offset_quat_wp)[:, 3] = 1.0

        # Create data buffers
        self._data.create_buffers(
            num_envs=self._num_envs,
            num_target_frames=self._num_target_frames,
            target_frame_names=self._target_frame_names,
            device=self._device,
        )

    def _update_buffers_impl(self, env_mask: wp.array | None = None) -> None:
        """Fills the buffers of the sensor data."""
        env_mask = self._resolve_indices_and_mask(None, env_mask)
        if (
            self._raw_transforms is None
            or self._source_raw_indices is None
            or self._target_raw_indices is None
            or self._source_offset_pos_wp is None
            or self._source_offset_quat_wp is None
            or self._target_offset_pos_wp is None
            or self._target_offset_quat_wp is None
        ):
            raise RuntimeError(
                f"FrameTransformer '{self.cfg.prim_path}': not initialized."
                " Access sensor data only after sim.reset() has been called."
            )

        # Step 1: refresh each per-body RIGID_BODY_POSE binding and gather rows into _raw_transforms.
        for binding, read_buf, dst_indices in zip(
            self._body_bindings, self._body_read_bufs, self._body_dst_flat_indices
        ):
            binding.read(read_buf)
            pose_buf_tf = read_buf.view(wp.transformf)  # (num_envs, 7) float32 -> (num_envs,) transformf
            wp.launch(
                gather_body_pose_kernel,
                dim=self._num_envs,
                inputs=[env_mask, pose_buf_tf, dst_indices, self._raw_transforms],
                device=self._device,
            )

        # Step 2: compute source/target world poses (with offsets) and target poses relative to source.
        wp.launch(
            frame_transformer_update_kernel,
            dim=(self._num_envs, self._num_target_frames),
            inputs=[
                env_mask,
                self._raw_transforms,
                self._source_raw_indices,
                self._target_raw_indices,
                self._source_offset_pos_wp,
                self._source_offset_quat_wp,
                self._target_offset_pos_wp,
                self._target_offset_quat_wp,
                self._data._source_pos_w,
                self._data._source_quat_w,
                self._data._target_pos_w,
                self._data._target_quat_w,
                self._data._target_pos_source,
                self._data._target_quat_source,
            ],
            device=self._device,
        )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "frame_visualizer"):
                self.frame_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)

            # set their visibility to true
            self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "frame_visualizer"):
                self.frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.is_initialized or self._raw_transforms is None or not hasattr(self, "frame_visualizer"):
            return

        # Convert warp -> torch at the boundary for visualization
        source_pos_w = wp.to_torch(self._data._source_pos_w)
        source_quat_w = wp.to_torch(self._data._source_quat_w)
        target_pos_w = wp.to_torch(self._data._target_pos_w)
        target_quat_w = wp.to_torch(self._data._target_quat_w)

        # Get the all frames pose
        frames_pos = torch.cat([source_pos_w, target_pos_w.view(-1, 3)], dim=0)
        frames_quat = torch.cat([source_quat_w, target_quat_w.view(-1, 4)], dim=0)

        # Get the all connecting lines between frames pose
        lines_pos, lines_quat, lines_length = self._get_connecting_lines(
            start_pos=source_pos_w.repeat_interleave(target_pos_w.size(1), dim=0),
            end_pos=target_pos_w.view(-1, 3),
        )

        # Initialize default (identity) scales and marker indices for all markers (frames + lines)
        marker_scales = torch.ones(frames_pos.size(0) + lines_pos.size(0), 3)
        marker_indices = torch.zeros(marker_scales.size(0))

        # Set the z-scale of line markers to represent their actual length
        marker_scales[-lines_length.size(0) :, -1] = lines_length

        # Assign marker config index 1 to line markers
        marker_indices[-lines_length.size(0) :] = 1

        # Update the frame and the connecting line visualizer
        self.frame_visualizer.visualize(
            translations=torch.cat((frames_pos, lines_pos), dim=0),
            orientations=torch.cat((frames_quat, lines_quat), dim=0),
            scales=marker_scales,
            marker_indices=marker_indices,
        )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event) -> None:
        """Drop OVPhysX handles and cached buffers when physics stops."""
        super()._invalidate_initialize_callback(event)
        self._body_read_bufs = []
        self._body_dst_flat_indices = []
        self._body_bindings = []
        self._physx_instance = None
        self._raw_transforms = None
        self._source_raw_indices = None
        self._target_raw_indices = None
        self._source_offset_pos_wp = None
        self._source_offset_quat_wp = None
        self._target_offset_pos_wp = None
        self._target_offset_quat_wp = None
        for attr_name in (
            "_target_pose_source_ta",
            "_target_pos_source_ta",
            "_target_quat_source_ta",
            "_target_pose_w_ta",
            "_target_pos_w_ta",
            "_target_quat_w_ta",
            "_source_pose_w_ta",
            "_source_pos_w_ta",
            "_source_quat_w_ta",
        ):
            if hasattr(self._data, attr_name):
                setattr(self._data, attr_name, None)

    """
    Internal helpers.
    """

    def _get_connecting_lines(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draws connecting lines between frames.

        Given start and end points, this function computes the positions (mid-point), orientations,
        and lengths of the connecting lines.

        Args:
            start_pos: The start positions of the connecting lines. Shape is (N, 3).
            end_pos: The end positions of the connecting lines. Shape is (N, 3).

        Returns:
            A tuple containing:
            - The positions of each connecting line. Shape is (N, 3).
            - The orientations of each connecting line in quaternion. Shape is (N, 4).
            - The lengths of each connecting line. Shape is (N,).
        """
        direction = end_pos - start_pos
        lengths = torch.linalg.norm(direction, dim=-1)
        positions = (start_pos + end_pos) / 2

        # Get default direction (along z-axis)
        default_direction = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(start_pos.size(0), -1)

        # Normalize direction vector
        direction_norm = normalize(direction)

        # Calculate rotation from default direction to target direction
        rotation_axis = torch.linalg.cross(default_direction, direction_norm)
        rotation_axis_norm = torch.linalg.norm(rotation_axis, dim=-1)

        # Handle case where vectors are parallel
        mask = rotation_axis_norm > 1e-6
        rotation_axis = torch.where(
            mask.unsqueeze(-1),
            normalize(rotation_axis),
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(start_pos.size(0), -1),
        )

        # Calculate rotation angle
        cos_angle = torch.sum(default_direction * direction_norm, dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        orientations = quat_from_angle_axis(angle, rotation_axis)

        return positions, orientations, lengths

    @staticmethod
    def _env_wildcardify(prim_path: str) -> str:
        """Convert an env-0 prim path into an ovphysx fnmatch glob matching all envs.

        Extends the two-substitution pattern used by
        :class:`~isaaclab_ovphysx.sensors.ContactSensor` with a third substitution
        for concrete ``env_<N>`` paths produced by ``sim_utils.find_matching_prims``.
        The three substitutions, in order:

        1. ``{ENV_REGEX_NS}`` → ``*`` (placeholder form).
        2. ``.*`` → ``*`` (IsaacLab regex form, e.g. ``env_.*``).
        3. ``/envs/env_<digits>`` → ``/envs/env_*`` (concrete env-0 path form).

        Args:
            prim_path: An env-0 prim path (e.g. ``"/World/envs/env_0/Robot/LF_FOOT"``) or an
                IsaacLab regex form (e.g. ``"/World/envs/env_.*/Robot/LF_FOOT"`` or
                ``"{ENV_REGEX_NS}/Robot/LF_FOOT"``).

        Returns:
            The same path with the env namespace replaced by an fnmatch wildcard
            (``*`` for the ``{ENV_REGEX_NS}`` placeholder, ``env_*`` for concrete or
            regex env paths). Assumes the standard IsaacLab ``/World/envs/env_<N>/...``
            layout; non-standard scene structures will only get the first two
            substitutions.
        """
        pattern = re.sub(r"\{ENV_REGEX_NS\}", "*", prim_path)
        pattern = re.sub(r"\.\*", "*", pattern)
        pattern = re.sub(r"/envs/env_\d+(/|$)", r"/envs/env_*\1", pattern)
        return pattern

    @staticmethod
    def _get_relative_body_path(prim_path: str) -> str:
        """Strip the ``/envs/env_<id>/`` prefix from a prim path so paths can be compared across environments.

        Args:
            prim_path: Absolute USD prim path that may contain an ``/envs/env_<digits>/`` segment.

        Returns:
            The prim path with that segment collapsed to ``/envs/``, so prim paths from any env compare equal.
        """
        pattern = re.compile(r"/envs/env_[^/]+/")
        return pattern.sub("/envs/", prim_path)
