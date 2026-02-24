# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.frame_transformer import BaseFrameTransformer
from isaaclab.utils.math import is_identity_pose, normalize, quat_from_angle_axis

from isaaclab_physx.physics import PhysxManager as SimulationManager

from .frame_transformer_data import FrameTransformerData
from .kernels import frame_transformer_update_kernel

if TYPE_CHECKING:
    from isaaclab.sensors.frame_transformer import FrameTransformerCfg

# import logger
logger = logging.getLogger(__name__)


class FrameTransformer(BaseFrameTransformer):
    """A PhysX sensor for reporting frame transforms.

    This class provides an interface for reporting the transform of one or more frames (target frames)
    with respect to another frame (source frame). The source frame is specified by the user as a prim path
    (:attr:`FrameTransformerCfg.prim_path`) and the target frames are specified by the user as a list of
    prim paths (:attr:`FrameTransformerCfg.target_frames`).

    The source frame and target frames are assumed to be rigid bodies. The transform of the target frames
    with respect to the source frame is computed by first extracting the transform of the source frame
    and target frames from the physics engine and then computing the relative transform between the two.

    Additionally, the user can specify an offset for the source frame and each target frame. This is useful
    for specifying the transform of the desired frame with respect to the body's center of mass, for instance.

    A common example of using this sensor is to track the position and orientation of the end effector of a
    robotic manipulator. In this case, the source frame would be the body corresponding to the base frame of the
    manipulator, and the target frame would be the body corresponding to the end effector. Since the end-effector is
    typically a fictitious body, the user may need to specify an offset from the end-effector to the body of the
    manipulator.

    """

    cfg: FrameTransformerCfg
    """The configuration parameters."""

    __backend_name__: str = "physx"
    """The name of the backend for the frame transformer sensor."""

    def __init__(self, cfg: FrameTransformerCfg):
        """Initializes the frame transformer object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data: FrameTransformerData = FrameTransformerData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"FrameTransformer @ '{self.cfg.prim_path}': \n"
            f"\ttracked body frames: {[self._source_frame_body_name] + self._target_frame_body_names} \n"
            f"\tnumber of envs: {self._num_envs}\n"
            f"\tsource body frame: {self._source_frame_body_name}\n"
            f"\ttarget frames (count: {self._target_frame_names}): {len(self._target_frame_names)}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> FrameTransformerData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_bodies(self) -> int:
        """Returns the number of target bodies being tracked.

        .. deprecated::
            Use ``len(data.target_frame_names)`` instead. This property will be removed in a future release.
        """
        warnings.warn(
            "The `num_bodies` property will be deprecated in a future release."
            " Please use `len(data.target_frame_names)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return len(self._target_frame_body_names)

    @property
    def body_names(self) -> list[str]:
        """Returns the names of the target bodies being tracked.

        .. deprecated::
            Use ``data.target_frame_names`` instead. This property will be removed in a future release.
        """
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

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        # resolve indices and mask
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        # reset the timers and counters
        super().reset(None, env_mask)

    """
    Implementation.
    """

    def _initialize_impl(self):
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
        # The frames whose offsets are not identity
        non_identity_offset_frames: list[str] = []

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
            # Find correct prim
            matching_prims = sim_utils.find_matching_prims(prim_path)
            if len(matching_prims) == 0:
                raise ValueError(
                    f"Failed to create frame transformer for frame '{frame}' with path '{prim_path}'."
                    " No matching prims were found."
                )
            for prim in matching_prims:
                # Get the prim path of the matching prim
                matching_prim_path = prim.GetPath().pathString
                # Check if it is a rigid prim
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(
                        f"While resolving expression '{prim_path}' found a prim '{matching_prim_path}' which is not a"
                        " rigid body. The class only supports transformations between rigid bodies."
                    )

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
                        non_identity_offset_frames.append(frame_name)
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
                f" {non_identity_offset_frames}"
            )

        # The names of bodies that RigidPrim will be tracking to later extract transforms from
        tracked_prim_paths = [body_names_to_frames[body_name]["prim_path"] for body_name in body_names_to_frames.keys()]
        tracked_body_names = [body_name for body_name in body_names_to_frames.keys()]

        body_names_regex = [tracked_prim_path.replace("env_0", "env_*") for tracked_prim_path in tracked_prim_paths]

        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # Create a prim view for all frames and initialize it
        # order of transforms coming out of view will be source frame followed by target frame(s)
        self._frame_physx_view = self._physics_sim_view.create_rigid_body_view(body_names_regex)

        # Determine the order in which regex evaluated body names so we can later index into frame transforms
        # by frame name correctly
        all_prim_paths = self._frame_physx_view.prim_paths

        if "env_" in all_prim_paths[0]:

            def extract_env_num_and_prim_path(item: str) -> tuple[int, str]:
                """Separates the environment number and prim_path from the item.

                Args:
                    item: The item to extract the environment number from. Assumes item is of the form
                        `/World/envs/env_1/blah` or `/World/envs/env_11/blah`.
                Returns:
                    The environment number and the prim_path.
                """
                match = re.search(r"env_(\d+)(.*)", item)
                return (int(match.group(1)), match.group(2))

            # Find the indices that would reorganize output to be per environment.
            # We want `env_1/blah` to come before `env_11/blah` and env_1/Robot/base
            # to come before env_1/Robot/foot so we need to use custom key function
            self._per_env_indices = [
                index
                for index, _ in sorted(
                    list(enumerate(all_prim_paths)), key=lambda x: extract_env_num_and_prim_path(x[1])
                )
            ]

            # Only need 0th env as the names and their ordering are the same across environments
            sorted_prim_paths = [
                all_prim_paths[index] for index in self._per_env_indices if "env_0" in all_prim_paths[index]
            ]

        else:
            # If no environment is present, then the order of the body names is the same as the order of the
            # prim paths sorted alphabetically
            self._per_env_indices = [index for index, _ in sorted(enumerate(all_prim_paths), key=lambda x: x[1])]
            sorted_prim_paths = [all_prim_paths[index] for index in self._per_env_indices]

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

    def _update_buffers_impl(self, env_mask: wp.array | None = None):
        """Fills the buffers of the sensor data."""
        # Resolve mask
        env_mask = self._resolve_indices_and_mask(None, env_mask)
        # Get raw transforms from PhysX view and reinterpret as transformf
        raw_transforms = self._frame_physx_view.get_transforms().view(wp.transformf)

        wp.launch(
            frame_transformer_update_kernel,
            dim=(self._num_envs, self._num_target_frames),
            inputs=[
                env_mask,
                raw_transforms,
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

    def _set_debug_vis_impl(self, debug_vis: bool):
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

    def _debug_vis_callback(self, event):
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

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._frame_physx_view = None

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
    def _get_relative_body_path(prim_path: str) -> str:
        """Extract a normalized body path from a prim path.

        Removes the environment instance segment `/envs/env_<id>/` to normalize paths
        across multiple environments, while preserving the `/envs/` prefix to
        distinguish environment-scoped paths from non-environment paths.

        Examples:
        - '/World/envs/env_0/Robot/torso' -> '/World/envs/Robot/torso'
        - '/World/envs/env_123/Robot/left_hand' -> '/World/envs/Robot/left_hand'
        - '/World/Robot' -> '/World/Robot'
        - '/World/Robot_2/left_hand' -> '/World/Robot_2/left_hand'

        Args:
            prim_path: The full prim path.

        Returns:
            The prim path with `/envs/env_<id>/` removed, preserving `/envs/`.
        """
        pattern = re.compile(r"/envs/env_[^/]+/")
        return pattern.sub("/envs/", prim_path)
