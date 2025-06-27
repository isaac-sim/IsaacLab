# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    combine_frame_transforms,
    convert_quat,
    is_identity_pose,
    normalize,
    quat_from_angle_axis,
    subtract_frame_transforms,
)

from ..sensor_base import SensorBase
from .frame_transformer_data import FrameTransformerData

if TYPE_CHECKING:
    from .frame_transformer_cfg import FrameTransformerCfg


class FrameTransformer(SensorBase):
    """A sensor for reporting frame transforms.

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

        Note:
            This is an alias used for consistency with other sensors. Otherwise, we recommend using
            :attr:`len(data.target_frame_names)` to access the number of target frames.
        """
        return len(self._target_frame_body_names)

    @property
    def body_names(self) -> list[str]:
        """Returns the names of the target bodies being tracked.

        Note:
            This is an alias used for consistency with other sensors. Otherwise, we recommend using
            :attr:`data.target_frame_names` to access the target frame names.
        """
        return self._target_frame_body_names

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = ...

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self._target_frame_names, preserve_order)

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
            omni.log.verbose(f"No offset application needed for source frame as it is identity: {self.cfg.prim_path}")
            self._apply_source_frame_offset = False
        else:
            omni.log.verbose(f"Applying offset to source frame as it is not identity: {self.cfg.prim_path}")
            # Store offsets as tensors (duplicating each env's offsets for ease of multiplication later)
            self._source_frame_offset_pos = source_frame_offset_pos.unsqueeze(0).repeat(self._num_envs, 1)
            self._source_frame_offset_quat = source_frame_offset_quat.unsqueeze(0).repeat(self._num_envs, 1)

        # Keep track of mapping from the rigid body name to the desired frames and prim path, as there may be multiple frames
        # based upon the same body name and we don't want to create unnecessary views
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

                # Get the name of the body
                body_name = matching_prim_path.rsplit("/", 1)[-1]
                # Use body name if frame isn't specified by user
                frame_name = frame if frame is not None else body_name

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
            omni.log.info(
                f"No offsets application needed from '{self.cfg.prim_path}' to target frames as all"
                f" are identity: {frames[1:]}"
            )
        else:
            omni.log.info(
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

            # Find the indices that would reorganize output to be per environment. We want `env_1/blah` to come before `env_11/blah`
            # and env_1/Robot/base to come before env_1/Robot/foot so we need to use custom key function
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
            # If no environment is present, then the order of the body names is the same as the order of the prim paths sorted alphabetically
            self._per_env_indices = [index for index, _ in sorted(enumerate(all_prim_paths), key=lambda x: x[1])]
            sorted_prim_paths = [all_prim_paths[index] for index in self._per_env_indices]

        # -- target frames
        self._target_frame_body_names = [prim_path.split("/")[-1] for prim_path in sorted_prim_paths]

        # -- source frame
        self._source_frame_body_name = self.cfg.prim_path.split("/")[-1]
        source_frame_index = self._target_frame_body_names.index(self._source_frame_body_name)

        # Only remove source frame from tracked bodies if it is not also a target frame
        if not self._source_is_also_target_frame:
            self._target_frame_body_names.remove(self._source_frame_body_name)

        # Determine indices into all tracked body frames for both source and target frames
        all_ids = torch.arange(self._num_envs * len(tracked_body_names))
        self._source_frame_body_ids = torch.arange(self._num_envs) * len(tracked_body_names) + source_frame_index

        # If source frame is also a target frame, then the target frame body ids are the same as the source frame body ids
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

        # fill the data buffer
        self._data.target_frame_names = self._target_frame_names
        self._data.source_pos_w = torch.zeros(self._num_envs, 3, device=self._device)
        self._data.source_quat_w = torch.zeros(self._num_envs, 4, device=self._device)
        self._data.target_pos_w = torch.zeros(self._num_envs, len(duplicate_frame_indices), 3, device=self._device)
        self._data.target_quat_w = torch.zeros(self._num_envs, len(duplicate_frame_indices), 4, device=self._device)
        self._data.target_pos_source = torch.zeros_like(self._data.target_pos_w)
        self._data.target_quat_source = torch.zeros_like(self._data.target_quat_w)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = ...

        # Extract transforms from view - shape is:
        # (the total number of source and target body frames being tracked * self._num_envs, 7)
        transforms = self._frame_physx_view.get_transforms()

        # Reorder the transforms to be per environment as is expected of SensorData
        transforms = transforms[self._per_env_indices]

        # Convert quaternions as PhysX uses xyzw form
        transforms[:, 3:] = convert_quat(transforms[:, 3:], to="wxyz")

        # Process source frame transform
        source_frames = transforms[self._source_frame_body_ids]
        # Only apply offset if the offsets will result in a coordinate frame transform
        if self._apply_source_frame_offset:
            source_pos_w, source_quat_w = combine_frame_transforms(
                source_frames[:, :3],
                source_frames[:, 3:],
                self._source_frame_offset_pos,
                self._source_frame_offset_quat,
            )
        else:
            source_pos_w = source_frames[:, :3]
            source_quat_w = source_frames[:, 3:]

        # Process target frame transforms
        target_frames = transforms[self._target_frame_body_ids]
        duplicated_target_frame_pos_w = target_frames[self._duplicate_frame_indices, :3]
        duplicated_target_frame_quat_w = target_frames[self._duplicate_frame_indices, 3:]

        # Only apply offset if the offsets will result in a coordinate frame transform
        if self._apply_target_frame_offset:
            target_pos_w, target_quat_w = combine_frame_transforms(
                duplicated_target_frame_pos_w,
                duplicated_target_frame_quat_w,
                self._target_frame_offset_pos,
                self._target_frame_offset_quat,
            )
        else:
            target_pos_w = duplicated_target_frame_pos_w
            target_quat_w = duplicated_target_frame_quat_w

        # Compute the transform of the target frame with respect to the source frame
        total_num_frames = len(self._target_frame_names)
        target_pos_source, target_quat_source = subtract_frame_transforms(
            source_pos_w.unsqueeze(1).expand(-1, total_num_frames, -1).reshape(-1, 3),
            source_quat_w.unsqueeze(1).expand(-1, total_num_frames, -1).reshape(-1, 4),
            target_pos_w,
            target_quat_w,
        )

        # Update buffers
        # note: The frame names / ordering don't change so no need to update them after initialization
        self._data.source_pos_w[:] = source_pos_w.view(-1, 3)
        self._data.source_quat_w[:] = source_quat_w.view(-1, 4)
        self._data.target_pos_w[:] = target_pos_w.view(-1, total_num_frames, 3)
        self._data.target_quat_w[:] = target_quat_w.view(-1, total_num_frames, 4)
        self._data.target_pos_source[:] = target_pos_source.view(-1, total_num_frames, 3)
        self._data.target_quat_source[:] = target_quat_source.view(-1, total_num_frames, 4)

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
        # Get the all frames pose
        frames_pos = torch.cat([self._data.source_pos_w, self._data.target_pos_w.view(-1, 3)], dim=0)
        frames_quat = torch.cat([self._data.source_quat_w, self._data.target_quat_w.view(-1, 4)], dim=0)

        # Get the all connecting lines between frames pose
        lines_pos, lines_quat, lines_length = self._get_connecting_lines(
            start_pos=self._data.source_pos_w.repeat_interleave(self._data.target_pos_w.size(1), dim=0),
            end_pos=self._data.target_pos_w.view(-1, 3),
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
        """
        Given start and end points, compute the positions (mid-point), orientations, and lengths of the connecting lines.

        Args:
            start_pos: The start positions of the connecting lines. Shape is (N, 3).
            end_pos: The end positions of the connecting lines. Shape is (N, 3).

        Returns:
            positions: The position of each connecting line. Shape is (N, 3).
            orientations: The orientation of each connecting line in quaternion. Shape is (N, 4).
            lengths: The length of each connecting line. Shape is (N,).
        """
        direction = end_pos - start_pos
        lengths = torch.norm(direction, dim=-1)
        positions = (start_pos + end_pos) / 2

        # Get default direction (along z-axis)
        default_direction = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(start_pos.size(0), -1)

        # Normalize direction vector
        direction_norm = normalize(direction)

        # Calculate rotation from default direction to target direction
        rotation_axis = torch.linalg.cross(default_direction, direction_norm)
        rotation_axis_norm = torch.norm(rotation_axis, dim=-1)

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
