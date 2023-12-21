# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import carb
import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.utils.math import (
    combine_frame_transforms,
    convert_quat,
    is_identity_pose,
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

    Note:
        Currently, this implementation only handles frames within an articulation. This is because the frame
        regex expressions are resolved based on their parent prim path. This can be extended to handle
        frames outside of articulation by using the frame prim path instead. However, this would require
        additional checks to ensure that the user-specified frames are valid.

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
            f"\ttracked body frames: {self._tracked_body_names} \n"
            f"\tnumber of envs: {self._num_envs}\n"
            f"\tsource frame: {self._tracked_body_names[0]}\n"
            f"\ttarget frames: {self._target_frame_names} count: {len(self._target_frame_names)}\n"
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

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = ...

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()

        # resolve source frame offset
        source_frame_offset_pos = torch.tensor(self.cfg.source_frame_offset.pos, device=self.device)
        source_frame_offset_rot = torch.tensor(self.cfg.source_frame_offset.rot, device=self.device)
        # Only need to perform offsetting of source frame if the position offsets is non-zero and rotation offset is
        # not the identity quaternion for efficiency in _update_buffer_impl
        self._apply_source_frame_offset = True
        # Handle source frame offsets
        if is_identity_pose(source_frame_offset_pos, source_frame_offset_rot):
            carb.log_verbose(f"Not applying offset for source frame as it is identity: {self.cfg.prim_path}")
            self._apply_source_frame_offset = False
        else:
            carb.log_verbose(f"Applying offset for source frame as it is not identity: {self.cfg.prim_path}")
            # Store offsets as tensors (duplicating each env's offsets for ease of multiplication later)
            self._source_frame_offset_pos = source_frame_offset_pos.unsqueeze(0).repeat(self._num_envs, 1)
            self._source_frame_offset_rot = source_frame_offset_rot.unsqueeze(0).repeat(self._num_envs, 1)

        # The offsets associated with each target frame
        target_offsets = {}

        # Keep track of mapping from the rigid body name to the desired frame, as there may be multiple frames
        # based upon the same body name and we don't want to create unnecessary views
        body_names_to_frames = {}

        # The frames whose offsets are not identity
        non_identity_offset_frames = []

        # Only need to perform offsetting of target frame if any of the position offsets are non-zero or any of the
        # rotation offsets are not the identity quaternion for efficiency in _update_buffer_impl
        self._apply_target_frame_offset = False

        # Collect all target frames, their associated body prim paths and their offsets so that we can extract
        # the prim, check that it has the appropriate rigid body API in a single loop.
        # First element is None because user can't specify source frame name
        frames = [None] + [target_frame.name for target_frame in self.cfg.target_frames]
        frame_prim_paths = [self.cfg.prim_path] + [target_frame.prim_path for target_frame in self.cfg.target_frames]
        # First element is None because source frame offset is handled above
        frame_offsets = [None] + [target_frame.offset for target_frame in self.cfg.target_frames]
        for frame, prim_path, offset in zip(frames, frame_prim_paths, frame_offsets):
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
                # check if it is a rigid prim
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(
                        f"While resolving expression '{prim_path}' found a prim '{matching_prim_path}' which is not a"
                        " rigid body. The class only supports transformations between rigid bodies."
                    )

                body_name = matching_prim_path.rsplit("/", 1)[-1]

                # Use body_name if frame isn't specified by user
                if frame is None:
                    frame_name = body_name
                else:
                    frame_name = frame

                if body_name in body_names_to_frames:
                    body_names_to_frames[body_name].add(frame_name)
                else:
                    body_names_to_frames[body_name] = {frame_name}

                if offset is not None:
                    offset_pos = torch.tensor(offset.pos, device=self.device)
                    offset_rot = torch.tensor(offset.rot, device=self.device)
                    # Check if we need to apply offsets
                    if not is_identity_pose(offset_pos, offset_rot):
                        non_identity_offset_frames.append(frame_name)
                        self._apply_target_frame_offset = True

                    target_offsets[frame_name] = {"pos": offset_pos, "rot": offset_rot}

        if not self._apply_target_frame_offset:
            carb.log_info(
                f"Not applying offset from {self.cfg.prim_path} to target frames as all are identity: {frames}"
            )
        else:
            carb.log_info(
                f"Applying offset from {self.cfg.prim_path} as the following frames are non-identity:"
                f" {non_identity_offset_frames}"
            )

        # The names of bodies that RigidPrimView will be tracking to later extract transforms from
        self._tracked_body_names = list(body_names_to_frames.keys())
        num_tracked_bodies = len(self._tracked_body_names)

        # The number of target body frames being tracked by RigidPrimView. Subtract one to remove source frame from
        # count of frames
        self._num_target_body_frames = num_tracked_bodies - 1

        # Determine indices into all tracked body frames for both source and target frames
        all_idxs = torch.arange(self._num_envs * num_tracked_bodies)
        self._source_frame_idxs = torch.arange(self._num_envs) * num_tracked_bodies
        self._target_frame_idxs = all_idxs[~torch.isin(all_idxs, self._source_frame_idxs)]

        # Construct regex expression for the body names
        body_names_regex = r"(" + "|".join(self._tracked_body_names) + r")"
        body_names_regex = f"{self.cfg.prim_path.rsplit('/', 1)[0]}/{body_names_regex}"

        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # Create a prim view for all frames and initialize it
        # order of transforms coming out of view will be source frame followed by target frame(s)
        self._frame_physx_view = self._physics_sim_view.create_rigid_body_view(body_names_regex.replace(".*", "*"))

        # Determine the order in which regex evaluated body names so we can later index into frame transforms
        # by frame name correctly
        all_prim_paths = self._frame_physx_view.prim_paths

        # Only need first env as the names and their orderring are the same across environments
        first_env_prim_paths = all_prim_paths[0 : self._num_target_body_frames + 1]
        first_env_body_names = [first_env_prim_path.split("/")[-1] for first_env_prim_path in first_env_prim_paths]

        target_frame_body_names = first_env_body_names[1:]

        # The position and rotation components of target frame offsets
        target_frame_offset_pos = []
        target_frame_offset_rot = []

        # The name of each of the target frame(s) - either user specified or defaulted to the body name
        self._target_frame_names = []

        # Stores the indices of bodies that need to be duplicated. For instance, if body "LF_SHANK" is needed
        # for 2 frames, this list enables us to duplicate the body to both frames when doing the calculations
        # when updating sensor in _update_buffers_impl
        duplicate_frame_indices = []

        # Go through each body name and determine the number of duplicates we need for that frame
        # and extract the offsets. This is all done to handles the case where multiple frames
        # reference the same body, but have different names and/or offsets
        for i, body_name in enumerate(target_frame_body_names):
            for frame in body_names_to_frames[body_name]:
                target_frame_offset_pos.append(target_offsets[frame]["pos"])
                target_frame_offset_rot.append(target_offsets[frame]["rot"])
                self._target_frame_names.append(frame)
                duplicate_frame_indices.append(i)

        duplicate_frame_indices = torch.tensor(duplicate_frame_indices, device=self.device)

        # To handle multiple environments, need to expand so [0, 1, 1, 2] with 2 environments becomes
        # [0, 1, 1, 2, 3, 4, 4, 5]. Again, this is a optimization to make _update_buffer_impl more efficient
        self._duplicate_frame_indices = torch.cat(
            [duplicate_frame_indices + self._num_target_body_frames * env_num for env_num in range(self._num_envs)]
        )

        # Stack up all the frame offsets
        self._target_frame_offset_pos = torch.stack(target_frame_offset_pos).repeat(self._num_envs, 1)
        self._target_frame_offset_rot = torch.stack(target_frame_offset_rot).repeat(self._num_envs, 1)

        # fill the data buffer
        self._data.target_frame_names = self._target_frame_names
        self._data.source_pos_w = torch.zeros(self._num_envs, 3, device=self._device)
        self._data.source_rot_w = torch.zeros(self._num_envs, 4, device=self._device)
        self._data.target_pos_w = torch.zeros(self._num_envs, len(duplicate_frame_indices), 3, device=self._device)
        self._data.target_rot_w = torch.zeros(self._num_envs, len(duplicate_frame_indices), 4, device=self._device)
        self._data.target_pos_source = torch.zeros(self._num_envs, len(duplicate_frame_indices), 3, device=self._device)
        self._data.target_rot_source = torch.zeros(self._num_envs, len(duplicate_frame_indices), 4, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = ...

        # Extract transforms from view - shape is:
        # (the total number of source and target body frames being tracked * self._num_envs, 7)
        transforms = self._frame_physx_view.get_transforms()
        # Convert quaternions as PhysX uses xyzw form
        transforms[:, 3:] = convert_quat(transforms[:, 3:], to="wxyz")

        # Process source frame transform
        source_frames = transforms[self._source_frame_idxs]
        # Only apply offset if the offsets will result in a coordinate frame transform
        if self._apply_source_frame_offset:
            source_pos_w, source_rot_w = combine_frame_transforms(
                source_frames[:, :3],
                source_frames[:, 3:],
                self._source_frame_offset_pos,
                self._source_frame_offset_rot,
            )
        else:
            source_pos_w = source_frames[:, :3]
            source_rot_w = source_frames[:, 3:]

        # Process target frame transforms
        target_frames = transforms[self._target_frame_idxs]
        duplicated_target_frame_pos_w = target_frames[self._duplicate_frame_indices, :3]
        duplicated_target_frame_rot_w = target_frames[self._duplicate_frame_indices, 3:]
        # Only apply offset if the offsets will result in a coordinate frame transform
        if self._apply_target_frame_offset:
            target_pos_w, target_rot_w = combine_frame_transforms(
                duplicated_target_frame_pos_w,
                duplicated_target_frame_rot_w,
                self._target_frame_offset_pos,
                self._target_frame_offset_rot,
            )
        else:
            target_pos_w = duplicated_target_frame_pos_w
            target_rot_w = duplicated_target_frame_rot_w

        # Compute the transform of the target frame with respect to the source frame
        total_num_frames = len(self._target_frame_names)
        target_pos_source, target_rot_source = subtract_frame_transforms(
            source_pos_w.unsqueeze(1).expand(-1, total_num_frames, -1).reshape(-1, 3),
            source_rot_w.unsqueeze(1).expand(-1, total_num_frames, -1).reshape(-1, 4),
            target_pos_w,
            target_rot_w,
        )

        # Update buffers
        # note: The frame names / ordering don't change so no need to update them after initialization
        self._data.source_pos_w[:] = source_pos_w.view(-1, 3)
        self._data.source_rot_w[:] = source_rot_w.view(-1, 4)
        self._data.target_pos_w[:] = target_pos_w.view(-1, total_num_frames, 3)
        self._data.target_rot_w[:] = target_rot_w.view(-1, total_num_frames, 4)
        self._data.target_pos_source[:] = target_pos_source.view(-1, total_num_frames, 3)
        self._data.target_rot_source[:] = target_rot_source.view(-1, total_num_frames, 4)

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
        # Update the visualized markers
        if self.frame_visualizer is not None:
            self.frame_visualizer.visualize(self._data.target_pos_w.view(-1, 3), self._data.target_rot_w.view(-1, 4))

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._frame_physx_view = None
