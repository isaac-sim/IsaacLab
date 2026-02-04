# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.math import normalize, quat_from_angle_axis

from ..sensor_base import SensorBase
from .frame_transformer_data import FrameTransformerData

if TYPE_CHECKING:
    from .frame_transformer_cfg import FrameTransformerCfg

"""
Warp kernels
"""


@wp.kernel
def set_env_mask(env_mask: wp.array(dtype=bool), env_ids: wp.array(dtype=wp.int32)):
    """Create an environment mask from the environment ids.

    Args:
        env_mask: The environment mask (num_envs,). (modified)
        env_ids: The environment ids.
    """

    idx = wp.tid()
    env_mask[env_ids[idx]] = True


@wp.func
def split_transform_to_quat4lab(transform: wp.transformf) -> wp.vec4f:
    """Split a frame transform into a quaternion in wxyz order.

    Args:
        transform: The frame transform in xyzw order.

    Returns:
        The quaternion in wxyz order.
    """
    quat = wp.transform_get_rotation(transform)
    return wp.vec4f(quat[3], quat[0], quat[1], quat[2])


@wp.kernel
def update_source_transform(
    offset: wp.transformf,
    source_index: int,
    source_transforms: wp.array2d(dtype=wp.transformf),
    output_transforms: wp.array(dtype=wp.transformf),
    output_pos: wp.array(dtype=wp.vec3f),
    output_quat: wp.array(dtype=wp.vec4f),
    env_mask: wp.array(dtype=bool),
):
    """Update the source transform.

    This kernel applies the offset to the source frame transform and outputs the results
    in the output transforms, positions, and quaternions.

    .. note:: The output positions and quaternions are used to keep everything compatible
        with the rest of the code but they are not needed.

    Args:
        offset: The offset of the source frame.
        source_index: The index of the source frame.
        source_transforms: The source frame transforms.
        output_transforms: The output transforms (modified).
        output_pos: The output positions (modified).
        output_quat: The output quaternions (modified).
        env_mask: The environment mask.
    """
    idx = wp.tid()
    if env_mask[idx]:
        output_transforms[idx] = source_transforms[idx, source_index] * offset
        output_pos[idx] = wp.transform_get_translation(output_transforms[idx])
        output_quat[idx] = split_transform_to_quat4lab(output_transforms[idx])


@wp.kernel
def update_frame_transforms(
    frame_offsets: wp.array(dtype=wp.transformf),
    origin_transforms: wp.array(dtype=wp.transformf),
    frames_transforms_world: wp.array2d(dtype=wp.transformf),
    frames_transforms_origin: wp.array2d(dtype=wp.transformf),
    frames_pos_world: wp.array2d(dtype=wp.vec3f),
    frames_quat_world: wp.array2d(dtype=wp.vec4f),
    frames_pos_origin: wp.array2d(dtype=wp.vec3f),
    frames_quat_origin: wp.array2d(dtype=wp.vec4f),
    frame_to_view_ids: wp.array(dtype=wp.int32),
    body_to_view_ids: wp.array(dtype=wp.int32),
    env_mask: wp.array(dtype=bool),
):
    """Update the frame transforms.

    This kernel updates the frame transforms in the origin frame and the world frame.

    .. note:: The output positions and quaternions are used to keep everything compatible
        with the rest of the code but they are not needed.

    Args:
        frame_offsets: The offsets of the frames.
        origin_transforms: The origin transforms.
        frames_transforms_world: The frame transforms in world frame.
        frames_transforms_origin: The frame transforms in origin frame (modified).
        frames_pos_world: The frame positions in world frame (modified).
        frames_quat_world: The frame quaternions in world frame (modified).
        frames_pos_origin: The frame positions in origin frame (modified).
        frames_quat_origin: The frame quaternions in origin frame (modified).
        frame_to_view_ids: The mapping from frame to view ids.
        body_to_view_ids: The mapping from body to view ids.
        env_mask: The environment mask.
    """
    env_idx, frame_idx = wp.tid()

    if env_mask[env_idx]:
        frames_transforms_origin[env_idx][frame_to_view_ids[frame_idx]] = (
            frames_transforms_world[env_idx][body_to_view_ids[frame_idx]] * frame_offsets[frame_to_view_ids[frame_idx]]
        )
        frames_pos_world[env_idx][frame_to_view_ids[frame_idx]] = wp.transform_get_translation(
            frames_transforms_origin[env_idx][frame_to_view_ids[frame_idx]]
        )
        frames_quat_world[env_idx][frame_to_view_ids[frame_idx]] = split_transform_to_quat4lab(
            frames_transforms_origin[env_idx][frame_to_view_ids[frame_idx]]
        )
        frames_transforms_origin[env_idx][frame_to_view_ids[frame_idx]] = (
            wp.transform_inverse(origin_transforms[env_idx])
            * frames_transforms_origin[env_idx][frame_to_view_ids[frame_idx]]
        )
        frames_pos_origin[env_idx][frame_to_view_ids[frame_idx]] = wp.transform_get_translation(
            frames_transforms_origin[env_idx][frame_to_view_ids[frame_idx]]
        )
        frames_quat_origin[env_idx][frame_to_view_ids[frame_idx]] = split_transform_to_quat4lab(
            frames_transforms_origin[env_idx][frame_to_view_ids[frame_idx]]
        )


"""
FrameTransformer class
"""


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
        # Warp buffers used to store the frame transforms.
        # Note we bind these buffers to the data fields in the _initialize_impl method. This way any changes
        # made to the buffers are reflected in the data fields.
        self._ALL_ENV_MASK = None
        self._ENV_MASK = None
        self._warp_source_pos_w = None
        self._warp_source_quat_w = None
        self._warp_source_transforms_w = None
        self._warp_target_pos_w = None
        self._warp_target_quat_w = None
        self._warp_target_pos_source = None
        self._warp_target_quat_source = None
        self._warp_target_transform_source = None

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

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
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

        # Collect all target frames, their associated body prim paths and their offsets
        frames = [target_frame.name for target_frame in self.cfg.target_frames]
        frame_prim_paths = [target_frame.prim_path for target_frame in self.cfg.target_frames]
        # First element is None because source frame offset is handled separately
        frame_offsets = [target_frame.offset for target_frame in self.cfg.target_frames]

        # Loop through all the views attached to the Newton manager to find the source frame body
        # and set view id and the id of the body within that view.
        source_frame_prim_path = self.cfg.prim_path
        frame_found = False
        body_name = source_frame_prim_path.rsplit("/", 1)[-1]
        for view_id, view in enumerate(NewtonManager.get_views()):
            for body_id, view_body_name in enumerate(view.body_names):
                if body_name == view_body_name:
                    self._warp_source_body_id = body_id
                    self._warp_source_view_id = view_id
                    # Convert the offset to a wp.transformf
                    self._warp_source_offset = wp.transformf(
                        self.cfg.source_frame_offset.pos[0],
                        self.cfg.source_frame_offset.pos[1],
                        self.cfg.source_frame_offset.pos[2],
                        self.cfg.source_frame_offset.rot[1],
                        self.cfg.source_frame_offset.rot[2],
                        self.cfg.source_frame_offset.rot[3],
                        self.cfg.source_frame_offset.rot[0],
                    )
                    frame_found = True
                    break
            if frame_found:
                break
        # Raise an error if the source frame body is not found
        if not frame_found:
            raise ValueError(f"Source frame '{body_name}' not found.")
        self._source_frame_body_name = body_name
        # Print the information about the source frame body
        print("[INFO]: Initializing FrameTransformer!")
        print(f"[INFO]: Using source body: {body_name} as reference frame.")
        print(f"[INFO]: + Body found in view id: {self._warp_source_view_id}.")
        print(f"[INFO]: + Body id in view {self._warp_source_view_id}: {self._warp_source_body_id}.")

        # Go through all the target frames and find the bodies matching the regex
        matching_prims = []
        offsets = []
        self._target_frame_names = []
        for prim_path, offset, frame_name in zip(frame_prim_paths, frame_offsets, frames):
            # Find the matching prims
            prims = sim_utils.find_matching_prims(prim_path)
            # Duplicate the target frame names if more than one prim is found
            self._target_frame_names.extend([frame_name] * len(prims))
            # Add to the list all the matching prims
            matching_prims.extend(prims)
            # Duplicate the offsets if more than one prim is found
            offsets.extend([offset] * len(prims))
        # Convert the matching prims to a list of prim paths
        matching_prims = [prim.GetPath().pathString for prim in matching_prims]

        # Set the number of bodies/frames found in the scene
        self._num_frames = len(matching_prims)
        # Create a buffer to store the pose of the target frames
        pose = torch.zeros((self._num_frames, 7), dtype=torch.float32, device=self._device)
        # Set the default to identity transform
        pose[:, -1] = 1.0

        # Create a dictionary to store the body id, frame id and body name for each view
        self._warp_view_body_id = {}
        self._warp_view_frame_id = {}
        self._warp_view_body_name = {}
        self._target_frame_body_names = []

        # Loop through all the matching prims and offsets to find the bodies in the scene
        for frame_id, (prim_path, offset) in enumerate(zip(matching_prims, offsets)):
            frame_found = False
            # Go through all the views to find the requested bodies
            body_name = prim_path.rsplit("/", 1)[-1]
            for view_id, view in enumerate(NewtonManager.get_views()):
                # Get the body names from the view
                view_body_names = view.body_names
                for body_id, view_body_name in enumerate(view_body_names):
                    if body_name == view_body_name:
                        frame_found = True
                        if view_id not in self._warp_view_body_id.keys():
                            self._warp_view_body_id[view_id] = [body_id]
                            self._warp_view_frame_id[view_id] = [frame_id]
                            self._warp_view_body_name[view_id] = [body_name]
                        else:
                            self._warp_view_body_id[view_id].append(body_id)
                            self._warp_view_frame_id[view_id].append(frame_id)
                            self._warp_view_body_name[view_id].append(body_name)
                        self._target_frame_body_names.append(body_name)
                        break
                    if frame_found:
                        break
                # If the offset is not None, then set the pose of the target frame
                if offset is not None:
                    pose[frame_id, :3] = torch.tensor(offset.pos, device=self._device)
                    # Warp wants quaternions in xyzw order
                    pose[frame_id, 3] = offset.rot[1]
                    pose[frame_id, 4] = offset.rot[2]
                    pose[frame_id, 5] = offset.rot[3]
                    pose[frame_id, 6] = offset.rot[0]
            # Raise an error if the frame is not found
            if not frame_found:
                raise ValueError(f"Frame '{body_name}' found.")
        # Set the target frame names
        self._data.target_frame_names = self._target_frame_names
        print(f"[INFO]: Found {self._num_frames} target frames.")
        for key in self._warp_view_body_name:
            print(f"[INFO]: + Found {len(self._warp_view_body_name[key])} bodies in view {key}.")
            for body_name, body_id, frame_id in zip(
                self._warp_view_body_name[key], self._warp_view_body_id[key], self._warp_view_frame_id[key]
            ):
                print(f"[INFO]:   + Found {body_name} in view {key} with body id {body_id} and frame id {frame_id}.")
        print("[INFO]: FrameTransformer initialized!")
        # Convert the pose to a wp.array
        self._warp_offset_buffer = wp.from_torch(pose, dtype=wp.transformf)

        for key in self._warp_view_body_id.keys():
            self._warp_view_body_id[key] = wp.array(self._warp_view_body_id[key], dtype=wp.int32)
        for key in self._warp_view_frame_id.keys():
            self._warp_view_frame_id[key] = wp.array(self._warp_view_frame_id[key], dtype=wp.int32)

        # Populate the warp buffers
        self._ALL_ENV_MASK = wp.ones((self._num_envs,), dtype=bool, device=self._device)
        self._ENV_MASK = wp.zeros((self._num_envs,), dtype=bool, device=self._device)
        self._warp_source_pos_w = wp.zeros((self._num_envs,), dtype=wp.vec3f, device=self._device)
        self._warp_source_quat_w = wp.zeros((self._num_envs,), dtype=wp.vec4f, device=self._device)
        self._warp_source_transforms_w = wp.zeros((self._num_envs,), dtype=wp.transformf, device=self._device)
        self._warp_target_pos_w = wp.zeros(
            (
                self._num_envs,
                self._num_frames,
            ),
            dtype=wp.vec3f,
            device=self._device,
        )
        self._warp_target_quat_w = wp.zeros(
            (
                self._num_envs,
                self._num_frames,
            ),
            dtype=wp.vec4f,
            device=self._device,
        )
        self._warp_target_pos_source = wp.zeros_like(self._warp_target_pos_w, device=self._device)
        self._warp_target_quat_source = wp.zeros_like(self._warp_target_quat_w, device=self._device)
        self._warp_target_transform_source = wp.zeros(
            (
                self._num_envs,
                self._num_frames,
            ),
            dtype=wp.transformf,
            device=self._device,
        )
        # Bindings with dataclass
        self._data.source_pos_w = wp.to_torch(self._warp_source_pos_w)
        self._data.source_quat_w = wp.to_torch(self._warp_source_quat_w)
        self._data.target_pos_w = wp.to_torch(self._warp_target_pos_w)
        self._data.target_quat_w = wp.to_torch(self._warp_target_quat_w)
        self._data.target_pos_source = wp.to_torch(self._warp_target_pos_source)
        self._data.target_quat_source = wp.to_torch(self._warp_target_quat_source)
        # Bind with Newton:
        self._warp_views = {}
        for key in self._warp_view_body_id:
            view = NewtonManager._views[key].get_link_transforms(NewtonManager.get_state_0())[:, 0]
            if len(view.shape) == 1:
                view = view.reshape((-1, 1))
            self._warp_views[key] = view

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # default to all sensors, create a mask for the environment ids if needed
        if (len(env_ids) == self._num_envs) or (env_ids is None):
            env_mask = self._ALL_ENV_MASK
        else:
            self._ENV_MASK.fill_(False)
            wp.launch(
                set_env_mask,
                dim=len(env_ids),
                inputs=[
                    self._ENV_MASK,
                    env_ids.to(torch.int32),
                ],
            )
            env_mask = self._ENV_MASK

        # Update the source frame transform
        wp.launch(
            update_source_transform,
            dim=self._num_envs,
            inputs=[
                self._warp_source_offset,
                self._warp_source_body_id,
                self._warp_views[self._warp_source_view_id],
                self._warp_source_transforms_w,
                self._warp_source_pos_w,
                self._warp_source_quat_w,
                env_mask,
            ],
        )
        # Update the frame transforms in the origin frame and the world frame
        for view_id in self._warp_view_body_id.keys():
            wp.launch(
                update_frame_transforms,
                dim=(self._num_envs, self._warp_view_frame_id[view_id].shape[0]),
                inputs=[
                    self._warp_offset_buffer,
                    self._warp_source_transforms_w,
                    self._warp_views[view_id],
                    self._warp_target_transform_source,
                    self._warp_target_pos_w,
                    self._warp_target_quat_w,
                    self._warp_target_pos_source,
                    self._warp_target_quat_source,
                    self._warp_view_frame_id[view_id],
                    self._warp_view_body_id[view_id],
                    env_mask,
                ],
            )

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #    # set visibility of markers
    #    # note: parent only deals with callbacks. not their visibility
    #    if debug_vis:
    #        if not hasattr(self, "frame_visualizer"):
    #            self.frame_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)

    #        # set their visibility to true
    #        self.frame_visualizer.set_visibility(True)
    #    else:
    #        if hasattr(self, "frame_visualizer"):
    #            self.frame_visualizer.set_visibility(False)

    # def _debug_vis_callback(self, event):
    #    return
    #    # Get the all frames pose
    #    frames_pos = torch.cat([self._data.source_pos_w, self._data.target_pos_w.view(-1, 3)], dim=0)
    #    frames_quat = torch.cat([self._data.source_quat_w, self._data.target_quat_w.view(-1, 4)], dim=0)

    #    # Get the all connecting lines between frames pose
    #    lines_pos, lines_quat, lines_length = self._get_connecting_lines(
    #        start_pos=self._data.source_pos_w.repeat_interleave(self._data.target_pos_w.size(1), dim=0),
    #        end_pos=self._data.target_pos_w.view(-1, 3),
    #    )

    #    # Initialize default (identity) scales and marker indices for all markers (frames + lines)
    #    marker_scales = torch.ones(frames_pos.size(0) + lines_pos.size(0), 3)
    #    marker_indices = torch.zeros(marker_scales.size(0))

    #    # Set the z-scale of line markers to represent their actual length
    #    marker_scales[-lines_length.size(0) :, -1] = lines_length

    #    # Assign marker config index 1 to line markers
    #    marker_indices[-lines_length.size(0) :] = 1

    #    # Update the frame and the connecting line visualizer
    #    self.frame_visualizer.visualize(
    #        translations=torch.cat((frames_pos, lines_pos), dim=0),
    #        orientations=torch.cat((frames_quat, lines_quat), dim=0),
    #        scales=marker_scales,
    #        marker_indices=marker_indices,
    #    )

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
