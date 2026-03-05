# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.sensors.frame_transformer.base_frame_transformer import BaseFrameTransformer
from isaaclab.utils.math import normalize, quat_from_angle_axis

from isaaclab_newton.physics import NewtonManager

from .frame_transformer_data import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg

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


class FrameTransformer(BaseFrameTransformer):
    """A sensor for reporting frame transforms using the Newton physics backend.

    This class provides an interface for reporting the transform of one or more frames (target frames)
    with respect to another frame (source frame). The source frame is specified by the user as a prim path
    (:attr:`FrameTransformerCfg.prim_path`) and the target frames are specified by the user as a list of
    prim paths (:attr:`FrameTransformerCfg.target_frames`).

    The source frame and target frames are assumed to be rigid bodies. The transform of the target frames
    with respect to the source frame is computed by first extracting the transform of the source frame
    and target frames from the Newton physics engine and then computing the relative transform between the two.

    Additionally, the user can specify an offset for the source frame and each target frame. This is useful
    for specifying the transform of the desired frame with respect to the body's center of mass, for instance.
    """

    cfg: FrameTransformerCfg
    """The configuration parameters."""

    __backend_name__: str = "newton"
    """The name of the backend for the frame transformer sensor."""

    def __init__(self, cfg: FrameTransformerCfg):
        """Initializes the frame transformer object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)
        # Create empty data container
        self._data: FrameTransformerData = FrameTransformerData()
        # Internal warp buffers (allocated in _initialize_impl)
        self._warp_source_transforms_w = None
        self._warp_target_transform_source = None

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
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_bodies(self) -> int:
        """Returns the number of target bodies being tracked."""
        return len(self._target_frame_body_names)

    @property
    def body_names(self) -> list[str]:
        """Returns the names of the target bodies being tracked."""
        return self._target_frame_body_names

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
        for view_id, view in enumerate(NewtonManager.get_physics_sim_view()):
            for body_id, view_body_name in enumerate(view.link_names):
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
            for view_id, view in enumerate(NewtonManager.get_physics_sim_view()):
                # Get the body names from the view
                view_body_names = view.link_names
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
                raise ValueError(f"Frame '{body_name}' not found.")
        # Set the target frame names
        self._data.create_buffers(self._num_envs, self._num_frames, self._target_frame_names, self._device)
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

        # Allocate intermediate buffers (not exposed through data class)
        self._warp_source_transforms_w = wp.zeros((self._num_envs,), dtype=wp.transformf, device=self._device)
        self._warp_target_transform_source = wp.zeros(
            (self._num_envs, self._num_frames), dtype=wp.transformf, device=self._device
        )

        # Bind Newton views
        self._warp_views = {}
        for key in self._warp_view_body_id:
            view = NewtonManager.get_physics_sim_view()[key].get_link_transforms(NewtonManager.get_state_0())[:, 0]
            if len(view.shape) == 1:
                view = view.reshape((-1, 1))
            self._warp_views[key] = view

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        # Update the source frame transform
        wp.launch(
            update_source_transform,
            dim=self._num_envs,
            inputs=[
                self._warp_source_offset,
                self._warp_source_body_id,
                self._warp_views[self._warp_source_view_id],
                self._warp_source_transforms_w,
                self._data._source_pos_w,
                self._data._source_quat_w,
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
                    self._data._target_pos_w,
                    self._data._target_quat_w,
                    self._data._target_pos_source,
                    self._data._target_quat_source,
                    self._warp_view_frame_id[view_id],
                    self._warp_view_body_id[view_id],
                    env_mask,
                ],
            )

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
        self._warp_views = None

    """
    Internal helpers.
    """

    def _get_connecting_lines(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given start and end points, compute positions, orientations, and lengths of connecting lines.

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
