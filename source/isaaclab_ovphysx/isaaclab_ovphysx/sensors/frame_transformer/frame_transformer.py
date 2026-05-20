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

from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.frame_transformer import BaseFrameTransformer
from isaaclab.utils.math import normalize, quat_from_angle_axis

from .frame_transformer_data import FrameTransformerData

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

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        raise NotImplementedError("FrameTransformer._initialize_impl lands in the next commit.")

    def _update_buffers_impl(self, env_mask: wp.array | None = None):
        raise NotImplementedError("FrameTransformer._update_buffers_impl lands in the next commit.")

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
        """Strip the ``/envs/env_<id>/`` prefix from a prim path so paths can be compared across environments.

        Args:
            prim_path: Absolute USD prim path that may contain an ``/envs/env_<digits>/`` segment.

        Returns:
            The prim path with that segment collapsed to ``/envs/``, so prim paths from any env compare equal.
        """
        pattern = re.compile(r"/envs/env_[^/]+/")
        return pattern.sub("/envs/", prim_path)
