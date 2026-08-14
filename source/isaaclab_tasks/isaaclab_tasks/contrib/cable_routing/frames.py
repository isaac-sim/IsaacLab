# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-independent contact-frame transforms for cable routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import combine_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def contact_frame_pose_w(
    robot: Articulation,
    body_id: int,
    offset_pos: tuple[float, float, float],
    offset_quat: tuple[float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a configured contact-frame pose in world coordinates."""
    body_pos_w = robot.data.body_pos_w.torch[:, body_id]
    body_quat_w = robot.data.body_quat_w.torch[:, body_id]
    frame_offset_pos = body_pos_w.new_tensor(offset_pos).expand_as(body_pos_w)
    frame_offset_quat = body_quat_w.new_tensor(offset_quat).expand_as(body_quat_w)
    return combine_frame_transforms(body_pos_w, body_quat_w, frame_offset_pos, frame_offset_quat)


def contact_frame_position_w(
    robot: Articulation,
    body_id: int,
    offset_pos: tuple[float, float, float],
) -> torch.Tensor:
    """Return a configured contact-frame position in world coordinates [m]."""
    body_pos_w = robot.data.body_pos_w.torch[:, body_id]
    body_quat_w = robot.data.body_quat_w.torch[:, body_id]
    frame_offset_pos = body_pos_w.new_tensor(offset_pos).expand_as(body_pos_w)
    return body_pos_w + quat_apply(body_quat_w, frame_offset_pos)
