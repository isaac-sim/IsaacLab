# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical task frames for the I2RT YAM gripper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import combine_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


YAM_CONTACT_FRAME_OFFSET_POS = (0.0, -0.044, 0.1297)
"""Midpoint of the YAM inner fingertip pads in ``link_6`` coordinates [m]."""

YAM_CONTACT_FRAME_OFFSET_QUAT = (0.0, 0.0, -0.7071067812, 0.7071067812)
"""Physical pinch-frame orientation in ``link_6``, quaternion ``(x, y, z, w)``."""


def yam_contact_frame_pose_w(robot: Articulation, body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the YAM inner-pad midpoint pose in world coordinates.

    Args:
        robot: YAM articulation containing the gripper body.
        body_id: Index of ``link_6`` in the articulation body tensors.

    Returns:
        Contact-frame positions [m] and quaternions, with shapes ``(N, 3)`` and
        ``(N, 4)`` respectively.
    """
    body_pos_w = robot.data.body_pos_w.torch[:, body_id]
    body_quat_w = robot.data.body_quat_w.torch[:, body_id]
    offset_pos = body_pos_w.new_tensor(YAM_CONTACT_FRAME_OFFSET_POS).expand_as(body_pos_w)
    offset_quat = body_quat_w.new_tensor(YAM_CONTACT_FRAME_OFFSET_QUAT).expand_as(body_quat_w)
    return combine_frame_transforms(body_pos_w, body_quat_w, offset_pos, offset_quat)


def yam_contact_frame_position_w(robot: Articulation, body_id: int) -> torch.Tensor:
    """Return the YAM inner-pad midpoint position in world coordinates [m]."""
    body_pos_w = robot.data.body_pos_w.torch[:, body_id]
    body_quat_w = robot.data.body_quat_w.torch[:, body_id]
    offset_pos = body_pos_w.new_tensor(YAM_CONTACT_FRAME_OFFSET_POS).expand_as(body_pos_w)
    return body_pos_w + quat_apply(body_quat_w, offset_pos)
