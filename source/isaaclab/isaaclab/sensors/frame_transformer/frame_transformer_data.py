# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass


@dataclass
class FrameTransformerData:
    """Data container for the frame transformer sensor."""

    target_frame_names: list[str] = None
    """Target frame names (this denotes the order in which that frame data is ordered).

    The frame names are resolved from the :attr:`FrameTransformerCfg.FrameCfg.name` field.
    This does not necessarily follow the order in which the frames are defined in the config due to
    the regex matching.
    """

    target_pos_source: torch.Tensor = None
    """Position of the target frame(s) relative to source frame.

    Shape is (N, M, 3), where N is the number of environments, and M is the number of target frames.
    """

    target_quat_source: torch.Tensor = None
    """Orientation of the target frame(s) relative to source frame quaternion (w, x, y, z).

    Shape is (N, M, 4), where N is the number of environments, and M is the number of target frames.
    """

    target_pos_w: torch.Tensor = None
    """Position of the target frame(s) after offset (in world frame).

    Shape is (N, M, 3), where N is the number of environments, and M is the number of target frames.
    """

    target_quat_w: torch.Tensor = None
    """Orientation of the target frame(s) after offset (in world frame) quaternion (w, x, y, z).

    Shape is (N, M, 4), where N is the number of environments, and M is the number of target frames.
    """

    source_pos_w: torch.Tensor = None
    """Position of the source frame after offset (in world frame).

    Shape is (N, 3), where N is the number of environments.
    """

    source_quat_w: torch.Tensor = None
    """Orientation of the source frame after offset (in world frame) quaternion (w, x, y, z).

    Shape is (N, 4), where N is the number of environments.
    """
