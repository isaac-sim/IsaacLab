# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warnings
from dataclasses import dataclass


@dataclass
class FrameTransformerData:
    """Data container for the frame transformer sensor."""

    target_frame_names: list[str] = None
    """Target frame names (this denotes the order in which that frame data is ordered).

    The frame names are resolved from the :attr:`FrameTransformerCfg.FrameCfg.name` field.
    This usually follows the order in which the frames are defined in the config. However, in
    the case of regex matching, the order may be different.
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

    @property
    def target_rot_source(self) -> torch.Tensor:
        """Alias for :attr:`target_quat_source`.

        .. deprecated:: v0.2.1
            Use :attr:`target_quat_source` instead. Will be removed in v0.3.0.
        """
        warnings.warn("'target_rot_source' is deprecated, use 'target_quat_source' instead.", DeprecationWarning)
        return self.target_quat_source

    @property
    def target_rot_w(self) -> torch.Tensor:
        """Alias for :attr:`target_quat_w`.

        .. deprecated:: v0.2.1
            Use :attr:`target_quat_w` instead. Will be removed in v0.3.0.
        """
        warnings.warn("'target_rot_w' is deprecated, use 'target_quat_w' instead.", DeprecationWarning)
        return self.target_quat_w

    @property
    def source_rot_w(self) -> torch.Tensor:
        """Alias for :attr:`source_quat_w`.

        .. deprecated:: v0.2.1
            Use :attr:`source_quat_w` instead. Will be removed in v0.3.0.
        """
        warnings.warn("'source_rot_w' is deprecated, use 'source_quat_w' instead.", DeprecationWarning)
        return self.source_quat_w
