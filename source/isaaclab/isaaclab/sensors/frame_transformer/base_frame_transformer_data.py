# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import warp as wp


class BaseFrameTransformerData(ABC):
    """Data container for the frame transformer sensor."""

    target_frame_names: list[str] | None = None
    """Target frame names in the order frame data is stored.

    Resolved from :attr:`FrameTransformerCfg.FrameCfg.name`; order may differ from
    the config due to regex matching.
    """

    @property
    @abstractmethod
    def target_pos_source(self) -> wp.array:
        """Target position(s) relative to source frame.
        (N, M) array of ``wp.vec3f`` — N envs, M target frames.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_quat_source(self) -> wp.array:
        """Target orientation(s) relative to source frame.
        (N, M) array of ``wp.quatf`` — N envs, M target frames.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_pos_w(self) -> wp.array:
        """Target position(s) in world frame (with offset applied).
        (N, M) array of ``wp.vec3f`` — N envs, M target frames.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_quat_w(self) -> wp.array:
        """Target orientation(s) in world frame (with offset applied).
        (N, M) array of ``wp.quatf`` — N envs, M target frames.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def source_pos_w(self) -> wp.array:
        """Source position in world frame (with offset applied).
        (N,) array of ``wp.vec3f`` — N envs.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def source_quat_w(self) -> wp.array:
        """Source orientation in world frame (with offset applied).
        (N,) array of ``wp.quatf`` — N envs.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_buffers(self, *args, **kwargs):
        """Allocates owned data buffers."""
        raise NotImplementedError
