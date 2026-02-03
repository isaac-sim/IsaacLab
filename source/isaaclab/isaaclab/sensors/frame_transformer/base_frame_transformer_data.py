# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for frame transformer sensor data containers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseFrameTransformerData(ABC):
    """Data container for the frame transformer sensor.

    This base class defines the interface for frame transformer sensor data. Backend-specific
    implementations should inherit from this class and provide the actual data storage.
    """

    @property
    @abstractmethod
    def target_frame_names(self) -> list[str]:
        """Target frame names (order matches data ordering).

        Resolved from :attr:`FrameTransformerCfg.FrameCfg.name`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_pose_source(self) -> list[int]:
        """Pose of the target frame(s) relative to source frame. Shape is (N, M, 7). Quaternion in xyzw order."""
        raise NotImplementedError

    @property
    @abstractmethod
    def target_pos_source(self) -> torch.Tensor:
        """Position of the target frame(s) relative to source frame. Shape is (N, M, 3)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def target_quat_source(self) -> torch.Tensor:
        """Orientation of the target frame(s) relative to source frame (x, y, z, w). Shape is (N, M, 4)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def target_pose_w(self) -> torch.Tensor:
        """Pose of the target frame(s) after offset in world frame. Shape is (N, M, 7). Quaternion in xyzw order."""
        raise NotImplementedError

    @property
    @abstractmethod
    def target_pos_w(self) -> torch.Tensor:
        """Position of the target frame(s) after offset in world frame. Shape is (N, M, 3)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def target_quat_w(self) -> torch.Tensor:
        """Orientation of the target frame(s) after offset in world frame (x, y, z, w). Shape is (N, M, 4)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def source_pose_w(self) -> torch.Tensor:
        """Pose of the source frame after offset in world frame. Shape is (N, 7). Quaternion in xyzw order."""
        raise NotImplementedError

    @property
    @abstractmethod
    def source_pos_w(self) -> torch.Tensor:
        """Position of the source frame after offset in world frame. Shape is (N, 3)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def source_quat_w(self) -> torch.Tensor:
        """Orientation of the source frame after offset in world frame (x, y, z, w). Shape is (N, 4)."""
        raise NotImplementedError
