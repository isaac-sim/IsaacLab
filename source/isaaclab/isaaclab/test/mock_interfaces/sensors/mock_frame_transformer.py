# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock frame transformer sensor for testing without Isaac Sim."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch


class MockFrameTransformerData:
    """Mock data container for frame transformer sensor.

    This class mimics the interface of BaseFrameTransformerData for testing purposes.
    All tensor properties return zero tensors with correct shapes if not explicitly set.
    """

    def __init__(
        self,
        num_instances: int,
        num_target_frames: int,
        target_frame_names: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize mock frame transformer data.

        Args:
            num_instances: Number of environment instances.
            num_target_frames: Number of target frames to track.
            target_frame_names: Names of target frames.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self._num_target_frames = num_target_frames
        self._target_frame_names = target_frame_names or [f"frame_{i}" for i in range(num_target_frames)]
        self._device = device

        # Internal storage for mock data
        self._source_pos_w: torch.Tensor | None = None
        self._source_quat_w: torch.Tensor | None = None
        self._target_pos_w: torch.Tensor | None = None
        self._target_quat_w: torch.Tensor | None = None
        self._target_pos_source: torch.Tensor | None = None
        self._target_quat_source: torch.Tensor | None = None

    # -- Properties --

    @property
    def target_frame_names(self) -> list[str]:
        """Names of target frames."""
        return self._target_frame_names

    @property
    def source_pos_w(self) -> torch.Tensor:
        """Position of source frame in world frame. Shape: (N, 3)."""
        if self._source_pos_w is None:
            return torch.zeros(self._num_instances, 3, device=self._device)
        return self._source_pos_w

    @property
    def source_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of source frame in world frame. Shape: (N, 4)."""
        if self._source_quat_w is None:
            quat = torch.zeros(self._num_instances, 4, device=self._device)
            quat[:, 0] = 1.0
            return quat
        return self._source_quat_w

    @property
    def source_pose_w(self) -> torch.Tensor:
        """Pose of source frame in world frame. Shape: (N, 7)."""
        return torch.cat([self.source_pos_w, self.source_quat_w], dim=-1)

    @property
    def target_pos_w(self) -> torch.Tensor:
        """Position of target frames in world frame. Shape: (N, M, 3)."""
        if self._target_pos_w is None:
            return torch.zeros(self._num_instances, self._num_target_frames, 3, device=self._device)
        return self._target_pos_w

    @property
    def target_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of target frames in world frame. Shape: (N, M, 4)."""
        if self._target_quat_w is None:
            quat = torch.zeros(self._num_instances, self._num_target_frames, 4, device=self._device)
            quat[..., 0] = 1.0
            return quat
        return self._target_quat_w

    @property
    def target_pose_w(self) -> torch.Tensor:
        """Pose of target frames in world frame. Shape: (N, M, 7)."""
        return torch.cat([self.target_pos_w, self.target_quat_w], dim=-1)

    @property
    def target_pos_source(self) -> torch.Tensor:
        """Position of target frames relative to source frame. Shape: (N, M, 3)."""
        if self._target_pos_source is None:
            return torch.zeros(self._num_instances, self._num_target_frames, 3, device=self._device)
        return self._target_pos_source

    @property
    def target_quat_source(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of target frames relative to source frame. Shape: (N, M, 4)."""
        if self._target_quat_source is None:
            quat = torch.zeros(self._num_instances, self._num_target_frames, 4, device=self._device)
            quat[..., 0] = 1.0
            return quat
        return self._target_quat_source

    @property
    def target_pose_source(self) -> torch.Tensor:
        """Pose of target frames relative to source frame. Shape: (N, M, 7)."""
        return torch.cat([self.target_pos_source, self.target_quat_source], dim=-1)

    # -- Setters --

    def set_source_pos_w(self, value: torch.Tensor) -> None:
        """Set source position in world frame."""
        self._source_pos_w = value.to(self._device)

    def set_source_quat_w(self, value: torch.Tensor) -> None:
        """Set source orientation in world frame."""
        self._source_quat_w = value.to(self._device)

    def set_target_pos_w(self, value: torch.Tensor) -> None:
        """Set target positions in world frame."""
        self._target_pos_w = value.to(self._device)

    def set_target_quat_w(self, value: torch.Tensor) -> None:
        """Set target orientations in world frame."""
        self._target_quat_w = value.to(self._device)

    def set_target_pos_source(self, value: torch.Tensor) -> None:
        """Set target positions relative to source."""
        self._target_pos_source = value.to(self._device)

    def set_target_quat_source(self, value: torch.Tensor) -> None:
        """Set target orientations relative to source."""
        self._target_quat_source = value.to(self._device)

    def set_mock_data(
        self,
        source_pos_w: torch.Tensor | None = None,
        source_quat_w: torch.Tensor | None = None,
        target_pos_w: torch.Tensor | None = None,
        target_quat_w: torch.Tensor | None = None,
        target_pos_source: torch.Tensor | None = None,
        target_quat_source: torch.Tensor | None = None,
    ) -> None:
        """Bulk setter for mock data.

        Args:
            source_pos_w: Source position in world frame. Shape: (N, 3).
            source_quat_w: Source orientation in world frame. Shape: (N, 4).
            target_pos_w: Target positions in world frame. Shape: (N, M, 3).
            target_quat_w: Target orientations in world frame. Shape: (N, M, 4).
            target_pos_source: Target positions relative to source. Shape: (N, M, 3).
            target_quat_source: Target orientations relative to source. Shape: (N, M, 4).
        """
        if source_pos_w is not None:
            self.set_source_pos_w(source_pos_w)
        if source_quat_w is not None:
            self.set_source_quat_w(source_quat_w)
        if target_pos_w is not None:
            self.set_target_pos_w(target_pos_w)
        if target_quat_w is not None:
            self.set_target_quat_w(target_quat_w)
        if target_pos_source is not None:
            self.set_target_pos_source(target_pos_source)
        if target_quat_source is not None:
            self.set_target_quat_source(target_quat_source)


class MockFrameTransformer:
    """Mock frame transformer sensor for testing without Isaac Sim.

    This class mimics the interface of BaseFrameTransformer for testing purposes.
    It provides the same properties and methods but without simulation dependencies.
    """

    def __init__(
        self,
        num_instances: int,
        num_target_frames: int,
        target_frame_names: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize mock frame transformer sensor.

        Args:
            num_instances: Number of environment instances.
            num_target_frames: Number of target frames to track.
            target_frame_names: Names of target frames.
            device: Device for tensor allocation.
        """
        self._num_instances = num_instances
        self._num_target_frames = num_target_frames
        self._target_frame_names = target_frame_names or [f"frame_{i}" for i in range(num_target_frames)]
        self._device = device
        self._data = MockFrameTransformerData(num_instances, num_target_frames, self._target_frame_names, device)

    # -- Properties --

    @property
    def data(self) -> MockFrameTransformerData:
        """Data container for the sensor."""
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of sensor instances."""
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        """Number of target bodies being tracked."""
        return self._num_target_frames

    @property
    def body_names(self) -> list[str]:
        """Names of target bodies being tracked."""
        return self._target_frame_names

    @property
    def device(self) -> str:
        """Device for tensor allocation."""
        return self._device

    # -- Methods --

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find target frames by name regex patterns.

        Args:
            name_keys: Regex pattern(s) to match frame names.
            preserve_order: If True, preserve order of name_keys in output.

        Returns:
            Tuple of (frame_indices, frame_names) matching the patterns.
        """
        if isinstance(name_keys, str):
            name_keys = [name_keys]

        matched_indices = []
        matched_names = []

        if preserve_order:
            for key in name_keys:
                pattern = re.compile(key)
                for i, name in enumerate(self._target_frame_names):
                    if pattern.fullmatch(name) and i not in matched_indices:
                        matched_indices.append(i)
                        matched_names.append(name)
        else:
            for i, name in enumerate(self._target_frame_names):
                for key in name_keys:
                    pattern = re.compile(key)
                    if pattern.fullmatch(name):
                        matched_indices.append(i)
                        matched_names.append(name)
                        break

        return matched_indices, matched_names

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset sensor state for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        # No-op for mock - data persists until explicitly changed
        pass

    def update(self, dt: float, force_recompute: bool = False) -> None:
        """Update sensor.

        Args:
            dt: Time step since last update.
            force_recompute: Force recomputation of buffers.
        """
        # No-op for mock - data is set explicitly
        pass
