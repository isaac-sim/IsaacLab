# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.sensors.frame_transformer.base_frame_transformer_data import BaseFrameTransformerData
from isaaclab.utils.warp.math_ops import transform_to_vec_quat


class FrameTransformerData(BaseFrameTransformerData):
    """Data container for the Newton frame transformer sensor.

    Transform buffers are populated from the Newton sensor via
    :func:`copy_from_newton_kernel`.
    """

    _source_transforms: wp.array | None
    """Source world transforms — ``(num_envs,)`` array of ``wp.transformf``."""

    _target_transforms: wp.array | None
    """Target-relative transforms — ``(num_envs, num_targets)`` array of ``wp.transformf``."""

    _target_transforms_w: wp.array | None
    """Target world transforms — ``(num_envs, num_targets)`` array of ``wp.transformf``."""

    _stride: int
    """Entries per env in the Newton sensor flat array: ``1 + num_targets``."""

    def __init__(self):
        self._target_frame_names: list[str] | None = None
        self._source_transforms = None
        self._target_transforms = None
        self._target_transforms_w = None
        self._stride = 0

    def _create_buffers(self, num_envs: int, num_targets: int, device: str):
        """Allocates transform buffers and zero-copy views."""
        self._source_transforms = wp.zeros(num_envs, dtype=wp.transformf, device=device)
        self._target_transforms = wp.zeros((num_envs, num_targets), dtype=wp.transformf, device=device)
        self._target_transforms_w = wp.zeros((num_envs, num_targets), dtype=wp.transformf, device=device)

        # Zero-copy views — auto-reflect kernel writes to underlying transforms
        self._source_pos_w, self._source_quat_w = transform_to_vec_quat(self._source_transforms)
        self._target_pos_source, self._target_quat_source = transform_to_vec_quat(self._target_transforms)
        self._target_pos_w, self._target_quat_w = transform_to_vec_quat(self._target_transforms_w)

    @property
    def target_frame_names(self) -> list[str]:
        return self._target_frame_names

    @property
    def target_pose_source(self) -> wp.array | None:
        return None

    @property
    def target_pos_source(self) -> wp.array:
        return self._target_pos_source

    @property
    def target_quat_source(self) -> wp.array:
        return self._target_quat_source

    @property
    def target_pose_w(self) -> wp.array | None:
        return None

    @property
    def target_pos_w(self) -> wp.array:
        return self._target_pos_w

    @property
    def target_quat_w(self) -> wp.array:
        return self._target_quat_w

    @property
    def source_pose_w(self) -> wp.array | None:
        return None

    @property
    def source_pos_w(self) -> wp.array:
        return self._source_pos_w

    @property
    def source_quat_w(self) -> wp.array:
        return self._source_quat_w
