# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for core learning tasks."""

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_from_angle_axis, quat_mul


class EpisodeErrorRecorder:
    """Record the minimum physical error reached in each episode.

    The recorder deliberately contains no success threshold. This keeps the
    measured task error separate from the policy that converts it to a success
    result.
    """

    def __init__(self, num_envs: int, device: str | torch.device):
        """Initialize per-environment error buffers.

        Args:
            num_envs: Number of parallel environments.
            device: Device on which to store the buffers.
        """
        self.minimum_error = torch.full((num_envs,), torch.inf, device=device)
        self._has_sample = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, error: torch.Tensor) -> None:
        """Record one error sample for every environment.

        Args:
            error: Per-environment physical errors, in task-defined units.

        Raises:
            ValueError: If :paramref:`error` does not match the recorder shape.
        """
        if error.shape != self.minimum_error.shape:
            raise ValueError(f"Expected error shape {self.minimum_error.shape}, got {error.shape}.")
        finite = torch.isfinite(error)
        # non-finite samples keep the running minimum; boolean advanced indexing would
        # force a host synchronization every step, so substitute-and-minimum instead
        torch.minimum(self.minimum_error, torch.where(finite, error, self.minimum_error), out=self.minimum_error)
        self._has_sample |= finite

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> dict[str, torch.Tensor]:
        """Summarize and clear completed episodes.

        Args:
            env_ids: Environments whose episodes completed, or ``None`` for all.

        Returns:
            Mean, median, and 90th-percentile episode-minimum errors as 0-dim
            device tensors, so logging them does not force a host
            synchronization in the reset path. The result is empty when none of
            the selected environments has a sample.
        """
        if env_ids is None:
            env_ids = slice(None)
        valid = self._has_sample[env_ids]
        values = self.minimum_error[env_ids][valid]
        statistics = {}
        if values.numel() > 0:
            statistics = {
                "mean": values.mean(),
                "median": values.median(),
                "p90": torch.quantile(values, 0.9),
            }
        self.minimum_error[env_ids] = torch.inf
        self._has_sample[env_ids] = False
        return statistics


def sample_joint_positions_within_limits(
    default_position: torch.Tensor,
    limits: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """Sample reset positions between each joint's default position and limits.

    Args:
        default_position: Default joint positions [m or rad, depending on joint type], shape ``(..., J)``.
        limits: Lower and upper joint-position limits [m or rad, depending on joint type], shape ``(..., J, 2)``.
        noise_scale: Dimensionless interpolation scale from the default position toward the sampled limits.

    Returns:
        Sampled joint positions [m or rad, depending on joint type], shape ``(..., J)``.

    Raises:
        ValueError: If :paramref:`noise_scale` is outside ``[0, 1]``.
    """
    if not 0.0 <= noise_scale <= 1.0:
        raise ValueError(f"Expected noise_scale in [0, 1], got {noise_scale}.")
    position_sample = math_utils.sample_uniform(
        -1.0,
        1.0,
        default_position.shape,
        device=default_position.device,
    )
    position_fraction = 0.5 * (position_sample + 1.0)
    position_delta = limits[..., 0] - default_position
    position_delta = position_delta + (limits[..., 1] - limits[..., 0]) * position_fraction
    joint_position = default_position + noise_scale * position_delta
    return torch.clamp(joint_position, min=limits[..., 0], max=limits[..., 1])


def random_xy_rotation(count: int, device: str | torch.device) -> torch.Tensor:
    """Sample the Direct tasks' sequential random X/Y rotation.

    Args:
        count: Number of rotations to sample.
        device: Device on which to sample.

    Returns:
        Sampled ``(x, y, z, w)`` unit quaternions, shape ``(count, 4)``.
    """
    random_values = math_utils.sample_uniform(-1.0, 1.0, (count, 2), device=device)
    x_unit = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(count, 1)
    y_unit = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(count, 1)
    return math_utils.quat_mul(
        math_utils.quat_from_angle_axis(random_values[:, 0] * torch.pi, x_unit),
        math_utils.quat_from_angle_axis(random_values[:, 1] * torch.pi, y_unit),
    )


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """Compose ``[-pi, pi]``-scaled random X- and Y-axis rotations into ``(x, y, z, w)`` quaternions."""
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
