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


class SuccessTracker:
    """Count the goals each episode reaches.

    A goal is reached whenever the task resamples on success, so the tracker
    counts resamples rather than re-deriving success from the error. Counting
    the resample is what keeps the total exact: the error can dip below the
    threshold on a step where the resample is suppressed, and only the resample
    marks a goal the policy actually banked.

    The count is unbounded, so it keeps separating a good policy from a better
    one long after every episode reaches its first goal.

    The call order is load-bearing, because resetting a task also samples the
    next episode's goal and so counts a goal that the finished episode never
    reached. Per step::

        earned()  # mask the post-reset step, then release
        record_goal_reached()  # once per goal the task resamples on

    and per reset::

        snapshot()              # read the finished episode, before the new goal
        <the task samples the new episode's goal>
        clear()                 # drop that goal, guard mid-step resets

    Calling :meth:`clear` before :meth:`snapshot` reports the new episode
    instead of the finished one, and folding the two into a single call credits
    every episode with the goal its own reset sampled.

    A goal reached on the step that ends an episode is not counted, because the
    reset moves the object before the task evaluates that step. The loss is at
    most one goal per episode, so it shifts the reported mean by well under one
    percent, and recovering it is not possible from this class alone.
    """

    def __init__(self, num_envs: int, device: str | torch.device):
        """Initialize per-environment goal buffers.

        Args:
            num_envs: Number of parallel environments.
            device: Device on which to store the buffers.
        """
        self._goals_reached = torch.zeros(num_envs, device=device)
        self._skip_update = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def record_goal_reached(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Count one reached goal for the given environments.

        Args:
            env_ids: Environments whose goal was just resampled.
        """
        self._goals_reached[env_ids] += 1.0

    def earned(self, reached: torch.Tensor) -> torch.Tensor:
        """Drop the goals a reset handed out, and release the guard.

        Resetting an environment samples its next goal, and the task evaluates
        that goal before any physics step has run against it. A reset pose that
        happens to sit within the success threshold of the new goal would
        otherwise bank a goal the policy never earned. This masks those out for
        the one step it can happen, then releases, so the caller cannot leave
        the guard armed.

        Args:
            reached: Environments whose goal error is within the threshold.

        Returns:
            The subset that earned the goal.
        """
        earned = reached & ~self._skip_update
        self._skip_update[:] = False
        return earned

    def snapshot(self, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        """Return the goals each episode reached, without clearing the counts.

        Call before the goal is resampled for the new episode, since that
        resample is counted too.

        Args:
            env_ids: Environments whose episodes completed.

        Returns:
            Goals reached per environment, as a float tensor.
        """
        return self._goals_reached[env_ids].clone()

    def clear(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice,
        skip_next_update: torch.Tensor,
    ) -> None:
        """Start a new episode for the given environments.

        Call after the new episode's goal has been sampled, so the goal that
        resample counted is discarded.

        Args:
            env_ids: Environments whose episodes completed.
            skip_next_update: Whether each environment is reset mid-step, with the
                task evaluating its new goal before any physics runs against it.
                Only those need the guard :meth:`earned` applies: a reset that ends
                the step leaves the task to evaluate the new goal a full step later,
                where a reach is earned.
        """
        self._goals_reached[env_ids] = 0.0
        self._skip_update[env_ids] = skip_next_update


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


@torch.jit.script
def randomize_rotation(
    rand0: torch.Tensor, rand1: torch.Tensor, x_unit_tensor: torch.Tensor, y_unit_tensor: torch.Tensor
) -> torch.Tensor:
    """Compose ``[-pi, pi]``-scaled random X- and Y-axis rotations into ``(x, y, z, w)`` quaternions.

    Args:
        rand0: Rotation amounts about the X axis in ``[-1, 1]``, scaled to ``[-pi, pi]`` [rad].
        rand1: Rotation amounts about the Y axis in ``[-1, 1]``, scaled to ``[-pi, pi]`` [rad].
        x_unit_tensor: Per-sample X axis, shape ``(count, 3)``.
        y_unit_tensor: Per-sample Y axis, shape ``(count, 3)``.

    Returns:
        Composed ``(x, y, z, w)`` unit quaternions, shape ``(count, 4)``.
    """
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
