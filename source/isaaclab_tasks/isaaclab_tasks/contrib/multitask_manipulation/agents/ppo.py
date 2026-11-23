# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-balanced PPO for contributed heterogeneous manipulation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


class TaskBalancedPPO(PPO):
    """PPO that normalizes rollout advantages independently for each task."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        task_names: Sequence[str],
        task_encoding_obs_group: str,
        task_encoding_slice: Sequence[int],
        **kwargs: Any,
    ) -> None:
        """Initialize task routing metadata and the base PPO algorithm.

        Args:
            actor: Task-headed policy model.
            critic: Task-headed value model.
            storage: On-policy rollout storage.
            task_names: Task names ordered by one-hot identity.
            task_encoding_obs_group: Observation group containing the task one-hot.
            task_encoding_slice: Half-open slice containing the task one-hot.
            **kwargs: Base PPO keyword arguments.

        Raises:
            ValueError: If task metadata is invalid or mixed mini-batch normalization is enabled.
        """
        self.task_names = tuple(task_names)
        if not self.task_names:
            raise ValueError("Expected at least one task name.")
        if len(task_encoding_slice) != 2:
            raise ValueError(f"Expected a two-element task encoding slice, got {task_encoding_slice}.")
        self.task_encoding_slice = (int(task_encoding_slice[0]), int(task_encoding_slice[1]))
        encoding_start, encoding_end = self.task_encoding_slice
        if encoding_start < 0 or encoding_end <= encoding_start:
            raise ValueError(f"Expected an ordered non-negative task encoding slice, got {self.task_encoding_slice}.")
        if encoding_end - encoding_start != len(self.task_names):
            raise ValueError(
                f"Task encoding slice {self.task_encoding_slice} has dimension {encoding_end - encoding_start},"
                f" expected {len(self.task_names)}."
            )
        if kwargs.get("normalize_advantage_per_mini_batch", False):
            raise ValueError("TaskBalancedPPO is incompatible with mixed-task mini-batch advantage normalization.")

        self.task_encoding_obs_group = task_encoding_obs_group
        self.task_statistics: dict[str, float] = {}
        super().__init__(actor, critic, storage, **kwargs)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute GAE targets and normalize advantages within each task rollout."""
        super().compute_returns(obs)
        storage = self.storage
        raw_advantages = storage.returns - storage.values
        task_encoding = storage.observations[self.task_encoding_obs_group][
            ..., self.task_encoding_slice[0] : self.task_encoding_slice[1]
        ]
        self._validate_task_encoding(task_encoding)
        task_ids = task_encoding.argmax(dim=-1)

        task_statistics = {}
        for task_id, task_name in enumerate(self.task_names):
            selected = task_ids == task_id
            if not torch.any(selected):
                raise ValueError(f"Rollout contains no samples for task '{task_name}'.")
            task_advantages = raw_advantages[selected]
            task_returns = storage.returns[selected]
            advantage_mean = task_advantages.mean()
            advantage_std = task_advantages.std(unbiased=False)
            storage.advantages[selected] = (task_advantages - advantage_mean) / (advantage_std + 1.0e-8)
            task_statistics[f"{task_name}_advantage_std"] = advantage_std.item()
            task_statistics[f"{task_name}_return_std"] = task_returns.std(unbiased=False).item()
        self.task_statistics = task_statistics

    def update(self) -> dict[str, float]:
        """Run PPO updates and append per-task rollout scale diagnostics."""
        losses = super().update()
        losses.update(self.task_statistics)
        return losses

    def _validate_task_encoding(self, task_encoding: torch.Tensor) -> None:
        """Validate one-hot task identities stored in the rollout."""
        if task_encoding.shape[-1] != len(self.task_names):
            raise ValueError(
                f"Expected {len(self.task_names)} task encoding dimensions, got {task_encoding.shape[-1]}."
            )
        is_binary = torch.logical_or(task_encoding == 0.0, task_encoding == 1.0).all()
        has_one_task = (task_encoding.sum(dim=-1) == 1.0).all()
        if not bool(is_binary and has_one_task):
            raise ValueError("Task encoding observations must be one-hot with exactly one active task per sample.")
