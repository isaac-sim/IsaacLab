# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for UR10 particle pushing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

if TYPE_CHECKING:
    from ..ur10_particle_push_env import UR10ParticlePushEnv
    from ..ur10_particle_push_env_cfg import UR10ParticlePushEnvCfg


def compute_capped_bin_goal_progress(
    bin_fraction: torch.Tensor,
    success_fraction: float | torch.Tensor,
) -> torch.Tensor:
    """Return a bounded delivery potential normalized by the success target."""
    target = torch.as_tensor(success_fraction, dtype=bin_fraction.dtype, device=bin_fraction.device)
    return (bin_fraction / target).clamp(0.0, 1.0)


def compute_transport_progress(
    particle_position_e: torch.Tensor,
    cfg: UR10ParticlePushEnvCfg,
    start_x: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return bounded source-centroid motion from episode reset toward the bin."""
    if start_x is None:
        start_x = cfg.pile_nominal_center[0]
    if isinstance(start_x, (float, int)) and cfg.bin_inner_x_bounds[0] <= start_x:
        raise ValueError("The bin mouth must lie in front of the source pile.")
    start = torch.as_tensor(start_x, dtype=particle_position_e.dtype, device=particle_position_e.device)
    travel = (cfg.bin_inner_x_bounds[0] - start).clamp_min(torch.finfo(particle_position_e.dtype).eps)
    centroid_x = particle_position_e[..., 0].mean(dim=1)
    return ((centroid_x - start) / travel).clamp(0.0, 1.0)


class _ProgressReward(ManagerTermBase):
    """Base class for signed, reset-aware potential differences."""

    def __init__(
        self,
        cfg: RewardTermCfg,
        env: UR10ParticlePushEnv,
        potential: Literal["bin", "transport"],
    ):
        super().__init__(cfg, env)
        self._potential = potential
        env.update_task_state()
        self._previous = self._value(env).clone()

    def _value(self, env: UR10ParticlePushEnv) -> torch.Tensor:
        if self._potential == "bin":
            return compute_capped_bin_goal_progress(env.bin_fraction, env.cfg.success_fraction)
        return env.transport_progress

    def __call__(self, env: UR10ParticlePushEnv) -> torch.Tensor:
        env.update_task_state()
        value = self._value(env)
        delta = value - self._previous
        self._previous.copy_(value)
        # RewardManager integrates rates over ``step_dt``; dividing here preserves a potential delta.
        return delta / env.step_dt

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._previous[selected] = self._value(self._env)[selected]


class BinProgressReward(_ProgressReward):
    """Signed progress in target-normalized bin occupancy with current-fill logging."""

    def __init__(self, cfg: RewardTermCfg, env: UR10ParticlePushEnv):
        super().__init__(cfg, env, potential="bin")

    def __call__(self, env: UR10ParticlePushEnv) -> torch.Tensor:
        reward = super().__call__(env)
        env.extras.setdefault("log", {})["Metrics/bin_fraction"] = env.bin_fraction.mean()
        return reward


class TransportProgressReward(_ProgressReward):
    """Signed progress of the material centroid toward the bin."""

    def __init__(self, cfg: RewardTermCfg, env: UR10ParticlePushEnv):
        super().__init__(cfg, env, potential="transport")


def success_event(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Return a unit-rate impulse for terminal success."""
    env.update_task_state()
    return env.success_this_step.float() / env.step_dt


def failure_event(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Return a unit-rate impulse for terminal numerical or spill failure."""
    env.update_task_state()
    failure = env.invalid_state | env.escaped_workspace | env.excessive_spill
    return failure.float() / env.step_dt


def spill_fraction(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Return the fraction of irreversibly spilled particles."""
    env.update_task_state()
    return env.spill_fraction


def action_magnitude(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Return squared finite policy-action magnitude."""
    return torch.square(env.arm_action.raw_actions).sum(dim=1)


def action_rate(env: UR10ParticlePushEnv) -> torch.Tensor:
    """Return squared finite policy-action change."""
    action = env.arm_action
    return torch.square(action.raw_actions - action.previous_actions).sum(dim=1)
