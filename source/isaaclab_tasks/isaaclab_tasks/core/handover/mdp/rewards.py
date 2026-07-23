# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the manager-based handover task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from isaaclab_tasks.core.reorient.mdp.rewards import EpisodeErrorRecorder

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def handover_reward(goal_distance: torch.Tensor, distance_scale: float) -> torch.Tensor:
    """Return one hand's Direct reward for the current object-goal distance."""
    return 2.0 * torch.exp(-distance_scale * goal_distance)


@torch.jit.script
def evaluate_handover_success(
    object_position: torch.Tensor, target_position: torch.Tensor, success_distance_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate handover success while exposing its physical error.

    Args:
        object_position: Object positions [m].
        target_position: Goal positions [m].
        success_distance_threshold: Exclusive successful goal-distance threshold [m].

    Returns:
        Per-environment success flags and object-to-goal distances [m].
    """
    goal_distance = torch.linalg.norm(object_position - target_position, ord=2, dim=-1)
    return goal_distance < success_distance_threshold, goal_distance


class HandoverReward(ManagerTermBase):
    """Compute summed hand rewards and track sticky per-episode success.

    The scalar task parameters arrive as term params, set at the configuration
    declaration site to match the Direct environment's values.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_distance = EpisodeErrorRecorder(self.num_envs, self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # 0-dim device tensor: avoids a host sync here; consumers read it at logging cadence
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self._episode_succeeded[env_ids].float().mean()
        for statistic, value in self._goal_distance.reset(env_ids).items():
            self._env.extras["log"][f"Diagnostics/episode_min_goal_distance_{statistic}"] = value
        self._episode_succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        distance_scale: float,
        success_distance_threshold: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        object_asset: RigidObject = env.scene[object_cfg.name]
        object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
        goal_pos = env.command_manager.get_command(command_name)[:, :3]
        succeeded, goal_distance = evaluate_handover_success(object_pos, goal_pos, success_distance_threshold)
        self._goal_distance.update(goal_distance)
        per_agent_reward = handover_reward(goal_distance, distance_scale)

        # tensors, not .item(): a host sync every step stalls the GPU at large env counts
        goal_distance_mean = goal_distance.mean()
        env.extras.setdefault("log", {})["dist_reward"] = per_agent_reward.mean()
        env.extras["log"]["dist_goal"] = goal_distance_mean
        env.extras["log"]["Metrics/goal_distance"] = goal_distance_mean
        self._episode_succeeded |= succeeded

        # RewardManager applies step_dt to all terms. Divide here to preserve the Direct
        # environment's per-control-step reward exposed after summing both agents.
        return 2.0 * per_agent_reward / env.step_dt
