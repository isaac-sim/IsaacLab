# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reach environment with the manager tasks' reset-driven curriculum."""

from __future__ import annotations

import functools
from collections.abc import Callable

import warp as wp
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv

from isaaclab.utils.warp import ProxyArray

from .reach_env_cfg import ReachMDP2EnvCfg


@wp.kernel
def _set_reach_reward_weights(
    reset_mask: wp.array(dtype=wp.bool),
    step: wp.array(dtype=wp.int32),
    applied: wp.array(dtype=wp.bool),
    reward_weights: wp.array2d(dtype=float),
    after_steps: int,
    action_rate: int,
    joint_vel: int,
    action_rate_weight: float,
    joint_vel_weight: float,
):
    if applied[0] or step[0] <= after_steps:
        return
    resetting = bool(False)
    for world in range(reset_mask.shape[0]):
        if reset_mask[world]:
            resetting = True
            break
    if resetting:
        for world in range(reset_mask.shape[0]):
            if action_rate >= 0:
                reward_weights[world, action_rate] = action_rate_weight
            if joint_vel >= 0:
                reward_weights[world, joint_vel] = joint_vel_weight
        applied[0] = True


class ReachMDP2RLEnv(MDP2RLEnv):
    """Apply the shared reward curriculum at the first reset after its step threshold, to the rewards that the task
    has: a reward that the task disables is left out."""

    cfg: ReachMDP2EnvCfg

    def __init__(self, cfg: ReachMDP2EnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._curriculum_step = ProxyArray(wp.zeros(1, dtype=wp.int32, device=self.device))
        self._curriculum_applied = wp.zeros(1, dtype=wp.bool, device=self.device)
        reward_indices = {op.use: op.index for op in self.mdp.ops[mdp2.Phase.REWARDS]}
        self._action_rate_index = reward_indices.get("rewards.action_rate", -1)
        self._joint_vel_index = reward_indices.get("rewards.joint_vel", -1)

    def _after_evaluate(self) -> None:
        super()._after_evaluate()
        self._curriculum_step.torch.add_(1)

    def _reset_work(self, mask: ProxyArray, **kwargs) -> list[tuple[bool, Callable[[], None]]]:
        return [
            (True, functools.partial(self._apply_curriculum, mask)),
            *super()._reset_work(mask, **kwargs),
        ]

    def _apply_curriculum(self, mask: ProxyArray) -> None:
        wp.launch(
            _set_reach_reward_weights,
            dim=1,
            inputs=[
                mask.warp,
                self._curriculum_step.warp,
                self._curriculum_applied,
                self.state.reward_weights.warp,
                self.cfg.curriculum_after_steps,
                self._action_rate_index,
                self._joint_vel_index,
                self.cfg.curriculum_action_rate,
                self.cfg.curriculum_joint_vel,
            ],
            device=self.device,
        )
