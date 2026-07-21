# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first curriculum terms for reach environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_experimental.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _mark_reset_requested(env_mask: wp.array(dtype=wp.bool), reset_requested: wp.array(dtype=wp.int32)):
    env_id = wp.tid()
    if env_mask[env_id]:
        wp.atomic_max(reset_requested, 0, 1)


@wp.kernel
def _update_reward_weight(
    reset_requested: wp.array(dtype=wp.int32),
    common_step_counter: int,
    num_steps: int,
    target_weight: float,
    term_weight: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    if reset_requested[0] != 0:
        if common_step_counter > num_steps:
            term_weight[0] = target_weight
        out[0] = term_weight[0]
    reset_requested[0] = 0


class ModifyRewardWeight(ManagerTermBase):
    """Update one reward weight after a reset reaches the configured step threshold."""

    # The Python-owned common step counter is intentionally read in eager mode.
    # A future captured path should first move this counter to persistent device storage.
    _warp_capturable = False

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        """Initialize persistent device state.

        Args:
            cfg: Curriculum term configuration.
            env: Environment containing the reward manager.
        """
        super().__init__(cfg, env)
        self._term_weight_wp = env.reward_manager.get_term_weight_wp(cfg.params["term_name"])
        self._reset_requested_wp = wp.zeros(1, dtype=wp.int32, device=self.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_mask: wp.array(dtype=wp.bool),
        out: wp.array(dtype=wp.float32),
        term_name: str,
        weight: float,
        num_steps: int,
    ) -> None:
        """Update the reward weight when at least one environment resets.

        Args:
            env: Environment containing the common step counter.
            env_mask: Boolean mask selecting resetting environments.
            out: Persistent one-element output containing the current weight.
            term_name: Reward term name, resolved during initialization.
            weight: Target reward weight.
            num_steps: Step threshold after which the target weight is applied.
        """
        del term_name
        wp.launch(
            _mark_reset_requested,
            dim=self.num_envs,
            inputs=[env_mask, self._reset_requested_wp],
            device=self.device,
        )
        wp.launch(
            _update_reward_weight,
            dim=1,
            inputs=[
                self._reset_requested_wp,
                env.common_step_counter,
                num_steps,
                weight,
                self._term_weight_wp,
                out,
            ],
            device=self.device,
        )
