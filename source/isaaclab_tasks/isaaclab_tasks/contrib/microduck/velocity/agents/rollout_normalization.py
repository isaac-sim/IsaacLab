# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional rollout-driven observation normalization for Swizzle."""

from rsl_rl.algorithms import PPO

from isaaclab_tasks.contrib.microduck.velstand.agents.model import MicroDuckVelStandMLPModel


class MicroDuckRolloutNormalizationPPO(PPO):
    """Update next observations once per rollout step, as locked RSL-RL 5.0 did.

    Both models must use :class:`MicroDuckVelStandMLPModel`, the existing training-only timing
    adapter, so native PPO's later batch update is suppressed. Its inference and exported normalizer
    are identical to the native MLP; actor and optimizer checkpoints retain their native layout.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not all(isinstance(model, MicroDuckVelStandMLPModel) for model in (self._raw_actor, self._raw_critic)):
            raise ValueError("Per-step normalization requires the rollout-normalization model for actor and critic")

    def process_env_step(self, obs, rewards, dones, extras):
        self._raw_actor.update_rollout_normalization(obs)
        self._raw_critic.update_rollout_normalization(obs)
        super().process_env_step(obs, rewards, dones, extras)
