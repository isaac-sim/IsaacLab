# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic settings for G1 velocity tracking with the state_obs observation preset.

The settings follow the official SuperMimic code: no world-model-only warm-up and no collection noise.
"""

from isaaclab.utils import configclass, replace

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlSuperMimicActorCfg,
    RslRlSuperMimicAlgorithmCfg,
    RslRlSuperMimicRunnerCfg,
    RslRlSuperMimicTaskCfg,
    RslRlWorldModelCfg,
)

SUPERMIMIC_ALGORITHM_CFG = RslRlSuperMimicAlgorithmCfg(
    replay_buffer_capacity=4_096_000,
    num_world_model_warmup_iterations=0,
    action_noise=0.0,
    use_mixed_precision=True,
    use_cuda_graph=True,
    task_cfg=RslRlSuperMimicTaskCfg(class_name="isaaclab_tasks.core.velocity.supermimic:VelocityTask"),
)

# dt 0.02 * (track_lin 1 + track_ang 1 + feet_air_time 0.75 * 0.4).
FLAT_MAX_STEP_REWARD = 0.046
# dt 0.02 * (track_lin 1 + track_ang 2 + feet_air_time 0.25 * 0.4).
ROUGH_MAX_STEP_REWARD = 0.062


@configclass
class G1FlatSuperMimicRunnerCfg(RslRlSuperMimicRunnerCfg):
    """Collect one physical step before applying fractional learner updates."""

    num_steps_per_env = 1
    max_iterations = 36_000
    save_interval = 1200
    experiment_name = "g1_flat_supermimic"
    clip_actions = None
    obs_groups = {
        "state": ["state"],
        "goal": ["goal"],
        "context": ["context"],
        "labels": ["labels"],
    }
    actor = RslRlSuperMimicActorCfg()
    world_model = RslRlWorldModelCfg()
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = replace(SUPERMIMIC_ALGORITHM_CFG, max_step_reward=FLAT_MAX_STEP_REWARD)


@configclass
class G1RoughSuperMimicRunnerCfg(G1FlatSuperMimicRunnerCfg):
    """Give the critic the rough PPO critic's widths for its additional height scan."""

    experiment_name = "g1_rough_supermimic"
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = replace(SUPERMIMIC_ALGORITHM_CFG, max_step_reward=ROUGH_MAX_STEP_REWARD)
