# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic settings for G1 velocity tracking with the state_obs observation preset.

The SuperMimic agents follow the official SuperMimic code: no world-model-only warm-up and no collection noise. The
BlinkMimic agents follow BlinkMimic (arXiv 2611.16422): Muon on the actor and world model, a four times higher actor
learning rate that decays after 1400 learner iterations, mirrored world-model batches, a smaller world model and policy
batch, and a deployment actor average.
"""

from isaaclab.utils import configclass, replace

from isaaclab_rl.rsl_rl import (
    RslRlLearnedDynamicsCfg,
    RslRlMLPModelCfg,
    RslRlSuperMimicActorCfg,
    RslRlSuperMimicAlgorithmCfg,
    RslRlSuperMimicRunnerCfg,
    RslRlSuperMimicTaskCfg,
    RslRlWorldModelCfg,
)

SUPERMIMIC_ALGORITHM_CFG = RslRlSuperMimicAlgorithmCfg(
    world_model_cfg=RslRlLearnedDynamicsCfg(replay_buffer_capacity=4_096_000, num_warmup_iterations=0),
    action_noise=0.0,
    use_mixed_precision=True,
    use_cuda_graph=True,
    task_cfg=RslRlSuperMimicTaskCfg(class_name="isaaclab_tasks.core.velocity.config.g1.supermimic:G1VelocityTask"),
)

BLINKMIMIC_ALGORITHM_CFG = replace(
    SUPERMIMIC_ALGORITHM_CFG,
    world_model_cfg=replace(
        SUPERMIMIC_ALGORITHM_CFG.world_model_cfg,
        model=RslRlWorldModelCfg(hidden_dims=(1024, 1024, 1024)),
        num_updates=5,
        mirror_fraction=0.5,
    ),
    policy_batch_size=2048,
    learning_rate=4e-4,
    muon_models=("actor", "world_model"),
    actor_learning_rate_multiplier=4.0,
    actor_learning_rate_decay_start=1400,
    # an exponential moving average with decay 0.995 per actor update
    actor_ema_halflife=138,
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
        "actor": ["policy", "history"],
        "state": ["state"],
        "goal": ["goal"],
        "context": ["context"],
        "labels": ["labels"],
    }
    actor = RslRlSuperMimicActorCfg()
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = replace(SUPERMIMIC_ALGORITHM_CFG, max_step_reward=FLAT_MAX_STEP_REWARD)


@configclass
class G1FlatBlinkMimicRunnerCfg(G1FlatSuperMimicRunnerCfg):
    """Train flat G1 velocity tracking with the BlinkMimic settings."""

    experiment_name = "g1_flat_blinkmimic"
    algorithm = replace(BLINKMIMIC_ALGORITHM_CFG, max_step_reward=FLAT_MAX_STEP_REWARD)


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


@configclass
class G1RoughBlinkMimicRunnerCfg(G1RoughSuperMimicRunnerCfg):
    """Train rough G1 velocity tracking with the BlinkMimic settings."""

    experiment_name = "g1_rough_blinkmimic"
    algorithm = replace(BLINKMIMIC_ALGORITHM_CFG, max_step_reward=ROUGH_MAX_STEP_REWARD)
