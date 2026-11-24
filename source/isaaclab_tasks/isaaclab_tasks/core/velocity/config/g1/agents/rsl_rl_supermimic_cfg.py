# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic settings for flat G1 velocity with the state_obs observation preset."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlLearnedDynamicsCfg,
    RslRlMLPModelCfg,
    RslRlSuperMimicActorCfg,
    RslRlSuperMimicAlgorithmCfg,
    RslRlSuperMimicRunnerCfg,
    RslRlSuperMimicTaskCfg,
)


@configclass
class G1FlatSuperMimicRunnerCfg(RslRlSuperMimicRunnerCfg):
    """Collect one physical step before applying fractional learner updates."""

    num_steps_per_env = 1
    max_iterations = 36_000
    save_interval = 1200
    experiment_name = "g1_flat_supermimic"
    clip_actions = None
    separate_goal_terms = ("velocity_commands",)
    obs_groups = {
        "actor": ["policy", "history", "goal"],
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
    algorithm = RslRlSuperMimicAlgorithmCfg(
        world_model_cfg=RslRlLearnedDynamicsCfg(replay_buffer_capacity=4_096_000),
        use_cuda_graph=True,
        # dt 0.02 * (track_lin 1 + track_ang 1 + feet_air_time 0.75 * 0.4).
        max_step_reward=0.046,
        task_cfg=RslRlSuperMimicTaskCfg(class_name="isaaclab_tasks.core.velocity.supermimic:VelocityTask"),
    )
