# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL SuperMimic configurations for G1 motion tracking."""

from copy import deepcopy

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlResidualMLPModelCfg
from isaaclab_rl.rsl_rl.supermimic_cfg import (
    RslRlLearnedDynamicsCfg,
    RslRlSuperMimicActorCfg,
    RslRlSuperMimicAlgorithmCfg,
    RslRlSuperMimicRunnerCfg,
    RslRlSuperMimicTaskCfg,
    RslRlWorldModelCfg,
)


@configclass
class MotionTrackingTaskCfg(RslRlSuperMimicTaskCfg):
    class_name = "isaaclab_tasks.core.motion_tracking.supermimic:MotionTrackingTask"

    def configure_env(self, env_cfg, runner_cfg, *, inference: bool = False):
        if env_cfg.observations.state is None:
            raise ValueError("SuperMimic requires presets=state_obs.")
        env_cfg.observations.labels.self_collisions.params = deepcopy(env_cfg.rewards.self_collisions.params)
        wm = runner_cfg.algorithm.world_model_cfg
        wm.replay_buffer_capacity = 1000 * env_cfg.scene.num_envs
        runner_cfg.algorithm.updates_per_step = (
            env_cfg.scene.num_envs * runner_cfg.replay_ratio / (wm.num_updates * wm.batch_size * (wm.window_length + 1))
        )


@configclass
class G1MotionSuperMimicRunnerCfg(RslRlSuperMimicRunnerCfg):
    class_name = "OffPolicyRunner"
    experiment_name = "g1_motion_tracking"
    num_steps_per_env = 1
    init_at_random_ep_len = False
    max_iterations = 1_000_000
    save_interval = 1000
    separate_goal_terms = ("command",)
    replay_ratio: int = 16
    obs_groups = {
        "actor": ["policy", "actor_goal"],
        "state": ["state"],
        "goal": ["goal"],
        "context": ["context"],
        "labels": ["labels"],
    }
    actor = RslRlSuperMimicActorCfg(activation="gelu", norm="layer", output_gain=1.0)
    critic = RslRlResidualMLPModelCfg(hidden_dims=[256, 128], activation="gelu", obs_normalization=False)
    algorithm = RslRlSuperMimicAlgorithmCfg(
        task_cfg=MotionTrackingTaskCfg(),
        history_length=1,
        update_schedule="from_start",
        normalize_before_action=False,
        use_cuda_graph=True,
        max_step_reward=0.1,
        final_learning_rate_ratio=1.0,
        critic_survival_weighting=False,
        critic_bootstrap_clipping=False,
        critic_shuffle_each_epoch=False,
        muon_models=("actor", "world_model"),
        world_model_cfg=RslRlLearnedDynamicsCfg(
            model=RslRlWorldModelCfg(
                activation="gelu", normalize_output=False, normalize_context=False, zero_initialize_output=False
            ),
            num_warmup_steps=8,
            num_warmup_iterations=0,
            window_length=7,
            batch_size=2048,
            loss_reduction="mean",
            full_windows=True,
        ),
    )

    def __post_init__(self):
        self.critic.network_cfg = {
            "class_name": "MLP",
            "hidden_dims": [256, 128],
            "activation": "gelu",
            "norm": "layer",
        }
