# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic settings for G1 motion tracking with the state_obs observation preset.

The training settings follow BlinkMimic (arXiv 2611.16422): Muon, a raised actor learning rate that decays after
1400 learner iterations, a policy that corrects the reference pose, mirrored world-model batches, and no
world-model-only warm-up.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlSuperMimicActorCfg,
    RslRlSuperMimicAlgorithmCfg,
    RslRlSuperMimicRunnerCfg,
    RslRlSuperMimicTaskCfg,
    RslRlWorldModelCfg,
)


@configclass
class G1MotionSuperMimicRunnerCfg(RslRlSuperMimicRunnerCfg):
    """Collect one physical step before applying fractional learner updates."""

    num_steps_per_env = 1
    max_iterations = 36_000
    save_interval = 1200
    experiment_name = "g1_motion_tracking_supermimic"
    clip_actions = None
    obs_groups = {
        "state": ["state"],
        "goal": ["goal"],
        "context": ["context"],
        "labels": ["labels"],
    }
    actor = RslRlSuperMimicActorCfg(use_reference_action=True)
    world_model = RslRlWorldModelCfg(hidden_dims=(1024, 1024, 1024))
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlSuperMimicAlgorithmCfg(
        replay_buffer_capacity=4_096_000,
        num_warmup_steps=64,
        num_world_model_warmup_iterations=0,
        num_world_model_updates=5,
        world_model_mirror_fraction=0.5,
        policy_batch_size=2048,
        action_noise=0.0,
        learning_rate=4e-4,
        muon_models=("actor", "world_model"),
        actor_learning_rate_multiplier=4.0,
        actor_learning_rate_decay_iterations=300,
        # episodes end with their clip, well before max_episode_length, so a normalized-return trigger never fires
        actor_learning_rate_decay_start=1400,
        # an exponential moving average with decay 0.995 per actor update
        actor_ema_halflife=138,
        use_mixed_precision=True,
        use_cuda_graph=True,
        # dt 0.02 * (anchor position 0.5 + anchor orientation 0.5 + four body tracking terms 1).
        max_step_reward=0.1,
        task_cfg=RslRlSuperMimicTaskCfg(
            class_name="isaaclab_tasks.core.motion_tracking.config.g1.supermimic:G1MotionTrackingTask"
        ),
    )
