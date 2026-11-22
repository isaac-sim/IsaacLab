# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic settings for G1 motion tracking with the state_obs observation preset."""

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
    actor = RslRlSuperMimicActorCfg()
    world_model = RslRlWorldModelCfg()
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlSuperMimicAlgorithmCfg(
        replay_buffer_capacity=4_096_000,
        use_cuda_graph=True,
        # dt 0.02 * (anchor position 0.5 + anchor orientation 0.5 + four body tracking terms 1).
        max_step_reward=0.1,
        task_cfg=RslRlSuperMimicTaskCfg(class_name="isaaclab_tasks.core.motion_tracking.supermimic:MotionTrackingTask"),
    )
