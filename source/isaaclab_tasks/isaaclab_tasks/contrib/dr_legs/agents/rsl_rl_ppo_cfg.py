# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import PresetCfg


@configclass
class DrLegsHoldPosePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    clip_actions = 1.0
    experiment_name = "dr_legs_hold_pose"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 512],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 512],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def _walk_runner_cfg(init_std: float) -> DrLegsHoldPosePPORunnerCfg:
    """Return a DR Legs walking runner configuration."""
    cfg = DrLegsHoldPosePPORunnerCfg()
    cfg.experiment_name = "dr_legs_walk"
    cfg.actor.distribution_cfg.init_std = init_std
    return cfg


@configclass
class DrLegsWalkPPORunnerCfg(PresetCfg):
    """Backend-specific DR Legs walking runner configurations."""

    default = _walk_runner_cfg(init_std=0.2)
    newton_kamino = default
    physx = _walk_runner_cfg(init_std=0.5)
    isaacsim_physx = physx
