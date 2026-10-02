# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking rsl_rl_ppo_cfg."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.utils import preset

from ..flat_env_cfg import MICRODUCK_STEPS_PER_ITERATION


@configclass
class MicroDuckPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the MicroDuck velocity-tracking tasks."""

    num_steps_per_env = MICRODUCK_STEPS_PER_ITERATION
    max_iterations = 50000
    save_interval = 250
    experiment_name = preset(default="microduck_velocity_flat", backlash="microduck_velocity_flat_backlash")
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MicroDuckRoughPPORunnerCfg(MicroDuckPPORunnerCfg):
    """The same PPO recipe with a separate experiment directory for rough terrain."""

    experiment_name = preset(default="microduck_velocity_rough", backlash="microduck_velocity_rough_backlash")


@configclass
class MicroDuckRecoveryPPORunnerCfg(MicroDuckPPORunnerCfg):
    """Single-policy PPO from scratch, with a longer horizon for getting up."""

    experiment_name = "microduck_recovery_velocity_backlash"

    def __post_init__(self):
        self.algorithm.gamma = 0.995
        self.algorithm.num_mini_batches = 16
