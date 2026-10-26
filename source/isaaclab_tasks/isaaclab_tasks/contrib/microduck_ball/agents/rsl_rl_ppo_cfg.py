# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configurations of the MicroDuck on a ball."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class MicroDuckBallPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner of the MicroDuck on a ball. The critic also reads the privileged observation group."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "microduck_ball"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MicroDuckBallFixedPPORunnerCfg(MicroDuckBallPPORunnerCfg):
    """PPO runner of the fixed-ball stage."""

    max_iterations = 400
    experiment_name = "microduck_ball_fixed"


@configclass
class MicroDuckBallBalancePPORunnerCfg(MicroDuckBallPPORunnerCfg):
    """PPO runner of the free-ball stage."""

    experiment_name = "microduck_ball_balance"


@configclass
class MicroDuckBallRollPPORunnerCfg(MicroDuckBallPPORunnerCfg):
    """PPO runner of the velocity-command stage, with more exploration to find the steps."""

    experiment_name = "microduck_ball_roll"

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.01


@configclass
class MicroDuckBallRobustPPORunnerCfg(MicroDuckBallPPORunnerCfg):
    """PPO runner of the robustness stage."""

    experiment_name = "microduck_ball_robust"
