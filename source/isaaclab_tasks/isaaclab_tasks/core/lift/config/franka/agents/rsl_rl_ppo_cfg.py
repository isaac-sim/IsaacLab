# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, dataclass
from typing import Any

from isaaclab.utils import config_field, replace_config

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab_tasks.utils import PresetCfg

STATE_POLICY_CFG = RslRlMLPModelCfg(
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    activation="elu",
)

STATE_CRITIC_CFG = RslRlMLPModelCfg(
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    activation="elu",
)


ALGO_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.995,
    lam=0.90,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@dataclass
class FrankaPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: Any = config_field(32)
    max_iterations: Any = config_field(15000)
    save_interval: Any = config_field(250)
    experiment_name: Any = config_field((MISSING,))  # type: ignore
    obs_groups: Any = config_field((MISSING,))  # type: ignore
    actor: Any = config_field((MISSING,))  # type: ignore
    critic: Any = config_field((MISSING,))  # type: ignore
    algorithm: Any = config_field(MISSING)  # type: ignore


@dataclass
class FrankaPPORunnerCfg(PresetCfg):
    default: Any = config_field(
        replace_config(
            FrankaPPOBaseRunnerCfg(),
            experiment_name="lift_franka",
            obs_groups={"actor": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]},
            actor=STATE_POLICY_CFG,
            critic=STATE_CRITIC_CFG,
            algorithm=ALGO_CFG,
        )
    )
