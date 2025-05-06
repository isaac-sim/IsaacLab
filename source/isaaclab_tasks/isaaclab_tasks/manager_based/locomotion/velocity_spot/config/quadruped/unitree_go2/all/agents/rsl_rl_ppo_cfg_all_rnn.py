# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from .rsl_rl_ppo_cfg_pos import UnitreeGo2RoughPPORunnerCfg


@configclass
class UnitreeGo2RoughPPORunnerCfg_all_rnn(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.policy.actor_hidden_dims = [256, 256, 256]   
        self.policy.critic_hidden_dims = [256, 256, 256]  

        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.activation = "elu"

        # self.algorithm.entropy_coef = 0

        self.max_iterations = 100_000_000
        self.save_interval = 500
        self.experiment_name = "go2_velocity_rough_spot_all_rnn"


@configclass
class UnitreeGo2FlatPPORunnerCfg_all_rnn(UnitreeGo2RoughPPORunnerCfg_all_rnn):
    def __post_init__(self):
        super().__post_init__()

        # self.algorithm.entropy_coef = 0

        self.max_iterations = 100_000_000
        self.save_interval = 500
        self.experiment_name = "go2_velocity_flat_spot_all_rnn"
