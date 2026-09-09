# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.torchrl import TorchRlPpoCfg


@configclass
class CartpolePPOCfg(TorchRlPpoCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole"
    actor_hidden_dims = [32, 32]
    critic_hidden_dims = [32, 32]
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    gamma = 0.99
    lam = 0.95
    entropy_coef = 0.005


@configclass
class CartpoleDirectPPOCfg(CartpolePPOCfg):
    experiment_name = "cartpole_direct"
