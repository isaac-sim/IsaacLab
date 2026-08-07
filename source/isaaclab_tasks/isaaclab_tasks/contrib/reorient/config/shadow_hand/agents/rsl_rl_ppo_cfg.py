# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configurations for the OpenAI Shadow Hand variant.

The feed-forward and recurrent policies differ only in the actor and critic models, so
both derive from a shared base rather than from each other. Select the recurrent one with
``--agent rsl_rl_lstm_cfg_entry_point``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg


@configclass
class ShadowHandAsymPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Everything the feed-forward and recurrent variants agree on.

    They differ only in the actor and critic models, so neither derives from the other.
    """

    num_steps_per_env = 16
    max_iterations = 10000
    save_interval = 250
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ShadowHandAsymFFPPORunnerCfg(ShadowHandAsymPPOBaseRunnerCfg):
    """RSL-RL feed-forward policy configuration for the asymmetric OpenAI observations."""

    experiment_name = "shadow_hand_openai_ff"
    actor = RslRlMLPModelCfg(
        hidden_dims=[400, 400, 200, 100],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )


@configclass
class ShadowHandAsymLSTMPPORunnerCfg(ShadowHandAsymPPOBaseRunnerCfg):
    """RSL-RL recurrent policy configuration for the asymmetric OpenAI observations."""

    experiment_name = "shadow_hand_openai_lstm"
    actor = RslRlRNNModelCfg(
        hidden_dims=[400, 400, 200, 100],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[512, 512, 256, 128],
        activation="elu",
        obs_normalization=True,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
