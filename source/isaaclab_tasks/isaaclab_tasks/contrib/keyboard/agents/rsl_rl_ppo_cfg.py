# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

##
# Custom model configurations.
##


@configclass
class SharedEncoderMLPModelCfg(RslRlMLPModelCfg):
    """Configuration for the shared-encoder MLP model."""

    class_name: str = "isaaclab_tasks.contrib.keyboard.agents.models:SharedEncoderMLPModel"
    """The model class name. Defaults to :class:`~.models.SharedEncoderMLPModel`."""

    @configclass
    class EncoderCfg:
        """Configuration for the MLP encoder of a single observation group."""

        hidden_dims: list[int] = MISSING
        """The hidden dimensions of the encoder MLP."""

        latent_dim: int = MISSING
        """The dimension of the encoder output latent."""

        activation: str = "elu"
        """The activation function of the encoder MLP. Defaults to elu."""

        last_activation: str | None = "elu"
        """The activation applied to the encoder output latent. Defaults to elu.

        If None, the latent is the output of the last linear layer.
        """

    encoder_cfg: dict[str, EncoderCfg] = MISSING
    """Mapping from observation group name to the MLP encoder configuration for that group."""


##
# Runner configuration.
##


@configclass
class SO101PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL configuration for SO-101 keyboard typing with shared observation encoders."""

    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 250
    experiment_name = "so101_keyboard_typing"
    obs_groups = {
        "actor": ["policy", "proprio", "perception"],
        "critic": ["policy", "proprio", "perception"],
    }
    actor = SharedEncoderMLPModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        obs_normalization=True,
        hidden_dims=[512, 256, 128],
        activation="elu",
        encoder_cfg={"policy": SharedEncoderMLPModelCfg.EncoderCfg(hidden_dims=[256], latent_dim=64)},
    )
    critic = SharedEncoderMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128],
        activation="elu",
        encoder_cfg={"policy": SharedEncoderMLPModelCfg.EncoderCfg(hidden_dims=[256], latent_dim=64)},
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="isaaclab_tasks.contrib.keyboard.agents.models:SharedEncoderPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.90,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
