# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for bimanual cable routing."""

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class SharedEncoderMLPModelCfg(RslRlMLPModelCfg):
    """Configuration for an MLP with encoded observation groups."""

    class_name: str = "isaaclab_tasks.contrib.cable_routing.agents.models:SharedEncoderMLPModel"
    """Model class resolved by RSL-RL."""

    @configclass
    class EncoderCfg:
        """Configuration for one observation-group encoder."""

        hidden_dims: list[int] = MISSING
        """Hidden dimensions of the encoder MLP."""

        latent_dim: int = MISSING
        """Dimension of the encoder output."""

        activation: str = "elu"
        """Activation function of the encoder MLP."""

        last_activation: str | None = "elu"
        """Activation applied to the encoder output, or ``None`` for a linear output."""

    encoder_cfg: dict[str, EncoderCfg] = MISSING
    """Mapping from observation-group names to encoder configurations."""


@configclass
class CableRoutingGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian distribution configuration with task-specific standard-deviation bounds."""

    class_name: str = "isaaclab_tasks.contrib.cable_routing.agents.models:BoundedGaussianDistribution"
    """Task-owned distribution that supports both release-image and current RSL-RL versions."""

    std_range: tuple[float, float] = MISSING
    """Minimum and maximum effective and learnable standard deviation."""


@configclass
class CableRoutingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for the goal-conditioned bimanual cable-routing policy."""

    num_steps_per_env = 36
    init_at_random_ep_len = False
    max_iterations = 15000
    save_interval = 250
    experiment_name = "yam_cable_routing"
    clip_actions = 1.0
    obs_groups = {
        "actor": ["policy", "proprio", "cable_state", "goal"],
        "critic": ["policy", "proprio", "cable_state", "goal"],
    }
    actor = SharedEncoderMLPModelCfg(
        distribution_cfg=CableRoutingGaussianDistributionCfg(
            init_std=0.25,
            std_type="log",
            std_range=(0.02, 0.5),
        ),
        obs_normalization=True,
        hidden_dims=[512, 256, 128],
        activation="elu",
        encoder_cfg={
            "goal": SharedEncoderMLPModelCfg.EncoderCfg(hidden_dims=[64, 64], latent_dim=32),
        },
    )
    critic = SharedEncoderMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128],
        activation="elu",
        encoder_cfg={
            "goal": SharedEncoderMLPModelCfg.EncoderCfg(hidden_dims=[64, 64], latent_dim=32),
        },
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="isaaclab_tasks.contrib.cable_routing.agents.models:SharedEncoderPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.995,
        lam=0.90,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
