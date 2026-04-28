# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DexsuiteKukaAllegroPPOResNetRunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner config for Kuka Allegro with ResNet18 features.

    RSL-RL PPO runner configuration for DexSuite Kuka Allegro with frozen ResNet18
    feature extraction. The ResNet backbone runs at the observation level (producing
    512-dim features), so the policy MLP only processes compact feature vectors.

    The ResNet18 model uses ImageNet pretrained weights and is automatically downloaded
    and cached by torchvision on first use.

    Observation groups:
        - actor: ["policy", "proprio", "resnet_features"] — 512-dim frozen ResNet features
        - critic: ["policy", "proprio", "resnet_features"] — same as actor (shared obs)

    Note:
        The policy network uses a smaller MLP (256, 128, 64) since ResNet already provides
        rich 512-dimensional features, compared to the CNN variant which processes raw images.
    """

    num_steps_per_env = 32
    max_iterations = 17000
    save_interval = 250
    experiment_name = "dexsuite_kuka_allegro_resnet_features"
    obs_groups = {"actor": ["policy", "proprio", "resnet_features"], "critic": ["policy", "proprio", "resnet_features"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        # ResNet18 outputs 512-dim features, so we can use a smaller first layer
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
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
