# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
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

# Camera-actor stability requires bounding the encoder update magnitude
# (learning_rate x latent size x input scale <~ 0.05, and LR <~ 1.5e-4): larger encoders or
# hotter/adaptive learning rates collapse into velocity-limit terminations or freezing.
# One shared configuration for single and duo cameras: 512-dim latent per camera at fixed
# LR 5e-5 sits under the stability budget in both rigs (duo doubles the latent mass), trading
# a few hundred iterations of onset in the single-camera case for a uniform, sub-band setup.
CNN_POLICY_CFG = RslRlCNNModelCfg(
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
    cnn_cfg=RslRlCNNModelCfg.CNNCfg(
        output_channels=[16, 32, 32],
        kernel_size=[8, 4, 3],
        stride=[4, 2, 1],
        activation="elu",
    ),
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
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class DexsuiteKukaAllegroPPOBaseRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 250
    experiment_name = (MISSING,)  # type: ignore
    obs_groups = (MISSING,)  # type: ignore
    actor = (MISSING,)  # type: ignore
    critic = (MISSING,)  # type: ignore
    algorithm = MISSING  # type: ignore


@configclass
class DexsuiteKukaAllegroPPORunnerCfg(PresetCfg):
    default = DexsuiteKukaAllegroPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_kuka_allegro",
        obs_groups={"actor": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]},
        actor=STATE_POLICY_CFG,
        critic=STATE_CRITIC_CFG,
        algorithm=ALGO_CFG,
    )

    single_camera = DexsuiteKukaAllegroPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_kuka_allegro_single_camera",
        obs_groups={"actor": ["policy", "proprio", "base_image"], "critic": ["policy", "proprio", "perception"]},
        actor=CNN_POLICY_CFG,
        critic=STATE_CRITIC_CFG,
        # fixed LR: the adaptive-KL schedule oscillates across the encoder stability threshold
        # (its KL signal is inflated by encoder feature churn) and never trains a camera actor
        algorithm=ALGO_CFG.replace(num_mini_batches=8, schedule="fixed", learning_rate=5.0e-5),
    )

    duo_camera = DexsuiteKukaAllegroPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_kuka_allegro_duo_camera",
        obs_groups={
            "actor": ["policy", "proprio", "base_image", "wrist_image"],
            "critic": ["policy", "proprio", "perception"],
        },
        actor=CNN_POLICY_CFG,
        critic=STATE_CRITIC_CFG,
        algorithm=ALGO_CFG.replace(num_mini_batches=8, schedule="fixed", learning_rate=5.0e-5),
    )
