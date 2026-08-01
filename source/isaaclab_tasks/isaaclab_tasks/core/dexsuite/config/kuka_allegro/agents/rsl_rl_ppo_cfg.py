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


@configclass
class RslRlSpatialSoftmaxCNNModelCfg(RslRlCNNModelCfg):
    """Configuration for the dexsuite spatial-softmax camera actor.

    The convolutional feature map is reduced to per-channel keypoint coordinates instead of being
    flattened, which shrinks the latent by roughly an order of magnitude while keeping the spatial
    information a pixels-only policy depends on. See :class:`.models.SpatialSoftmaxCNNModel`.
    """

    class_name: str = "isaaclab_tasks.core.dexsuite.config.kuka_allegro.agents.models:SpatialSoftmaxCNNModel"
    """The model class name resolved by rsl-rl."""

    init_temperature: float = 1.0
    """Initial softmax temperature of the keypoint layer. Defaults to 1.0."""


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

# Camera actor shared by both rigs. The encoder reduces its feature map to per-channel keypoint
# coordinates rather than flattening it, so the latent handed to the MLP stays at two numbers per
# channel regardless of the feature-map size.
CNN_POLICY_CFG = RslRlSpatialSoftmaxCNNModelCfg(
    obs_normalization=True,
    hidden_dims=[512, 256, 128],
    distribution_cfg=RslRlSpatialSoftmaxCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
    cnn_cfg=RslRlSpatialSoftmaxCNNModelCfg.CNNCfg(
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


# Camera actors need a fixed learning rate: the adaptive KL schedule varies it across the range
# where encoder features change faster than the policy head can track, and does not converge.
CAMERA_ALGO_CFG = ALGO_CFG.replace(num_mini_batches=8, schedule="fixed", learning_rate=7.0e-5)


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
        algorithm=CAMERA_ALGO_CFG,
    )

    duo_camera = DexsuiteKukaAllegroPPOBaseRunnerCfg().replace(
        experiment_name="dexsuite_kuka_allegro_duo_camera",
        obs_groups={
            "actor": ["policy", "proprio", "base_image", "wrist_image"],
            "critic": ["policy", "proprio", "perception"],
        },
        actor=CNN_POLICY_CFG,
        critic=STATE_CRITIC_CFG,
        algorithm=CAMERA_ALGO_CFG,
    )
