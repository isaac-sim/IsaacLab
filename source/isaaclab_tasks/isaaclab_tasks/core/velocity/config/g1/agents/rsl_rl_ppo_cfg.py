# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class G1RoughTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 250
    experiment_name = "g1_rough_teacher"
    obs_groups = {"actor": ["proprio", "height_scan"], "critic": ["proprio", "height_scan"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func="isaaclab_tasks.core.velocity.mdp.symmetry.g1:compute_symmetric_states",
            mirror_loss_coeff=0.0,
        ),
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughTeacherPPORunnerCfg):
    max_iterations = 1500
    save_interval = 50
    experiment_name = "g1_flat"
    obs_groups = {"actor": ["proprio"], "critic": ["proprio"]}

    def __post_init__(self):
        super().__post_init__()
        self.actor.hidden_dims = [256, 128, 128]
        self.critic.hidden_dims = [256, 128, 128]
        self.algorithm.symmetry_cfg = None


@configclass
class G1RoughDepthStudentRunnerCfg(RslRlDistillationRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 24000
    save_interval = 250
    experiment_name = "g1_rough_depth_student"
    obs_groups = {"student": ["proprio_history", "depth"], "teacher": ["proprio", "height_scan"]}
    student = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            padding="zeros",
            activation="relu",
            flatten=True,
        ),
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="isaaclab_tasks.core.velocity.config.g1.agents.distillation:MirrorDistillation",
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
