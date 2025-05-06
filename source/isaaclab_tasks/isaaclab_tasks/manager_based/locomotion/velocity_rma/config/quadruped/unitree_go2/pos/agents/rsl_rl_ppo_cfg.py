# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,  
    RslRlPpoActorCriticCfg,
    RslRlPpoCNNActorCriticCfg,
    RslRlPpoAlgorithmCfg, 
    RslRlDistillationCNNStudentTeacherCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg
)


@configclass
class Go2RoughPPORunnerCfg_Teacher(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 100_000_000
    save_interval = 500  
    experiment_name = "go2_velocity_rma_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128], 
        critic_hidden_dims=[128, 128, 128],  
        activation="elu"
    )
    # policy = RslRlPpoCNNActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[64, 256, 256], 
    #     critic_hidden_dims=[64, 256, 256],  
    #     activation="elu",
    #     cnn_kernel_sizes=[5, 5, 5, 5],
    #     cnn_strides=[1, 2, 2, 2],
    #     cnn_filters=[32, 32, 32, 16],
    #     cnn_paddings=[2, 2, 2, 2],
    #     cnn_dilations=[1, 1, 1, 1]
    # )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.01,  # 0.01  0.0025
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


@configclass
class Go2FlatPPORunnerCfg_Teacher(Go2RoughPPORunnerCfg_Teacher):
    def __post_init__(self):
        super().__post_init__()
    
        self.max_iterations = 100_000_000
        self.save_interval = 500
        self.experiment_name = "go2_velocity_rma_flat"


@configclass
class Go2RoughPPORunnerCfg_Policy(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 100_000_000
    save_interval = 500  
    experiment_name = "go2_velocity_rma_rough"
    empirical_normalization = False
    policy = RslRlDistillationCNNStudentTeacherCfg(
        init_noise_std=1.0,
        student_hidden_dims=[256, 256], 
        teacher_hidden_dims=[512, 256, 128], 
        activation="elu",
        student_cnn_kernel_size=3,
        student_cnn_stride=3,
        student_cnn_filters=[32, 16, 8],
        student_paddings=[0, 0, 1],
        teacher_cnn_kernel_size=3,
        teacher_cnn_stride=3,
        teacher_cnn_filters=[32, 16, 8],
        teacher_paddings=[0, 1, 1],
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=10,
        learning_rate=1e-4,
        gradient_length=1
    )


@configclass
class Go2FlatPPORunnerCfg_Policy(Go2RoughPPORunnerCfg_Policy):
    def __post_init__(self):
        super().__post_init__()
    
        self.max_iterations = 100_000_000
        self.save_interval = 500
        self.experiment_name = "go2_velocity_rma_flat"
