# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class TaskHeadedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Configuration for independently explored task action heads."""

    class_name: str = "isaaclab_tasks.contrib.multitask_manipulation.agents.models:TaskHeadedGaussianDistribution"
    """The qualified task-headed distribution class name."""


@configclass
class TaskHeadedMLPModelCfg(RslRlMLPModelCfg):
    """Configuration for the shared backbone and task-specific action heads."""

    class_name: str = "isaaclab_tasks.contrib.multitask_manipulation.agents.models:TaskHeadedMLPModel"
    """The qualified task-headed model class name."""

    task_action_dims: tuple[int, ...] = (8, 8, 6)
    """Action dimensions ordered as lift, cabinet, and reach."""

    task_encoding_slice: tuple[int, int] = (0, 3)
    """Half-open policy observation slice containing the task one-hot."""


@configclass
class TaskHeadedValueModelCfg(RslRlMLPModelCfg):
    """Configuration for the shared backbone and task-specific value heads."""

    class_name: str = "isaaclab_tasks.contrib.multitask_manipulation.agents.models:TaskHeadedValueModel"
    """The qualified task-headed value model class name."""

    task_head_count: int = 3
    """Number of scalar value heads."""

    task_encoding_slice: tuple[int, int] = (0, 3)
    """Half-open policy observation slice containing the task one-hot."""


@configclass
class TaskBalancedPPOCfg(RslRlPpoAlgorithmCfg):
    """Configuration for task-wise rollout advantage normalization."""

    class_name: str = "isaaclab_tasks.contrib.multitask_manipulation.agents.ppo:TaskBalancedPPO"
    """The qualified task-balanced PPO class name."""

    task_names: tuple[str, ...] = ("lift", "cabinet", "reach")
    """Task names ordered by the policy observation one-hot."""

    task_encoding_obs_group: str = "policy"
    """Observation group containing the task one-hot."""

    task_encoding_slice: tuple[int, int] = (0, 3)
    """Half-open observation slice containing the task one-hot."""


@configclass
class MultitaskManipulationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for the heterogeneous manipulation task."""

    num_steps_per_env = 16
    init_at_random_ep_len = False
    max_iterations = 2000
    save_interval = 100
    experiment_name = "multitask_manipulation"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = TaskHeadedMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=TaskHeadedGaussianDistributionCfg(init_std=1.0),
    )
    critic = TaskHeadedValueModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = TaskBalancedPPOCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
