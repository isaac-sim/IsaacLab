# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


class UR10ParticlePushGaussianDistribution(GaussianDistribution):
    """State-independent joint exploration with smooth, finite standard-deviation bounds."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.60,
        std_range: tuple[float, float] = (0.15, 0.65),
        std_type: str = "scalar",
        **kwargs,
    ) -> None:
        if len(std_range) != 2 or std_range[0] <= 0.0 or std_range[0] >= std_range[1]:
            raise ValueError("std_range must contain positive, increasing lower and upper bounds.")
        if not std_range[0] < init_std < std_range[1]:
            raise ValueError("init_std must lie strictly inside std_range.")
        if std_type != "scalar":
            raise ValueError("UR10ParticlePushGaussianDistribution supports only scalar standard deviations.")
        super().__init__(output_dim, init_std=init_std, std_type=std_type, **kwargs)
        self.std_range = (float(std_range[0]), float(std_range[1]))
        initial_fraction = (init_std - self.std_range[0]) / (self.std_range[1] - self.std_range[0])
        initial_logit = torch.logit(torch.tensor(initial_fraction, dtype=self.std_param.dtype))
        with torch.no_grad():
            self.std_param.fill_(initial_logit)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian while preserving gradients near both exploration bounds."""
        minimum_std, maximum_std = self.std_range
        std = minimum_std + (maximum_std - minimum_std) * torch.sigmoid(self.std_param)
        self._distribution = Normal(mlp_output, std)


@configclass
class UR10ParticlePushGaussianDistributionCfg(RslRlCNNModelCfg.GaussianDistributionCfg):
    """Smoothly bounded, per-joint exploration used by the push policy."""

    class_name: str = (
        "isaaclab_tasks.contrib.ur10_particle_push.agents.rsl_rl_ppo_cfg:UR10ParticlePushGaussianDistribution"
    )
    init_std: float = 0.60
    std_range: tuple[float, float] = (0.15, 0.65)


@configclass
class UR10ParticlePushPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO defaults for the deployable heightmap-and-proprioception policy."""

    # 72 policy steps span 1.2 s at 60 Hz.
    num_steps_per_env = 72
    max_iterations = 6000
    save_interval = 50
    experiment_name = "ur10_particle_push"
    run_name = "single_pile_push"
    # Keep the task usable offline by default. Distributed launch profiles select W&B explicitly
    # with ``--logger wandb`` and provide the project and credentials.
    logger = "tensorboard"
    wandb_project = "ur10-particle-push-mpm"
    clip_actions = 1.0
    obs_groups = {
        "actor": ["policy", "heightmap"],
        "critic": ["policy", "heightmap", "critic"],
    }
    actor = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[8, 16, 32, 32],
            kernel_size=[5, 3, 3, 3],
            stride=[2, 2, 2, 2],
            padding="zeros",
            activation="elu",
        ),
        hidden_dims=[256, 128, 64],
        activation="elu",
        # Both image and proprioceptive inputs are bounded at their source. Rank-local empirical
        # normalizers would otherwise make distributed checkpoints depend on the saved rank.
        obs_normalization=False,
        distribution_cfg=UR10ParticlePushGaussianDistributionCfg(),
    )
    critic = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[8, 16, 32, 32],
            kernel_size=[5, 3, 3, 3],
            stride=[2, 2, 2, 2],
            padding="zeros",
            activation="elu",
        ),
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=1.0e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.5e-4,
        schedule="fixed",
        gamma=0.99 ** (1.0 / 6.0),
        lam=0.95 ** (1.0 / 6.0),
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Keep actor and critic encoders independent: RSL-RL currently inserts shared
        # parameters twice into PPO's optimizer, producing duplicate updates and warnings.
        share_cnn_encoders=False,
    )
