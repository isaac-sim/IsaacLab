# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for Kuka-Allegro cube stacking."""

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from ...franka.agents.rsl_rl_ppo_cfg import FrankaStackPPORunnerCfg


class KukaAllegroGaussianDistribution(GaussianDistribution):
    """Bounded all-continuous exploration with separate arm and hand priors."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.35,
        hand_init_std: float = 0.15,
        arm_action_dim: int = 7,
        std_range: tuple[float, float] = (0.08, 0.45),
        std_type: str = "scalar",
        **kwargs,
    ) -> None:
        if not 0 < arm_action_dim < output_dim:
            raise ValueError("arm_action_dim must split the continuous arm and hand actions.")
        if len(std_range) != 2 or std_range[0] <= 0.0 or std_range[0] >= std_range[1]:
            raise ValueError("std_range must contain positive, increasing bounds.")
        if not std_range[0] < init_std < std_range[1]:
            raise ValueError("init_std must lie strictly inside std_range.")
        if not std_range[0] < hand_init_std < std_range[1]:
            raise ValueError("hand_init_std must lie strictly inside std_range.")
        if std_type != "scalar":
            raise ValueError("KukaAllegroGaussianDistribution supports scalar standard deviations.")
        super().__init__(
            output_dim,
            init_std=init_std,
            std_range=std_range,
            std_type=std_type,
            **kwargs,
        )
        self._bounded_std_range = (float(std_range[0]), float(std_range[1]))
        initial_std = torch.full_like(self.std_param, hand_init_std)
        initial_std[:arm_action_dim] = init_std
        initial_fraction = (initial_std - std_range[0]) / (std_range[1] - std_range[0])
        with torch.no_grad():
            self.std_param.copy_(torch.logit(initial_fraction))

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update a diagonal Gaussian without hard-clamp gradient dead zones."""
        minimum_std, maximum_std = self._bounded_std_range
        std = minimum_std + (maximum_std - minimum_std) * torch.sigmoid(self.std_param)
        self._distribution = Normal(mlp_output, std)


@configclass
class KukaAllegroGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Configuration for bounded 7-arm plus 16-hand exploration."""

    class_name: str = (
        "isaaclab_tasks.contrib.stack.config.kuka_allegro.agents.rsl_rl_ppo_cfg:KukaAllegroGaussianDistribution"
    )
    hand_init_std: float = 0.15
    arm_action_dim: int = 7
    std_range: tuple[float, float] = (0.08, 0.45)


@configclass
class KukaAllegroStackPPORunnerCfg(FrankaStackPPORunnerCfg):
    """PPO configuration for the 23-action KUKA-Allegro stack task."""

    experiment_name = "kuka_allegro_stack_full_hand"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=KukaAllegroGaussianDistributionCfg(init_std=0.35),
    )

    def __post_init__(self) -> None:
        # Entropy sums over all 23 Gaussian dimensions. Reusing the 8-action
        # coefficient drives every standard deviation to its upper bound and
        # makes the clipped physical action differ from PPO's sampled action.
        self.algorithm.entropy_coef = 1.0e-4
        # Full-hand production checkpoints use this fixed optimizer rate.
        # Keep it in the task config so training and local evaluation resolve
        # the same optimizer contract without a launcher override.
        self.algorithm.learning_rate = 5.0e-5
