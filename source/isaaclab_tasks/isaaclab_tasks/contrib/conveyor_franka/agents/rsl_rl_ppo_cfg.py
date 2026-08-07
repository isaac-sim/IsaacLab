# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for conveyor transfer."""

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Bernoulli, Normal

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


class _ConveyorDeterministicOutput(nn.Module):
    """Threshold the gripper logit into the action used by the environment."""

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        gripper = torch.where(
            output[..., -1:] >= 0.0,
            torch.ones_like(output[..., -1:]),
            -torch.ones_like(output[..., -1:]),
        )
        return torch.cat((output[..., :-1], gripper), dim=-1)


class ConveyorGaussianBernoulliDistribution(GaussianDistribution):
    """Use Gaussian exploration for the arm and a Bernoulli gripper.

    The final policy output controls a physically binary action. Sampling it
    from a Gaussian would assign different log probabilities to values that
    become the same open or close command after thresholding. A Bernoulli
    instead optimizes exactly the physical decision seen by the environment.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.45,
        std_range: tuple[float, float] = (0.15, 0.65),
        std_type: str = "scalar",
        **kwargs,
    ) -> None:
        if output_dim < 2:
            raise ValueError("The conveyor distribution requires arm outputs followed by one gripper output.")
        if len(std_range) != 2 or std_range[0] <= 0.0 or std_range[0] >= std_range[1]:
            raise ValueError("std_range must contain positive, increasing bounds.")
        if not std_range[0] < init_std < std_range[1]:
            raise ValueError("init_std must lie strictly inside std_range.")
        if std_type != "scalar":
            raise ValueError("The conveyor distribution supports only scalar standard-deviation parameters.")
        super().__init__(output_dim, init_std=init_std, std_type=std_type, **kwargs)
        self.std_range = (float(std_range[0]), float(std_range[1]))
        self._arm_distribution: Normal | None = None
        self._gripper_distribution: Bernoulli | None = None
        initial_fraction = (init_std - self.std_range[0]) / (self.std_range[1] - self.std_range[0])
        initial_logit = torch.logit(torch.tensor(initial_fraction, dtype=self.std_param.dtype))
        with torch.no_grad():
            self.std_param.fill_(initial_logit)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update continuous-arm and binary-gripper distributions."""
        minimum_std, maximum_std = self.std_range
        arm_std = minimum_std + (maximum_std - minimum_std) * torch.sigmoid(self.std_param[:-1])
        self._arm_distribution = Normal(mlp_output[..., :-1], arm_std)
        self._gripper_distribution = Bernoulli(logits=mlp_output[..., -1:])

    def sample(self) -> torch.Tensor:
        """Sample seven residuals followed by an exact signed binary action."""
        gripper_open = self._gripper_distribution.sample()
        return torch.cat((self._arm_distribution.sample(), 2.0 * gripper_open - 1.0), dim=-1)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return arm means and the most likely binary gripper command."""
        return _ConveyorDeterministicOutput()(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an exportable deterministic-output transform."""
        return _ConveyorDeterministicOutput()

    @property
    def mean(self) -> torch.Tensor:
        """Return arm means and the expected signed gripper command."""
        gripper_mean = 2.0 * self._gripper_distribution.probs - 1.0
        return torch.cat((self._arm_distribution.mean, gripper_mean), dim=-1)

    @property
    def std(self) -> torch.Tensor:
        """Return arm standard deviations and signed-Bernoulli spread."""
        gripper_std = 2.0 * torch.sqrt(self._gripper_distribution.probs * (1.0 - self._gripper_distribution.probs))
        return torch.cat((self._arm_distribution.stddev, gripper_std), dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        """Return joint Gaussian-plus-Bernoulli entropy."""
        return self._arm_distribution.entropy().sum(dim=-1) + self._gripper_distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return parameters needed to evaluate the mixed KL divergence."""
        return self._arm_distribution.mean, self._arm_distribution.stddev, self._gripper_distribution.logits

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the exact continuous/binary physical action."""
        arm_log_prob = self._arm_distribution.log_prob(outputs[..., :-1]).sum(dim=-1)
        gripper_open = (outputs[..., -1:] >= 0.0).to(outputs.dtype)
        return arm_log_prob + self._gripper_distribution.log_prob(gripper_open).sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return ``KL(old || new)`` for both action families."""
        old_arm_mean, old_arm_std, old_gripper_logits = old_params
        new_arm_mean, new_arm_std, new_gripper_logits = new_params
        arm_kl = torch.distributions.kl_divergence(
            Normal(old_arm_mean, old_arm_std),
            Normal(new_arm_mean, new_arm_std),
        ).sum(dim=-1)
        old_gripper_probability = torch.sigmoid(old_gripper_logits)
        gripper_kl = (
            old_gripper_probability * (old_gripper_logits - new_gripper_logits)
            - torch.nn.functional.softplus(old_gripper_logits)
            + torch.nn.functional.softplus(new_gripper_logits)
        ).sum(dim=-1)
        return arm_kl + gripper_kl.clamp_min(0.0)


@configclass
class ConveyorGaussianBernoulliDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Bounded Gaussian arm exploration with a Bernoulli gripper."""

    class_name: str = (
        "isaaclab_tasks.contrib.conveyor_franka.agents.rsl_rl_ppo_cfg:ConveyorGaussianBernoulliDistribution"
    )
    std_range: tuple[float, float] = (0.15, 0.65)


@configclass
class ConveyorFrankaPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for four-cube commanded transfer."""

    num_steps_per_env = 32
    max_iterations = 4000
    save_interval = 50
    experiment_name = "conveyor_franka_transfer"
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=ConveyorGaussianBernoulliDistributionCfg(init_std=0.45),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1.0e-4,
        schedule="fixed",
        # At 60 Hz, the ten-second pickup-to-placement horizon needs the same
        # long-horizon discount used by the reset-driven Franka stack task.
        gamma=0.999,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
