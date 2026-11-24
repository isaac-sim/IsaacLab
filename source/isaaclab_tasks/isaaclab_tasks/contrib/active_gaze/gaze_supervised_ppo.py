# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with gaze supervision by a teacher, for RSL-RL.

The update follows :meth:`rsl_rl.algorithms.PPO.update` (BSD-3-Clause) for feedforward models without RND or
symmetry, and adds the teacher's gaze loss to the PPO loss.
"""

from __future__ import annotations

import math

import torch
from rsl_rl.algorithms import PPO
from torch import nn

from .gaze_distribution import GAZE_ACTIONS


def gaze_loss(mean: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean distance between the policy's gaze-action means and the teacher's, over valid labels.

    Args:
        mean: The policy's gaze-action means, shape (batch, 2).
        labels: The teacher's gaze actions and their validity, shape (batch, 3).
    """
    valid = labels[:, 2] > 0.5
    # Invalid rows may hold NaN; replacing it keeps their (discarded) residuals finite, so no NaN reaches the gradients.
    residual = torch.where(valid[:, None], mean - torch.nan_to_num(labels[:, :2]), 0.0)
    return residual.norm(dim=-1).sum() / valid.sum().clamp_min(1)


class GazeSupervisedPPO(PPO):
    """PPO whose loss also pulls the policy's gaze toward a teacher's.

    The gaze is the last two actions. The teacher's labels are the observation group ``oracle_group``, as
    :class:`isaaclab_fovi.mdp.gaze_teacher_labels` computes them; they never enter the actor or the critic. Only the
    means of the gaze actions are supervised; pair it with
    :class:`~isaaclab_tasks.contrib.active_gaze.gaze_distribution.GazeGaussianDistribution`, which fixes their noise.

    Args:
        oracle_loss_coef: Weight of the gaze loss. Zero trains plain PPO.
        oracle_group: Observation group of the teacher's labels.
    """

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        oracle_loss_coef: float = 1.0,
        oracle_group: str = "oracle",
        **kwargs,
    ):
        if actor.is_recurrent or critic.is_recurrent or kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
            raise ValueError("GazeSupervisedPPO supports feedforward models without RND or symmetry.")
        if not math.isfinite(oracle_loss_coef) or oracle_loss_coef < 0:
            raise ValueError("oracle_loss_coef must be finite and nonnegative.")
        super().__init__(actor, critic, storage, **kwargs)
        self.oracle_loss_coef = oracle_loss_coef
        self.oracle_group = oracle_group

    def update(self) -> tuple[dict[str, float], dict[str, float]]:
        totals = torch.zeros(5, device=self.device)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch and batch.advantages.numel() > 1:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            with torch.amp.autocast(
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                self.actor(batch.observations, stochastic_output=True)
                log_prob = self.actor.get_output_log_prob(batch.actions)
                values = self.critic(batch.observations)
                entropy_loss = -self.actor.output_entropy.mean()
                if self.desired_kl is not None and self.schedule == "adaptive":
                    self._adapt_learning_rate(batch.old_distribution_params, self.actor.output_distribution_params)
                ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
                advantages = batch.advantages.squeeze(-1)
                surrogate = torch.max(
                    -advantages * ratio, -advantages * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
                ).mean()
                value_error = (values - batch.returns).pow(2)
                if self.use_clipped_value_loss:
                    clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                    value_error = torch.max(value_error, (clipped - batch.returns).pow(2))
                value_loss = value_error.mean()
                loss = surrogate + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                if self.oracle_loss_coef > 0:
                    labels = batch.observations[self.oracle_group].float()
                    oracle = gaze_loss(self.actor.output_mean[:, -GAZE_ACTIONS:].float(), labels)
                    valid = (labels[:, 2] > 0.5).float().mean()
                    loss = loss + self.oracle_loss_coef * oracle
                else:
                    # Plain PPO: the teacher's labels are neither needed nor read.
                    oracle = valid = torch.zeros((), device=self.device)
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals += torch.stack((value_loss, surrogate, entropy_loss, oracle, valid)).detach().float()
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.storage.clear()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        value, surrogate, entropy, oracle, oracle_valid = (totals / num_updates).tolist()
        return {"value": value, "surrogate": surrogate, "entropy": entropy, "oracle": oracle}, {
            "Optimizer/learning_rate": self.learning_rate,
            "Model/mean_policy_std": self.actor.output_std.mean().item(),
            "Loss/oracle_valid": oracle_valid,
        }

    def _adapt_learning_rate(self, old_params, new_params):
        """Adapt the learning rate to the policy's KL divergence, as PPO's adaptive schedule does."""
        with torch.inference_mode():
            kl = self.actor.get_kl_divergence(old_params, new_params).mean()
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                kl /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif 0.0 < kl < self.desired_kl / 2.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                rate = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(rate, src=0)
                self.learning_rate = rate.item()
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
