# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial motion-prior extension for RSL-RL PPO."""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from .discriminator import AMPDiscriminator
from .replay_buffer import AMPReplayBuffer


class AMPPPO(PPO):
    """PPO with an adversarial motion-prior reward and discriminator update."""

    def __init__(
        self,
        actor,
        critic,
        storage: RolloutStorage,
        *,
        amp_data=None,
        amp_observation_dim: int | None = None,
        command_manager=None,
        amp_obs_key: str = "amp",
        amp_reward_coef: float = 0.3,
        amp_task_reward_lerp: float = 0.7,
        amp_discr_hidden_dims: list[int] | tuple[int, ...] = (256, 256),
        amp_discr_activation: str = "relu",
        amp_feature_normalization: bool = True,
        amp_grad_pen_lambda: float = 10.0,
        amp_replay_buffer_size: int = 100_000,
        amp_discr_trunk_weight_decay: float = 1.0e-3,
        amp_discr_head_weight_decay: float = 1.0e-1,
        amp_update_interval: int = 1,
        amp_reward_command_gate: bool = False,
        amp_reward_command_name: str | None = None,
        amp_reward_command_threshold: float = 0.0,
        amp_rollout_obs_clip: float | None = 500.0,
        amp_min_normalized_std: float | None = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO and its task-local AMP components."""
        if amp_observation_dim is None or amp_data is None:
            raise ValueError(
                "AMPPPO requires an AMP observation dimension and motion dataset. Both are "
                "prepared by AMPPPO.construct_algorithm; instantiate through the runner."
            )

        super().__init__(actor, critic, storage, **kwargs)

        self.discriminator = AMPDiscriminator(
            observation_dim=amp_observation_dim,
            hidden_dims=amp_discr_hidden_dims,
            activation=amp_discr_activation,
            feature_normalization=amp_feature_normalization,
            device=self.device,
        ).to(self.device)
        self.optimizer.add_param_group(
            {
                "params": self.discriminator.trunk.parameters(),
                "weight_decay": amp_discr_trunk_weight_decay,
                "name": "amp_trunk",
            }
        )
        self.optimizer.add_param_group(
            {
                "params": self.discriminator.linear.parameters(),
                "weight_decay": amp_discr_head_weight_decay,
                "name": "amp_head",
            }
        )

        self.amp_data = amp_data
        self.command_manager = command_manager
        self.amp_obs_key = amp_obs_key
        self.amp_reward_coef = amp_reward_coef
        self.amp_task_reward_lerp = amp_task_reward_lerp
        self.amp_grad_pen_lambda = amp_grad_pen_lambda
        self.amp_storage = AMPReplayBuffer(amp_observation_dim, amp_replay_buffer_size, self.device)
        self.amp_update_interval = max(1, int(amp_update_interval))
        self.amp_reward_command_gate = amp_reward_command_gate
        self.amp_reward_command_name = amp_reward_command_name
        self.amp_reward_command_threshold = amp_reward_command_threshold
        self.amp_rollout_obs_clip = amp_rollout_obs_clip
        self.amp_min_normalized_std = amp_min_normalized_std

        self._update_counter = 0
        self._warned_missing_gate_command = False

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:
        """Construct AMP data from the environment before delegating model setup to PPO."""
        alg_cfg = cfg["algorithm"]

        amp_data_cfg = dict(alg_cfg.pop("amp_data"))
        dataset_class = resolve_callable(amp_data_cfg.pop("class_type"))
        amp_data = dataset_class(env=env.unwrapped, device=device, **amp_data_cfg)

        amp_obs_key = alg_cfg.get("amp_obs_key", "amp")
        if amp_obs_key not in obs.keys():
            raise ValueError(
                f"AMPPPO expects an observation group '{amp_obs_key}' from the environment. "
                f"Available groups: {list(obs.keys())}"
            )

        amp_obs_dim = obs[amp_obs_key].shape[-1]
        expected_dim = amp_data.observation_dim
        if amp_obs_dim != expected_dim:
            raise ValueError(
                f"AMP observation group '{amp_obs_key}' has {amp_obs_dim} features but the motion "
                f"dataset produces frames of {expected_dim}. Check that the group's terms and "
                "joint order match the dataset's amp_obs_terms."
            )

        alg_cfg["amp_observation_dim"] = amp_obs_dim
        alg_cfg["amp_data"] = amp_data
        alg_cfg["command_manager"] = getattr(env.unwrapped, "command_manager", None)

        return PPO.construct_algorithm(obs, env, cfg, device)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Blend task and AMP rewards, then record the PPO transition."""
        amp_state = self.transition.observations[self.amp_obs_key]
        amp_next_state = obs[self.amp_obs_key]
        if self.amp_rollout_obs_clip is not None:
            done_mask = dones.bool().reshape(-1)
            if torch.any(done_mask):
                amp_next_state = amp_next_state.clone()
                amp_next_state[done_mask] = amp_next_state[done_mask].clamp(
                    -self.amp_rollout_obs_clip, self.amp_rollout_obs_clip
                )

        amp_rewards, discriminator_predictions = self.discriminator.predict_amp_reward(amp_state, amp_next_state)
        amp_rewards = self.amp_reward_coef * amp_rewards
        gate = self._amp_command_gate()
        self.amp_storage.insert(amp_state, amp_next_state)
        if gate is not None:
            amp_rewards = amp_rewards * gate
        total_rewards = (1.0 - self.amp_task_reward_lerp) * amp_rewards + self.amp_task_reward_lerp * rewards

        log = extras.setdefault("log", {})
        log["Train/mean_amp_reward"] = amp_rewards.mean()
        log["Train/mean_task_reward"] = rewards.mean()
        log["Train/mean_discriminator_prediction"] = discriminator_predictions.mean()

        self._clip_rollout_observations(obs)
        super().process_env_step(obs, total_rewards, dones, extras)

    def update(self) -> dict[str, float]:
        """Update PPO and AMP together, as in the standalone Asimov task."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None
        mean_amp_loss = 0.0
        mean_grad_pen = 0.0
        mean_policy_d = 0.0
        mean_expert_d = 0.0

        train_amp = self._update_counter % self.amp_update_interval == 0
        self._update_counter += 1
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        if train_amp:
            mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
            policy_generator = self.amp_storage.generator(num_updates, mini_batch_size)
            expert_generator = self.amp_data.feed_forward_generator(num_updates, mini_batch_size)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss += self.symmetry.mirror_loss_coeff * symmetry_loss
            if self.rnd:
                rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size])

            if train_amp:
                policy_state, policy_next_state = next(policy_generator)
                expert_state, expert_next_state = next(expert_generator)
                normalized_policy_state = self.discriminator.normalize(policy_state)
                normalized_policy_next_state = self.discriminator.normalize(policy_next_state)
                normalized_expert_state = self.discriminator.normalize(expert_state)
                normalized_expert_next_state = self.discriminator.normalize(expert_next_state)
                policy_d = self.discriminator(
                    torch.cat((normalized_policy_state, normalized_policy_next_state), dim=-1)
                )
                expert_d = self.discriminator(
                    torch.cat((normalized_expert_state, normalized_expert_next_state), dim=-1)
                )
                expert_loss = nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
                policy_loss = nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen = self.discriminator.compute_grad_pen(
                    expert_state, expert_next_state, lambda_=self.amp_grad_pen_lambda
                )
                loss += amp_loss + grad_pen

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            self._clip_actor_critic_gradients()
            self.optimizer.step()
            self._clamp_action_std()
            if self.rnd:
                self.rnd.optimizer.step()
            if train_amp:
                self.discriminator.update_normalization(policy_state, expert_state)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if train_amp:
                mean_amp_loss += amp_loss.item()
                mean_grad_pen += grad_pen.item()
                mean_policy_d += policy_d.mean().item()
                mean_expert_d += expert_d.mean().item()

        if train_amp and self.is_multi_gpu:
            self._sync_discriminator_normalizer()

        self.storage.clear()
        losses = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if mean_rnd_loss is not None:
            losses["rnd"] = mean_rnd_loss / num_updates
        if mean_symmetry_loss is not None:
            losses["symmetry"] = mean_symmetry_loss / num_updates
        if train_amp:
            losses.update(
                {
                    "amp": mean_amp_loss / num_updates,
                    "amp_grad_pen": mean_grad_pen / num_updates,
                    "amp_policy_pred": mean_policy_d / num_updates,
                    "amp_expert_pred": mean_expert_d / num_updates,
                }
            )
        return losses

    def _clip_actor_critic_gradients(self) -> torch.Tensor:
        return nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)

    def _amp_command_gate(self) -> torch.Tensor | None:
        if not self.amp_reward_command_gate:
            return None
        command = self._get_velocity_command()
        if command is None:
            if not self._warned_missing_gate_command:
                print(
                    "[WARN] amp_reward_command_gate is enabled, but no velocity command was found."
                    " AMP rewards are left ungated."
                )
                self._warned_missing_gate_command = True
            return None
        command_norm = torch.linalg.norm(command[:, :3].to(self.device), dim=1)
        return (command_norm > self.amp_reward_command_threshold).to(dtype=torch.float, device=self.device)

    def _get_velocity_command(self) -> torch.Tensor | None:
        if self.command_manager is None:
            return None
        command_names = [self.amp_reward_command_name, "twist", "base_velocity"]
        for name in dict.fromkeys(n for n in command_names if n):
            try:
                return self.command_manager.get_command(name)
            except Exception:
                continue
        return None

    def _clip_rollout_observations(self, obs: TensorDict) -> None:
        if self.amp_rollout_obs_clip is None:
            return
        model_obs_groups = set(self.actor.obs_groups) | set(self.critic.obs_groups)
        for group_name in model_obs_groups:
            obs[group_name].clamp_(-self.amp_rollout_obs_clip, self.amp_rollout_obs_clip)

    def _clamp_action_std(self) -> None:
        if self.amp_min_normalized_std is None:
            return
        distribution = getattr(self.actor, "distribution", None)
        std_param = getattr(distribution, "std_param", None)
        if std_param is not None:
            std_param.data.clamp_(min=self.amp_min_normalized_std)

    def _sync_discriminator_normalizer(self) -> None:
        if not self.discriminator.feature_normalization:
            return
        state = [self.discriminator.feature_norm.state_dict()]
        torch.distributed.broadcast_object_list(state, src=0)
        self.discriminator.feature_norm.load_state_dict(state[0])

    def reduce_parameters(self) -> None:
        """Synchronize the policy and discriminator gradients across workers."""
        params = list(self.actor.parameters())
        params.extend(self.critic.parameters())
        params.extend(self.discriminator.parameters())
        if self.rnd:
            params.extend(self.rnd.parameters())
        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in params:
            if param.grad is None:
                continue
            numel = param.numel()
            param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
            offset += numel

    def broadcast_parameters(self) -> None:
        """Broadcast PPO and discriminator parameters from the primary worker."""
        super().broadcast_parameters()
        discriminator_state = [self.discriminator.state_dict()]
        torch.distributed.broadcast_object_list(discriminator_state, src=0)
        self.discriminator.load_state_dict(discriminator_state[0])

    def train_mode(self) -> None:
        """Put PPO models and the discriminator in training mode."""
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        """Put PPO models and the discriminator in evaluation mode."""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """Return PPO and discriminator state, including their joint optimizer."""
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load joint-optimizer checkpoints and earlier split-optimizer checkpoints."""
        requested_load_cfg = load_cfg
        if requested_load_cfg is None:
            requested_load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        split_optimizer = "amp_optimizer_state_dict" in loaded_dict
        ppo_load_cfg = dict(requested_load_cfg)
        if split_optimizer:
            ppo_load_cfg["optimizer"] = False

        load_iteration = super().load(loaded_dict, ppo_load_cfg, strict)
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)

        if split_optimizer and requested_load_cfg.get("optimizer"):
            self._load_split_optimizer_states(
                loaded_dict["optimizer_state_dict"], loaded_dict["amp_optimizer_state_dict"]
            )

        return load_iteration

    def _load_split_optimizer_states(self, policy_state: dict, discriminator_state: dict) -> None:
        merged = self.optimizer.state_dict()
        merged["state"] = {}

        old_policy_ids = policy_state["param_groups"][0]["params"]
        new_policy_ids = merged["param_groups"][0]["params"]
        for old_id, new_id in zip(old_policy_ids, new_policy_ids, strict=True):
            if old_id in policy_state["state"]:
                merged["state"][new_id] = policy_state["state"][old_id]

        old_discriminator_ids = [param for group in discriminator_state["param_groups"] for param in group["params"]]
        new_discriminator_ids = merged["param_groups"][1]["params"] + merged["param_groups"][2]["params"]
        for old_id, new_id in zip(old_discriminator_ids, new_discriminator_ids, strict=True):
            if old_id in discriminator_state["state"]:
                merged["state"][new_id] = discriminator_state["state"][old_id]

        policy_options = {key: value for key, value in policy_state["param_groups"][0].items() if key != "params"}
        merged["param_groups"][0].update(policy_options)
        for index, source_group in enumerate(discriminator_state["param_groups"], start=1):
            merged["param_groups"][index].update({key: value for key, value in source_group.items() if key != "params"})
        self.optimizer.load_state_dict(merged)
        self.learning_rate = self.optimizer.param_groups[0]["lr"]
