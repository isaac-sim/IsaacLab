# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO for Isaac Lab environments built from TorchRL's collector, GAE, and clipped PPO loss."""

from __future__ import annotations

import os

import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchrl.collectors import Collector
from torchrl.data import Composite
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, IndependentNormal, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from .ppo_cfg import TorchRlPpoCfg
from .vecenv_wrapper import IsaacLabTorchRLWrapper


def make_actor(env: IsaacLabTorchRLWrapper, cfg: TorchRlPpoCfg) -> ProbabilisticActor:
    """Gaussian MLP policy over the ``"policy"`` observation group."""
    net = nn.Sequential(
        _mlp(env, "policy", 2 * env.action_spec.shape[-1], cfg.actor_hidden_dims, cfg.activation),
        NormalParamExtractor(scale_mapping=f"biased_softplus_{cfg.init_noise_std}"),
    )
    return ProbabilisticActor(
        TensorDictModule(net, in_keys=["policy"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=IndependentNormal,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
        default_interaction_type=ExplorationType.RANDOM,
    )


def make_critic(env: IsaacLabTorchRLWrapper, cfg: TorchRlPpoCfg) -> ValueOperator:
    """MLP value function over the flat ``"critic"`` observation group when the task defines one, else ``"policy"``."""
    critic_spec = env.observation_spec.get("critic", None)
    key = "critic" if critic_spec is not None and not isinstance(critic_spec, Composite) else "policy"
    return ValueOperator(_mlp(env, key, 1, cfg.critic_hidden_dims, cfg.activation), in_keys=[key])


def train_ppo(env: IsaacLabTorchRLWrapper, cfg: TorchRlPpoCfg, log_dir: str) -> ProbabilisticActor:
    """Trains a PPO agent for ``cfg.max_iterations`` iterations and returns the actor.

    Actor checkpoints (``model_<iteration>.pt``, loadable with :meth:`torch.nn.Module.load_state_dict` on
    :func:`make_actor`) and TensorBoard scalars, including the episode statistics Isaac Lab reports under
    ``extras["log"]``, are written to ``log_dir``.
    """
    torch.manual_seed(cfg.seed)
    actor = make_actor(env, cfg).to(cfg.device)
    critic = make_critic(env, cfg).to(cfg.device)
    loss = ClipPPOLoss(
        actor,
        critic,
        clip_epsilon=cfg.clip_param,
        entropy_bonus=cfg.entropy_coef > 0,
        entropy_coeff=cfg.entropy_coef,
        critic_coeff=cfg.value_loss_coef,
    )
    loss.set_keys(sample_log_prob="sample_log_prob")
    loss.make_value_estimator(ValueEstimators.GAE, gamma=cfg.gamma, lmbda=cfg.lam)
    optimizer = torch.optim.Adam(loss.parameters(), lr=cfg.learning_rate)
    frames_per_batch = env.batch_size[0] * cfg.num_steps_per_env
    collector = Collector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * cfg.max_iterations,
        device=cfg.device,
        auto_register_policy_transforms=False,
    )
    writer = SummaryWriter(log_dir)

    for iteration, batch in enumerate(collector, start=1):
        with torch.no_grad():
            loss.value_estimator(batch)
        samples = batch.reshape(-1)
        for _ in range(cfg.num_learning_epochs):
            for indices in torch.randperm(samples.shape[0], device=samples.device).chunk(cfg.num_mini_batches):
                terms = loss(samples[indices])
                optimizer.zero_grad()
                sum(value for key, value in terms.items() if key.startswith("loss_")).backward()
                nn.utils.clip_grad_norm_(loss.parameters(), cfg.max_grad_norm)
                optimizer.step()

        stats = {f"Loss/{key}": value.item() for key, value in terms.items() if key.startswith("loss_")}
        stats["Train/mean_step_reward"] = batch["next", "reward"].mean().item()
        stats.update({key: float(value) for key, value in env.unwrapped.extras.get("log", {}).items()})
        for key, value in stats.items():
            writer.add_scalar(key, value, iteration)
        print(f"[TorchRL] iteration {iteration}/{cfg.max_iterations}: reward {stats['Train/mean_step_reward']:.4f}")
        if iteration % cfg.save_interval == 0 or iteration == cfg.max_iterations:
            torch.save(actor.state_dict(), os.path.join(log_dir, f"model_{iteration}.pt"))

    collector.shutdown()
    writer.close()
    return actor


def _mlp(env: IsaacLabTorchRLWrapper, key: str, out_features: int, hidden_dims: list[int], activation: str) -> MLP:
    in_features = env.observation_spec[key].shape[len(env.batch_size) :].numel()
    return MLP(in_features, out_features, num_cells=hidden_dims, activation_class=getattr(nn, activation))
