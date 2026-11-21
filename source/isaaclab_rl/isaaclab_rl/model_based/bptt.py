# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""Policy learners using a differentiable world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab_rl.model_based.optim import (
    CapturedUpdate,
    _optimize_step,
    build_optimizers,
    graph_state,
)

if TYPE_CHECKING:
    from .rollout import MultiTaskWorldModelEnv


@dataclass
class BpttPolicyLearnerCfg:
    lr: float = 2e-4
    optimizer: str = "muon_adam"
    num_batches: int = 1
    batch_size: int = 4096
    noise: float = 0.1
    horizon: int = 32
    gamma: float = 1.0
    survival_masking: bool = True
    beta2: float = 0.95  # faster Adam EMA recovery after gradient spikes
    grad_clip_value: float = 0.1  # per-parameter gradient element clipping


@dataclass
class ActorCriticPolicyLearnerCfg(BpttPolicyLearnerCfg):
    gamma: float = 0.99
    lam: float = 0.95


def gae_returns(
    values: torch.Tensor,
    last_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """GAE returns ``A_t + V(s_t)`` of a time-major rollout. A soft done ``p_t``
    discounts the next state by ``(1 - p_t) * gamma``."""
    returns = torch.empty_like(values)
    advantage = 0.0
    next_value = last_value
    for t in reversed(range(values.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + not_done * gamma * next_value - values[t]
        advantage = delta + not_done * gamma * lam * advantage
        returns[t] = advantage + values[t]
        next_value = values[t]
    return returns


class BpttPolicyLearner:
    """Policy Learner using direct Backpropagation Through Time (BPTT)."""

    def __init__(
        self,
        policy: nn.Module,
        wm_env: MultiTaskWorldModelEnv,
        cfg: BpttPolicyLearnerCfg,
        device: str = "cpu",
    ):
        self.actor = policy
        self.wm_env = wm_env
        self.cfg = cfg
        self.actor_params = list(self.actor.parameters())
        self.actor_optimizers = build_optimizers(
            [self.actor], self.cfg.lr, betas=(0.9, self.cfg.beta2), device=device, optimizer=self.cfg.optimizer
        )
        self._rollout_actor = torch.compile(self.actor) if torch.device(device).type == "cuda" else self.actor
        self._captured_update = CapturedUpdate(self._update_from, self._graph_state)

    def update(self) -> dict[str, torch.Tensor]:
        self.actor.train()
        for module in (self.wm_env.dynamics, self.wm_env.representation):
            module.eval()
            module.requires_grad_(False)
        starts = [self.wm_env.sample_initial_states() for _ in range(self.cfg.num_batches)]
        return self._captured_update(torch.stack(starts))

    def _update_from(self, starts: TensorDict) -> dict[str, torch.Tensor]:
        totals: dict[str, torch.Tensor] = {}
        for start in starts.unbind(0):
            for k, v in self._train_batch(start).items():
                totals[k] = totals[k] + v if k in totals else v
        n = self.cfg.num_batches
        reward = totals.pop("return") / self.cfg.horizon / n / self.wm_env.representation.dt
        return {"Policy/reward": reward} | {f"Policy/{k}": v / n for k, v in totals.items()}

    def _train_batch(self, start: TensorDict) -> dict[str, torch.Tensor]:
        """Imagine one rollout from ``start`` and take one actor step on its return."""
        cfg = self.cfg
        obs = self.wm_env.reset(start)
        ret, aux_loss, survival = 0.0, 0.0, 1.0
        rollout: list[TensorDict] = []
        for t in range(cfg.horizon):
            actions = self._rollout_actor(obs)
            next_obs, reward, term_prob, _ = self.wm_env.step(actions)
            step = {"critic": obs["critic"], "reward": reward, "done": term_prob}
            rollout.append(TensorDict(step, batch_size=reward.shape).detach())
            ret = ret + cfg.gamma**t * survival * reward
            aux_loss = aux_loss + getattr(self.actor, "auxiliary_loss", lambda: 0.0)()
            if cfg.survival_masking:
                survival = survival * (1 - term_prob)
            obs = next_obs
        ret = ret + self._bootstrap(obs, survival, term_prob)
        loss = (-ret + aux_loss).mean()
        grad_norm = _optimize_step(self.actor_optimizers, loss, self.actor_params, 1.0, cfg.grad_clip_value)
        return {
            "return": ret.mean().detach(),
            "grad_norm": grad_norm,
            **self._fit_critic(rollout, obs),
            **getattr(self.actor, "auxiliary_stats", lambda: {})(),
        }

    def _bootstrap(self, obs: TensorDict, survival: torch.Tensor, term_prob: torch.Tensor) -> torch.Tensor | float:
        """Value of the rollout's last state, discounted into its return."""
        del obs, survival, term_prob
        return 0.0

    def _fit_critic(self, rollout: list[TensorDict], last_obs: TensorDict) -> dict[str, torch.Tensor]:
        """Fit a critic to the rollout after the actor step."""
        del rollout, last_obs
        return {}

    def _graph_state(self) -> list[torch.Tensor]:
        """What the captured update reads that Python can rebind."""
        wm = self.wm_env
        modules = [self.actor, wm.dynamics, wm.representation]
        return graph_state(modules, self.actor_optimizers)


class ActorCriticPolicyLearner(BpttPolicyLearner):
    """Actor-critic variant that adds a critic bootstrap and GAE critic update."""

    def __init__(
        self,
        policy: nn.Module,
        critic: nn.Module,
        wm_env: MultiTaskWorldModelEnv,
        cfg: ActorCriticPolicyLearnerCfg,
        device: str = "cpu",
    ):
        super().__init__(policy, wm_env, cfg, device)
        self.critic = critic
        self.critic_optimizers = build_optimizers([critic], self.cfg.lr, device=device, optimizer=self.cfg.optimizer)
        self._v_max = self.wm_env.max_step_reward / (1 - self.cfg.gamma)

    def _bootstrap(self, obs: TensorDict, survival: torch.Tensor, term_prob: torch.Tensor) -> torch.Tensor:
        self.critic.requires_grad_(False)
        value = self.critic(obs["critic"]).clamp(max=self._v_max).squeeze(-1)
        alive = survival if self.cfg.survival_masking else 1.0 - term_prob
        return self.cfg.gamma**self.cfg.horizon * value * alive

    def _fit_critic(self, rollout: list[TensorDict], last_obs: TensorDict) -> dict[str, torch.Tensor]:
        steps = torch.stack(rollout)
        with torch.no_grad():
            values = self.critic(steps["critic"])
            last_value = self.critic(last_obs["critic"])
        returns = gae_returns(
            values,
            last_value,
            steps["reward"].unsqueeze(-1),
            steps["done"].unsqueeze(-1),
            self.cfg.gamma,
            self.cfg.lam,
        ).clamp(max=self._v_max)
        obs, returns = steps["critic"].flatten(0, 1), returns.flatten(0, 1)
        perm = torch.randperm(len(returns) // 4 * 4, device=returns.device)
        self.critic.requires_grad_(True)
        losses = []
        for _ in range(5):
            for idx in perm.view(4, -1):
                loss = (returns[idx] - self.critic(obs[idx])).pow(2).mean()
                _optimize_step(self.critic_optimizers, loss, self.critic.parameters(), max_norm=1.0)
                losses.append(loss.detach())
        return {"value_function": torch.stack(losses).mean()}

    def _graph_state(self) -> list[torch.Tensor]:
        return super()._graph_state() + graph_state([self.critic], self.critic_optimizers)
