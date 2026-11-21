# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""World-model and policy learners with shared optimization and CUDA graph updates."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict

if TYPE_CHECKING:
    from .models import Dynamics
    from .representations import StateRepresentation
    from .rollout import MultiTaskWorldModelEnv


def build_optimizers(
    modules: Sequence[nn.Module],
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    device: str = "cpu",
    optimizer: str = "muon_adam",
) -> list[torch.optim.Optimizer]:
    """Build fused capturable Adam, optionally using Muon for intermediate Linear weights.

    ``muon_adam`` retains the pinned source optimizer; ``adam`` supports the paper
    recipe with a tensor-valued learning rate. Muon's rate is fixed after capture.
    """
    if optimizer not in {"adam", "muon_adam"}:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    hidden = [
        linear.weight
        for module in modules
        for seq in module.modules()
        if optimizer == "muon_adam" and isinstance(seq, nn.Sequential)
        for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
    ]
    hidden_ids = {id(p) for p in hidden}
    rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
    adam = torch.optim.Adam(rest, lr=torch.tensor(lr, device=device), betas=betas, fused=True, capturable=True)
    if not hidden:
        return [adam]
    return [adam, torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw")]


def _optimize_step(
    optimizers: Sequence[torch.optim.Optimizer],
    loss: torch.Tensor,
    params: Iterable[torch.Tensor] | None = None,
    max_norm: float | None = None,
    clip_value: float = 0.0,
) -> torch.Tensor | None:
    """Run one optimizer step: zero_grad -> backward -> optional clip -> step."""
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if dist.is_initialized() and dist.get_world_size() > 1:
        # Average before clipping so every rank applies the combined-batch update.
        gradients = [
            parameter.grad
            for optimizer in optimizers
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        flat = torch.cat([gradient.flatten() for gradient in gradients])
        dist.all_reduce(flat)
        flat.div_(dist.get_world_size())
        offset = 0
        for gradient in gradients:
            gradient.copy_(flat[offset : offset + gradient.numel()].view_as(gradient))
            offset += gradient.numel()
    grad_norm = None
    if params is not None and max_norm is not None:
        # Don't set error_if_nonfinite=True: it forces a per-step CPU<->GPU sync.
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
        if clip_value > 0:
            torch.nn.utils.clip_grad_value_(params, clip_value)
    for optimizer in optimizers:
        optimizer.step()
    return grad_norm


def graph_state(modules: Iterable[nn.Module], optimizers: Iterable[torch.optim.Optimizer]) -> list[torch.Tensor]:
    """What a captured update reads that Python can rebind: parameters, buffers,
    optimizer state and learning-rate tensors."""
    state: list[torch.Tensor] = []
    for module in modules:
        state += [*module.parameters(), *module.buffers()]
    for optimizer in optimizers:
        state += [t for s in optimizer.state.values() for t in s.values()]
        state += [group["lr"] for group in optimizer.param_groups if torch.is_tensor(group["lr"])]
    return state


def _identity(state: list[torch.Tensor]) -> list[int]:
    return [x.data_ptr() for x in state]


@functools.cache
def _capture_stream_and_pool(device: torch.device):
    return torch.cuda.Stream(device), torch.cuda.graph_pool_handle()


class CapturedUpdate:
    """Runs ``fn`` eagerly for its first ``warmup`` calls, then records it once as
    a CUDA graph and replays it on the device of ``state()``. ``fn`` must be
    capture-safe: no host syncs, fixed shapes, and every tensor it reads besides
    its inputs updated in place. Python values it reads are fixed at capture, and
    each replay overwrites the outputs it returns. All CapturedUpdates on a device
    record on one stream into one memory pool and must replay in the order they
    were recorded. A replay raises if a tensor in ``state()`` moved; a learning rate
    that changes is a device tensor, changed in place with ``group["lr"].fill_(lr)``."""

    def __init__(
        self,
        fn: Callable[[TensorDict], dict[str, torch.Tensor]],
        state: Callable[[], list[torch.Tensor]],
        warmup: int = 3,
    ):
        self._fn = fn
        self._state = state
        self._warmup = warmup
        self._inputs: TensorDict | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._identity: list[int] = []
        self._outputs: dict[str, torch.Tensor] = {}

    def close(self) -> None:
        """Release graph-held NCCL communicators before process-group shutdown."""
        if self._graph is not None:
            self._graph.reset()
            self._graph = None

    def __call__(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
        state = self._state()
        if not state[0].is_cuda:
            return self._fn(inputs)
        with torch.cuda.device(state[0].device):
            stream, pool = _capture_stream_and_pool(state[0].device)
            if self._inputs is None:
                self._inputs = inputs.clone()
            else:
                self._inputs.update_(inputs)
            if self._warmup > 0:
                self._warmup -= 1
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    outputs = self._fn(self._inputs)
                torch.cuda.current_stream().wait_stream(stream)
                return outputs
            if self._graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._outputs = self._fn(self._inputs)
                self._graph, self._identity = graph, _identity(self._state())
            elif _identity(state) != self._identity:
                raise RuntimeError("a captured tensor moved")
            self._graph.replay()
            torch.cuda.current_stream().synchronize()
            return self._outputs


@dataclass
class WorldModelLearnerCfg:
    lr: float = 2e-4
    optimizer: str = "muon_adam"
    num_batches: int = 10
    batch_size: int = 2048
    horizon: int = 8

    # Loss Weights (Bryson's Rule) — integrated root pose, both representations
    nominal_error_root_pose_pos: float = 0.01
    nominal_error_root_pose_ori: float = 0.02

    # Loss Weights — gencoord representation
    nominal_error_joint_pos: float = 0.1
    nominal_error_joint_vel: float = 3.14

    # Loss Weights — maxcoord representation
    nominal_error_body_pos: float = 0.05
    nominal_error_body_rot: float = 0.1
    nominal_error_body_vel: float = 1.0
    nominal_error_body_avel: float = 3.14


class WorldModelLearner:
    """Trainer for World Model (Supervised)."""

    def __init__(
        self,
        dynamics: Dynamics,
        representation: StateRepresentation,
        cfg: WorldModelLearnerCfg,
        device: str = "cpu",
    ) -> None:
        self.dynamics = dynamics
        self.representation = representation
        self.cfg = cfg
        self.device = device

        self._all_params = list(dynamics.parameters())
        self.optimizers = build_optimizers([dynamics], self.cfg.lr, device=self.device, optimizer=self.cfg.optimizer)

        # Bryson's rule: each named channel group is weighted 1 / nominal_error^2.
        self.loss_group_weights = {
            name: 1.0 / getattr(cfg, attr) ** 2 for name, attr in representation.LOSS_WEIGHT_CONFIG.items()
        }

        self._compiled_rollout = torch.compile(self._rollout)
        self._compiled_compute_losses = torch.compile(self._compute_losses)
        self._compiled_represent = torch.compile(self.representation.represent)
        self._captured_update = CapturedUpdate(
            self._update_from,
            lambda: graph_state([dynamics, representation], self.optimizers),
        )

    def update_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """Update the dynamics state/action normalizers from fresh collected data.

        ``actions`` are raw policy outputs (what the replay buffer stores); the
        action normalizer sees proc (matching what ``forward`` feeds the MLP after
        ``Dynamics.proc``).
        """
        rep_states = self.representation.represent(states)
        rep_states_flat = rep_states.reshape(-1, rep_states.shape[-1])
        actions_flat = actions.reshape(-1, actions.shape[-1])

        # EmpiricalNormalization.update is a no-op in eval mode; toggle to train
        # for the updates and restore afterward.
        normalizers = self.dynamics.normalizers()
        for n in normalizers:
            n.train()

        self.dynamics.state_normalizer.update(rep_states_flat)
        self.dynamics.action_normalizer.update(self.dynamics.proc(actions_flat))

        for n in normalizers:
            n.eval()

    def update(self, data: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
        self.dynamics.train()
        for p in self._all_params:
            p.requires_grad_(True)
        batches = torch.stack([td.select("state", "action", "sim_ctx") for td in data])
        return self._captured_update(batches)

    def _update_from(self, batches: TensorDict) -> dict[str, torch.Tensor]:
        stats = []

        for td in batches.unbind(0):
            full_states = td["state"]
            actions = td["action"]

            all_rep_states = self._compiled_represent(full_states)

            all_aux_states = self.representation.extract_aux_state(full_states)
            init_rep_state = all_rep_states[:, 0, :]
            init_aux = all_aux_states[:, 0, :]

            sim_ctx = td["sim_ctx"][:, 0]

            pred_rep_states, pred_aux_states = self._compiled_rollout(init_rep_state, actions, init_aux, sim_ctx)

            # 2. Vectorized Loss Calculation (always FP32, compiled for kernel fusion)
            loss_state, group_mse = self._compiled_compute_losses(
                pred_rep_states,
                pred_aux_states,
                all_rep_states[:, 1:],
                all_aux_states[:, 1:],
            )

            # 3. Backprop
            _optimize_step(self.optimizers, loss_state, self._all_params, max_norm=1.0)

            # 4. Logging
            stats.append(torch.cat([loss_state.unsqueeze(0), group_mse]).detach())

        names = ["state", *self.loss_group_weights]
        means = torch.stack(stats).mean(0)
        return {f"Loss/{name}": m for name, m in zip(names, means, strict=True)}

    def _rollout(
        self,
        init_rep_state: torch.Tensor,
        actions: torch.Tensor,
        init_aux: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive rollout over the window's actions.

        The prediction from the last action is never scored, so the rollout stops
        one step short.

        Returns:
          (pred_rep_states, pred_aux_states), stacked on dim 1.
        """
        rep_state, aux = init_rep_state, init_aux
        pred_rep_states: list[torch.Tensor] = []
        pred_aux_states: list[torch.Tensor] = []
        for k in range(actions.shape[1] - 1):
            aux, rep_state = self.dynamics(rep_state, actions[:, k], aux, sim_ctx)
            pred_rep_states.append(rep_state)
            pred_aux_states.append(aux)
        return torch.stack(pred_rep_states, dim=1), torch.stack(pred_aux_states, dim=1)

    def _compute_losses(
        self,
        pred_rep_states: torch.Tensor,
        pred_aux_states: torch.Tensor,
        gt_rep_states: torch.Tensor,
        gt_aux_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Squared error per named channel group. Bryson's rule scales each group by
        # its weight; the state loss is the plain mean over every channel.
        loss_blocks = self.representation.loss_blocks
        pred_blocks = loss_blocks(pred_rep_states, pred_aux_states)
        gt_blocks = loss_blocks(gt_rep_states, gt_aux_states)
        sq_errs = {name: (pred_blocks[name] - gt_blocks[name]) ** 2 for name in pred_blocks}
        group_mse = torch.stack([sq.mean() for sq in sq_errs.values()])
        # Filled on the device: a captured update cannot copy from the host.
        widths = torch.stack([group_mse.new_full((), sq.shape[-1]) for sq in sq_errs.values()])
        weights = torch.stack([group_mse.new_full((), self.loss_group_weights[name]) for name in sq_errs])
        loss_state = (group_mse * weights * widths).sum() / widths.sum()
        return loss_state, group_mse


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
