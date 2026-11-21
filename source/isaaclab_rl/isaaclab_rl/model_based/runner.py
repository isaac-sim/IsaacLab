# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified to compose task adapters and use Isaac Lab entrypoints for logging and checkpoints.

"""Training settings and physical collection, dynamics learning, and imagined policy updates."""

import math

import torch
import torch.distributed as dist

from isaaclab.utils import configclass

from .learners import (
    ActorCriticPolicyLearner,
    ActorCriticPolicyLearnerCfg,
    BpttPolicyLearner,
    BpttPolicyLearnerCfg,
    WorldModelLearner,
    WorldModelLearnerCfg,
)
from .models import Dynamics, MultiEncoderAE, build_mlp
from .rollout import ReplayBuffer


@configclass
class ModelBasedRunnerCfg:
    """Learning iterations count vectorized physical environment steps."""

    seed: int = 42
    max_iterations: int = 1000000
    buffer_size: int = 1000
    replay_ratio: int = 16
    dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    latent_dim: int = 32
    actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    latent_bc_weight: float = 0.1
    history_length: int = 1
    allow_tf32: bool = False
    lr_decay_iterations: int | None = None
    lr_final_factor: float = 0.03
    termination_softness: float = 0.05
    wm: WorldModelLearnerCfg = WorldModelLearnerCfg()
    policy: BpttPolicyLearnerCfg | ActorCriticPolicyLearnerCfg = ActorCriticPolicyLearnerCfg()
    log_interval: int = 10
    save_interval: int = 1000
    wandb_project: str | None = None
    wandb_name: str | None = None


REFERENCE_REVISION = "5efbf114e11f725c31d83e217be987c13c0e4ce2"


class Ratio:
    """Reference update scheduler: advance fractional credit without rounding each step."""

    def __init__(self, ratio: float):
        self._ratio = ratio
        self._prev = None

    def __call__(self, step: int) -> int:
        step = int(step)
        if self._ratio == 0:
            return 0
        if self._ratio < 0:
            return 1
        if self._prev is None:
            self._prev = step
            return 1
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


def build_actor(adapter, cfg: ModelBasedRunnerCfg) -> MultiEncoderAE:
    """Build the same actor for learning and checkpoint playback."""
    return MultiEncoderAE(
        adapter.obs_dim,
        adapter.num_actions,
        adapter.task_specs,
        adapter.slot_mix,
        cfg.latent_dim,
        cfg.actor_hidden_dims,
        cfg.latent_bc_weight,
        cfg.history_length,
    ).to(adapter.device)


class ModelBasedRunner:
    """Train on task-provided snapshots and differentiable rollouts.

    ``adapter`` owns physical state extraction, goal construction, and reset sampling.
    Checkpoints record model and optimizer state; restoring physical state and replay
    requires a separate simulator snapshot and is not implied by loading these weights.
    """

    def __init__(self, adapter, cfg: ModelBasedRunnerCfg):
        self.adapter = adapter
        self.cfg = cfg
        if cfg.lr_decay_iterations is not None:
            if cfg.lr_decay_iterations <= 0 or not 0 <= cfg.lr_final_factor <= 1:
                raise ValueError("Learning-rate decay requires a positive duration and final factor in [0, 1].")
            if cfg.wm.optimizer != "adam" or cfg.policy.optimizer != "adam":
                raise ValueError("Captured learning-rate schedules require tensor-valued Adam learning rates.")
        device = adapter.device
        self.representation = adapter.representation
        self.dynamics = Dynamics(
            self.representation,
            adapter.num_actions,
            adapter.sim_context.output_dim,
            adapter.action_scale,
            adapter.action_offset,
            cfg.dynamics_hidden_dims,
        ).to(device)
        self.actor = build_actor(adapter, cfg)
        self.critic = build_mlp(adapter.critic_dim, cfg.critic_hidden_dims, 1).to(device)
        if dist.is_initialized():
            for module in (self.dynamics, self.actor, self.critic):
                for tensor in (*module.parameters(), *module.buffers()):
                    dist.broadcast(tensor.detach(), src=0)
        self.replay_buffer = ReplayBuffer(adapter.buffer_schema, cfg.buffer_size, adapter.num_envs, device)
        self.world_model_learner = WorldModelLearner(self.dynamics, self.representation, cfg.wm, device)
        self.wm_env = adapter.create_rollout(self.dynamics, self.replay_buffer)
        if torch.device(device).type == "cuda":
            self.wm_env.step = torch.compile(self.wm_env.step)
            self.wm_env.reset = torch.compile(self.wm_env.reset)
        if isinstance(cfg.policy, ActorCriticPolicyLearnerCfg):
            self.policy_learner = ActorCriticPolicyLearner(self.actor, self.critic, self.wm_env, cfg.policy, device)
        else:
            self.policy_learner = BpttPolicyLearner(self.actor, self.wm_env, cfg.policy, device)
        self.should_train = Ratio(
            adapter.num_envs / cfg.wm.num_batches / cfg.wm.batch_size / cfg.wm.horizon * cfg.replay_ratio
        )
        self.iteration = 0
        self.learning_updates = 0
        self._latest_losses: dict[str, torch.Tensor] = {}

    def close(self) -> None:
        """Release captured collectives before destroying the distributed group."""
        self.world_model_learner._captured_update.close()
        self.policy_learner._captured_update.close()

    def step(self) -> dict[str, torch.Tensor]:
        """Collect one vector step, then perform the reference ratio of learning updates."""
        with torch.inference_mode():
            transition, actor_obs, logs = self.adapter.collect(self.actor, self.cfg.policy.noise)
            self.replay_buffer.insert(transition)
            states, actions, actor_inputs = transition["state"].clone(), transition["action"].clone(), actor_obs.clone()
        self.world_model_learner.update_normalization(states, actions)
        self.actor.train()
        self.actor.update_normalization(actor_inputs)
        for _ in range(self.should_train(self.iteration)):
            if self.replay_buffer.num_transitions < self.cfg.wm.horizon:
                continue
            self._update_learning_rate()
            data = (
                self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
                for _ in range(self.cfg.wm.num_batches)
            )
            self._latest_losses.update(self.world_model_learner.update(data))
            self._latest_losses.update(self.policy_learner.update())
            self.learning_updates += 1
        self.iteration += 1
        return logs | self._latest_losses

    def _update_learning_rate(self) -> None:
        """Update captured Adam learning-rate tensors without replacing their storage."""
        if self.cfg.lr_decay_iterations is None:
            return
        progress = min(self.iteration / self.cfg.lr_decay_iterations, 1.0)
        factor = self.cfg.lr_final_factor + (1 - self.cfg.lr_final_factor) * (1 + math.cos(math.pi * progress)) / 2
        groups = [(self.world_model_learner.optimizers, self.cfg.wm.lr)]
        groups.append((self.policy_learner.actor_optimizers, self.cfg.policy.lr))
        if isinstance(self.policy_learner, ActorCriticPolicyLearner):
            groups.append((self.policy_learner.critic_optimizers, self.cfg.policy.lr))
        for optimizers, base_lr in groups:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"].fill_(base_lr * factor)

    def checkpoint(self) -> dict:
        """Return weights and training state with explicit algorithm provenance."""
        state = {
            "reference_revision": REFERENCE_REVISION,
            "iteration": self.iteration,
            "learning_updates": self.learning_updates,
            "dynamics": self.dynamics.state_dict(),
            "sim_context": self.adapter.sim_context.state_dict(),
            "actor": self.actor.state_dict(),
            "world_model_optimizers": [opt.state_dict() for opt in self.world_model_learner.optimizers],
            "actor_optimizers": [opt.state_dict() for opt in self.policy_learner.actor_optimizers],
        }
        if isinstance(self.policy_learner, ActorCriticPolicyLearner):
            state["critic"] = self.critic.state_dict()
            state["critic_optimizers"] = [opt.state_dict() for opt in self.policy_learner.critic_optimizers]
        return state
