# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

"""World model learner (supervised training)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from isaaclab_rl.model_based.optim import (
    CapturedUpdate,
    _optimize_step,
    build_optimizers,
    graph_state,
)

if TYPE_CHECKING:
    from isaaclab_rl.model_based.models import Dynamics

    from .interfaces import StateRepresentation


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
