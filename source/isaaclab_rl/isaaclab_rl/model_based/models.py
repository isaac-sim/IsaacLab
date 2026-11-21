# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 the SuperMimic authors.
# Modified for Isaac Lab package boundaries and simulator interfaces.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict

if TYPE_CHECKING:
    from .representations import StateRepresentation, TaskSpec


class _CudaGraphNormalization(EmpiricalNormalization):
    """EmpiricalNormalization with in-place _std update for CUDA graph compatibility.

    The upstream update() does ``self._std = torch.sqrt(self._var)`` which creates
    a new tensor, invalidating data pointers captured by CUDA graphs (the
    captured training updates). This subclass uses copy_() instead.
    """

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        std_buf = self._std
        super().update(x)
        std_buf.copy_(self._std)
        self._std = std_buf


def build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    """Build an MLP: ``input_dim → (Linear → LN → GELU)* → Linear(output_dim)``."""
    layers: list[nn.Module] = []
    curr = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(curr, h), nn.LayerNorm(h), nn.GELU()]
        curr = h
    layers.append(nn.Linear(curr, output_dim))
    return nn.Sequential(*layers)


class Dynamics(nn.Module):
    """Differentiable forward model: ``(rep_state, action, aux_state) -> (next_aux, next_rep)``.

    ``action`` is the raw post-clamp policy output — same distribution the
    policy sees back as ``last_action`` in its observation. Dynamics applies
    ``proc = action * scale + offset`` internally before normalization.
    """

    def __init__(
        self,
        representation: StateRepresentation,
        num_actions: int,
        sim_ctx_dim: int,
        action_scale: torch.Tensor,
        action_offset: torch.Tensor,
        hidden_dims: tuple[int, ...],
    ):
        super().__init__()

        self.num_rep_state = representation.rep_state_dim
        self.num_actions = num_actions
        self.dt = representation.dt
        self._apply_prediction = representation.apply_prediction

        self.register_buffer("action_scale", action_scale.clone())
        self.register_buffer("action_offset", action_offset.clone())

        self.state_normalizer = _CudaGraphNormalization(self.num_rep_state)
        self.action_normalizer = _CudaGraphNormalization(num_actions)

        self.mlp = build_mlp(
            self.num_rep_state + num_actions + sim_ctx_dim,
            hidden_dims,
            output_dim=representation.prediction_dim,
        )

    def proc(self, action: torch.Tensor) -> torch.Tensor:
        """Map raw policy output to proc joint commands: ``action * scale + offset``."""
        return action * self.action_scale + self.action_offset

    def forward(
        self,
        rep_state: torch.Tensor,
        action: torch.Tensor,
        aux_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_norm = self.state_normalizer(rep_state)
        action_norm = self.action_normalizer(self.proc(action))
        prediction = self.mlp(torch.cat([state_norm, action_norm, sim_ctx], dim=-1))
        return self._apply_prediction(aux_state, rep_state, prediction, self.dt)

    def normalizers(self) -> list[_CudaGraphNormalization]:
        return [self.state_normalizer, self.action_normalizer]


class MultiEncoderAE(nn.Module):
    """Per-task goal encoders + shared decoder, tanh-bounded latent.

    Slots are contiguous blocks of the batch routed to each task's encoder.
    ``slot_mix`` is resolved into ``(lo, hi)`` per forward from the current
    batch size, so one instance serves both sim collection and BPTT rollout.

    Aux loss: latent BC pulling each non-anchor encoder's latent toward the
    anchor encoder's latent on the same state (anchor query stop-grad).

    Obs TensorDict layout:
      ``obs_td["history"]``            — last ``history_length`` obs frames
                                         (sensor or rep_state), (B, H, obs_dim),
                                         newest last; flattened into the
                                         decoder input
      ``obs_td[f"goal_{task.name}"]``  — full-batch goal per task; only the
                                         task's slot range is read
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        task_specs: list[TaskSpec],
        slot_mix: dict[str, float],
        latent_dim: int,
        hidden_dims: tuple[int, ...],
        latent_bc_weight: float,
        history_length: int,
    ):
        super().__init__()
        if not task_specs:
            raise ValueError("task_specs must be non-empty")
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.latent_dim = latent_dim
        self.latent_bc_weight = latent_bc_weight
        self.task_specs = task_specs
        self._task_names = [ts.name for ts in task_specs]
        # Anchor: the first task. The aux-loss hook regularizes each non-anchor
        # encoder against the anchor encoder on the same state.
        self._anchor = self._task_names[0]
        self._aux_loss = torch.zeros(0)

        self.obs_normalizer = _CudaGraphNormalization(obs_dim)
        self.goal_normalizers = nn.ModuleDict({ts.name: _CudaGraphNormalization(ts.goal_dim) for ts in task_specs})
        self.encoders = nn.ModuleDict(
            {ts.name: build_mlp(ts.goal_dim, hidden_dims, output_dim=latent_dim) for ts in task_specs}
        )
        self.decoder = build_mlp(
            latent_dim + history_length * obs_dim,
            hidden_dims,
            output_dim=action_dim,
        )

        self.set_slot_mix(slot_mix)

    def set_slot_mix(self, slot_mix: dict[str, float]) -> None:
        """Set fractional slot allocation per task. Missing entries default to 0."""
        unknown = set(slot_mix) - set(self._task_names)
        if unknown:
            raise ValueError(f"slot_mix has unknown tasks {sorted(unknown)}")
        fractions = [slot_mix.get(name, 0.0) for name in self._task_names]
        if any(frac < 0.0 for frac in fractions):
            raise ValueError(f"slot_mix fractions must be non-negative; got {slot_mix}")
        total = sum(fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"slot_mix must sum to 1.0; got {total}")
        self._slot_fractions = fractions

    def slot_ranges(self, batch_size: int) -> list[tuple[int, int]]:
        """Contiguous ``(lo, hi)`` per task at the given batch size; last absorbs remainder."""
        ranges: list[tuple[int, int]] = []
        off = 0
        last = len(self._slot_fractions) - 1
        for i, frac in enumerate(self._slot_fractions):
            hi = batch_size if i == last else off + int(round(batch_size * frac))
            ranges.append((off, hi))
            off = hi
        return ranges

    def forward(self, obs_td: TensorDict) -> torch.Tensor:
        slots = self.slot_ranges(obs_td.shape[0])
        enc_outs: list[torch.Tensor] = []
        for name, (lo, hi) in zip(self._task_names, slots, strict=True):
            if lo == hi:
                continue
            goal_n = self.goal_normalizers[name](obs_td[f"goal_{name}"][lo:hi])
            enc_outs.append(self.encoders[name](goal_n))

        self._aux_loss = self._compute_aux_loss(obs_td, slots)

        z = torch.cat([torch.tanh(eo) for eo in enc_outs], dim=0)
        history_n = self.obs_normalizer(obs_td["history"])
        return self.decoder(torch.cat([z, history_n.flatten(start_dim=1)], dim=-1))

    def _compute_aux_loss(self, obs_td: TensorDict, slots: list[tuple[int, int]]) -> torch.Tensor:
        """Per-env BC loss pulling each non-anchor encoder's latent toward the
        anchor encoder's latent on the same state.

        Goals are detached before the BC encoder forwards so BC's gradient only
        updates the non-anchor encoder's parameters — never the decoder or the
        anchor encoder.

        Anchor query is also under ``no_grad``. Skipped in eval mode.
        Returns ``(num_envs,)``; anchor / empty slots contribute zero.
        """
        out = torch.zeros(obs_td.shape[0], device=obs_td.device)
        if not self.training or self.latent_bc_weight == 0.0:
            return out
        anchor = self._anchor
        anchor_goal_d = self.goal_normalizers[anchor](obs_td[f"goal_{anchor}"]).detach()
        for name, (lo, hi) in zip(self._task_names, slots, strict=True):
            if name == anchor or lo == hi:
                continue
            goal_n_d = self.goal_normalizers[name](obs_td[f"goal_{name}"][lo:hi]).detach()
            z = torch.tanh(self.encoders[name](goal_n_d))
            with torch.no_grad():
                z_anchor = torch.tanh(self.encoders[anchor](anchor_goal_d[lo:hi]))
            out[lo:hi] = self.latent_bc_weight * ((z - z_anchor) ** 2).mean(dim=-1)
        return out

    def auxiliary_loss(self) -> torch.Tensor:
        return self._aux_loss

    def auxiliary_stats(self) -> dict[str, torch.Tensor]:
        if self._aux_loss.numel() == 0:
            return {}
        return {"bc_loss": self._aux_loss.mean().detach()}

    @torch.jit.unused
    def update_normalization(self, obs_td: TensorDict) -> None:
        obs = obs_td["history"][:, -1]
        self.obs_normalizer.update(obs)
        for name, (lo, hi) in zip(self._task_names, self.slot_ranges(obs.shape[0]), strict=True):
            if lo == hi:
                continue
            self.goal_normalizers[name].update(obs_td[f"goal_{name}"][lo:hi])
