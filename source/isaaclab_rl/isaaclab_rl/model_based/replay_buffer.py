# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Derived from HybridRobotics/SuperMimic, revision 5efbf114e11f725c31d83e217be987c13c0e4ce2.
# Copyright 2026 the SuperMimic authors. See docs/licenses/dependencies/supermimic-official-license.txt.
# Modified for Isaac Lab package boundaries and simulator interfaces.

from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass(frozen=True)
class FieldSpec:
    dim: int = 0  # 0 = scalar (no trailing axis); >0 = vector with trailing dim
    dtype: torch.dtype = torch.float32

    def shape(self, num_envs: int, buffer_size: int) -> tuple[int, ...]:
        return (num_envs, buffer_size, self.dim) if self.dim else (num_envs, buffer_size)


class ReplayBuffer:
    """Per-env circular buffer over a TensorDict, with env-stratified window sampling."""

    def __init__(
        self,
        schema: dict[str, FieldSpec],
        buffer_size: int,
        num_envs: int,
        device: str = "cpu",
    ):
        if not {"terminated", "truncated"} <= schema.keys():
            raise ValueError("schema needs 'terminated' and 'truncated'")
        self.num_transitions = 0
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self._step = 0
        self._storage = TensorDict(
            {
                name: torch.zeros(spec.shape(num_envs, buffer_size), dtype=spec.dtype, device=device)
                for name, spec in schema.items()
            },
            batch_size=[num_envs, buffer_size],
        )

    def insert(self, td: TensorDict) -> None:
        if td.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"insert expects batch_size=[{self.num_envs}], got {list(td.batch_size)}")
        if set(td.keys()) != set(self._storage.keys()):
            raise ValueError("schema mismatch")
        self._storage[:, self._step] = td
        self._step = (self._step + 1) % self.buffer_size
        self.num_transitions = min(self.buffer_size, self.num_transitions + 1)

    def sample_causal(self, sequence_length: int, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """Sample a window of ``L`` frames ENDING at a uniformly-drawn filled frame.

        The newest frame (index ``-1``) is the sampled "current" state — drawn like
        a 1-step ``sample``, so its distribution is independent of ``L``. Preceding
        slots hold the current frame's real in-episode predecessors; slots before
        the episode start (a stored ``terminated``/``truncated``) or before the
        oldest filled frame are clamped to the first in-episode frame, matching the
        real-sim history ring's reset padding.

        Returns ``(window, padded)``: the ``(B, L)`` window and a ``(B, L)`` bool
        mask that is ``True`` on the clamp-padded slots, so the caller can noise the
        oldest real frame once and share that realization across them.
        """
        L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
        dev, n = self.device, self.num_transitions
        if L > T:
            raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
        if n < 1:
            raise RuntimeError("sample_causal on empty buffer")
        base = self._step if n == T else 0
        env_idx = torch.randint(E, (B,), device=dev)
        current_age = torch.randint(n, (B,), device=dev)
        ages = current_age[:, None] + (torch.arange(L, device=dev) - (L - 1))  # (B, L), newest last
        ended = (self._storage["terminated"] | self._storage["truncated"])[
            env_idx[:, None], (base + ages.clamp(min=0)) % T
        ]
        # Reachable = contiguous in-episode suffix ending at the (always-valid) newest
        # frame; a stored done at an earlier slot ends the prior episode there.
        valid = (ages >= 0) & ~ended.bool()
        valid[:, -1] = True
        reachable = torch.cummin(valid.flip(-1).int(), dim=-1).values.flip(-1).bool()
        first_age = torch.where(reachable, ages, current_age[:, None]).amin(dim=-1, keepdim=True)
        slot = (base + torch.where(reachable, ages, first_age)) % T
        return self._storage[env_idx[:, None], slot], ~reachable

    def sample(self, sequence_length: int, batch_size: int) -> TensorDict:
        L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
        dev, n = self.device, self.num_transitions
        if L > T:
            raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
        # Index by age (0 = oldest filled, n-1 = newest); slot = (base + age) % T.
        # In age-space windows are linear, so wrap and warm-up collapse into one predicate.
        base = self._step if n == T else 0
        ages = torch.arange(T, device=dev)
        ended = (self._storage["terminated"] | self._storage["truncated"])[:, (base + ages) % T].long()
        # Interior [a, a+L-2] is done-free iff c[a+L-1] == c[a] (c is cumsum with leading 0).
        c = torch.cat([ended.new_zeros(E, 1), ended.cumsum(1), ended.new_zeros(E, L - 1)], dim=1)
        valid = (c[:, L - 1 : L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)
        csum = valid.cumsum(dim=1, dtype=torch.long)
        total = csum[:, -1]
        if not (total > 0).any():
            raise RuntimeError(f"no valid sequences L={L} n={n}")
        env_idx = torch.multinomial((total > 0).float(), B, replacement=True)
        u = (torch.rand(B, device=dev) * total[env_idx].float()).long()
        age_start = torch.searchsorted(csum[env_idx], u[:, None], right=True).squeeze(-1)
        storage_idx = (base + age_start[:, None] + torch.arange(L, device=dev)) % T
        return self._storage[env_idx[:, None], storage_idx]
