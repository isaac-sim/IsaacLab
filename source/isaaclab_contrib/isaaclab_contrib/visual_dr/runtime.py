# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized runtime: decides which frames are generated, and composites the result.

The runtime tracks episode identity per environment and derives every random
choice from it arithmetically, so a style depends on *which episode of which
environment* a frame belongs to and not on call order, replica assignment or
global RNG state. Reading the same observation twice returns the same pixels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from .backends import DRBackend, DRFrame, DRRequest
from .prompts import load_prompt_bank

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .cfg import VisualDRCfg

# splitmix64 constants, written as the signed values torch's int64 actually holds:
# the unsigned literals exceed int64 max and cannot be converted. A hash rather
# than a stateful torch.Generator, so identical inputs give identical draws
# regardless of call order, device or which replica serves the request.
_GOLDEN = 0x9E3779B97F4A7C15 - (1 << 64)
_MIX_A = 0xBF58476D1CE4E5B9 - (1 << 64)
_MIX_B = 0x94D049BB133111EB - (1 << 64)


def _shift(value: torch.Tensor, bits: int) -> torch.Tensor:
    """Logical right shift. torch's ``>>`` on int64 propagates the sign bit."""
    return (value >> bits) & ((1 << (64 - bits)) - 1)


def _mix(*values: torch.Tensor) -> torch.Tensor:
    """Hash int64 tensors into a well-distributed int64 tensor."""
    state = torch.zeros_like(values[0])
    for value in values:
        state = state + value * _GOLDEN
        state = (state ^ _shift(state, 30)) * _MIX_A
        state = (state ^ _shift(state, 27)) * _MIX_B
        state = state ^ _shift(state, 31)
    return state


def _unit(hashed: torch.Tensor) -> torch.Tensor:
    """Map a hash to a float in [0, 1) without touching the sign bit."""
    return (hashed & 0x7FFFFFFFFFFFFFFF).double().div(float(0x8000000000000000)).float()


class VisualDRRuntime:
    """Applies visual DR to a vectorized environment's camera observations.

    The application owns residency -- :meth:`activate` before collection,
    :meth:`offload` before a learner claims the GPU -- but not scheduling. Episode
    boundaries, action-chunk gating and partial resets are read off the
    environment, so no training loop has to call the runtime at the right moment.
    """

    def __init__(self, cfg: VisualDRCfg, num_envs: int, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.backend: DRBackend = cfg.backend.class_type(cfg.backend)
        self.prompts = load_prompt_bank(getattr(cfg.backend, "prompts", None))

        zeros = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._episode_ids = zeros.clone()
        self._step_in_chunk = zeros.clone()
        self._env_ids = torch.arange(num_envs, dtype=torch.long, device=self.device)
        self._consumed = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self._restyle = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._style_seeds = zeros.clone()

        self._cache: dict[str, torch.Tensor] = {}
        self._synced_step: int | None = None
        self._observation_index = 0
        self._active = False
        self._closed = False
        self.errors: list[str] = []
        """Reasons frames were left unrandomized, when ``on_error`` is ``passthrough``."""

    # -- residency ---------------------------------------------------------

    def activate(self) -> None:
        if self._closed:
            raise RuntimeError("Visual DR runtime is closed")
        if not self._active:
            self.backend.activate()
            self._active = True

    def offload(self) -> None:
        """Release model memory. Reads still work; they return raw frames."""
        if self._active:
            self._active = False
            self._cache.clear()
            self.backend.offload()

    def close(self) -> None:
        if not self._closed:
            self._active = False
            self._cache.clear()
            self.backend.close()
            self._closed = True

    # -- scheduling --------------------------------------------------------

    def sync(self, env: ManagerBasedRLEnv) -> None:
        """Advance episode and chunk state once per environment step.

        Called by every camera term; the step counter makes all calls after the
        first a no-op, so cameras within one step share a scope and repeated
        reads of the same observation are free.
        """
        step = int(env.common_step_counter)
        if self._synced_step == step:
            return
        self._synced_step = step
        self._observation_index += 1
        self._cache.clear()

        # An environment reset by the previous step arrives here with its episode
        # length back at zero. This is the only partial-reset signal the runtime
        # needs: it is per environment and it does not disturb the others.
        was_reset = env.episode_length_buf == 0
        self._episode_ids = self._episode_ids + was_reset.long()
        self._step_in_chunk = torch.where(was_reset, torch.zeros_like(self._step_in_chunk), self._step_in_chunk + 1)

        # A frame is consumed when a policy actually reads it: the first frame of
        # an episode, and thereafter one per action chunk. The frames in between
        # are still rendered and stepped; they are just never generated.
        period = self.cfg.decision_period
        self._consumed = was_reset | (self._step_in_chunk % period == 0)
        self._step_in_chunk = torch.where(self._consumed, torch.zeros_like(self._step_in_chunk), self._step_in_chunk)

        base = torch.full_like(self._env_ids, self.cfg.seed)
        if self.cfg.scope == "per_batch":
            draw = _unit(_mix(base, torch.full_like(self._env_ids, self._observation_index)))[:1].expand(self.num_envs)
        else:
            draw = _unit(
                _mix(base, self._env_ids, self._episode_ids, torch.full_like(self._env_ids, self._observation_index))
            )
        self._restyle = self._consumed & (draw < self.cfg.probability)

        if self.cfg.style == "per_episode":
            self._style_seeds = _mix(base, self._env_ids, self._episode_ids)
        else:
            self._style_seeds = _mix(
                base, self._env_ids, self._episode_ids, torch.full_like(self._env_ids, self._observation_index)
            )

    # -- generation --------------------------------------------------------

    def read(self, camera: str, rgb: torch.Tensor, make_frame: Callable[[], DRFrame]) -> torch.Tensor:
        """Return the camera's observation, randomized where this scope calls for it.

        ``make_frame`` is deferred so depth and segmentation are never fetched for
        a scope in which nothing will be generated.
        """
        if camera in self._cache:
            return self._cache[camera]
        # Clone rather than hand back the renderer's own buffer: it is reused by
        # the next render, and an observation that mutates underneath a rollout
        # buffer is close to impossible to diagnose later.
        if not self._active or not bool(self._restyle.any()):
            return rgb.clone()
        selected = self._restyle.nonzero(as_tuple=False).squeeze(-1)
        frame = make_frame()
        frame.validate()
        output = self._generate(camera, frame, selected)
        self._cache[camera] = output
        return output

    @torch.no_grad()
    def _generate(self, camera: str, frame: DRFrame, selected: torch.Tensor) -> torch.Tensor:
        output = frame.rgb.clone()
        seeds = self._style_seeds[selected]
        for start in range(0, selected.numel(), self.cfg.backend.max_batch):
            chunk = selected[start : start + self.cfg.backend.max_batch]
            chunk_seeds = seeds[start : start + self.cfg.backend.max_batch]
            prompts = tuple(self.prompts[int(s) % len(self.prompts)] for s in chunk_seeds.tolist())
            sub = frame.index(chunk)
            try:
                generated = self.backend.generate(sub, DRRequest(chunk_seeds, prompts, camera))
            except Exception as error:  # noqa: BLE001 - policy is the caller's to set
                if self.cfg.on_error == "raise":
                    raise
                self.errors.append(f"{camera}: {type(error).__name__}: {error}")
                continue
            if (generated.shape, generated.dtype, generated.device) != (sub.rgb.shape, sub.rgb.dtype, sub.rgb.device):
                raise ValueError(
                    f"Backend returned {tuple(generated.shape)} {generated.dtype} on {generated.device}; "
                    f"expected {tuple(sub.rgb.shape)} {sub.rgb.dtype} on {sub.rgb.device}"
                )
            # Foreground preservation is this paste. Without mask-guided denoising
            # the generated background does not know what will be pasted over it,
            # so boundary_px exists to hide the seam.
            output[chunk] = torch.where(sub.preserve, sub.rgb, generated)
        return output
