# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from collections.abc import Callable, Iterable
from typing import Any

from .raw_interface import RawMPInterface


class FunctionalMPWrapper(RawMPInterface):
    """Plug MP into an env by passing callables instead of subclassing `RawMPInterface`.

    Use this wrapper when the environment already exposes state accessors and you want
    to avoid a new class. Provide functions for the mask and tracking state; the rest of
    the MP pipeline (clamping, rollout, replanning) is handled by `BlackBoxWrapper`.
    """

    def __init__(
        self,
        env: gym.Env,
        context_mask_fn: Callable[[gym.Env], torch.Tensor] | None = None,
        current_pos_fn: Callable[[gym.Env], Any] | None = None,
        current_vel_fn: Callable[[gym.Env], Any] | None = None,
        action_bounds_fn: Callable[[gym.Env], tuple[Iterable, Iterable]] | None = None,
    ):
        """Wrap an environment with functional accessors for MP.

        Args:
            env (gym.Env): Environment to wrap; should expose `.device` and `.action_space`.
            context_mask_fn (Callable | None): Returns a boolean mask `(obs_dim,)` on
                `env.device` selecting MP-relevant observation entries. Defaults to
                `RawMPInterface.context_mask`.
            current_pos_fn (Callable | None): Returns position broadcastable to
                `(num_envs, dof)`. Required for rollout.
            current_vel_fn (Callable | None): Returns velocity broadcastable to
                `(num_envs, dof)`. Required for rollout.
            action_bounds_fn (Callable | None): Returns `(low, high)` tensors for step
                actions; optional safety clamp beyond `env.action_space`.

        Raises:
            NotImplementedError: If `current_pos_fn` or `current_vel_fn` is missing and
            the corresponding property is accessed.
        """
        super().__init__(env)
        self._context_mask_fn = context_mask_fn
        self._current_pos_fn = current_pos_fn
        self._current_vel_fn = current_vel_fn
        self._action_bounds_fn = action_bounds_fn

    @property
    def context_mask(self) -> torch.Tensor:
        """Return the policy mask using the provided callable or the default."""
        if self._context_mask_fn is not None:
            return torch.as_tensor(
                self._context_mask_fn(self.env), dtype=torch.bool, device=getattr(self.env, "device", "cpu")
            )
        return super().context_mask

    @property
    def current_pos(self):
        """Return current positions via `current_pos_fn`; required for MP conditioning."""
        if self._current_pos_fn is None:
            raise NotImplementedError("current_pos_fn must be provided for FunctionalMPWrapper.")
        return self._current_pos_fn(self.env)

    @property
    def current_vel(self):
        """Return current velocities via `current_vel_fn`; required for MP conditioning."""
        if self._current_vel_fn is None:
            raise NotImplementedError("current_vel_fn must be provided for FunctionalMPWrapper.")
        return self._current_vel_fn(self.env)

    @property
    def action_bounds(self):
        """Return step-action bounds via `action_bounds_fn` or defer to the base logic."""
        if self._action_bounds_fn is not None:
            return self._action_bounds_fn(self.env)
        return super().action_bounds
