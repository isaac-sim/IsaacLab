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
    """Lightweight MP wrapper that uses callables instead of subclassing.

    This is useful when the underlying environment already exposes state accessors or when you want
    to plug MP on top of an env without writing a dedicated wrapper class. You supply functions to
    fetch the context mask and tracking state; everything else is handled by :class:`BlackBoxMPWrapper`.
    """

    def __init__(
        self,
        env: gym.Env,
        context_mask_fn: Callable[[gym.Env], torch.Tensor] | None = None,
        current_pos_fn: Callable[[gym.Env], Any] | None = None,
        current_vel_fn: Callable[[gym.Env], Any] | None = None,
        action_bounds_fn: Callable[[gym.Env], tuple[Iterable, Iterable]] | None = None,
    ):
        super().__init__(env)
        self._context_mask_fn = context_mask_fn
        self._current_pos_fn = current_pos_fn
        self._current_vel_fn = current_vel_fn
        self._action_bounds_fn = action_bounds_fn

    @property
    def context_mask(self) -> torch.Tensor:
        if self._context_mask_fn is not None:
            return torch.as_tensor(
                self._context_mask_fn(self.env), dtype=torch.bool, device=getattr(self.env, "device", "cpu")
            )
        return super().context_mask

    @property
    def current_pos(self):
        if self._current_pos_fn is None:
            raise NotImplementedError("current_pos_fn must be provided for FunctionalMPWrapper.")
        return self._current_pos_fn(self.env)

    @property
    def current_vel(self):
        if self._current_vel_fn is None:
            raise NotImplementedError("current_vel_fn must be provided for FunctionalMPWrapper.")
        return self._current_vel_fn(self.env)

    @property
    def action_bounds(self):
        if self._action_bounds_fn is not None:
            return self._action_bounds_fn(self.env)
        return super().action_bounds
