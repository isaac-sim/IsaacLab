# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native modifier base class (experimental)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .modifier_cfg import ModifierCfg


class ModifierBase(ABC):
    """Base class for Warp-native class-based modifiers.

    Experimental fork of :class:`isaaclab.utils.modifiers.ModifierBase` adapted for the
    Warp-first calling convention.  Subclasses operate **in-place** on ``wp.array``
    buffers and return ``None``.

    A class implementation of a modifier can be used to store state information between
    calls.  This is useful for modifiers that require stateful operations, such as
    rolling averages, delays, or decaying filters.
    """

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        """Initializes the modifier class.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified.  First element is the
                batch size (number of environments).
            device: The device to run the modifier on.
        """
        self._cfg = cfg
        self._data_dim = data_dim
        self._device = device

    @abstractmethod
    def reset(self, env_mask: wp.array | None = None):
        """Resets the modifier.

        Args:
            env_mask: Boolean env mask of shape ``(num_envs,)`` selecting environments
                to reset. Defaults to None, in which case all environments are
                considered.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data: wp.array) -> None:
        """Apply the modification in-place.

        Args:
            data: The ``wp.array`` buffer to modify.  Shape should match the
                *data_dim* passed during initialization.
        """
        raise NotImplementedError
