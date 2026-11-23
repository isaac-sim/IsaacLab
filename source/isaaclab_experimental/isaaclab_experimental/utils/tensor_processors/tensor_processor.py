# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native processor base class (experimental)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .tensor_processor_cfg import TensorProcessorCfg


class TensorProcessor(ABC):
    """Base class for Warp-native class-based tensor processors.

    Experimental fork of :class:`isaaclab.utils.tensor_processors.TensorProcessor` adapted for the
    Warp-first calling convention.  Subclasses operate **in-place** on ``wp.array``
    buffers and return ``None``.

    A class implementation of a processor can be used to store state information between
    calls.  This is useful for tensor processors that require stateful operations, such as
    rolling averages, delays, or decaying filters.
    """

    def __init__(self, cfg: TensorProcessorCfg, data_dim: tuple[int, ...], device: str) -> None:
        """Initializes the processor class.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified.  First element is the
                batch size (number of environments).
            device: The device to run the processor on.
        """
        self._cfg = cfg
        self._data_dim = data_dim
        self._device = device

    @abstractmethod
    def reset(self, env_mask: wp.array | None = None):
        """Resets the processor.

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
