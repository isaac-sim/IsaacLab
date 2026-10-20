# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .tensor_processor_cfg import TensorProcessorCfg


class TensorProcessor(ABC):
    """Transform one tensor into one tensor, optionally retaining state between calls.

    Subclasses implement ``__call__`` and ``reset``. Observation processors preserve the input shape.
    Configuration and dimensions
    are supplied by the observation manager once during construction.
    """

    def __init__(self, cfg: TensorProcessorCfg, data_dim: tuple[int, ...], device: str) -> None:
        """Initializes the processor class.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the processor on.
        """
        self._cfg = cfg
        self._data_dim = data_dim
        self._device = device

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the Processor.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Abstract method for defining the modification function.

        Args:
            data: The data to be modified. Shape should match the data_dim passed during initialization.

        Returns:
            Modified data. Shape is the same as the input data.
        """
        raise NotImplementedError
