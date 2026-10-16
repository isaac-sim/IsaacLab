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
    """Base class for tensor_processors implemented as classes.

    Tensor processors implementations can be functions or classes. If a processor is a class, it should
    inherit from this class and implement the required methods.

    A class implementation of a processor can be used to store state information between calls.
    This is useful for tensor_processors that require stateful operations, such as rolling averages
    or delays or decaying filters.

    Example pseudo-code to create and use the class:

    .. code-block:: python

        from isaaclab.utils import tensor_processors

        # define custom keyword arguments to pass to TensorProcessorCfg
        kwarg_dict = {"arg_1": VAL_1, "arg_2": VAL_2}

        # create processor configuration object
        # func is the class name of the processor and params is the dictionary of arguments
        processor_config = tensor_processors.TensorProcessorCfg(
            func=tensor_processors.TensorProcessor, params=kwarg_dict
        )

        # define processor instance
        my_processor = MyProcessor(processor_config, data_dim=(32, 4), device="cpu")

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

    def close(self) -> None:
        """Release resources owned by this processor."""
        pass

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
