# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modifier_cfg import ModifierCfg


class ModifierBase(ABC):
    """Base class for modifiers implemented as classes.

    Modifiers implementations can be functions or classes. If a modifier is a class, it should
    inherit from this class and implement the required methods.

    A class implementation of a modifier can be used to store state information between calls.
    This is useful for modifiers that require stateful operations, such as rolling averages
    or delays or decaying filters.

    Example pseudo-code to create and use the class:

    .. code-block:: python

        from isaaclab.utils import modifiers

        # define custom keyword arguments to pass to ModifierCfg
        kwarg_dict = {"arg_1" : VAL_1, "arg_2" : VAL_2}

        # create modifier configuration object
        # func is the class name of the modifier and params is the dictionary of arguments
        modifier_config = modifiers.ModifierCfg(func=modifiers.ModifierBase, params=kwarg_dict)

        # define modifier instance
        my_modifier = modifiers.ModifierBase(cfg=modifier_config)

    """

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        """Initializes the modifier class.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the modifier on.
        """
        self._cfg = cfg
        self._data_dim = data_dim
        self._device = device

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the Modifier.

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
