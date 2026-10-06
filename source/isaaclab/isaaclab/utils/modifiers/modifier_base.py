# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...envs import ManagerBasedEnv
    from .modifier_cfg import ModifierCfg


@dataclass(frozen=True)
class ModifierOutput:
    """One observation value plus named intermediate results for later modifiers.

    Named tensors are borrowed read-only. The observation manager returns ``data`` after
    the final modifier, while modifiers can consume any named result in between.
    """

    data: torch.Tensor
    named: dict[str, torch.Tensor]
    name: str

    def clone(self) -> ModifierOutput:
        """Give a modifier an independent primary value without copying intermediates."""
        return self.with_data(self.data.clone())

    def with_data(self, data: torch.Tensor) -> ModifierOutput:
        """Replace the selected value while retaining other named intermediates."""
        return ModifierOutput(data, {**self.named, self.name: data}, self.name)


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
        kwarg_dict = {"arg_1": VAL_1, "arg_2": VAL_2}

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
    def __call__(self, env: ManagerBasedEnv, data: torch.Tensor | ModifierOutput) -> torch.Tensor | ModifierOutput:
        """Abstract method for defining the modification function.

        Args:
            env: The environment that owns the observation.
            data: The observation or named output to modify.

        Returns:
            Modified data, normally with the same shape as the input.
        """
        raise NotImplementedError
