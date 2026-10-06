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

    A class modifier receives the environment and observation on each call. It may keep state
    between calls and implement ``reset(env_ids)`` and ``close()`` when it owns resources.
    The observation manager constructs it from :class:`ModifierCfg` with the input dimensions
    and device. A modifier that changes shape may expose ``output_dim`` for the next modifier.
    """

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str, *, env: ManagerBasedEnv) -> None:
        """Initializes the modifier class.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the modifier on.
            env: The environment that owns the modifier.
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
