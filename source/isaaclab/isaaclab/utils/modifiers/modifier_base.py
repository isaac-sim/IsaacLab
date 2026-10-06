# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...envs import ManagerBasedEnv
    from .modifier_cfg import ModifierCfg


class ModifierBase(ABC):
    """Base class for modifiers implemented as classes.

    A class modifier receives the environment and observation on each call. It may keep state
    between calls and implement ``reset(env_ids)`` and ``close()`` when it owns resources.
    The observation manager constructs it from :class:`ModifierCfg` with the input dimensions
    and environment. A modifier that changes shape may expose ``output_dim`` for the next modifier.
    """

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], *, env: ManagerBasedEnv) -> None:
        """Initializes the modifier class.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            env: The environment that owns the modifier.
        """
        self._cfg = cfg
        self._data_dim = data_dim

    @abstractmethod
    def __call__(self, env: ManagerBasedEnv, data: torch.Tensor) -> torch.Tensor:
        """Abstract method for defining the modification function.

        Args:
            env: The environment that owns the observation.
            data: The observation to modify.

        Returns:
            Modified data, normally with the same shape as the input.
        """
        raise NotImplementedError
