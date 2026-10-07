# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence

import torch

from ..string import string_to_callable
from .modifier_base import ModifierBase
from .modifier_cfg import ModifierCfg


class ModifierChain:
    """Apply modifiers to one tensor in configuration order.

    Class modifiers are constructed on the first call with the shape of the tensor they receive, so
    a chain needs no shape information up front and each modifier may change the shape or type for
    the next one. Function modifiers receive their :attr:`ModifierCfg.params` on every call.
    """

    def __init__(self, cfgs: Sequence[ModifierCfg], device: str):
        """Initialize the chain.

        Args:
            cfgs: Modifier configurations, applied in order.
            device: Device passed to class modifiers.

        Raises:
            TypeError: If an entry is not a :class:`ModifierCfg`.
        """
        for cfg in cfgs:
            if not isinstance(cfg, ModifierCfg):
                raise TypeError(f"Modifier configuration must be a ModifierCfg, received '{type(cfg)}'.")
        self._cfgs = tuple(cfgs)
        self._device = device
        self._stages: list[Callable[[torch.Tensor], torch.Tensor] | None] = [None] * len(self._cfgs)
        self._instances: list[ModifierBase] = []

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply every modifier in order.

        Args:
            data: Input tensor. Modifiers must not change it in place.

        Returns:
            Output of the last modifier, or ``data`` for an empty chain.
        """
        for index, cfg in enumerate(self._cfgs):
            stage = self._stages[index]
            if stage is None:
                stage = self._stages[index] = self._build(cfg, data)
            data = stage(data)
        return data

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset the state of constructed class modifiers.

        Args:
            env_ids: Environment indices to reset. Defaults to None, which resets all environments.
        """
        for modifier in self._instances:
            modifier.reset(env_ids=env_ids)

    def close(self) -> None:
        """Close constructed class modifiers. The chain rebuilds them if it is called again."""
        instances, self._instances = self._instances, []
        self._stages = [None] * len(self._cfgs)
        for modifier in instances:
            modifier.close()

    def _build(self, cfg: ModifierCfg, data: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        func = string_to_callable(str(cfg.func)) if isinstance(cfg.func, str) else cfg.func
        if inspect.isclass(func):
            modifier = func(cfg=cfg, data_dim=tuple(data.shape), device=self._device)
            if not isinstance(modifier, ModifierBase):
                raise TypeError(f"Modifier class '{func.__name__}' must inherit from ModifierBase.")
            self._instances.append(modifier)
            return modifier
        if not callable(func):
            raise TypeError(f"Modifier function must be callable, received '{func}'.")
        return functools.partial(func, **cfg.params)
