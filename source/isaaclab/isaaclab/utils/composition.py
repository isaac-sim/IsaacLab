# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration and construction of nested callable mechanisms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from .configclass import configclass


@configclass
class WrapperCfg:
    """Configure a mechanism around a callable or another configured term.

    The owner constructs the leaf and supplies its evaluation boundary to :meth:`wrap`.
    Each mechanism owns its entire ``__call__`` implementation and any state it needs.
    """

    class_type: type = MISSING
    """Runtime implementation, constructed with this config, the callable, batch size, and device."""

    term: Any = MISSING
    """Enclosed callable, leaf configuration, or another wrapper configuration."""

    params: dict[str, Any] = {}
    """Parameters of an enclosed function. Configure class-based leaves through their own config."""

    def unwrap(self) -> tuple[Any, dict[str, Any]]:
        """Return the enclosed leaf and its parameters without evaluating it."""
        if isinstance(self.term, WrapperCfg):
            if self.params:
                raise ValueError("Put function parameters on the wrapper directly enclosing the function.")
            return self.term.unwrap()
        return self.term, self.params

    def wrap(self, term: Callable, num_envs: int, device: str, **capabilities) -> Callable:
        """Construct a nested callable without splitting or reordering its computation.

        Args:
            term: Constructed leaf evaluation callable.
            num_envs: Batch size.
            device: Execution device.
            **capabilities: Boundary capabilities supplied by the owner.
        """
        if isinstance(self.term, WrapperCfg):
            term = self.term.wrap(term, num_envs, device, **capabilities)
        return self.class_type(self, term, num_envs, device, **capabilities)
