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
    """Configure a callable around a term. The owner constructs the leaf; wrappers own their computation."""

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

    def wrap(
        self,
        term: Callable,
        num_envs: int,
        device: str,
        *,
        input_supported: bool = True,
        output_supported: bool = True,
        split_calls: bool = False,
    ) -> Callable:
        """Construct wrappers around the owner's leaf callable, preserving their nesting.

        ``input_supported`` and ``output_supported`` identify exposed signals. ``split_calls``
        means calls with arguments stage inputs and calls without arguments evaluate outputs
        on a separate clock, as action terms do. Wrappers own how they handle those calls.
        """
        if isinstance(self.term, WrapperCfg):
            term = self.term.wrap(
                term,
                num_envs,
                device,
                input_supported=input_supported,
                output_supported=output_supported,
                split_calls=split_calls,
            )
        return self.class_type(
            self,
            term,
            num_envs,
            device,
            input_supported=input_supported,
            output_supported=output_supported,
            split_calls=split_calls,
        )
