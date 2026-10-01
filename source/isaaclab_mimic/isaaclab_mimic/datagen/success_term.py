# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for evaluating success terms outside the termination manager."""

import inspect
from collections.abc import Sequence
from typing import Any

from isaaclab.managers import ManagerTermBase, TerminationTermCfg


def initialize_success_term(success_term: TerminationTermCfg, env: Any) -> TerminationTermCfg:
    """Instantiate a class-based success term for direct evaluation."""
    if inspect.isclass(success_term.func):
        if not issubclass(success_term.func, ManagerTermBase):
            raise TypeError(
                "Class-based success terms must inherit from ManagerTermBase."
                f" Received: '{success_term.func.__name__}'."
            )
        success_term.func = success_term.func(cfg=success_term, env=env)
    return success_term


def reset_success_term(success_term: TerminationTermCfg, env_ids: Sequence[int] | None = None) -> None:
    """Reset state maintained by a class-based success term."""
    if isinstance(success_term.func, ManagerTermBase):
        success_term.func.reset(env_ids=env_ids)
