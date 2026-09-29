# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped RL-Games registry helpers for repeatable programmatic benchmark runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_MISSING = object()


def register_scoped_rl_games_environment(
    vecenv: Any, env_configurations: Any, environment: Any, vecenv_factory: Any
) -> Callable[[], None]:
    """Register an environment and return an idempotent restoration callback.

    Args:
        vecenv: RL-Games vector-environment registry module.
        env_configurations: RL-Games environment-configuration registry module.
        environment: Wrapped benchmark environment.
        vecenv_factory: Factory registered for ``IsaacRlgWrapper``.

    Returns:
        Callback that restores the two prior registry entries.
    """
    previous_vecenv = vecenv.vecenv_config.get("IsaacRlgWrapper", _MISSING)
    previous_environment = env_configurations.configurations.get("rlgpu", _MISSING)
    vecenv.register("IsaacRlgWrapper", vecenv_factory)
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: environment}
    )
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        if previous_vecenv is _MISSING:
            vecenv.vecenv_config.pop("IsaacRlgWrapper", None)
        else:
            vecenv.vecenv_config["IsaacRlgWrapper"] = previous_vecenv
        if previous_environment is _MISSING:
            env_configurations.configurations.pop("rlgpu", None)
        else:
            env_configurations.configurations["rlgpu"] = previous_environment

    return restore
