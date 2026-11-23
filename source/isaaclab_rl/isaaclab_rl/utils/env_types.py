# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment compatibility validation shared by the learning framework wrappers."""

from __future__ import annotations

import contextlib
from typing import Any

import gymnasium as gym


def check_env_type(env: Any, *, allow_multi_agent: bool = False, allow_manager_based_env: bool = False) -> None:
    """Raise unless the environment supports the learning wrappers' reset behavior.

    Supported types are :class:`~isaaclab.envs.ManagerBasedRLEnv` and :class:`~isaaclab.envs.DirectRLEnv`,
    plus their Warp counterparts when ``isaaclab_experimental`` is installed.
    Environments that advertise an automatic reset mode must use same-step resets.

    Args:
        env: Environment or wrapper exposing ``unwrapped``.
        allow_multi_agent: Whether :class:`~isaaclab.envs.DirectMARLEnv` is supported.
        allow_manager_based_env: Whether the non-RL :class:`~isaaclab.envs.ManagerBasedEnv` is supported.

    Raises:
        ValueError: If the unwrapped environment type or automatic reset mode is unsupported.
    """
    # concrete environment classes load simulation modules, so import them only when a wrapper is built
    from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedEnv, ManagerBasedRLEnv

    allowed_types: list[type] = [ManagerBasedRLEnv, DirectRLEnv]
    if allow_multi_agent:
        allowed_types.append(DirectMARLEnv)
    if allow_manager_based_env:
        allowed_types.append(ManagerBasedEnv)
    with contextlib.suppress(ImportError):
        from isaaclab_experimental.envs import DirectRLEnvWarp, ManagerBasedEnvWarp, ManagerBasedRLEnvWarp

        allowed_types += [ManagerBasedRLEnvWarp, DirectRLEnvWarp]
        if allow_manager_based_env:
            allowed_types.append(ManagerBasedEnvWarp)

    if not isinstance(env.unwrapped, tuple(allowed_types)):
        names = " / ".join(cls.__name__ for cls in allowed_types)
        raise ValueError(f"The environment must be inherited from {names}. Environment type: {type(env.unwrapped)}")

    # Non-RL environments have no Gym metadata; Warp environments may omit the reset mode.
    if isinstance(env.unwrapped, gym.Env) and "autoreset_mode" in env.unwrapped.metadata:
        autoreset_mode = env.unwrapped.metadata["autoreset_mode"]
        if autoreset_mode != gym.vector.AutoresetMode.SAME_STEP:
            raise ValueError(
                f"RL wrappers require same-step automatic resets; got {autoreset_mode!r}. "
                "Set cfg.autoreset_mode to gymnasium.vector.AutoresetMode.SAME_STEP."
            )
