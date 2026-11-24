# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared spawn-function wrapping for contributed asset preparation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from isaaclab.sim import UsdFileCfg

if TYPE_CHECKING:
    from pxr import Usd

_WrapperCfgT = TypeVar("_WrapperCfgT", bound=UsdFileCfg)


def wrap_spawn(
    spawn_cfg: UsdFileCfg, wrapper_cfg_type: type[_WrapperCfgT], wrapper: Callable[..., Usd.Prim]
) -> _WrapperCfgT:
    """Route a USD spawn configuration through ``wrapper``, keeping its current spawn function.

    The wrapper configuration declares ``original_spawn``, which ``wrapper`` calls before
    preparing the spawned prim. Wrapping is idempotent: an already wrapped configuration keeps
    its original spawn function, while a function that replaced the wrapper after an earlier
    preparation becomes the new original. ``func`` may be the wrapper itself or its
    ``"module:function"`` string form after configuration serialization.

    Args:
        spawn_cfg: Spawn configuration to wrap.
        wrapper_cfg_type: Configuration type that declares ``original_spawn`` and the wrapper's settings.
        wrapper: Spawn function that calls ``cfg.original_spawn`` and prepares the result.

    Returns:
        The wrapped spawn configuration. It is ``spawn_cfg`` itself when it already has
        ``wrapper_cfg_type``, otherwise a new configuration copied from it.
    """
    if not isinstance(spawn_cfg, wrapper_cfg_type):
        original_spawn = spawn_cfg.func
        spawn_cfg = wrapper_cfg_type(**vars(spawn_cfg))
        spawn_cfg.original_spawn = original_spawn
    elif spawn_cfg.func is not wrapper and spawn_cfg.func != f"{wrapper.__module__}:{wrapper.__name__}":
        spawn_cfg.original_spawn = spawn_cfg.func
    spawn_cfg.func = wrapper
    return spawn_cfg
