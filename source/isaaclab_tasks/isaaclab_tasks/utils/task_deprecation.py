# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper for retiring per-variant gym task IDs in favor of consolidated tasks.

When a task consolidation PR collapses N per-variant gym registrations into a
single consolidated task plus a preset selector, the retired IDs stay
registered for one release with an ``env_cfg_entry_point`` that emits a
:class:`DeprecationWarning` naming the new task and the equivalent
``presets=<name>`` (and optionally ``--agent=<name>``) invocation.

:func:`deprecated_task_alias` factors out the per-deprecation warning + cfg
resolution boilerplate. Call sites read as data: ``(old_id, new_command,
cfg_path)``, where *new_command* is the literal command a user should run
after migration -- the same text that appears in the deprecation warning.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Callable
from typing import Any


def _user_stacklevel() -> int:
    """Compute a ``warnings.warn`` stacklevel that lands on the first frame
    outside this module, so the deprecation warning cites user code rather
    than the gym/parse_cfg loader that invoked the entry-point factory.

    Walks a bounded number of frames; falls back to ``stacklevel=2`` if no
    non-local frame is found within the bound.
    """
    max_walk = 16
    level = 1
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__file__") == __file__:
        level += 1
        frame = frame.f_back
        if level > max_walk:
            return 2
    return level


def deprecated_task_alias(
    old_task_id: str,
    new_command: str,
    consolidated_cfg_path: str,
    cfg_factory: Callable[[], Any] | None = None,
) -> Callable[[], Any]:
    """Wrap a retired gym task ID with a :class:`DeprecationWarning` + cfg resolution.

    The returned callable is meant for use as a gym registry
    ``env_cfg_entry_point``. On invocation it emits a warning of the form::

        Task '<old_task_id>' is deprecated and will be removed in a future
        release. Use '<new_command>'.

    *new_command* is the literal command a user should run after migration --
    e.g. ``"--task=Isaac-Cartpole-Camera-v0 presets=rgb"`` -- and surfaces
    character-for-character in the warning. Whatever extra CLI tokens the new
    task needs (``presets=...``, ``--agent=...``, Hydra overrides, ...) go
    straight into this string.

    Default cfg resolution: imports *consolidated_cfg_path* via
    :mod:`importlib`, instantiates the class, and returns
    ``getattr(instance, <preset>)`` when *new_command* contains a
    ``presets=<preset>`` token, else the instance itself. The import is
    lazy -- it runs on first ``gym.make()``, not at registration time --
    matching gym's own handling of string ``"module:Name"`` entry points.
    Override via *cfg_factory* when resolution needs custom logic (e.g.
    a nested ``PresetCfg`` walk).

    Args:
        old_task_id: The deprecated gym task ID, quoted in the warning body.
        new_command: The replacement command, rendered verbatim inside
            single quotes in the warning. Typically
            ``"--task=NEW [presets=NAME] [--agent=NAME]"``.
        consolidated_cfg_path: ``"module.path:ClassName"`` string for the
            consolidated :class:`PresetCfg` subclass. Same format gym
            accepts for ``env_cfg_entry_point``. Resolved lazily.
        cfg_factory: Optional zero-arg callable that builds the env cfg the
            retired task should load. Use when default resolution doesn't
            fit -- e.g. a two-axis nested ``PresetCfg`` that needs both
            the root and a nested attribute pinned. When set, takes
            precedence over *consolidated_cfg_path*'s default resolution.

    Returns:
        A zero-arg callable suitable for use as ``env_cfg_entry_point``.
    """

    def factory():
        warnings.warn(
            f"Task '{old_task_id}' is deprecated and will be removed in a future release. Use '{new_command}'.",
            DeprecationWarning,
            stacklevel=_user_stacklevel(),
        )
        if cfg_factory is not None:
            return cfg_factory()
        # Default resolution: pick the variant named by ``presets=<name>`` in
        # the new_command, or return the bare consolidated cfg when absent.
        preset = next(
            (tok.split("=", 1)[1].split(",")[0] for tok in new_command.split() if tok.startswith("presets=")),
            None,
        )
        mod_name, cls_name = consolidated_cfg_path.split(":")
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return getattr(cls(), preset) if preset else cls()

    return factory
