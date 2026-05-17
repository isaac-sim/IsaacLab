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
construction boilerplate. Call sites read as data: ``(old_id, new_command,
cfg_factory)``, where *new_command* is the list of CLI tokens a user should
type after migration and *cfg_factory* builds the cfg the retired ID should
return -- typically the historical per-variant cfg class so the retired ID
stays bit-for-bit identical to its pre-deprecation behavior.
"""

from __future__ import annotations

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
    new_command: list[str],
    cfg_factory: Callable[[], Any],
) -> Callable[[], Any]:
    """Wrap a retired gym task ID with a :class:`DeprecationWarning` + cfg construction.

    The returned callable is meant for use as a gym registry
    ``env_cfg_entry_point``. On invocation it emits a warning of the form::

        Task '<old_task_id>' is deprecated and will be removed in a future
        release. Use '<new_command joined with spaces>'.

    then returns ``cfg_factory()``.

    *new_command* is the list of CLI tokens a user should type after
    migration -- e.g. ``["--task=Isaac-Cartpole-Camera-v0", "presets=rgb"]``
    -- joined with single spaces and rendered verbatim inside the warning's
    quoted command. One element per CLI token (``--flag=value``,
    ``key=value``, Hydra override, ...). Convention at call sites:
    ``--task=`` first, ``--agent=`` next when present, and the
    ``presets=NAME`` selector at the end.

    *cfg_factory* is invoked verbatim. The retired ID's cfg payload is
    whatever the factory returns -- typically the historical per-variant cfg
    class instance, so the retired ID stays bit-for-bit identical to its
    pre-deprecation behavior and only the deprecation warning is layered on
    top. The factory body is the natural place to import the historical cfg
    class lazily so the import cost is paid at first ``gym.make()`` rather
    than at registration time.

    Args:
        old_task_id: The deprecated gym task ID, quoted in the warning body.
        new_command: The replacement command split into CLI tokens, joined
            with single spaces in the warning. Typically
            ``["--task=NEW", "--agent=NAME"?, "presets=NAME"?]``.
        cfg_factory: Zero-arg callable that returns the cfg instance the
            retired task should load.

    Returns:
        A zero-arg callable suitable for use as ``env_cfg_entry_point``.
    """

    def factory():
        warnings.warn(
            f"Task '{old_task_id}' is deprecated and will be removed in a future release. Use"
            f" '{' '.join(new_command)}'.",
            DeprecationWarning,
            stacklevel=_user_stacklevel(),
        )
        return cfg_factory()

    return factory
