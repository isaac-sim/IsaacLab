# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch (default) frontend.

The torch frontend is a thin wrapper around :func:`gym.make`. It does not
mutate the cfg — the standard ``isaaclab.envs.*`` env classes use
``FactoryBase`` to dispatch on the active physics backend at construction
time, so PhysX and Newton physics both run through this single path.

Its rule set is small: one warning when the user asks for the torch
frontend on a task that's *registered* against the warp runtime
(``entry_point`` under :data:`WARP_ROOT_PREFIXES`). That's a contradiction
of intent — the env class is a warp env regardless of the frontend flag —
so we surface it but don't block.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import (
    Change,
    CompatRule,
    Frontend,
    Issue,
    ResolveContext,
    Runtime,
    Severity,
    TaskMeta,
)


class WarnIfTaskIsWarpRegistered(CompatRule):
    """Warn when a warp-registered task is asked to run on the torch frontend.

    The env class itself is a warp implementation; ``--frontend=torch`` will
    still happily build it via ``gym.make`` (since the entry-point class is
    what runs), but the user's stated intent says torch. We emit a warning
    so the contradiction is visible.
    """

    name = "warn_if_task_is_warp_registered"

    def run(self, cfg: Any, ctx: ResolveContext) -> Iterable[Issue | Change]:
        if ctx.task.runtime == Runtime.WARP:
            yield Issue(
                rule=self.name,
                severity=Severity.WARNING,
                message=(
                    f"task is registered against the warp runtime"
                    f" (entry_point={ctx.task.entry_point!r}); --frontend=torch"
                    f" will still build the warp env class. Consider --frontend=warp."
                ),
                location="task.entry_point",
                detail={"runtime": ctx.task.runtime.value},
            )


class TorchFrontend(Frontend):
    """Default frontend: ``gym.make`` against the registered ``entry_point``."""

    name = "torch"
    rules = (WarnIfTaskIsWarpRegistered,)

    def construct(self, cfg: Any, meta: TaskMeta, **kwargs: Any) -> Any:
        import gymnasium as gym

        return gym.make(meta.task_id, cfg=cfg, **kwargs)
