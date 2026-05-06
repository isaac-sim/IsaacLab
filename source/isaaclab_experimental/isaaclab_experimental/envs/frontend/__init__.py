# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frontend framework: pluggable runtime selectors for IsaacLab tasks.

A frontend (chosen via ``--frontend {torch,warp}``) is a thin dispatcher that
takes a stable env cfg + task id, runs a pluggable :class:`CompatRule`
pipeline against the (cfg, task, frontend) triple, and constructs the env on
the matching runtime. New runtimes plug in by subclassing :class:`Frontend`
and calling :func:`register_frontend`; new compatibility checks plug in by
subclassing :class:`CompatRule` and adding it to a frontend's
:attr:`Frontend.rules`.

Public API::

    from isaaclab_experimental.envs.frontend import get_frontend
    frontend = get_frontend("warp")
    env = frontend.build(env_cfg, task_id="Isaac-Cartpole-v0")
    print(env.unwrapped.frontend_report.format())
"""

from __future__ import annotations

from .base import (
    WARP_ROOT_PREFIXES,
    Change,
    CompatRule,
    Frontend,
    FrontendIncompatibleError,
    Issue,
    Report,
    ResolveContext,
    Runtime,
    Severity,
    TaskMeta,
    TaskResolver,
    Workflow,
    available_frontends,
    get_frontend,
    iter_term_attrs,
    register_frontend,
    resolve_warp_twin,
    walk_attrs,
)
from .torch import TorchFrontend, WarnIfTaskIsWarpRegistered
from .warp import (
    CheckPhysicsIsNewton,
    DropUnsupportedSensors,
    PromoteSceneEntityCfg,
    ResolvePhysicsPreset,
    SwapActionClassType,
    SwapMdpFunctions,
    VerifyDirectIsWarp,
    WarpFrontend,
)

# Register built-ins so ``get_frontend("torch" | "warp")`` works without
# the caller importing the concrete classes.
register_frontend(TorchFrontend.name, TorchFrontend)
register_frontend(WarpFrontend.name, WarpFrontend)


__all__ = [
    # core abstractions
    "Frontend",
    "CompatRule",
    "TaskMeta",
    "TaskResolver",
    "Report",
    "Issue",
    "Change",
    "Severity",
    "Workflow",
    "Runtime",
    "ResolveContext",
    "FrontendIncompatibleError",
    "WARP_ROOT_PREFIXES",
    # registry
    "register_frontend",
    "get_frontend",
    "available_frontends",
    # rule helpers
    "walk_attrs",
    "resolve_warp_twin",
    "iter_term_attrs",
    # built-in frontends
    "TorchFrontend",
    "WarpFrontend",
    # built-in rules
    "WarnIfTaskIsWarpRegistered",
    "CheckPhysicsIsNewton",
    "ResolvePhysicsPreset",
    "DropUnsupportedSensors",
    "PromoteSceneEntityCfg",
    "SwapMdpFunctions",
    "SwapActionClassType",
    "VerifyDirectIsWarp",
]
