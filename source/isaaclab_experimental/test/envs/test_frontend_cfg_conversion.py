# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coverage test: every manager-based ``*-Warp-v0`` task adapts cleanly from its stable cfg.

For each registered manager-based warp task we load the *stable* env cfg it points
at, force Newton physics, and run :func:`adapt_cfg_for_warp` — the same adaptation
the warp env runs in its ``__init__``. A pass proves the SceneEntityCfg promotion
and the MDP twin swap succeed for every warp-managed term (observations / rewards /
terminations / actions). This is the guard against drift: if a stable cfg later
gains a warp-managed term with no warp twin, the matching case fails here instead
of at training time.
"""

from __future__ import annotations

import importlib

import gymnasium as gym
import isaaclab_tasks_experimental  # noqa: F401
import pytest
from isaaclab_experimental.envs.frontend import adapt_cfg_for_warp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

# Registering the task packages is the whole point — import for side effects.
import isaaclab_tasks  # noqa: F401

# Manager-based warp tasks are exactly those whose entry point is the shared warp
# env class; direct ``*-Direct-Warp-v0`` tasks register their own env class and
# are not cfg-adapted, so they are excluded here.
_MANAGER_WARP_ENTRY_POINT = "isaaclab_experimental.envs:ManagerBasedRLEnvWarp"


def _manager_warp_tasks() -> list[tuple[str, str]]:
    """Return ``(task_id, env_cfg_entry_point)`` for every manager-based warp task."""
    tasks: list[tuple[str, str]] = []
    for task_id, spec in gym.registry.items():
        if spec.entry_point != _MANAGER_WARP_ENTRY_POINT:
            continue
        cfg_entry = (spec.kwargs or {}).get("env_cfg_entry_point")
        if isinstance(cfg_entry, str):
            tasks.append((task_id, cfg_entry))
    return sorted(tasks)


def _instantiate_cfg(cfg_entry_point: str):
    """Import ``module:Class`` and instantiate the stable env cfg."""
    module_path, class_name = cfg_entry_point.split(":")
    return getattr(importlib.import_module(module_path), class_name)()


_MANAGER_WARP_TASKS = _manager_warp_tasks()

# Tasks whose stable cfg still includes a warp-managed term with no warp twin.
# These are tracked for implementation; delete the entry once the twin lands
# (xfail is strict, so an unexpected pass fails the suite and flags the cleanup).
_PENDING_WARP_TWINS = {
    "Isaac-Cartpole-Warp-v0": "reward 'survival_success_rate'",
    "Isaac-Ant-Warp-v0": "observation 'body_incoming_wrench'",
    "Isaac-Humanoid-Warp-v0": "observation 'body_incoming_wrench'",
}


def _params():
    cases = []
    for task_id, cfg_entry_point in _MANAGER_WARP_TASKS:
        marks = ()
        if task_id in _PENDING_WARP_TWINS:
            marks = pytest.mark.xfail(
                strict=True,
                reason=f"pending warp twin for {_PENDING_WARP_TWINS[task_id]}",
            )
        cases.append(pytest.param(task_id, cfg_entry_point, marks=marks, id=task_id))
    return cases


def test_manager_warp_tasks_are_registered():
    """Sanity: the package actually registered manager-based warp tasks."""
    assert _MANAGER_WARP_TASKS, "no manager-based '*-Warp-v0' tasks registered"


@pytest.mark.parametrize("task_id, cfg_entry_point", _params())
def test_stable_cfg_adapts_to_warp(task_id: str, cfg_entry_point: str):
    """The stable cfg behind each warp task adapts without a missing twin."""
    cfg = _instantiate_cfg(cfg_entry_point)
    # The warp env requires Newton physics (normally via ``presets=newton_mjwarp``);
    # set it directly so the test does not depend on Hydra preset resolution.
    cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1)
    # Raises FrontendIncompatibleError if any warp-managed term lacks a warp twin.
    adapt_cfg_for_warp(cfg)
