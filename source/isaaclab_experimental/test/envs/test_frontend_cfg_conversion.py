# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coverage tests for manager-based Warp task configuration adaptation.

For each registered manager-based Warp task, load its environment configuration,
force Newton physics, and run :func:`adapt_cfg_for_warp` — the same adaptation the
Warp environment runs in its ``__init__``. A dedicated stable Cartpole case also
guards the stable-to-experimental module routing used by ``--frontend warp``.
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


def _params():
    return [pytest.param(task_id, cfg_entry_point, id=task_id) for task_id, cfg_entry_point in _MANAGER_WARP_TASKS]


def test_manager_warp_tasks_are_registered():
    """Sanity: the package actually registered manager-based warp tasks."""
    assert _MANAGER_WARP_TASKS, "no manager-based '*-Warp-v0' tasks registered"


_WARP_ROOTS = ("isaaclab_experimental", "isaaclab_tasks_experimental")


@pytest.mark.parametrize("task_id, cfg_entry_point", _params())
def test_registered_cfg_adapts_to_warp(task_id: str, cfg_entry_point: str):
    """Each registered manager-based Warp cfg adapts without a missing twin."""
    cfg = _instantiate_cfg(cfg_entry_point)
    # The warp env requires Newton physics (normally via ``presets=newton_mjwarp``);
    # set it directly so the test does not depend on Hydra preset resolution.
    cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1)
    # Raises FrontendIncompatibleError if any warp-managed term lacks a warp twin.
    adapt_cfg_for_warp(cfg)

    # Action terms carry a ``class_type`` (not a ``func``) and live on a base that
    # is not a ManagerTermBaseCfg; guard that the adapter still swaps them to the
    # warp ActionTerm, otherwise the warp ActionManager rejects them at runtime.
    actions = getattr(cfg, "actions", None)
    if actions is not None:
        for name, term in vars(actions).items():
            class_type = getattr(term, "class_type", None)
            if class_type is not None:
                assert class_type.__module__.startswith(_WARP_ROOTS), (
                    f"{task_id}: action term '{name}' class_type was not swapped to a warp twin"
                    f" (got {class_type.__module__}.{class_type.__name__})"
                )


def test_stable_cartpole_cfg_adapts_to_current_warp_module_layout():
    """The stable Cartpole cfg resolves task-specific twins in the current package layout."""
    from isaaclab_experimental.managers.action_manager import ActionTerm

    from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg

    cfg = CartpoleEnvCfg()
    cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1)

    adapt_cfg_for_warp(cfg)

    assert cfg.rewards.pole_pos.func.__module__.startswith(
        "isaaclab_tasks_experimental.manager_based.classic.cartpole.mdp"
    )
    assert cfg.rewards.success_rate.func.__module__.startswith(
        "isaaclab_tasks_experimental.manager_based.classic.cartpole.mdp"
    )
    assert issubclass(cfg.actions.joint_effort.class_type, ActionTerm)
