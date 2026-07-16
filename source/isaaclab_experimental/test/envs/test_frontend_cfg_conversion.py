# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coverage tests for warp task configuration adaptation.

Two sweeps, both running :meth:`WarpFrontend.adapt_cfg` — the same adaptation
:class:`~isaaclab_experimental.envs.ManagerBasedRLEnvWarp` runs in its
``__init__``:

* Every *stable* task id with declared warp twin coverage: resolve the
  ``newton_mjwarp`` preset and adapt the stable cfg, exactly what
  ``--frontend warp`` does.
* Every registered manager-based ``*-Warp-v0`` variant (velocity deltas whose
  stable cfgs still contain terms without warp twins): load its env cfg,
  resolve presets, and adapt.

A dedicated stable Cartpole case also guards the stable-to-experimental
module routing used by ``--frontend warp``.
"""

from __future__ import annotations

import importlib

import gymnasium as gym
import isaaclab_tasks_experimental  # noqa: F401
import pytest
from isaaclab_experimental.envs.frontend import WarpFrontend
from isaaclab_newton.physics import NewtonCfg

# Registering the task packages is the whole point — import for side effects.
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets

# Stable task ids whose MDP terms are fully twinned: ``--frontend warp`` adapts
# their cfgs directly, with no parallel ``*-Warp-v0`` registration. Extend this
# list when a new task family gains full twin coverage.
_STABLE_TASKS_WITH_WARP_COVERAGE = [
    "Isaac-Ant",
    "Isaac-Cartpole",
    "Isaac-Humanoid",
    "Isaac-Reach-Franka",
    "Isaac-Reach-Franka-Play",
    "Isaac-Reach-UR10",
    "Isaac-Reach-UR10-Play",
]

# Manager-based warp tasks are exactly those whose entry point is the shared warp
# env class; direct tasks provide their own env class (via ``warp_entry_point``
# or a warp-native registration) and are not cfg-adapted, so they are excluded here.
_MANAGER_WARP_ENTRY_POINT = "isaaclab_experimental.envs:ManagerBasedRLEnvWarp"

_WARP_ROOTS = ("isaaclab_experimental", "isaaclab_tasks_experimental")


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


def _load_adapted_cfg(cfg_entry_point: str):
    """Instantiate an env cfg, resolve the Newton preset, and adapt it for warp."""
    module_path, class_name = cfg_entry_point.split(":")
    cfg = getattr(importlib.import_module(module_path), class_name)()
    cfg = resolve_presets(cfg, selected=("newton_mjwarp",))
    assert isinstance(cfg.sim.physics, NewtonCfg), "task does not provide a newton_mjwarp physics preset"
    # Raises FrontendIncompatibleError if any warp-managed term lacks a warp twin.
    WarpFrontend.adapt_cfg(cfg)
    return cfg


def _cfg_entry_point(task_id: str) -> str:
    cfg_entry = (gym.spec(task_id).kwargs or {}).get("env_cfg_entry_point")
    assert isinstance(cfg_entry, str), f"{task_id}: no env_cfg_entry_point"
    return cfg_entry


_MANAGER_WARP_TASKS = _manager_warp_tasks()


def test_manager_warp_tasks_are_registered():
    """Sanity: every expected velocity warp task variant is registered."""
    expected = {
        f"Isaac-Velocity-Flat-{robot}-Warp{suffix}-v0"
        for robot in ("AnymalB", "AnymalC", "AnymalD", "Cassie", "G1", "H1", "UnitreeA1", "UnitreeGo1", "UnitreeGo2")
        for suffix in ("", "-Play")
    }
    registered = {task_id for task_id, _ in _MANAGER_WARP_TASKS}
    assert registered == expected, f"missing: {sorted(expected - registered)}; extra: {sorted(registered - expected)}"


@pytest.mark.parametrize("task_id", _STABLE_TASKS_WITH_WARP_COVERAGE, ids=_STABLE_TASKS_WITH_WARP_COVERAGE)
def test_stable_task_cfg_adapts_to_warp(task_id: str):
    """Each covered stable task adapts without a missing twin (the --frontend warp path)."""
    _load_adapted_cfg(_cfg_entry_point(task_id))


@pytest.mark.parametrize(
    "task_id, cfg_entry_point",
    [pytest.param(task_id, cfg_entry, id=task_id) for task_id, cfg_entry in _MANAGER_WARP_TASKS],
)
def test_registered_warp_variant_cfg_adapts(task_id: str, cfg_entry_point: str):
    """Each registered ``*-Warp-v0`` variant cfg adapts without a missing twin."""
    cfg = _load_adapted_cfg(cfg_entry_point)

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


def _direct_tasks_with_warp_entry_point() -> list[tuple[str, str]]:
    """Return ``(task_id, warp_entry_point)`` for stable direct tasks declaring a warp env."""
    tasks: list[tuple[str, str]] = []
    for task_id, spec in gym.registry.items():
        target = (spec.kwargs or {}).get("warp_entry_point")
        if isinstance(target, str):
            tasks.append((task_id, target))
    return sorted(tasks)


_DIRECT_WARP_TASKS = _direct_tasks_with_warp_entry_point()


def test_direct_tasks_declare_expected_warp_entry_points():
    """Sanity: the stable direct tasks with warp implementations declare them."""
    declared = {task_id for task_id, _ in _DIRECT_WARP_TASKS}
    assert {"Isaac-Cartpole-Direct", "Isaac-Ant-Direct", "Isaac-Humanoid-Direct"} <= declared


@pytest.mark.parametrize(
    "task_id, warp_entry_point",
    [pytest.param(task_id, target, id=task_id) for task_id, target in _DIRECT_WARP_TASKS],
)
def test_declared_direct_warp_entry_point_resolves(task_id: str, warp_entry_point: str):
    """Each declared ``warp_entry_point`` imports and names a warp-rooted class."""
    module_path, class_name = warp_entry_point.split(":")
    env_class = getattr(importlib.import_module(module_path), class_name)
    assert isinstance(env_class, type), f"{task_id}: {warp_entry_point} is not a class"
    assert env_class.__module__.startswith(_WARP_ROOTS)


def test_stable_cartpole_cfg_adapts_to_current_warp_module_layout():
    """The stable Cartpole cfg resolves task-specific twins in the current package layout."""
    from isaaclab_experimental.managers.action_manager import ActionTerm

    cfg = _load_adapted_cfg(_cfg_entry_point("Isaac-Cartpole"))

    assert cfg.rewards.pole_pos.func.__module__.startswith("isaaclab_tasks_experimental.core.cartpole.mdp")
    assert cfg.rewards.success_rate.func.__module__.startswith("isaaclab_tasks_experimental.core.cartpole.mdp")
    assert issubclass(cfg.actions.joint_effort.class_type, ActionTerm)
