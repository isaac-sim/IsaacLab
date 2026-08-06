# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test: the ``IsaacContrib-Factory-Franka`` env cfg constructs.

Used as a fast integration safety net before / after structural
refactors of the nist subpackage. Mirrors what the hydra pipeline
does up to (but not including) Kit initialisation:

1. Triggers gym task registration via ``import isaaclab_tasks``.
2. Resolves ``env_cfg_entry_point`` from the gym registry.
3. Instantiates the env cfg class.
4. Round-trips the cfg through ``cfg.to_dict()`` (the same path hydra
   feeds to ``OmegaConf.create`` in :func:`register_task`). Catches
   serialisation issues that show up only when training launches --
   e.g. nested cfgs whose annotations OmegaConf rejects, or
   ``class_type`` fields not registered as ``ResolvableString``.

If any of the above breaks (broken import path, missing module after a
move, cfg construction error, OmegaConf-incompatible annotation), this
test catches it without paying the cost of the heavier
``test_env_cfg_no_forbidden_imports.py``.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

import isaaclab_tasks  # noqa: F401 -- registers the gym tasks


@pytest.mark.parametrize("task_name", ["IsaacContrib-Factory-Franka"])
def test_env_cfg_constructs(task_name: str) -> None:
    """The env cfg referenced by ``env_cfg_entry_point`` imports + constructs."""
    spec = gym.spec(task_name)
    entry = spec.kwargs["env_cfg_entry_point"]
    module_path, cls_name = entry.split(":")
    cfg_cls = getattr(importlib.import_module(module_path), cls_name)
    cfg = cfg_cls()
    # Standard manager-based env cfg fields. If a structural move left a
    # field unresolved (e.g. curriculum cfg pointing at a missing module),
    # construction would have raised before this assertion.
    assert hasattr(cfg, "scene")
    assert hasattr(cfg, "actions")
    assert hasattr(cfg, "observations")
    assert hasattr(cfg, "events")


def test_factory_accumulator_success_rate_callback_targets_monitor_success_rate() -> None:
    """Factory accumulator curriculum should bind the reset monitor tensor."""
    spec = gym.spec("IsaacContrib-Factory-Franka")
    module_path, cls_name = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg_cls = getattr(importlib.import_module(module_path), cls_name)
    cfg = cfg_cls()
    callback = cfg.curriculum.difficulty_scheduler.params["success_rate_callback"]

    expected = "env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate"
    assert callback.default == expected
    assert callback.accumulator == expected

    rates = torch.tensor([0.5, 1.0])
    reset_accumulator = SimpleNamespace(monitor_success_rate=rates)
    eval_env = SimpleNamespace(
        event_manager=SimpleNamespace(get_term_cfg=lambda _name: SimpleNamespace(func=reset_accumulator))
    )
    assert eval(callback.accumulator, {}, {"env": eval_env}) is rates  # noqa: S307


def test_factory_difficulty_scheduler_waits_for_accumulator_rates() -> None:
    """Initial reset may run curriculum before accumulator reset materializes rates."""
    from isaaclab_tasks.contrib.nist.mdp.curriculums import DifficultyScheduler

    scheduler = DifficultyScheduler.__new__(DifficultyScheduler)
    scheduler.current_adr_difficulties = torch.ones(3) * 2
    scheduler.difficulty_frac = torch.tensor(0.2)
    reset_accumulator = SimpleNamespace(monitor_success_rate=None)
    env = SimpleNamespace(
        device=torch.device("cpu"),
        event_manager=SimpleNamespace(get_term_cfg=lambda _name: SimpleNamespace(func=reset_accumulator)),
    )

    result = DifficultyScheduler.__call__(
        scheduler,
        env,
        torch.arange(3),
        "env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
        max_difficulty=10,
    )

    assert result is scheduler.difficulty_frac
    torch.testing.assert_close(scheduler.current_adr_difficulties, torch.ones(3) * 2)


def test_factory_difficulty_scheduler_averages_ready_success_rates() -> None:
    """Difficulty scheduler should average bound rate tensors internally."""
    from isaaclab_tasks.contrib.nist.mdp.curriculums import DifficultyScheduler

    scheduler = DifficultyScheduler.__new__(DifficultyScheduler)
    scheduler.current_adr_difficulties = torch.ones(3) * 2
    scheduler.difficulty_frac = torch.tensor(0.2)
    reset_accumulator = SimpleNamespace(monitor_success_rate=torch.ones(4))
    env = SimpleNamespace(
        device=torch.device("cpu"),
        event_manager=SimpleNamespace(get_term_cfg=lambda _name: SimpleNamespace(func=reset_accumulator)),
    )

    result = DifficultyScheduler.__call__(
        scheduler,
        env,
        torch.arange(3),
        "env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
        max_difficulty=10,
    )

    torch.testing.assert_close(scheduler.current_adr_difficulties, torch.ones(3) * 3)
    torch.testing.assert_close(result, torch.tensor(0.3))


@pytest.mark.parametrize("task_name", ["IsaacContrib-Factory-Franka"])
def test_env_cfg_to_dict_serialises(task_name: str) -> None:
    """``cfg.to_dict()`` produces a fully-flattened dict with no dataclass instances.

    Hydra's :func:`register_task` feeds ``cfg.to_dict()`` into
    ``OmegaConf.create``. If any nested dataclass cfg uses an
    annotation OmegaConf can't validate (e.g. ``type[X]`` without ``| str``,
    or untyped containers leaving dataclass instances exposed), the
    training launch crashes during ``register_task``. This test checks
    that the dict is OmegaConf-clean by asserting no dataclass instances
    leak through.
    """
    spec = gym.spec(task_name)
    module_path, cls_name = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg_cls = getattr(importlib.import_module(module_path), cls_name)
    cfg = cfg_cls()
    cfg_dict = cfg.to_dict()

    def _assert_no_dataclasses(value, path: str = "") -> None:
        if hasattr(value, "__dataclass_fields__"):
            raise AssertionError(
                f"Unflattened dataclass {type(value).__name__} at {path!r} -- "
                "this would crash OmegaConf during hydra's register_task."
            )
        if isinstance(value, dict):
            for k, v in value.items():
                _assert_no_dataclasses(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _assert_no_dataclasses(v, f"{path}[{i}]")

    _assert_no_dataclasses(cfg_dict)


@pytest.mark.parametrize(
    "module_path",
    [
        # Pure-Python leaf modules that the restructure relocated. Importing
        # them catches stale ``from .X import Y`` references where ``X`` is
        # now a sibling-package or has been renamed. Modules that pull in
        # Kit / Newton / USD must NOT be added here -- they segfault when
        # imported outside a launched Kit app.
        "isaaclab_tasks.contrib.nist.utils.sampling",
        "isaaclab_tasks.contrib.nist.utils.sampling.sampler",
        "isaaclab_tasks.contrib.nist.utils.sampling.sampler_cfg",
        "isaaclab_tasks.contrib.nist.utils.sampling.sampling_strategies",
        "isaaclab_tasks.contrib.nist.utils.sampling.sampling_strategies_cfg",
        "isaaclab_tasks.contrib.nist.utils.state_layout",
        "isaaclab_tasks.contrib.nist.utils.reset_state",
        "isaaclab_tasks.contrib.nist.assembly_profile",
        "isaaclab_tasks.contrib.nist.assembly_profile_cfg",
    ],
)
def test_relocated_module_imports(module_path: str) -> None:
    """Modules touched by the directory restructure import without errors.

    A targeted version of "import everything" -- limited to pure-Python
    leaf modules so the test doesn't pull in Kit/Newton-bound modules
    that segfault outside a launched Kit app. Catches stale relative
    imports that the restructure left behind (e.g.
    ``from .reset_state import X`` after ``reset_state.py`` moved
    sibling).
    """
    importlib.import_module(module_path)
