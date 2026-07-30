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
    "Isaac-Reach-UR10",
    "Isaac-Velocity-Flat-AnymalD",
    "Isaac-Velocity-Flat-Cassie",
    "Isaac-Velocity-Flat-G1",
    "Isaac-Velocity-Flat-H1",
    "Isaac-Velocity-Flat-UnitreeGo2",
]

# Manager-based warp tasks are exactly those whose entry point is the shared warp
# env class; direct tasks provide their own env class (resolved by name-based
# mirror, or an explicit ``warp_entry_point`` override) and are not cfg-adapted,
# so they are excluded here.
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


def test_no_manager_warp_variants_remain():
    """End-state pin: manager-based warp execution needs no parallel registrations."""
    assert _manager_warp_tasks() == [], "unexpected ManagerBasedRLEnvWarp registrations reappeared"


def test_no_warp_task_registrations_remain():
    """End-state pin: warp execution needs no parallel ``-Warp`` task ids at all."""
    warp_ids = sorted(task_id for task_id in gym.registry if "-Warp" in task_id)
    assert warp_ids == [], f"unexpected warp task registrations: {warp_ids}"


@pytest.mark.parametrize("task_id", _STABLE_TASKS_WITH_WARP_COVERAGE, ids=_STABLE_TASKS_WITH_WARP_COVERAGE)
def test_stable_task_cfg_adapts_to_warp(task_id: str):
    """Each covered stable task adapts without a missing twin (the --frontend warp path)."""
    cfg = _load_adapted_cfg(_cfg_entry_point(task_id))

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


_DIRECT_WARP_TASKS = [
    "Isaac-Cartpole-Direct",
    "Isaac-Ant-Direct",
    "Isaac-Humanoid-Direct",
    "Isaac-Reorient-Cube-Allegro-Direct",
]


@pytest.mark.parametrize("task_id", _DIRECT_WARP_TASKS)
def test_direct_task_resolves_to_warp_env_class_by_name(task_id: str):
    """Stable direct tasks resolve their warp env class by name-based mirror
    resolution alone — no ``warp_entry_point`` declaration is needed."""
    from isaaclab_experimental.envs.frontend import WarpFrontend

    assert (gym.spec(task_id).kwargs or {}).get("warp_entry_point") is None
    env_class = WarpFrontend._resolve_direct_warp_class(task_id)
    assert isinstance(env_class, type), f"{task_id}: no warp env class resolved"
    assert env_class.__module__.startswith(_WARP_ROOTS)
    assert env_class.__name__.endswith("WarpEnv")


def test_stable_cartpole_cfg_adapts_to_current_warp_module_layout():
    """The stable Cartpole cfg resolves task-specific twins in the current package layout."""
    from isaaclab_experimental.managers.action_manager import ActionTerm

    cfg = _load_adapted_cfg(_cfg_entry_point("Isaac-Cartpole"))

    assert cfg.rewards.pole_pos.func.__module__.startswith("isaaclab_tasks_experimental.core.cartpole.mdp")
    assert cfg.rewards.success_rate.func.__module__.startswith("isaaclab_tasks_experimental.core.cartpole.mdp")
    assert issubclass(cfg.actions.joint_effort.class_type, ActionTerm)


def _iter_obs_terms(cfg):
    """Yield (name, term cfg) for every observation term that carries a noise slot."""
    from isaaclab.managers.manager_term_cfg import ObservationGroupCfg

    for group in vars(cfg.observations).values():
        if not isinstance(group, ObservationGroupCfg):
            continue
        for name, term in vars(group).items():
            if hasattr(term, "noise"):
                yield name, term


def test_stable_observation_noise_converts_to_warp_twins():
    """Stable noise cfgs (e.g. Unoise) become warp-native twins with the same parameters."""
    from isaaclab.utils import noise as stable_noise

    entry = _cfg_entry_point("Isaac-Velocity-Flat-UnitreeGo2")
    module_path, class_name = entry.split(":")
    raw = getattr(importlib.import_module(module_path), class_name)()
    stable_params = {
        name: (term.noise.n_min, term.noise.n_max)
        for name, term in _iter_obs_terms(raw)
        if isinstance(term.noise, stable_noise.UniformNoiseCfg)
    }
    assert stable_params, "expected uniform noise on the stable velocity observations"

    cfg = _load_adapted_cfg(entry)
    converted = dict(_iter_obs_terms(cfg))
    for name, (n_min, n_max) in stable_params.items():
        twin = converted[name].noise
        assert type(twin).__module__.startswith("isaaclab_experimental."), name
        assert (twin.n_min, twin.n_max) == (n_min, n_max), name


def test_noise_cfg_without_warp_twin_is_a_hard_error():
    """Class-based noise models have no warp twin yet: adapting must fail loudly."""
    from isaaclab_experimental.envs.frontend import FrontendIncompatibleError

    from isaaclab.utils.noise import NoiseModelCfg, UniformNoiseCfg

    entry = _cfg_entry_point("Isaac-Velocity-Flat-UnitreeGo2")
    module_path, class_name = entry.split(":")
    cfg = getattr(importlib.import_module(module_path), class_name)()
    cfg = resolve_presets(cfg, selected=("newton_mjwarp",))
    name, term = next(iter(_iter_obs_terms(cfg)))
    term.noise = NoiseModelCfg(noise_cfg=UniformNoiseCfg())
    with pytest.raises(FrontendIncompatibleError, match="noise"):
        WarpFrontend.adapt_cfg(cfg)
