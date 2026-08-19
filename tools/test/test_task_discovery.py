# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the registry-independent parts of task discovery.

The registry walk and selector detection need Isaac Lab importable and are
exercised by running the tool; everything here is pure and runs offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _bootstrap_paths() -> None:
    """Prepend ``tools/`` so the module imports the same way the tool does."""
    tools_dir = Path(__file__).resolve().parents[1]
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))


_bootstrap_paths()

import task_discovery  # noqa: E402
from task_discovery import (  # noqa: E402
    DiscoveredTask,
    DiscoveryError,
    _build_modes,
    _domain_presets,
    _rl_libraries_from_kwargs,
    is_training_task,
)

Mode = DiscoveredTask.Mode


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"rsl_rl_cfg_entry_point": "x"}, ("rsl_rl",)),
        # Variant entry points belong to their library; matching the exact name
        # would drop every recurrent, distillation and per-terrain config.
        ({"rsl_rl_recurrent_cfg_entry_point": "x"}, ("rsl_rl",)),
        ({"skrl_flat_ppo_cfg_entry_point": "x"}, ("skrl",)),
        # Ordering follows RL_LIBRARY_PRIORITY, not registration order.
        ({"skrl_cfg_entry_point": "x", "rsl_rl_cfg_entry_point": "x"}, ("rsl_rl", "skrl")),
        # The env config is not an agent config.
        ({"env_cfg_entry_point": "x"}, ()),
        ({}, ()),
    ],
)
def test_rl_libraries_are_read_from_entry_point_stems(kwargs: dict, expected: tuple[str, ...]) -> None:
    assert _rl_libraries_from_kwargs(kwargs) == expected


@pytest.mark.parametrize(
    ("task_id", "expected"),
    [
        ("Isaac-Ant", True),
        ("IsaacContrib-Walk", True),
        ("Isaac-Ant-Eval", False),
        ("Isaac-Benchmark-Cartpole", False),
        ("Some-Other-Env", False),
    ],
)
def test_only_trainable_isaac_tasks_are_walked(task_id: str, expected: bool) -> None:
    assert is_training_task(task_id) is expected


def test_domain_presets_drop_names_the_task_also_declares_as_a_typed_selector() -> None:
    # Reachable as ``physics=ovphysx`` / ``renderer=ovrtx``, so reporting them again
    # as ``presets=`` tokens would double-count the same run.
    names = ["rgb", "ovphysx", "depth", "ovrtx"]

    assert _domain_presets(names, ("ovphysx", "ovrtx")) == ("depth", "rgb")


def test_a_backend_exposed_only_as_a_domain_preset_is_kept() -> None:
    # Isaac-Open-Drawer-Franka buckets its backends under DOMAIN because their cfg
    # classes do not subclass PhysicsCfg, so ``presets=newton_mjwarp`` is the only
    # way to select them and ``physics=newton_mjwarp`` is rejected. Dropping them by
    # name left the task reporting one mode carrying no tokens, hiding four runs.
    names = ["isaacsim_physx", "newton_kamino", "newton_mjwarp", "ovphysx", "physx"]

    assert _domain_presets(names, ()) == tuple(sorted(names))


def test_modes_are_the_raw_cross_product_when_resolution_is_skipped() -> None:
    # Nothing has been resolved, so nothing can be collapsed and there is no default.
    modes, default = _build_modes("Isaac-X", ("physx", "newton_mjwarp"), ("ovrtx",), (), resolve=False)

    assert modes == (Mode("physx", "ovrtx", None), Mode("newton_mjwarp", "ovrtx", None))
    assert default is None


def test_a_task_without_presets_gets_one_mode_carrying_no_tokens() -> None:
    assert _build_modes("Isaac-X", (), (), (), resolve=False)[0] == (Mode(None, None, None),)


def test_domain_presets_are_expanded_one_at_a_time_beside_the_default() -> None:
    # Presets targeting the same field conflict, so they are never combined; the
    # ``None`` entry keeps the task's own default reachable.
    modes, _ = _build_modes("Isaac-X", (), (), ("rgb", "depth"), resolve=False)

    assert modes == (Mode(None, None, None), Mode(None, None, "rgb"), Mode(None, None, "depth"))


def _resolver(runs: dict[tuple[str | None, str | None, str | None], tuple[str, str]]):
    """Return a ``_mode_resolves`` stand-in driven by a combination -> run table."""

    def fake(task_id, physics, renderer, presets=None):
        return runs.get((physics, renderer, presets))

    return fake


def test_spellings_that_resolve_to_the_same_run_collapse_to_one_mode(monkeypatch) -> None:
    # Isaac-Open-Drawer-Franka in miniature: passing nothing lands on the same config
    # as naming the preset the task already defaults to, and the ``physx`` alias lands
    # on the same config as the concrete backend it resolves to.
    monkeypatch.setattr(
        task_discovery,
        "_mode_resolves",
        _resolver(
            {
                (None, None, None): ("fp-physx", "PhysxCfg"),
                (None, None, "isaacsim_physx"): ("fp-physx", "PhysxCfg"),
                (None, None, "ovphysx"): ("fp-ov", "OvPhysxCfg"),
                (None, None, "physx"): ("fp-ov", "OvPhysxCfg"),
                (None, None, "newton_mjwarp"): ("fp-newton", "NewtonCfg(MJWarpSolverCfg)"),
            }
        ),
    )

    modes, default = _build_modes(
        "Isaac-X", (), (), ("isaacsim_physx", "newton_mjwarp", "ovphysx", "physx"), resolve=True
    )

    # Five spellings, three distinct runs.
    assert modes == (
        Mode(None, None, "isaacsim_physx"),
        Mode(None, None, "newton_mjwarp"),
        Mode(None, None, "ovphysx"),
    )
    assert default == DiscoveredTask.Default(backend="PhysxCfg", mode=Mode(None, None, "isaacsim_physx"))


def test_runs_sharing_a_backend_but_not_a_config_stay_separate(monkeypatch) -> None:
    # Isaac-Reach-Franka's controller presets all run on Newton MJWarp and are four
    # different runs, so collapsing on the backend instead of the config is wrong.
    monkeypatch.setattr(
        task_discovery,
        "_mode_resolves",
        _resolver(
            {
                (None, None, None): ("fp-joint", "NewtonCfg(MJWarpSolverCfg)"),
                (None, None, "joint_pos"): ("fp-joint", "NewtonCfg(MJWarpSolverCfg)"),
                (None, None, "diffik"): ("fp-diffik", "NewtonCfg(MJWarpSolverCfg)"),
            }
        ),
    )

    modes, default = _build_modes("Isaac-X", (), (), ("joint_pos", "diffik"), resolve=True)

    assert modes == (Mode(None, None, "joint_pos"), Mode(None, None, "diffik"))
    assert default.mode == Mode(None, None, "joint_pos")


def test_a_default_that_matches_no_named_preset_survives_as_its_own_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        task_discovery,
        "_mode_resolves",
        _resolver(
            {
                (None, None, None): ("fp-own", "PhysxCfg"),
                (None, None, "rgb"): ("fp-rgb", "PhysxCfg"),
            }
        ),
    )

    modes, default = _build_modes("Isaac-X", (), (), ("rgb",), resolve=True)

    assert modes == (Mode(None, None, None), Mode(None, None, "rgb"))
    assert default == DiscoveredTask.Default(backend="PhysxCfg", mode=Mode(None, None, None))


def _validator_broken_on(backend: str):
    """Return a ``_mode_resolves`` stand-in that fails structurally on *backend*."""

    def fake(task_id, physics, renderer, presets=None):
        if physics == backend:
            raise DiscoveryError("AttributeError: no attribute 'solver_cfg'")
        return ("fp", "PhysxCfg")

    return fake


def test_a_combination_the_validator_cannot_judge_is_dropped_and_the_walk_continues(monkeypatch) -> None:
    # One task whose config breaks the validator must not cost the caller every
    # other task in the registry. Unknown is not legal either, so the combination
    # is dropped rather than reported.
    monkeypatch.setattr(task_discovery, "_mode_resolves", _validator_broken_on("newton_mjwarp"))

    modes, _ = _build_modes("Isaac-X", ("physx", "newton_mjwarp"), (), (), resolve=True)

    assert modes == (Mode("physx", None, None),)


def test_strict_raises_on_a_structural_failure_instead_of_dropping_it(monkeypatch) -> None:
    # Callers policing Isaac Lab API drift want the canary, not a survivable walk.
    monkeypatch.setattr(task_discovery, "_mode_resolves", _validator_broken_on("newton_mjwarp"))

    with pytest.raises(DiscoveryError):
        _build_modes("Isaac-X", ("physx", "newton_mjwarp"), (), (), resolve=True, strict=True)


def test_uncollapsed_mode_keeps_every_validated_spelling(monkeypatch) -> None:
    # A preset naming the default of its own axis resolves to the default config, so
    # the collapse drops it -- but it is still a token a reader can type, so the
    # documentation view has to keep it.
    monkeypatch.setattr(
        task_discovery,
        "_mode_resolves",
        _resolver(
            {
                (None, None, None): ("fp-default", "PhysxCfg"),
                (None, None, "shapes"): ("fp-default", "PhysxCfg"),
                (None, None, "cube"): ("fp-cube", "PhysxCfg"),
            }
        ),
    )

    collapsed, _ = _build_modes("Isaac-X", (), (), ("shapes", "cube"), resolve=True)
    every, default = _build_modes("Isaac-X", (), (), ("shapes", "cube"), resolve=True, collapse=False)

    assert collapsed == (Mode(None, None, "shapes"), Mode(None, None, "cube"))
    assert every == (Mode(None, None, None), Mode(None, None, "shapes"), Mode(None, None, "cube"))
    # The default still names an explicit spelling, not the bare no-token mode.
    assert default == DiscoveredTask.Default(backend="PhysxCfg", mode=Mode(None, None, "shapes"))


def test_uncollapsed_mode_still_drops_combinations_the_validator_rejects(monkeypatch) -> None:
    monkeypatch.setattr(
        task_discovery,
        "_mode_resolves",
        _resolver(
            {
                (None, None, None): ("fp-default", "PhysxCfg"),
                (None, None, "rgb"): ("fp-rgb", "PhysxCfg"),
            }
        ),
    )

    every, _ = _build_modes("Isaac-X", (), (), ("rgb", "raycaster_depth"), resolve=True, collapse=False)

    assert Mode(None, None, "raycaster_depth") not in every
