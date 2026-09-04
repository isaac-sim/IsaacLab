# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the parts of task discovery that need no Isaac Lab.

Resolution is stubbed here so the module stays importable on pytest alone; the real
resolver is covered by ``test_task_discovery_resolve.py``.
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
Default = DiscoveredTask.Default


@pytest.fixture
def resolver(monkeypatch):
    """Install a ``_mode_resolves`` driven by a ``combination -> (fingerprint, backend)`` table.

    A combination absent from the table resolves to ``None``, i.e. cannot run.
    """

    def install(runs):
        monkeypatch.setattr(
            task_discovery,
            "_mode_resolves",
            lambda task, physics, renderer, presets=None: runs.get((physics, renderer, presets)),
        )

    return install


@pytest.fixture
def broken_validator(monkeypatch):
    """Install a ``_mode_resolves`` that fails structurally on one physics token."""

    def install(backend):
        def fake(task, physics, renderer, presets=None):
            if physics == backend:
                raise DiscoveryError("AttributeError: no attribute 'solver_cfg'")
            return ("fp", "PhysxCfg")

        monkeypatch.setattr(task_discovery, "_mode_resolves", fake)

    return install


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"rsl_rl_cfg_entry_point": "x"}, ("rsl_rl",)),
        # Variants belong to their library; exact-name matching would drop every
        # recurrent, distillation and per-terrain config.
        ({"rsl_rl_recurrent_cfg_entry_point": "x"}, ("rsl_rl",)),
        ({"skrl_flat_ppo_cfg_entry_point": "x"}, ("skrl",)),
        # Ordering follows RL_LIBRARY_PRIORITY, not registration order.
        ({"skrl_cfg_entry_point": "x", "rsl_rl_cfg_entry_point": "x"}, ("rsl_rl", "skrl")),
        # The env config is not an agent config.
        ({"env_cfg_entry_point": "x"}, ()),
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


@pytest.mark.parametrize(
    ("names", "typed", "expected"),
    [
        # Declared on a typed axis too, so reachable as ``physics=``/``renderer=`` and
        # reporting them again as ``presets=`` would double-count the same run.
        (["rgb", "ovphysx", "depth", "ovrtx"], ("ovphysx", "ovrtx"), ("depth", "rgb")),
        # Isaac-Open-Drawer-Franka: backends bucket under DOMAIN because their cfg
        # classes do not subclass PhysicsCfg, so ``presets=`` is the only way to select
        # them. Dropping them by name left the task reporting one empty mode.
        (["newton_mjwarp", "ovphysx", "physx"], (), ("newton_mjwarp", "ovphysx", "physx")),
    ],
)
def test_domain_presets_drop_only_what_a_typed_axis_already_offers(names, typed, expected) -> None:
    assert _domain_presets(names, typed) == expected


@pytest.mark.parametrize(
    ("physics", "renderers", "domains", "expected"),
    [
        # Renderers are expanded across: a camera task reported headless-only would omit
        # the thing under test.
        (
            ("physx", "newton_mjwarp"),
            ("ovrtx",),
            (),
            (Mode("physx", "ovrtx", None), Mode("newton_mjwarp", "ovrtx", None)),
        ),
        # Declaring nothing still leaves one way to run.
        ((), (), (), (Mode(None, None, None),)),
        # Domain presets go one at a time, never combined; ``None`` keeps the task's own
        # default reachable beside them.
        ((), (), ("rgb", "depth"), (Mode(None, None, None), Mode(None, None, "rgb"), Mode(None, None, "depth"))),
    ],
)
def test_declared_modes_are_the_raw_cross_product(physics, renderers, domains, expected) -> None:
    # Nothing has been resolved, so nothing can be collapsed and there is no default.
    assert _build_modes("Isaac-X", physics, renderers, domains, resolve=False) == (expected, None)


def test_spellings_that_resolve_to_the_same_run_collapse_to_one_mode(resolver) -> None:
    # Isaac-Open-Drawer-Franka in miniature: passing nothing lands on the same config as
    # naming the preset the task already defaults to, and ``physx`` lands on the same
    # config as the concrete backend it aliases.
    resolver(
        {
            (None, None, None): ("fp-physx", "PhysxCfg"),
            (None, None, "isaacsim_physx"): ("fp-physx", "PhysxCfg"),
            (None, None, "ovphysx"): ("fp-ov", "OvPhysxCfg"),
            (None, None, "physx"): ("fp-ov", "OvPhysxCfg"),
            (None, None, "newton_mjwarp"): ("fp-newton", "NewtonCfg(MJWarpSolverCfg)"),
        }
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
    assert default == Default(backend="PhysxCfg", mode=Mode(None, None, "isaacsim_physx"))


def test_runs_sharing_a_backend_but_not_a_config_stay_separate(resolver) -> None:
    # Isaac-Reach-Franka's controller presets all run on Newton MJWarp and are different
    # runs, so collapsing on the backend instead of the config would merge them.
    resolver(
        {
            (None, None, None): ("fp-joint", "NewtonCfg(MJWarpSolverCfg)"),
            (None, None, "joint_pos"): ("fp-joint", "NewtonCfg(MJWarpSolverCfg)"),
            (None, None, "diffik"): ("fp-diffik", "NewtonCfg(MJWarpSolverCfg)"),
        }
    )

    modes, default = _build_modes("Isaac-X", (), (), ("joint_pos", "diffik"), resolve=True)

    assert modes == (Mode(None, None, "joint_pos"), Mode(None, None, "diffik"))
    assert default.mode == Mode(None, None, "joint_pos")


def test_a_default_matching_no_named_preset_survives_as_its_own_mode(resolver) -> None:
    resolver({(None, None, None): ("fp-own", "PhysxCfg"), (None, None, "rgb"): ("fp-rgb", "PhysxCfg")})

    modes, default = _build_modes("Isaac-X", (), (), ("rgb",), resolve=True)

    assert modes == (Mode(None, None, None), Mode(None, None, "rgb"))
    assert default == Default(backend="PhysxCfg", mode=Mode(None, None, None))


def test_uncollapsed_keeps_every_spelling_but_still_drops_rejections(resolver) -> None:
    # ``shapes`` names the default of its own axis, so it resolves to the default config
    # and the collapse drops it -- but it is still a token a reader can type, which is
    # what the documentation view needs. ``raycaster`` is absent from the table, so it
    # does not resolve and must be dropped either way.
    resolver(
        {
            (None, None, None): ("fp-default", "PhysxCfg"),
            (None, None, "shapes"): ("fp-default", "PhysxCfg"),
            (None, None, "cube"): ("fp-cube", "PhysxCfg"),
        }
    )
    domains = ("shapes", "cube", "raycaster")

    collapsed, _ = _build_modes("Isaac-X", (), (), domains, resolve=True)
    every, default = _build_modes("Isaac-X", (), (), domains, resolve=True, collapse=False)

    assert collapsed == (Mode(None, None, "shapes"), Mode(None, None, "cube"))
    assert every == (Mode(None, None, None), Mode(None, None, "shapes"), Mode(None, None, "cube"))
    assert default == Default(backend="PhysxCfg", mode=Mode(None, None, "shapes"))


def test_a_combination_the_validator_cannot_judge_is_dropped_and_the_walk_continues(broken_validator) -> None:
    # One task breaking the validator must not cost the caller the rest of the registry.
    # Unknown is not legal either, so the combination is dropped rather than reported.
    broken_validator("newton_mjwarp")

    modes, _ = _build_modes("Isaac-X", ("physx", "newton_mjwarp"), (), (), resolve=True)

    assert modes == (Mode("physx", None, None),)


def test_strict_raises_on_a_structural_failure_instead_of_dropping_it(broken_validator) -> None:
    # Callers policing Isaac Lab API drift want the canary, not a survivable walk.
    broken_validator("newton_mjwarp")

    with pytest.raises(DiscoveryError):
        _build_modes("Isaac-X", ("physx", "newton_mjwarp"), (), (), resolve=True, strict=True)
