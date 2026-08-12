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

from task_discovery import (  # noqa: E402
    DiscoveredTask,
    _build_modes,
    _canonical_physics,
    _domain_presets,
    _is_training_task,
    _rl_libraries_from_kwargs,
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
    assert _is_training_task(task_id) is expected


def test_proxy_physics_variants_are_dropped() -> None:
    assert _canonical_physics(("newton_mjwarp", "newton_mjwarp_vbd_proxy")) == ("newton_mjwarp",)


def test_the_physx_selector_is_reported_alongside_concrete_backends() -> None:
    # Whether a selector duplicates a concrete backend depends on how the run is
    # launched, so the decision belongs to the caller, not to discovery.
    assert _canonical_physics(("isaacsim_physx", "ovphysx", "physx")) == ("isaacsim_physx", "ovphysx", "physx")


def test_domain_presets_drop_names_that_mirror_a_backend_selector() -> None:
    assert _domain_presets(["rgb", "ovphysx", "depth", "newton_mjwarp"]) == ("depth", "rgb")


def test_modes_are_the_cross_product_when_resolution_is_skipped() -> None:
    modes = _build_modes("Isaac-X", ("physx", "newton_mjwarp"), ("ovrtx",), (), resolve=False)

    assert modes == (Mode("physx", "ovrtx", None), Mode("newton_mjwarp", "ovrtx", None))


def test_a_task_without_presets_gets_one_mode_carrying_no_tokens() -> None:
    assert _build_modes("Isaac-X", (), (), (), resolve=False) == (Mode(None, None, None),)


def test_domain_presets_are_expanded_one_at_a_time_beside_the_default() -> None:
    # Presets targeting the same field conflict, so they are never combined; the
    # ``None`` entry keeps the task's own default reachable.
    modes = _build_modes("Isaac-X", (), (), ("rgb", "depth"), resolve=False)

    assert modes == (Mode(None, None, None), Mode(None, None, "rgb"), Mode(None, None, "depth"))
