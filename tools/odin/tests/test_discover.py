# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy expansion and filtering for registry-discovered task lists.

The registry walk itself needs Isaac Lab and is exercised separately; everything
here is pure and runs offline.
"""

from pathlib import Path

import pytest
import yaml

from tools.odin.discover import DiscoveredTask, DiscoveryError, filter_rows, rows_for_policy, write_task_list
from tools.odin.plan import load_task_rows, plan_rows

_ANT = DiscoveredTask(
    task_id="Isaac-Ant",
    scope="core",
    rl_libraries=("rsl_rl", "rl_games", "skrl"),
    physics=("newton_kamino", "newton_mjwarp", "physx"),
    renderers=(),
)
_CONTRIB = DiscoveredTask(
    task_id="IsaacContrib-Walk",
    scope="contrib",
    rl_libraries=("rsl_rl",),
    physics=("newton_mjwarp",),
    renderers=(),
)
# 35 real tasks declare no physics preset and reject any physics= token.
_NO_PHYSICS = DiscoveredTask(
    task_id="Isaac-Intrinsic", scope="core", rl_libraries=("skrl",), physics=(), renderers=()
)
_TASKS = [_ANT, _CONTRIB, _NO_PHYSICS]


def test_standard_picks_one_library_per_task() -> None:
    rows = rows_for_policy(_TASKS, "standard")
    assert {row["rl_library"] for row in rows if row["task_id"] == "Isaac-Ant"} == {"rsl_rl"}


def test_all_libraries_expands_every_declared_library() -> None:
    rows = rows_for_policy(_TASKS, "all-libraries")
    assert {row["rl_library"] for row in rows if row["task_id"] == "Isaac-Ant"} == {"rsl_rl", "rl_games", "skrl"}


def test_core_only_drops_contrib() -> None:
    rows = rows_for_policy(_TASKS, "core-only")
    assert all(row["scope"] == "core" for row in rows)


def test_cross_backend_keeps_only_the_two_compared_backends() -> None:
    task = DiscoveredTask(
        task_id="Isaac-X", scope="core", rl_libraries=("rsl_rl",),
        physics=("newton_mjwarp", "ovphysx", "newton_kamino"), renderers=(),
    )
    rows = rows_for_policy([task], "cross-backend")
    assert {row["physics"] for row in rows} == {"newton_mjwarp", "ovphysx"}


def test_physx_is_dropped_when_ovphysx_is_also_declared() -> None:
    # Headless, physics=physx resolves to OvPhysX, so running both duplicates.
    task = DiscoveredTask(
        task_id="Isaac-X", scope="core", rl_libraries=("rsl_rl",), physics=("physx", "ovphysx"), renderers=()
    )
    assert {row["physics"] for row in rows_for_policy([task], "standard")} == {"ovphysx"}


def test_physx_is_kept_when_ovphysx_is_absent() -> None:
    task = DiscoveredTask(
        task_id="Isaac-X", scope="core", rl_libraries=("rsl_rl",), physics=("physx",), renderers=()
    )
    assert {row["physics"] for row in rows_for_policy([task], "standard")} == {"physx"}


def test_proxy_backend_is_never_dispatched() -> None:
    task = DiscoveredTask(
        task_id="Isaac-X", scope="core", rl_libraries=("rsl_rl",),
        physics=("newton_mjwarp", "newton_mjwarp_vbd_proxy"), renderers=(),
    )
    assert {row["physics"] for row in rows_for_policy([task], "standard")} == {"newton_mjwarp"}


def test_task_without_physics_gets_exactly_one_row_with_no_token() -> None:
    rows = [row for row in rows_for_policy(_TASKS, "standard") if row["task_id"] == "Isaac-Intrinsic"]
    assert len(rows) == 1
    assert "physics" not in rows[0]


def test_task_without_a_supported_library_is_dropped() -> None:
    task = DiscoveredTask(task_id="Isaac-Rlinf", scope="core", rl_libraries=(), physics=(), renderers=())
    assert rows_for_policy([task], "standard") == []


def test_rows_are_deterministic() -> None:
    assert rows_for_policy(_TASKS, "standard") == rows_for_policy(list(reversed(_TASKS)), "standard")


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(DiscoveryError, match="quantum"):
        rows_for_policy(_TASKS, "quantum")


def test_scope_filter_uses_the_explicit_field() -> None:
    rows = rows_for_policy(_TASKS, "standard")
    assert {row["task_id"] for row in filter_rows(rows, scope="contrib")} == {"IsaacContrib-Walk"}


def test_include_and_exclude_globs_compose() -> None:
    rows = rows_for_policy(_TASKS, "standard")
    assert filter_rows(rows, include="Isaac-*", exclude="Isaac-Intrinsic*") != []
    assert all(r["task_id"] != "Isaac-Intrinsic" for r in filter_rows(rows, exclude="Isaac-Intrinsic*"))


def test_physics_filter_matches_untokened_rows_via_default() -> None:
    rows = rows_for_policy(_TASKS, "standard")
    assert [r["task_id"] for r in filter_rows(rows, physics=["default"])] == ["Isaac-Intrinsic"]


def test_max_rows_is_a_deterministic_head() -> None:
    rows = rows_for_policy(_TASKS, "standard")
    assert filter_rows(rows, max_rows=2) == rows[:2]


def test_written_list_is_consumable_by_the_planner(tmp_path: Path) -> None:
    # The whole point of the contract: discovery output feeds plan_rows unchanged.
    out = tmp_path / "tasks.yaml"
    rows = rows_for_policy(_TASKS, "standard")

    write_task_list(out, rows, meta={"policy": "standard", "row_count": len(rows)})

    planned = plan_rows(task_rows=load_task_rows(out), seeds=[42])
    assert len(planned) == len(rows)
    assert any(row.physics is None for row in planned)


def test_provenance_header_is_ignored_by_the_planner(tmp_path: Path) -> None:
    out = tmp_path / "tasks.yaml"
    write_task_list(out, rows_for_policy(_TASKS, "standard"), meta={"policy": "standard"})
    assert yaml.safe_load(out.read_text())["discovery"]["policy"] == "standard"
    assert load_task_rows(out)  # header does not leak into the rows
