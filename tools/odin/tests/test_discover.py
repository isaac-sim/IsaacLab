# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Row expansion and filtering for registry-discovered task lists.

The registry walk itself needs Isaac Lab and is exercised separately; everything
here is pure and runs offline.
"""

from pathlib import Path

import yaml

from tools.odin import discover
from tools.odin.discover import DiscoveredTask, expand_rows, filter_rows, write_task_list
from tools.odin.plan import load_task_rows, plan_rows

Mode = DiscoveredTask.Mode

_ANT = DiscoveredTask(
    task_id="Isaac-Ant",
    scope="core",
    rl_libraries=("rsl_rl", "rl_games", "skrl"),
    modes=(Mode("newton_kamino", None, None), Mode("newton_mjwarp", None, None), Mode("physx", None, None)),
)
_CONTRIB = DiscoveredTask(
    task_id="IsaacContrib-Walk",
    scope="contrib",
    rl_libraries=("rsl_rl",),
    modes=(Mode("newton_mjwarp", None, None),),
)
# 35 real tasks declare no physics preset and reject any physics= token.
_NO_PHYSICS = DiscoveredTask(
    task_id="Isaac-Intrinsic", scope="core", rl_libraries=("skrl",), modes=(Mode(None, None, None),)
)
# A camera task: renderers are always expanded, and one pairing is illegal.
_CAMERA = DiscoveredTask(
    task_id="Isaac-Cartpole-Camera",
    scope="core",
    rl_libraries=("rsl_rl",),
    modes=(Mode("newton_mjwarp", "ovrtx", None), Mode("isaacsim_physx", "isaacsim_rtx", None)),
)
_TASKS = [_ANT, _CONTRIB, _NO_PHYSICS]


def test_expansion_is_total_across_every_declared_library() -> None:
    rows = expand_rows(_TASKS)
    assert {row["rl_library"] for row in rows if row["task_id"] == "Isaac-Ant"} == {"rsl_rl", "rl_games", "skrl"}


def test_library_filter_narrows_the_library_axis() -> None:
    rows = filter_rows(expand_rows(_TASKS), libraries=["rsl_rl"])
    assert {row["rl_library"] for row in rows} == {"rsl_rl"}


def test_scope_filter_replaces_a_core_only_selection() -> None:
    rows = filter_rows(expand_rows(_TASKS), scope="core")
    assert all(row["scope"] == "core" for row in rows)


def test_physics_filter_replaces_a_cross_backend_selection() -> None:
    # Two backends on the same tasks is a filter, not its own vocabulary.
    task = DiscoveredTask(
        task_id="Isaac-X",
        scope="core",
        rl_libraries=("rsl_rl",),
        modes=(Mode("newton_mjwarp", None, None), Mode("ovphysx", None, None), Mode("newton_kamino", None, None)),
    )
    rows = filter_rows(expand_rows([task]), physics=["newton_mjwarp", "ovphysx"])
    assert {row["physics"] for row in rows} == {"newton_mjwarp", "ovphysx"}


def test_task_without_physics_gets_exactly_one_row_with_no_token() -> None:
    rows = [row for row in expand_rows(_TASKS) if row["task_id"] == "Isaac-Intrinsic"]
    assert len(rows) == 1
    assert "physics" not in rows[0]


def test_task_without_a_supported_library_is_dropped() -> None:
    task = DiscoveredTask(task_id="Isaac-Rlinf", scope="core", rl_libraries=(), modes=())
    assert expand_rows([task]) == []


def test_camera_task_rows_carry_a_renderer() -> None:
    rows = expand_rows([_CAMERA])
    assert len(rows) == 2
    assert all(row["renderer"] for row in rows)
    assert {(r["physics"], r["renderer"]) for r in rows} == {
        ("newton_mjwarp", "ovrtx"),
        ("isaacsim_physx", "isaacsim_rtx"),
    }


def test_renderer_filter_narrows_the_renderer_axis() -> None:
    rows = filter_rows(expand_rows([_CAMERA]), renderers=["ovrtx"])
    assert {row["renderer"] for row in rows} == {"ovrtx"}


def test_renderer_filter_keeps_headless_rows_via_none() -> None:
    # Rows are headless unless a renderer was requested.
    rows = expand_rows(_TASKS)
    assert filter_rows(rows, renderers=["none"]) == rows


def test_rows_are_deterministic() -> None:
    assert expand_rows(_TASKS) == expand_rows(list(reversed(_TASKS)))


def test_scope_filter_uses_the_explicit_field() -> None:
    rows = expand_rows(_TASKS)
    assert {row["task_id"] for row in filter_rows(rows, scope="contrib")} == {"IsaacContrib-Walk"}


def test_include_and_exclude_globs_compose() -> None:
    rows = expand_rows(_TASKS)
    assert filter_rows(rows, include="Isaac-*", exclude="Isaac-Intrinsic*") != []
    assert all(r["task_id"] != "Isaac-Intrinsic" for r in filter_rows(rows, exclude="Isaac-Intrinsic*"))


def test_physics_filter_matches_untokened_rows_via_default() -> None:
    rows = expand_rows(_TASKS)
    assert [r["task_id"] for r in filter_rows(rows, physics=["default"])] == ["Isaac-Intrinsic"]


def test_max_rows_is_a_deterministic_head() -> None:
    rows = expand_rows(_TASKS)
    assert filter_rows(rows, max_rows=2) == rows[:2]


def test_written_list_is_consumable_by_the_planner(tmp_path: Path) -> None:
    # The whole point of the contract: discovery output feeds plan_rows unchanged.
    out = tmp_path / "tasks.yaml"
    rows = expand_rows(_TASKS)

    write_task_list(out, rows, meta={"row_count": len(rows)})

    planned = plan_rows(task_rows=load_task_rows(out), seeds=[42])
    assert len(planned) == len(rows)
    assert any(row.physics is None for row in planned)


def test_broken_imports_are_not_mistaken_for_rejected_presets() -> None:
    # An import regression that swallows to False marks every mode illegal, and
    # the CLI then blames the operator's filters for the empty row set.
    assert ImportError in discover._INFRASTRUCTURE_ERRORS
    assert AttributeError in discover._INFRASTRUCTURE_ERRORS
    # A rejected preset combination raises these; they must stay swallowed.
    assert ValueError not in discover._INFRASTRUCTURE_ERRORS
    assert RuntimeError not in discover._INFRASTRUCTURE_ERRORS


def test_provenance_header_is_ignored_by_the_planner(tmp_path: Path) -> None:
    out = tmp_path / "tasks.yaml"
    write_task_list(out, expand_rows(_TASKS), meta={"row_count": 1})
    assert yaml.safe_load(out.read_text())["discovery"]["row_count"] == 1
    assert load_task_rows(out)  # header does not leak into the rows
