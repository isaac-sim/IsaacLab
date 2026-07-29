# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Row expansion, key derivation, and uv extras selection."""

from pathlib import Path

import pytest

from tools.odin.plan import (
    PlanError,
    apply_metadata,
    build_row_key,
    load_task_rows,
    plan_rows,
    uv_extras_for_profile,
)

# A seed list carries only (task_id, rl_library, physics). Sizing is deliberately
# absent: upstream defaults --num_envs and --max_iterations to None and falls
# back to the task's shipped agent config, and the resolved values are harvested
# back out of the emitted bundles afterwards.
_ROWS = [
    {"task_id": "Isaac-Ant", "rl_library": "rsl_rl", "physics": "physx"},
    {"task_id": "Isaac-Cartpole-Direct", "rl_library": "skrl", "physics": "newton_mjwarp"},
]


def test_row_key_format_is_stable() -> None:
    assert build_row_key("rsl_rl", "physx", "Isaac-Ant", 42) == "rsl_rl_physx_Isaac-Ant_seed42"


def test_row_key_includes_the_renderer_when_set() -> None:
    # A camera task expands across renderers on the same physics; without this
    # segment those rows collide on one key.
    assert build_row_key("rsl_rl", "physx", "Isaac-Cam", 42, renderer="ovrtx") == "rsl_rl_physx_ovrtx_Isaac-Cam_seed42"


def test_renderer_variants_do_not_collide() -> None:
    entries = [
        {"task_id": "Isaac-Cam", "rl_library": "rsl_rl", "physics": "newton_mjwarp", "renderer": r}
        for r in ("ovrtx", "newton_renderer", "isaacsim_rtx")
    ]
    keys = [row.row_key for row in plan_rows(task_rows=entries, seeds=[42])]
    assert len(keys) == len(set(keys)) == 3


def test_full_profile_holds_every_backend() -> None:
    # One virtualenv covers Kit PhysX, OvPhysX, Newton and OVRTX, so a row's
    # physics preset never has to be resolved to a runtime before dispatch.
    extras = uv_extras_for_profile("full")
    for expected in ("isaacsim", "ovphysx", "ovrtx", "rsl-rl", "skrl", "rl-games", "sb3", "rerun"):
        assert expected in extras


def test_full_profile_excludes_extras_that_cannot_co_resolve() -> None:
    # teleop, viser, mimic, test and the `all` aggregate still conflict with
    # isaacsim in [tool.uv].conflicts.
    extras = uv_extras_for_profile("full")
    for forbidden in ("teleop", "viser", "mimic", "test", "all"):
        assert forbidden not in extras


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(PlanError, match="quantum"):
        uv_extras_for_profile("quantum")


def test_rows_default_to_the_full_profile() -> None:
    assert plan_rows(task_rows=_ROWS, seeds=[42])[0].profile == "full"


def test_rows_expand_across_seeds() -> None:
    rows = plan_rows(task_rows=_ROWS, seeds=[42, 43])
    assert len(rows) == 4
    assert {r.seed for r in rows} == {42, 43}


def test_sizing_defaults_to_none_when_absent() -> None:
    row = plan_rows(task_rows=_ROWS, seeds=[42])[0]
    assert row.num_envs is None
    assert row.max_iterations is None
    assert row.timeout_s is None


def test_sizing_is_carried_when_present() -> None:
    entry = dict(_ROWS[0], num_envs=4096, max_iterations=500, timeout_s=3600)
    row = plan_rows(task_rows=[entry], seeds=[42])[0]
    assert (row.num_envs, row.max_iterations, row.timeout_s) == (4096, 500, 3600)


def test_include_glob_filters_by_task_id() -> None:
    rows = plan_rows(task_rows=_ROWS, seeds=[42], include="Isaac-Ant*")
    assert [r.task_id for r in rows] == ["Isaac-Ant"]


def test_osmo_task_name_is_dns_safe() -> None:
    row = plan_rows(task_rows=_ROWS, seeds=[42])[0]
    assert row.osmo_task_name == row.osmo_task_name.lower()
    assert "_" not in row.osmo_task_name
    assert len(row.osmo_task_name) <= 63


def test_planning_is_deterministic() -> None:
    first = [r.row_key for r in plan_rows(task_rows=_ROWS, seeds=[43, 42])]
    second = [r.row_key for r in plan_rows(task_rows=_ROWS, seeds=[43, 42])]
    assert first == second


def test_duplicate_row_keys_are_rejected() -> None:
    with pytest.raises(PlanError, match="duplicate"):
        plan_rows(task_rows=_ROWS + [_ROWS[0]], seeds=[42])


def test_missing_field_names_the_task() -> None:
    with pytest.raises(PlanError, match="Isaac-Ant"):
        plan_rows(task_rows=[{"task_id": "Isaac-Ant"}], seeds=[42])


def test_unknown_rl_library_is_rejected() -> None:
    with pytest.raises(PlanError, match="tensorforce"):
        plan_rows(task_rows=[dict(_ROWS[0], rl_library="tensorforce")], seeds=[42])


def test_load_task_rows_requires_a_tasks_list(tmp_path: Path) -> None:
    path = tmp_path / "tasks.yaml"
    path.write_text("not_tasks: []\n")
    with pytest.raises(PlanError, match="tasks"):
        load_task_rows(path)


def test_shipped_seed_list_plans_cleanly() -> None:
    shipped = Path(__file__).resolve().parents[1] / "config" / "tasks.yaml"
    rows = plan_rows(task_rows=load_task_rows(shipped), seeds=[42])
    assert rows
    assert all(r.osmo_task_name for r in rows)


def test_metadata_overlay_fills_missing_sizing() -> None:
    metadata = [
        {
            "task_id": "Isaac-Ant",
            "rl_library": "rsl_rl",
            "physics": "physx",
            "num_envs": 4096,
            "max_iterations": 500,
            "timeout_s": 900,
        }
    ]
    merged = apply_metadata(_ROWS, metadata)
    row = plan_rows(task_rows=merged, seeds=[42])[0]
    assert (row.num_envs, row.max_iterations, row.timeout_s) == (4096, 500, 900)


def test_metadata_overlay_never_overrides_an_explicit_value() -> None:
    # A hand-set override must survive a later harvest.
    seed = [dict(_ROWS[0], num_envs=64)]
    metadata = [{"task_id": "Isaac-Ant", "rl_library": "rsl_rl", "physics": "physx", "num_envs": 4096}]
    assert apply_metadata(seed, metadata)[0]["num_envs"] == 64


def test_metadata_for_unlisted_combinations_is_ignored() -> None:
    metadata = [{"task_id": "Isaac-Nonexistent", "rl_library": "rsl_rl", "physics": "physx", "num_envs": 1}]
    assert apply_metadata(_ROWS, metadata) == _ROWS


def test_metadata_matches_on_physics_too() -> None:
    # Same task and library on a different backend must not inherit sizing.
    metadata = [{"task_id": "Isaac-Ant", "rl_library": "rsl_rl", "physics": "newton_mjwarp", "num_envs": 4096}]
    assert apply_metadata(_ROWS, metadata)[0].get("num_envs") is None


def test_metadata_matches_on_renderer_and_presets_too() -> None:
    # harvest groups on five fields; matching on three would let a camera row
    # inherit a headless row's budget, which is the wrong workload entirely.
    seed = [dict(_ROWS[0], renderer="ovrtx", presets="depth")]
    headless = [{"task_id": "Isaac-Ant", "rl_library": "rsl_rl", "physics": "physx", "timeout_s": 60}]
    assert apply_metadata(seed, headless)[0].get("timeout_s") is None

    camera = [dict(headless[0], renderer="ovrtx", presets="depth")]
    assert apply_metadata(seed, camera)[0]["timeout_s"] == 60


def test_preset_variants_have_a_fixed_order() -> None:
    # Without renderer and presets in the sort key these rows tie, so their
    # layout follows input order and reruns disagree.
    entries = [
        {"task_id": "Isaac-Cam", "rl_library": "rsl_rl", "physics": "newton_mjwarp", "renderer": "ovrtx", "presets": p}
        for p in ("rgb", "depth")
    ]
    forward = [row.row_key for row in plan_rows(task_rows=entries, seeds=[42])]
    reversed_ = [row.row_key for row in plan_rows(task_rows=list(reversed(entries)), seeds=[42])]
    assert forward == reversed_


def test_task_without_a_physics_preset_is_planned() -> None:
    # 35 of the 87 usable tasks declare no physics preset and hard-fail on any
    # physics= token, so physics must be optional.
    row = plan_rows(task_rows=[{"task_id": "Isaac-X", "rl_library": "rsl_rl"}], seeds=[42])[0]
    assert row.physics is None
    assert row.row_key == "rsl_rl_default_Isaac-X_seed42"


def test_presets_accept_a_single_token() -> None:
    row = plan_rows(task_rows=[dict(_ROWS[0], presets="depth")], seeds=[42])[0]
    assert row.presets == ("depth",)


def test_presets_accept_a_comma_separated_string() -> None:
    # Discovery emits one token, but a hand-written list may compose several.
    row = plan_rows(task_rows=[dict(_ROWS[0], presets="depth,simple_shading_full_mdl")], seeds=[42])[0]
    assert row.presets == ("depth", "simple_shading_full_mdl")


def test_presets_accept_a_yaml_list() -> None:
    row = plan_rows(task_rows=[dict(_ROWS[0], presets=["depth", "albedo"])], seeds=[42])[0]
    assert row.presets == ("depth", "albedo")


def test_absent_presets_are_empty() -> None:
    assert plan_rows(task_rows=_ROWS, seeds=[42])[0].presets == ()
