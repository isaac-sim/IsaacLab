# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for environment documentation generation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from gymnasium.envs.registration import EnvSpec


def _bootstrap_paths() -> None:
    """Prepend editable ``source/*`` packages and ``tools/`` for dev-tree runs."""
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = repo_root / "source"
    tools_dir = repo_root / "tools"

    prepend: list[str] = [str(tools_dir)]
    if source_dir.is_dir():
        for package_dir in sorted(source_dir.iterdir()):
            if not package_dir.is_dir():
                continue
            module_root = package_dir / package_dir.name
            if module_root.is_dir():
                prepend.append(str(package_dir))

    for path in reversed(prepend):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_paths()

from environ_docs import (  # noqa: E402
    COMPREHENSIVE_LIST_END_MARKER,
    COMPREHENSIVE_LIST_START_MARKER,
    ENVIRONMENT_BROWSER_TASKS_END_MARKER,
    ENVIRONMENT_BROWSER_TASKS_START_MARKER,
    EnvironmentDocRow,
    _physics_names_for_docs,
    apply_rl_library_overrides,
    collect_environment_doc_rows,
    format_presets_rst,
    format_rl_libraries,
    is_training_task,
    parse_rl_libraries_from_kwargs,
    patch_curated_environment_tables,
    patch_environment_browser_javascript,
    patch_environments_rst,
    render_comprehensive_list_table,
    render_environment_browser_task_rows,
)

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.preset_target import PresetTarget  # noqa: E402


def test_is_training_task_filters_inference_variants():
    assert is_training_task("Isaac-Cartpole")
    assert not is_training_task("IsaacContrib-Assemble-Trocar-G129-Dex3-Eval")
    assert not is_training_task("Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0")


def test_parse_rl_libraries_from_kwargs_handles_multi_agent_and_amp():
    kwargs = {
        "env_cfg_entry_point": "ignored",
        "rl_games_cfg_entry_point": "agents:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": "agents:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": "agents:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "agents:skrl_mappo_cfg.yaml",
        "skrl_amp_cfg_entry_point": "agents:skrl_amp_cfg.yaml",
        "rsl_rl_with_symmetry_cfg_entry_point": "agents.rsl_rl_ppo_cfg:RunnerCfg",
        "rl_games_cfg_entry_point_ignored": "nope",
    }
    agents = parse_rl_libraries_from_kwargs(kwargs)
    assert agents["rl_games"] == ["PPO"]
    assert agents["skrl"] == ["PPO", "AMP", "IPPO", "MAPPO"]


def test_parse_rl_libraries_detects_vision_config_from_filename():
    kwargs = {
        "env_cfg_entry_point": "ignored",
        "rl_games_cfg_entry_point": "agents:rl_games_ppo_vision_cfg.yaml",
        "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg:RunnerCfg",
    }
    agents = parse_rl_libraries_from_kwargs(kwargs)
    assert agents["rl_games"] == ["VISION"]
    assert agents["rsl_rl"] == ["PPO"]


def test_apply_rl_library_overrides_supplements_registry_gaps():
    agents = apply_rl_library_overrides(
        "IsaacContrib-Assemble-Trocar-G129-Dex3",
        {},
    )
    assert agents == {"rlinf": ["PPO"]}


def test_format_presets_rst_single_and_multi_line():
    single = format_presets_rst({PresetTarget.PHYSICS: ["physx", "isaacsim_physx", "newton_mjwarp"]})
    assert single == "**physics=** ``isaacsim_physx``, ``newton_mjwarp``"

    multi = format_presets_rst(
        {
            PresetTarget.PHYSICS: ["physx", "isaacsim_physx"],
            PresetTarget.RENDERER: ["rtx", "isaacsim_rtx", "ovrtx"],
            PresetTarget.DOMAIN: ["rgb", "depth"],
        }
    )
    assert "| **physics=** ``isaacsim_physx``" in multi
    assert "**renderer=** ``isaacsim_rtx``, ``ovrtx``" in multi
    assert "**presets=** ``rgb``, ``depth``" in multi
    assert "``physx``" not in multi
    assert "``rtx``" not in multi


def test_format_presets_rst_hides_domain_names_duplicated_by_physics():
    formatted = format_presets_rst(
        {
            PresetTarget.PHYSICS: ["isaacsim_physx", "newton_kamino", "newton_mjwarp", "physx"],
            PresetTarget.DOMAIN: ["newton_mjwarp", "physx"],
        }
    )
    assert formatted == "**physics=** ``isaacsim_physx``, ``newton_kamino``, ``newton_mjwarp``"
    assert "presets=" not in formatted


def test_format_presets_rst_hides_physics_backend_mirrors_without_physics_preset():
    formatted = format_presets_rst(
        {
            PresetTarget.PHYSICS: ["isaacsim_physx", "newton_mjwarp", "physx"],
            PresetTarget.DOMAIN: ["newton_mjwarp", "ovphysx", "physx"],
        }
    )
    assert formatted == "**physics=** ``isaacsim_physx``, ``newton_mjwarp``"
    assert "ovphysx" not in formatted


def test_format_presets_rst_hides_concrete_backend_mirrors_without_typed_selectors():
    formatted = format_presets_rst(
        {
            PresetTarget.PHYSICS: [],
            PresetTarget.RENDERER: [],
            PresetTarget.DOMAIN: ["isaacsim_physx", "isaacsim_rtx", "ovrtx", "rgb"],
        }
    )
    assert formatted == "**presets=** ``rgb``"


def test_format_presets_rst_keeps_ovphysx_on_physics():
    formatted = format_presets_rst(
        {
            PresetTarget.PHYSICS: ["isaacsim_physx", "newton_kamino", "newton_mjwarp", "ovphysx", "physx"],
            PresetTarget.DOMAIN: ["newton_mjwarp", "physx"],
        }
    )
    assert formatted == ("**physics=** ``isaacsim_physx``, ``newton_kamino``, ``newton_mjwarp``, ``ovphysx``")


def test_physics_names_for_docs_infers_physx_from_default():
    names = _physics_names_for_docs(
        "Isaac-Velocity-Flat-G1",
        {PresetTarget.PHYSICS: ["newton_mjwarp"], PresetTarget.DOMAIN: [], PresetTarget.RENDERER: []},
    )
    assert names == ["newton_mjwarp", "physx"]


def test_collect_environment_doc_rows_from_mock_specs():
    specs = [
        EnvSpec(
            id="Isaac-Cartpole-Direct",
            entry_point="isaaclab_tasks.core.cartpole.cartpole_direct_env:CartpoleEnv",
            kwargs={
                "env_cfg_entry_point": "cfg:CartpoleEnvCfg",
                "rl_games_cfg_entry_point": "agents:rl_games_ppo_cfg.yaml",
                "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
                "skrl_cfg_entry_point": "agents:skrl_ppo_cfg.yaml",
                "sb3_cfg_entry_point": "agents:sb3_ppo_cfg.yaml",
            },
        ),
        EnvSpec(
            id="Isaac-Cartpole-Direct-Eval",
            entry_point="isaaclab_tasks.core.cartpole.cartpole_direct_env:CartpoleEnv",
            kwargs={"env_cfg_entry_point": "cfg:CartpoleEnvCfg"},
        ),
    ]
    rows = collect_environment_doc_rows(specs)
    assert len(rows) == 1
    assert rows[0].task_name == "Isaac-Cartpole-Direct"
    assert rows[0].workflow == "Direct"
    assert "sb3" in rows[0].rl_libraries


def test_collect_environment_doc_rows_excludes_deprecated_task_aliases():
    specs = [
        EnvSpec(
            id="Isaac-Example",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": "cfg:ExampleEnvCfg"},
        ),
        EnvSpec(
            id="Isaac-Example-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": "cfg:ExampleEnvCfg",
                "deprecated": {"alias": "--task Isaac-Example"},
            },
        ),
    ]

    rows = collect_environment_doc_rows(specs)

    assert [row.task_name for row in rows] == ["Isaac-Example"]


def test_collect_environment_doc_rows_applies_rlinf_override():
    specs = [
        EnvSpec(
            id="IsaacContrib-Assemble-Trocar-G129-Dex3",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": "cfg:G1AssembleTrocarEnvCfg"},
        ),
    ]
    rows = collect_environment_doc_rows(specs)
    assert rows[0].rl_libraries == {"rlinf": ["PPO"]}


def test_format_presets_rst_returns_empty_string_when_unavailable():
    assert format_presets_rst(None) == ""
    assert format_presets_rst({PresetTarget.PHYSICS: [], PresetTarget.RENDERER: [], PresetTarget.DOMAIN: []}) == ""


def test_format_rl_libraries_returns_empty_string_when_unavailable():
    assert format_rl_libraries({}) == ""


def test_render_comprehensive_list_table_uses_blank_cells_for_missing_values():
    table = render_comprehensive_list_table(
        [
            EnvironmentDocRow(
                task_name="Isaac-Ant-v0",
                workflow="Manager Based",
                rl_libraries={},
                presets=None,
            )
        ]
    )
    assert "Isaac-Ant-v0" in table
    assert "      - -\n" not in table
    assert "    * - Isaac-Ant-v0\n      - Manager Based" in table


def test_render_comprehensive_list_table_uses_narrower_task_column_width():
    table = render_comprehensive_list_table([])
    assert ":widths: 22 12 30 36" in table


def test_patch_environments_rst_replaces_marked_section():
    original = (
        "Header\n\n"
        f"{COMPREHENSIVE_LIST_START_MARKER}\n\n"
        ".. list-table::\n"
        "    old\n\n"
        f"{COMPREHENSIVE_LIST_END_MARKER}\n"
        "Footer"
    )
    updated = patch_environments_rst(original, ".. list-table::\n    new")
    assert "old" not in updated
    assert ".. list-table::\n    new" in updated
    assert updated.endswith("Footer")


def test_patch_curated_environment_tables_synchronizes_concrete_presets():
    original = (
        ".. table::\n\n"
        "    +-------+------------------+-------------+------------------+\n"
        "    | World | Environment ID   | Description | Presets          |\n"
        "    +=======+==================+=============+==================+\n"
        "    | demo  | |cartpole-link|  | Example     | **physics=**     |\n"
        "    |       |                  |             | ``physx``        |\n"
        "    +-------+------------------+-------------+------------------+\n\n"
        ".. |cartpole-link| replace:: :isaaclab-source:`Isaac-Cartpole <cfg.py>`\n\n"
        f"{COMPREHENSIVE_LIST_START_MARKER}\n"
    )
    rows = [
        EnvironmentDocRow(
            task_name="Isaac-Cartpole",
            workflow="Manager Based",
            rl_libraries={},
            presets={
                PresetTarget.PHYSICS: ["isaacsim_physx", "newton_mjwarp", "ovphysx"],
                PresetTarget.RENDERER: ["isaacsim_rtx", "ovrtx"],
                PresetTarget.DOMAIN: ["rgb"],
            },
        )
    ]

    updated = patch_curated_environment_tables(original, rows)

    assert "``isaacsim_physx``" in updated
    assert "``newton_mjwarp``" in updated
    assert "``ovphysx``" in updated
    assert "``isaacsim_rtx``" in updated
    assert "``ovrtx``" in updated
    assert "``rgb``" in updated
    assert "``physx``" not in updated


def test_environment_browser_rows_include_only_concrete_core_selectors():
    rows = [
        EnvironmentDocRow(
            task_name="Isaac-Cartpole",
            workflow="Manager Based",
            rl_libraries={"rsl_rl": ["PPO"], "skrl": ["PPO"]},
            presets={
                PresetTarget.PHYSICS: ["isaacsim_physx", "newton_mjwarp"],
                PresetTarget.RENDERER: ["isaacsim_rtx", "ovrtx"],
                PresetTarget.DOMAIN: ["rgb"],
            },
        ),
        EnvironmentDocRow(
            task_name="IsaacContrib-Cartpole",
            workflow="Manager Based",
            rl_libraries={"rsl_rl": ["PPO"]},
            presets={PresetTarget.PHYSICS: ["ovphysx"]},
        ),
    ]
    rendered = render_environment_browser_task_rows(rows)
    original = (
        f"        {ENVIRONMENT_BROWSER_TASKS_START_MARKER}\n"
        "        const taskRows = [];\n"
        f"        {ENVIRONMENT_BROWSER_TASKS_END_MARKER}\n"
        "        const preserved = true;\n"
    )

    updated = patch_environment_browser_javascript(original, rendered)

    assert '"Isaac-Cartpole"' in updated
    assert '"rsl_rl,skrl"' in updated
    assert '"isaacsim_physx,newton_mjwarp"' in updated
    assert '"isaacsim_rtx,ovrtx"' in updated
    assert '"rgb"' in updated
    assert "IsaacContrib-Cartpole" not in updated
    assert "const preserved = true;" in updated


def test_patch_environment_browser_rejects_markers_around_non_generated_code():
    original = (
        f"{ENVIRONMENT_BROWSER_TASKS_START_MARKER}\n"
        "const taskRows = [];\n"
        "const preserved = true;\n"
        f"{ENVIRONMENT_BROWSER_TASKS_END_MARKER}\n"
    )

    with pytest.raises(ValueError, match="only the generated taskRows array"):
        patch_environment_browser_javascript(original, "const taskRows = [];")


def test_render_comprehensive_list_table_includes_header():
    table = render_comprehensive_list_table(
        [
            collect_environment_doc_rows(
                [
                    EnvSpec(
                        id="Isaac-Cartpole",
                        entry_point="isaaclab.envs:ManagerBasedRLEnv",
                        kwargs={
                            "env_cfg_entry_point": "cfg:CartpoleEnvCfg",
                            "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
                        },
                    )
                ]
            )[0]
        ]
    )
    assert "**Task Name**" in table
    assert "Isaac-Cartpole" in table
    assert format_rl_libraries({"rsl_rl": ["PPO"]}) in table
