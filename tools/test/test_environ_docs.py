# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for environment documentation generation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

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
    EnvironmentDocRow,
    _physics_names_for_docs,
    apply_rl_library_overrides,
    collect_environment_doc_rows,
    find_inference_task_name,
    format_presets_rst,
    format_rl_libraries,
    is_training_task,
    parse_rl_libraries_from_kwargs,
    patch_environments_rst,
    render_comprehensive_list_table,
)

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.preset_target import PresetTarget  # noqa: E402


def test_is_training_task_filters_inference_variants():
    assert is_training_task("Isaac-Cartpole-v0")
    assert not is_training_task("Isaac-Cartpole-Play-v0")
    assert not is_training_task("Isaac-Assemble-Trocar-G129-Dex3-Eval-v0")
    assert not is_training_task("Isaac-Repose-Cube-Shadow-Vision-Direct-Play-v0")
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
        "Isaac-Assemble-Trocar-G129-Dex3-v0",
        {},
    )
    assert agents == {"rlinf": ["PPO"]}


def test_find_inference_task_name_supports_play_and_eval():
    registry_ids = {
        "Isaac-Ant-v0",
        "Isaac-Ant-Play-v0",
        "Isaac-Assemble-Trocar-G129-Dex3-v0",
        "Isaac-Assemble-Trocar-G129-Dex3-Eval-v0",
    }
    assert find_inference_task_name("Isaac-Ant-v0", registry_ids) == "Isaac-Ant-Play-v0"
    assert (
        find_inference_task_name("Isaac-Assemble-Trocar-G129-Dex3-v0", registry_ids)
        == "Isaac-Assemble-Trocar-G129-Dex3-Eval-v0"
    )
    assert find_inference_task_name("Isaac-Cartpole-v0", registry_ids) is None


def test_format_presets_rst_single_and_multi_line():
    single = format_presets_rst({PresetTarget.PHYSICS: ["physx", "newton_mjwarp"]})
    assert single == "**physics=** ``physx``, ``newton_mjwarp``"

    multi = format_presets_rst(
        {
            PresetTarget.PHYSICS: ["physx"],
            PresetTarget.RENDERER: ["isaacsim_rtx_renderer"],
            PresetTarget.DOMAIN: ["rgb", "depth"],
        }
    )
    assert "| **physics=** ``physx``" in multi
    assert "**renderer=** ``isaacsim_rtx_renderer``" in multi
    assert "**presets=** ``rgb``, ``depth``" in multi


def test_format_presets_rst_hides_domain_names_duplicated_by_physics():
    formatted = format_presets_rst(
        {
            PresetTarget.PHYSICS: ["newton_kamino", "newton_mjwarp", "physx"],
            PresetTarget.DOMAIN: ["newton_mjwarp", "physx"],
        }
    )
    assert formatted == "**physics=** ``newton_kamino``, ``newton_mjwarp``, ``physx``"
    assert "presets=" not in formatted


def test_format_presets_rst_hides_physics_backend_mirrors_without_physics_preset():
    formatted = format_presets_rst(
        {
            PresetTarget.PHYSICS: ["newton_mjwarp", "physx"],
            PresetTarget.DOMAIN: ["newton_mjwarp", "ovphysx", "physx"],
        }
    )
    assert formatted == "**physics=** ``newton_mjwarp``, ``physx``"
    assert "ovphysx" not in formatted


def test_format_presets_rst_keeps_ovphysx_on_physics():
    formatted = format_presets_rst(
        {
            PresetTarget.PHYSICS: ["newton_kamino", "newton_mjwarp", "ovphysx", "physx"],
            PresetTarget.DOMAIN: ["newton_mjwarp", "physx"],
        }
    )
    assert formatted == "**physics=** ``newton_kamino``, ``newton_mjwarp``, ``ovphysx``, ``physx``"


def test_physics_names_for_docs_infers_physx_from_default():
    names = _physics_names_for_docs(
        "Isaac-Velocity-Flat-G1-v0",
        {PresetTarget.PHYSICS: ["newton_mjwarp"], PresetTarget.DOMAIN: [], PresetTarget.RENDERER: []},
    )
    assert names == ["newton_mjwarp", "physx"]


def test_collect_environment_doc_rows_from_mock_specs():
    specs = [
        EnvSpec(
            id="Isaac-Cartpole-Direct-v0",
            entry_point="isaaclab_tasks.direct.cartpole.cartpole_env:CartpoleEnv",
            kwargs={
                "env_cfg_entry_point": "cfg:CartpoleEnvCfg",
                "rl_games_cfg_entry_point": "agents:rl_games_ppo_cfg.yaml",
                "rsl_rl_cfg_entry_point": "agents.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
                "skrl_cfg_entry_point": "agents:skrl_ppo_cfg.yaml",
                "sb3_cfg_entry_point": "agents:sb3_ppo_cfg.yaml",
            },
        ),
        EnvSpec(
            id="Isaac-Cartpole-Direct-Play-v0",
            entry_point="isaaclab_tasks.direct.cartpole.cartpole_env:CartpoleEnv",
            kwargs={"env_cfg_entry_point": "cfg:CartpoleEnvCfg_PLAY"},
        ),
    ]
    rows = collect_environment_doc_rows(specs)
    assert len(rows) == 1
    assert rows[0].task_name == "Isaac-Cartpole-Direct-v0"
    assert rows[0].inference_task_name == "Isaac-Cartpole-Direct-Play-v0"
    assert rows[0].workflow == "Direct"
    assert "sb3" in rows[0].rl_libraries


def test_collect_environment_doc_rows_applies_rlinf_override():
    specs = [
        EnvSpec(
            id="Isaac-Assemble-Trocar-G129-Dex3-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": "cfg:G1AssembleTrocarEnvCfg"},
        ),
    ]
    rows = collect_environment_doc_rows(specs)
    assert rows[0].rl_libraries == "**rlinf** (PPO)"


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
                inference_task_name=None,
                workflow="Manager Based",
                rl_libraries="",
                presets="",
            )
        ]
    )
    assert "Isaac-Ant-v0" in table
    assert "      - -\n" not in table
    assert "      -\n      - Manager Based" in table


def test_render_comprehensive_list_table_uses_narrower_task_column_width():
    table = render_comprehensive_list_table([])
    assert ":widths: 18 17 10 22 33" in table


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


def test_render_comprehensive_list_table_includes_header():
    table = render_comprehensive_list_table(
        [
            collect_environment_doc_rows(
                [
                    EnvSpec(
                        id="Isaac-Cartpole-v0",
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
    assert "Isaac-Cartpole-v0" in table
    assert format_rl_libraries({"rsl_rl": ["PPO"]}) in table
