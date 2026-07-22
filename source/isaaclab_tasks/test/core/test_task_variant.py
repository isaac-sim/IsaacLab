# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task preset and agent compatibility discovery."""

from __future__ import annotations

import argparse
import json

import gymnasium as gym
import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.task_variant import (
    enumerate_task_agents,
    enumerate_task_variants,
    validate_task_variant,
)


def test_enumerate_task_agents_filters_by_library() -> None:
    """Agent discovery returns only entry points for the requested library."""
    agents = enumerate_task_agents("Isaac-Cartpole-Camera", "rl_games")

    assert list(agents) == ["rl_games_cfg_entry_point", "rl_games_feature_cfg_entry_point"]


def test_cartpole_camera_declares_raw_and_feature_compatibility() -> None:
    """Cartpole camera exposes the intended preset-to-agent matrix."""
    variants = enumerate_task_variants("Isaac-Cartpole-Camera", "rl_games")
    agents = {agent["name"]: agent for agent in variants["agents"]}

    assert variants["default_preset"] == "rgb"
    assert agents["rl_games_cfg_entry_point"]["default"] is True
    assert agents["rl_games_cfg_entry_point"]["compatible_presets"] == [
        "albedo",
        "depth",
        "rgb",
        "semantic_segmentation",
        "simple_shading_constant_diffuse",
        "simple_shading_diffuse_mdl",
        "simple_shading_full_mdl",
    ]
    assert agents["rl_games_feature_cfg_entry_point"]["compatible_presets"] == ["resnet18", "theia_tiny"]


@pytest.mark.parametrize("preset_name", ["resnet18", "theia_tiny"])
def test_cartpole_camera_rejects_default_agent_for_feature_presets(preset_name: str) -> None:
    """Feature presets fail early with the raw-image agent and suggest the matching agent."""
    with pytest.raises(ValueError, match="rl_games_feature_cfg_entry_point") as exc_info:
        validate_task_variant(
            "Isaac-Cartpole-Camera",
            "rl_games_cfg_entry_point",
            [preset_name],
        )
    assert "rsl_rl_feature_cfg_entry_point" not in str(exc_info.value)


def test_cartpole_camera_rejects_feature_agent_for_default_rgb() -> None:
    """Omitting presets still validates the task's effective default preset."""
    with pytest.raises(ValueError, match="presets=rgb"):
        validate_task_variant(
            "Isaac-Cartpole-Camera",
            "rl_games_feature_cfg_entry_point",
            [],
        )


def test_cartpole_camera_accepts_backend_and_feature_presets_together() -> None:
    """Physics and renderer selectors remain orthogonal to agent compatibility."""
    validate_task_variant(
        "Isaac-Cartpole-Camera",
        "rsl_rl_feature_cfg_entry_point",
        ["newton_mjwarp", "newton_renderer", "resnet18"],
    )


@pytest.mark.parametrize(
    "task_name,preset_name,agent_entry",
    [
        (
            "IsaacContrib-Cartpole-Showcase-Direct",
            "tuple_multidiscrete",
            "skrl_tuple_multidiscrete_cfg_entry_point",
        ),
        (
            "IsaacContrib-Cartpole-Camera-Showcase-Direct",
            "dict_discrete",
            "skrl_dict_discrete_cfg_entry_point",
        ),
    ],
)
def test_cartpole_showcases_declare_matching_space_agents(task_name: str, preset_name: str, agent_entry: str) -> None:
    """Both showcase tasks declare their non-default space-specific agents."""
    validate_task_variant(task_name, agent_entry, [preset_name])
    with pytest.raises(ValueError, match=agent_entry):
        validate_task_variant(task_name, "skrl_cfg_entry_point", [preset_name])


def test_all_declared_task_variant_metadata_matches_live_registrations() -> None:
    """Every task-owned compatibility declaration references live presets and agents."""
    tasks = sorted(spec.id for spec in gym.registry.values() if "task_variant_cfg" in spec.kwargs)

    assert tasks == [
        "Isaac-Cartpole-Camera",
        "IsaacContrib-Cartpole-Camera-Showcase-Direct",
        "IsaacContrib-Cartpole-Showcase-Direct",
    ]
    for task_name in tasks:
        enumerate_task_variants(task_name)


def test_resolver_validates_agent_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal config resolution fails before sim launch for an incompatible pair."""
    monkeypatch.setattr("sys.argv", ["train.py", "presets=resnet18"])
    from isaaclab_tasks.utils import resolve_task_config

    with pytest.raises(ValueError, match="incompatible with presets=resnet18"):
        resolve_task_config("Isaac-Cartpole-Camera", "rl_games_cfg_entry_point")


def test_resolver_validates_path_targeted_preset_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path-targeted domain presets receive the same agent compatibility check."""
    monkeypatch.setattr("sys.argv", ["train.py", "env=resnet18"])
    from isaaclab_tasks.utils import resolve_task_config

    with pytest.raises(ValueError, match="incompatible with presets=resnet18"):
        resolve_task_config("Isaac-Cartpole-Camera", "rl_games_cfg_entry_point")


def test_task_help_lists_agents_and_compatibility(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Task-specific help places registered agents beside compatible presets."""
    monkeypatch.setattr("sys.argv", ["train.py", "--task", "Isaac-Cartpole-Camera", "--help"])
    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--task")
    parser.add_argument("--agent", default="rl_games_cfg_entry_point")
    with pytest.raises(SystemExit):
        setup_preset_cli(parser, agent_library="rl_games")

    output = capsys.readouterr().out
    assert "Registered --agent values for rl_games:" in output
    assert "rl_games_cfg_entry_point (default)" in output
    assert "rl_games_feature_cfg_entry_point" in output
    assert "compatible presets: resnet18, theia_tiny" in output


def test_task_help_lists_alternate_agents_without_compatibility_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Alternate algorithm configs remain discoverable when no preset mapping is needed."""
    monkeypatch.setattr("sys.argv", ["train.py", "--task", "Isaac-Cartpole", "--help"])
    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--task")
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    with pytest.raises(SystemExit):
        setup_preset_cli(parser, agent_library="rsl_rl")

    output = capsys.readouterr().out
    assert "rsl_rl_cfg_entry_point (default)" in output
    assert "rsl_rl_with_symmetry_cfg_entry_point" in output
    assert "Preset compatibility is not declared for this task." in output


def test_list_variants_json_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The JSON listing contains no config-loader messages on stdout."""
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task", "Isaac-Cartpole-Camera", "--list_variants", "json"],
    )
    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--task")
    parser.add_argument("--agent", default="rl_games_cfg_entry_point")
    with pytest.raises(SystemExit):
        setup_preset_cli(parser, agent_library="rl_games")

    output = json.loads(capsys.readouterr().out)
    assert output["task"] == "Isaac-Cartpole-Camera"
    assert output["agent_library"] == "rl_games"
    assert [agent["name"] for agent in output["agents"]] == [
        "rl_games_cfg_entry_point",
        "rl_games_feature_cfg_entry_point",
    ]
