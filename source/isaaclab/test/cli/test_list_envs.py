# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the installed environment-listing command."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym

from isaaclab.cli.commands.list_envs import _belongs_to_project, _find_project_task_modules, command_list_envs


def test_find_project_task_modules_uses_nearest_parent(tmp_path):
    """Task discovery must recognize the project that contains the current directory."""
    project_dir = tmp_path / "project"
    nested_dir = project_dir / "scripts" / "nested"
    nested_dir.mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "example"\n[project.entry-points."isaaclab.tasks"]\nexample = "example.tasks"\n'
    )

    assert _find_project_task_modules(nested_dir) == ("example.tasks",)


def test_belongs_to_project_checks_task_configuration_module():
    """Manager-based tasks must match through their downstream environment config."""
    spec = SimpleNamespace(
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": "example.tasks.balance.config.robot.env_cfg:BalanceEnvCfg"},
    )

    assert _belongs_to_project(spec, ("example.tasks",))
    assert not _belongs_to_project(spec, ("another_project.tasks",))


def test_command_lists_non_isaac_prefixed_task_from_current_project(tmp_path, monkeypatch, capsys):
    """Downstream tasks must not need an ``Isaac`` prefix to appear in project-scoped output."""
    task_id = "Example-Balance-Robot"
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": "example.tasks.balance.env_cfg:BalanceEnvCfg"},
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "example"\n[project.entry-points."isaaclab.tasks"]\nexample = "example.tasks"\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("isaaclab.cli.commands.list_envs.importlib.metadata.entry_points", lambda **kwargs: [])

    try:
        command_list_envs([])
    finally:
        del gym.registry[task_id]

    output = capsys.readouterr().out
    assert task_id in output
    assert "Isaac-Cartpole" not in output
