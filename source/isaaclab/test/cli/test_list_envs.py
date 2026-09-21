# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the installed environment-listing command."""

from __future__ import annotations

import gymnasium as gym

from isaaclab.cli.commands.list_envs import command_list_envs


def test_command_lists_non_isaac_prefixed_task_from_current_project(tmp_path, monkeypatch, capsys):
    """Downstream tasks must not need an ``Isaac`` prefix to appear in project-scoped output."""
    task_id = "Example-Balance-Robot"
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": "example.tasks.balance.env_cfg:BalanceEnvCfg"},
    )
    project_dir = tmp_path / "project"
    working_dir = project_dir / "scripts" / "nested"
    working_dir.mkdir(parents=True)
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "example"\n[project.entry-points."isaaclab.tasks"]\nexample = "example.tasks"\n'
    )
    monkeypatch.chdir(working_dir)
    monkeypatch.setattr("isaaclab.cli.commands.list_envs.importlib.metadata.entry_points", lambda **kwargs: [])

    try:
        command_list_envs([])
    finally:
        del gym.registry[task_id]

    output = capsys.readouterr().out
    assert task_id in output
    assert "Isaac-Cartpole" not in output
