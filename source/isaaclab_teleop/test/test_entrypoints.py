# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`isaaclab_teleop.entrypoints`."""

import importlib
import importlib.metadata
import sys
from contextlib import contextmanager
from unittest import mock

import pytest

pytestmark = pytest.mark.unit


def test_live_entrypoint_preserves_resolved_camera_selection(monkeypatch):
    """The public workflow carries its resolved PiP selection into the selected runtime."""
    from isaaclab_teleop.entrypoints import teleop_se3_agent as workflow

    config, teleop_config, camera = object(), object(), object()
    selection = [("robot_pov_cam", camera, object())]
    resolved = (config, teleop_config, True, selection)
    monkeypatch.setattr(workflow, "_resolve_env_cfg", lambda: resolved)
    calls = []

    @contextmanager
    def launch(env_config, args, *, cleanup_context):
        assert env_config is config and args.using_ovxr
        assert not cleanup_context
        calls.append("launch")
        yield
        calls.append("close")

    monkeypatch.setattr(workflow, "launch_simulation", launch)
    monkeypatch.setattr(workflow, "run_teleoperation", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["caller"])
    assert workflow.run_teleop_se3_agent_cli(["--ovxr", "--ovxr_pip_camera", "robot_pov_cam"]) == 0
    assert calls == ["launch", resolved, "close"]
    assert sys.argv == ["caller"]


@pytest.mark.parametrize(
    ("module_name", "cli_name"),
    [
        ("isaaclab_teleop.entrypoints.teleop_se3_agent", "run_teleop_se3_agent_cli"),
        ("isaaclab_teleop.entrypoints.record_demos", "run_record_demos_cli"),
        ("isaaclab_teleop.entrypoints.replay_demos", "run_replay_demos_cli"),
        ("isaaclab_teleop.entrypoints.teleop_replay_agent", "run_teleop_replay_agent_cli"),
    ],
)
def test_entrypoints_load_out_of_tree_tasks_before_parsing(monkeypatch, module_name, cli_name):
    """Tasks registered through ``isaaclab.tasks`` load before parsing; ``sys.argv`` and the exit code survive."""
    module = importlib.import_module(module_name)
    calls = []
    task_entry_point = mock.Mock()
    task_entry_point.load.side_effect = lambda: calls.append("load tasks")

    def parse_args(argv):
        calls.append("parse")
        sys.argv = [sys.argv[0], "hydra.override=1"]

    monkeypatch.setattr(sys, "argv", ["caller", "--caller-flag"])
    monkeypatch.setattr(module, "_parse_args", parse_args)
    monkeypatch.setattr(module, "main", lambda: calls.append("main") or 3)
    with mock.patch.object(importlib.metadata, "entry_points", return_value=[task_entry_point]) as entry_points:
        status = getattr(module, cli_name)(["--task", "Out-Of-Tree-Task"])

    entry_points.assert_called_once_with(group="isaaclab.tasks")
    assert calls == ["load tasks", "parse", "main"]
    assert sys.argv == ["caller", "--caller-flag"]
    # only the session replay agent reports an outcome; the other workflows always exit 0
    assert status == (3 if cli_name == "run_teleop_replay_agent_cli" else 0)
