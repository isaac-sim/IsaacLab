# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`isaaclab_teleop.entrypoints`."""

import argparse
import importlib
import importlib.metadata
import sys
from contextlib import ExitStack
from unittest import mock

import pytest

pytestmark = pytest.mark.unit


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


@pytest.mark.parametrize("failing_step", ["environment", "teleop device"])
def test_teleop_se3_agent_propagates_startup_failures(monkeypatch, failing_step):
    """A session that cannot start raises, so the command exits non-zero instead of reporting success."""
    import isaaclab_teleop
    from isaaclab_teleop.entrypoints import teleop_se3_agent

    env = mock.MagicMock()
    camera_feed_session = mock.MagicMock()
    error = RuntimeError(f"{failing_step} failure")

    def make(task, cfg):
        if failing_step == "environment":
            raise error
        return env

    def create_teleop_device(*args, **kwargs):
        raise error

    args_cli = argparse.Namespace(
        task="Test-Task-v0",
        device="cpu",
        xr=False,
        disable_external_cameras=True,
        kit_visualizer=False,
        cloudxr_env=None,
        auto_launch_cloudxr=False,
        enable_debug_visualization=False,
    )
    monkeypatch.setattr(teleop_se3_agent, "args_cli", args_cli, raising=False)
    monkeypatch.setattr(teleop_se3_agent, "has_kit", lambda: False)
    monkeypatch.setattr(teleop_se3_agent.gym, "make", make)
    monkeypatch.setattr(teleop_se3_agent, "create_teleop_device", create_teleop_device)
    monkeypatch.setattr(isaaclab_teleop, "XrCameraFeedSession", mock.Mock(prepare=lambda *a, **k: camera_feed_session))

    with pytest.raises(RuntimeError) as raised, ExitStack() as cleanup:
        teleop_se3_agent.run_teleoperation(mock.MagicMock(), object(), cleanup)

    assert raised.value is error
    camera_feed_session.close.assert_called_once()
    assert env.unwrapped.close.call_count == (failing_step == "teleop device")
