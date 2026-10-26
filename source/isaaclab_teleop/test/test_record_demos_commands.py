# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for recording command configuration."""

import math
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.unit


def test_recording_defers_command_resampling(monkeypatch, tmp_path):
    """Recording defers command resampling beyond a realistic demonstration."""
    script = Path(__file__).resolve().parents[3] / "scripts/tools/record_demos.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--task", "Isaac-Reach-Franka", "--teleop_device", "spacemouse"])
    recorder = runpy.run_path(str(script), run_name="record_demos_test")
    cfg, _, _ = recorder["create_environment_config"](str(tmp_path), "reach")

    resampling_time_range = cfg.commands.ee_pose.resampling_time_range
    assert all(math.isfinite(bound) for bound in resampling_time_range)
    assert min(resampling_time_range) > 24 * 60 * 60


def test_recording_success_counts_simulation_steps_while_paused(monkeypatch):
    """An ignored reset during success recording must not let render-only frames complete the demo."""
    import isaaclab_teleop
    from isaaclab_teleop import ControlEvents, IsaacTeleopCfg

    script = Path(__file__).resolve().parents[3] / "scripts/tools/record_demos.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--task", "Isaac-Reach-Franka", "--num_success_steps", "3", "--num_demos", "0"],
    )
    recorder = runpy.run_path(str(script), run_name="record_demos_test")
    loop = recorder["run_simulation_loop"]
    events = iter(
        [
            ControlEvents(is_active=True),
            ControlEvents(is_active=False, should_reset=True),
            ControlEvents(is_active=False),
            ControlEvents(is_active=False),
            ControlEvents(is_active=True),
            ControlEvents(is_active=True),
        ]
    )
    device = MagicMock()
    device.__enter__.return_value = device

    def advance():
        device.last_control_events = next(events)
        return torch.zeros(6)

    device.advance.side_effect = advance
    env = MagicMock(num_envs=1, device="cpu")
    env.sim.is_running.side_effect = [True] * 6 + [False]
    env.sim.is_stopped.return_value = False
    env.scene.env_prim_paths = ["/World/envs/env_0"]
    env.recorder_manager.exported_successful_episode_count = 0
    env.step.return_value = ({},)
    exported_at_steps = []

    def export_episodes(env_ids):
        exported_at_steps.append(env.step.call_count)
        env.recorder_manager.exported_successful_episode_count += 1

    env.recorder_manager.export_episodes.side_effect = export_episodes
    monkeypatch.setitem(loop.__globals__, "create_teleop_device", lambda *args, **kwargs: device)
    monkeypatch.setitem(loop.__globals__, "setup_ui", lambda *args: MagicMock())
    monkeypatch.setattr(isaaclab_teleop, "create_haptic_feedback_driver", lambda *args: None)
    camera_feed_session = MagicMock()
    camera_feed_session.bind.return_value = nullcontext()
    success_term = SimpleNamespace(func=lambda env: torch.tensor([True]), params={})

    count = loop(env, success_term, None, camera_feed_session, IsaacTeleopCfg())

    assert count == 1
    assert exported_at_steps == [3]
