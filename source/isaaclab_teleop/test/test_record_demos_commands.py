# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral coverage for command sampling during demonstration recording."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.managers import CommandManager, TerminationManager
from isaaclab.markers.vis_marker_registry import VisMarkerRegistry

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

pytestmark = pytest.mark.unit


@pytest.fixture
def command_env():
    """Provide robot pose inputs to real command and termination managers."""
    # The real command generator needs robot poses to compute its metrics, but
    # command sampling itself does not require a physics scene or input device.
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    robot = SimpleNamespace(
        find_bodies=lambda _: ([0], ["panda_hand"]),
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.zeros(1, 3)),
            root_quat_w=SimpleNamespace(torch=identity),
            body_pos_w=SimpleNamespace(torch=torch.zeros(1, 1, 3)),
            body_quat_w=SimpleNamespace(torch=identity[:, None]),
        ),
    )
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={"robot": robot},
        extras={},
        sim=SimpleNamespace(vis_marker_registry=VisMarkerRegistry(), is_playing=lambda: True),
    )


def test_recorded_reach_goal_changes_only_on_episode_reset(monkeypatch, tmp_path, command_env):
    """Recording holds one target while ordinary training still samples new targets."""
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts/tools/record_demos.py"
    overrides = ["physics=newton_mjwarp", "presets=newton_ik"]
    with monkeypatch.context() as patch:
        patch.setattr(
            sys,
            "argv",
            [str(script), "--task", "Isaac-Reach-Franka", "--teleop_device", "spacemouse", *overrides],
        )
        recorder = runpy.run_path(str(script), run_name="record_demos_test")
        recording_cfg, _, _ = recorder["create_environment_config"](str(tmp_path), "reach")

    training_cfg = parse_env_cfg("Isaac-Reach-Franka", device="cpu", num_envs=1, overrides=overrides)
    recording_cfg.commands.ee_pose.debug_vis = False
    training_cfg.commands.ee_pose.debug_vis = False
    recording_command = UniformPoseCommand(recording_cfg.commands.ee_pose, command_env)
    training_command = UniformPoseCommand(training_cfg.commands.ee_pose, command_env)
    recording_command.reset()
    training_command.reset()
    first_recorded_goal = recording_command.command.clone()
    first_training_goal = training_command.command.clone()

    # Twenty seconds spans several of Reach's ordinary four-second intervals.
    for _ in range(600):
        recording_command.compute(dt=1.0 / 30.0)
        training_command.compute(dt=1.0 / 30.0)
        torch.testing.assert_close(recording_command.command, first_recorded_goal, rtol=0.0, atol=0.0)

    assert not torch.equal(training_command.command, first_training_goal)
    recording_command.reset()
    assert not torch.equal(recording_command.command, first_recorded_goal)


def test_mcap_replay_keeps_success_rewards_valid_without_automatic_resets(monkeypatch, tmp_path, command_env):
    """Manual replay success remains available while its registered term stays inert."""
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts/environments/teleoperation/teleop_replay_agent.py"
    with monkeypatch.context() as patch:
        patch.setattr(
            sys,
            "argv",
            [str(script), "--task", "Isaac-Reach-Franka", "--replay_file", str(tmp_path / "replay.mcap")],
        )
        replay = runpy.run_path(str(script), run_name="teleop_replay_test")
        cfg, success = replay["_prepare_env_cfg"]("Isaac-Reach-Franka", 1, "cpu")

    cfg.commands.ee_pose.debug_vis = False
    command_env.command_manager = CommandManager(cfg.commands, command_env)
    command_env.termination_manager = TerminationManager(cfg.terminations, command_env)
    success_reward = mdp.is_terminated_term(cfg.rewards.success, command_env)
    command = command_env.command_manager.get_term("ee_pose")
    command.pose_command_b[:] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    assert success.func(command_env, **success.params).all()
    assert not command_env.termination_manager.compute().any()
    assert success_reward(command_env, **cfg.rewards.success.params).eq(0.0).all()
