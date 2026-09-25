# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU coverage for keyboard reset-buffer capacity and environment selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab_tasks.contrib.keyboard.mdp.commands import typing_commands
from isaaclab_tasks.contrib.keyboard.so101_env_cfg import CommandsCfg


def _make_command(buffer_size: int | None) -> typing_commands.LetterTypingCommand:
    cfg = CommandsCfg().typing
    assert cfg.reset.buffer_size is None
    cfg.reset.buffer_size = buffer_size
    cfg.max_len = 2
    cfg.letter_length = (1, 2)
    cfg.typeable_slots = (0, 1)
    cfg.backspace_slot = 2
    num_envs = 16
    keyboard = SimpleNamespace(
        root_view=SimpleNamespace(count=num_envs),
        joint_names=[f"key_{index:03d}_joint" for index in range(3)],
        body_names=[f"key_{index:03d}" for index in range(3)],
        data=SimpleNamespace(body_pos_w=SimpleNamespace(torch=torch.zeros(num_envs, 3, 3))),
    )
    robot = SimpleNamespace(
        find_joints=lambda names: ([0], [names[0]]),
        find_bodies=lambda name: ([0], [name]),
        is_fixed_base=True,
        num_base_dofs=0,
        data=SimpleNamespace(
            body_pos_w=SimpleNamespace(torch=torch.zeros(num_envs, 1, 3)),
            body_quat_w=SimpleNamespace(torch=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(num_envs, 1, 1)),
        ),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene={"keyboard": keyboard, "robot": robot},
        sim=SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=Mock())),
    )
    return typing_commands.LetterTypingCommand(cfg, env)


@pytest.mark.parametrize("buffer_size", [None, 3, 19])
def test_reset_buffer_captures_commands_and_reach_from_selected_environments(monkeypatch, buffer_size):
    """Default full batches and explicit partial/tail batches keep commands, snapshots, and residuals aligned."""
    command = _make_command(buffer_size)
    cap = command.num_envs if buffer_size is None else buffer_size
    assert command._cur_buffer_size == cap
    assert command.success_monitor.partition_size == cap
    rows = torch.arange(cap)
    target = torch.stack([rows % 3, (rows + 1) % 3], dim=1)
    typed = target.clone()
    typed[1::2, 0] = (typed[1::2, 0] + 1) % 3
    target_len = torch.full((cap,), 2)
    typed_len = rows % 2 + 1
    prefix = 1 - rows % 2
    monkeypatch.setattr(command, "_sample_diverse_states", lambda count: (target, typed, target_len, typed_len))
    monkeypatch.setattr(command, "_log_buffer_stats", Mock())
    command._ik_offset.zero_()
    command._ik_hover.zero_()
    batches = []

    def solve(ids):
        start = sum(batch.numel() for batch in batches)
        batch = slice(start, start + ids.numel())
        torch.testing.assert_close(command.target[ids], target[batch])
        torch.testing.assert_close(command.typed[ids], typed[batch])
        torch.testing.assert_close(command.target_len[ids], target_len[batch])
        torch.testing.assert_close(command.typed_len[ids], typed_len[batch])
        torch.testing.assert_close(command.prefix_len[ids], prefix[batch])
        command.robot.data.body_pos_w.torch[ids, 0, 0] = ids.float() + 1.0
        batches.append(ids.clone())

    def snapshot(env, ids, assets, *, is_relative):
        return torch.stack([ids.float(), command.target[ids, 0].float(), command.prefix_len[ids].float()], dim=1)

    monkeypatch.setattr(command, "_solve_reset_pose", solve)
    monkeypatch.setattr(typing_commands, "get_reset_state", snapshot)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        command._build_buffer()

    for ids in batches:
        assert ids.unique().numel() == ids.numel()
        if ids.numel() == command.num_envs:
            torch.testing.assert_close(ids, torch.arange(command.num_envs))
        else:
            assert bool((ids >= ids.numel()).any()), "Partial batches must not use only the environment prefix."
    selected = torch.cat(batches)
    torch.testing.assert_close(command._buf_target, target)
    torch.testing.assert_close(command._buf_typed, typed)
    torch.testing.assert_close(command._buf_state[:, 0], selected.float())
    torch.testing.assert_close(command._buf_state[:, 1], target[:, 0].float())
    torch.testing.assert_close(command._buf_state[:, 2], prefix.float())
    torch.testing.assert_close(command._buf_reach, selected.float() + 1.0)
    assert command._buffer_built


@pytest.mark.parametrize("buffer_size", [0, -1])
def test_reset_buffer_rejects_nonpositive_capacity(buffer_size):
    with pytest.raises(ValueError, match="buffer_size must be positive"):
        _make_command(buffer_size)
