# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect targeted tracking rewards and independent head/foot failure limits."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.utils import load_cfg_from_registry


def test_per_foot_reward_excludes_other_bodies():
    """A missed foot loses its own reward even if all other bodies match perfectly."""
    actual = torch.zeros(3, 3, 3)
    actual[1, 2, 0] = 0.03
    actual[2, [0, 2], 2] = 0.03
    actual[:, 1, :] = 10.0
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["ankle_right", "jaw_soft", "ankle_left"]),
        body_pos_relative_w=torch.zeros_like(actual),
        robot_body_pos_w=actual,
    )
    environment = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    configuration = load_cfg_from_registry("Isaac-Motion-Tracking-MicroDuck-Feet-Head", "env_cfg_entry_point")
    feet = configuration.rewards.motion_feet_pos
    torch.testing.assert_close(
        feet.weight * feet.func(environment, **feet.params),
        4.0 * torch.tensor([1.0, 0.5526996123, 0.1053992246]),
    )


def test_independent_foot_and_head_failure_limits():
    """Foot and head height errors use separate limits and tracked bodies."""
    configuration = load_cfg_from_registry("Isaac-Motion-Tracking-MicroDuck-Feet-Head", "env_cfg_entry_point")
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["ankle_left", "ankle_right", "jaw_soft"]),
        body_pos_relative_w=torch.zeros(4, 3, 3),
        robot_body_pos_w=torch.zeros(4, 3, 3),
    )
    command.robot_body_pos_w[0, 0, 2] = 0.091
    command.robot_body_pos_w[1, 2, 2] = 0.10
    command.robot_body_pos_w[2, 2, 2] = 0.181
    command.robot_body_pos_w[3, 1, 2] = 0.20
    environment = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    feet = configuration.terminations.feet_pos
    head = configuration.terminations.ee_body_pos
    torch.testing.assert_close(feet.func(environment, **feet.params), torch.tensor([True, False, False, True]))
    torch.testing.assert_close(head.func(environment, **head.params), torch.tensor([False, False, True, False]))
    feet.params["threshold"] = 0.18
    torch.testing.assert_close(
        feet.func(environment, **feet.params) | head.func(environment, **head.params),
        torch.tensor([False, False, True, True]),
    )


def test_head_reward_tracks_xyz_independently_of_feet():
    """Head displacement on any axis matters, while unrelated foot errors do not."""
    configuration = load_cfg_from_registry("Isaac-Motion-Tracking-MicroDuck-Feet-Head", "env_cfg_entry_point")
    actual = torch.full((6, 3, 3), 10.0)
    actual[:, 1] = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, -0.02], [0.01, 0.01, 0.01], [0.06, 0.0, 0.0]]
    )
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["ankle_left", "jaw_soft", "ankle_right"]),
        body_pos_relative_w=torch.zeros_like(actual),
        robot_body_pos_w=actual,
    )
    environment = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    head = configuration.rewards.motion_head_pos
    torch.testing.assert_close(
        head.weight * head.func(environment, **head.params),
        4.0 * torch.tensor([1.0, 0.7788007831, 0.3678794412, 0.3678794412, 0.4723665527, 0.0001234098]),
    )


def test_registered_training_defaults_are_independent():
    """The common training defaults must not be changed by another task instance."""
    configuration = load_cfg_from_registry("Isaac-Motion-Tracking-MicroDuck-Feet-Head", "env_cfg_entry_point")
    other = load_cfg_from_registry("Isaac-Motion-Tracking-MicroDuck-Feet-Head", "env_cfg_entry_point")
    assert configuration.rewards.action_rate_l2.weight == -0.05
    assert configuration.terminations.anchor_pos.params["threshold"] == 0.15
    assert configuration.terminations.anchor_ori.params["threshold"] == 0.8
    configuration.terminations.feet_pos.params["threshold"] = 0.02
    assert other.terminations.feet_pos.params["threshold"] == 0.09
    assert configuration.terminations.ee_body_pos.params["threshold"] == 0.18
