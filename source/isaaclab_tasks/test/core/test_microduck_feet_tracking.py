# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect targeted tracking rewards and independent head/foot failure limits."""

from types import SimpleNamespace

import torch

from isaaclab.utils import to_dict

from isaaclab_tasks.core.motion_tracking.config.micro_duck.feet_env_cfg import MicroDuckFeetTrackingEnvCfg
from isaaclab_tasks.core.motion_tracking.config.micro_duck.rewards import motion_feet_position_error_exp
from isaaclab_tasks.core.motion_tracking.config.micro_duck.tracking_env_cfg import MicroDuckMotionTrackingEnvCfg
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
    reward = motion_feet_position_error_exp(environment, "motion", ["ankle_left", "ankle_right"], 0.02)
    torch.testing.assert_close(reward, torch.tensor([1.0, 0.5526996123, 0.1053992246]))


def test_independent_foot_and_head_failure_limits():
    """Tightening foot tolerance must neither tighten head tolerance nor alter the baseline."""
    baseline = MicroDuckMotionTrackingEnvCfg()
    focused = MicroDuckFeetTrackingEnvCfg()
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["ankle_left", "ankle_right", "jaw_soft"]),
        body_pos_relative_w=torch.zeros(3, 3, 3),
        robot_body_pos_w=torch.zeros(3, 3, 3),
    )
    command.robot_body_pos_w[0, 0, 2] = 0.031
    command.robot_body_pos_w[1, 2, 2] = 0.04
    command.robot_body_pos_w[2, 2, 2] = 0.061
    environment = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    feet = focused.terminations.feet_pos
    head = focused.terminations.ee_body_pos
    combined = baseline.terminations.ee_body_pos
    torch.testing.assert_close(feet.func(environment, **feet.params), torch.tensor([True, False, False]))
    torch.testing.assert_close(head.func(environment, **head.params), torch.tensor([False, False, True]))
    torch.testing.assert_close(combined.func(environment, **combined.params), torch.tensor([False, False, True]))
    feet.params["threshold"] = 0.06
    torch.testing.assert_close(
        feet.func(environment, **feet.params) | head.func(environment, **head.params),
        combined.func(environment, **combined.params),
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


def test_registered_head_task_only_adds_head_reward():
    """Opting in must preserve observations, dynamics, foot rewards and failure limits."""
    feet = MicroDuckFeetTrackingEnvCfg()
    focused = load_cfg_from_registry("Isaac-Motion-Tracking-MicroDuck-Feet-Head", "env_cfg_entry_point")
    focused_configuration = to_dict(focused)
    head_reward = focused_configuration["rewards"].pop("motion_head_pos")
    assert head_reward["weight"] == 4.0
    assert focused_configuration == to_dict(feet)
    focused.terminations.feet_pos.params["threshold"] = 0.02
    assert feet.terminations.feet_pos.params["threshold"] == 0.03
    assert focused.terminations.ee_body_pos.params["threshold"] == 0.06
