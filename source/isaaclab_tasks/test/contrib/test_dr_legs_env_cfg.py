# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DR Legs task configuration."""

import math

import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.dr_legs.dr_legs_env import DrLegsRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.parametrize(
    ("overrides", "expected_effort_limit", "expected_tilt_range", "expected_armature_randomization"),
    [
        ((), math.inf, None, False),
        (("physics=physx",), 3.1, (-0.1, 0.1), True),
    ],
)
def test_dr_legs_stability_profile_matches_physics_backend(
    overrides: tuple[str, ...],
    expected_effort_limit: float,
    expected_tilt_range: tuple[float, float] | None,
    expected_armature_randomization: bool,
):
    """Apply solver-specific safeguards only to the default Kamino backend."""
    env_cfg = parse_env_cfg("IsaacContrib-DrLegs-Walk", overrides=overrides)
    pose_range = env_cfg.events.reset_base.params["pose_range"]

    assert env_cfg.scene.robot.actuators["driven_joints"].joint_effort_limit == expected_effort_limit
    assert pose_range.get("roll") == expected_tilt_range
    assert pose_range.get("pitch") == expected_tilt_range
    assert (env_cfg.events.randomize_joint_params is not None) is expected_armature_randomization


def test_dr_legs_discards_non_finite_rewards(monkeypatch: pytest.MonkeyPatch):
    """Keep an invalid solver reward from reaching the trainer."""
    rewards = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0, float("nan")])
    terminated = torch.tensor([True, True, True, True, False])
    time_outs = torch.zeros(5, dtype=torch.bool)

    def base_step(_env, _action):
        return {}, rewards, terminated, time_outs, {}

    monkeypatch.setattr(ManagerBasedRLEnv, "step", base_step)
    env = object.__new__(DrLegsRLEnv)
    _, actual_rewards, _, _, _ = env.step(torch.empty(0))

    torch.testing.assert_close(actual_rewards, torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
