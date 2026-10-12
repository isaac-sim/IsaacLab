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
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.parametrize(
    ("overrides", "expected_effort_limit"),
    [
        ((), math.inf),
        (("physics=newton_kamino",), math.inf),
        (("physics=isaacsim_physx",), 3.1),
        (("physics=physx",), 3.1),
    ],
)
def test_dr_legs_driven_joint_effort_limit_matches_physics_backend(
    overrides: tuple[str, ...], expected_effort_limit: float
):
    """Disable the unstable effort constraint only for the Kamino backend."""
    env_cfg = parse_env_cfg("IsaacContrib-DrLegs-Walk", overrides=overrides)
    actual_effort_limit = env_cfg.scene.robot.actuators["driven_joints"].joint_effort_limit
    assert actual_effort_limit == expected_effort_limit


@pytest.mark.parametrize(
    ("overrides", "expected_tilt_range", "expected_armature_randomization"),
    [
        ((), None, False),
        (("physics=physx",), (-0.1, 0.1), True),
    ],
)
def test_dr_legs_reset_randomization_matches_physics_backend(
    overrides: tuple[str, ...], expected_tilt_range: tuple[float, float] | None, expected_armature_randomization: bool
):
    """Disable base tilt and armature randomization only for the Kamino backend."""
    env_cfg = parse_env_cfg("IsaacContrib-DrLegs-HoldPose", overrides=overrides)
    pose_range = env_cfg.events.reset_base.params["pose_range"]
    assert pose_range.get("roll") == expected_tilt_range
    assert pose_range.get("pitch") == expected_tilt_range
    assert (env_cfg.events.randomize_joint_params is not None) is expected_armature_randomization


def test_dr_legs_discards_non_finite_rewards_only_for_terminated_envs(monkeypatch: pytest.MonkeyPatch):
    """Keep a failed solver state from returning an invalid reward to the trainer."""
    rewards = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0, float("nan")])
    terminated = torch.tensor([True, True, True, True, False])
    time_outs = torch.zeros(5, dtype=torch.bool)

    def base_step(_env, _action):
        return {}, rewards, terminated, time_outs, {}

    monkeypatch.setattr(ManagerBasedRLEnv, "step", base_step)
    env = object.__new__(DrLegsRLEnv)
    _, actual_rewards, _, _, _ = env.step(torch.empty(0))

    torch.testing.assert_close(actual_rewards[:4], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.isnan(actual_rewards[4])


@pytest.mark.parametrize(
    (
        "overrides",
        "expected_action_scale",
        "expected_stiffness",
        "expected_armature",
        "expected_gait_period",
        "expected_velocity_feedback",
        "expected_init_std",
    ),
    [
        ((), 0.4, 15.0, 0.01, 0.6, True, 0.2),
        (("physics=physx",), 0.1, 5.0, None, 1.0, False, 1.0),
    ],
)
def test_dr_legs_walk_training_profile_matches_physics_backend(
    overrides: tuple[str, ...],
    expected_action_scale: float,
    expected_stiffness: float,
    expected_armature: float | None,
    expected_gait_period: float,
    expected_velocity_feedback: bool,
    expected_init_std: float,
):
    """Resolve the stable Newton walking profile without changing the PhysX profile."""
    env_cfg, agent_cfg = resolve_task_config("IsaacContrib-DrLegs-Walk", "rsl_rl_cfg_entry_point", overrides=overrides)

    assert env_cfg.actions.joint_pos.scale == expected_action_scale
    assert env_cfg.scene.robot.actuators["driven_joints"].stiffness == expected_stiffness
    assert env_cfg.scene.robot.actuators["driven_joints"].armature == expected_armature
    assert env_cfg.observations.policy.gait_phase.params["period"] == expected_gait_period
    assert (env_cfg.observations.policy.base_lin_vel is not None) is expected_velocity_feedback
    assert env_cfg.commands.base_velocity.ranges.heading == (
        (0.0, 0.0) if expected_velocity_feedback else (-math.pi, math.pi)
    )
    assert agent_cfg.actor.distribution_cfg.init_std == expected_init_std
