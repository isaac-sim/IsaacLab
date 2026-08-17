# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral contracts for cable-routing actions and terminal outcomes."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.cable_routing.mdp.actions import FiniteRelativeJointPositionAction
from isaaclab_tasks.contrib.cable_routing.mdp.commands import CableRoutingCommand
from isaaclab_tasks.contrib.cable_routing.mdp.observations import finite_last_action
from isaaclab_tasks.contrib.cable_routing.mdp.rewards import finite_action_rate_l2, route_failure, route_success


def test_relative_joint_action_holds_one_limit_clamped_target() -> None:
    """A policy delta is resolved once, independent of physics decimation."""
    captured: dict[str, torch.Tensor] = {}
    joint_position = torch.tensor(((0.95, -0.95),))
    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_position),
            default_joint_pos=SimpleNamespace(torch=torch.zeros_like(joint_position)),
            soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor((((-1.0, 1.0), (-1.0, 1.0)),))),
        ),
        set_joint_position_target_index=lambda *, target, joint_ids: captured.update(
            target=target.clone(), joint_ids=torch.as_tensor(joint_ids)
        ),
    )
    term = FiniteRelativeJointPositionAction.__new__(FiniteRelativeJointPositionAction)
    term.cfg = SimpleNamespace(clip=None)
    term._asset = asset
    term._joint_ids = [0, 1]
    term._raw_actions = torch.zeros((1, 2))
    term._scale = 0.1
    term._offset = 0.0

    term.process_actions(torch.tensor(((torch.inf, -1.0),)))
    joint_position.zero_()
    term.apply_actions()
    term.apply_actions()

    torch.testing.assert_close(term.raw_actions, torch.tensor(((1.0, -1.0),)))
    torch.testing.assert_close(captured["target"], torch.tensor(((1.0, -1.0),)))


def test_binary_actions_have_canonical_observation_and_rate_semantics() -> None:
    """Only gripper sign, not raw magnitude, reaches observations and penalties."""
    manager = SimpleNamespace(
        active_terms=["left_arm", "left_gripper", "right_arm", "right_gripper"],
        action_term_dim=[2, 1, 2, 1],
        action=torch.tensor(((0.1, -0.2, 0.01, 0.3, -0.4, -0.01),)),
        prev_action=torch.tensor(((0.1, -0.2, 0.9, 0.3, -0.4, -0.8),)),
    )
    env = SimpleNamespace(action_manager=manager)
    binary_names = ("left_gripper", "right_gripper")

    torch.testing.assert_close(
        finite_last_action(env, binary_action_names=binary_names),
        torch.tensor(((0.1, -0.2, 1.0, 0.3, -0.4, -1.0),)),
    )
    torch.testing.assert_close(finite_action_rate_l2(env, binary_action_names=binary_names), torch.zeros(1))


def test_invalid_termination_overrides_success_for_reward_and_replay_credit() -> None:
    """A simultaneous route completion and invalid state is always a failure."""
    term_values = {
        "invalid_cable": torch.tensor((False, True, False)),
        "invalid_robot_or_action": torch.tensor((False, False, True)),
    }
    command = SimpleNamespace(succeeded=torch.ones(3, dtype=torch.bool), ensure_route_state_current=lambda **_: None)
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=1.0 / 30.0,
        command_manager=SimpleNamespace(get_term=lambda _: command),
        termination_manager=SimpleNamespace(get_term=lambda name: term_values[name]),
    )
    failure_names = tuple(term_values)

    torch.testing.assert_close(
        route_success(env, "route", failure_names) * 20.0 * env.step_dt, torch.tensor((20.0, 0.0, 0.0))
    )
    torch.testing.assert_close(
        route_failure(env, failure_names) * -20.0 * env.step_dt, torch.tensor((0.0, -20.0, -20.0))
    )

    command_term = CableRoutingCommand.__new__(CableRoutingCommand)
    command_term.succeeded = command.succeeded
    command_term.cfg = SimpleNamespace(failure_termination_names=failure_names)
    command_term._env = env
    torch.testing.assert_close(command_term._terminal_success(torch.arange(3)), torch.tensor((True, False, False)))
