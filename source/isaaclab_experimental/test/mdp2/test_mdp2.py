# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for building and running MDP2 tasks on an analytic point-mass scene."""

import pytest
import torch
from isaaclab_experimental import mdp2

robot = mdp2.Entity("robot")
joint_state = mdp2.Term(torch=lambda value: value, output=mdp2.TensorSpec(("worlds", "joints")))
alive = mdp2.Term(torch=lambda joint_pos: torch.ones_like(joint_pos[:, 0]))
position_l2 = mdp2.Term(torch=lambda joint_pos: joint_pos.square().sum(dim=-1))
out_of_bounds = mdp2.Term(torch=lambda joint_pos, *, limit: (joint_pos.abs() > limit).any(dim=-1))


def make_task(scene: mdp2.PointMassScene, seconds: float = 1.0) -> tuple[mdp2.MDP, mdp2.MDPState]:
    cfg = mdp2.TaskCfg(
        actions={"effort": mdp2.DirectAction(robot.joint_effort_target, scale=2.0)},
        observations={
            "policy": mdp2.ObservationGroup(
                joint_pos=mdp2.Observation(joint_state, inputs={"value": robot.joint_pos}),
                joint_vel=mdp2.Observation(joint_state, inputs={"value": robot.joint_vel}),
            )
        },
        rewards={
            "alive": mdp2.Reward(alive, inputs={"joint_pos": robot.joint_pos}),
            "position": mdp2.Reward(position_l2, inputs={"joint_pos": robot.joint_pos}, weight=-10.0),
            "disabled": None,
        },
        terminations={
            "time_limit": mdp2.TimeLimit(seconds=seconds),
            "out_of_bounds": mdp2.Termination(
                out_of_bounds, inputs={"joint_pos": robot.joint_pos}, params={"limit": 1e-3}
            ),
        },
    )
    builder = mdp2.MDPBuilder(scene.schema)
    builder.add_config(cfg)
    mdp = builder.finalize(device="cpu")
    return mdp, mdp.state(scene)


def env_step(mdp, state, scene, actions, transition):
    """The phase order of ``MDP2RLEnv.step``, with the analytic scene standing in for the simulation."""
    mdp2.apply_actions(mdp, state, actions)
    scene.step()
    mdp2.evaluate(mdp, state, transition)
    ended = transition.terminated | transition.truncated
    scene.reset(ended)
    mdp2.reset(mdp, state, ended)
    mdp2.observe(mdp, state, transition.observations)


def env_reset(mdp, state, scene, observations):
    """The phase order of ``MDP2RLEnv.reset`` for all worlds."""
    every_world = torch.ones(mdp.num_worlds, dtype=torch.bool)
    scene.reset(every_world)
    mdp2.reset(mdp, state, every_world)
    mdp2.observe(mdp, state, observations)


def test_finalize_reports_all_errors():
    builder = mdp2.MDPBuilder(mdp2.PointMassScene(num_worlds=1).schema)
    builder.add_reward("unknown_entity", mdp2.Reward(alive, inputs={"joint_pos": mdp2.Entity("table").joint_pos}))
    builder.add_reward("unknown_field", mdp2.Reward(alive, inputs={"joint_pos": robot.joint_angle}))
    builder.add_reward("no_torch", mdp2.Reward(mdp2.Term(), inputs={"joint_pos": robot.joint_pos}))

    with pytest.raises(ValueError) as error:
        builder.finalize()

    message = str(error.value)
    assert "rewards.unknown_entity: input 'joint_pos' selects unknown entity 'table'" in message
    assert "rewards.unknown_field: input 'joint_pos' selects unknown field 'robot.joint_angle'" in message
    assert "rewards.no_torch: the term has no 'torch' implementation" in message


def test_step_matches_point_mass_dynamics():
    scene = mdp2.PointMassScene(num_worlds=2, joint_names=("x", "y"))
    mdp, state = make_task(scene)
    transition = mdp.transition()

    assert mdp.schedule == (
        "actions.effort",
        "terminations.out_of_bounds",
        "rewards.alive",
        "rewards.position",
        "observations.policy.joint_pos",
        "observations.policy.joint_vel",
    )
    assert (mdp.action_dim, mdp.observation_dims["policy"], mdp.max_episode_length) == (2, 4, 50)

    actions = torch.tensor([[0.5, -0.5], [0.1, 0.0]])
    env_step(mdp, state, scene, actions, transition)

    # Two substeps of semi-implicit Euler: v = 2 a dt, p = 3 a dt^2.
    effort = 2.0 * actions
    joint_pos, joint_vel = 3.0 * effort * 0.01**2, 2.0 * effort * 0.01
    torch.testing.assert_close(transition.observations["policy"], torch.cat([joint_pos, joint_vel], dim=-1))
    torch.testing.assert_close(transition.rewards, (1.0 - 10.0 * joint_pos.square().sum(dim=-1)) * 0.02)
    assert not transition.terminated.any() and not transition.truncated.any()
    assert state.episode_length.tolist() == [1, 1]


def test_step_resets_ended_worlds():
    scene = mdp2.PointMassScene(num_worlds=3)
    mdp, state = make_task(scene, seconds=0.04)
    transition = mdp.transition()
    actions = torch.tensor([[0.0], [0.0], [100.0]])

    env_step(mdp, state, scene, actions, transition)

    assert transition.terminated.tolist() == [False, False, True]
    assert transition.observations["policy"][2].abs().sum() == 0.0
    assert state.episode_length.tolist() == [1, 1, 0]

    env_step(mdp, state, scene, torch.zeros(3, 1), transition)

    assert transition.truncated.tolist() == [True, True, False]
    assert state.episode_length.tolist() == [0, 0, 1]


def test_reset_restores_all_worlds():
    scene = mdp2.PointMassScene(num_worlds=2)
    mdp, state = make_task(scene)
    observations = mdp.observations()
    env_step(mdp, state, scene, torch.full((2, 1), 0.01), mdp.transition())

    env_reset(mdp, state, scene, observations)

    assert observations["policy"].abs().sum() == 0.0
    assert state.actions.abs().sum() == 0.0
    assert scene.data("robot", "joint_effort_target").abs().sum() == 0.0
    assert state.episode_length.tolist() == [0, 0]
