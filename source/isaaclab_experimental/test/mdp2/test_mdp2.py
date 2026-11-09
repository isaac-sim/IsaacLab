# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for building and running MDP2 tasks on an analytic point-mass scene."""

import pytest
import torch
import warp as wp
from isaaclab_experimental import mdp2

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@wp.kernel
def _position_l2_kernel(joint_pos: wp.array2d(dtype=float), out: wp.array(dtype=float)):
    world = wp.tid()
    total = float(0.0)
    for j in range(joint_pos.shape[1]):
        total += joint_pos[world, j] * joint_pos[world, j]
    out[world] = total


@wp.kernel
def _out_of_bounds_kernel(joint_pos: wp.array2d(dtype=float), limit: float, out: wp.array(dtype=bool)):
    world = wp.tid()
    out[world] = False
    for j in range(joint_pos.shape[1]):
        if wp.abs(joint_pos[world, j]) > limit:
            out[world] = True


def position_l2_warp(out, joint_pos):
    wp.launch(_position_l2_kernel, dim=out.shape, inputs=[joint_pos], outputs=[out], device=out.device)


def out_of_bounds_warp(out, joint_pos, *, limit):
    wp.launch(_out_of_bounds_kernel, dim=out.shape, inputs=[joint_pos, limit], outputs=[out], device=out.device)


robot = mdp2.Entity("robot")
joint_state = mdp2.Term(
    torch=lambda value: value, warp=lambda out, value: wp.copy(out, value), output=mdp2.TensorSpec(("worlds", "joints"))
)
alive = mdp2.Term(torch=lambda joint_pos: torch.ones_like(joint_pos[:, 0]), warp=lambda out, joint_pos: out.fill_(1.0))
position_l2 = mdp2.Term(torch=lambda joint_pos: joint_pos.square().sum(dim=-1), warp=position_l2_warp)
out_of_bounds = mdp2.Term(
    torch=lambda joint_pos, *, limit: (joint_pos.abs() > limit).any(dim=-1), warp=out_of_bounds_warp
)


def make_task(
    scene: mdp2.PointMassScene, backend: str = "torch", seconds: float = 1.0, weight: float = -10.0
) -> tuple[mdp2.MDP, mdp2.MDPState]:
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
            "position": mdp2.Reward(position_l2, inputs={"joint_pos": robot.joint_pos}, weight=weight),
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
    mdp = builder.finalize(device=DEVICE, backend=backend)
    return mdp, mdp.state(scene)


def env_step(mdp, state, scene, actions, transition):
    """The phase order of ``MDP2RLEnv.step``, with the analytic scene standing in for the simulation."""
    mdp2.apply_actions(mdp, state, actions)
    scene.step()
    mdp2.evaluate(mdp, state, transition)
    ended = transition.terminated.torch | transition.truncated.torch
    scene.reset(ended)
    mdp2.reset(mdp, state, ended)
    mdp2.observe(mdp, state, transition.observations)


def env_reset(mdp, state, scene, observations):
    """The phase order of ``MDP2RLEnv.reset`` for all worlds."""
    every_world = torch.ones(mdp.num_worlds, dtype=torch.bool, device=DEVICE)
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
    scene = mdp2.PointMassScene(num_worlds=2, joint_names=("x", "y"), device=DEVICE)
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

    actions = torch.tensor([[0.5, -0.5], [0.1, 0.0]], device=DEVICE)
    env_step(mdp, state, scene, actions, transition)

    # Two substeps of semi-implicit Euler: v = 2 a dt, p = 3 a dt^2.
    effort = 2.0 * actions
    joint_pos, joint_vel = 3.0 * effort * 0.01**2, 2.0 * effort * 0.01
    torch.testing.assert_close(transition.observations["policy"].torch, torch.cat([joint_pos, joint_vel], dim=-1))
    torch.testing.assert_close(transition.rewards.torch, (1.0 - 10.0 * joint_pos.square().sum(dim=-1)) * 0.02)
    assert not transition.terminated.torch.any() and not transition.truncated.torch.any()
    assert state.episode_length.torch.tolist() == [1, 1]


def test_step_resets_ended_worlds():
    scene = mdp2.PointMassScene(num_worlds=3, device=DEVICE)
    mdp, state = make_task(scene, seconds=0.04)
    transition = mdp.transition()
    actions = torch.tensor([[0.0], [0.0], [100.0]], device=DEVICE)

    env_step(mdp, state, scene, actions, transition)

    assert transition.terminated.torch.tolist() == [False, False, True]
    assert transition.observations["policy"].torch[2].abs().sum() == 0.0
    assert state.episode_length.torch.tolist() == [1, 1, 0]

    env_step(mdp, state, scene, torch.zeros(3, 1, device=DEVICE), transition)

    assert transition.truncated.torch.tolist() == [True, True, False]
    assert state.episode_length.torch.tolist() == [0, 0, 1]


def test_reset_restores_all_worlds():
    scene = mdp2.PointMassScene(num_worlds=2, device=DEVICE)
    mdp, state = make_task(scene)
    observations = mdp.observations()
    env_step(mdp, state, scene, torch.full((2, 1), 0.01, device=DEVICE), mdp.transition())

    env_reset(mdp, state, scene, observations)

    assert observations["policy"].torch.abs().sum() == 0.0
    assert state.actions.torch.abs().sum() == 0.0
    assert scene.data("robot", "joint_effort_target").torch.abs().sum() == 0.0
    assert state.episode_length.torch.tolist() == [0, 0]


def test_warp_matches_torch():
    runs = {}
    for backend in ("torch", "warp"):
        scene = mdp2.PointMassScene(num_worlds=64, joint_names=("x", "y", "z"), device=DEVICE)
        mdp, state = make_task(scene, backend=backend, seconds=0.2)
        transition = mdp.transition()
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        history = []
        for _ in range(30):
            actions = torch.randn(64, 3, device=DEVICE, generator=generator) * 2.0
            actions[:32] *= 0.01  # these worlds stay in bounds until the time limit
            env_step(mdp, state, scene, actions, transition)
            buffers = (transition.rewards, transition.terminated, transition.truncated, state.episode_length)
            history.append(
                [buffer.torch.clone() for buffer in buffers] + [transition.observations["policy"].torch.clone()]
            )
        runs[backend] = history

    for torch_step, warp_step in zip(runs["torch"], runs["warp"]):
        for expected, actual in zip(torch_step, warp_step):
            torch.testing.assert_close(actual, expected)
    assert any(step[1].any() for step in runs["torch"]) and any(step[2].any() for step in runs["torch"])
