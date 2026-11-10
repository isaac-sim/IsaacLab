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


@wp.func
def joint_pos_warp(world: int, joint: int, joint_pos: wp.array2d(dtype=float)) -> float:
    return joint_pos[world, joint]


@wp.func
def joint_vel_warp(world: int, joint: int, joint_vel: wp.array2d(dtype=float)) -> float:
    return joint_vel[world, joint]


@wp.func
def generated_commands_warp(world: int, command: wp.array(dtype=float)) -> float:
    return command[world]


@wp.func
def is_alive_warp(world: int, joint_pos: wp.array2d(dtype=float)) -> float:
    return 1.0


@wp.func
def joint_pos_l2_warp(world: int, joint_pos: wp.array2d(dtype=float)) -> float:
    total = float(0.0)
    for j in range(joint_pos.shape[1]):
        total += joint_pos[world, j] * joint_pos[world, j]
    return total


@wp.func
def joint_pos_out_of_manual_limit_warp(world: int, joint_pos: wp.array2d(dtype=float), limit: float) -> bool:
    outside = bool(False)
    for j in range(joint_pos.shape[1]):
        outside = outside or wp.abs(joint_pos[world, j]) > limit
    return outside


@wp.func
def delta_warp(world: int, joint: int, value: wp.array2d(dtype=float), last_value: wp.array2d(dtype=float)) -> float:
    return value[world, joint] - last_value[world, joint]


@wp.func
def delta_update_warp(world: int, value: wp.array2d(dtype=float), last_value: wp.array2d(dtype=float)):
    for j in range(value.shape[1]):
        last_value[world, j] = value[world, j]


@wp.kernel
def _copy_masked_kernel(
    mask: wp.array(dtype=bool), value: wp.array2d(dtype=float), last_value: wp.array2d(dtype=float)
):
    world, j = wp.tid()
    if mask[world]:
        last_value[world, j] = value[world, j]


def delta_reset_warp(mask, value, last_value):
    wp.launch(_copy_masked_kernel, dim=value.shape, inputs=[mask, value], outputs=[last_value], device=value.device)


@wp.func
def joint_vel_delta_warp(
    world: int, joint_vel: wp.array2d(dtype=float), last_joint_vel: wp.array(dtype=float)
) -> float:
    return joint_vel[world, 0] - last_joint_vel[world]


@wp.func
def joint_vel_delta_update_warp(world: int, joint_vel: wp.array2d(dtype=float), last_joint_vel: wp.array(dtype=float)):
    last_joint_vel[world] = joint_vel[world, 0]


@wp.func
def uniform_command_warp(world: int, rng: wp.int32) -> float:
    return wp.randf(wp.rand_init(rng, 0), -1.0, 1.0)


@wp.kernel
def _add_uniform_noise_kernel(
    joint_pos: wp.array2d(dtype=float), rng: wp.array(dtype=wp.int32), out: wp.array2d(dtype=float)
):
    world, j = wp.tid()
    out[world, j] = joint_pos[world, j] + wp.randf(wp.rand_init(rng[world], j), -0.1, 0.1)


def noisy_joint_pos_warp(out, joint_pos, rng):
    wp.launch(_add_uniform_noise_kernel, dim=out.shape, inputs=[joint_pos, rng], outputs=[out], device=out.device)


@wp.func
def push_by_adding_velocity_warp(world: int, joint_vel: wp.array2d(dtype=float)):
    for j in range(joint_vel.shape[1]):
        joint_vel[world, j] += 1.0


@wp.kernel
def _reset_joints_by_offset_kernel(
    mask: wp.array(dtype=bool), rng: wp.array(dtype=wp.int32), joint_pos: wp.array2d(dtype=float)
):
    world, j = wp.tid()
    if mask[world]:
        joint_pos[world, j] += wp.randf(wp.rand_init(rng[world], j), -0.5, 0.5)


def reset_joints_by_offset_warp(mask, joint_pos, rng):
    wp.launch(
        _reset_joints_by_offset_kernel,
        dim=joint_pos.shape,
        inputs=[mask, rng],
        outputs=[joint_pos],
        device=joint_pos.device,
    )


robot = mdp2.Entity("robot")
joint_pos = mdp2.Term(torch=lambda joint_pos: joint_pos, warp=joint_pos_warp, shape=("worlds", "joints"))
joint_vel = mdp2.Term(torch=lambda joint_vel: joint_vel, warp=joint_vel_warp, shape=("worlds", "joints"))
generated_commands = mdp2.Term(torch=lambda command: command, warp=generated_commands_warp)
is_alive = mdp2.Term(torch=lambda joint_pos: torch.ones_like(joint_pos[:, 0]), warp=is_alive_warp)
joint_pos_l2 = mdp2.Term(torch=lambda joint_pos: joint_pos.square().sum(dim=-1), warp=joint_pos_l2_warp)
joint_pos_out_of_manual_limit = mdp2.Term(
    torch=lambda joint_pos, *, limit: (joint_pos.abs() > limit).any(dim=-1), warp=joint_pos_out_of_manual_limit_warp
)
delta = mdp2.Term(
    torch=mdp2.Implementation(
        compute=lambda value, last_value: value - last_value,
        update=lambda value, last_value: last_value.copy_(value),
        reset=lambda mask, value, last_value: last_value.copy_(torch.where(mask[:, None], value, last_value)),
    ),
    warp=mdp2.Implementation(compute=delta_warp, update=delta_update_warp, reset=delta_reset_warp),
    shape=("worlds", "joints"),
    state={"last_value": mdp2.State(("worlds", "joints"))},
)
"""Change of a field since the last step, zero after a reset."""
joint_vel_delta = mdp2.Term(
    torch=mdp2.Implementation(
        compute=lambda joint_vel, last_joint_vel: joint_vel[:, 0] - last_joint_vel,
        update=lambda joint_vel, last_joint_vel: last_joint_vel.copy_(joint_vel[:, 0]),
    ),
    warp=mdp2.Implementation(compute=joint_vel_delta_warp, update=joint_vel_delta_update_warp),
    state={"last_joint_vel": mdp2.State()},
)
"""Change of the first joint's velocity since the last step. Its state restarts at zero."""
uniform_command = mdp2.Term(torch=lambda rng: mdp2.uniform(rng, -1.0, 1.0), warp=uniform_command_warp, random=True)
noisy_joint_pos = mdp2.Term(
    torch=lambda joint_pos, rng: joint_pos + mdp2.uniform(rng, -0.1, 0.1, joint_pos.shape[1:]),
    warp=noisy_joint_pos_warp,
    shape=("worlds", "joints"),
    random=True,
)
push_by_adding_velocity = mdp2.Term(
    torch=lambda mask, joint_vel: joint_vel.add_(mask[:, None].float()), warp=push_by_adding_velocity_warp
)
"""Add one to the joint velocities of the masked worlds."""
reset_joints_by_offset = mdp2.Term(
    torch=lambda mask, joint_pos, rng: joint_pos.add_(
        mask[:, None] * mdp2.uniform(rng, -0.5, 0.5, joint_pos.shape[1:])
    ),
    warp=reset_joints_by_offset_warp,
    random=True,
)
"""Offset the joint positions of the masked worlds by uniform samples."""
push_robot = mdp2.Event(
    push_by_adding_velocity,
    inputs={"joint_vel": robot.joint_vel},
    mode=mdp2.EventMode.INTERVAL,
    interval_range_s=(0.04, 0.04),
)


def make_task(
    scene: mdp2.PointMassScene, backend: str = "torch", episode_length_s: float = 1.0, random: bool = False
) -> tuple[mdp2.MDP, mdp2.MDPState]:
    """A point-mass task. With ``random``, it also has a command, noisy observations, and events."""
    task = mdp2.TaskCfg(
        actions={"joint_effort": mdp2.DirectAction(robot.joint_effort_target, scale=2.0)},
        observations={
            "policy": mdp2.ObservationGroup(
                joint_pos=mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos}),
                joint_vel=mdp2.Observation(joint_vel, inputs={"joint_vel": robot.joint_vel}),
                joint_pos_delta=mdp2.Observation(delta, inputs={"value": robot.joint_pos}),
                joint_vel_delta=mdp2.Observation(delta, inputs={"value": robot.joint_vel}),
            )
        },
        rewards={
            "alive": mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_pos}),
            "joint_pos_l2": mdp2.Reward(joint_pos_l2, inputs={"joint_pos": robot.joint_pos}, weight=-10.0),
            "disabled": None,
        },
        terminations={
            "time_out": mdp2.TimeLimit(episode_length_s=episode_length_s),
            "out_of_bounds": mdp2.Termination(
                joint_pos_out_of_manual_limit,
                inputs={"joint_pos": robot.joint_pos},
                params={"limit": 0.5 if random else 1e-3},
            ),
        },
    )
    builder = mdp2.MDPBuilder(scene.schema)
    builder.add_task(task)
    if random:
        builder.add_command("target", mdp2.Command(uniform_command, resampling_time_range=(0.02, 0.06)))
        builder.add_reward("target", mdp2.Reward(generated_commands, inputs={"command": mdp2.commands.target}))
        # Its state must advance before the push event writes the velocity.
        builder.add_reward("joint_vel_delta", mdp2.Reward(joint_vel_delta, inputs={"joint_vel": robot.joint_vel}))
        for name in ("joint_pos", "joint_pos_again"):
            observation = mdp2.Observation(noisy_joint_pos, inputs={"joint_pos": robot.joint_pos})
            builder.add_observation("noisy", name, observation)
        builder.add_observation(
            "policy", "target", mdp2.Observation(generated_commands, inputs={"command": mdp2.commands.target})
        )
        builder.add_event("push_robot", push_robot)
        builder.add_event(
            "reset_robot_joints", mdp2.Event(reset_joints_by_offset, inputs={"joint_pos": robot.joint_pos})
        )
    mdp = builder.finalize(device=DEVICE, backend=backend)
    return mdp, mdp.state(scene)


def env_step(mdp, state, scene, actions, transition) -> dict[str, torch.Tensor]:
    """The phase order of ``MDP2RLEnv.step``, with the analytic scene standing in for the simulation.

    Returns the bootstrap observations: the successor observations before ended worlds are reset.
    """
    mdp2.apply_actions(mdp, state, actions)
    scene.step()
    mdp2.evaluate(mdp, state, transition)
    bootstrap = {group: values.torch.clone() for group, values in transition.observations.items()}
    ended = transition.terminated.torch | transition.truncated.torch
    scene.reset(ended)
    mdp2.reset(mdp, state, ended)
    mdp2.observe(mdp, state, transition.observations, ended)
    return bootstrap


def env_reset(mdp, state, scene, observations, mask):
    """The phase order of ``MDP2RLEnv.reset``."""
    scene.reset(mask)
    mdp2.reset(mdp, state, mask)
    mdp2.observe(mdp, state, observations, mask)


def test_finalize_reports_all_errors():
    builder = mdp2.MDPBuilder(mdp2.PointMassScene(num_worlds=1).schema)
    builder.add_reward("unknown_entity", mdp2.Reward(is_alive, inputs={"joint_pos": mdp2.Entity("table").joint_pos}))
    builder.add_reward("unknown_field", mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_angle}))
    builder.add_reward("no_warp", mdp2.Reward(mdp2.Term(), inputs={"joint_pos": robot.joint_pos}))
    builder.add_observation(
        "policy",
        "name_clash",
        mdp2.Observation(delta, inputs={"value": robot.joint_pos, "last_value": robot.joint_pos}),
    )
    builder.add_event(
        "no_interval",
        mdp2.Event(push_by_adding_velocity, inputs={"joint_vel": robot.joint_vel}, mode=mdp2.EventMode.INTERVAL),
    )
    builder.add_reward("func_arguments", mdp2.Reward(joint_pos_l2, inputs={"pos": robot.joint_pos}))
    builder.add_event("function_arguments", mdp2.Event(reset_joints_by_offset, inputs={"pos": robot.joint_pos}))

    with pytest.raises(ValueError) as error:
        builder.finalize(backend="warp")

    message = str(error.value)
    assert "rewards.unknown_entity: input 'joint_pos' selects unknown entity 'table'" in message
    assert "rewards.unknown_field: input 'joint_pos' selects unknown field 'robot.joint_angle'" in message
    assert "rewards.no_warp: the term has no 'warp' implementation" in message
    assert "observations.policy.name_clash: inputs, state, params, and rng must have distinct names" in message
    assert "events.no_interval: interval events need an interval_range_s" in message
    assert (
        "rewards.func_arguments: wp.func 'joint_pos_l2_warp' must take the world, 0 output indices, and ['pos']"
        in message
    )
    assert "events.function_arguments: 'reset_joints_by_offset_warp' is called with ['mask', 'pos', 'rng']" in message


def test_step_matches_point_mass_dynamics():
    scene = mdp2.PointMassScene(num_worlds=2, joint_names=("x", "y"), device=DEVICE)
    mdp, state = make_task(scene)
    transition = mdp.transition()

    assert mdp.schedule == (
        "actions.joint_effort",
        "terminations.out_of_bounds",
        "rewards.alive",
        "rewards.joint_pos_l2",
        "observations.policy.joint_pos",
        "observations.policy.joint_vel",
        "observations.policy.joint_pos_delta",
        "observations.policy.joint_vel_delta",
    )
    assert (mdp.action_dim, mdp.observation_dims["policy"], mdp.max_episode_length) == (2, 8, 50)

    actions = torch.tensor([[0.25, -0.25], [0.1, 0.0]], device=DEVICE)
    env_step(mdp, state, scene, actions, transition)
    env_step(mdp, state, scene, actions, transition)

    # Four substeps of semi-implicit Euler: v = 4 a dt, p = 10 a dt^2. The second step added 2 a dt and 7 a dt^2,
    # which the delta observations report only if their state advanced exactly once per step.
    effort, dt = 2.0 * actions, 0.01
    pos, vel = 10.0 * effort * dt**2, 4.0 * effort * dt
    expected = torch.cat([pos, vel, 7.0 * effort * dt**2, 2.0 * effort * dt], dim=-1)
    torch.testing.assert_close(transition.observations["policy"].torch, expected)
    torch.testing.assert_close(transition.rewards.torch, (1.0 - 10.0 * pos.square().sum(dim=-1)) * 0.02)
    assert not transition.terminated.torch.any() and not transition.truncated.torch.any()
    assert state.episode_length.torch.tolist() == [2, 2]


def test_step_ends_and_bootstraps_worlds():
    """In one step, world 0 continues, world 1 times out, and world 2 leaves the bounds."""
    scene = mdp2.PointMassScene(num_worlds=3, device=DEVICE)
    mdp, state = make_task(scene, episode_length_s=0.2)
    transition = mdp.transition()
    state.episode_length.torch[1] = mdp.max_episode_length - 1

    bootstrap = env_step(mdp, state, scene, torch.tensor([[0.1], [0.1], [100.0]], device=DEVICE), transition)

    assert transition.terminated.torch.tolist() == [False, False, True]
    assert transition.truncated.torch.tolist() == [False, True, False]
    assert state.episode_length.torch.tolist() == [1, 0, 0]
    observations, bootstrap = transition.observations["policy"].torch, bootstrap["policy"]
    assert torch.equal(observations[0], bootstrap[0])  # continuing worlds keep their successor sample
    assert torch.equal(bootstrap[1], bootstrap[0])  # truncated worlds bootstrap from the pre-reset sample
    assert bootstrap[0].abs().sum() > 0.0 and bootstrap[2].isfinite().all()
    assert observations[1:].abs().sum() == 0.0  # ended worlds observe their new episode


def test_reset_touches_only_selected_worlds():
    scene = mdp2.PointMassScene(num_worlds=3, device=DEVICE)
    mdp, state = make_task(scene)
    observations = mdp.observations()
    env_reset(mdp, state, scene, observations, torch.ones(3, dtype=torch.bool, device=DEVICE))
    for _ in range(2):
        env_step(mdp, state, scene, torch.full((3, 1), 0.01, device=DEVICE), mdp.transition())
    mdp2.observe(mdp, state, observations, torch.ones(3, dtype=torch.bool, device=DEVICE))

    def snapshot():
        buffers = [state.episode_length, state.actions, observations["policy"], scene.data("robot", "joint_pos")]
        buffers += [value for term in state.term_state.values() for value in term.values()]
        return [buffer.torch.clone() for buffer in buffers]

    before = snapshot()
    env_reset(mdp, state, scene, observations, torch.zeros(3, dtype=torch.bool, device=DEVICE))
    for expected, actual in zip(before, snapshot()):
        assert torch.equal(actual, expected)

    env_reset(mdp, state, scene, observations, torch.tensor([True, False, False], device=DEVICE))
    for expected, actual in zip(before, snapshot()):
        assert expected[0].abs().sum() > 0.0 and actual[0].abs().sum() == 0.0
        assert torch.equal(actual[1:], expected[1:])


def test_commands_and_interval_events_follow_their_timers():
    """Rewards see the command that governed the step, and observations see the resampled one."""
    scene = mdp2.PointMassScene(num_worlds=8, device=DEVICE)
    builder = mdp2.MDPBuilder(scene.schema)
    builder.add_command("target", mdp2.Command(uniform_command, resampling_time_range=(0.02, 0.06)))
    builder.add_reward("target", mdp2.Reward(generated_commands, inputs={"command": mdp2.commands.target}))
    builder.add_observation(
        "policy", "target", mdp2.Observation(generated_commands, inputs={"command": mdp2.commands.target})
    )
    builder.add_event("push_robot", push_robot)
    mdp = builder.finalize(device=DEVICE)
    state, transition = mdp.state(scene), mdp.transition()
    env_reset(mdp, state, scene, transition.observations, torch.ones(8, dtype=torch.bool, device=DEVICE))

    resamples = torch.zeros(8, device=DEVICE)
    for _ in range(6):
        previous = transition.observations["policy"].torch[:, 0].clone()
        env_step(mdp, state, scene, torch.zeros(8, 1, device=DEVICE), transition)
        torch.testing.assert_close(transition.rewards.torch, previous * mdp.step_dt)
        resamples += transition.observations["policy"].torch[:, 0] != previous

    assert (resamples >= 2).all() and (resamples < 6).any()  # every one to three steps
    assert torch.equal(scene.data("robot", "joint_vel").torch, torch.full((8, 1), 3.0, device=DEVICE))


def test_random_streams_are_independent_per_world_and_use():
    def noisy_observations(reset_world_0: bool, env: bool) -> torch.Tensor:
        scene = mdp2.PointMassScene(num_worlds=2, joint_names=("x", "y"), device=DEVICE)
        mdp, state = make_task(scene, random=True)
        transition = mdp.transition()
        samples = []
        for step in range(3):
            if reset_world_0 and step == 1:
                env_reset(mdp, state, scene, transition.observations, torch.tensor([True, False], device=DEVICE))
            if env:
                env_step(mdp, state, scene, torch.zeros(2, 1, device=DEVICE), transition)
            else:
                scene.step()
                mdp2.evaluate(mdp, state, transition)
            samples.append(transition.observations["noisy"].torch.clone())
        return torch.stack(samples)

    samples = noisy_observations(reset_world_0=False, env=True)
    assert (samples[..., :2] != samples[..., 2:]).all()  # two uses of one term
    assert (samples[1:] != samples[:-1]).all()  # every step
    # Resetting world 0, or resets and observations of no worlds, leave world 1's draws unchanged.
    assert torch.equal(noisy_observations(reset_world_0=True, env=True)[:, 1], samples[:, 1])
    assert torch.equal(noisy_observations(reset_world_0=False, env=False)[:, 1], samples[:, 1])


def test_warp_fuses_functions_into_few_kernels():
    mdp, _ = make_task(mdp2.PointMassScene(num_worlds=2, device=DEVICE), backend="warp")

    # Observations share the kernel of the terminations and rewards. The state updates wait for them in a second one.
    program = mdp.program
    assert [len(steps) for steps in (program.apply_actions, program.evaluate, program.observe)] == [1, 2, 1]


def test_warp_matches_torch():
    runs = {}
    for backend in ("torch", "warp"):
        scene = mdp2.PointMassScene(num_worlds=64, joint_names=("x", "y", "z"), device=DEVICE)
        mdp, state = make_task(scene, backend=backend, episode_length_s=0.2, random=True)
        transition = mdp.transition()
        env_reset(mdp, state, scene, transition.observations, torch.ones(64, dtype=torch.bool, device=DEVICE))
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        history = []
        for _ in range(30):
            actions = torch.randn(64, 3, device=DEVICE, generator=generator) * 2.0
            actions[:32] *= 0.01  # these worlds stay in bounds until they time out
            bootstrap = env_step(mdp, state, scene, actions, transition)
            buffers = (transition.rewards, transition.terminated, transition.truncated, state.episode_length)
            history.append(
                [buffer.torch.clone() for buffer in buffers]
                + [transition.observations["policy"].torch.clone(), bootstrap["policy"]]
                + [state.rng.torch.clone(), transition.observations["noisy"].torch.clone()]
            )
        runs[backend] = history

    for torch_step, warp_step in zip(runs["torch"], runs["warp"]):
        for expected, actual in zip(torch_step[:-2], warp_step[:-2]):
            torch.testing.assert_close(actual, expected)
        for expected, actual in zip(torch_step[-2:], warp_step[-2:]):
            assert torch.equal(actual, expected)  # random draws are bitwise identical
    assert any(step[1].any() for step in runs["torch"]) and any(step[2].any() for step in runs["torch"])
