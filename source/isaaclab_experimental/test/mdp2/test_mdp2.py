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


@wp.kernel
def _subtract_kernel(value: wp.array2d(dtype=float), previous: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    world, j = wp.tid()
    out[world, j] = value[world, j] - previous[world, j]


@wp.kernel
def _copy_selected_kernel(
    mask: wp.array(dtype=bool), value: wp.array2d(dtype=float), previous: wp.array2d(dtype=float)
):
    world, j = wp.tid()
    if mask[world]:
        previous[world, j] = value[world, j]


@wp.kernel
def _uniform_kernel(rng: wp.array(dtype=wp.int32), low: float, high: float, out: wp.array(dtype=float)):
    world = wp.tid()
    out[world] = wp.randf(wp.rand_init(rng[world], 0), low, high)


@wp.kernel
def _add_noise_kernel(value: wp.array2d(dtype=float), rng: wp.array(dtype=wp.int32), out: wp.array2d(dtype=float)):
    world, j = wp.tid()
    out[world, j] = value[world, j] + wp.randf(wp.rand_init(rng[world], j), -0.1, 0.1)


@wp.kernel
def _push_kernel(mask: wp.array(dtype=bool), joint_vel: wp.array2d(dtype=float)):
    world, j = wp.tid()
    if mask[world]:
        joint_vel[world, j] += 1.0


@wp.kernel
def _scatter_kernel(mask: wp.array(dtype=bool), rng: wp.array(dtype=wp.int32), joint_pos: wp.array2d(dtype=float)):
    world, j = wp.tid()
    if mask[world]:
        joint_pos[world, j] = wp.randf(wp.rand_init(rng[world], j), -0.5, 0.5)


def position_l2_warp(out, joint_pos):
    wp.launch(_position_l2_kernel, dim=out.shape, inputs=[joint_pos], outputs=[out], device=out.device)


def out_of_bounds_warp(out, joint_pos, *, limit):
    wp.launch(_out_of_bounds_kernel, dim=out.shape, inputs=[joint_pos, limit], outputs=[out], device=out.device)


def change_warp(out, value, previous):
    wp.launch(_subtract_kernel, dim=out.shape, inputs=[value, previous], outputs=[out], device=out.device)


def change_reset_warp(mask, value, previous):
    wp.launch(_copy_selected_kernel, dim=value.shape, inputs=[mask, value], outputs=[previous], device=value.device)


def target_warp(out, rng):
    wp.launch(_uniform_kernel, dim=out.shape, inputs=[rng, -1.0, 1.0], outputs=[out], device=out.device)


def noisy_warp(out, value, rng):
    wp.launch(_add_noise_kernel, dim=out.shape, inputs=[value, rng], outputs=[out], device=out.device)


def push_warp(mask, joint_vel):
    wp.launch(_push_kernel, dim=joint_vel.shape, inputs=[mask], outputs=[joint_vel], device=joint_vel.device)


def scatter_warp(mask, joint_pos, rng):
    wp.launch(_scatter_kernel, dim=joint_pos.shape, inputs=[mask, rng], outputs=[joint_pos], device=joint_pos.device)


robot = mdp2.Entity("robot")
joint_state = mdp2.Term(
    torch=lambda value: value, warp=lambda out, value: wp.copy(out, value), shape=("worlds", "joints")
)
alive = mdp2.Term(torch=lambda joint_pos: torch.ones_like(joint_pos[:, 0]), warp=lambda out, joint_pos: out.fill_(1.0))
position_l2 = mdp2.Term(torch=lambda joint_pos: joint_pos.square().sum(dim=-1), warp=position_l2_warp)
out_of_bounds = mdp2.Term(
    torch=lambda joint_pos, *, limit: (joint_pos.abs() > limit).any(dim=-1), warp=out_of_bounds_warp
)
change = mdp2.Term(
    torch=mdp2.Implementation(
        compute=lambda value, previous: value - previous,
        update=lambda value, previous: previous.copy_(value),
        reset=lambda mask, value, previous: previous.copy_(torch.where(mask[:, None], value, previous)),
    ),
    warp=mdp2.Implementation(
        compute=change_warp, update=lambda value, previous: wp.copy(previous, value), reset=change_reset_warp
    ),
    shape=("worlds", "joints"),
    state={"previous": mdp2.State(("worlds", "joints"))},
)
"""Change of a field since the last step, zero after a reset."""
identity = mdp2.Term(torch=lambda value: value, warp=lambda out, value: wp.copy(out, value))
target = mdp2.Term(torch=lambda rng: mdp2.uniform(rng, -1.0, 1.0), warp=target_warp, random=True)
noisy = mdp2.Term(
    torch=lambda value, rng: value + mdp2.uniform(rng, -0.1, 0.1, value.shape[1:]),
    warp=noisy_warp,
    shape=("worlds", "joints"),
    random=True,
)
push = mdp2.Term(torch=lambda mask, joint_vel: joint_vel.add_(mask[:, None].float()), warp=push_warp)
"""Add one to the velocity of the masked worlds."""
scatter = mdp2.Term(
    torch=lambda mask, joint_pos, rng: joint_pos.copy_(
        torch.where(mask[:, None], mdp2.uniform(rng, -0.5, 0.5, joint_pos.shape[1:]), joint_pos)
    ),
    warp=scatter_warp,
    random=True,
)
"""Place the masked worlds at random positions."""
push_every_two_steps = mdp2.Event(
    push, inputs={"joint_vel": robot.joint_vel}, mode=mdp2.EventMode.INTERVAL, interval=(0.04, 0.04)
)


def make_task(
    scene: mdp2.PointMassScene, backend: str = "torch", seconds: float = 1.0, random: bool = False
) -> tuple[mdp2.MDP, mdp2.MDPState]:
    """A point-mass task. With ``random``, it also has a command, noisy observations, and events."""
    task = mdp2.TaskCfg(
        actions={"effort": mdp2.DirectAction(robot.joint_effort_target, scale=2.0)},
        observations={
            "policy": mdp2.ObservationGroup(
                joint_pos=mdp2.Observation(joint_state, inputs={"value": robot.joint_pos}),
                joint_vel=mdp2.Observation(joint_state, inputs={"value": robot.joint_vel}),
                joint_pos_change=mdp2.Observation(change, inputs={"value": robot.joint_pos}),
                joint_vel_change=mdp2.Observation(change, inputs={"value": robot.joint_vel}),
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
                out_of_bounds, inputs={"joint_pos": robot.joint_pos}, params={"limit": 0.5 if random else 1e-3}
            ),
        },
    )
    builder = mdp2.MDPBuilder(scene.schema)
    builder.add_task(task)
    if random:
        builder.add_command("target", mdp2.Command(target, resampling_time=(0.02, 0.06)))
        builder.add_reward("target", mdp2.Reward(identity, inputs={"value": mdp2.commands.target}))
        builder.add_observation("policy", "target", mdp2.Observation(identity, inputs={"value": mdp2.commands.target}))
        for name in ("noisy_pos", "noisy_pos_again"):
            builder.add_observation("noisy", name, mdp2.Observation(noisy, inputs={"value": robot.joint_pos}))
        builder.add_event("push", push_every_two_steps)
        builder.add_event("scatter", mdp2.Event(scatter, inputs={"joint_pos": robot.joint_pos}))
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
    builder.add_reward("unknown_entity", mdp2.Reward(alive, inputs={"joint_pos": mdp2.Entity("table").joint_pos}))
    builder.add_reward("unknown_field", mdp2.Reward(alive, inputs={"joint_pos": robot.joint_angle}))
    builder.add_reward("no_torch", mdp2.Reward(mdp2.Term(), inputs={"joint_pos": robot.joint_pos}))
    builder.add_observation(
        "policy", "name_clash", mdp2.Observation(change, inputs={"value": robot.joint_pos, "previous": robot.joint_pos})
    )
    builder.add_event("no_interval", mdp2.Event(push, mode=mdp2.EventMode.INTERVAL))

    with pytest.raises(ValueError) as error:
        builder.finalize()

    message = str(error.value)
    assert "rewards.unknown_entity: input 'joint_pos' selects unknown entity 'table'" in message
    assert "rewards.unknown_field: input 'joint_pos' selects unknown field 'robot.joint_angle'" in message
    assert "rewards.no_torch: the term has no 'torch' implementation" in message
    assert "observations.policy.name_clash: inputs, state, params, and rng must have distinct names" in message
    assert "events.no_interval: interval events need an interval" in message


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
        "observations.policy.joint_pos_change",
        "observations.policy.joint_vel_change",
    )
    assert (mdp.action_dim, mdp.observation_dims["policy"], mdp.max_episode_length) == (2, 8, 50)

    actions = torch.tensor([[0.25, -0.25], [0.1, 0.0]], device=DEVICE)
    env_step(mdp, state, scene, actions, transition)
    env_step(mdp, state, scene, actions, transition)

    # Four substeps of semi-implicit Euler: v = 4 a dt, p = 10 a dt^2. The second step added 2 a dt and 7 a dt^2,
    # which the change observations report only if their state advanced exactly once per step.
    effort, dt = 2.0 * actions, 0.01
    joint_pos, joint_vel = 10.0 * effort * dt**2, 4.0 * effort * dt
    expected = torch.cat([joint_pos, joint_vel, 7.0 * effort * dt**2, 2.0 * effort * dt], dim=-1)
    torch.testing.assert_close(transition.observations["policy"].torch, expected)
    torch.testing.assert_close(transition.rewards.torch, (1.0 - 10.0 * joint_pos.square().sum(dim=-1)) * 0.02)
    assert not transition.terminated.torch.any() and not transition.truncated.torch.any()
    assert state.episode_length.torch.tolist() == [2, 2]


def test_step_ends_and_bootstraps_worlds():
    """In one step, world 0 continues, world 1 reaches the time limit, and world 2 leaves the bounds."""
    scene = mdp2.PointMassScene(num_worlds=3, device=DEVICE)
    mdp, state = make_task(scene, seconds=0.2)
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
    builder.add_command("target", mdp2.Command(target, resampling_time=(0.02, 0.06)))
    builder.add_reward("target", mdp2.Reward(identity, inputs={"value": mdp2.commands.target}))
    builder.add_observation("policy", "target", mdp2.Observation(identity, inputs={"value": mdp2.commands.target}))
    builder.add_event("push", push_every_two_steps)
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


def test_warp_matches_torch():
    runs = {}
    for backend in ("torch", "warp"):
        scene = mdp2.PointMassScene(num_worlds=64, joint_names=("x", "y", "z"), device=DEVICE)
        mdp, state = make_task(scene, backend=backend, seconds=0.2, random=True)
        transition = mdp.transition()
        env_reset(mdp, state, scene, transition.observations, torch.ones(64, dtype=torch.bool, device=DEVICE))
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        history = []
        for _ in range(30):
            actions = torch.randn(64, 3, device=DEVICE, generator=generator) * 2.0
            actions[:32] *= 0.01  # these worlds stay in bounds until the time limit
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
