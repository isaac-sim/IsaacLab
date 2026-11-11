# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for building and running MDP2 tasks on an analytic point-mass scene."""

import math

import pytest
import torch
import warp as wp
from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2.fusion import Fusion

from isaaclab.utils.warp import ProxyArray

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

robot = mdp2.Entity("robot")


@wp.func
def joint_pos_warp(world: int, joint: int, joint_pos: wp.array2d(dtype=float)) -> float:
    return joint_pos[world, joint]


joint_pos = mdp2.Term(torch=lambda joint_pos: joint_pos, warp=joint_pos_warp, shape=("worlds", "joints"))


@wp.func
def joint_vel_warp(world: int, joint: int, joint_vel: wp.array2d(dtype=float)) -> float:
    return joint_vel[world, joint]


joint_vel = mdp2.Term(torch=lambda joint_vel: joint_vel, warp=joint_vel_warp, shape=("worlds", "joints"))


@wp.func
def joint_pos_subset_warp(
    world: int, joint: int, joint_pos: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)
) -> float:
    return joint_pos[world, joint_ids[joint]]


joint_pos_subset = mdp2.Term(
    torch=lambda joint_pos, joint_ids: joint_pos[:, joint_ids],
    warp=joint_pos_subset_warp,
    shape=("worlds", "joint_ids"),
)


@wp.func
def is_alive_warp(world: int, joint_pos: wp.array2d(dtype=float)) -> float:
    return 1.0


is_alive = mdp2.Term(torch=lambda joint_pos: torch.ones_like(joint_pos[:, 0]), warp=is_alive_warp)


@wp.func
def joint_pos_l2_warp(world: int, joint_pos: wp.array2d(dtype=float)) -> float:
    total = float(0.0)
    for j in range(joint_pos.shape[1]):
        total += joint_pos[world, j] * joint_pos[world, j]
    return total


joint_pos_l2 = mdp2.Term(torch=lambda joint_pos: joint_pos.square().sum(dim=-1), warp=joint_pos_l2_warp)


@wp.func
def joint_pos_out_of_manual_limit_warp(world: int, joint_pos: wp.array2d(dtype=float), limit: float) -> bool:
    outside = bool(False)
    for j in range(joint_pos.shape[1]):
        outside = outside or wp.abs(joint_pos[world, j]) > limit
    return outside


joint_pos_out_of_manual_limit = mdp2.Term(
    torch=lambda joint_pos, *, limit: (joint_pos.abs() > limit).any(dim=-1), warp=joint_pos_out_of_manual_limit_warp
)


@wp.func
def last_action_warp(world: int, action: int, actions: wp.array2d(dtype=float)) -> float:
    return actions[world, action]


last_action = mdp2.Term(torch=lambda actions: actions, warp=last_action_warp, shape=("worlds", "actions"))


@wp.func
def episode_length_warp(world: int, episode_length: wp.array(dtype=wp.int32)) -> float:
    return float(episode_length[world])


episode_length = mdp2.Term(torch=lambda episode_length: episode_length.float(), warp=episode_length_warp)


@wp.func
def generated_commands_warp(world: int, command: wp.array(dtype=float)) -> float:
    return command[world]


generated_commands = mdp2.Term(torch=lambda command: command, warp=generated_commands_warp)

mdp_observations = {
    "last_action": mdp2.Observation(last_action, inputs={"actions": mdp2.last_action}),
    "episode_length": mdp2.Observation(episode_length, inputs={"episode_length": mdp2.episode_length}),
    "joint_pos_subset": mdp2.Observation(
        joint_pos_subset, inputs={"joint_pos": robot.joint_pos, "joint_ids": robot.joints("[xz]")}
    ),
}
"""Observations of the MDP's own fields and of the joints named x and z."""


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


@wp.func
def joint_vel_delta_warp(
    world: int, joint_vel: wp.array2d(dtype=float), last_joint_vel: wp.array(dtype=float)
) -> float:
    return joint_vel[world, 0] - last_joint_vel[world]


@wp.func
def joint_vel_delta_update_warp(world: int, joint_vel: wp.array2d(dtype=float), last_joint_vel: wp.array(dtype=float)):
    last_joint_vel[world] = joint_vel[world, 0]


joint_vel_delta = mdp2.Term(
    torch=mdp2.Implementation(
        compute=lambda joint_vel, last_joint_vel: joint_vel[:, 0] - last_joint_vel,
        update=lambda joint_vel, last_joint_vel: last_joint_vel.copy_(joint_vel[:, 0]),
    ),
    warp=mdp2.Implementation(compute=joint_vel_delta_warp, update=joint_vel_delta_update_warp),
    state={"last_joint_vel": mdp2.State()},
)
"""Change of the first joint's velocity since the last step. Its state restarts at zero."""


@wp.func
def uniform_command_warp(world: int, command: wp.array(dtype=float), rng: wp.int32):
    command[world] = wp.randf(wp.rand_init(rng, 0), -1.0, 1.0)


def uniform_command_torch(mask, command, rng):
    command.copy_(torch.where(mask, mdp2.uniform(rng, -1.0, 1.0), command))


uniform_command = mdp2.Term(torch=uniform_command_torch, warp=uniform_command_warp, random=True)


@wp.func
def copy_command_warp(world: int, command: wp.array(dtype=float), source: wp.array(dtype=float)):
    command[world] = source[world]


copy_command = mdp2.Term(
    torch=lambda mask, command, source: command.copy_(torch.where(mask, source, command)), warp=copy_command_warp
)


@wp.kernel
def _add_uniform_noise_kernel(
    joint_pos: wp.array2d(dtype=float), rng: wp.array(dtype=wp.int32), out: wp.array2d(dtype=float)
):
    world, j = wp.tid()
    out[world, j] = joint_pos[world, j] + wp.randf(wp.rand_init(rng[world], j), -0.1, 0.1)


def noisy_joint_pos_warp(out, joint_pos, rng):
    wp.launch(_add_uniform_noise_kernel, dim=out.shape, inputs=[joint_pos, rng], outputs=[out], device=out.device)


noisy_joint_pos = mdp2.Term(
    torch=lambda joint_pos, rng: joint_pos + mdp2.uniform(rng, -0.1, 0.1, joint_pos.shape[1:]),
    warp=noisy_joint_pos_warp,
    shape=("worlds", "joints"),
    random=True,
)


@wp.func
def push_by_adding_velocity_warp(world: int, joint_vel: wp.array2d(dtype=float)):
    for j in range(joint_vel.shape[1]):
        joint_vel[world, j] += 1.0


push_by_adding_velocity = mdp2.Term(
    torch=lambda mask, joint_vel: joint_vel.add_(mask[:, None].float()), warp=push_by_adding_velocity_warp
)
"""Add one to the joint velocities of the masked worlds."""


@wp.func
def scale_joint_vel_warp(world: int, joint_vel: wp.array2d(dtype=float), factor: float):
    for j in range(joint_vel.shape[1]):
        joint_vel[world, j] *= factor


scale_joint_vel = mdp2.Term(
    torch=lambda mask, joint_vel, *, factor: joint_vel.mul_(torch.where(mask[:, None], factor, 1.0)),
    warp=scale_joint_vel_warp,
)


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


def _reset_joints_by_offset_torch(mask, joint_pos, rng):
    joint_pos.add_(mask[:, None] * mdp2.uniform(rng, -0.5, 0.5, joint_pos.shape[1:]))


reset_joints_by_offset = mdp2.Term(torch=_reset_joints_by_offset_torch, warp=reset_joints_by_offset_warp, random=True)
"""Offset the joint positions of the masked worlds by uniform samples."""

push_robot = mdp2.Event(
    push_by_adding_velocity,
    outputs={"joint_vel": robot.joint_vel},
    mode=mdp2.EventMode.INTERVAL,
    interval_range_s=(0.04, 0.04),
)


def eager_joint_pos_l2_fn(ctx: mdp2.TermContext):
    joint_pos = ctx.inputs["joint_pos"]

    # Reading a value on the host fails inside a CUDA graph, which eager terms never run in.
    if joint_pos.isfinite().all().item():
        ctx.out.copy_(joint_pos.square().sum(dim=-1))


def eager_reset_joints_by_offset_fn(ctx: mdp2.TermContext):
    joint_pos = ctx.outputs["joint_pos"]
    joint_pos.add_(ctx.mask[:, None] * mdp2.uniform(ctx.rng, -0.5, 0.5, joint_pos.shape[1:]))


eager_joint_pos_l2 = mdp2.Term(eager=eager_joint_pos_l2_fn)
eager_reset_joints_by_offset = mdp2.Term(eager=eager_reset_joints_by_offset_fn, random=True)
"""Eager versions of joint_pos_l2 and reset_joints_by_offset."""


def make_task(
    scene: mdp2.PointMassScene,
    backend: str = "torch",
    episode_length_s: float = 1.0,
    random: bool = False,
    eager: bool = False,
) -> tuple[mdp2.MDP, mdp2.MDPState]:
    """Finalize the task of :func:`make_builder` and create its state."""
    mdp = make_builder(scene.schema, episode_length_s, random, eager).finalize(device=DEVICE, backend=backend)
    return mdp, mdp.state(scene)


def make_builder(
    schema: mdp2.SceneSchema,
    episode_length_s: float = 1.0,
    random: bool = False,
    eager: bool = False,
    joint_pos_weight: float = -10.0,
) -> mdp2.MDPBuilder:
    """Return a point-mass task. With ``random``, it also has a command, noisy observations, and events.

    With ``eager``, the joint position penalty and the reset event are eager terms.
    """
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
            "joint_pos_l2": mdp2.Reward(
                eager_joint_pos_l2 if eager else joint_pos_l2,
                inputs={"joint_pos": robot.joint_pos},
                weight=joint_pos_weight,
            ),
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
    builder = mdp2.MDPBuilder(schema)
    builder.add_task(task)
    if not random:
        return builder

    builder.add_command("target", mdp2.Command(uniform_command, resampling_time_range=(0.02, 0.06)))
    builder.add_reward("target", mdp2.Reward(generated_commands, inputs={"command": mdp2.commands.target}))

    # Its state must advance before the push event writes the velocity.
    builder.add_reward("joint_vel_delta", mdp2.Reward(joint_vel_delta, inputs={"joint_vel": robot.joint_vel}))

    for name in ("joint_pos", "joint_pos_again"):
        observation = mdp2.Observation(noisy_joint_pos, inputs={"joint_pos": robot.joint_pos})
        builder.add_observation("noisy", name, observation)
    modified_joint_vel = mdp2.Observation(
        joint_vel,
        inputs={"joint_vel": robot.joint_vel},
        noise=mdp2.UniformNoise(-0.05, 0.05),
        clip=(-1.5, 1.5),
        scale=0.5,
    )
    builder.add_observation("noisy", "joint_vel", modified_joint_vel)
    noisy_history = mdp2.Observation(
        noisy_joint_pos,
        inputs={"joint_pos": robot.joint_pos},
        noise=mdp2.UniformNoise(-0.01, 0.01),
        history_length=3,
    )
    builder.add_observation("noisy", "joint_pos_history", noisy_history)

    target = mdp2.Observation(generated_commands, inputs={"command": mdp2.commands.target})
    builder.add_observation("policy", "target", target)
    for name, observation in mdp_observations.items():
        builder.add_observation("policy", name, observation)

    reset_term = eager_reset_joints_by_offset if eager else reset_joints_by_offset
    builder.add_event("push_robot", push_robot)
    builder.add_event("reset_robot_joints", mdp2.Event(reset_term, outputs={"joint_pos": robot.joint_pos}))
    return builder


def env_step(mdp, state, scene, actions, transition) -> dict[str, torch.Tensor]:
    """Step in the phase order of ``MDP2RLEnv.step``, with the analytic scene standing in for the simulation.

    Returns the bootstrap observations: the successor observations before ended worlds are reset.
    """
    mdp2.apply_actions(mdp, state, ProxyArray(wp.from_torch(actions)))
    scene.step()
    mdp2.evaluate(mdp, state, transition)
    bootstrap = {group: values.torch.clone() for group, values in transition.observations.items()}

    scene.reset(transition.done.torch)
    mdp2.reset(mdp, state, transition.done)
    mdp2.observe(mdp, state, transition.observations, transition.done)
    return bootstrap


def env_reset(mdp, state, scene, observations, mask):
    """Reset in the phase order of ``MDP2RLEnv.reset``."""
    scene.reset(mask)
    mdp2.reset(mdp, state, ProxyArray(wp.from_torch(mask)))
    mdp2.observe(mdp, state, observations, ProxyArray(wp.from_torch(mask)))


def test_finalize_reports_all_errors():
    """Finalizing reports every invalid use at once, each prefixed with its name."""
    builder = mdp2.MDPBuilder(mdp2.PointMassScene(num_worlds=1).schema)
    builder.add_reward("unknown_entity", mdp2.Reward(is_alive, inputs={"joint_pos": mdp2.Entity("table").joint_pos}))
    builder.add_reward("unknown_field", mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_angle}))
    builder.add_reward("unknown_command", mdp2.Reward(generated_commands, inputs={"command": mdp2.commands.typo}))
    builder.add_reward("unmatched_subset", mdp2.Reward(is_alive, inputs={"joint_pos": robot.joints("x", "elbow")}))
    for name in ("joint_effort", "second_writer"):
        builder.add_action(name, mdp2.DirectAction(robot.joint_effort_target))
    builder.add_reward("array_type", mdp2.Reward(generated_commands, inputs={"command": robot.joint_pos}))
    builder.add_termination("return_type", mdp2.Termination(is_alive, inputs={"joint_pos": robot.joint_pos}))
    builder.add_reward("no_warp", mdp2.Reward(mdp2.Term(), inputs={"joint_pos": robot.joint_pos}))
    name_clash = mdp2.Observation(delta, inputs={"value": robot.joint_pos, "last_value": robot.joint_pos})
    builder.add_observation("policy", "name_clash", name_clash)
    no_interval = mdp2.Event(
        push_by_adding_velocity, outputs={"joint_vel": robot.joint_vel}, mode=mdp2.EventMode.INTERVAL
    )
    builder.add_event("no_interval", no_interval)
    builder.add_reward("func_arguments", mdp2.Reward(joint_pos_l2, inputs={"pos": robot.joint_pos}))
    builder.add_event("function_arguments", mdp2.Event(reset_joints_by_offset, outputs={"pos": robot.joint_pos}))

    with pytest.raises(ValueError) as error:
        builder.finalize(backend="warp")

    message = str(error.value)
    assert "rewards.unknown_entity: unknown entity 'table' in 'table.joint_pos'" in message
    assert "rewards.unknown_field: unknown field 'robot.joint_angle'" in message
    assert "rewards.unknown_command: unknown field 'commands.typo'. Fields of the MDP:" in message
    assert "rewards.unmatched_subset: 'robot.joints('x', 'elbow')' matches no name with ['elbow']" in message
    assert "actions.second_writer: 'robot.joint_effort_target' is already written by 'actions.joint_effort'" in message
    assert "rewards.array_type: wp.func 'generated_commands_warp' takes 'command' as a 1-dimensional array" in message
    assert "terminations.return_type: wp.func 'is_alive_warp' must return bool" in message
    assert "rewards.no_warp: the term has no 'warp' implementation" in message
    assert "observations.policy.name_clash: inputs, outputs, state, params, and rng must have distinct names" in message
    assert "events.no_interval: interval events need an interval_range_s" in message
    assert (
        "rewards.func_arguments: wp.func 'joint_pos_l2_warp' must take the world, 0 output indices, and any of"
        " ['pos']" in message
    )
    assert (
        "events.function_arguments: 'reset_joints_by_offset_warp' is called with ['mask', 'rng'] of ['pos', 'rng']:"
        " missing a required argument: 'joint_pos'" in message
    )


def test_schedule_edits_keep_phases_and_dependencies():
    """Edits that would break a phase or a read-after-write order fail and leave the schedule unchanged."""
    builder = mdp2.MDPBuilder(mdp2.PointMassScene(num_worlds=1).schema)
    target = builder.add_command("target", mdp2.Command(uniform_command))
    target_copy = builder.add_command(
        "target_copy", mdp2.Command(copy_command, inputs={"source": mdp2.commands.target})
    )
    alive = builder.add_reward("alive", mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_pos}))
    schedule = builder.schedule()

    with pytest.raises(ValueError, match="'commands.target_copy' reads 'commands.target', so it must run after"):
        builder.move_before(target_copy, target)
    with pytest.raises(ValueError, match="a use stays in its phase"):
        builder.move_after(alive, target)
    with pytest.raises(ValueError, match="must list each of its entries once"):
        builder.set_order(mdp2.Phase.COMMANDS, [target_copy])

    assert builder.schedule() == schedule
    assert builder.describe() == "\n".join(
        [
            "rewards",
            "  rewards.alive         reads robot.joint_pos",
            "commands",
            "  commands.target       writes commands.target",
            "  commands.target_copy  reads commands.target; writes commands.target_copy",
        ]
    )

    # Added again, the command runs after its reader.
    builder.remove(target)
    builder.add_command("target", mdp2.Command(uniform_command))

    with pytest.raises(ValueError, match="'commands.target_copy' reads 'commands.target', so it must run after"):
        builder.validate()


def test_schedule_edits_change_execution_order():
    """Pushing then doubling the velocity differs from doubling then pushing. Finalized MDPs keep their order."""
    builder = mdp2.MDPBuilder(mdp2.PointMassScene(num_worlds=2).schema)
    push = builder.add_event("push_robot", push_robot)
    double_joint_vel = mdp2.Event(
        scale_joint_vel,
        outputs={"joint_vel": robot.joint_vel},
        params={"factor": 2.0},
        mode=mdp2.EventMode.INTERVAL,
        interval_range_s=(0.04, 0.04),
    )
    double = builder.add_event("double_joint_vel", double_joint_vel)

    for backend in ("torch", "warp"):
        builder.set_order(mdp2.Phase.INTERVAL_EVENTS, [push, double])
        push_first = builder.finalize(device=DEVICE, backend=backend)
        builder.move_before(double, push)
        double_first = builder.finalize(device=DEVICE, backend=backend)

        assert push_first.schedule == ("events.push_robot", "events.double_joint_vel")
        assert double_first.schedule == ("events.double_joint_vel", "events.push_robot")

        for mdp, expected in ((push_first, 2.0), (double_first, 1.0)):
            scene = mdp2.PointMassScene(num_worlds=2, device=DEVICE)
            state, transition = mdp.state(scene), mdp.transition()
            env_reset(mdp, state, scene, transition.observations, torch.ones(2, dtype=torch.bool, device=DEVICE))

            # The events run on the second step.
            for _ in range(2):
                env_step(mdp, state, scene, torch.zeros(2, 1, device=DEVICE), transition)

            assert torch.equal(scene.data("robot", "joint_vel").torch, torch.full((2, 1), expected, device=DEVICE))


def test_step_matches_point_mass_dynamics():
    """Two steps give the analytic positions, velocities, deltas, and rewards of the point mass."""
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
    assert torch.equal(observations[0], bootstrap[0])  # Continuing worlds keep their successor sample.
    assert torch.equal(bootstrap[1], bootstrap[0])  # Truncated worlds bootstrap from the pre-reset sample.
    assert bootstrap[0].abs().sum() > 0.0 and bootstrap[2].isfinite().all()
    assert observations[1:].abs().sum() == 0.0  # Ended worlds observe their new episode.


def test_reset_touches_only_selected_worlds():
    """A reset of no worlds changes nothing, and a reset of world 0 changes only world 0."""
    scene = mdp2.PointMassScene(num_worlds=3, device=DEVICE)
    mdp, state = make_task(scene)
    observations = mdp.observations()
    env_reset(mdp, state, scene, observations, torch.ones(3, dtype=torch.bool, device=DEVICE))
    for _ in range(2):
        env_step(mdp, state, scene, torch.full((3, 1), 0.01, device=DEVICE), mdp.transition())
    mdp2.observe(mdp, state, observations, ProxyArray(wp.ones(3, dtype=wp.bool, device=DEVICE)))

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


def test_state_resets_after_reset_events():
    """The delta observation restarts from the joint positions that the reset event offset."""
    for backend in ("torch", "warp"):
        scene = mdp2.PointMassScene(num_worlds=2, device=DEVICE)
        mdp, state = make_task(scene, backend=backend, random=True)
        observations = mdp.observations()

        env_reset(mdp, state, scene, observations, torch.ones(2, dtype=torch.bool, device=DEVICE))

        joint_pos, joint_pos_delta = observations["policy"].torch[:, 0], observations["policy"].torch[:, 2]
        assert (joint_pos != 0.0).all() and (joint_pos_delta == 0.0).all()


def test_mdp_fields_subsets_weights_and_diagnostics():
    """MDP field inputs, joint subsets, mutable reward weights, and per-term diagnostics."""
    for backend in ("torch", "warp"):
        scene = mdp2.PointMassScene(num_worlds=2, joint_names=("x", "y", "z"), device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)

        # The actions drive the joints in the order z, x, y.
        builder.add_action("effort_z", mdp2.DirectAction(robot.joint_effort_target, subset=robot.joints("z")))
        builder.add_action("effort_xy", mdp2.DirectAction(robot.joint_effort_target, subset=robot.joints("[xy]")))
        for name, observation in mdp_observations.items():
            builder.add_observation("policy", name, observation)
        builder.add_reward("alive", mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_pos}))
        builder.add_reward("joint_pos_l2", mdp2.Reward(joint_pos_l2, inputs={"joint_pos": robot.joint_pos}))
        limit = mdp2.Termination(joint_pos_out_of_manual_limit, inputs={"joint_pos": robot.joint_pos})
        limit.params["limit"] = 0.01
        builder.add_termination("out_of_bounds", limit)

        mdp = builder.finalize(device=DEVICE, backend=backend, diagnostics=True)
        limit.params["limit"] = 1.0e6  # A finalized MDP keeps its params.
        state, transition = mdp.state(scene), mdp.transition()
        env_reset(mdp, state, scene, transition.observations, torch.ones(2, dtype=torch.bool, device=DEVICE))

        actions = torch.tensor([[0.1, 0.2, 0.3], [100.0, 0.0, 0.0]], device=DEVICE)
        mdp2.apply_actions(mdp, state, ProxyArray(wp.from_torch(actions)))
        scene.step()
        mdp2.evaluate(mdp, state, transition)

        joint_pos = scene.data("robot", "joint_pos").torch
        assert torch.equal(scene.data("robot", "joint_effort_target").torch, actions[:, [1, 2, 0]])
        expected = torch.cat([actions, torch.ones(2, 1, device=DEVICE), joint_pos[:, [0, 2]]], dim=-1)
        torch.testing.assert_close(transition.observations["policy"].torch, expected)
        assert transition.terminated.torch.tolist() == transition.done.torch.tolist() == [False, True]
        assert torch.equal(transition.termination_terms.torch[:, 0], transition.terminated.torch)
        torch.testing.assert_close(transition.reward_terms.torch.sum(dim=-1), transition.rewards.torch)

        # Weights are state, and may change while the MDP runs.
        state.reward_weights.torch[1] = 0.0
        mdp2.evaluate(mdp, state, transition)

        torch.testing.assert_close(transition.rewards.torch, torch.full((2,), mdp.step_dt, device=DEVICE))


def test_commands_and_interval_events_follow_their_timers():
    """Rewards see the command that governed the step, and observations see the resampled one."""
    scene = mdp2.PointMassScene(num_worlds=8, device=DEVICE)
    builder = mdp2.MDPBuilder(scene.schema)
    builder.add_command("target", mdp2.Command(uniform_command, resampling_time_range=(0.02, 0.06)))
    builder.add_reward("target", mdp2.Reward(generated_commands, inputs={"command": mdp2.commands.target}))
    target = mdp2.Observation(generated_commands, inputs={"command": mdp2.commands.target})
    builder.add_observation("policy", "target", target)
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

    # Commands resample every one to three steps, and the push event fires every other step.
    assert (resamples >= 2).all() and (resamples < 6).any()
    assert torch.equal(scene.data("robot", "joint_vel").torch, torch.full((8, 1), 3.0, device=DEVICE))


def test_random_streams_are_independent_per_world_and_use():
    """Each world and each use of a random term draws from its own stream, once per step."""

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

    assert (samples[..., :2] != samples[..., 2:4]).all()  # Two uses of one term.
    assert (samples[..., 4:].abs() <= 0.75).all() and (samples[1:, :, 4:] != samples[:-1, :, 4:]).any()
    assert (samples[1:, ..., :4] != samples[:-1, ..., :4]).all()  # Every step.

    # Resetting world 0, or resets and observations of no worlds, leave world 1's draws unchanged.
    assert torch.equal(noisy_observations(reset_world_0=True, env=True)[:, 1], samples[:, 1])
    assert torch.equal(noisy_observations(reset_world_0=False, env=False)[:, 1], samples[:, 1])


def test_fusion_cuts_kernels_where_threads_would_race():
    """A world statement reading elements that other threads write starts a new kernel."""
    code = Fusion("racing", 2, DEVICE)
    values = code.argument("values", wp.array2d(dtype=wp.float32), lambda call: call)
    code.elements((3,), f"{values}[world, element] = 1.0", writes=["values"])
    code.world(f"{values}[world, 0] = {values}[world, 1] + {values}[world, 2]", writes=["values"])
    launches = code.steps()

    array = wp.zeros((2, 3), dtype=wp.float32, device=DEVICE)
    for launch in launches:
        launch(array)

    assert len(launches) == 2 and array.numpy()[:, 0].tolist() == [2.0, 2.0]


def test_warp_fuses_functions_into_few_kernels():
    """Each MDP function of the Warp backend runs in one or two kernels."""
    mdp, _ = make_task(mdp2.PointMassScene(num_worlds=2, device=DEVICE), backend="warp")

    # Observations share the kernel of the terminations and rewards. The state updates wait for them in a second one.
    launches = mdp.launches
    num_kernels = [len(function) for function in (launches.apply_actions, launches.evaluate, launches.observe)]
    assert num_kernels == [1, 2, 1]


def test_warp_matches_torch():
    """The Warp and Torch backends agree over episodes with resets, and draw bitwise the same random numbers."""
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
            actions[:32] *= 0.01  # These worlds stay in bounds until they time out.
            bootstrap = env_step(mdp, state, scene, actions, transition)

            buffers = (transition.rewards, transition.terminated, transition.truncated, state.episode_length)
            results = [buffer.torch.clone() for buffer in buffers]
            results += [transition.observations["policy"].torch.clone(), bootstrap["policy"]]
            draws = [state.rng.torch.clone(), transition.observations["noisy"].torch.clone()]
            history.append((results, draws))
        runs[backend] = history

    for (torch_results, torch_draws), (warp_results, warp_draws) in zip(runs["torch"], runs["warp"]):
        for expected, actual in zip(torch_results, warp_results):
            torch.testing.assert_close(actual, expected)
        for expected, actual in zip(torch_draws, warp_draws):
            assert torch.equal(actual, expected)

    # Both ways an episode ends happened.
    assert any(results[1].any() for results, _ in runs["torch"])
    assert any(results[2].any() for results, _ in runs["torch"])


def test_eager_terms_split_segments():
    """An eager term splits its function into capturable and eager segments."""
    builder = mdp2.MDPBuilder(mdp2.PointMassScene(num_worlds=1).schema)
    builder.add_reward("alive", mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_pos}))
    eager = builder.add_reward("eager_l2", mdp2.Reward(eager_joint_pos_l2, inputs={"joint_pos": robot.joint_pos}))
    l2 = builder.add_reward("joint_pos_l2", mdp2.Reward(joint_pos_l2, inputs={"joint_pos": robot.joint_pos}))

    def evaluate_segments():
        segments = builder.preview_segments()["evaluate"]
        return [(segment.capturable, [step.op.name for step in segment.steps]) for segment in segments]

    assert evaluate_segments() == [
        (True, ["rewards.alive"]),
        (False, ["rewards.eager_l2"]),
        (True, ["rewards.joint_pos_l2"]),
    ]

    builder.move_after(eager, l2)

    assert evaluate_segments() == [(True, ["rewards.alive", "rewards.joint_pos_l2"]), (False, ["rewards.eager_l2"])]
    assert not builder.finalize().is_fully_capturable

    builder.remove(eager)

    assert builder.finalize().is_fully_capturable


def test_eager_terms_match_compiled_terms():
    """Eager terms see the same data and draw from the same streams as compiled terms, for their worlds only."""
    for backend in ("torch", "warp"):
        runs = []
        for eager in (False, True):
            scene = mdp2.PointMassScene(num_worlds=4, joint_names=("x", "y"), device=DEVICE)
            mdp, state = make_task(scene, backend=backend, random=True, eager=eager)
            transition = mdp.transition()

            mask = torch.tensor([True, True, False, True], device=DEVICE)
            env_reset(mdp, state, scene, transition.observations, mask)
            joint_pos = scene.data("robot", "joint_pos").torch.clone()
            env_step(mdp, state, scene, torch.full((4, mdp.action_dim), 0.1, device=DEVICE), transition)
            runs.append((joint_pos, transition.rewards.torch.clone()))

        (compiled_pos, compiled_rewards), (eager_pos, eager_rewards) = runs
        assert torch.equal(eager_pos, compiled_pos) and (eager_pos[2] == 0.0).all()
        torch.testing.assert_close(eager_rewards, compiled_rewards)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_captured_mdp_replays_like_eager_calls():
    """Graph replay, with eager terms between the graphs, gives bitwise the same results as eager calls."""
    for backend in ("torch", "warp"):
        runs = []
        for capture in (False, True):
            scene = mdp2.PointMassScene(num_worlds=64, joint_names=("x", "y", "z"), device=DEVICE)
            mdp, state = make_task(scene, backend=backend, episode_length_s=0.2, random=True, eager=True)
            transition = mdp.transition()
            actions = ProxyArray(wp.zeros((64, mdp.action_dim), dtype=wp.float32, device=DEVICE))
            captured = mdp2.CapturedMDP(mdp, state, transition, actions, transition.done) if capture else None
            env_reset(mdp, state, scene, transition.observations, torch.ones(64, dtype=torch.bool, device=DEVICE))

            generator = torch.Generator(device=DEVICE).manual_seed(0)
            history = []
            for _ in range(20):
                actions.torch.copy_(torch.randn(64, 3, device=DEVICE, generator=generator))
                if captured is None:
                    env_step(mdp, state, scene, actions.torch, transition)
                else:
                    captured.apply_actions()
                    scene.step()
                    captured.evaluate()
                    scene.reset(transition.done.torch)
                    captured.reset()
                    captured.observe()
                step = torch.cat([transition.rewards.torch[:, None], transition.observations["policy"].torch], 1)
                history.append(step)
            runs.append(torch.stack(history))

        # Warp runs hooks ahead of the kernel statements they do not depend on, which can merge graphs.
        num_segments = sum(segment.capturable for function in mdp.segments.values() for segment in function)
        assert captured.num_graphs == num_segments if backend == "torch" else captured.num_graphs < num_segments
        assert torch.equal(runs[1], runs[0])


def test_world_configs_report_errors():
    """Replicated builders report extra uses, uncovered worlds, and differing time limits."""
    schema = mdp2.PointMassScene(num_worlds=4).schema
    builder = mdp2.MDPBuilder(schema)
    builder.replicate(make_builder(schema, episode_length_s=1.0), 2)
    builder.replicate(make_builder(schema, episode_length_s=2.0), 1)
    builder.add_reward("alive", mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_pos}))

    with pytest.raises(ValueError) as error:
        builder.validate()

    message = str(error.value)
    assert "Add term uses to the replicated builders" in message
    assert "The world configs cover 3 of 4 worlds" in message
    assert "The world configs must share one time limit" in message


def test_world_configs_match_single_config_runs():
    """Each world of a mixed MDP gives bitwise the results of its config's MDP run alone, with zero padding."""
    num_worlds, split = 6, 3
    for backend in ("torch", "warp"):
        schema = mdp2.PointMassScene(num_worlds=num_worlds, joint_names=("x", "y")).schema
        configs = [
            make_builder(schema, episode_length_s=0.2, random=True),
            make_builder(schema, episode_length_s=0.2, joint_pos_weight=-1.0),
        ]
        mixed = mdp2.MDPBuilder(schema)
        mixed.replicate(configs[0], split)
        mixed.replicate(configs[1], num_worlds - split)

        runs = []
        for builder in (*configs, mixed):
            scene = mdp2.PointMassScene(num_worlds=num_worlds, joint_names=("x", "y"), device=DEVICE)
            mdp = builder.finalize(device=DEVICE, backend=backend)
            state, transition = mdp.state(scene), mdp.transition()
            env_reset(
                mdp, state, scene, transition.observations, torch.ones(num_worlds, dtype=torch.bool, device=DEVICE)
            )

            generator = torch.Generator(device=DEVICE).manual_seed(0)
            history = []
            for _ in range(15):
                actions = torch.randn(num_worlds, 2, device=DEVICE, generator=generator) * 0.5
                env_step(mdp, state, scene, actions, transition)

                results = torch.stack(
                    [transition.rewards.torch, transition.terminated.torch, transition.truncated.torch], 1
                )
                observations = {group: values.torch.clone() for group, values in transition.observations.items()}
                history.append({"results": results, **observations})
            runs.append((mdp, history))

        (first, first_run), (second, second_run), (mixed_mdp, mixed_run) = runs
        names = {op.name: op.worlds for ops in mixed_mdp.ops.values() for op in ops}
        assert names["rewards.alive"] is None  # A use both configs share runs once, in every world.
        assert names["rewards.joint_pos_l2[0]"] == (0, 1, 2) and names["rewards.joint_pos_l2[1]"] == (3, 4, 5)

        observation_dims = [config.observation_dims["policy"] for config in mixed_mdp.world_configs]
        assert observation_dims == [first.observation_dims["policy"], second.observation_dims["policy"]]

        for run, worlds in ((first_run, slice(0, split)), (second_run, slice(split, None))):
            for alone, mixed_step in zip(run, mixed_run):
                for key, values in alone.items():
                    assert torch.equal(mixed_step[key][worlds, : values.shape[1]], values[worlds])

        width = second.observation_dims["policy"]
        for mixed_step in mixed_run:
            assert (mixed_step["policy"][split:, width:] == 0.0).all() and (mixed_step["noisy"][split:] == 0.0).all()


def test_library_action_and_termination_terms():
    """The library's action and termination terms, which read MDP fields rather than the scene."""
    for backend in ("torch", "warp"):
        scene = mdp2.PointMassScene(num_worlds=2, joint_names=("x", "y"), device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_action("joint_effort", mdp2.DirectAction(robot.joint_effort_target))

        actions = {"actions": mdp2.last_action}
        builder.add_observation("policy", "last_action", mdp2.Observation(mdp2.mdp.last_action, inputs=actions))
        builder.add_observation("policy", "action_l2", mdp2.Observation(mdp2.mdp.action_l2, inputs=actions))
        builder.add_observation("policy", "action_rate_l2", mdp2.Observation(mdp2.mdp.action_rate_l2, inputs=actions))
        terminated = {"terminated": mdp2.terminated}
        builder.add_observation("policy", "is_alive", mdp2.Observation(mdp2.mdp.is_alive, inputs=terminated))
        builder.add_observation("policy", "is_terminated", mdp2.Observation(mdp2.mdp.is_terminated, inputs=terminated))

        out_of_bounds = mdp2.Termination(
            joint_pos_out_of_manual_limit, inputs={"joint_pos": robot.joint_pos}, params={"limit": 0.1}
        )
        builder.add_termination("out_of_bounds", out_of_bounds)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()

        steps = [
            torch.tensor([[1.0, 2.0], [0.0, 0.0]], device=DEVICE),
            torch.tensor([[0.0, 2.0], [500.0, 0.0]], device=DEVICE),
        ]
        for step_actions in steps:
            mdp2.apply_actions(mdp, state, ProxyArray(wp.from_torch(step_actions)))
            scene.step()
            mdp2.evaluate(mdp, state, transition)

        policy = transition.observations["policy"].torch
        torch.testing.assert_close(policy[:, :2], steps[1])
        torch.testing.assert_close(policy[:, 2], steps[1].square().sum(-1))
        torch.testing.assert_close(policy[:, 3], (steps[1] - steps[0]).square().sum(-1))
        assert policy[:, 4].tolist() == [1.0, 0.0] and policy[:, 5].tolist() == [0.0, 1.0]


def test_observation_history():
    """The group holds the last values oldest first, and a new episode starts with its first value repeated."""
    for backend in ("torch", "warp"):
        scene = mdp2.PointMassScene(num_worlds=2, device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_action("joint_effort", mdp2.DirectAction(robot.joint_effort_target))
        history = mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos}, history_length=3)
        builder.add_observation("policy", "joint_pos", history)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()

        assert mdp.observation_dims["policy"] == 3

        scene.data("robot", "joint_pos").torch.fill_(0.5)
        env_reset(mdp, state, scene, transition.observations, torch.tensor([False, True], device=DEVICE))

        # The reset world starts its episode from the scene's reset positions.
        assert transition.observations["policy"].torch[1].tolist() == [0.0, 0.0, 0.0]

        positions = []
        for _ in range(2):
            env_step(mdp, state, scene, torch.ones(2, 1, device=DEVICE), transition)
            positions.append(scene.data("robot", "joint_pos").torch[:, 0].clone())

        expected = torch.stack([torch.tensor([0.0, 0.0], device=DEVICE), *positions], dim=1)
        expected[0, 0] = 0.0  # World 0 was never reset, so its history started at zero.
        torch.testing.assert_close(transition.observations["policy"].torch, expected)


@wp.func
def count_samples_warp(world: int, command: wp.array(dtype=float), samples: wp.array(dtype=float)):
    samples[world] = samples[world] + 1.0
    command[world] = samples[world]


def count_samples_torch(mask, command, samples):
    samples.add_(mask.float())
    command.copy_(torch.where(mask, samples, command))


sample_count = mdp2.Term(torch=count_samples_torch, warp=count_samples_warp, state={"samples": mdp2.State()})
"""A command whose value counts its samples in its state."""


def test_command_resamples_keep_their_state():
    """A command resamples on episode resets and when its timer expires, and its state outlives episodes."""
    results = {}
    for backend in ("torch", "warp"):
        scene = mdp2.PointMassScene(num_worlds=4, device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_command("count", mdp2.Command(sample_count, resampling_time_range=(0.03, 0.03)))
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()

        mdp2.reset(mdp, state, ProxyArray(wp.ones(4, dtype=wp.bool, device=DEVICE)))
        for _ in range(4):
            mdp2.evaluate(mdp, state, transition)
        mdp2.reset(mdp, state, ProxyArray(wp.ones(4, dtype=wp.bool, device=DEVICE)))
        results[backend] = state.fields[mdp2.commands.count].torch.clone()

    # The reset samples once, then the 0.03 s timer expires every second 0.02 s step, then the reset samples again.
    assert torch.equal(results["torch"], torch.full((4,), 4.0, device=DEVICE))
    assert torch.equal(results["warp"], results["torch"])


def test_library_commands():
    """Commands stay in their ranges, agree across backends, and generated_commands observes them."""
    pose_range = ((0.4, 0.6), (-0.2, 0.2), (0.1, 0.3), (-0.5, 0.5), (-0.5, 0.5), (-math.pi, math.pi))
    velocity_range = {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.3, 0.3)}
    commands = {
        "base_velocity": mdp2.Command(
            mdp2.mdp.uniform_velocity_command, params={**velocity_range, "rel_standing_envs": 0.25}
        ),
        "object_pose": mdp2.Command(
            mdp2.mdp.uniform_pose_command, params={"pose_range": pose_range, "make_quat_unique": True}
        ),
    }

    results = {}
    for backend in ("torch", "warp"):
        scene = mdp2.PointMassScene(num_worlds=256, device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        for name, command in commands.items():
            builder.add_command(name, command)
            command_input = {"command": mdp2.FieldSelector("commands", name, scene=False)}
            builder.add_observation("policy", name, mdp2.Observation(mdp2.mdp.generated_commands, inputs=command_input))
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, observations = mdp.state(scene), mdp.observations()

        env_reset(mdp, state, scene, observations, torch.ones(256, dtype=torch.bool, device=DEVICE))
        results[backend] = observations["policy"].torch.clone()

    assert torch.equal(results["warp"][:, :3], results["torch"][:, :3])
    torch.testing.assert_close(results["warp"], results["torch"])

    velocity, pose = results["warp"][:, :3], results["warp"][:, 3:]
    standing = (velocity == 0.0).all(-1)
    assert 0.1 < standing.float().mean() < 0.4
    assert (
        (velocity[:, 0].abs() <= 1.0).all()
        and (velocity[:, 1].abs() <= 0.5).all()
        and (velocity[:, 2].abs() <= 0.3).all()
    )
    for axis, (low, high) in enumerate(pose_range[:3]):
        assert ((pose[:, axis] >= low) & (pose[:, axis] <= high)).all()
    torch.testing.assert_close(pose[:, 3:].norm(dim=-1), torch.ones(256, device=DEVICE))
    assert (pose[:, 6] >= 0.0).all()
