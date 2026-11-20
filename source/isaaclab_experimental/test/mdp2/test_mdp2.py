# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for building and running MDP2 tasks on an analytic point-mass scene."""

import dataclasses
import math
from collections.abc import Collection

import pytest
import torch
import warp as wp
from isaaclab_experimental import mdp2
from isaaclab_experimental.mdp2.fusion import Fusion

from isaaclab.utils.warp import ProxyArray

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.kitless

robot = mdp2.Entity("robot")


class PointMassScene:
    """Scene for the tests: one ``"robot"``, or other entities, whose joints are unit point masses driven by their
    effort targets."""

    def __init__(
        self,
        num_worlds: int,
        joint_names: tuple[str, ...] = ("x",),
        dt: float = 0.01,
        decimation: int = 2,
        device: str = "cpu",
        entities: tuple[str, ...] = ("robot",),
    ):
        fields = dict.fromkeys(
            ("joint_pos", "joint_vel", "joint_effort_target"), mdp2.FieldSchema(("joints",), writable=True)
        )
        # Derived by the simulation, as joint accelerations are: read-only. The tip is computed from the joints, as body
        # poses are, so reading it after joint writes needs a commit.
        fields["joint_acc"] = mdp2.FieldSchema(("joints",))
        fields["tip"] = mdp2.FieldSchema(("joints",), derived_from=("joint_pos",))

        self.schema = mdp2.SceneSchema(
            num_worlds=num_worlds,
            step_dt=dt * decimation,
            entities={entity: mdp2.EntitySchema(names={"joints": joint_names}, fields=fields) for entity in entities},
        )
        self.dt = dt
        self.decimation = decimation

        shape = (num_worlds, len(joint_names))
        self._data = {
            entity: {name: ProxyArray(wp.zeros(shape, dtype=wp.float32, device=device)) for name in fields}
            for entity in entities
        }

    def data(self, entity: str, field: str) -> ProxyArray:
        return self._data[entity][field]

    def entity(self, name: str) -> "PointMassScene":
        """The scene object of an entity: the scene itself, whose data it reads."""
        return self

    def commit(self, mask: wp.array, fields: Collection[mdp2.FieldSelector]) -> None:
        """Nothing to commit: the fields are the simulation state."""

    def step(self) -> None:
        """Advance one control step: ``decimation`` substeps of semi-implicit Euler."""
        robot = {name: value.torch for name, value in self._data["robot"].items()}
        for _ in range(self.decimation):
            robot["joint_vel"].add_(robot["joint_effort_target"], alpha=self.dt)
            robot["joint_pos"].add_(robot["joint_vel"], alpha=self.dt)

    def reset(self, mask: torch.Tensor) -> None:
        """Zero every field of the worlds selected by a boolean mask."""
        for value in self._data["robot"].values():
            value.torch.masked_fill_(mask[:, None], 0.0)


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


def host_joint_pos_l2_fn(joint_pos):
    # Reading a value on the host fails inside a CUDA graph, which hooks that are not capturable never run in.
    if joint_pos.isfinite().all().item():
        return joint_pos.square().sum(dim=-1)
    return torch.zeros_like(joint_pos[:, 0])


def host_reset_joints_by_offset_fn(mask, joint_pos, rng):
    joint_pos.add_(mask[:, None] * mdp2.uniform(rng, -0.5, 0.5, joint_pos.shape[1:]))


host_joint_pos_l2 = mdp2.Term(torch=host_joint_pos_l2_fn, capturable=False)
host_reset_joints_by_offset = mdp2.Term(torch=host_reset_joints_by_offset_fn, random=True, capturable=False)
"""Versions of joint_pos_l2 and reset_joints_by_offset that run on the host, between CUDA graphs."""


def make_task(
    scene: PointMassScene,
    backend: str = "torch",
    episode_length_s: float = 1.0,
    random: bool = False,
    host: bool = False,
    torch_hooks: bool = False,
) -> tuple[mdp2.MDP, mdp2.MDPState]:
    """Finalize the task of :func:`make_builder` and create its state."""
    builder = make_builder(scene.schema, episode_length_s, random, host, torch_hooks=torch_hooks)
    mdp = builder.finalize(device=DEVICE, backend=backend)
    return mdp, mdp.state(scene)


def make_builder(
    schema: mdp2.SceneSchema,
    episode_length_s: float = 1.0,
    random: bool = False,
    host: bool = False,
    joint_pos_weight: float = -10.0,
    torch_hooks: bool = False,
) -> mdp2.MDPBuilder:
    """Return a point-mass task. With ``random``, it also has a command, noisy observations, and events.

    With ``host``, the joint position penalty and the reset event run on the host, between CUDA graphs. With
    ``torch_hooks``, terms with
    state, random draws, or masks have Torch hooks only, which a Warp MDP runs between its kernels.
    """

    def torch_only(term: mdp2.Term) -> mdp2.Term:
        return dataclasses.replace(term, warp=None) if torch_hooks else term

    task = mdp2.TaskCfg(
        actions={"joint_effort": mdp2.DirectAction(robot.joint_effort_target, scale=2.0)},
        observations={
            "policy": {
                "joint_pos": mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos}),
                "joint_vel": mdp2.Observation(joint_vel, inputs={"joint_vel": robot.joint_vel}),
                "joint_pos_delta": mdp2.Observation(torch_only(delta), inputs={"value": robot.joint_pos}),
                "joint_vel_delta": mdp2.Observation(delta, inputs={"value": robot.joint_vel}),
            }
        },
        rewards={
            "alive": mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_pos}),
            "joint_pos_l2": mdp2.Reward(
                host_joint_pos_l2 if host else joint_pos_l2,
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

    builder.add_command("target", mdp2.Command(torch_only(uniform_command), resampling_time_range=(0.02, 0.06)))
    builder.add_reward("target", mdp2.Reward(generated_commands, inputs={"command": mdp2.commands.target}))

    # Its state must advance before the push event writes the velocity.
    builder.add_reward(
        "joint_vel_delta", mdp2.Reward(torch_only(joint_vel_delta), inputs={"joint_vel": robot.joint_vel})
    )

    for name in ("joint_pos", "joint_pos_again"):
        observation = mdp2.Observation(torch_only(noisy_joint_pos), inputs={"joint_pos": robot.joint_pos})
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

    reset_term = host_reset_joints_by_offset if host else torch_only(reset_joints_by_offset)
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
    builder = mdp2.MDPBuilder(PointMassScene(num_worlds=1).schema)
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
    builder.add_event("read_only", mdp2.Event(push_by_adding_velocity, outputs={"joint_vel": robot.joint_acc}))

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
        "events.function_arguments: 'reset_joints_by_offset_warp' is called with ['mask', 'rng'] of"
        " ['cache', 'pos', 'rng']: missing a required argument: 'joint_pos'" in message
    )
    assert "events.read_only: 'robot.joint_acc' is read-only" in message


def test_schedule_edits_keep_phases_and_dependencies():
    """Edits that would break a phase or a read-after-write order fail and leave the schedule unchanged."""
    builder = mdp2.MDPBuilder(PointMassScene(num_worlds=1).schema)
    target = builder.add_command("target", mdp2.Command(uniform_command))
    target_copy = builder.add_command(
        "target_copy", mdp2.Command(copy_command, inputs={"source": mdp2.commands.target})
    )
    alive = builder.add_reward("alive", mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_pos}))
    # A use that reads another's state runs after it too.
    total = mdp2.Quantity(accumulate, inputs={"joint_pos": robot.joint_pos}, params={"gain": 1.0})
    total = builder.add_quantity("sum", total)
    total_copy = mdp2.Quantity(mdp2.mdp.value, inputs={"value": mdp2.term_state("quantities.sum", "total")})
    total_copy = builder.add_quantity("sum_copy", total_copy)
    schedule = builder.schedule()

    with pytest.raises(ValueError, match="'commands.target_copy' reads 'commands.target', so it must run after"):
        builder.move_before(target_copy, target)
    with pytest.raises(ValueError, match="'quantities.sum_copy' reads 'quantities.sum.total', so it must run after"):
        builder.move_before(total_copy, total)
    with pytest.raises(ValueError, match="a use stays in its phase"):
        builder.move_after(alive, target)
    with pytest.raises(ValueError, match="must list each of its entries once"):
        builder.set_order(mdp2.Phase.COMMANDS, [target_copy])

    assert builder.schedule() == schedule

    # Added again, the command runs after its reader.
    builder.remove(target)
    builder.add_command("target", mdp2.Command(uniform_command))

    with pytest.raises(ValueError, match="'commands.target_copy' reads 'commands.target', so it must run after"):
        builder.validate()


def test_schedule_edits_change_execution_order():
    """Pushing then doubling the velocity differs from doubling then pushing. Finalized MDPs keep their order."""
    builder = mdp2.MDPBuilder(PointMassScene(num_worlds=2).schema)
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
            scene = PointMassScene(num_worlds=2, device=DEVICE)
            state, transition = mdp.state(scene), mdp.transition()
            env_reset(mdp, state, scene, transition.observations, torch.ones(2, dtype=torch.bool, device=DEVICE))

            # The events run on the second step.
            for _ in range(2):
                env_step(mdp, state, scene, torch.zeros(2, 1, device=DEVICE), transition)

            assert torch.equal(scene.data("robot", "joint_vel").torch, torch.full((2, 1), expected, device=DEVICE))


def test_step_matches_point_mass_dynamics():
    """Two steps give the analytic positions, velocities, deltas, and rewards of the point mass."""
    scene = PointMassScene(num_worlds=2, joint_names=("x", "y"), device=DEVICE)
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
    """In one step, world 0 continues, world 1 times out, and world 2 times out but also leaves the bounds."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=3, device=DEVICE)
        builder = make_builder(scene.schema, episode_length_s=0.2)
        mdp = builder.finalize(device=DEVICE, backend=backend, final_observations=True)
        state, transition = mdp.state(scene), mdp.transition()
        state.episode_length.torch[1:] = mdp.max_episode_length - 1

        bootstrap = env_step(mdp, state, scene, torch.tensor([[0.1], [0.1], [100.0]], device=DEVICE), transition)

        # A world that terminates is not truncated.
        assert transition.terminated.torch.tolist() == [False, False, True]
        assert transition.truncated.torch.tolist() == [False, True, False]
        assert state.episode_length.torch.tolist() == [1, 0, 0]

        observations, bootstrap = transition.observations["policy"].torch, bootstrap["policy"]
        final = transition.final_observations["policy"].torch
        assert torch.equal(observations[0], bootstrap[0])  # Continuing worlds keep their successor sample.
        assert torch.equal(bootstrap[1], bootstrap[0])  # Truncated worlds bootstrap from the pre-reset sample.
        assert bootstrap[0].abs().sum() > 0.0 and bootstrap[2].isfinite().all()
        assert observations[1:].abs().sum() == 0.0  # Ended worlds observe their new episode.
        assert torch.equal(final[1:], bootstrap[1:]) and final[0].abs().sum() == 0.0


def test_reset_touches_only_selected_worlds():
    """A reset of no worlds changes nothing, and a reset of world 0 changes only world 0."""
    scene = PointMassScene(num_worlds=3, device=DEVICE)
    mdp, state = make_task(scene)
    observations = mdp.observations()
    env_reset(mdp, state, scene, observations, torch.ones(3, dtype=torch.bool, device=DEVICE))
    for _ in range(2):
        env_step(mdp, state, scene, torch.full((3, 1), 0.01, device=DEVICE), mdp.transition())
    mdp2.observe(mdp, state, observations, ProxyArray(wp.ones(3, dtype=wp.bool, device=DEVICE)))

    def snapshot():
        buffers = [state.episode_length, state.actions, observations["policy"], scene.data("robot", "joint_pos")]
        buffers += [state.arrays[mdp2.term_state(op.name, name)] for op in mdp.term_ops for name in op.state]
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
        scene = PointMassScene(num_worlds=2, device=DEVICE)
        mdp, state = make_task(scene, backend=backend, random=True)
        observations = mdp.observations()

        env_reset(mdp, state, scene, observations, torch.ones(2, dtype=torch.bool, device=DEVICE))

        joint_pos, joint_pos_delta = observations["policy"].torch[:, 0], observations["policy"].torch[:, 2]
        assert (joint_pos != 0.0).all() and (joint_pos_delta == 0.0).all()


def test_mdp_fields_subsets_weights_and_diagnostics():
    """MDP field inputs, joint subsets, mutable reward weights, and per-term diagnostics."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=2, joint_names=("x", "y", "z"), device=DEVICE)
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
        far = mdp2.Termination(
            joint_pos_out_of_manual_limit, inputs={"joint_pos": robot.joint_pos}, params={"limit": 5e-5}, time_out=True
        )
        builder.add_termination("far", far)

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
        # A termination with time_out truncates instead, in the worlds that do not terminate.
        assert transition.terminated.torch.tolist() == [False, True]
        assert transition.truncated.torch.tolist() == [True, False]
        assert transition.done.torch.tolist() == [True, True]
        assert torch.equal(transition.termination_terms.torch[:, 0], transition.terminated.torch)
        torch.testing.assert_close(transition.reward_terms.torch.sum(dim=-1), transition.rewards.torch)

        # Weights are state, and may change while the MDP runs.
        state.reward_weights.torch[:, 1] = 0.0
        mdp2.evaluate(mdp, state, transition)

        torch.testing.assert_close(transition.rewards.torch, torch.full((2,), mdp.step_dt, device=DEVICE))


def test_commands_and_interval_events_follow_their_timers():
    """Rewards see the command that governed the step, and observations see the resampled one."""
    scene = PointMassScene(num_worlds=8, device=DEVICE)
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
        scene = PointMassScene(num_worlds=2, joint_names=("x", "y"), device=DEVICE)
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


def test_warp_matches_torch():
    """The Warp and Torch backends agree over episodes with resets, and draw bitwise the same random numbers."""
    runs = {}
    for backend, torch_hooks in (("torch", False), ("warp", False), ("warp", True)):
        scene = PointMassScene(num_worlds=64, joint_names=("x", "y", "z"), device=DEVICE)
        mdp, state = make_task(scene, backend=backend, episode_length_s=0.2, random=True, torch_hooks=torch_hooks)
        assert any(op.torch_hooks for op in mdp.term_ops) == torch_hooks
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
        runs[backend, torch_hooks] = history

    for mixed in (False, True):
        for (torch_results, torch_draws), (warp_results, warp_draws) in zip(runs["torch", False], runs["warp", mixed]):
            for expected, actual in zip(torch_results, warp_results):
                torch.testing.assert_close(actual, expected)
            for expected, actual in zip(torch_draws, warp_draws):
                assert torch.equal(actual, expected)

    # Both ways an episode ends happened.
    assert any(results[1].any() for results, _ in runs["torch", False])
    assert any(results[2].any() for results, _ in runs["torch", False])


def test_host_hooks_match_compiled_terms():
    """Hooks that run on the host see the same data and draw from the same streams as compiled terms, for their worlds
    only."""
    for backend in ("torch", "warp"):
        runs = []
        for host in (False, True):
            scene = PointMassScene(num_worlds=4, joint_names=("x", "y"), device=DEVICE)
            mdp, state = make_task(scene, backend=backend, random=True, host=host)
            transition = mdp.transition()

            mask = torch.tensor([True, True, False, True], device=DEVICE)
            env_reset(mdp, state, scene, transition.observations, mask)
            joint_pos = scene.data("robot", "joint_pos").torch.clone()
            env_step(mdp, state, scene, torch.full((4, mdp.action_dim), 0.1, device=DEVICE), transition)
            runs.append((joint_pos, transition.rewards.torch.clone()))

        (compiled_pos, compiled_rewards), (host_pos, host_rewards) = runs
        assert torch.equal(host_pos, compiled_pos) and (host_pos[2] == 0.0).all()
        torch.testing.assert_close(host_rewards, compiled_rewards)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_captured_mdp_replays_like_eager_calls():
    """Graph replay, with eager terms between the graphs, gives bitwise the same results as eager calls, also with
    Torch hooks inside the graphs of a Warp MDP."""
    for backend, torch_hooks in (("torch", False), ("warp", False), ("warp", True)):
        runs = []
        for capture in (False, True):
            scene = PointMassScene(num_worlds=64, joint_names=("x", "y", "z"), device=DEVICE)
            mdp, state = make_task(
                scene, backend=backend, episode_length_s=0.2, random=True, host=True, torch_hooks=torch_hooks
            )
            assert not mdp.is_fully_capturable
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

        num_segments = sum(segment.capturable for function in mdp.segments.values() for segment in function)
        assert captured.num_graphs == num_segments
        assert torch.equal(runs[1], runs[0])


def test_world_configs_report_errors():
    """Replicated builders report extra uses, uncovered worlds, and differing time limits."""
    schema = PointMassScene(num_worlds=4).schema
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
        schema = PointMassScene(num_worlds=num_worlds, joint_names=("x", "y")).schema
        configs = [
            make_builder(schema, episode_length_s=0.2, random=True),
            make_builder(schema, episode_length_s=0.2, joint_pos_weight=-1.0),
        ]
        mixed = mdp2.MDPBuilder(schema)
        mixed.replicate(configs[0], split)
        mixed.replicate(configs[1], num_worlds - split)

        runs = []
        for builder in (*configs, mixed):
            scene = PointMassScene(num_worlds=num_worlds, joint_names=("x", "y"), device=DEVICE)
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
        ops = {op.name: op for ops in mixed_mdp.ops.values() for op in ops}
        assert ops["rewards.alive"].worlds is None  # A use both configs share runs once, in every world.
        assert ops["rewards.joint_pos_l2"].weights == (-10.0, -1.0)  # Each config keeps its own weight.
        assert ops["terminations.out_of_bounds[0]"].worlds == (0, 1, 2)  # Different uses run apart.

        observation_dims = [config.observation_dims["policy"] for config in mixed_mdp.world_configs]
        assert observation_dims == [first.observation_dims["policy"], second.observation_dims["policy"]]

        for run, worlds in ((first_run, slice(0, split)), (second_run, slice(split, None))):
            for alone, mixed_step in zip(run, mixed_run):
                for key, values in alone.items():
                    assert torch.equal(mixed_step[key][worlds, : values.shape[1]], values[worlds])

        width = second.observation_dims["policy"]
        for mixed_step in mixed_run:
            assert (mixed_step["policy"][split:, width:] == 0.0).all() and (mixed_step["noisy"][split:] == 0.0).all()


host_action_l2 = mdp2.Term(torch=lambda actions: actions.square().sum(dim=-1), capturable=False)


def test_world_configs_with_fewer_actions_read_their_own_actions():
    """The action terms of a world config, eager ones included, see its actions only: the last actions zero the
    columns that pad them for a wider config, so equal terms of both configs are one op."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=4, joint_names=("x", "y"), device=DEVICE)
        configs = []
        for subset in (None, robot.joints("x")):
            config = mdp2.MDPBuilder(scene.schema)
            config.add_action("joint_effort", mdp2.DirectAction(robot.joint_effort_target, subset=subset))
            actions = {"actions": mdp2.last_action}
            config.add_observation("policy", "last_action", mdp2.Observation(mdp2.mdp.last_action, inputs=actions))
            config.add_observation("policy", "action_l2", mdp2.Observation(mdp2.mdp.action_l2, inputs=actions))
            rate = mdp2.Observation(mdp2.mdp.action_rate_l2, inputs=actions)
            config.add_observation("policy", "narrow_rate" if subset else "action_rate_l2", rate)
            config.add_observation("policy", "eager_l2", mdp2.Observation(host_action_l2, inputs=actions))
            configs.append(config)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.replicate(configs[0], 2)
        builder.replicate(configs[1], 2)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()

        steps = [
            torch.tensor([[1.0, 2.0], [0.5, 0.0], [3.0, 7.0], [-1.0, 9.0]], device=DEVICE),
            torch.tensor([[0.0, 2.0], [0.5, 1.0], [1.0, -4.0], [2.0, 5.0]], device=DEVICE),
        ]
        for step_actions in steps:
            mdp2.apply_actions(mdp, state, ProxyArray(wp.from_torch(step_actions)))
            scene.step()
            mdp2.evaluate(mdp, state, transition)

        # Observations of the same name share columns, as wide as the widest, in the order of the merged ops:
        # last_action (0-1), action_l2 (2), action_rate_l2 (3), narrow_rate (4), eager_l2 (5).
        policy = transition.observations["policy"].torch
        torch.testing.assert_close(policy[:2, :2], steps[1][:2])
        torch.testing.assert_close(policy[:2, 2], steps[1][:2].square().sum(-1))
        torch.testing.assert_close(policy[:2, 3], (steps[1][:2] - steps[0][:2]).square().sum(-1))
        # The second column of the actions pads the narrow config: its terms ignore it, and its columns stay zero.
        torch.testing.assert_close(policy[2:, 0], steps[1][2:, 0])
        assert (policy[2:, 1] == 0.0).all() and (policy[2:, 3] == 0.0).all() and (policy[:2, 4] == 0.0).all()
        torch.testing.assert_close(policy[2:, 2], steps[1][2:, 0].square())
        torch.testing.assert_close(policy[2:, 4], (steps[1][2:, 0] - steps[0][2:, 0]).square())
        # Eager terms too.
        torch.testing.assert_close(policy[:2, 5], steps[1][:2].square().sum(-1))
        torch.testing.assert_close(policy[2:, 5], steps[1][2:, 0].square())
        assert mdp.action_mask(state).tolist() == [[True, True]] * 2 + [[True, False]] * 2


def test_library_action_and_termination_terms():
    """The library's action and termination terms, which read MDP fields rather than the scene."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=2, joint_names=("x", "y"), device=DEVICE)
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


def test_observation_noise_then_clip_then_scale():
    """An observation adds its noise, then clips, then scales, as in MDP1."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=1, joint_names=("x", "y"), device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        observation = mdp2.Observation(
            joint_vel,
            inputs={"joint_vel": robot.joint_vel},
            noise=mdp2.UniformNoise(0.25, 0.25),
            clip=(-1.5, 1.5),
            scale=0.5,
        )
        builder.add_observation("policy", "joint_vel", observation)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, observations = mdp.state(scene), mdp.observations()

        scene.data("robot", "joint_vel").torch.copy_(torch.tensor([[1.0, 2.0]], device=DEVICE))
        mdp2.observe(mdp, state, observations, ProxyArray(wp.ones(1, dtype=wp.bool, device=DEVICE)))

        assert observations["policy"].torch.tolist() == [[0.625, 0.75]]


def test_observation_history():
    """The group holds the last values oldest first, and a new episode starts with its first value repeated."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=2, device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_action("joint_effort", mdp2.DirectAction(robot.joint_effort_target))
        history = mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos}, history_length=3)
        builder.add_observation("policy", "joint_pos", history)
        builder.add_termination("time_out", mdp2.TimeLimit(episode_length_s=1.0))
        mdp = builder.finalize(device=DEVICE, backend=backend, final_observations=True)
        state, transition = mdp.state(scene), mdp.transition()

        assert mdp.observation_dims["policy"] == 3

        # World 1 resets to a nonzero position, as a reset event would place it.
        joint_positions = scene.data("robot", "joint_pos").torch
        joint_positions.fill_(0.5)
        reset = torch.tensor([False, True], device=DEVICE)
        scene.reset(reset)
        joint_positions[1] = 0.25
        mdp2.reset(mdp, state, ProxyArray(wp.from_torch(reset)))
        mdp2.observe(mdp, state, transition.observations, ProxyArray(wp.from_torch(reset)))

        # The new episode fills every slot with its first value.
        assert transition.observations["policy"].torch[1].tolist() == [0.25, 0.25, 0.25]

        positions = []
        for _ in range(2):
            env_step(mdp, state, scene, torch.ones(2, 1, device=DEVICE), transition)
            positions.append(joint_positions[:, 0].clone())

        # World 0 was never reset, so its history started at zero.
        expected = torch.stack([torch.tensor([0.0, 0.25], device=DEVICE), *positions], dim=1)
        torch.testing.assert_close(transition.observations["policy"].torch, expected)

        # World 0 times out: its final observation holds the history of the step, before the reset refills it.
        state.episode_length.torch[0] = mdp.max_episode_length - 1
        bootstrap = env_step(mdp, state, scene, torch.ones(2, 1, device=DEVICE), transition)

        final = transition.final_observations["policy"].torch[0]
        assert torch.equal(final, bootstrap["policy"][0]) and torch.equal(final[:2], expected[0, 1:])
        assert transition.observations["policy"].torch[0].tolist() == [0.0] * 3


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
        scene = PointMassScene(num_worlds=4, device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_command("count", mdp2.Command(sample_count, resampling_time_range=(0.03, 0.03)))
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()

        mdp2.reset(mdp, state, ProxyArray(wp.ones(4, dtype=wp.bool, device=DEVICE)))
        for _ in range(4):
            mdp2.evaluate(mdp, state, transition)
        mdp2.reset(mdp, state, ProxyArray(wp.ones(4, dtype=wp.bool, device=DEVICE)))
        results[backend] = state.arrays[mdp2.commands.count].torch.clone()

    # The reset samples once, then the 0.03 s timer expires every second 0.02 s step, then the reset samples again.
    assert torch.equal(results["torch"], torch.full((4,), 4.0, device=DEVICE))
    assert torch.equal(results["warp"], results["torch"])


def drift_torch(mask, command, *, step):
    command.add_(step if mask is None else mask.float() * step)


def drift_warp(mask, command, *, step):
    # A hook that launches its own kernels, here through Torch.
    drift_torch(None if mask is None else wp.to_torch(mask), wp.to_torch(command), step=step)


drifting_count = mdp2.Term(
    torch=mdp2.Implementation(compute=count_samples_torch, update=drift_torch),
    warp=mdp2.Implementation(compute=count_samples_warp, update=drift_warp),
    state={"samples": mdp2.State()},
)
"""A command that counts its samples and drifts by ``step`` every step, only in the worlds where it runs."""


def test_command_updates_change_only_their_worlds():
    """Update hooks that write a field shared by world configs get the worlds where their op runs as ``mask``."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=4, device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        for step in (1.0, 2.0):
            config = mdp2.MDPBuilder(scene.schema)
            command = mdp2.Command(drifting_count, resampling_time_range=(10.0, 10.0), params={"step": step})
            config.add_command("count", command)
            builder.replicate(config, 2)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()

        mdp2.reset(mdp, state, ProxyArray(wp.ones(4, dtype=wp.bool, device=DEVICE)))
        for _ in range(3):
            mdp2.evaluate(mdp, state, transition)

        expected = torch.tensor([4.0, 4.0, 7.0, 7.0], device=DEVICE)
        assert torch.equal(state.arrays[mdp2.commands.count].torch, expected)


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
        scene = PointMassScene(num_worlds=256, device=DEVICE)
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


@wp.func
def reached_warp(world: int, joint_pos: wp.array2d(dtype=float), reached: wp.array(dtype=float)) -> float:
    if joint_pos[world, 0] > 0.0:
        reached[world] = 1.0
    return reached[world]


def reached_torch(joint_pos, reached):
    reached.copy_(torch.where(joint_pos[:, 0] > 0.0, 1.0, reached))
    return reached


reached = mdp2.Term(torch=reached_torch, warp=reached_warp, state={"reached": mdp2.State()})
"""Whether the first joint position was ever positive this episode, sticky like MDP1's success flags.

Its compute writes its state, which a quantity may do; repeating it changes nothing, so observations may read it.
"""


def test_curriculum_reads_quantity_and_scales_noise():
    """A sticky quantity is shared by a reward, an observation, and a curriculum, which reads what the episodes that
    ended left and scales observation noise by its progress, in an MDP that can be captured whole."""
    results = {}
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=4, joint_names=("x", "y"), device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_quantity("reached", mdp2.Quantity(reached, inputs={"joint_pos": robot.joint_pos}))
        builder.add_reward("reach", mdp2.Reward(mdp2.mdp.value, inputs={"value": mdp2.quantities.reached}))
        curriculum = mdp2.Event(
            mdp2.mdp.difficulty_scheduler,
            inputs={"succeeded": mdp2.quantities.reached},
            params={"max_difficulty": 2, "min_progress": 0.5},
        )
        builder.add_event("curriculum", curriculum)
        noise = mdp2.UniformNoise(-1.0, 1.0, scale=mdp2.term_state("events.curriculum", "progress"))
        observation = mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos}, noise=noise)
        builder.add_observation("policy", "joint_pos", observation)
        observation = mdp2.Observation(mdp2.mdp.value, inputs={"value": mdp2.quantities.reached})
        builder.add_observation("policy", "reached", observation)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()
        all_worlds = torch.ones(4, dtype=torch.bool, device=DEVICE)
        joint_positions = scene.data("robot", "joint_pos").torch

        assert mdp.is_fully_capturable

        # No episode succeeded yet, so there is no progress and no noise.
        env_reset(mdp, state, scene, transition.observations, all_worlds)
        assert torch.equal(transition.observations["policy"].torch[:, 0], joint_positions[:, 0])

        # Every world reaches, then resets: each difficulty rises to 1 of 2, and the noise grows to half its range.
        joint_positions[:, 0] = 0.5
        mdp2.evaluate(mdp, state, transition)
        assert (transition.rewards.torch > 0.0).all()
        env_reset(mdp, state, scene, transition.observations, all_worlds)

        assert state.arrays[mdp2.term_state("events.curriculum", "progress")].torch.tolist() == [0.5] * 4
        # The sticky state restarts after the curriculum read it, and the new episode's observation sees that.
        assert state.arrays[mdp2.term_state("quantities.reached", "reached")].torch.tolist() == [0.0] * 4
        assert transition.observations["policy"].torch[:, 2].tolist() == [0.0] * 4
        noise_values = transition.observations["policy"].torch[:, :2] - joint_positions
        assert (noise_values.abs() <= 0.5).all() and (noise_values != 0.0).any()
        results[backend] = noise_values

    assert torch.equal(results["torch"], results["warp"])


def test_noise_scaled_by_a_quantity_follows_the_new_episode():
    """A quantity that only scales observation noise is computed again for the observations of a new episode."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=4, joint_names=("x", "y"), device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_quantity("reached", mdp2.Quantity(reached, inputs={"joint_pos": robot.joint_pos}))
        noise = mdp2.UniformNoise(-1.0, 1.0, scale=mdp2.quantities.reached)
        observation = mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos}, noise=noise)
        builder.add_observation("policy", "joint_pos", observation)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()
        all_worlds = torch.ones(4, dtype=torch.bool, device=DEVICE)
        joint_positions = scene.data("robot", "joint_pos").torch

        # The episode reaches, so its observations are noisy; the new one has not reached yet, so they are not.
        joint_positions[:, 0] = 0.5
        mdp2.evaluate(mdp, state, transition)
        assert (transition.observations["policy"].torch != joint_positions).any()
        joint_positions[:, 0] = -0.5
        env_reset(mdp, state, scene, transition.observations, all_worlds)
        assert torch.equal(transition.observations["policy"].torch, joint_positions)


@wp.func
def hold_second_joint_warp(world: int, joint_pos: wp.array2d(dtype=float)):
    joint_pos[world, 1] = 1.0


def hold_second_joint_torch(mask, joint_pos):
    joint_pos[:, 1] = torch.where(mask, 1.0, joint_pos[:, 1])


hold_second_joint = mdp2.Term(
    torch=hold_second_joint_torch, warp=hold_second_joint_warp, state={"spare": mdp2.State(persistent=True)}
)
"""Reset event that sets the second joint position to one. Its hooks do not take its state."""


def test_action_offset_follows_a_quantity_through_resets():
    """An action offset by a quantity uses the value of the last step, and of the new episode after a reset, while
    a quantity that nothing reads between steps keeps what the episode that ended left."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=4, joint_names=("x", "y"), device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_action("effort", mdp2.DirectAction(robot.joint_effort_target, offset=mdp2.quantities.reference))
        builder.add_quantity("reference", mdp2.Quantity(joint_pos, inputs={"joint_pos": robot.joint_pos}))
        builder.add_quantity("reached", mdp2.Quantity(reached, inputs={"joint_pos": robot.joint_pos}))
        builder.add_event("hold", mdp2.Event(hold_second_joint, outputs={"joint_pos": robot.joint_pos}))
        builder.add_observation(
            "policy", "joint_pos", mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos})
        )
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()
        all_worlds = torch.ones(4, dtype=torch.bool, device=DEVICE)
        actions = ProxyArray(wp.zeros((4, 2), dtype=wp.float32, device=DEVICE))
        targets = scene.data("robot", "joint_effort_target").torch

        env_reset(mdp, state, scene, transition.observations, all_worlds)
        scene.data("robot", "joint_pos").torch[:, 0] = 0.5
        mdp2.evaluate(mdp, state, transition)
        mdp2.apply_actions(mdp, state, actions)
        assert targets.tolist() == [[0.5, 1.0]] * 4

        env_reset(mdp, state, scene, transition.observations, all_worlds)
        mdp2.apply_actions(mdp, state, actions)
        assert targets.tolist() == [[0.0, 1.0]] * 4
        assert state.arrays[mdp2.quantities.reached].torch.tolist() == [1.0] * 4


@wp.func
def accumulate_warp(world: int, joint_pos: wp.array2d(dtype=float), total: wp.array(dtype=float), gain: float) -> float:
    total[world] = total[world] + gain * joint_pos[world, 0]
    return total[world]


def accumulate_torch(joint_pos, total, *, gain):
    total.add_(gain * joint_pos[:, 0])
    return total


accumulate = mdp2.Term(torch=accumulate_torch, warp=accumulate_warp, state={"total": mdp2.State()})
"""Sum of ``gain`` times the first joint position over the episode."""


def test_world_configs_share_the_state_of_a_use_they_define_differently():
    """Two world configs give a quantity different params: it is two ops with one state, each in its own worlds,
    which a reward of both configs reads."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=4, device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        for gain in (1.0, 2.0):
            config = mdp2.MDPBuilder(scene.schema)
            quantity = mdp2.Quantity(accumulate, inputs={"joint_pos": robot.joint_pos}, params={"gain": gain})
            config.add_quantity("sum", quantity)
            total = mdp2.term_state("quantities.sum", "total")
            config.add_reward("sum", mdp2.Reward(mdp2.mdp.value, inputs={"value": total}))
            builder.replicate(config, 2)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()
        assert [op.name for op in mdp.ops[mdp2.Phase.QUANTITIES]] == ["quantities.sum[0]", "quantities.sum[1]"]

        scene.data("robot", "joint_pos").torch.fill_(1.0)
        for _ in range(2):
            mdp2.evaluate(mdp, state, transition)
        assert state.arrays[mdp2.term_state("quantities.sum", "total")].torch.tolist() == [2.0, 2.0, 4.0, 4.0]
        torch.testing.assert_close(
            transition.rewards.torch / mdp.step_dt, torch.tensor([2.0, 2.0, 4.0, 4.0], device=DEVICE)
        )


def add_reset_count_torch(mask, total, *, gain):
    total.add_(10.0 * gain * mask.float())


def torch_accumulate(initial: float) -> mdp2.Term:
    """:data:`accumulate` with Torch hooks only, its state starting at ``initial``, and a reset hook that adds
    ``10 * gain``."""
    hooks = mdp2.Implementation(compute=accumulate_torch, reset=add_reset_count_torch)
    return mdp2.Term(torch=hooks, state={"total": mdp2.State(initial=initial)})


def test_world_configs_keep_python_hooks_in_their_worlds():
    """Torch hooks of a use that world configs define differently, which a Warp MDP also runs over every world, act
    in their own worlds: the state starts at each config's initial value, compute and reset hooks change it in their
    worlds only, and an untimed command resamples in its worlds."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=4, device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        for gain in (1.0, 2.0):
            config = mdp2.MDPBuilder(scene.schema)
            quantity = mdp2.Quantity(
                torch_accumulate(gain), inputs={"joint_pos": robot.joint_pos}, params={"gain": gain}
            )
            config.add_quantity("sum", quantity)
            # Torch-only, so that a Warp MDP runs its hook over every world; the step only sets the configs apart.
            command = mdp2.Command(mdp2.Term(torch=count_samples_torch, state={"samples": mdp2.State()}))
            config.add_command("count", dataclasses.replace(command, params={"step": gain}))
            builder.replicate(config, 2)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()
        total = state.arrays[mdp2.term_state("quantities.sum", "total")].torch

        scene.data("robot", "joint_pos").torch.fill_(1.0)
        mdp2.evaluate(mdp, state, transition)
        assert total.tolist() == [2.0, 2.0, 4.0, 4.0]

        mdp2.reset(mdp, state, ProxyArray(wp.from_torch(torch.tensor([True, False, True, False], device=DEVICE))))
        assert total.tolist() == [12.0, 2.0, 24.0, 4.0]
        assert state.arrays[mdp2.commands.count].torch.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_uses_that_read_each_other_are_reported_as_a_cycle():
    """A quantity that reads the state of an event that reads the quantity cannot be ordered: the only error, also
    in one world config of several."""
    counting = mdp2.Term(torch=lambda mask, value, count: None, state={"count": mdp2.State()})
    schema = PointMassScene(num_worlds=2).schema
    cyclic = make_builder(schema)
    count = mdp2.term_state("events.counting", "count")
    cyclic.add_quantity("mirror", mdp2.Quantity(mdp2.mdp.value, inputs={"value": count}))
    cyclic.add_event("counting", mdp2.Event(counting, inputs={"value": mdp2.quantities.mirror}))
    builder = mdp2.MDPBuilder(schema)
    builder.replicate(cyclic, 1)
    builder.replicate(make_builder(schema), 1)
    with pytest.raises(ValueError) as error:
        builder.validate()
    assert "in a cycle: quantities.mirror, events.counting" in str(error.value)
    assert "unknown field" not in str(error.value) and "time limit" not in str(error.value)


def test_reset_commits_before_events_that_ask_for_it():
    """The scene commits the writes of the earlier reset events before an event that asks for it, and before an event
    that takes the entity they wrote, which may read any of its data."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=2, device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_event("push", mdp2.Event(push_by_adding_velocity, outputs={"joint_vel": robot.joint_vel}))
        scale = mdp2.Event(scale_joint_vel, outputs={"joint_vel": robot.joint_vel}, params={"factor": 3.0}, commit=True)
        builder.add_event("scale", scale)
        seen = []
        record = mdp2.Term(
            torch=lambda mask, robot: seen.append(robot.data("robot", "joint_vel").torch.clone()), capturable=False
        )
        builder.add_event("record", mdp2.Event(record, inputs={"robot": robot}))
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state = mdp.state(scene)
        joint_velocities = scene.data("robot", "joint_vel").torch

        committed = []
        mask = ProxyArray(wp.from_torch(torch.ones(2, dtype=torch.bool, device=DEVICE)))
        mdp2.reset(mdp, state, mask, commit=lambda: committed.append(joint_velocities.clone()))

        assert len(committed) == 2 and (committed[0] == 1.0).all() and (committed[1] == 3.0).all()
        assert (joint_velocities == 3.0).all() and (seen[0] == 3.0).all()


record_tip = mdp2.Term(torch=lambda mask, tip: None)
"""An event that reads the tip, a field derived from the joint positions."""


def test_reset_commits_before_events_that_read_derived_fields():
    """A commit is inferred before a reset event that reads a field derived from fields earlier events wrote, and
    nowhere else; an event can opt out."""
    scene = PointMassScene(num_worlds=1, device=DEVICE)
    builder = mdp2.MDPBuilder(scene.schema)
    builder.add_event("push", mdp2.Event(push_by_adding_velocity, outputs={"joint_vel": robot.joint_vel}))
    builder.add_event("read_tip_early", mdp2.Event(record_tip, inputs={"tip": robot.tip}))
    builder.add_event("offset", mdp2.Event(reset_joints_by_offset, outputs={"joint_pos": robot.joint_pos}))
    builder.add_event("read_tip", mdp2.Event(record_tip, inputs={"tip": robot.tip}))
    builder.add_event("read_tip_again", mdp2.Event(record_tip, inputs={"tip": robot.tip}))
    builder.add_event("offset_again", mdp2.Event(reset_joints_by_offset, outputs={"joint_pos": robot.joint_pos}))
    builder.add_event("opt_out", mdp2.Event(record_tip, inputs={"tip": robot.tip}, commit=False))
    mdp = builder.finalize(device=DEVICE, backend="torch")

    commits = [step.op.name for step in mdp.plan.reset if step.kind is mdp2.StepKind.COMMIT]
    assert commits == ["events.read_tip"]


def test_entity_outputs_write_every_writable_field_of_the_entity():
    """An event that writes through an entity covers its writable fields: events that read one run after it, the
    scene commits them, and fused wp.func hooks cannot take an entity."""
    schema = PointMassScene(num_worlds=1).schema
    through_robot = mdp2.Event(mdp2.Term(torch=lambda mask, robot: None), outputs={"robot": robot})
    read_joints = mdp2.Event(mdp2.Term(torch=lambda mask, joint_pos: None), inputs={"joint_pos": robot.joint_pos})

    builder = mdp2.MDPBuilder(schema)
    reader = builder.add_event("read_joints", read_joints)
    writer = builder.add_event("through_robot", through_robot)
    with pytest.raises(ValueError, match="'events.read_joints' reads 'robot.joint_pos', so it must run after"):
        builder.validate()
    builder.move_after(reader, writer)
    mdp = builder.finalize(device=DEVICE, backend="torch")
    written = {"robot.joint_pos", "robot.joint_vel", "robot.joint_effort_target"}
    assert {str(selector) for selector in mdp.reset_writes} == written

    builder = mdp2.MDPBuilder(schema)
    builder.add_event("fused", mdp2.Event(push_by_adding_velocity, inputs={"robot": robot}))
    with pytest.raises(ValueError, match="wp.func hooks cannot take scene entities"):
        builder.validate(backend="warp")


@pytest.mark.parametrize("close_above", [True, False])
def test_action_terms_take_their_columns_after_direct_actions(close_above: bool):
    """An action term takes the next columns of the action and writes its outputs, here a binary command that closes
    on either side of its threshold."""
    for backend in ("torch", "warp"):
        scene = PointMassScene(num_worlds=2, joint_names=("x", "y"), device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_action("x", mdp2.DirectAction(robot.joint_effort_target, subset=robot.joints("x"), scale=2.0))
        gripper = mdp2.Action(
            mdp2.mdp.binary_joint_position,
            inputs={"joint_ids": robot.joints("y")},
            outputs={"joint_pos_target": robot.joint_effort_target},
            params={"open_position": -1.0, "close_position": 1.0, "threshold": 0.5, "close_above": close_above},
        )
        builder.add_action("y", gripper)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state = mdp.state(scene)

        actions = torch.tensor([[0.5, 0.9], [-1.0, 0.2]], device=DEVICE)
        mdp2.apply_actions(mdp, state, ProxyArray(wp.from_torch(actions)))
        assert mdp.action_dim == 2
        gripper_targets = [1.0, -1.0] if close_above else [-1.0, 1.0]
        assert scene.data("robot", "joint_effort_target").torch.tolist() == [
            [1.0, gripper_targets[0]],
            [-2.0, gripper_targets[1]],
        ]
        torch.testing.assert_close(state.actions.torch, actions)


def image_scene(num_worlds: int) -> PointMassScene:
    """A point-mass scene with a ``camera`` entity whose ``rgb`` field is a 2x3 image with 3 channels."""
    scene = PointMassScene(num_worlds=num_worlds, device=DEVICE)
    names = {"height": ("0", "1"), "width": ("0", "1", "2"), "rgb_channels": ("0", "1", "2")}
    rgb = mdp2.FieldSchema(("height", "width", "rgb_channels"), wp.uint8)
    camera = mdp2.EntitySchema(names=names, fields={"rgb": rgb})
    scene.schema = dataclasses.replace(scene.schema, entities={**scene.schema.entities, "camera": camera})
    images = torch.arange(num_worlds * 18, device=DEVICE).reshape(num_worlds, 2, 3, 3).to(torch.uint8)
    scene._data["camera"] = {"rgb": ProxyArray(wp.from_torch(images, dtype=wp.uint8))}
    return scene


def test_groups_that_are_not_concatenated_keep_each_term_shape_and_type():
    """A group that is not concatenated gives each term its own buffer, of the term's shape and element type."""
    camera = mdp2.Entity("camera")
    for backend in ("torch", "warp"):
        scene = image_scene(num_worlds=2)
        task = mdp2.TaskCfg(
            observations={
                "policy": {"joint_pos": mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos})},
                "cameras": mdp2.ObservationGroup(
                    terms={
                        "rgb": mdp2.Observation(mdp2.mdp.image_rgb, inputs={"image": camera.rgb}),
                        "joint_pos": mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos}),
                    },
                    concatenate=False,
                ),
            }
        )
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_task(task)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()
        scene.data("robot", "joint_pos").torch.copy_(torch.tensor([[1.0], [2.0]]))
        mdp2.evaluate(mdp, state, transition)

        assert dict(mdp.observation_layouts) == {
            "policy": ((1,), wp.float32),
            "cameras/rgb": ((2, 3, 3), wp.uint8),
            "cameras/joint_pos": ((1,), wp.float32),
        }
        rgb = transition.observations["cameras/rgb"].torch
        assert rgb.dtype == torch.uint8
        assert torch.equal(rgb.view(2, 2, 3, 3), scene.data("camera", "rgb").torch)
        assert transition.observations["cameras/joint_pos"].torch.tolist() == [[1.0], [2.0]]

    # Other types need their own buffer, without processing.
    builder = mdp2.MDPBuilder(image_scene(num_worlds=1).schema)
    builder.add_observation("policy", "rgb", mdp2.Observation(mdp2.mdp.image_rgb, inputs={"image": camera.rgb}))
    scaled = mdp2.Observation(mdp2.mdp.image_rgb, inputs={"image": camera.rgb}, scale=2.0)
    builder.add_observation("cameras", "scaled", scaled, concatenate=False)
    with pytest.raises(ValueError) as error:
        builder.validate()
    for name in ("policy.rgb", "cameras.scaled"):
        assert f"observations.{name}: an observation of type uint8 must be in a group with concatenate=False" in str(
            error.value
        )


def test_compose_merges_tasks_in_order():
    """Composed tasks merge term by term in the order of the parts, a later part removes a use by setting it to None,
    and two parts that define the same use conflict."""
    event = mdp2.Event(push_by_adding_velocity, outputs={"joint_vel": robot.joint_vel})
    observation = mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos})
    scene_part = mdp2.TaskCfg(events={"place": event}, observations={"policy": {"a": observation}})
    embodiment = mdp2.TaskCfg(events={"reset_arm": event, "hold": event}, observations={"policy": {"b": observation}})
    task_part = mdp2.TaskCfg(events={"hold": None, "settle": event})
    task = mdp2.compose(scene_part, embodiment, task_part)
    assert list(task.observations["policy"].terms) == ["a", "b"] and task.events["hold"] is None

    builder = mdp2.MDPBuilder(PointMassScene(num_worlds=1).schema)
    builder.add_task(task)
    events = [entry.path[1] for entry in builder.schedule() if entry.path[0] == "events"]
    assert events == ["place", "reset_arm", "settle"]

    with pytest.raises(ValueError, match="Tasks 0 and 1 both define 'observations.policy.a'"):
        mdp2.compose(scene_part, scene_part)


@wp.func
def joint_above_warp(world: int, joint_pos: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)) -> float:
    return wp.where(joint_pos[world, joint_ids[0]] > 0.5, 1.0, 0.0)


joint_above = mdp2.Term(
    torch=lambda joint_pos, joint_ids: (joint_pos[:, joint_ids][:, 0] > 0.5).float(), warp=joint_above_warp
)


def above(*joints: str) -> dict[str, mdp2.Quantity]:
    """Quantities, one per joint, that are one where the joint is above 0.5."""
    return {
        joint: mdp2.Quantity(joint_above, inputs={"joint_pos": robot.joint_pos, "joint_ids": robot.joints(joint)})
        for joint in joints
    }


def run_progress(
    task: mdp2.TaskCfg, joint_values: list, backend: str, capture: bool, num_worlds: int = 1, joints: tuple = ()
):
    """Run a task whose joints, named after its quantities unless given, are set at every step, and return per step
    the fields of every task's progress by name, every other field of the MDP, such as term state, by selector, and
    the rewards, terminations, per-term rewards, and observations."""
    joint_names = joints or tuple(task.quantities)
    scene = PointMassScene(num_worlds=num_worlds, joint_names=joint_names, device=DEVICE)
    builder = mdp2.MDPBuilder(scene.schema)
    builder.add_task(task)
    mdp = builder.finalize(device=DEVICE, backend=backend, diagnostics=True)
    state, transition = mdp.state(scene), mdp.transition()
    actions = ProxyArray(wp.zeros((num_worlds, max(mdp.action_dim, 1)), dtype=wp.float32, device=DEVICE))
    captured = mdp2.CapturedMDP(mdp, state, transition, actions, transition.done) if capture else None
    env_reset(mdp, state, scene, transition.observations, torch.ones(num_worlds, dtype=torch.bool, device=DEVICE))

    joints = scene.data("robot", "joint_pos").torch
    fields = [key for key in state.arrays if isinstance(key, mdp2.FieldSelector) and not key.scene]
    history = []
    for values in joint_values:
        joints.copy_(torch.tensor(values, device=DEVICE).expand_as(joints))
        if captured is None:
            mdp2.evaluate(mdp, state, transition)
        else:
            captured.evaluate()
        step = {key.field if key.entity == "progress" else str(key): state.arrays[key].torch.clone() for key in fields}
        step["rewards"] = transition.rewards.torch.clone()
        step["reward_terms"] = transition.reward_terms.torch.clone()
        step["terminated"] = transition.terminated.torch.clone()
        step |= {group: values.torch.clone() for group, values in transition.observations.items()}
        history.append(step)
    return mdp, state, history


def check_progress(task: mdp2.TaskCfg, joint_values: list, joints: tuple = ()) -> list[dict[str, torch.Tensor]]:
    """Run a task on both backends, eagerly and captured, check that the four runs agree bitwise, and return one."""
    runs = []
    for backend in ("torch", "warp"):
        for capture in (False, True) if torch.cuda.is_available() else (False,):
            runs.append(run_progress(task, joint_values, backend, capture, joints=joints)[2])
    for run in runs[1:]:
        for expected, actual in zip(runs[0], run):
            for name, value in expected.items():
                assert torch.equal(value, actual[name]), name
    return runs[0]


def test_milestones_count_their_hold_from_the_step_after_the_previous_one():
    """Lift, then upright for 10 steps: the hold starts the step after the lift, and a break restarts it."""
    lift_and_hold = mdp2.compose(
        mdp2.TaskCfg(quantities=above("lifted", "upright")),
        mdp2.sequence(
            lifted=mdp2.Milestone(mdp2.quantities.lifted),
            upright=mdp2.Milestone(mdp2.quantities.upright, hold=10),
        ),
    )
    # Upright from the start; lifted from step 5 (episode steps count from 1).
    steps = [[0.0, 1.0]] * 4 + [[1.0, 1.0]] * 16
    history = check_progress(lift_and_hold, steps)
    done = [int(step["done"].item()) for step in history]
    assert done.index(1) == 14  # Step 15: 10 upright steps after the lift at step 5
    assert history[-1]["lifted.done_at"].item() == 5 and history[-1]["done_at"].item() == 15
    assert history[5]["upright.started_at"].item() == 5 and history[4]["stage"].item() == 1

    # Tipping over at step 10 restarts the count.
    steps = [[0.0, 1.0]] * 4 + [[1.0, 1.0]] * 5 + [[1.0, 0.0]] + [[1.0, 1.0]] * 12
    history = check_progress(lift_and_hold, steps)
    assert [int(step["done"].item()) for step in history].index(1) == 19


def test_sequence_rechecks_the_final_completion_of_its_subtasks():
    """Put in, then close: the second subtask starts the step after the first, even if already satisfied, and the
    task is done only while both still hold; reopening keeps the recorded completion of the first."""
    task = mdp2.compose(
        mdp2.TaskCfg(quantities=above("inside", "closed")),
        mdp2.sequence(
            put=mdp2.Milestone(mdp2.quantities.inside, hold=2, final=True),
            close=mdp2.Milestone(mdp2.quantities.closed, final=True),
        ),
    )
    # Closed from the start, inside from step 3: put is done at step 4, close at step 5, and so the task.
    history = check_progress(task, [[0.0, 1.0]] * 2 + [[1.0, 1.0]] * 4)
    final = history[-1]
    assert (final["put.done_at"].item(), final["close.done_at"].item(), final["done_at"].item()) == (4, 5, 5)

    # Taken out again at step 5: close is done, but the task is not, since put no longer holds.
    history = check_progress(task, [[0.0, 1.0]] * 2 + [[1.0, 1.0]] * 2 + [[0.0, 1.0]] * 3)
    final = history[-1]
    assert final["put.done_at"].item() == 4 and final["close.done"].item() and not final["done"].item()


def test_together_requires_some_subtasks_and_scores_the_best():
    """Two of three subtasks: done once two are, scored by the mean of the best two, each a weighted sequence."""

    def sort(joint: str) -> mdp2.TaskCfg:
        return mdp2.sequence(
            grasped=mdp2.Milestone(getattr(mdp2.quantities, joint), score=0.25),
            binned=mdp2.Milestone(getattr(mdp2.quantities, joint), score=0.75),
        )

    task = mdp2.compose(
        mdp2.TaskCfg(quantities=above("red", "blue", "green")),
        red=sort("red"),
        blue=sort("blue"),
        green=sort("green"),
        require=2,
    )
    history = check_progress(task, [[1.0, 0.0, 0.0]] * 2 + [[1.0, 1.0, 0.0]])
    assert history[0]["red.score"].item() == 0.25 and history[0]["score"].item() == 0.125
    assert history[1]["red.done"].item() and not history[1]["done"].item()
    assert history[2]["score"].item() == 0.625 and not history[2]["done"].item()


def test_tasks_of_a_sequence_run_their_uses_in_turn():
    """Rewards follow the subtask that ran when the step started, and observations of the same name share their
    columns, written by the subtask that runs, with a history that restarts with it. Terms read the progress."""

    def stage(joint: str, reward: float) -> mdp2.TaskCfg:
        position = mdp2.Observation(
            joint_pos_subset,
            inputs={"joint_pos": robot.joint_pos, "joint_ids": robot.joints(joint)},
            history_length=2,
        )
        return mdp2.TaskCfg(
            rewards={"alive": mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_pos}, weight=reward)},
            observations={"policy": {"position": position}},
            done=getattr(mdp2.quantities, joint),
        )

    stage_index = mdp2.Observation(mdp2.mdp.one_hot, inputs={"value": mdp2.progress.stage}, params={"size": 2})
    task = mdp2.compose(
        mdp2.TaskCfg(
            quantities=above("reach", "lift"),
            observations={"policy": {"stage": stage_index}},
            terminations={"success": mdp2.Termination(mdp2.mdp.is_true, inputs={"value": mdp2.progress.done})},
        ),
        mdp2.sequence(reach=stage("reach", 1.0), lift=stage("lift", 2.0)),
    )
    history = check_progress(task, [[0.0, 0.25], [1.0, 0.25], [1.0, 0.75]])
    assert [step["terminated"].item() for step in history] == [False, False, True]
    step_dt = 0.02
    rewards = [round(step["rewards"].item() / step_dt, 6) for step in history]
    assert rewards == [1.0, 1.0, 2.0]  # Reach until its step 2 is done, lift from step 3
    # The stage (columns 0-1), then the position of the subtask that runs, with its history (columns 2-3).
    policy = [step["policy"][0].tolist() for step in history]
    assert policy[0] == [1.0, 0.0, 0.0, 0.0]
    assert policy[1] == [0.0, 1.0, 0.25, 0.25]  # Lift's position, its history filled at its start
    assert policy[2] == [0.0, 0.0, 0.25, 0.75]  # The last subtask keeps running once the sequence is done

    mdp, state, _ = run_progress(task, [[0.0, 0.25]], "warp", capture=False)
    position = mdp.spec().observations["policy"][1]
    assert (position.name, position.columns, position.history_length) == ("position", slice(2, 4), 2)
    assert dict(position.labels) == {
        "observations.policy.reach.position": ("reach",),
        "observations.policy.lift.position": ("lift",),
    }
    assert mdp.observation_mask(state, "policy").tolist() == [[True] * 4]
    assert "during reach" in mdp.describe()


def test_tasks_restart_their_state_timers_and_commands_when_they_start():
    """A task of a sequence restarts its term state (to its initial value, or with its reset hook), its timers, and
    its commands when it starts, and its per-term rewards are zero once it stopped."""
    gain = {"joint_pos": robot.joint_pos}

    def stage(joint: str, hooked: bool) -> mdp2.TaskCfg:
        rewards = {"fused": mdp2.Reward(accumulate, inputs=gain, params={"gain": 1.0})}
        commands = {}
        if hooked:
            rewards["hooked"] = mdp2.Reward(torch_accumulate(5.0), inputs=gain, params={"gain": 1.0})
            commands["count"] = mdp2.Command(sample_count, resampling_time_range=(0.04, 0.04))
        return mdp2.TaskCfg(rewards=rewards, commands=commands, done=getattr(mdp2.quantities, joint))

    task = mdp2.compose(
        mdp2.TaskCfg(quantities=above("a", "b")), mdp2.sequence(first=stage("a", False), second=stage("b", True))
    )
    # The first task is done at step 2, the second runs from step 3.
    history = check_progress(task, [[0.25, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    total = [step[str(mdp2.term_state("rewards.second.fused", "total"))].item() for step in history]
    hooked = [step[str(mdp2.term_state("rewards.second.hooked", "total"))].item() for step in history]
    samples = [step[str(mdp2.term_state("commands.second.count", "samples"))].item() for step in history]
    assert total[2:] == [1.0, 2.0, 3.0]  # Restarted at zero, then the first joint at every step
    assert hooked[1:] == [15.0, 16.0, 17.0, 18.0]  # Its initial 5, then its reset hook's 10, at its start
    assert samples == [0.0, 1.0, 1.0, 2.0, 2.0]  # Sampled at its start, then every two steps
    first = [step[str(mdp2.term_state("rewards.first.fused", "total"))].item() for step in history]
    assert first == [0.25, 1.25, 1.25, 1.25, 1.25]
    assert all(step["reward_terms"][0, 0].item() == 0.0 for step in history[2:])


def test_subtasks_read_their_own_fields_by_name():
    """Subtasks written as tasks of their own read their quantities, the state of their uses, and their progress by
    their names, which their prefix keeps apart."""

    def reach(joint: str) -> mdp2.TaskCfg:
        inputs = {"joint_pos": robot.joint_pos, "joint_ids": robot.joints(joint)}
        started = mdp2.Observation(episode_length, inputs={"episode_length": mdp2.progress.started_at})
        total = {"command": mdp2.term_state("rewards.held", "total")}
        return mdp2.TaskCfg(
            quantities={
                "reached": mdp2.Quantity(joint_above, inputs=inputs),
                "held": mdp2.Quantity(generated_commands, inputs=total),
            },
            rewards={"held": mdp2.Reward(accumulate, inputs={"joint_pos": robot.joint_pos}, params={"gain": 1.0})},
            observations={"policy": {"started": started}},
            done=mdp2.quantities.reached,
        )

    task = mdp2.sequence(first=reach("a"), second=reach("b"))
    history = check_progress(task, [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], joints=("a", "b"))
    assert [step["first.done_at"].item() for step in history] == [-1, 2, 2]
    assert [step["second.done_at"].item() for step in history] == [-1, -1, 3]
    # The step at which the subtask that runs started: the second from step 2.
    assert [step["policy"][0, 0].item() for step in history] == [0.0, 2.0, 2.0]
    # Each subtask's quantity reads the state of its own reward, from the step before.
    assert [step["quantities.second.held"].item() for step in history] == [0.0, 0.0, 0.0]
    assert [step["quantities.first.held"].item() for step in history] == [0.0, 0.0, 1.0]


def test_split_runs_a_sequence_beside_a_plain_task():
    """A split gives a sequence and a plain task their own worlds, and the root's progress is each world's own."""
    sequence = mdp2.sequence(first=mdp2.Milestone(mdp2.quantities.a), second=mdp2.Milestone(mdp2.quantities.b))
    task = mdp2.compose(
        mdp2.TaskCfg(quantities=above("a", "b")),
        mdp2.split(staged=sequence, plain=mdp2.TaskCfg(done=mdp2.quantities.b), worlds={"staged": 3, "plain": 1}),
    )
    mdp, _, history = run_progress(task, [[1.0, 1.0], [1.0, 1.0]], "warp", capture=False, num_worlds=4)
    assert [config.worlds for config in mdp.world_configs] == [range(0, 3), range(3, 4)]
    assert history[0]["done"].tolist() == [False] * 3 + [True]
    assert history[1]["done"].tolist() == [True] * 4
    assert history[1]["staged.done"].tolist() == [True] * 3 + [False]


def test_splits_divide_the_same_worlds_and_may_be_steps_of_a_sequence():
    """Two subtasks that split divide the same worlds, each world running the subtask of its share of each, and a step
    of a sequence may split the worlds."""
    a, b = mdp2.Milestone(mdp2.quantities.a), mdp2.Milestone(mdp2.quantities.b)
    task = mdp2.compose(
        mdp2.TaskCfg(quantities=above("a", "b")),
        x=mdp2.split(p=a, r=b),
        y=mdp2.split(worlds={"s": 1, "t": 3}, s=a, t=b),
        z=mdp2.sequence(first=a, second=mdp2.split(p=a, r=b)),
    )
    mdp, _, history = run_progress(task, [[1.0, 0.0], [1.0, 0.0]], "warp", capture=False, num_worlds=4)
    assert [config.worlds for config in mdp.world_configs] == [range(0, 1), range(1, 2), range(2, 4)]
    assert history[0]["x.done"].tolist() == [True, True, False, False]
    assert history[0]["y.done"].tolist() == [True, False, False, False]
    # The sequence's second step runs from step 2: on "a" in the first two worlds, on "b" in the others.
    assert history[0]["z.done"].tolist() == [False] * 4
    assert history[1]["z.done"].tolist() == [True, True, False, False]


def test_task_trees_report_errors():
    """A sequence task without a completion and a task with subtasks and a done are errors."""
    schema = PointMassScene(num_worlds=2, joint_names=("a",)).schema
    quantities = mdp2.TaskCfg(quantities=above("a"))
    milestone = mdp2.Milestone(mdp2.quantities.a)
    cases = [
        (mdp2.sequence(first=milestone, second=mdp2.TaskCfg()), "needs a completion"),
        (mdp2.TaskCfg(done=mdp2.quantities.a, subtasks={"first": milestone}), "cannot have a done"),
    ]
    for task, message in cases:
        with pytest.raises(ValueError, match=message):
            builder = mdp2.MDPBuilder(schema)
            builder.add_task(mdp2.compose(quantities, task))
            builder.validate()


@wp.func
def scaled_sum_warp(world: int, joint_pos: wp.array2d(dtype=float), gain: float) -> float:
    total = float(0.0)
    for j in range(joint_pos.shape[1]):
        total += joint_pos[world, j]
    return gain * total


scaled_sum = mdp2.Term(torch=lambda joint_pos, *, gain: gain * joint_pos.sum(-1), warp=scaled_sum_warp)


@wp.func
def outside_warp(world: int, joint_pos: wp.array2d(dtype=float), bounds: wp.vec2f) -> bool:
    return joint_pos[world, 0] < bounds[0] or joint_pos[world, 0] > bounds[1]


def outside_torch(joint_pos, *, bounds):
    low, high = bounds
    return (joint_pos[:, 0] < low) | (joint_pos[:, 0] > high)


outside = mdp2.Term(torch=outside_torch, warp=outside_warp)


def test_tunable_values_change_at_resets_or_steps_while_graphs_replay():
    """Tunable params and weights hold one value per world, which terms read as their world's value and observations
    as a field. A tuning waits for each world's next reset, or applies at the next step, also between graph replays,
    and every world counts the tunings it applied."""
    task = mdp2.TaskCfg(
        observations={
            "policy": {
                "gain": mdp2.Observation(generated_commands, inputs={"command": mdp2.tunable("rewards.scaled", "gain")})
            }
        },
        rewards={
            "scaled": mdp2.Reward(
                scaled_sum,
                inputs={"joint_pos": robot.joint_pos},
                params={"gain": mdp2.Tunable(1.0)},
                weight=mdp2.Tunable(1.0),
            )
        },
        terminations={
            "outside": mdp2.Termination(
                outside, inputs={"joint_pos": robot.joint_pos}, params={"bounds": mdp2.Tunable((-10.0, 10.0))}
            )
        },
    )
    num_worlds = 5  # Not the number of joints, so that a wrong broadcast cannot go unseen

    def run(backend: str, capture: bool) -> list[dict[str, torch.Tensor]]:
        scene = PointMassScene(num_worlds=num_worlds, joint_names=("a", "b", "c"), device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_task(task)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()
        mask = ProxyArray(wp.zeros(num_worlds, dtype=wp.bool, device=DEVICE))
        actions = ProxyArray(wp.zeros((num_worlds, 1), dtype=wp.float32, device=DEVICE))
        captured = mdp2.CapturedMDP(mdp, state, transition, actions, mask) if capture else None
        env_reset(mdp, state, scene, transition.observations, torch.ones(num_worlds, dtype=torch.bool, device=DEVICE))
        scene.data("robot", "joint_pos").torch.fill_(0.5)

        tunings = {
            1: [("rewards.scaled.gain", 2.0, [1, 3], "reset")],
            3: [("rewards.scaled.weight", torch.arange(1.0, 6.0, device=DEVICE), None, "step")],
            4: [("terminations.outside.bounds", (-1.0, 0.25), [4], "step")],
        }
        resets = {2: [1]}
        history = []
        for step in range(1, 6):
            if captured is None:
                mdp2.evaluate(mdp, state, transition)
            else:
                captured.evaluate()
            mask.torch.zero_()
            mask.torch[resets.get(step, [])] = True
            if captured is None:
                mdp2.reset(mdp, state, mask)
                mdp2.observe(mdp, state, transition.observations, mask)
            else:
                captured.reset()
                captured.observe()
            history.append(
                {
                    "rewards": transition.rewards.torch.clone(),
                    "terminated": transition.terminated.torch.clone(),
                    "policy": transition.observations["policy"].torch.clone(),
                    "revision": state.revision.torch.clone(),
                }
            )
            for path, value, worlds, at in tunings.get(step, []):
                mdp2.tune(mdp, state, path, value, worlds, at)
        return history

    runs = [run(backend, capture) for backend in ("torch", "warp") for capture in (False, True)]
    for other in runs[1:]:
        for expected, actual in zip(runs[0], other):
            for name in expected:
                assert torch.equal(expected[name], actual[name]), name

    history = runs[0]
    step_dt = 0.02
    rewards = [(step["rewards"] / (step_dt * 1.5)).tolist() for step in history]
    assert rewards[1] == pytest.approx([1.0] * 5)  # Tuned for the next reset: no change yet
    # World 1 reset after step 2 and took the gain of 2; world 3 still waits for its reset.
    assert rewards[2] == pytest.approx([1.0, 2.0, 1.0, 1.0, 1.0])
    assert history[2]["policy"][:, 0].tolist() == [1.0, 2.0, 1.0, 1.0, 1.0]
    assert rewards[3] == pytest.approx([1.0, 4.0, 3.0, 4.0, 5.0])  # Weights of 1 to 5 from the next step
    assert history[4]["terminated"].tolist() == [False] * 4 + [True]  # World 4's tighter bounds
    assert history[4]["revision"].tolist() == [1, 2, 1, 1, 2]

    builder = mdp2.MDPBuilder(PointMassScene(num_worlds=2).schema)
    builder.add_task(dataclasses.replace(task, observations={}))
    mdp = builder.finalize(device=DEVICE)
    with pytest.raises(KeyError, match="not tunable: declare it with mdp2.Tunable"):
        mdp2.tune(mdp, mdp.state(PointMassScene(num_worlds=2, device=DEVICE)), "rewards.scaled.std", 1.0)


def test_uses_with_when_run_where_their_field_held_at_the_start_of_the_step():
    """A use with ``when`` follows its field as it was at the start of the step, as rewards follow the task that
    governed the action: it outputs zero while off, and its state and history restart where it turns on. A reset
    prepares it, and a new episode starts with the field's value at the reset. A tunable ``when`` is a switch."""
    inputs = {"joint_pos": robot.joint_pos}
    position = mdp2.Observation(joint_pos, inputs=inputs, history_length=2, when=mdp2.quantities.a)
    task = mdp2.TaskCfg(
        quantities=above("a"),
        observations={"policy": {"position": position}},
        rewards={"gated": mdp2.Reward(accumulate, inputs=inputs, params={"gain": 1.0}, when=mdp2.quantities.a)},
        terminations={
            "switched": mdp2.Termination(
                mdp2.mdp.is_positive, inputs={"value": mdp2.quantities.a}, when=mdp2.Tunable(True)
            )
        },
    )
    flags = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]  # Joint "a" at each step; joint "b" is 0.25 throughout

    def run(backend: str, capture: bool) -> list[dict[str, torch.Tensor]]:
        scene = PointMassScene(num_worlds=1, joint_names=("b", "a"), device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_task(task)
        mdp = builder.finalize(device=DEVICE, backend=backend, diagnostics=True)
        state, transition = mdp.state(scene), mdp.transition()
        mask = ProxyArray(wp.ones(1, dtype=wp.bool, device=DEVICE))
        actions = ProxyArray(wp.zeros((1, 1), dtype=wp.float32, device=DEVICE))
        captured = mdp2.CapturedMDP(mdp, state, transition, actions, mask) if capture else None
        env_reset(mdp, state, scene, transition.observations, mask.torch)
        joints = scene.data("robot", "joint_pos").torch
        history = []
        for step, flag in enumerate(flags, start=1):
            joints.copy_(torch.tensor([[0.25, flag]], device=DEVICE))
            if step == 6:
                # A new episode, whose reset sees the joint "a" up.
                mdp2.reset(mdp, state, mask) if captured is None else captured.reset()
                mdp2.observe(mdp, state, transition.observations, mask) if captured is None else captured.observe()
                mdp2.tune(mdp, state, "terminations.switched.when", False, at="step")
            if captured is None:
                mdp2.evaluate(mdp, state, transition)
            else:
                captured.evaluate()
            total = state.arrays[mdp2.term_state("rewards.gated", "total")].torch
            history.append(
                {
                    "reward": transition.reward_terms.torch[0, 0].clone(),
                    "total": total.clone(),
                    "terminated": transition.terminated.torch.clone(),
                    "policy": transition.observations["policy"].torch.clone(),
                }
            )
        return history

    runs = [run(backend, capture) for backend in ("torch", "warp") for capture in (False, True)]
    for other in runs[1:]:
        for expected, actual in zip(runs[0], other):
            for name in expected:
                assert torch.equal(expected[name], actual[name]), name

    history = runs[0]
    # Off at the start, on from the step after "a" rose, off from the step after it fell, and restarted when back on.
    assert [round(step["reward"].item() / 0.02, 6) for step in history[:5]] == [0.0, 0.0, 0.25, 0.5, 0.0]
    assert [step["total"].item() for step in history[:5]] == [0.0, 0.0, 0.25, 0.5, 0.5]
    # The observation is zero while off, and its history restarts where it turns on: [b, a] per step of history.
    policy = [step["policy"][0].tolist() for step in history]
    assert policy[0] == [0.0] * 4 and policy[1] == [0.25, 1.0, 0.25, 1.0] and policy[3] == [0.0] * 4
    assert policy[2] == [0.25, 1.0, 0.25, 1.0]
    # The reset saw "a" rise: the new episode's first step runs, with the state the reset restarted.
    assert history[5]["total"].item() == 0.25
    # The termination ran while its switch was on, and stops from the step boundary after it was tuned off.
    assert [step["terminated"].item() for step in history] == [False, True, True, False, False, True, False]

    gate = mdp2.quantities.a
    reset_event = mdp2.Event(push_by_adding_velocity, outputs={"joint_vel": robot.joint_vel}, when=gate)
    errors = {
        "quantities.b": mdp2.TaskCfg(quantities={"b": dataclasses.replace(above("a")["a"], when=gate)}),
        "events.push": mdp2.TaskCfg(events={"push": reset_event}),
        "commands.c": mdp2.TaskCfg(commands={"c": mdp2.Command(uniform_command, when=mdp2.commands.c)}),
    }
    for name, part in errors.items():
        builder = mdp2.MDPBuilder(PointMassScene(num_worlds=1, joint_names=("a",)).schema)
        builder.add_task(mdp2.compose(mdp2.TaskCfg(quantities=above("a")), part))
        with pytest.raises(ValueError, match=f"{name}: (only rewards|when 'commands.c' is written by the use)"):
            builder.validate()


def test_tasks_that_repeat_count_each_time_they_are_done():
    """A task that repeats counts its completions, waits until its completion no longer holds, and then restarts: its
    uses restart, as when a task of a sequence starts, and its ``done`` stays true from the first time."""
    held = mdp2.Reward(accumulate, inputs={"joint_pos": robot.joint_pos}, params={"gain": 1.0})
    task = mdp2.compose(
        mdp2.TaskCfg(quantities=above("a")),
        drops=mdp2.TaskCfg(rewards={"held": held}, done=mdp2.quantities.a, repeat=True),
    )
    # Joint "a" rises at steps 2, 5, and 8; joint "b", which the reward sums, is 0.25 throughout.
    steps = [[0.25, a] for a in (0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0)]
    history = check_progress(task, steps, joints=("b", "a"))
    assert [step["drops.count"].item() for step in history] == [0, 1, 1, 1, 2, 2, 2, 3]
    assert history[-1]["drops.done_at"].item() == 2 and history[-1]["done_at"].item() == 2
    assert all(step["done"].item() for step in history[1:])
    # The reward's state restarts at the end of the steps where "a" fell after a completion.
    total = [step[str(mdp2.term_state("rewards.drops.held", "total"))].item() for step in history]
    assert total == [0.25, 0.5, 0.75, 0.0, 0.25, 0.0, 0.25, 0.5]

    # A sequence that repeats restarts its subtasks: each up-then-down cycle counts once.
    cycle = dataclasses.replace(
        mdp2.sequence(up=mdp2.Milestone(mdp2.quantities.a), down=mdp2.Milestone(mdp2.quantities.b)), repeat=True
    )
    task = mdp2.compose(mdp2.TaskCfg(quantities=above("a", "b")), cycle)
    history = check_progress(task, [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert [step["count"].item() for step in history] == [0, 1, 1, 1, 2]
    assert [step["up.done"].item() for step in history] == [True, True, False, True, True]
    assert [step["stage"].item() for step in history] == [1, 2, 0, 1, 2]


def test_tasks_of_a_sequence_take_their_own_actions():
    """Each task of a sequence writes its actions while it runs. Actions of one name share their columns, so a policy
    keeps its outputs across tasks, and the last actions zero the columns of a task that does not run."""

    def stage(joint: str, actions: dict[str, mdp2.DirectAction]) -> mdp2.TaskCfg:
        return mdp2.TaskCfg(actions=actions, done=getattr(mdp2.quantities, joint))

    effort = mdp2.DirectAction(robot.joint_effort_target, subset=robot.joints("a"))
    grip = mdp2.DirectAction(robot.joint_effort_target, subset=robot.joints("b"))
    task = mdp2.compose(
        mdp2.TaskCfg(quantities=above("a", "b")),
        mdp2.sequence(
            reach=stage("a", {"effort": effort}),
            lift=stage("b", {"effort": dataclasses.replace(effort, scale=3.0), "grip": grip}),
        ),
    )

    def run(backend: str, capture: bool) -> list[dict[str, torch.Tensor]]:
        scene = PointMassScene(num_worlds=1, joint_names=("a", "b"), device=DEVICE)
        builder = mdp2.MDPBuilder(scene.schema)
        builder.add_task(task)
        mdp = builder.finalize(device=DEVICE, backend=backend)
        state, transition = mdp.state(scene), mdp.transition()
        actions = ProxyArray(wp.from_torch(torch.tensor([[0.5, 0.25]], device=DEVICE)))
        captured = mdp2.CapturedMDP(mdp, state, transition, actions, transition.done) if capture else None
        env_reset(mdp, state, scene, transition.observations, torch.ones(1, dtype=torch.bool, device=DEVICE))
        spec = mdp.spec().actions
        assert [(columns.name, columns.columns) for columns in spec] == [("effort", slice(0, 1)), ("grip", slice(1, 2))]
        assert dict(spec[0].labels) == {"actions.reach.effort": ("a",), "actions.lift.effort": ("a",)}
        assert dict(spec[1].labels) == {"actions.lift.grip": ("b",)}
        history = []
        for joints in ([0.0, 0.0], [1.0, 0.0], [1.0, 0.0]):
            mdp2.apply_actions(mdp, state, actions) if captured is None else captured.apply_actions()
            scene.data("robot", "joint_pos").torch.copy_(torch.tensor([joints], device=DEVICE))
            mdp2.evaluate(mdp, state, transition) if captured is None else captured.evaluate()
            target = scene.data("robot", "joint_effort_target").torch
            history.append({"target": target.clone(), "last": state.actions.torch.clone()})
            history[-1]["mask"] = mdp.action_mask(state).clone()
        return history

    runs = [run(backend, capture) for backend in ("torch", "warp") for capture in (False, True)]
    for other in runs[1:]:
        for expected, actual in zip(runs[0], other):
            for name in expected:
                assert torch.equal(expected[name], actual[name]), name
    history = runs[0]
    # Reach writes the effort until it is done at step 2; lift scales it by 3 and grips from step 3.
    assert [step["target"][0].tolist() for step in history] == [[0.5, 0.0], [0.5, 0.0], [1.5, 0.25]]
    assert [step["last"][0].tolist() for step in history] == [[0.5, 0.0], [0.5, 0.0], [0.5, 0.25]]
    # The mask is of the next step: lift runs from step 3.
    assert [step["mask"][0].tolist() for step in history] == [[True, False], [True, True], [True, True]]


@wp.func
def count_applications_warp(
    world: int, action: wp.array2d(dtype=float), target: wp.array2d(dtype=float), calls: wp.array(dtype=float)
):
    calls[world] = calls[world] + 1.0
    target[world, 0] = action[world, 0] * calls[world]


def count_applications_torch(action, target, calls):
    calls.add_(1.0)
    target[:, 0] = action[:, 0] * calls


count_applications = mdp2.Term(
    torch=count_applications_torch,
    warp=count_applications_warp,
    shape=("worlds", 1),
    state={"calls": mdp2.State(persistent=True)},
)
"""Write the action's column times the number of times the action applied."""


def test_actions_apply_once_per_control_step_or_before_every_physics_step():
    """An action at the control rate applies once per control step, before the physics steps, and one at the physics
    rate before each of them. The actions are stored once."""
    task = mdp2.TaskCfg(
        actions={
            "plan": mdp2.Action(
                count_applications, outputs={"target": robot.joint_effort_target}, rate=mdp2.ActionRate.CONTROL
            ),
            "track": mdp2.DirectAction(robot.joint_effort_target, subset=robot.joints("b"), scale=2.0),
        }
    )
    for backend in ("torch", "warp"):
        for capture in (False, True):
            scene = PointMassScene(num_worlds=2, joint_names=("a", "b"), device=DEVICE)
            builder = mdp2.MDPBuilder(scene.schema)
            builder.add_task(task)
            mdp = builder.finalize(device=DEVICE, backend=backend)
            state, transition = mdp.state(scene), mdp.transition()
            actions = ProxyArray(wp.from_torch(torch.tensor([[1.0, 0.5], [2.0, 0.25]], device=DEVICE)))
            captured = mdp2.CapturedMDP(mdp, state, transition, actions, transition.done) if capture else None
            for _ in range(3):
                mdp2.process_actions(mdp, state, actions) if captured is None else captured.process_actions()
                for _ in range(scene.decimation):
                    mdp2.apply_actions(mdp, state, actions) if captured is None else captured.apply_actions()
            calls = state.arrays[mdp2.term_state("actions.plan", "calls")].torch
            assert calls.tolist() == [3.0, 3.0], (backend, capture)
            target = scene.data("robot", "joint_effort_target").torch
            assert target.tolist() == [[3.0, 1.0], [6.0, 0.5]] and torch.equal(state.actions.torch, actions.torch)
            assert len(mdp.launches.process_actions) == 1 and len(mdp.plan.apply_actions) == 1


def test_choice_draws_one_subtask_per_episode_and_world():
    """A choice draws one subtask per world at each reset, by its weights: only that subtask's uses run, the task is
    done once it is, and its ``stage`` is the drawn subtask."""

    def skill(joint: str, reward: float) -> mdp2.TaskCfg:
        alive = mdp2.Reward(is_alive, inputs={"joint_pos": robot.joint_pos}, weight=reward)
        return mdp2.TaskCfg(rewards={"alive": alive}, done=getattr(mdp2.quantities, joint))

    task = mdp2.compose(
        mdp2.TaskCfg(quantities=above("a", "b")),
        skills=mdp2.choice(weights={"left": 3.0, "right": 1.0}, left=skill("a", 1.0), right=skill("b", 2.0)),
    )
    runs = []
    for backend in ("torch", "warp"):
        for capture in (False, True) if torch.cuda.is_available() else (False,):
            runs.append(run_progress(task, [[1.0, 0.0]], backend, capture, num_worlds=256)[2][0])
    for run in runs[1:]:
        for name, value in runs[0].items():
            assert torch.equal(value, run[name]), name

    step = runs[0]
    left = step["skills.stage"] == 0
    assert 0.65 < left.float().mean().item() < 0.85
    assert torch.equal(step["skills.left.active"], left) and torch.equal(step["skills.right.active"], ~left)
    assert torch.equal(step["done"], left)  # Joint "a" is up, "b" is down
    torch.testing.assert_close(step["rewards"] / 0.02, torch.where(left, 1.0, 2.0))


def test_parts_renamed_to_other_entities_run_together_in_columns_of_their_own():
    """One part, renamed to act on two robots, runs twice as subtasks together: each copy's actions and observations
    get columns of their own, named with the subtask's name."""
    arm = mdp2.TaskCfg(
        actions={"effort": mdp2.DirectAction(robot.joint_effort_target)},
        observations={"policy": {"position": mdp2.Observation(joint_pos, inputs={"joint_pos": robot.joint_pos})}},
    )
    task = mdp2.compose(left=mdp2.rename(arm, robot="left"), right=mdp2.rename(arm, robot="right"))
    scene = PointMassScene(num_worlds=2, joint_names=("a", "b"), device=DEVICE, entities=("left", "right"))
    builder = mdp2.MDPBuilder(scene.schema)
    builder.add_task(task)
    mdp = builder.finalize(device=DEVICE, backend="warp")
    spec = mdp.spec()
    assert [(columns.name, columns.columns) for columns in spec.actions] == [
        ("left.effort", slice(0, 2)),
        ("right.effort", slice(2, 4)),
    ]
    assert [columns.name for columns in spec.observations["policy"]] == ["left.position", "right.position"]

    state = mdp.state(scene)
    actions = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 2, device=DEVICE)
    mdp2.apply_actions(mdp, state, ProxyArray(wp.from_torch(actions)))
    assert scene.data("left", "joint_effort_target").torch.tolist() == [[1.0, 2.0]] * 2
    assert scene.data("right", "joint_effort_target").torch.tolist() == [[3.0, 4.0]] * 2
