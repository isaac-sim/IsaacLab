# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp backend: each runtime function runs as a few generated kernels, see :class:`.fusion.Fusion`.

One thread per world runs a function's scalar work in schedule order: terminations, rewards, timers, commands,
events, state updates, and resets. Observation terms run one thread per output element, in the same kernel
where their inputs allow. ``wp.func`` hooks are called from these threads. Hooks that launch kernels themselves
run as their own steps, and compute into scratch buffers that the next kernel reads like a func's result.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from .fusion import Fusion
from .mdp import MDP, MDPState, TermOp, Transition


@dataclass
class Call:
    """Arguments of one call to a runtime function, which the kernel arguments are read from."""

    state: MDPState
    actions: wp.array | None = None
    mask: wp.array | None = None
    transition: Transition | None = None
    observations: dict[str, ProxyArray] | None = None


@dataclass(frozen=True)
class Program:
    """Steps of every runtime function: generated kernels, and the hooks that launch their own."""

    apply_actions: tuple[Callable[[Call], None], ...]
    evaluate: tuple[Callable[[Call], None], ...]
    reset: tuple[Callable[[Call], None], ...]
    observe: tuple[Callable[[Call], None], ...]


@wp.func
def _fill(world: int, values: wp.array2d(dtype=wp.float32), value: float):
    for i in range(values.shape[1]):
        values[world, i] = value


@wp.func
def _advance(world: int, rng: wp.array2d(dtype=wp.int32)):
    for stream in range(rng.shape[0]):
        rng[stream, world] = wp.int32(wp.rand_init(rng[stream, world]))


@wp.func
def _restart(world: int, low: float, high: float, seed: wp.int32, time_left: wp.array(dtype=wp.float32)):
    time_left[world] = wp.randf(wp.rand_init(seed, -1), low, high)


@wp.func
def _tick(
    world: int, low: float, high: float, seed: wp.int32, time_left: wp.array(dtype=wp.float32), step_dt: float
) -> bool:
    """Count down a timer and restart it where it expired. Returns whether it expired."""
    time_left[world] = time_left[world] - step_dt
    if time_left[world] > 0.0:
        return False
    _restart(world, low, high, seed, time_left)
    return True


def compile_program(mdp: MDP) -> Program:
    return Program(_apply_actions(mdp), _evaluate(mdp), _reset(mdp), _observe(mdp))


def apply_actions(mdp: MDP, state: MDPState, actions: torch.Tensor) -> None:
    call = Call(state, actions=wp.from_torch(actions))
    for step in mdp.program.apply_actions:
        step(call)


def evaluate(mdp: MDP, state: MDPState, transition: Transition) -> None:
    call = Call(state, transition=transition, observations=transition.observations)
    for step in mdp.program.evaluate:
        step(call)


def reset(mdp: MDP, state: MDPState, mask: torch.Tensor) -> None:
    call = Call(state, mask=wp.from_torch(mask))
    for step in mdp.program.reset:
        step(call)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray], mask: torch.Tensor) -> None:
    call = Call(state, mask=wp.from_torch(mask), observations=observations)
    for step in mdp.program.observe:
        step(call)


def _apply_actions(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("apply_actions", mdp.num_worlds, mdp.device)
    actions = code.argument("actions", wp.array2d(dtype=wp.float32), lambda call: call.actions)
    last_actions = code.argument("last_actions", wp.array2d(dtype=wp.float32), lambda call: call.state.actions.warp)
    code.elements((mdp.action_dim,), f"{last_actions}[world, element] = {actions}[world, element]")
    for op in mdp.action_ops:
        target = code.argument(
            op.target,
            wp.array2d(dtype=wp.float32),
            lambda call, op=op: call.state.fields[op.target].warp.reshape((mdp.num_worlds, -1)),
        )
        value = f"{actions}[world, {op.columns.start} + element] * {code.constant(float, op.scale)}"
        code.elements(
            (op.columns.stop - op.columns.start,),
            f"{target}[world, element] = {value} + {code.constant(float, op.offset)}",
        )
    return code.steps()


def _evaluate(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("evaluate", mdp.num_worlds, mdp.device)
    episode_length = code.argument(
        "episode_length", wp.array(dtype=wp.int32), lambda call: call.state.episode_length.warp
    )
    terminated = code.argument("terminated", wp.array(dtype=wp.bool), lambda call: call.transition.terminated.warp)
    truncated = code.argument("truncated", wp.array(dtype=wp.bool), lambda call: call.transition.truncated.warp)
    rewards = code.argument("rewards", wp.array(dtype=wp.float32), lambda call: call.transition.rewards.warp)
    time_out = "False"
    if mdp.max_episode_length is not None:
        time_out = f"{episode_length}[world] >= {code.constant(int, mdp.max_episode_length)}"
    code.world(
        f"{code.function(_advance)}(world, {_rng(code)})",
        f"{episode_length}[world] = {episode_length}[world] + 1",
        f"{terminated}[world] = False",
        f"{truncated}[world] = {time_out}",
        f"{rewards}[world] = 0.0",
        writes=["rng"],
    )
    for op in mdp.termination_ops:
        code.world(*_nest(f"if {_value(code, op)}:", f"{truncated if op.time_out else terminated}[world] = True"))
        _update(code, op)
    for op in mdp.reward_ops:
        scale = code.constant(float, op.weight * mdp.step_dt)
        code.world(f"{rewards}[world] = {rewards}[world] + {scale} * {_value(code, op)}")
        _update(code, op)
    for op in mdp.command_ops:
        if op.timer is not None:
            _resample(code, op, [f"if {_due(code, mdp, op)}:"])
    for op in mdp.interval_event_ops:
        due = _due(code, mdp, op)
        _run(code, op, op.compute, op.inputs.values(), [f"if {due}:"], lambda call, op=op: call.state.due[op.name])
    _observations(code, mdp)
    for op in mdp.observation_ops:
        _update(code, op)
    return code.steps()


def _reset(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("reset", mdp.num_worlds, mdp.device, _mask)
    episode_length = code.argument(
        "episode_length", wp.array(dtype=wp.int32), lambda call: call.state.episode_length.warp
    )
    last_actions = code.argument("last_actions", wp.array2d(dtype=wp.float32), lambda call: call.state.actions.warp)
    code.world(
        f"{code.function(_advance)}(world, {_rng(code)})",
        f"{episode_length}[world] = 0",
        f"{code.function(_fill)}(world, {last_actions}, 0.0)",
    )
    for op in mdp.term_ops:
        if op.reset is not None:
            _run(code, op, op.reset, [(op.name, name) for name in op.state], mask=_mask)
            continue
        for name, spec in op.state.items():
            values = code.argument(
                (op.name, name),
                wp.array2d(dtype=wp.float32),
                lambda call, op=op, name=name: call.state.term_state[op.name][name].warp.reshape((mdp.num_worlds, -1)),
            )
            code.world(f"{code.function(_fill)}(world, {values}, {code.constant(float, spec.initial)})")
    for op in mdp.interval_event_ops + mdp.command_ops:
        if op.timer is not None:
            code.world(f"{code.function(_restart)}({_timer(code, op)})")
    for op in mdp.reset_event_ops:
        _run(code, op, op.compute, op.inputs.values(), mask=_mask)
    for op in mdp.command_ops:
        _resample(code, op)
    return code.steps()


def _observe(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("observe", mdp.num_worlds, mdp.device, _mask)
    _observations(code, mdp)
    return code.steps()


def _observations(code: Fusion, mdp: MDP) -> None:
    """Write every observation term into its columns of the call's observations, one thread per element."""
    for op in mdp.observation_ops:
        out = code.argument(
            ("observations", op.group),
            wp.array2d(dtype=wp.float32),
            lambda call, op=op: call.observations[op.group].warp,
        )
        code.elements(op.shape, f"{out}[world, {op.columns.start} + element] = {_value(code, op)}")


def _resample(code: Fusion, op: TermOp, conditions: Sequence[str] = ()) -> None:
    """Write fresh samples of a command in the world thread, under ``conditions``."""
    target = code.argument(
        op.target,
        wp.array(dtype=wp.float32, ndim=1 + len(op.shape)),
        lambda call: call.state.fields[op.target].warp,
    )
    value = _value(code, op)
    loops = [f"for i{d} in range({n}):" for d, n in enumerate(op.shape)]
    code.world(*_nest(*conditions, *loops, f"{target}[{_index(op.shape)}] = {value}"), writes=[op.target])


def _update(code: Fusion, op: TermOp) -> None:
    """Advance an op's state, after its compute."""
    if op.update is not None:
        _run(code, op, op.update, [(op.name, name) for name in op.state])


def _run(
    code: Fusion,
    op: TermOp,
    hook: Callable,
    writes: Iterable[Hashable],
    conditions: Sequence[str] = (),
    mask: Callable[[Call], wp.array] | None = None,
) -> None:
    """Run a hook that has no output.

    A func runs in the world thread under ``conditions``, and ``writes`` are the keys of the arguments it writes.
    A function that launches its own kernels runs as a step, and takes ``mask(call)`` if it has a mask.
    """
    if isinstance(hook, wp.Function):
        code.world(*_nest(*conditions, _call(code, op, hook)), writes=writes)
    elif mask is None:
        code.call(lambda call: hook(**_arguments(op, call.state)))
    else:
        code.call(lambda call: hook(mask=mask(call), **_arguments(op, call.state)))


def _value(code: Fusion, op: TermOp) -> str:
    """Source of an op's output for the thread's world and element.

    A compute function that launches its own kernels first runs as a step, and the source reads its scratch.
    """
    if isinstance(op.compute, wp.Function):
        return _call(code, op, op.compute, op.shape)
    code.call(lambda call: op.compute(out=call.state.scratch[op.name], **_arguments(op, call.state)))
    scratch = code.argument(
        ("scratch", op.name),
        wp.array(dtype=op.dtype, ndim=1 + len(op.shape)),
        lambda call: call.state.scratch[op.name],
    )
    return f"{scratch}[{_index(op.shape)}]"


def _call(code: Fusion, op: TermOp, hook: wp.Function, shape: tuple[int, ...] = ()) -> str:
    """Source of a call to a func hook for the thread's world and, with an output ``shape``, its element."""
    values = [_index(shape)]
    for name, annotation in list(hook.input_types.items())[1 + len(shape) :]:
        if name in op.inputs:
            selector = op.inputs[name]
            get = lambda call, selector=selector: call.state.fields[selector].warp  # noqa: E731
            values.append(code.argument(selector, annotation, get))
        elif name in op.state:
            get = lambda call, name=name: call.state.term_state[op.name][name].warp  # noqa: E731
            values.append(code.argument((op.name, name), annotation, get))
        elif name in op.params:
            values.append(code.constant(annotation, op.params[name]))
        else:
            values.append(f"{_rng(code)}[{op.stream}, world]")
    return f"{code.function(hook)}({', '.join(values)})"


def _due(code: Fusion, mdp: MDP, op: TermOp) -> str:
    """Source that counts down an op's timer, true where it expired. Functions get the result as a mask."""
    tick = f"{code.function(_tick)}({_timer(code, op)}, {code.constant(float, mdp.step_dt)})"
    if isinstance(op.compute, wp.Function):
        return tick
    due = code.argument(("due", op.name), wp.array(dtype=wp.bool), lambda call: call.state.due[op.name])
    code.world(f"{due}[world] = {tick}")
    return f"{due}[world]"


def _timer(code: Fusion, op: TermOp) -> str:
    """Source of the arguments that :func:`_restart` and :func:`_tick` share."""
    low, high = (code.constant(float, value) for value in op.timer)
    time_left = code.argument(
        ("timer", op.name), wp.array(dtype=wp.float32), lambda call: call.state.timers[op.name].warp
    )
    return f"world, {low}, {high}, {_rng(code)}[{op.stream}, world], {time_left}"


def _rng(code: Fusion) -> str:
    return code.argument("rng", wp.array2d(dtype=wp.int32), lambda call: call.state.rng.warp)


def _mask(call: Call) -> wp.array:
    return call.mask


def _index(shape: tuple[int, ...]) -> str:
    """Source of the indices of the thread's world and element of ``shape``."""
    return ", ".join(["world", *(f"i{d}" for d in range(len(shape)))])


def _nest(*lines: str) -> list[str]:
    """Indent each line one level deeper than the line before."""
    return [f"{'    ' * depth}{line}" for depth, line in enumerate(lines)]


def _arguments(op: TermOp, state: MDPState) -> dict[str, Any]:
    inputs = {arg: state.fields[selector].warp for arg, selector in op.inputs.items()}
    term_state = {name: value.warp for name, value in state.term_state[op.name].items()}
    if op.random:
        term_state["rng"] = state.rng.warp[op.stream]
    return {**inputs, **term_state, **op.params}
