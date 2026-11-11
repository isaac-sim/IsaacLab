# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp backend: each runtime function runs as a few generated kernels, see :class:`.fusion.Fusion`.

One thread per world runs a function's per-world steps in order: terminations, rewards, timers, commands, events,
state updates, and resets. Observations run one thread per output element, in the same kernel when their inputs
allow it. ``wp.func`` hooks are called from these threads. Hooks that launch their own kernels run separately and
write into scratch buffers, which the next kernel reads like a func's result.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence

import warp as wp

from .config import Phase, episode_length, last_action
from .eager import call_eager
from .fusion import Fusion
from .mdp import MDP
from .plan import Call, Launches, Step, StepKind, TermOp


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


@wp.kernel
def _select(mask: wp.array(dtype=wp.bool), active: wp.array(dtype=wp.bool), selected: wp.array(dtype=wp.bool)):
    world = wp.tid()
    selected[world] = mask[world] and active[world]


_RESULTS = {
    "rewards": (wp.array(dtype=wp.float32), lambda call: call.transition.rewards.warp),
    "terminated": (wp.array(dtype=wp.bool), lambda call: call.transition.terminated.warp),
    "truncated": (wp.array(dtype=wp.bool), lambda call: call.transition.truncated.warp),
    "done": (wp.array(dtype=wp.bool), lambda call: call.transition.done.warp),
    "reward_terms": (wp.array2d(dtype=wp.float32), lambda call: call.transition.reward_terms.warp),
    "termination_terms": (wp.array2d(dtype=wp.bool), lambda call: call.transition.termination_terms.warp),
}
"""Kernel arguments read from the transition of a call, by key."""


def compile_launches(mdp: MDP) -> Launches:
    """Generate the kernels of every runtime function, with the hooks that launch their own work between them."""
    return Launches(_apply_actions(mdp), _evaluate(mdp), _reset(mdp), _observe(mdp))


def _apply_actions(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("apply_actions", mdp.num_worlds, mdp.device)
    actions = code.argument("actions", wp.array2d(dtype=wp.float32), lambda call: call.actions.warp)
    last_actions = _field(code, last_action, wp.array2d(dtype=wp.float32))
    code.elements(
        (mdp.action_dim,), f"{last_actions}[world, element] = {actions}[world, element]", writes=[last_action]
    )
    for step in mdp.plan.apply_actions:
        op = step.op
        target = code.argument(
            op.target,
            wp.array2d(dtype=wp.float32),
            lambda call, op=op: call.state.fields[op.target].warp.reshape((mdp.num_worlds, -1)),
        )
        value = f"{actions}[world, {op.columns.start} + element] * {code.constant(float, op.scale)}"
        code.elements(
            (op.columns.stop - op.columns.start,),
            *_under(_active(code, op), f"{target}[world, element] = {value} + {code.constant(float, op.offset)}"),
            writes=[op.target],
        )
    return code.steps()


def _evaluate(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("evaluate", mdp.num_worlds, mdp.device)
    length = _field(code, episode_length, wp.array(dtype=wp.int32))
    terminated, truncated, done, rewards = (
        _result(code, key) for key in ("terminated", "truncated", "done", "rewards")
    )
    time_out = "False"
    if mdp.max_episode_length is not None:
        time_out = f"{length}[world] >= {code.constant(int, mdp.max_episode_length)}"
    code.world(
        f"{code.function(_advance)}(world, {_rng(code)})",
        f"{length}[world] = {length}[world] + 1",
        f"{terminated}[world] = False",
        f"{truncated}[world] = {time_out}",
        f"{done}[world] = {truncated}[world]",
        f"{rewards}[world] = 0.0",
        writes=["rng", episode_length, "terminated", "truncated", "done", "rewards"],
    )
    _emit(code, mdp, mdp.plan.evaluate)
    return code.steps()


def _reset(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("reset", mdp.num_worlds, mdp.device, _mask)
    length = _field(code, episode_length, wp.array(dtype=wp.int32))
    last_actions = _field(code, last_action, wp.array2d(dtype=wp.float32))
    code.world(
        f"{code.function(_advance)}(world, {_rng(code)})",
        f"{length}[world] = 0",
        f"{code.function(_fill)}(world, {last_actions}, 0.0)",
        writes=["rng", episode_length, last_action],
    )
    _emit(code, mdp, mdp.plan.reset)
    return code.steps()


def _observe(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("observe", mdp.num_worlds, mdp.device, _mask)
    _emit(code, mdp, mdp.plan.observe)
    return code.steps()


def _emit(code: Fusion, mdp: MDP, steps: Iterable[Step]) -> None:
    """Add the statements for the steps of a runtime function."""
    for step in steps:
        op = step.op
        state = [(op.name, name) for name in op.state]
        active = _active(code, op)
        match step.kind:
            case StepKind.COMPUTE:
                _compute(code, mdp, op, active)
            case StepKind.COMPUTE_IF_DUE:
                _compute(code, mdp, op, [*active, f"if {_due(code, mdp, op, active)}:"])
            case StepKind.UPDATE_STATE:
                _run(code, mdp, op, op.update, state, active)
            case StepKind.RESET_TIMER:
                restart = f"{code.function(_restart)}({_timer(code, op)})"
                code.world(*_under(active, restart), writes=[("timer", op.name)])
            case StepKind.RESET_STATE if op.reset is not None:
                _run(code, mdp, op, op.reset, state, active, mask=_mask)
            case StepKind.RESET_STATE:
                for name, spec in op.state.items():
                    values = code.argument(
                        (op.name, name),
                        wp.array2d(dtype=wp.float32),
                        lambda call, op=op, name=name: call.state.arguments[op.name][name].reshape(
                            (mdp.num_worlds, -1)
                        ),
                    )
                    fill = f"{code.function(_fill)}(world, {values}, {code.constant(float, spec.initial)})"
                    code.world(*_under(active, fill), writes=[(op.name, name)])


def _compute(code: Fusion, mdp: MDP, op: TermOp, conditions: Sequence[str] = ()) -> None:
    """Compute an op into its destination, nested under ``conditions``."""
    if op.worlds is not None and op.phase is Phase.RESET_EVENTS and not isinstance(op.compute, wp.Function):
        # The hook gets the reset worlds where the op is active.
        selected = lambda call: _selected(mdp, op, call)  # noqa: E731
        _run(code, mdp, op, op.compute, op.writes, mask=selected)
        return
    match op.phase:
        case Phase.TERMINATIONS:
            writes = ["truncated" if op.time_out else "terminated", "done"]
            lines = [f"ended = {_value(code, mdp, op)}"]
            lines += _under(["if ended:"], *(f"{_result(code, key)}[world] = True" for key in writes))
            if mdp.diagnostics:
                lines.append(f"{_result(code, 'termination_terms')}[world, {op.index}] = ended")
                writes.append("termination_terms")
            code.world(*_under(conditions, *lines), writes=writes)
        case Phase.REWARDS:
            weights = code.argument(
                "reward_weights", wp.array(dtype=wp.float32), lambda call: call.state.reward_weights.warp
            )
            scale = f"{weights}[{op.index}] * {code.constant(float, mdp.step_dt)}"
            rewards = _result(code, "rewards")
            lines = [f"reward = {scale} * {_value(code, mdp, op)}", f"{rewards}[world] = {rewards}[world] + reward"]
            writes = ["rewards"]
            if mdp.diagnostics:
                lines.append(f"{_result(code, 'reward_terms')}[world, {op.index}] = reward")
                writes.append("reward_terms")
            code.world(*_under(conditions, *lines), writes=writes)
        case Phase.COMMANDS:
            target = _field(code, op.writes[0], wp.array(dtype=wp.float32, ndim=1 + len(op.shape)))
            value = _value(code, mdp, op)
            loops = [f"for i{d} in range({n}):" for d, n in enumerate(op.shape)]
            code.world(*_under([*conditions, *loops], f"{target}[{_index(op.shape)}] = {value}"), writes=op.writes)
        case Phase.INTERVAL_EVENTS:
            _run(code, mdp, op, op.compute, op.writes, conditions, lambda call: call.state.selected[op.name])
        case Phase.RESET_EVENTS:
            _run(code, mdp, op, op.compute, op.writes, conditions, mask=_mask)
        case Phase.OBSERVATIONS:
            out = code.argument(
                ("observations", op.group),
                wp.array2d(dtype=wp.float32),
                lambda call: call.observations[op.group].warp,
            )
            line = f"{out}[world, {op.columns.start} + element] = {_value(code, mdp, op)}"
            code.elements(op.shape, *_under(conditions, line))


def _run(
    code: Fusion,
    mdp: MDP,
    op: TermOp,
    hook: Callable,
    writes: Iterable[Hashable],
    conditions: Sequence[str] = (),
    mask: Callable[[Call], wp.array] | None = None,
) -> None:
    """Run a hook that has no output.

    A func runs in the world thread under ``conditions``, and ``writes`` are the keys of the arguments it writes.
    A function that launches its own kernels, or an eager term, runs on its own, and gets ``mask(call)`` if given.
    """
    if isinstance(hook, wp.Function):
        code.world(*_under(conditions, _call(code, op, hook)), writes=writes)
    elif op.eager:
        code.call(lambda call: call_eager(mdp, op, call.state, wp.to_torch(mask(call))), False, *_accesses(op))
    elif mask is None:
        code.call(lambda call: hook(**call.state.arguments[op.name]), op.capturable, *_accesses(op))
    else:
        code.call(lambda call: hook(mask=mask(call), **call.state.arguments[op.name]), op.capturable, *_accesses(op))


def _value(code: Fusion, mdp: MDP, op: TermOp) -> str:
    """Return source for an op's output at the thread's world and element.

    A compute hook that launches its own kernels, or an eager term, runs first, and the source reads its scratch
    buffer.
    """
    if isinstance(op.compute, wp.Function):
        return _call(code, op, op.compute, op.shape)
    if op.eager:
        code.call(lambda call: call_eager(mdp, op, call.state), False, *_accesses(op))
    else:
        code.call(
            lambda call: op.compute(out=call.state.scratch[op.name].warp, **call.state.arguments[op.name]),
            op.capturable,
            *_accesses(op),
        )
    scratch = code.argument(
        ("scratch", op.name),
        wp.array(dtype=op.dtype, ndim=1 + len(op.shape)),
        lambda call: call.state.scratch[op.name].warp,
    )
    return f"{scratch}[{_index(op.shape)}]"


def _call(code: Fusion, op: TermOp, hook: wp.Function, shape: tuple[int, ...] = ()) -> str:
    """Return source that calls a func hook for the thread's world and, given an output ``shape``, its element."""
    values = [_index(shape)]
    for name, annotation in list(hook.input_types.items())[1 + len(shape) :]:
        if name in op.params:
            values.append(code.constant(annotation, op.params[name]))
        elif name in op.inputs or name in op.state:
            get = lambda call, name=name: call.state.arguments[op.name][name]  # noqa: E731
            values.append(code.argument(op.inputs.get(name, (op.name, name)), annotation, get))
        else:
            values.append(f"{_rng(code)}[{op.stream}, world]")
    return f"{code.function(hook)}({', '.join(values)})"


def _due(code: Fusion, mdp: MDP, op: TermOp, active: Sequence[str]) -> str:
    """Return source that counts down an op's timer and is true where it expired. Function hooks get a mask."""
    tick = f"{code.function(_tick)}({_timer(code, op)}, {code.constant(float, mdp.step_dt)})"
    if isinstance(op.compute, wp.Function):
        return tick
    due = code.argument(("selected", op.name), wp.array(dtype=wp.bool), lambda call: call.state.selected[op.name])
    code.world(*_under(active, f"{due}[world] = {tick}"), writes=[("timer", op.name), ("selected", op.name)])
    return f"{due}[world]"


def _active(code: Fusion, op: TermOp) -> list[str]:
    """Return the condition under which an op's statements run, if it is active in only some worlds."""
    if op.worlds is None:
        return []
    active = code.argument(("active", op.name), wp.array(dtype=wp.bool), lambda call: call.state.active[op.name].warp)
    return [f"if {active}[world]:"]


def _selected(mdp: MDP, op: TermOp, call: Call) -> wp.array:
    """Write the reset worlds where an op is active to its selected buffer, and return the buffer."""
    selected = call.state.selected[op.name]
    inputs = [call.mask.warp, call.state.active[op.name].warp]
    wp.launch(_select, dim=mdp.num_worlds, inputs=inputs, outputs=[selected], device=mdp.device)
    return selected


def _timer(code: Fusion, op: TermOp) -> str:
    """Return source for the arguments shared by :func:`_restart` and :func:`_tick`."""
    low, high = (code.constant(float, value) for value in op.timer)
    time_left = code.argument(
        ("timer", op.name), wp.array(dtype=wp.float32), lambda call: call.state.timers[op.name].warp
    )
    return f"world, {low}, {high}, {_rng(code)}[{op.stream}, world], {time_left}"


def _field(code: Fusion, selector: Hashable, annotation: wp.array) -> str:
    return code.argument(selector, annotation, lambda call: call.state.fields[selector].warp)


def _result(code: Fusion, key: str) -> str:
    return code.argument(key, *_RESULTS[key])


def _accesses(op: TermOp) -> tuple[set[Hashable], set[Hashable]]:
    """Keys of the kernel arguments that a hook of an op may read and write, for :meth:`Fusion.call`."""
    state = {(op.name, name) for name in op.state}
    reads = {*op.inputs.values(), *state, ("selected", op.name), *(["rng"] if op.random else [])}
    return reads, {*op.writes, *state, ("scratch", op.name)}


def _rng(code: Fusion) -> str:
    return code.argument("rng", wp.array2d(dtype=wp.int32), lambda call: call.state.rng.warp)


def _mask(call: Call) -> wp.array:
    return call.mask.warp


def _index(shape: tuple[int, ...]) -> str:
    """Return source for the indices of the thread's world and element of ``shape``."""
    return ", ".join(["world", *(f"i{d}" for d in range(len(shape)))])


def _under(conditions: Sequence[str], *lines: str) -> list[str]:
    """Indent ``lines`` under ``conditions``, nesting each condition one level deeper than the previous one."""
    nested = [f"{'    ' * depth}{condition}" for depth, condition in enumerate(conditions)]
    return nested + [f"{'    ' * len(conditions)}{line}" for line in lines]
