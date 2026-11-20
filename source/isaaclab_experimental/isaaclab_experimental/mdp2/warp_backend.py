# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp backend: each runtime function runs as a few generated kernels, see :class:`.fusion.Fusion`.

One thread per world runs a function's per-world steps in order: terminations, rewards, timers, commands, events,
state updates, and resets. Observations run one thread per output element, in the same kernel when their inputs
allow it. ``wp.func`` hooks are called from these threads. Hooks that launch their own kernels run separately and
write into scratch buffers, which the next kernel reads like a func's result. Terms without Warp hooks run their
Torch hooks the same way.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Hashable, Iterable, Sequence

import torch
import warp as wp

from .compiled import MDP
from .config import (
    Entity,
    FieldSelector,
    Phase,
    TaskMode,
    episode_length,
    last_action,
    progress_field,
    term_state,
    terminated,
)
from .fusion import Fusion
from .plan import Call, Launches, ObservationSlot, Step, StepKind, TaskNode, TaskTree, TermOp
from .terms import hook_arguments

_RESULTS = {
    "rewards": (wp.array(dtype=wp.float32), lambda call: call.transition.rewards.warp),
    "terminated": (wp.array(dtype=wp.bool), lambda call: call.transition.terminated.warp),
    "truncated": (wp.array(dtype=wp.bool), lambda call: call.transition.truncated.warp),
    "done": (wp.array(dtype=wp.bool), lambda call: call.transition.done.warp),
    "reward_terms": (wp.array2d(dtype=wp.float32), lambda call: call.transition.reward_terms.warp),
    "termination_terms": (wp.array2d(dtype=wp.bool), lambda call: call.transition.termination_terms.warp),
}
"""Kernel arguments read from the transition of a call, by key."""


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


def compile_launches(mdp: MDP) -> Launches:
    """Generate the kernels of every runtime function, with the hooks that launch their own work between them."""
    return Launches(_startup(mdp), _apply_actions(mdp), _evaluate(mdp), _reset(mdp), _observe(mdp))


def _startup(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("startup", mdp.num_worlds, mdp.device, _mask)
    _emit(code, mdp, mdp.plan.startup)
    return code.steps()


def _apply_actions(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("apply_actions", mdp.num_worlds, mdp.device)
    actions = code.argument("actions", wp.array2d(dtype=wp.float32), lambda call: call.actions.warp)
    last_actions = _array(code, last_action, wp.array2d(dtype=wp.float32))
    code.elements(
        (mdp.action_dim,), f"{last_actions}[world, element] = {actions}[world, element]", writes=[last_action]
    )
    for action_dim in mdp.narrow_action_dims:
        key = last_action, action_dim
        narrow = _array(code, key, wp.array2d(dtype=wp.float32))
        code.elements((action_dim,), f"{narrow}[world, element] = {actions}[world, element]", writes=[key])

    # Fields with more dimensions per world are flattened; joint fields are views of the solver state that may be
    # strided, so they are passed as they are.
    def rows(call: Call, selector: FieldSelector) -> wp.array:
        field = call.state.arrays[selector].warp
        return field if field.ndim == 2 else field.reshape((mdp.num_worlds, -1))

    for step in mdp.plan.apply_actions:
        op = step.op
        if isinstance(op, TermOp):
            _apply_action_term(code, op)
            continue
        target = code.argument(op.target, wp.array2d(dtype=wp.float32), lambda call, op=op: rows(call, op.target))
        index = "element" if op.subset is None else f"{_array(code, op.subset, wp.array(dtype=wp.int32))}[element]"
        offset = code.constant(float, op.offset)
        if isinstance(op.offset, FieldSelector):
            offsets = code.argument(op.offset, wp.array2d(dtype=wp.float32), lambda call, op=op: rows(call, op.offset))
            offset = f"{offsets}[world, {index}]"
        value = f"{actions}[world, {op.columns.start} + element] * {code.constant(float, op.scale)} + {offset}"
        code.elements(
            (op.columns.stop - op.columns.start,),
            *_under(_active(code, op), f"{target}[world, {index}] = {value}"),
            writes=[op.target],
        )

    return code.steps()


def _apply_action_term(code: Fusion, op: TermOp) -> None:
    """Run an action term's compute hook, which writes its outputs from its columns of the action."""
    if isinstance(op.compute, wp.Function):
        writes = [*op.writes, *(term_state(op.use, name) for name in op.state)]
        code.world(*_under(_active(code, op), _call(code, op, op.compute)), writes=writes)
        return

    def run(call: Call) -> None:
        # Torch hooks get Torch views, and hooks that launch kernels Warp ones.
        views = "torch" if op.torch_hooks else "warp"
        active = None if op.active_key is None else call.state.arrays[op.active_key]
        arguments = {
            **call.state.arguments[op.name],
            "action": getattr(call.actions, views)[:, op.columns],
            "mask": None if active is None else getattr(active, views),
        }
        op.compute(**hook_arguments(op.compute, arguments))

    code.call(run, op.capturable, *_accesses(op))


def _evaluate(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("evaluate", mdp.num_worlds, mdp.device)
    length = _array(code, episode_length, wp.array(dtype=wp.int32))
    ended = _array(code, terminated, wp.array(dtype=wp.bool))
    terminated_result, truncated, done, rewards = (
        _result(code, key) for key in ("terminated", "truncated", "done", "rewards")
    )

    time_out = "False"
    if mdp.max_episode_length is not None:
        time_out = f"{length}[world] >= {code.constant(int, mdp.max_episode_length)}"
    lines, writes = [], []
    if mdp.interval_writes:
        lines.append(f"{_array(code, 'interval_worlds', wp.array(dtype=wp.bool))}[world] = False")
        writes.append("interval_worlds")
    code.world(
        f"{code.function(_advance)}(world, {_rng(code)})",
        f"{length}[world] = {length}[world] + 1",
        f"{terminated_result}[world] = False",
        f"{ended}[world] = False",
        f"{truncated}[world] = {time_out}",
        f"{done}[world] = {truncated}[world]",
        f"{rewards}[world] = 0.0",
        *lines,
        writes=["rng", episode_length, terminated, "terminated", "truncated", "done", "rewards", *writes],
    )

    _emit(code, mdp, mdp.plan.evaluate, stepping=True)
    return code.steps()


def _reset(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("reset", mdp.num_worlds, mdp.device, _mask)
    length = _array(code, episode_length, wp.array(dtype=wp.int32))
    keys = [last_action, *((last_action, action_dim) for action_dim in mdp.narrow_action_dims)]
    fills = [f"{code.function(_fill)}(world, {_array(code, key, wp.array2d(dtype=wp.float32))}, 0.0)" for key in keys]
    code.world(
        f"{code.function(_advance)}(world, {_rng(code)})",
        f"{length}[world] = 0",
        *fills,
        writes=["rng", episode_length, *keys],
    )

    _emit(code, mdp, mdp.plan.reset)
    return code.steps()


def _observe(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("observe", mdp.num_worlds, mdp.device, _mask)
    _emit(code, mdp, mdp.plan.observe)
    return code.steps()


def _emit(code: Fusion, mdp: MDP, steps: Iterable[Step], stepping: bool = False) -> None:
    """Add the statements for the steps of a runtime function. ``stepping`` is true for ``evaluate``."""
    for step in steps:
        match step.kind:
            case StepKind.UPDATE_PROGRESS:
                _update_progress(code, step.op)
                continue
            case StepKind.START_TASKS:
                _start_tasks(code, step.op)
                continue
            case StepKind.RESET_PROGRESS:
                _reset_progress(code, step.op)
                continue
            case StepKind.CLEAR:
                _clear(code, mdp, step.op, stepping)
                continue
        op = step.op
        state = [term_state(op.use, name) for name in op.state]
        active = _active(code, op)
        if step.entry:
            # Where the op's scope started at this step, instead of the reset worlds.
            entered = op.entry_key
            started = lambda call, entered=entered: call.state.arrays[entered].warp  # noqa: E731
            conditions = [f"if {_array(code, entered, wp.array(dtype=wp.bool))}[world]:"]
            match step.kind:
                case StepKind.RESET_TIMER:
                    restart = f"{code.function(_restart)}({_timer(code, op)})"
                    code.world(*_under(conditions, restart), writes=[("timer", op.name)])
                case StepKind.COMPUTE:
                    _run(code, mdp, op, op.compute, [*op.writes, *state], conditions, mask=started)
                case StepKind.RESET_STATE if op.reset is not None:
                    _run(code, mdp, op, op.reset, state, conditions, mask=started)
                case StepKind.RESET_STATE:
                    _reset_state(code, op, conditions)
            continue
        match step.kind:
            case StepKind.COMPUTE:
                _compute(code, mdp, op, active, stepping)
            case StepKind.COMPUTE_IF_DUE:
                due = [*active, f"if {_due(code, mdp, op, active)}:"]
                _compute(code, mdp, op, due, stepping)
                if _writes_scene(op):
                    # The scene commits the event's writes in the worlds where it ran.
                    ran = _array(code, "interval_worlds", wp.array(dtype=wp.bool))
                    code.world(*_under(due, f"{ran}[world] = True"), writes=["interval_worlds"])
            case StepKind.UPDATE_STATE:
                # A function hook gets the worlds where the op runs, None for every world.
                running = None if op.active_key is None else lambda call, op=op: call.state.arrays[op.active_key].warp
                _run(code, mdp, op, op.update, [*state, *op.writes], active, running)
            case StepKind.RESET_TIMER:
                restart = f"{code.function(_restart)}({_timer(code, op)})"
                code.world(*_under(active, restart), writes=[("timer", op.name)])
            case StepKind.RESET_STATE if op.reset is not None:
                # A function hook gets the reset worlds where the op is active.
                mask = _mask if op.active_key is None else lambda call, op=op: _selected(mdp, op, call)
                _run(code, mdp, op, op.reset, state, active, mask=mask)
            case StepKind.RESET_STATE:
                _reset_state(code, op, active)
            case StepKind.COMMIT:
                code.call(lambda call: call.commit())


def _reset_state(code: Fusion, op: TermOp, conditions: Sequence[str]) -> None:
    """Fill the state of an op that has no reset hook with its initial values, under ``conditions``."""
    for name, spec in op.state.items():
        if spec.persistent:
            continue
        key, shape = term_state(op.use, name), spec.shape[1:]
        values = _array(code, key, wp.array(dtype=wp.float32, ndim=len(spec.shape)))
        loops = [f"for i{d} in range({size}):" for d, size in enumerate(shape)]
        fill = f"{values}[{_index(shape)}] = {code.constant(float, spec.initial)}"
        code.world(*_under([*conditions, *loops], fill), writes=[key])


def _compute(code: Fusion, mdp: MDP, op: TermOp, conditions: Sequence[str] = (), stepping: bool = False) -> None:
    """Compute an op into its destination, nested under ``conditions``. Stepping advances observation histories."""
    state = [term_state(op.use, name) for name in op.state]
    reset_like = op.phase in (Phase.STARTUP_EVENTS, Phase.RESET_EVENTS) or op.phase is Phase.COMMANDS and not stepping
    if op.active_key is not None and reset_like and not isinstance(op.compute, wp.Function):
        # The hook gets the reset worlds where the op is active.
        _run(code, mdp, op, op.compute, [*op.writes, *state], mask=lambda call: _selected(mdp, op, call))
        return

    match op.phase:
        case Phase.QUANTITIES:
            value = _value(code, mdp, op)
            field = _array(code, op.writes[0], wp.array(dtype=wp.float32, ndim=1 + len(op.shape)))
            loops = [f"for i{d} in range({size}):" for d, size in enumerate(op.shape)]
            assignment = f"{field}[{_index(op.shape)}] = {value}"
            code.world(*_under([*conditions, *loops], assignment), writes=[*op.writes, *_computed_state(op)])
        case Phase.TERMINATIONS:
            # A world that terminates is not truncated, so that learners do not bootstrap its value.
            done, truncated = _result(code, "done"), _result(code, "truncated")
            writes = ["truncated", "done"]
            if op.time_out:
                flags = [f"{truncated}[world] = not {_result(code, 'terminated')}[world]"]
            else:
                flags = [
                    f"{_result(code, 'terminated')}[world] = True",
                    f"{truncated}[world] = False",
                    f"{_array(code, terminated, wp.array(dtype=wp.bool))}[world] = True",
                ]
                writes += ["terminated", terminated]
            flags.append(f"{done}[world] = True")
            lines = [f"ended = {_value(code, mdp, op)}", *_under(["if ended:"], *flags)]
            if mdp.diagnostics:
                terms = _result(code, "termination_terms")
                lines.append(f"{terms}[world, {op.index}] = ended")
                writes.append("termination_terms")
                if op.entry_key is not None:
                    # An op whose scope stopped leaves no flag.
                    code.world(f"{terms}[world, {op.index}] = False", writes=["termination_terms"])
            code.world(*_under(conditions, *lines), writes=[*writes, *_computed_state(op)])
        case Phase.REWARDS:
            weights = _array(code, "reward_weights", wp.array2d(dtype=wp.float32))
            scale = f"{weights}[world, {op.index}] * {code.constant(float, mdp.step_dt)}"
            rewards = _result(code, "rewards")
            lines = [f"reward = {scale} * {_value(code, mdp, op)}", f"{rewards}[world] = {rewards}[world] + reward"]
            writes = ["rewards"]
            if mdp.diagnostics:
                terms = _result(code, "reward_terms")
                lines.append(f"{terms}[world, {op.index}] = reward")
                writes.append("reward_terms")
                if op.entry_key is not None:
                    # An op whose scope stopped leaves no reward.
                    code.world(f"{terms}[world, {op.index}] = 0.0", writes=["reward_terms"])
            code.world(*_under(conditions, *lines), writes=[*writes, *_computed_state(op)])
        case Phase.COMMANDS | Phase.INTERVAL_EVENTS if stepping:
            writes = [*op.writes, *state]
            _run(
                code, mdp, op, op.compute, writes, conditions, lambda call: call.state.arrays["selected", op.name].warp
            )
        case Phase.STARTUP_EVENTS | Phase.RESET_EVENTS | Phase.COMMANDS:
            _run(code, mdp, op, op.compute, [*op.writes, *state], conditions, mask=_mask)
        case Phase.OBSERVATIONS:
            _compute_observation(code, mdp, op, conditions, stepping)


def _compute_observation(code: Fusion, mdp: MDP, op: TermOp, conditions: Sequence[str], stepping: bool) -> None:
    """Compute an observation with its noise, clip, scale, and history into its columns of the group."""
    observation = op.observation
    group, start = observation.group, observation.columns.start
    dtype = mdp.observation_layouts[group][1]
    out = code.argument(("observations", group), wp.array2d(dtype=dtype), lambda call: call.observations[group].warp)

    value = _value(code, mdp, op)
    if observation.noise is not None:
        low, high = code.constant(float, observation.noise.n_min), code.constant(float, observation.noise.n_max)
        seed = f"{_rng(code)}[{observation.noise_stream}, world]"
        noise = f"wp.randf(wp.rand_init({seed}, element), {low}, {high})"
        if observation.noise.scale is not None:
            noise = f"{noise} * {_array(code, observation.noise.scale, wp.array(dtype=wp.float32))}[world]"
        value = f"({value} + {noise})"
    if observation.clip is not None:
        low, high = (code.constant(float, bound) for bound in observation.clip)
        value = f"wp.clamp({value}, {low}, {high})"
    if observation.scale is not None:
        value = f"{value} * {code.constant(float, observation.scale)}"

    # With final observations, evaluate also copies what it wrote in the worlds that ended, which reset overwrites.
    final = []
    if stepping and mdp.final_observations:
        columns = [f"{start} + element"]
        if observation.history_length:
            columns = [f"{start} + {h * observation.size} + element" for h in range(observation.history_length)]
        copies = code.argument(
            ("final_observations", group),
            wp.array2d(dtype=dtype),
            lambda call: call.transition.final_observations[group].warp,
        )
        final = _under(
            [f"if {_result(code, 'done')}[world]:"], *(f"{copies}[world, {c}] = {out}[world, {c}]" for c in columns)
        )

    if not observation.history_length:
        code.elements(op.shape, *_under(conditions, f"{out}[world, {start} + element] = {value}", *final))
        return

    # Each thread keeps the history of its element: stepping shifts it and appends the value, and observing fills it
    # for new episodes. An op whose scope changes also fills it when stepping, where its scope started at this step.
    # The group holds the history oldest first, each step in ``size`` columns.
    history = _array(code, ("history", op.name), wp.array3d(dtype=wp.float32))
    size, last = observation.size, observation.history_length - 1
    fill = [f"for h in range({observation.history_length}):", f"    {history}[world, h, element] = value"]
    shift = [
        f"for h in range({last}):",
        f"    {history}[world, h, element] = {history}[world, h + 1, element]",
        f"{history}[world, {last}, element] = value",
    ]
    if stepping and op.entry_key is None:
        update = [*shift[:-1], f"{history}[world, {last}, element] = {value}"]
    elif stepping:
        update = [
            "value = " + value,
            f"if {_array(code, op.entry_key, wp.array(dtype=wp.bool))}[world]:",
            *(f"    {line}" for line in fill),
            "else:",
            *(f"    {line}" for line in shift),
        ]
    else:
        length = f"{_array(code, episode_length, wp.array(dtype=wp.int32))}[world]"
        update = [f"if {length} == 0:", "    value = " + value, *(f"    {line}" for line in fill)]
    write = [
        f"for h in range({observation.history_length}):",
        f"    {out}[world, {start} + h * {size} + element] = {history}[world, h, element]",
    ]
    code.elements(op.shape, *_under(conditions, *update, *write, *final))


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
    A Python hook, which launches its own kernels or is a Torch hook, runs on its own, and gets ``mask(call)`` as
    ``mask`` if it takes it.
    """
    if isinstance(hook, wp.Function):
        code.world(*_under(conditions, _call(code, op, hook)), writes=writes)
    else:

        def run(call: Call) -> None:
            selected = None if mask is None else mask(call)
            if op.torch_hooks and selected is not None:
                selected = wp.to_torch(selected)
            hook(**hook_arguments(hook, {**call.state.arguments[op.name], "mask": selected}))

        code.call(run, op.capturable, *_accesses(op, mdp))


def _value(code: Fusion, mdp: MDP, op: TermOp) -> str:
    """Return source for an op's output at the thread's world and element.

    A Python compute hook, which launches its own kernels or is a Torch hook, runs first, and the source reads its
    scratch buffer.
    """
    if isinstance(op.compute, wp.Function):
        return _call(code, op, op.compute, op.shape)

    if op.torch_hooks:
        code.call(
            _own_state(
                op,
                lambda call: call.state.arrays["scratch", op.name].torch.copy_(
                    op.compute(**hook_arguments(op.compute, call.state.arguments[op.name]))
                ),
            ),
            op.capturable,
            *_accesses(op, mdp),
        )
    else:
        code.call(
            _own_state(
                op,
                lambda call: op.compute(
                    out=call.state.arrays["scratch", op.name].warp,
                    **hook_arguments(op.compute, call.state.arguments[op.name]),
                ),
            ),
            op.capturable,
            *_accesses(op, mdp),
        )
    scratch = _array(code, ("scratch", op.name), wp.array(dtype=op.dtype, ndim=1 + len(op.shape)))
    return f"{scratch}[{_index(op.shape)}]"


def _own_state(op: TermOp, run: Callable[[Call], None]) -> Callable[[Call], None]:
    """Wrap a compute hook that runs for every world, so that it keeps the state it writes in the op's worlds only.

    The ops of different world configs may share the state of their use, as in the Torch backend.
    """
    if op.active_key is None or not op.state:
        return run

    def run_in_own_worlds(call: Call) -> None:
        active = call.state.arrays[op.active_key].torch
        states = [call.state.arrays[term_state(op.use, name)].torch for name in op.state]
        states = [values for values in states if values.shape[:1] == active.shape]
        before = [values.clone() for values in states]
        run(call)
        for values, kept in zip(states, before):
            values.copy_(torch.where(active.view(-1, *[1] * (values.ndim - 1)), values, kept))

    return run_in_own_worlds


def _computed_state(op: TermOp) -> list[FieldSelector]:
    """Return the keys of the state that the compute func of a quantity, termination, or reward may write."""
    if not isinstance(op.compute, wp.Function):
        return []
    return [term_state(op.use, name) for name in op.state if name in op.compute.input_types]


def _call(code: Fusion, op: TermOp, hook: wp.Function, shape: tuple[int, ...] = ()) -> str:
    """Return source that calls a func hook for the thread's world and, given an output ``shape``, its element."""
    values = [_index(shape)]
    for name, annotation in list(hook.input_types.items())[1 + len(shape) :]:
        if name in op.params:
            values.append(code.constant(annotation, op.params[name]))
        elif name in op.inputs or name in op.state:
            key = op.array_key(name) if name in op.inputs else term_state(op.use, name)
            values.append(_array(code, key, annotation))
        elif name == "action" and op.columns is not None:
            # The action's columns, as a strided view of the actions.
            columns = op.columns
            values.append(code.argument(("action", op.name), annotation, lambda call: call.actions.warp[:, columns]))
        else:
            values.append(f"{_rng(code)}[{op.stream}, world]")
    return f"{code.function(hook)}({', '.join(values)})"


def _due(code: Fusion, mdp: MDP, op: TermOp, active: Sequence[str]) -> str:
    """Return source that counts down an op's timer and is true where it expired.

    Function hooks get it as a mask, and events that write scene fields read it again, so it is kept in a buffer.
    """
    tick = f"{code.function(_tick)}({_timer(code, op)}, {code.constant(float, mdp.step_dt)})"
    if isinstance(op.compute, wp.Function) and not _writes_scene(op):
        return tick

    due = _array(code, ("selected", op.name), wp.array(dtype=wp.bool))
    code.world(*_under(active, f"{due}[world] = {tick}"), writes=[("timer", op.name), ("selected", op.name)])
    return f"{due}[world]"


def _writes_scene(op: TermOp) -> bool:
    """Return whether an op writes scene fields, which the scene must commit."""
    return any(selector.scene for selector in op.writes)


def _active(code: Fusion, op: TermOp) -> list[str]:
    """Return the condition under which an op's statements run, if it runs in only some worlds or steps."""
    return [] if op.active_key is None else [f"if {_runs(code, op)}:"]


def _runs(code: Fusion, op: TermOp) -> str:
    """Return source for whether an op runs at the thread's world, for an op that runs in only some worlds or steps."""
    return f"{_array(code, op.active_key, wp.array(dtype=wp.bool))}[world]"


def _selected(mdp: MDP, op: TermOp, call: Call) -> wp.array:
    """Write the reset worlds where an op is active to its selected buffer, and return the buffer."""
    selected = call.state.arrays["selected", op.name].warp
    inputs = [call.mask.warp, call.state.arrays[op.active_key].warp]
    wp.launch(_select, dim=mdp.num_worlds, inputs=inputs, outputs=[selected], device=mdp.device)
    return selected


def _timer(code: Fusion, op: TermOp) -> str:
    """Return source for the arguments shared by :func:`_restart` and :func:`_tick`."""
    low, high = (code.constant(float, value) for value in op.timer)
    time_left = _array(code, ("timer", op.name), wp.array(dtype=wp.float32))
    return f"world, {low}, {high}, {_rng(code)}[{op.stream}, world], {time_left}"


def _accesses(op: TermOp, mdp: MDP) -> tuple[set[Hashable], set[Hashable]]:
    """Return the keys of the kernel arguments that a hook of an op may read and write, for :meth:`Fusion.call`.

    A hook that takes a scene entity may read any of its fields; it writes those its entity outputs cover.
    """
    state = {term_state(op.use, name) for name in op.state}
    reads = {*state, ("selected", op.name), *(["rng"] if op.random else [])}
    # Hooks of an op whose scope changes read where it runs and where it started.
    reads |= {op.active_key, op.entry_key} if op.entry_key is not None else set()
    for arg, selector in op.inputs.items():
        if isinstance(selector, Entity):
            reads.update(FieldSelector(selector.name, field) for field in mdp.schema.entities[selector.name].fields)
        else:
            reads.add(op.array_key(arg))
    return reads, {*op.writes, *state, ("scratch", op.name)}


def _array(code: Fusion, key: Hashable, annotation: wp.array) -> str:
    """Return the source name of an array of the call's state, by its key in :attr:`MDPState.arrays`."""
    return code.argument(key, annotation, lambda call: call.state.arrays[key].warp)


def _result(code: Fusion, key: str) -> str:
    """Return the source name of a transition array, see :data:`_RESULTS`."""
    return code.argument(key, *_RESULTS[key])


def _rng(code: Fusion) -> str:
    """Return the source name of the random streams, shaped (num_streams, num_worlds)."""
    return _array(code, "rng", wp.array2d(dtype=wp.int32))


def _mask(call: Call) -> wp.array:
    """Return the worlds to reset or observe."""
    return call.mask.warp


def _index(shape: tuple[int, ...]) -> str:
    """Return source for the indices of the thread's world and element of ``shape``."""
    return ", ".join(["world", *(f"i{d}" for d in range(len(shape)))])


def _under(conditions: Sequence[str], *lines: str) -> list[str]:
    """Indent ``lines`` under ``conditions``, nesting each condition one level deeper than the previous one."""
    nested = [f"{'    ' * depth}{condition}" for depth, condition in enumerate(conditions)]
    return nested + [f"{'    ' * len(conditions)}{line}" for line in lines]


_PROGRESS_TYPES = {"done": wp.bool, "holding": wp.bool, "active": wp.bool, "entered": wp.bool, "score": wp.float32}
"""Element types of the progress fields other than the ``int32`` ones."""


def _progress(code: Fusion, path: str, name: str) -> str:
    """Return source for a field of a task's progress at the thread's world."""
    key = progress_field(path, name)
    return f"{_array(code, key, wp.array(dtype=_PROGRESS_TYPES.get(name, wp.int32)))}[world]"


def _in_tree(trees: tuple[TaskTree, ...], tree: TaskTree) -> list[str]:
    """Return the condition for the worlds of a tree's world config, if there are several."""
    return [] if len(trees) == 1 else [f"if world >= {tree.worlds.start} and world < {tree.worlds.stop}:"]


def _tree_writes(tree: TaskTree, names: Iterable[str]) -> list[FieldSelector]:
    """Return the keys of the progress fields of a tree that a statement may write."""
    return [progress_field(node.path, name) for node in tree.nodes for name in names]


def _update_progress(code: Fusion, trees: tuple[TaskTree, ...]) -> None:
    """Count the completions of the tasks of every tree, subtasks first, and advance the sequences once."""
    length = f"{_array(code, episode_length, wp.array(dtype=wp.int32))}[world]"
    for tree in trees:
        lines = []
        for node in reversed(tree.nodes):
            if node.path in tree.completes:
                lines += _node_progress(code, tree, node, length)
        names = ("done", "holding", "score", "done_at", "streak", "stage")
        if lines:
            code.world(*_under(_in_tree(trees, tree), *lines), writes=_tree_writes(tree, names))


def _node_progress(code: Fusion, tree: TaskTree, node: TaskNode, length: str) -> list[str]:
    """Return the statements that update the progress of one task, after those of its subtasks."""
    field = functools.partial(_progress, code, node.path)
    active, done, holding, score = field("active"), field("done"), field("holding"), field("score")

    if node.done is not None:
        # Done holds for ``hold`` steps in a row, counted while the task runs and, once done, after it.
        condition = f"{_array(code, node.done, wp.array(dtype=node.done_dtype))}[world]"
        positive = {wp.bool: "", wp.float32: " > 0.0", wp.int32: " > 0"}[node.done_dtype]
        streak = field("streak")
        lines = [
            f"if {active} or {done}:",
            f"    if {condition}{positive}:",
            f"        {streak} = wp.min({streak} + 1, {node.hold})",
            "    else:",
            f"        {streak} = 0",
            f"{holding} = ({active} or {done}) and {streak} >= {node.hold}",
            f"complete = {holding}",
        ]
    else:
        counted = tree.counted(node)
        children = [functools.partial(_progress, code, child.path) for child in counted]
        finals = [f"{child('holding')} == {child_node.final}" for child, child_node in zip(children, counted)]
        finals = [final for final, child_node in zip(finals, counted) if child_node.final is not None]
        finals = " and ".join(finals) or "True"
        if node.mode is TaskMode.SEQUENCE:
            # One subtask at most per step: the next one starts at the step after.
            stage = field("stage")
            lines = [f"if {active} and not {done}:"]
            for k, child in enumerate(children):
                lines += [
                    f"    {'if' if k == 0 else 'elif'} {stage} == {k} and {child('done')}:",
                    f"        {stage} = {k + 1}",
                ]
            lines += [
                f"complete = {stage} == {len(counted)} and ({finals})",
                f"{holding} = {children[-1]('holding')}",
            ]
        else:
            required = node.require or len(counted)
            lines = [
                "count = int(0)",
                "held = int(0)",
                *(f"count = count + wp.where({child('done')}, 1, 0)" for child in children),
                *(f"held = held + wp.where({child('holding')}, 1, 0)" for child in children),
                f"complete = count >= {required} and ({finals})",
                f"{holding} = held >= {required}",
            ]

    lines += [
        f"if {active} and not {done} and complete:",
        f"    {done} = True",
        f"    {field('done_at')} = {length}",
    ]
    if node.done is not None:
        return [*lines, f"{score} = wp.where({done}, 1.0, 0.0)"]

    counted = tree.counted(node)
    scores = [_progress(code, child.path, "score") for child in counted]
    if node.mode is TaskMode.TOGETHER and node.require is not None:
        # The mean of the best ``require`` scores, picked one at a time.
        lines += ["taken = int(0)", "total = float(0.0)"]
        for _ in range(node.require):
            lines += ["best = float(-1.0)", "pick = int(0)"]
            for k, child_score in enumerate(scores):
                lines += [
                    f"if (taken & {1 << k}) == 0 and {child_score} > best:",
                    f"    best = {child_score}",
                    f"    pick = {1 << k}",
                ]
            lines += ["taken = taken | pick", "total = total + best"]
        return [*lines, f"{score} = total / {float(node.require)}"]
    weights = [child.score / sum(child.score for child in counted) for child in counted]
    return [*lines, f"{score} = " + " + ".join(f"{weight!r} * {child}" for weight, child in zip(weights, scores))]


def _start_tasks(code: Fusion, trees: tuple[TaskTree, ...]) -> None:
    """Start the tasks of the sequences that advanced, parents first, and stop the tasks they left."""
    length = f"{_array(code, episode_length, wp.array(dtype=wp.int32))}[world]"
    for tree in trees:
        lines = []
        for node in tree.nodes:
            if not node.staged:
                continue
            parent = tree.parent(node)
            runs = _progress(code, parent.path, "active")
            if parent.mode is TaskMode.SEQUENCE:
                # The last task keeps running once it is done.
                stage = f"wp.min({_progress(code, parent.path, 'stage')}, {len(parent.children) - 1})"
                runs = f"{runs} and {stage} == {parent.children.index(node.path)}"
            field = functools.partial(_progress, code, node.path)
            lines += [
                f"runs = {runs}",
                f"{field('entered')} = runs and not {field('active')}",
                f"if {field('entered')}:",
                f"    {field('started_at')} = {length}",
                f"{field('active')} = runs",
            ]
        if lines:
            writes = _tree_writes(tree, ("entered", "started_at", "active"))
            code.world(*_under(_in_tree(trees, tree), *lines), writes=writes)


def _reset_progress(code: Fusion, trees: tuple[TaskTree, ...]) -> None:
    """Restart the progress of every task in the masked worlds, as at the start of an episode."""
    for tree in trees:
        lines = []
        for node in tree.nodes:
            field = functools.partial(_progress, code, node.path)
            lines += [
                f"{field('done')} = False",
                f"{field('holding')} = False",
                f"{field('done_at')} = -1",
                f"{field('started_at')} = 0",
                f"{field('score')} = 0.0",
                f"{field('active')} = {tree.runs_at_start(node)}",
                *([f"{field('stage')} = 0"] if node.mode is TaskMode.SEQUENCE else []),
                *([f"{field('streak')} = 0"] if node.done is not None else []),
                *([f"{field('entered')} = False"] if node.staged else []),
            ]
        names = ("done", "holding", "done_at", "started_at", "score", "active", "stage", "streak", "entered")
        code.world(*_under(_in_tree(trees, tree), *lines), writes=_tree_writes(tree, names))


def _clear(code: Fusion, mdp: MDP, slot: ObservationSlot, stepping: bool) -> None:
    """Zero the columns of a slot that no op writes at this step.

    A column is written by the op that runs and is that wide, so the threads of different ops never write the same
    column of a world.
    """
    dtype = mdp.observation_layouts[slot.buffer][1]
    group = slot.buffer
    out = code.argument(("observations", group), wp.array2d(dtype=dtype), lambda call: call.observations[group].warp)
    covers = []
    for op in slot.ops:
        width = f"element % {slot.size} < {math.prod(op.shape)}"
        covers.append(width if op.active_key is None else f"({_runs(code, op)} and {width})")
    column, zero = f"{slot.columns.start} + element", f"wp.{dtype.__name__}(0)"
    lines = [f"if not ({' or '.join(covers)}):", f"    {out}[world, {column}] = {zero}"]
    if stepping and mdp.final_observations:
        copies = code.argument(
            ("final_observations", group),
            wp.array2d(dtype=dtype),
            lambda call: call.transition.final_observations[group].warp,
        )
        lines += [f"    if {_result(code, 'done')}[world]:", f"        {copies}[world, {column}] = {zero}"]
    code.elements((slot.columns.stop - slot.columns.start,), *lines)
