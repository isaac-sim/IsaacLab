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

import math
from collections.abc import Callable, Hashable, Iterable, Sequence

import warp as wp

from .compiled import MDP
from .config import FieldSelector, Phase, episode_length, last_action, term_state, terminated
from .eager import call_eager
from .fusion import Fusion
from .plan import Call, Launches, Step, StepKind, TermOp
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

    # Fields with more dimensions per world are flattened; joint fields are views of the solver state that may be
    # strided, so they are passed as they are.
    def rows(call: Call, selector: FieldSelector) -> wp.array:
        field = call.state.arrays[selector].warp
        return field if field.ndim == 2 else field.reshape((mdp.num_worlds, -1))

    for step in mdp.plan.apply_actions:
        op = step.op
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
    code.world(
        f"{code.function(_advance)}(world, {_rng(code)})",
        f"{length}[world] = {length}[world] + 1",
        f"{terminated_result}[world] = False",
        f"{ended}[world] = False",
        f"{truncated}[world] = {time_out}",
        f"{done}[world] = {truncated}[world]",
        f"{rewards}[world] = 0.0",
        writes=["rng", episode_length, terminated, "terminated", "truncated", "done", "rewards"],
    )

    _emit(code, mdp, mdp.plan.evaluate, stepping=True)
    return code.steps()


def _reset(mdp: MDP) -> tuple[Callable[[Call], None], ...]:
    code = Fusion("reset", mdp.num_worlds, mdp.device, _mask)
    length = _array(code, episode_length, wp.array(dtype=wp.int32))
    last_actions = _array(code, last_action, wp.array2d(dtype=wp.float32))
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


def _emit(code: Fusion, mdp: MDP, steps: Iterable[Step], stepping: bool = False) -> None:
    """Add the statements for the steps of a runtime function. ``stepping`` is true for ``evaluate``."""
    for step in steps:
        op = step.op
        state = [term_state(op.use, name) for name in op.state]
        active = _active(code, op)
        match step.kind:
            case StepKind.COMPUTE:
                _compute(code, mdp, op, active, stepping)
            case StepKind.COMPUTE_IF_DUE:
                _compute(code, mdp, op, [*active, f"if {_due(code, mdp, op, active)}:"], stepping)
            case StepKind.UPDATE_STATE:
                # A function hook gets the worlds where the op runs, None for every world.
                running = None if op.worlds is None else lambda call, op=op: call.state.arrays["active", op.name].warp
                _run(code, mdp, op, op.update, [*state, *op.writes], active, running)
            case StepKind.RESET_TIMER:
                restart = f"{code.function(_restart)}({_timer(code, op)})"
                code.world(*_under(active, restart), writes=[("timer", op.name)])
            case StepKind.RESET_STATE if op.reset is not None:
                _run(code, mdp, op, op.reset, state, active, mask=_mask)
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
    if op.worlds is not None and reset_like and not isinstance(op.compute, wp.Function):
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
            writes = ["truncated" if op.time_out else "terminated", "done"]
            flags = [f"{_result(code, key)}[world] = True" for key in writes]
            if not op.time_out:
                flags.append(f"{_array(code, terminated, wp.array(dtype=wp.bool))}[world] = True")
                writes.append(terminated)
            lines = [f"ended = {_value(code, mdp, op)}", *_under(["if ended:"], *flags)]
            if mdp.diagnostics:
                lines.append(f"{_result(code, 'termination_terms')}[world, {op.index}] = ended")
                writes.append("termination_terms")
            code.world(*_under(conditions, *lines), writes=[*writes, *_computed_state(op)])
        case Phase.REWARDS:
            weights = _array(code, "reward_weights", wp.array2d(dtype=wp.float32))
            scale = f"{weights}[world, {op.index}] * {code.constant(float, mdp.step_dt)}"
            rewards = _result(code, "rewards")
            lines = [f"reward = {scale} * {_value(code, mdp, op)}", f"{rewards}[world] = {rewards}[world] + reward"]
            writes = ["rewards"]
            if mdp.diagnostics:
                lines.append(f"{_result(code, 'reward_terms')}[world, {op.index}] = reward")
                writes.append("reward_terms")
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
    out = code.argument(
        ("observations", group), wp.array2d(dtype=wp.float32), lambda call: call.observations[group].warp
    )

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

    if not observation.history_length:
        code.elements(op.shape, *_under(conditions, f"{out}[world, {start} + element] = {value}"))
        return

    # Each thread keeps the history of its element: stepping shifts it and appends the value, observing fills it for
    # new episodes. The group holds the history oldest first.
    history = _array(code, ("history", op.name), wp.array3d(dtype=wp.float32))
    size, last = math.prod(op.shape), observation.history_length - 1
    if stepping:
        update = [
            f"for h in range({last}):",
            f"    {history}[world, h, element] = {history}[world, h + 1, element]",
            f"{history}[world, {last}, element] = {value}",
        ]
    else:
        length = _array(code, episode_length, wp.array(dtype=wp.int32))
        update = [
            f"if {length}[world] == 0:",
            "    value = " + value,
            f"    for h in range({observation.history_length}):",
            f"        {history}[world, h, element] = value",
        ]
    write = [
        f"for h in range({observation.history_length}):",
        f"    {out}[world, {start} + h * {size} + element] = {history}[world, h, element]",
    ]
    code.elements(op.shape, *_under(conditions, *update, *write))


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
    A function that launches its own kernels, a Torch hook, or an eager term, runs on its own, and gets ``mask(call)``
    as ``mask`` if it takes it.
    """
    if isinstance(hook, wp.Function):
        code.world(*_under(conditions, _call(code, op, hook)), writes=writes)
    elif op.eager:
        code.call(lambda call: call_eager(mdp, op, call.state, wp.to_torch(mask(call))), False, *_accesses(op))
    else:

        def run(call: Call) -> None:
            selected = None if mask is None else mask(call)
            if op.torch_hooks and selected is not None:
                selected = wp.to_torch(selected)
            hook(**hook_arguments(hook, {**call.state.arguments[op.name], "mask": selected}))

        code.call(run, op.capturable, *_accesses(op))


def _value(code: Fusion, mdp: MDP, op: TermOp) -> str:
    """Return source for an op's output at the thread's world and element.

    A compute hook that launches its own kernels, a Torch hook, or an eager term, runs first, and the source reads its
    scratch buffer.
    """
    if isinstance(op.compute, wp.Function):
        return _call(code, op, op.compute, op.shape)

    if op.eager:
        code.call(lambda call: call_eager(mdp, op, call.state), False, *_accesses(op))
    elif op.torch_hooks:
        code.call(
            lambda call: call.state.arrays["scratch", op.name].torch.copy_(
                op.compute(**hook_arguments(op.compute, call.state.arguments[op.name]))
            ),
            op.capturable,
            *_accesses(op),
        )
    else:
        code.call(
            lambda call: op.compute(
                out=call.state.arrays["scratch", op.name].warp,
                **hook_arguments(op.compute, call.state.arguments[op.name]),
            ),
            op.capturable,
            *_accesses(op),
        )
    scratch = _array(code, ("scratch", op.name), wp.array(dtype=op.dtype, ndim=1 + len(op.shape)))
    return f"{scratch}[{_index(op.shape)}]"


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
            key = op.inputs.get(name, term_state(op.use, name))
            values.append(_array(code, key, annotation))
        else:
            values.append(f"{_rng(code)}[{op.stream}, world]")
    return f"{code.function(hook)}({', '.join(values)})"


def _due(code: Fusion, mdp: MDP, op: TermOp, active: Sequence[str]) -> str:
    """Return source that counts down an op's timer and is true where it expired. Function hooks get a mask."""
    tick = f"{code.function(_tick)}({_timer(code, op)}, {code.constant(float, mdp.step_dt)})"
    if isinstance(op.compute, wp.Function):
        return tick

    due = _array(code, ("selected", op.name), wp.array(dtype=wp.bool))
    code.world(*_under(active, f"{due}[world] = {tick}"), writes=[("timer", op.name), ("selected", op.name)])
    return f"{due}[world]"


def _active(code: Fusion, op: TermOp) -> list[str]:
    """Return the condition under which an op's statements run, if it is active in only some worlds."""
    if op.worlds is None:
        return []
    active = _array(code, ("active", op.name), wp.array(dtype=wp.bool))
    return [f"if {active}[world]:"]


def _selected(mdp: MDP, op: TermOp, call: Call) -> wp.array:
    """Write the reset worlds where an op is active to its selected buffer, and return the buffer."""
    selected = call.state.arrays["selected", op.name].warp
    inputs = [call.mask.warp, call.state.arrays["active", op.name].warp]
    wp.launch(_select, dim=mdp.num_worlds, inputs=inputs, outputs=[selected], device=mdp.device)
    return selected


def _timer(code: Fusion, op: TermOp) -> str:
    """Return source for the arguments shared by :func:`_restart` and :func:`_tick`."""
    low, high = (code.constant(float, value) for value in op.timer)
    time_left = _array(code, ("timer", op.name), wp.array(dtype=wp.float32))
    return f"world, {low}, {high}, {_rng(code)}[{op.stream}, world], {time_left}"


def _accesses(op: TermOp) -> tuple[set[Hashable], set[Hashable]]:
    """Return the keys of the kernel arguments that a hook of an op may read and write, for :meth:`Fusion.call`."""
    state = {term_state(op.use, name) for name in op.state}
    reads = {*op.inputs.values(), *state, ("selected", op.name), *(["rng"] if op.random else [])}
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
