# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch backend: runs batched term functions, one launch per step of the execution plan."""

from __future__ import annotations

import functools
import itertools
import math

import torch

from . import rng
from .compiled import MDP, MDPState
from .config import FieldSelector, Phase, TaskMode, progress_field, term_state, terminated
from .plan import Call, Launch, Launches, ObservationSlot, Step, StepKind, TaskTree, TermOp, TunedValue
from .terms import hook_arguments


def compile_launches(mdp: MDP) -> Launches:
    """Turn every step of the execution plan into a launch, after the bookkeeping that starts each function."""
    return Launches(
        startup=_launches(mdp, mdp.plan.startup),
        # The actions are stored once per control step: with the actions that apply once, if any.
        process_actions=(
            (Launch(lambda call: _apply_actions(mdp, call, mdp.plan.process_actions, store=True)),)
            if mdp.plan.process_actions
            else ()
        ),
        apply_actions=(
            Launch(lambda call: _apply_actions(mdp, call, mdp.plan.apply_actions, store=not mdp.plan.process_actions)),
        ),
        evaluate=(Launch(lambda call: _begin_evaluate(mdp, call)), *_launches(mdp, mdp.plan.evaluate, stepping=True)),
        reset=(Launch(lambda call: _begin_reset(mdp, call)), *_launches(mdp, mdp.plan.reset)),
        observe=_launches(mdp, mdp.plan.observe),
    )


def _launches(mdp: MDP, steps: tuple[Step, ...], stepping: bool = False) -> tuple[Launch, ...]:
    """Return one launch per step."""
    return tuple(Launch(lambda call, step=step: _run(mdp, step, call, stepping), step.capturable) for step in steps)


def _apply_actions(mdp: MDP, call: Call, steps: tuple[Step, ...], store: bool) -> None:
    """Store the actions as the last actions if ``store``, and run the action steps: write each action's scaled
    columns to its target field, or run an action term."""
    actions = call.actions.torch
    if store:
        _store_actions(mdp, call)

    for step in steps:
        op = step.op
        if isinstance(op, TermOp):
            # The hook writes its outputs; it may take the worlds where it is active as ``mask``.
            active = _active(op, call.state)
            arguments = {**call.state.arguments[op.name], "action": actions[:, op.columns]}
            arguments["mask"] = None if active is None else active.torch
            op.compute(**hook_arguments(op.compute, arguments))
            continue
        target = call.state.arrays[op.target].torch.view(mdp.num_worlds, -1)
        ids = slice(None) if op.subset is None else call.state.arrays[op.subset].torch
        offset = op.offset
        if isinstance(offset, FieldSelector):
            offset = call.state.arrays[offset].torch.view(mdp.num_worlds, -1)[:, ids]
        value = actions[:, op.columns] * op.scale + offset

        active = _active(op, call.state)
        if active is not None:
            value = torch.where(active.torch[:, None], value, target[:, ids])
        target[:, ids] = value


def _store_actions(mdp: MDP, call: Call) -> None:
    """Store the actions as the last actions, which keep the columns that each world takes at this step and zero the
    others."""
    call.state.actions.torch.copy_(call.actions.torch)
    for slot in (slot for slot in mdp.action_slots if slot.partial):
        columns = torch.arange(slot.columns.start, slot.columns.stop, device=mdp.device)
        covered = torch.zeros(mdp.num_worlds, len(columns), dtype=torch.bool, device=mdp.device)
        for op in slot.ops:
            active = _active(op, call.state)
            width = columns < op.columns.stop
            covered |= width if active is None else active.torch[:, None] & width
        call.state.actions.torch[:, slot.columns].masked_fill_(~covered, 0.0)


def _begin_evaluate(mdp: MDP, call: Call) -> None:
    """Advance the random streams and episode lengths, and clear the results of the last step."""
    state, transition = call.state, call.transition
    rng.advance(state.rng.torch)
    state.episode_length.torch.add_(1)

    transition.terminated.torch.zero_()
    transition.truncated.torch.zero_()
    if mdp.max_episode_length is not None:
        transition.truncated.torch.copy_(state.episode_length.torch >= mdp.max_episode_length)
    transition.done.torch.copy_(transition.truncated.torch)
    state.arrays[terminated].torch.zero_()
    transition.rewards.torch.zero_()
    if "event_worlds" in state.arrays:
        state.arrays["event_worlds"].torch.zero_()


def _begin_reset(mdp: MDP, call: Call) -> None:
    """Advance the random streams of the masked worlds, and clear their episode lengths and last actions."""
    mask = call.mask.torch
    rng.advance(call.state.rng.torch, mask)
    call.state.episode_length.torch.masked_fill_(mask, 0)
    call.state.actions.torch.masked_fill_(mask[:, None], 0.0)


def _active(op, state: MDPState):
    """Return the worlds where an op runs at this step, None for every world."""
    return None if op.active_key is None else state.arrays[op.active_key]


def _run(mdp: MDP, step: Step, call: Call, stepping: bool) -> None:
    """Run one step, for the masked worlds if the call has a mask, and only where the op is active.

    A step on entry runs for the worlds where the op's task just started instead.
    """
    op, state = step.op, call.state
    match step.kind:
        case StepKind.UPDATE_PROGRESS:
            _update_progress(op, state)
            return
        case StepKind.START_TASKS:
            _start_tasks(op, state)
            return
        case StepKind.RESET_PROGRESS:
            _reset_progress(op, state, call.mask.torch)
            return
        case StepKind.CLEAR:
            _clear(mdp, op, call, stepping)
            return
        case StepKind.APPLY_TUNED:
            _apply_tuned(op, state, call.mask.torch)
            return
        case StepKind.SWITCH:
            _switch(op, state, None if call.mask is None else call.mask.torch)
            return

    mask = None if call.mask is None else call.mask.torch
    active = _active(op, state)
    if step.entry:
        mask, active = state.arrays[op.entry_key].torch, None
    if active is not None:
        mask = active.torch if mask is None else mask & active.torch

    match step.kind:
        case StepKind.COMPUTE:
            # A compute hook that advances state does so for every world, but the op may share the state of its use
            # with the ops of other world configs: keep what the hook wrote in its own worlds only.
            stepped = op.phase in (Phase.QUANTITIES, Phase.TERMINATIONS, Phase.REWARDS)
            names = [name for name in op.state if stepped and active is not None]
            states = [state.arrays[term_state(op.use, name)].torch for name in names]
            states = [values for values in states if values.shape[:1] == active.torch.shape]
            before = [values.clone() for values in states]
            _compute(mdp, op, call, mask, stepping)
            for values, kept in zip(states, before):
                own = active.torch.view(-1, *[1] * (values.ndim - 1))
                values.copy_(torch.where(own, values, kept))
            if step.entry and op.phase is Phase.RESET_EVENTS and any(selector.scene for selector in op.writes):
                # The scene commits a start event's writes in the worlds where it ran.
                state.arrays["event_worlds"].torch.logical_or_(mask)
        case StepKind.COMPUTE_IF_DUE:
            due = _tick(mdp, op, state)
            due = due if active is None else due & active.torch
            _compute(mdp, op, call, due, stepping)
            if any(selector.scene for selector in op.writes):
                # The scene commits the event's writes in the worlds where it ran.
                state.arrays["event_worlds"].torch.logical_or_(due)
        case StepKind.UPDATE_STATE:
            op.update(**hook_arguments(op.update, {**state.arguments[op.name], "mask": mask}))
        case StepKind.RESET_TIMER:
            _restart(op, state, mask)
        case StepKind.COMMIT:
            call.commit()
        case StepKind.RESET_STATE if op.reset is not None:
            op.reset(mask=mask, **hook_arguments(op.reset, state.arguments[op.name]))
        case StepKind.RESET_STATE:
            for name, spec in op.state.items():
                if spec.persistent:
                    continue
                values = state.arrays[term_state(op.use, name)].torch.view(mdp.num_worlds, -1)
                values.masked_fill_(mask[:, None], spec.initial)


def _compute(mdp: MDP, op: TermOp, call: Call, mask: torch.Tensor | None, stepping: bool) -> None:
    """Compute an op and write the result to its destination, for the masked worlds if a mask is given."""
    state, transition = call.state, call.transition
    # Events and commands write their outputs in place, for the masked worlds.
    if op.phase in (Phase.STARTUP_EVENTS, Phase.COMMANDS, Phase.INTERVAL_EVENTS, Phase.RESET_EVENTS):
        op.compute(mask=mask, **hook_arguments(op.compute, state.arguments[op.name]))
        return

    value = op.compute(**hook_arguments(op.compute, state.arguments[op.name]))
    if op.phase is Phase.QUANTITIES:
        destination = state.arrays[op.writes[0]].torch
        if mask is not None:
            value = torch.where(mask.view(-1, *[1] * len(op.shape)), value, destination)
        destination.copy_(value)
        return

    if mask is not None and op.phase in (Phase.TERMINATIONS, Phase.REWARDS):
        value = torch.where(mask, value, torch.zeros_like(value))

    if op.phase is Phase.TERMINATIONS:
        # A world that terminates is not truncated, so that learners do not bootstrap its value.
        if op.time_out:
            transition.truncated.torch.logical_or_(value & ~transition.terminated.torch)
        else:
            transition.terminated.torch.logical_or_(value)
            transition.truncated.torch.logical_and_(~value)
            state.arrays[terminated].torch.logical_or_(value)
        transition.done.torch.logical_or_(value)
        if mdp.diagnostics:
            transition.termination_terms.torch[:, op.index] = value
        return

    if op.phase is Phase.REWARDS:
        value = value * (state.reward_weights.torch[:, op.index] * mdp.step_dt)
        transition.rewards.torch.add_(value)
        if mdp.diagnostics:
            transition.reward_terms.torch[:, op.index] = value
        return

    observation = op.observation
    # The op writes the first columns of each step of history of its slot.
    steps = max(observation.history_length, 1)
    slot = call.observations[observation.group].torch[:, observation.columns]
    destination = slot.view(mdp.num_worlds, steps, observation.size)[:, :, : math.prod(op.shape)]
    if observation.noise is not None:
        noise = rng.uniform(
            state.rng.torch[observation.noise_stream], observation.noise.n_min, observation.noise.n_max, op.shape
        )
        if observation.noise.scale is not None:
            noise = noise * state.arrays[observation.noise.scale].torch.view(-1, *[1] * len(op.shape))
        value = value + noise
    if observation.clip is not None:
        value = value.clamp(*observation.clip)
    if observation.scale is not None:
        value = value * observation.scale
    if observation.history_length:
        value = _history(op, state, value.reshape(mdp.num_worlds, -1), mask, push=stepping)

    value = value.reshape(destination.shape)
    destination.copy_(value if mask is None else torch.where(mask[:, None, None], value, destination))
    if stepping and transition.final_observations is not None:
        ended = transition.done.torch if mask is None else transition.done.torch & mask
        final = transition.final_observations[observation.group].torch[:, observation.columns]
        final = final.view(destination.shape[0], steps, observation.size)[:, :, : destination.shape[2]]
        final.copy_(torch.where(ended[:, None, None], value, final))


def _history(op: TermOp, state: MDPState, value: torch.Tensor, mask: torch.Tensor | None, push: bool) -> torch.Tensor:
    """Push a value into an observation's history, or fill it where it starts: in new episodes if not ``push``,
    and, for an op whose scope changes, where its scope started at this step.

    Returns the history, oldest first, as ``(num_worlds, history_length, size)``.
    """
    history = state.arrays["history", op.name].torch
    selected = torch.ones_like(value[:, 0], dtype=torch.bool) if mask is None else mask
    if push:
        starts = torch.zeros_like(selected) if op.entry_key is None else state.arrays[op.entry_key].torch
    else:
        starts = state.episode_length.torch == 0

    filled = value[:, None].expand_as(history)
    if push:
        shifted = torch.where(starts[:, None, None], filled, torch.cat([history[:, 1:], value[:, None]], dim=1))
    else:
        selected = selected & starts
        shifted = filled

    history.copy_(torch.where(selected[:, None, None], shifted, history))
    return history


def _tick(mdp: MDP, op: TermOp, state: MDPState) -> torch.Tensor:
    """Count down a timer and restart it where it expired. Returns where it expired."""
    time_left = state.arrays["timer", op.name].torch
    time_left -= mdp.step_dt
    due = time_left <= 0.0
    _restart(op, state, due)
    return due


def _restart(op: TermOp, state: MDPState, mask: torch.Tensor) -> None:
    """Draw a new interval for the timer of an op in the masked worlds."""
    time_left = state.arrays["timer", op.name].torch
    time_left.copy_(torch.where(mask, rng.uniform(state.rng.torch[op.stream], *op.timer, offset=-1), time_left))


def _switch(ops: tuple[TermOp, ...], state: MDPState, mask: torch.Tensor | None) -> None:
    """Turn each op with ``when`` on where its task runs and its field is positive, and each op of several tasks on
    where any of them runs. Stepping (no mask) marks the worlds where it turned on; observing new episodes, in the
    masked worlds, marks none, since resets restarted them."""
    for op in ops:
        runs = torch.ones(state.episode_length.torch.shape, dtype=torch.bool, device=state.episode_length.torch.device)
        if op.stages:
            runs = torch.stack([state.arrays[progress_field(stage, "active")].torch for stage in op.stages]).any(0)
        elif op.scope_key is not None:
            runs = state.arrays[op.scope_key].torch
        if op.when is not None:
            runs = runs & (state.arrays[op.when].torch > 0)
        on, entered = state.arrays[op.active_key].torch, state.arrays[op.entry_key].torch
        if mask is None:
            torch.logical_and(runs, ~on, out=entered)
            on.copy_(runs)
        else:
            entered.masked_fill_(mask, False)
            on.copy_(torch.where(mask, runs, on))


def _apply_tuned(tuned: tuple[TunedValue, ...], state: MDPState, mask: torch.Tensor) -> None:
    """Apply the values tuned for the next reset in the reset worlds that have them, and count the tuning."""
    pending = state.arrays["pending"].torch
    applied = pending & mask
    for value in tuned:
        live, waiting = state.arrays[value.key].torch, state.arrays["pending", value.key].torch
        if value.column is not None:
            live, waiting = live[:, value.column], waiting[:, value.column]
        live.copy_(torch.where(applied.view(-1, *[1] * (live.ndim - 1)), waiting, live))
    state.arrays["revision"].torch.add_(applied.int())
    pending.logical_and_(~applied)


def _clear(mdp: MDP, slot: ObservationSlot, call: Call, stepping: bool) -> None:
    """Zero the columns of a slot that no op writes at this step, in the masked worlds."""
    steps = slot.columns.stop - slot.columns.start
    columns = torch.arange(steps, device=mdp.device) % slot.size
    covered = torch.zeros(mdp.num_worlds, steps, dtype=torch.bool, device=mdp.device)
    for op in slot.ops:
        active = _active(op, call.state)
        width = columns < math.prod(op.shape)
        covered |= width if active is None else active.torch[:, None] & width
    cleared = ~covered if call.mask is None else ~covered & call.mask.torch[:, None]
    call.observations[slot.buffer].torch[:, slot.columns].masked_fill_(cleared, 0.0)
    if stepping and call.transition.final_observations is not None:
        final = call.transition.final_observations[slot.buffer].torch[:, slot.columns]
        final.masked_fill_(cleared & call.transition.done.torch[:, None], 0.0)


def _progress(state: MDPState, tree: TaskTree, path: str, name: str) -> torch.Tensor:
    """Return a field of a task's progress in the worlds of its tree."""
    return state.arrays[progress_field(path, name)].torch[tree.worlds.start : tree.worlds.stop]


def _update_progress(trees: tuple[TaskTree, ...], state: MDPState) -> None:
    """Count the completions of the tasks of every tree, subtasks first, and advance the sequences once."""
    for tree in trees:
        length = state.episode_length.torch[tree.worlds.start : tree.worlds.stop]
        for node in reversed(tree.nodes):
            if node.path not in tree.completes:
                continue
            field = functools.partial(_progress, state, tree, node.path)
            active, done, holding, score = field("active"), field("done"), field("holding"), field("score")

            if node.done is not None:
                # Done holds for ``hold`` steps in a row, counted while the task runs and, once done, after it.
                condition = state.arrays[node.done].torch[tree.worlds.start : tree.worlds.stop] > 0
                streak, counting = field("streak"), active | done
                streak.copy_(
                    torch.where(counting, torch.where(condition, (streak + 1).clamp(max=node.hold), 0), streak)
                )
                holding.copy_(counting & (streak >= node.hold))
                complete = holding
            else:
                counted = tree.counted(node)
                done_children = torch.stack([_progress(state, tree, child.path, "done") for child in counted], 1)
                held = torch.stack([_progress(state, tree, child.path, "holding") for child in counted], 1)
                finals = torch.ones_like(active)
                for k, child in enumerate(counted):
                    if child.final is not None:
                        finals &= held[:, k] == child.final
                if node.mode is TaskMode.SEQUENCE:
                    stage = field("stage")
                    current = done_children.gather(1, stage.long().clamp(max=len(counted) - 1)[:, None])[:, 0]
                    # A sequence that repeats advances again once it restarted.
                    finished = field("spent") if node.repeat else done
                    stage.add_((active & ~finished & (stage < len(counted)) & current).int())
                    complete = (stage == len(counted)) & finals
                    holding.copy_(held[:, -1])
                elif node.mode is TaskMode.CHOICE:
                    # Done once the drawn subtask is.
                    stage, complete = field("stage"), torch.zeros_like(active)
                    holding.zero_()
                    for k, child in enumerate(counted):
                        chosen = stage == node.children.index(child.path)
                        complete = torch.where(chosen, done_children[:, k], complete)
                        holding.copy_(torch.where(chosen, held[:, k], holding))
                else:
                    required = node.require or len(counted)
                    complete = (done_children.sum(1) >= required) & finals
                    holding.copy_(held.sum(1) >= required)

            if node.repeat:
                _repeat(state, tree, node, complete, length)
            else:
                started = active & ~done & complete
                field("done_at").copy_(torch.where(started, length, field("done_at")))
                done.logical_or_(started)
            score.copy_(_score(state, tree, node) if node.done is None else done.float())
            if node.repeat:
                # A task that repeats keeps the score of being done once it was.
                score.copy_(torch.where(done, 1.0, score))


def _repeat(state: MDPState, tree: TaskTree, node, complete: torch.Tensor, length: torch.Tensor) -> None:
    """Count the completions of a task that repeats, and restart it, with its subtasks' progress, once its completion
    no longer holds."""
    field = functools.partial(_progress, state, tree, node.path)
    active, done, spent = field("active"), field("done"), field("spent")
    counted = active & complete & ~spent
    spent.logical_or_(counted)
    field("count").add_(counted.int())
    first = counted & ~done
    field("done_at").copy_(torch.where(first, length, field("done_at")))
    done.logical_or_(first)
    restarted = field("restarted")
    torch.logical_and(active & spent, ~field("holding"), out=restarted)
    spent.logical_and_(~restarted)
    initial = {"done": False, "holding": False, "done_at": -1, "score": 0.0, "stage": 0, "streak": 0, "count": 0}
    initial["spent"] = False
    if node.mode is TaskMode.SEQUENCE:
        field("stage").masked_fill_(restarted, 0)
    for child in tree.descendants(node):
        for name, value in initial.items():
            key = progress_field(child.path, name)
            # A choice keeps the subtask it drew.
            if key in state.arrays and not (name == "stage" and child.mode is TaskMode.CHOICE):
                state.arrays[key].torch[tree.worlds.start : tree.worlds.stop].masked_fill_(restarted, value)


def _score(state: MDPState, tree: TaskTree, node) -> torch.Tensor:
    """Return a task's score from its subtasks': their weighted mean, or the mean of the best ``require`` of them."""
    counted = tree.counted(node)
    scores = torch.stack([_progress(state, tree, child.path, "score") for child in counted], 1)
    if node.mode is TaskMode.CHOICE:
        # The score of the drawn subtask.
        stage, score = _progress(state, tree, node.path, "stage"), torch.zeros_like(scores[:, 0])
        for k, child in enumerate(counted):
            score = torch.where(stage == node.children.index(child.path), scores[:, k], score)
        return score
    if node.mode is TaskMode.TOGETHER and node.require is not None:
        return scores.topk(node.require, dim=1).values.mean(1)
    total = sum(child.score for child in counted)
    return sum(scores[:, k] * (child.score / total) for k, child in enumerate(counted))


def _start_tasks(trees: tuple[TaskTree, ...], state: MDPState) -> None:
    """Start the tasks of the sequences that advanced, parents first, and stop the tasks they left."""
    for tree in trees:
        length = state.episode_length.torch[tree.worlds.start : tree.worlds.stop]
        for node in tree.nodes:
            if not node.staged:
                continue
            parent = tree.parent(node)
            active, entered = _progress(state, tree, node.path, "active"), _progress(state, tree, node.path, "entered")
            runs = torch.ones_like(active) if parent is None else _progress(state, tree, parent.path, "active").clone()
            if parent is not None and parent.mode is TaskMode.SEQUENCE:
                # The last task keeps running once it is done.
                stage = _progress(state, tree, parent.path, "stage").clamp(max=len(parent.children) - 1)
                runs &= stage == parent.children.index(node.path)
            elif parent is not None and parent.mode is TaskMode.CHOICE:
                runs &= _progress(state, tree, parent.path, "stage") == parent.children.index(node.path)
            # A task starts where it did not run, and restarts with a parent that started or a repeat.
            fresh = ~active
            if parent is not None and tree.restarts(parent):
                fresh = fresh | _progress(state, tree, parent.path, "entered")
            if node.repeat:
                fresh = fresh | _progress(state, tree, node.path, "restarted")
            torch.logical_and(runs, fresh, out=entered)
            started = _progress(state, tree, node.path, "started_at")
            started.copy_(torch.where(entered, length, started))
            active.copy_(runs)


def _reset_progress(trees: tuple[TaskTree, ...], state: MDPState, mask: torch.Tensor) -> None:
    """Restart the progress of every task in the masked worlds, as at the start of an episode, parents first: a choice
    draws its subtask."""
    for tree in trees:
        selected = mask[tree.worlds.start : tree.worlds.stop]
        for node in tree.nodes:
            initial = {"done": False, "holding": False, "done_at": -1, "started_at": 0, "score": 0.0, "stage": 0}
            initial |= {"streak": 0, "entered": False, "active": tree.runs_at_start(node)}
            initial |= {"count": 0, "spent": False, "restarted": False}
            for name, value in initial.items():
                key = progress_field(node.path, name)
                if key in state.arrays:
                    state.arrays[key].torch[tree.worlds.start : tree.worlds.stop].masked_fill_(selected, value)
            if node.mode is TaskMode.CHOICE:
                draw = rng.uniform(state.rng.torch[node.stream], 0.0, 1.0)[tree.worlds.start : tree.worlds.stop]
                drawn = sum((draw >= bound).int() for bound in itertools.accumulate(node.weights[:-1]))
                stage = _progress(state, tree, node.path, "stage")
                stage.copy_(torch.where(selected, drawn, stage))
            if tree.drawn(node):
                parent, index = tree.parent(node), tree.parent(node).children.index(node.path)
                runs = _progress(state, tree, parent.path, "active")
                if parent.mode is TaskMode.CHOICE:
                    runs = runs & (_progress(state, tree, parent.path, "stage") == index)
                elif parent.mode is TaskMode.SEQUENCE and index > 0:
                    runs = torch.zeros_like(runs)
                active = _progress(state, tree, node.path, "active")
                active.copy_(torch.where(selected, runs, active))
