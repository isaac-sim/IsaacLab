# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch backend: runs batched term functions, one launch per step of the execution plan."""

from __future__ import annotations

import torch

from . import rng
from .config import Phase
from .eager import call_eager
from .mdp import MDP, MDPState
from .plan import Call, Launch, Launches, Step, StepKind, TermOp


def compile_launches(mdp: MDP) -> Launches:
    """Turn every step of the execution plan into a launch, after the bookkeeping that starts each function."""
    return Launches(
        apply_actions=(Launch(lambda call: _apply_actions(mdp, call)),),
        evaluate=(Launch(lambda call: _begin_evaluate(mdp, call)), *_launches(mdp, mdp.plan.evaluate)),
        reset=(Launch(_begin_reset), *_launches(mdp, mdp.plan.reset)),
        observe=_launches(mdp, mdp.plan.observe),
    )


def _launches(mdp: MDP, steps: tuple[Step, ...]) -> tuple[Launch, ...]:
    return tuple(Launch(lambda call, step=step: _run(mdp, step, call), step.capturable) for step in steps)


def _apply_actions(mdp: MDP, call: Call) -> None:
    actions = call.actions.torch
    call.state.actions.torch.copy_(actions)
    for step in mdp.plan.apply_actions:
        op = step.op
        target = call.state.fields[op.target].torch.view(mdp.num_worlds, -1)
        value = actions[:, op.columns] * op.scale + op.offset
        active = call.state.active.get(op.name)
        target.copy_(value if active is None else torch.where(active.torch[:, None], value, target))


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
    transition.rewards.torch.zero_()


def _begin_reset(call: Call) -> None:
    """Advance the random streams of the masked worlds, and clear their episode lengths and last actions."""
    mask = call.mask.torch
    rng.advance(call.state.rng.torch, mask)
    call.state.episode_length.torch.masked_fill_(mask, 0)
    call.state.actions.torch.masked_fill_(mask[:, None], 0.0)


def _run(mdp: MDP, step: Step, call: Call) -> None:
    """Run one step, for the masked worlds if the call has a mask, and only where the op is active."""
    op, state = step.op, call.state
    mask = None if call.mask is None else call.mask.torch
    active = state.active.get(op.name)
    if active is not None:
        mask = active.torch if mask is None else mask & active.torch
    match step.kind:
        case StepKind.COMPUTE:
            _compute(mdp, op, call, mask)
        case StepKind.COMPUTE_IF_DUE:
            due = _tick(mdp, op, state)
            _compute(mdp, op, call, due if active is None else due & active.torch)
        case StepKind.UPDATE_STATE:
            op.update(**state.arguments[op.name])
        case StepKind.RESET_TIMER:
            _restart(op, state, mask)
        case StepKind.RESET_STATE if op.reset is not None:
            op.reset(mask=mask, **state.arguments[op.name])
        case StepKind.RESET_STATE:
            for name, spec in op.state.items():
                values = state.term_state[op.name][name].torch.view(mdp.num_worlds, -1)
                values.masked_fill_(mask[:, None], spec.initial)


def _compute(mdp: MDP, op: TermOp, call: Call, mask: torch.Tensor | None) -> None:
    """Compute an op and write the result to its destination, for the masked worlds if a mask is given."""
    state, transition = call.state, call.transition
    if op.phase in (Phase.INTERVAL_EVENTS, Phase.RESET_EVENTS):
        if op.eager:
            call_eager(mdp, op, state, mask)
        else:
            op.compute(mask=mask, **state.arguments[op.name])
        return
    if op.eager:
        call_eager(mdp, op, state)
        value = state.scratch[op.name].torch
    else:
        value = op.compute(**state.arguments[op.name])
    if mask is not None and op.phase in (Phase.TERMINATIONS, Phase.REWARDS):
        value = torch.where(mask, value, torch.zeros_like(value))
    match op.phase:
        case Phase.TERMINATIONS:
            flags = transition.truncated if op.time_out else transition.terminated
            flags.torch.logical_or_(value)
            transition.done.torch.logical_or_(value)
            if mdp.diagnostics:
                transition.termination_terms.torch[:, op.index] = value
        case Phase.REWARDS:
            value = value * (state.reward_weights.torch[op.index] * mdp.step_dt)
            transition.rewards.torch.add_(value)
            if mdp.diagnostics:
                transition.reward_terms.torch[:, op.index] = value
        case Phase.COMMANDS | Phase.OBSERVATIONS:
            if op.phase is Phase.COMMANDS:
                destination = state.fields[op.writes[0]].torch.view(mdp.num_worlds, -1)
            else:
                destination = call.observations[op.group].torch[:, op.columns]
            value = value.reshape(destination.shape)
            destination.copy_(value if mask is None else torch.where(mask[:, None], value, destination))


def _tick(mdp: MDP, op: TermOp, state: MDPState) -> torch.Tensor:
    """Count down a timer and restart it where it expired. Returns where it expired."""
    time_left = state.timers[op.name].torch
    time_left -= mdp.step_dt
    due = time_left <= 0.0
    _restart(op, state, due)
    return due


def _restart(op: TermOp, state: MDPState, mask: torch.Tensor) -> None:
    time_left = state.timers[op.name].torch
    time_left.copy_(torch.where(mask, rng.uniform(state.rng.torch[op.stream], *op.timer, offset=-1), time_left))
