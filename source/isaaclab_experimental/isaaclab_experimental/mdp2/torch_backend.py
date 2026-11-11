# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch backend: runs batched term functions eagerly, one step at a time."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from isaaclab.utils.warp import ProxyArray

from . import rng
from .config import Phase
from .mdp import MDP, MDPState, Transition
from .plan import Step, StepKind, TermOp


def apply_actions(mdp: MDP, state: MDPState, actions: ProxyArray) -> None:
    state.actions.torch.copy_(actions.torch)
    for step in mdp.plan.apply_actions:
        op = step.op
        state.fields[op.target].torch.copy_(actions.torch[:, op.columns] * op.scale + op.offset)


def evaluate(mdp: MDP, state: MDPState, transition: Transition) -> None:
    rng.advance(state.rng.torch)
    state.episode_length.torch.add_(1)
    transition.terminated.torch.zero_()
    transition.truncated.torch.zero_()
    if mdp.max_episode_length is not None:
        transition.truncated.torch.copy_(state.episode_length.torch >= mdp.max_episode_length)
    transition.rewards.torch.zero_()
    _run(mdp, mdp.plan.evaluate, state, None, transition, transition.observations)
    transition.done.torch.copy_(transition.terminated.torch | transition.truncated.torch)


def reset(mdp: MDP, state: MDPState, mask: ProxyArray) -> None:
    rng.advance(state.rng.torch, mask.torch)
    state.episode_length.torch.masked_fill_(mask.torch, 0)
    state.actions.torch.masked_fill_(mask.torch[:, None], 0.0)
    _run(mdp, mdp.plan.reset, state, mask.torch)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray], mask: ProxyArray) -> None:
    _run(mdp, mdp.plan.observe, state, mask.torch, observations=observations)


def _run(
    mdp: MDP,
    steps: Iterable[Step],
    state: MDPState,
    mask: torch.Tensor | None,
    transition: Transition | None = None,
    observations: dict[str, ProxyArray] | None = None,
) -> None:
    """Run the steps of a runtime function, for the masked worlds if a mask is given."""
    for step in steps:
        op, arguments = step.op, state.arguments[step.op.name]
        match step.kind:
            case StepKind.COMPUTE:
                _compute(mdp, op, state, mask, transition, observations)
            case StepKind.COMPUTE_IF_DUE:
                _compute(mdp, op, state, _tick(mdp, op, state), transition, observations)
            case StepKind.UPDATE_STATE:
                op.update(**arguments)
            case StepKind.RESET_TIMER:
                _restart(op, state, mask)
            case StepKind.RESET_STATE if op.reset is not None:
                op.reset(mask=mask, **arguments)
            case StepKind.RESET_STATE:
                for name, spec in op.state.items():
                    values = state.term_state[op.name][name].torch.view(mdp.num_worlds, -1)
                    values.masked_fill_(mask[:, None], spec.initial)


def _compute(
    mdp: MDP,
    op: TermOp,
    state: MDPState,
    mask: torch.Tensor | None,
    transition: Transition | None,
    observations: dict[str, ProxyArray] | None,
) -> None:
    """Compute an op and write the result to its destination, for the masked worlds if a mask is given."""
    arguments = state.arguments[op.name]
    if op.phase in (Phase.INTERVAL_EVENTS, Phase.RESET_EVENTS):
        op.compute(mask=mask, **arguments)
        return
    value = op.compute(**arguments)
    match op.phase:
        case Phase.TERMINATIONS:
            flags = transition.truncated if op.time_out else transition.terminated
            flags.torch.logical_or_(value)
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
                destination = observations[op.group].torch[:, op.columns]
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
