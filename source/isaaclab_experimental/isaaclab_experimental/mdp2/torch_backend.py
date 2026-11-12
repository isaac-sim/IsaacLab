# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch backend: runs batched term functions, one launch per step of the execution plan."""

from __future__ import annotations

import torch

from . import rng
from .compiled import MDP, MDPState
from .config import FieldSelector, Phase, term_state, terminated
from .eager import call_eager
from .plan import Call, Launch, Launches, Step, StepKind, TermOp
from .terms import hook_arguments


def compile_launches(mdp: MDP) -> Launches:
    """Turn every step of the execution plan into a launch, after the bookkeeping that starts each function."""
    return Launches(
        startup=_launches(mdp, mdp.plan.startup),
        apply_actions=(Launch(lambda call: _apply_actions(mdp, call)),),
        evaluate=(Launch(lambda call: _begin_evaluate(mdp, call)), *_launches(mdp, mdp.plan.evaluate, stepping=True)),
        reset=(Launch(_begin_reset), *_launches(mdp, mdp.plan.reset)),
        observe=_launches(mdp, mdp.plan.observe),
    )


def _launches(mdp: MDP, steps: tuple[Step, ...], stepping: bool = False) -> tuple[Launch, ...]:
    """Return one launch per step."""
    return tuple(Launch(lambda call, step=step: _run(mdp, step, call, stepping), step.capturable) for step in steps)


def _apply_actions(mdp: MDP, call: Call) -> None:
    """Store the actions as the last actions, and write each action's scaled columns to its target field."""
    actions = call.actions.torch
    call.state.actions.torch.copy_(actions)

    for step in mdp.plan.apply_actions:
        op = step.op
        target = call.state.arrays[op.target].torch.view(mdp.num_worlds, -1)
        ids = slice(None) if op.subset is None else call.state.arrays[op.subset].torch.long()
        offset = op.offset
        if isinstance(offset, FieldSelector):
            offset = call.state.arrays[offset].torch.view(mdp.num_worlds, -1)[:, ids]
        value = actions[:, op.columns] * op.scale + offset

        active = call.state.arrays.get(("active", op.name))
        if active is not None:
            value = torch.where(active.torch[:, None], value, target[:, ids])
        target[:, ids] = value


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


def _begin_reset(call: Call) -> None:
    """Advance the random streams of the masked worlds, and clear their episode lengths and last actions."""
    mask = call.mask.torch
    rng.advance(call.state.rng.torch, mask)
    call.state.episode_length.torch.masked_fill_(mask, 0)
    call.state.actions.torch.masked_fill_(mask[:, None], 0.0)


def _run(mdp: MDP, step: Step, call: Call, stepping: bool) -> None:
    """Run one step, for the masked worlds if the call has a mask, and only where the op is active."""
    op, state = step.op, call.state
    mask = None if call.mask is None else call.mask.torch
    active = state.arrays.get(("active", op.name))
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
        case StepKind.COMPUTE_IF_DUE:
            due = _tick(mdp, op, state)
            _compute(mdp, op, call, due if active is None else due & active.torch, stepping)
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
        if op.eager:
            call_eager(mdp, op, state, mask)
        else:
            op.compute(mask=mask, **hook_arguments(op.compute, state.arguments[op.name]))
        return

    if op.eager:
        call_eager(mdp, op, state)
        value = state.arrays["scratch", op.name].torch
    else:
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
        flags = transition.truncated if op.time_out else transition.terminated
        flags.torch.logical_or_(value)
        transition.done.torch.logical_or_(value)
        if not op.time_out:
            state.arrays[terminated].torch.logical_or_(value)
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
    destination = call.observations[observation.group].torch[:, observation.columns]
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
    destination.copy_(value if mask is None else torch.where(mask[:, None], value, destination))


def _history(op: TermOp, state: MDPState, value: torch.Tensor, mask: torch.Tensor | None, push: bool) -> torch.Tensor:
    """Push a value into an observation's history, or fill the history of new episodes if not ``push``.

    Returns the history, oldest first, flattened per world.
    """
    history = state.arrays["history", op.name].torch
    selected = torch.ones_like(value[:, 0], dtype=torch.bool) if mask is None else mask

    if push:
        shifted = torch.cat([history[:, 1:], value[:, None]], dim=1)
    else:
        selected = selected & (state.episode_length.torch == 0)
        shifted = value[:, None].expand_as(history)

    history.copy_(torch.where(selected[:, None, None], shifted, history))
    return history.flatten(1)


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
