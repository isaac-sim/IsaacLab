# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch backend: batched term functions, run eagerly one term at a time."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.utils.warp import ProxyArray

from . import rng
from .mdp import MDP, MDPState, TermOp, Transition


def apply_actions(mdp: MDP, state: MDPState, actions: torch.Tensor) -> None:
    state.actions.torch.copy_(actions)
    for op in mdp.action_ops:
        state.fields[op.target].torch.copy_(actions[:, op.columns] * op.scale + op.offset)


def evaluate(mdp: MDP, state: MDPState, transition: Transition) -> None:
    rng.advance(state.rng.torch)
    episode_length = state.episode_length.torch
    episode_length += 1

    terminated, truncated = transition.terminated.torch, transition.truncated.torch
    terminated.zero_()
    truncated.zero_()
    if mdp.max_episode_length is not None:
        truncated |= episode_length >= mdp.max_episode_length
    for op in mdp.termination_ops:
        flags = truncated if op.time_out else terminated
        flags |= _step(op, state).bool()

    rewards = transition.rewards.torch
    rewards.zero_()
    for op in mdp.reward_ops:
        rewards += _step(op, state) * (op.weight * mdp.step_dt)

    for op in mdp.command_ops:
        if op.timer is not None:
            _resample(mdp, op, state, _tick(mdp, op, state))
    for op in mdp.interval_event_ops:
        op.compute(mask=_tick(mdp, op, state), **_arguments(op, state))

    for op in mdp.observation_ops:
        transition.observations[op.group].torch[:, op.columns] = _step(op, state).reshape(mdp.num_worlds, -1)


def reset(mdp: MDP, state: MDPState, mask: torch.Tensor) -> None:
    rng.advance(state.rng.torch, mask)
    state.episode_length.torch.masked_fill_(mask, 0)
    state.actions.torch.masked_fill_(mask[:, None], 0.0)
    for op in mdp.term_ops:
        if op.reset is not None:
            op.reset(mask=mask, **_arguments(op, state))
            continue
        for name, spec in op.state.items():
            state.term_state[op.name][name].torch.view(mdp.num_worlds, -1).masked_fill_(mask[:, None], spec.initial)
    for op in mdp.interval_event_ops + mdp.command_ops:
        if op.timer is not None:
            _restart(op, state, mask)
    for op in mdp.reset_event_ops:
        op.compute(mask=mask, **_arguments(op, state))
    for op in mdp.command_ops:
        _resample(mdp, op, state, mask)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray], mask: torch.Tensor) -> None:
    for op in mdp.observation_ops:
        columns = observations[op.group].torch[:, op.columns]
        value = op.compute(**_arguments(op, state)).reshape(mdp.num_worlds, -1)
        columns.copy_(torch.where(mask[:, None], value, columns))


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


def _resample(mdp: MDP, op: TermOp, state: MDPState, mask: torch.Tensor) -> None:
    """Write fresh command samples to the masked worlds."""
    command = state.fields[op.target].torch.view(mdp.num_worlds, -1)
    value = op.compute(**_arguments(op, state)).reshape(command.shape)
    command.copy_(torch.where(mask[:, None], value, command))


def _step(op: TermOp, state: MDPState) -> torch.Tensor:
    """Compute a term, then advance its state."""
    arguments = _arguments(op, state)
    value = op.compute(**arguments)
    if op.update is not None:
        op.update(**arguments)
    return value


def _arguments(op: TermOp, state: MDPState) -> dict[str, Any]:
    inputs = {arg: state.fields[selector].torch for arg, selector in op.inputs.items()}
    term_state = {name: value.torch for name, value in state.term_state[op.name].items()}
    if op.random:
        term_state["rng"] = state.rng.torch[op.stream]
    return {**inputs, **term_state, **op.params}
