# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch backend: batched term functions, run eagerly one term at a time."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.utils.warp import ProxyArray

from .mdp import MDP, MDPState, TermOp, Transition


def apply_actions(mdp: MDP, state: MDPState, actions: torch.Tensor) -> None:
    state.actions.torch.copy_(actions)
    for op in mdp.action_ops:
        state.fields[op.target].torch.copy_(actions[:, op.columns] * op.scale + op.offset)


def evaluate(mdp: MDP, state: MDPState, transition: Transition) -> None:
    episode_length = state.episode_length.torch
    episode_length += 1

    terminated, truncated = transition.terminated.torch, transition.truncated.torch
    terminated.zero_()
    truncated.zero_()
    if mdp.max_episode_length is not None:
        truncated |= episode_length >= mdp.max_episode_length
    for op in mdp.termination_ops:
        flags = truncated if op.truncation else terminated
        flags |= _step(op, state).bool()

    rewards = transition.rewards.torch
    rewards.zero_()
    for op in mdp.reward_ops:
        rewards += _step(op, state) * (op.weight * mdp.step_dt)

    for op in mdp.observation_ops:
        transition.observations[op.group].torch[:, op.columns] = _step(op, state).reshape(mdp.num_worlds, -1)


def reset(mdp: MDP, state: MDPState, mask: torch.Tensor) -> None:
    state.episode_length.torch.masked_fill_(mask, 0)
    state.actions.torch.masked_fill_(mask[:, None], 0.0)
    for op in mdp.termination_ops + mdp.reward_ops + mdp.observation_ops:
        if op.reset is not None:
            op.reset(mask=mask, **_arguments(op, state))
            continue
        for name, spec in op.state.items():
            value = state.term_state[op.name][name].torch
            value.masked_fill_(mask.reshape(-1, *[1] * (value.dim() - 1)), spec.initial)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray], mask: torch.Tensor) -> None:
    for op in mdp.observation_ops:
        columns = observations[op.group].torch[:, op.columns]
        value = op.compute(**_arguments(op, state)).reshape(mdp.num_worlds, -1)
        columns.copy_(torch.where(mask[:, None], value, columns))


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
    return {**inputs, **term_state, **op.params}
