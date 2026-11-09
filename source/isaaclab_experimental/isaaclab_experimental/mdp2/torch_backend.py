# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch backend: batched term functions, run eagerly one term at a time."""

from __future__ import annotations

import torch

from isaaclab.utils.warp import ProxyArray

from .mdp import MDP, MDPState, TermOp, Transition


def apply_actions(mdp: MDP, state: MDPState, actions: torch.Tensor) -> None:
    state.actions.torch.copy_(actions)
    for op in mdp.action_ops:
        state.views[op.target].torch.copy_(actions[:, op.columns] * op.scale + op.offset)


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
        flags |= _compute(op, state).bool()

    rewards = transition.rewards.torch
    rewards.zero_()
    for op in mdp.reward_ops:
        rewards += _compute(op, state) * (op.weight * mdp.step_dt)


def reset(mdp: MDP, state: MDPState, mask: torch.Tensor) -> None:
    state.episode_length.torch.masked_fill_(mask, 0)
    state.actions.torch.masked_fill_(mask[:, None], 0.0)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray]) -> None:
    for op in mdp.observation_ops:
        observations[op.group].torch[:, op.columns] = _compute(op, state).reshape(mdp.num_worlds, -1)


def _compute(op: TermOp, state: MDPState) -> torch.Tensor:
    return op.compute(**{arg: state.views[selector].torch for arg, selector in op.inputs.items()}, **op.params)
