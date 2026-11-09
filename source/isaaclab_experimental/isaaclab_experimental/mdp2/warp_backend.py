# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp backend: each term writes its output buffer, then one kernel each combines the terminations and rewards.

Reward and termination terms write rows of stacked buffers, and observation terms write strided views of their
columns, so a step launches the term kernels plus two fixed kernels and copies nothing.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from .mdp import MDP, MDPState, TermOp, Transition

_NO_LIMIT = 2**31 - 1


@wp.kernel
def _write_action(
    actions: wp.array2d(dtype=wp.float32),
    start: int,
    scale: float,
    offset: float,
    target: wp.array2d(dtype=wp.float32),
):
    world, i = wp.tid()
    target[world, i] = actions[world, start + i] * scale + offset


@wp.kernel
def _end_episodes(
    values: wp.array2d(dtype=wp.bool),
    truncation: wp.array(dtype=wp.bool),
    max_episode_length: int,
    episode_length: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
):
    world = wp.tid()
    length = episode_length[world] + 1
    episode_length[world] = length
    is_terminated = bool(False)
    is_truncated = length >= max_episode_length
    for i in range(values.shape[0]):
        if values[i, world]:
            if truncation[i]:
                is_truncated = True
            else:
                is_terminated = True
    terminated[world] = is_terminated
    truncated[world] = is_truncated


@wp.kernel
def _sum_rewards(
    values: wp.array2d(dtype=wp.float32), scales: wp.array(dtype=wp.float32), rewards: wp.array(dtype=wp.float32)
):
    world = wp.tid()
    total = float(0.0)
    for i in range(values.shape[0]):
        total += values[i, world] * scales[i]
    rewards[world] = total


@wp.kernel
def _reset(
    mask: wp.array(dtype=wp.bool), episode_length: wp.array(dtype=wp.int32), actions: wp.array2d(dtype=wp.float32)
):
    world = wp.tid()
    if mask[world]:
        episode_length[world] = 0
        for i in range(actions.shape[1]):
            actions[world, i] = 0.0


def apply_actions(mdp: MDP, state: MDPState, actions: torch.Tensor) -> None:
    actions = wp.from_torch(actions)
    wp.copy(state.actions.warp, actions)
    for op in mdp.action_ops:
        target = state.views[op.target].warp.reshape((mdp.num_worlds, -1))
        inputs = [actions, op.columns.start, op.scale, op.offset]
        wp.launch(_write_action, dim=target.shape, inputs=inputs, outputs=[target], device=mdp.device)


def evaluate(mdp: MDP, state: MDPState, transition: Transition) -> None:
    terms = state.terms
    for i, op in enumerate(mdp.termination_ops):
        _compute(op, state, terms.terminations[i])
    max_episode_length = _NO_LIMIT if mdp.max_episode_length is None else mdp.max_episode_length
    wp.launch(
        _end_episodes,
        dim=mdp.num_worlds,
        inputs=[terms.terminations, terms.truncation, max_episode_length],
        outputs=[state.episode_length.warp, transition.terminated.warp, transition.truncated.warp],
        device=mdp.device,
    )
    for i, op in enumerate(mdp.reward_ops):
        _compute(op, state, terms.rewards[i])
    inputs = [terms.rewards, terms.reward_scales]
    wp.launch(_sum_rewards, dim=mdp.num_worlds, inputs=inputs, outputs=[transition.rewards.warp], device=mdp.device)


def reset(mdp: MDP, state: MDPState, mask: torch.Tensor) -> None:
    inputs = [wp.from_torch(mask)]
    outputs = [state.episode_length.warp, state.actions.warp]
    wp.launch(_reset, dim=mdp.num_worlds, inputs=inputs, outputs=outputs, device=mdp.device)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray]) -> None:
    for op in mdp.observation_ops:
        _compute(op, state, _columns(observations[op.group].warp, op.columns.start, op.shape))


def _compute(op: TermOp, state: MDPState, out: wp.array) -> None:
    op.compute(out=out, **{arg: state.views[selector].warp for arg, selector in op.inputs.items()}, **op.params)


def _columns(buffer: wp.array, start: int, shape: tuple[int, ...]) -> wp.array:
    """A view of shape ``(num_worlds, *shape)`` into the columns of a 2D buffer, starting at ``start``."""
    item = buffer.strides[1]
    inner = [math.prod(shape[k + 1 :]) * item for k in range(len(shape))]
    return wp.array(
        ptr=buffer.ptr + start * item,
        dtype=buffer.dtype,
        shape=(buffer.shape[0], *shape),
        strides=(buffer.strides[0], *inner),
        device=buffer.device,
    )
