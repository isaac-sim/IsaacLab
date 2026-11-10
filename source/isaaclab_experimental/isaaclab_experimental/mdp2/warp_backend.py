# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp backend: terms write into output buffers, then fixed kernels combine terminations and rewards."""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray, memory

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
    is_truncation: wp.array(dtype=wp.bool),
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
            if is_truncation[i]:
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
def _tick(
    step_dt: float,
    low: float,
    high: float,
    rng: wp.array(dtype=wp.int32),
    time_left: wp.array(dtype=wp.float32),
    due: wp.array(dtype=wp.bool),
):
    world = wp.tid()
    remaining = time_left[world] - step_dt
    expired = remaining <= 0.0
    if expired:
        remaining = wp.randf(wp.rand_init(rng[world], -1), low, high)
    time_left[world] = remaining
    due[world] = expired


@wp.kernel
def _restart(
    mask: wp.array(dtype=wp.bool),
    low: float,
    high: float,
    rng: wp.array(dtype=wp.int32),
    time_left: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    if mask[world]:
        time_left[world] = wp.randf(wp.rand_init(rng[world], -1), low, high)


@wp.kernel
def _advance(mask: wp.array(dtype=wp.bool), rng: wp.array2d(dtype=wp.int32)):
    stream, world = wp.tid()
    if mask and not mask[world]:
        return
    rng[stream, world] = wp.int32(wp.rand_init(rng[stream, world]))


def apply_actions(mdp: MDP, state: MDPState, actions: torch.Tensor) -> None:
    actions = wp.from_torch(actions)
    wp.copy(state.actions.warp, actions)
    for op in mdp.action_ops:
        target = state.fields[op.target].warp.reshape((mdp.num_worlds, -1))
        inputs = [actions, op.columns.start, op.scale, op.offset]
        wp.launch(_write_action, dim=target.shape, inputs=inputs, outputs=[target], device=mdp.device)


def evaluate(mdp: MDP, state: MDPState, transition: Transition) -> None:
    outputs = state.outputs
    for i, op in enumerate(mdp.termination_ops):
        _step(op, state, outputs.terminations[i])
    max_episode_length = _NO_LIMIT if mdp.max_episode_length is None else mdp.max_episode_length
    wp.launch(
        _end_episodes,
        dim=mdp.num_worlds,
        inputs=[outputs.terminations, outputs.is_truncation, max_episode_length],
        outputs=[state.episode_length.warp, transition.terminated.warp, transition.truncated.warp],
        device=mdp.device,
    )
    for i, op in enumerate(mdp.reward_ops):
        _step(op, state, outputs.rewards[i])
    inputs = [outputs.rewards, outputs.reward_scales]
    wp.launch(_sum_rewards, dim=mdp.num_worlds, inputs=inputs, outputs=[transition.rewards.warp], device=mdp.device)
    for op in mdp.command_ops:
        if op.timer is not None:
            _resample(mdp, op, state, _tick_timer(mdp, op, state))
    for op in mdp.interval_event_ops:
        op.compute(mask=_tick_timer(mdp, op, state), **_arguments(op, state))

    for op in mdp.observation_ops:
        _step(op, state, _columns(transition.observations[op.group].warp, op.columns.start, op.shape))
    wp.launch(_advance, state.rng.warp.shape, [None], [state.rng.warp], device=mdp.device)


def reset(mdp: MDP, state: MDPState, mask: torch.Tensor) -> None:
    mask = wp.from_torch(mask)
    episode_length, actions = state.episode_length.warp, state.actions.warp
    fill = memory.make_masked_fill(wp.float32, 2)
    wp.launch(
        memory.make_masked_fill(wp.int32, 1), episode_length.shape, [0, mask, None], [episode_length], device=mdp.device
    )
    wp.launch(fill, actions.shape, [0.0, mask, None], [actions], device=mdp.device)
    for op in mdp.term_ops:
        if op.reset is not None:
            op.reset(mask=mask, **_arguments(op, state))
            continue
        for name, spec in op.state.items():
            values = state.term_state[op.name][name].warp.reshape((mdp.num_worlds, -1))
            wp.launch(fill, values.shape, [spec.initial, mask, None], [values], device=mdp.device)
    for op in mdp.interval_event_ops:
        _restart_timer(mdp, op, state, mask)
    for op in mdp.reset_event_ops:
        op.compute(mask=mask, **_arguments(op, state))
    for op in mdp.command_ops:
        if op.timer is not None:
            _restart_timer(mdp, op, state, mask)
        _resample(mdp, op, state, mask)
    wp.launch(_advance, state.rng.warp.shape, [mask], [state.rng.warp], device=mdp.device)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray], mask: torch.Tensor) -> None:
    fresh = state.outputs.observations
    for op in mdp.observation_ops:
        op.compute(out=_columns(fresh[op.group], op.columns.start, op.shape), **_arguments(op, state))
    mask = wp.from_torch(mask)
    for group, values in fresh.items():
        outputs = [observations[group].warp]
        wp.launch(
            memory.make_masked_copy(wp.float32, 2), values.shape, [values, mask, None], outputs, device=mdp.device
        )
    wp.launch(_advance, state.rng.warp.shape, [mask], [state.rng.warp], device=mdp.device)


def _tick_timer(mdp: MDP, op: TermOp, state: MDPState) -> wp.array:
    """Count down a timer and restart it where it expired. Returns where it expired."""
    due = state.outputs.due[op.name]
    inputs = [mdp.step_dt, *op.timer, state.rng.warp[op.stream]]
    wp.launch(_tick, mdp.num_worlds, inputs, [state.timers[op.name].warp, due], device=mdp.device)
    return due


def _restart_timer(mdp: MDP, op: TermOp, state: MDPState, mask: wp.array) -> None:
    inputs = [mask, *op.timer, state.rng.warp[op.stream]]
    wp.launch(_restart, mdp.num_worlds, inputs, [state.timers[op.name].warp], device=mdp.device)


def _resample(mdp: MDP, op: TermOp, state: MDPState, mask: wp.array) -> None:
    """Write fresh command samples to the masked worlds."""
    samples = state.outputs.commands[op.name]
    op.compute(out=_columns(samples, 0, op.shape), **_arguments(op, state))
    command = state.fields[op.target].warp.reshape((mdp.num_worlds, -1))
    wp.launch(
        memory.make_masked_copy(wp.float32, 2), samples.shape, [samples, mask, None], [command], device=mdp.device
    )


def _step(op: TermOp, state: MDPState, out: wp.array) -> None:
    """Compute a term into ``out``, then advance its state."""
    arguments = _arguments(op, state)
    op.compute(out=out, **arguments)
    if op.update is not None:
        op.update(**arguments)


def _arguments(op: TermOp, state: MDPState) -> dict[str, Any]:
    inputs = {arg: state.fields[selector].warp for arg, selector in op.inputs.items()}
    term_state = {name: value.warp for name, value in state.term_state[op.name].items()}
    if op.random:
        term_state["rng"] = state.rng.warp[op.stream]
    return {**inputs, **term_state, **op.params}


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
