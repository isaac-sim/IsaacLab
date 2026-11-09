# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions that run a compiled MDP on scene data views. The environment steps and resets the scene."""

from __future__ import annotations

import torch

from .mdp import MDP, MDPState, Observations, Transition


def apply_actions(mdp: MDP, state: MDPState, actions: torch.Tensor) -> None:
    """Store the actions, shape ``(num_worlds, action_dim)``, and write them to their target fields."""
    state.actions.copy_(actions)
    for op in mdp.action_ops:
        state.bindings[op.name]["target"].copy_(actions[:, op.columns] * op.scale + op.offset)


def evaluate(mdp: MDP, state: MDPState, transition: Transition) -> None:
    """Advance episode lengths and write endings and rewards, ``sum(weight * value) * step_dt``."""
    state.episode_length += 1

    transition.terminated.zero_()
    transition.truncated.zero_()
    if mdp.max_episode_length is not None:
        transition.truncated |= state.episode_length >= mdp.max_episode_length
    for op in mdp.termination_ops:
        flags = transition.truncated if op.truncation else transition.terminated
        flags |= op.compute(**state.bindings[op.name], **op.params).bool()

    transition.rewards.zero_()
    for op in mdp.reward_ops:
        transition.rewards += op.compute(**state.bindings[op.name], **op.params) * (op.weight * mdp.step_dt)


def reset(mdp: MDP, state: MDPState, mask: torch.Tensor) -> None:
    """Reset the MDP state of the worlds selected by a boolean mask of shape ``(num_worlds,)``."""
    state.episode_length.masked_fill_(mask, 0)
    state.actions.masked_fill_(mask[:, None], 0.0)


def observe(mdp: MDP, state: MDPState, observations: Observations) -> None:
    """Write the observations of every group."""
    for op in mdp.observation_ops:
        value = op.compute(**state.bindings[op.name], **op.params)
        observations[op.group][:, op.columns] = value.reshape(mdp.num_worlds, -1)
