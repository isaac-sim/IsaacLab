# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions that run a compiled MDP on scene data. The environment steps and resets the scene itself."""

from __future__ import annotations

from isaaclab.utils.warp import ProxyArray

from . import torch_backend, warp_backend
from .mdp import MDP, Backend, MDPState, Transition

_BACKENDS = {Backend.TORCH: torch_backend, Backend.WARP: warp_backend}


def apply_actions(mdp: MDP, state: MDPState, actions: ProxyArray) -> None:
    """Store the actions, shape ``(num_worlds, action_dim)``, and write them to their target fields."""
    _BACKENDS[mdp.backend].apply_actions(mdp, state, actions)


def evaluate(mdp: MDP, state: MDPState, transition: Transition) -> None:
    """Compute terminations and rewards, resample due commands, run due interval events, and observe."""
    _BACKENDS[mdp.backend].evaluate(mdp, state, transition)


def reset(mdp: MDP, state: MDPState, mask: ProxyArray) -> None:
    """Reset the masked worlds: restart timers, run reset events, resample commands, then reset term state."""
    _BACKENDS[mdp.backend].reset(mdp, state, mask)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray], mask: ProxyArray) -> None:
    """Compute observations for the masked worlds and leave the others unchanged. Changes no MDP state."""
    _BACKENDS[mdp.backend].observe(mdp, state, observations, mask)
