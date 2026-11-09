# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions that run a compiled MDP on scene data views. The environment steps and resets the scene."""

from __future__ import annotations

import torch

from isaaclab.utils.warp import ProxyArray

from . import torch_backend, warp_backend
from .mdp import MDP, Backend, MDPState, Transition

_BACKENDS = {Backend.TORCH: torch_backend, Backend.WARP: warp_backend}


def apply_actions(mdp: MDP, state: MDPState, actions: torch.Tensor) -> None:
    """Store the actions, shape ``(num_worlds, action_dim)``, and write them to their target fields."""
    _BACKENDS[mdp.backend].apply_actions(mdp, state, actions)


def evaluate(mdp: MDP, state: MDPState, transition: Transition) -> None:
    """Advance episode lengths and write endings and rewards, ``sum(weight * value) * step_dt``."""
    _BACKENDS[mdp.backend].evaluate(mdp, state, transition)


def reset(mdp: MDP, state: MDPState, mask: torch.Tensor) -> None:
    """Reset the MDP state of the worlds selected by a boolean mask of shape ``(num_worlds,)``."""
    _BACKENDS[mdp.backend].reset(mdp, state, mask)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray]) -> None:
    """Write the observations of every group."""
    _BACKENDS[mdp.backend].observe(mdp, state, observations)
