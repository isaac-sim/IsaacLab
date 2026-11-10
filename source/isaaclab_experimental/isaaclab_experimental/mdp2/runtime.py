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
    """Write endings and rewards, then resample due commands, run due interval events, and observe successors."""
    _BACKENDS[mdp.backend].evaluate(mdp, state, transition)


def reset(mdp: MDP, state: MDPState, mask: torch.Tensor) -> None:
    """Reset term state and timers of the masked worlds, run reset events, then resample their commands."""
    _BACKENDS[mdp.backend].reset(mdp, state, mask)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray], mask: torch.Tensor) -> None:
    """Write fresh observations of the masked worlds; other worlds keep their successor sample."""
    _BACKENDS[mdp.backend].observe(mdp, state, observations, mask)
