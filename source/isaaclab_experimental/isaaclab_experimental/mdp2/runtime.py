# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions that run a compiled MDP on scene data. The environment steps and resets the scene itself.

Each function runs its launches eagerly. To replay them as CUDA graphs, bind fixed buffers with
:class:`~.capture.CapturedMDP`.
"""

from __future__ import annotations

from isaaclab.utils.warp import ProxyArray

from .compiled import MDP, MDPState, Transition
from .plan import Call


def apply_actions(mdp: MDP, state: MDPState, actions: ProxyArray) -> None:
    """Store the actions, shape ``(num_worlds, action_dim)``, and write them to their target fields."""
    call = Call(state, actions=actions)
    for launch in mdp.launches.apply_actions:
        launch(call)


def evaluate(mdp: MDP, state: MDPState, transition: Transition) -> None:
    """Compute terminations and rewards, resample due commands, run due interval events, and observe."""
    call = Call(state, transition=transition, observations=transition.observations)
    for launch in mdp.launches.evaluate:
        launch(call)


def reset(mdp: MDP, state: MDPState, mask: ProxyArray) -> None:
    """Reset the masked worlds: restart timers, run reset events, resample commands, then reset term state."""
    call = Call(state, mask=mask)
    for launch in mdp.launches.reset:
        launch(call)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray], mask: ProxyArray) -> None:
    """Compute observations for the masked worlds and leave the others unchanged. Changes no MDP state."""
    call = Call(state, mask=mask, observations=observations)
    for launch in mdp.launches.observe:
        launch(call)
