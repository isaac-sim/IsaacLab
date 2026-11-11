# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions that run a compiled MDP on scene data. The environment steps and resets the scene itself.

Each function runs its launches eagerly. To replay them as CUDA graphs, bind fixed buffers with
:class:`~.capture.CapturedMDP`.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from isaaclab.utils.warp import ProxyArray

from .compiled import MDP, MDPState, Transition
from .eager import call_eager
from .plan import Call


def startup(mdp: MDP, state: MDPState) -> None:
    """Run the ``init`` hooks of the terms, then the startup events, once, in every world."""
    for op in mdp.term_ops:
        if op.init is not None:
            call_eager(mdp, op, state, hook=op.init)

    mask = ProxyArray(wp.ones(mdp.num_worlds, dtype=wp.bool, device=mdp.device))
    call = Call(state, mask=mask)
    for launch in mdp.launches.startup:
        launch(call)


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


def reset(mdp: MDP, state: MDPState, mask: ProxyArray, commit: Callable[[], None] | None = None) -> None:
    """Reset the masked worlds: restart timers, run reset events, resample commands, then reset term state.

    ``commit`` makes the scene see the writes of the reset events so far. It is called before every event that asks
    for it, and tasks without such events need none. The caller commits the writes once more after the reset.
    """
    call = Call(state, mask=mask, commit=commit)
    for launch in mdp.launches.reset:
        launch(call)


def observe(mdp: MDP, state: MDPState, observations: dict[str, ProxyArray], mask: ProxyArray) -> None:
    """Compute observations for the masked worlds and leave the others unchanged. Changes no MDP state."""
    call = Call(state, mask=mask, observations=observations)
    for launch in mdp.launches.observe:
        launch(call)
