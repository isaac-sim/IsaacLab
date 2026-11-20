# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions that run a compiled MDP on scene data. The environment steps and resets the scene itself.

Each function runs its launches eagerly. To replay them as CUDA graphs, bind fixed buffers with
:class:`~.capture.CapturedMDP`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from .compiled import MDP, MDPState, Transition
from .eager import call_init
from .plan import Call


def startup(mdp: MDP, state: MDPState) -> None:
    """Run the ``init`` hooks of the terms, then the startup events, once, in every world."""
    for op in mdp.term_ops:
        if op.init is not None:
            call_init(mdp, op, state, op.init)

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
    """Compute observations for the masked worlds and leave the others unchanged.

    Recomputes the quantities that observations read, and fills the observation histories of new episodes, in the
    masked worlds. Changes no other MDP state.
    """
    call = Call(state, mask=mask, observations=observations)
    for launch in mdp.launches.observe:
        launch(call)


def tune(
    mdp: MDP,
    state: MDPState,
    path: str,
    value: Any,
    worlds: slice | Sequence[int] | torch.Tensor | None = None,
    at: Literal["reset", "step"] = "reset",
) -> None:
    """Change a value declared :class:`~.config.Tunable`, such as ``"rewards.track.std"`` or ``"rewards.track.weight"``.

    ``value`` is one value, or one per selected world. ``worlds`` selects worlds by index, slice, or ``bool`` mask, all
    of them if None. At ``"reset"``, each world takes the value when its episode next resets, so that no episode mixes
    two values; at ``"step"``, from the next step on, except a ``when``, which uses take at the end of a step and so
    switch one step later. Each world counts the tunings it applied in
    :attr:`MDPState.revision`. Writes in place, so captured steps keep replaying. Raises ``KeyError`` for a value that
    is not tunable.
    """
    tuned = {value.path: value for value in mdp.tuned}
    if path not in tuned:
        raise KeyError(f"{path!r} is not tunable: declare it with mdp2.Tunable. Tunable values: {sorted(tuned)}.")
    if at not in ("reset", "step"):
        raise ValueError(f"Tunings apply at 'reset' or 'step', got {at!r}.")
    target = tuned[path]
    selected = torch.zeros(mdp.num_worlds, dtype=torch.bool, device=mdp.device)
    selected[slice(None) if worlds is None else worlds] = True

    # The value waits in the pending copy; a tuning for this step also writes it in place, and keeps the copy equal.
    destinations = [state.arrays["pending", target.key].torch]
    destinations += [state.arrays[target.key].torch] if at == "step" else []
    for destination in destinations:
        values = destination if target.column is None else destination[:, target.column]
        values[selected] = torch.as_tensor(value, dtype=values.dtype, device=mdp.device)
    if at == "step":
        state.arrays["revision"].torch.add_(selected.int())
    else:
        state.arrays["pending"].torch.logical_or_(selected)
