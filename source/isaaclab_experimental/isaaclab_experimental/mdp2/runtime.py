# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions that run a compiled MDP on scene data. The environment steps and resets the scene itself.

Each function runs its launches eagerly. To replay them as CUDA graphs, bind fixed buffers with
:class:`~.capture.CapturedMDP`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from .compiled import MDP, MDPState, Transition
from .config import FieldSelector
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


def process_actions(mdp: MDP, state: MDPState, actions: ProxyArray) -> None:
    """Store the actions, shape ``(num_worlds, action_dim)``, and apply those that apply once per control step, see
    :class:`~.config.ActionRate`. Call it once per control step, before :func:`apply_actions`. Without such actions it
    does nothing, and :func:`apply_actions` stores the actions."""
    call = Call(state, actions=actions)
    for launch in mdp.launches.process_actions:
        launch(call)


def apply_actions(mdp: MDP, state: MDPState, actions: ProxyArray) -> None:
    """Write the actions that apply before every physics step to their target fields; call it before each."""
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

    # A tuning for the next reset waits in the pending copy, and sets the value's bit; a tuning for this step writes
    # the value in place, and drops what waited for the next reset, as it is the newer tuning.
    destination = state.arrays[("pending", target.key) if at == "reset" else target.key].torch
    for values in [destination[:, column] for column in target.columns] or [destination]:
        values[selected] = torch.as_tensor(value, dtype=values.dtype, device=mdp.device)
    word, bit = divmod(mdp.tuned.index(target), 32)
    flags = state.arrays["pending"].torch[:, word]
    if at == "reset":
        flags.bitwise_or_(selected.int() << bit)
    else:
        flags.bitwise_and_(~(selected.int() << bit))
        state.arrays["revision"].torch.add_(selected.int())


_CARRIED = ("timer", "history", "on", "entered")
"""Kinds of the arrays of single ops that :func:`migrate` carries over, by the first part of their keys."""


def migrate(previous: MDP, previous_state: MDPState, mdp: MDP, state: MDPState) -> None:
    """Carry the state of an MDP over to another built on the same scene, such as for an edited task.

    Every array of the MDP that both key alike and shape alike carries over: the state, timers, and histories of the
    uses that both have, the commands, quantities, and progress, the episode lengths, and the last actions. So do the
    random streams of the uses that both have, and the values that both declare :class:`~.config.Tunable`, with those
    waiting for the next reset, as they may have been tuned; other reward weights and params are the new task's. The
    init hooks of the uses that the previous MDP did not have, or whose params changed, run, as at startup.
    """
    for key, array in state.arrays.items():
        carried = isinstance(key, FieldSelector) and not key.scene or key == "revision"
        carried |= isinstance(key, tuple) and key[0] in _CARRIED
        old = previous_state.arrays.get(key)
        if (
            carried
            and old is not None
            and old.torch.shape == array.torch.shape
            and old.torch.dtype == array.torch.dtype
        ):
            array.torch.copy_(old.torch)

    for row, name in enumerate(mdp.streams):
        if name in previous.streams:
            state.rng.torch[row] = previous_state.rng.torch[previous.streams.index(name)]
    _migrate_tuned(previous, previous_state, mdp, state)

    params = {op.use: op.params for op in previous.term_ops}
    for op in mdp.term_ops:
        if op.init is not None and params.get(op.use) != op.params:
            call_init(mdp, op, state, op.init)


def _migrate_tuned(previous: MDP, previous_state: MDPState, mdp: MDP, state: MDPState) -> None:
    """Carry over the tunable values that both MDPs have: the reward weights, which the arrays did not, and what waits
    for the next reset, with the bits that mark it."""
    indices = {value.path: index for index, value in enumerate(previous.tuned)}
    for index, value in enumerate(mdp.tuned):
        if value.path not in indices:
            continue
        old_index = indices[value.path]
        old = previous.tuned[old_index]
        for key in [value.key, ("pending", value.key)] if value.columns else [("pending", value.key)]:
            source, target = previous_state.arrays[key].torch, state.arrays[key].torch
            if value.columns:  # By column, as rewards may have moved
                for column, old_column in zip(value.columns, itertools.cycle(old.columns)):
                    target[:, column] = source[:, old_column]
            elif source.shape == target.shape and source.dtype == target.dtype:
                target.copy_(source)
        word, bit = divmod(index, 32)
        old_word, old_bit = divmod(old_index, 32)
        waits = (previous_state.arrays["pending"].torch[:, old_word] >> old_bit) & 1
        state.arrays["pending"].torch[:, word].bitwise_or_(waits << bit)
