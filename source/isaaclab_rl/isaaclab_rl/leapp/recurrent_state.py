# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Framework-neutral recurrent-state helpers for LEAPP policy export."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def is_two_tensor_lstm_state(states: object) -> bool:
    """Return whether *states* looks like an LSTM ``[hidden, cell]`` state."""
    return (
        isinstance(states, (list, tuple))
        and len(states) == 2
        and all(isinstance(state, torch.Tensor) for state in states)
    )


def state_dict_from_sequence(states: Sequence[torch.Tensor], prefix: str = "actor_state") -> dict[str, torch.Tensor]:
    """Convert an ordered recurrent-state sequence to a LEAPP named-state mapping."""
    return {f"{prefix}_{index}": state for index, state in enumerate(states)}


def state_sequence_from_registered(
    registered_state: object,
    names: Sequence[str],
    original_states: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Restore registered LEAPP state to the ordered sequence expected by an RL framework."""
    if isinstance(registered_state, dict):
        return [registered_state[name] for name in names]
    if isinstance(registered_state, (list, tuple)):
        return list(registered_state)
    if len(original_states) == 1:
        return [registered_state]
    raise TypeError(f"Expected registered recurrent state for {list(names)}, got {type(registered_state).__name__}.")
