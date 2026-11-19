# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantities that combine conditions into task progress and success.

Conditions are quantities of one value per world, true where positive. The combinators take any number of them as
inputs, under any names other than the reserved ``cache``, in declaration order::

    quantities = {
        "lifted": mdp2.Quantity(lifted, inputs=...),
        "placed": mdp2.Quantity(placed, inputs=...),
        "success": mdp2.Quantity(
            mdp.sequence, inputs={"lift": mdp2.quantities.lifted, "place": mdp2.quantities.placed}, record=True
        ),
    }

They run as Torch hooks, also in a Warp MDP, between its kernels.
"""

from __future__ import annotations

import torch

from ..terms import State, Term


def _all_of(cache: dict, **conditions: torch.Tensor) -> torch.Tensor:
    return torch.stack(list(conditions.values())).gt(0.0).all(dim=0).float()


def _any_of(cache: dict, **conditions: torch.Tensor) -> torch.Tensor:
    return torch.stack(list(conditions.values())).gt(0.0).any(dim=0).float()


all_of = Term(torch=_all_of)
"""One where every condition holds, zero elsewhere."""

any_of = Term(torch=_any_of)
"""One where any condition holds, zero elsewhere."""


def _sequence(stage: torch.Tensor, cache: dict, **conditions: torch.Tensor) -> torch.Tensor:
    # The condition of each world's current stage; worlds past the last stage stay there.
    met = torch.stack(list(conditions.values()), dim=1).gt(0.0)
    current = stage.long().clamp(max=met.shape[1] - 1)
    advance = met.gather(1, current[:, None])[:, 0] & (stage < met.shape[1])
    stage.add_(advance.float())
    return (stage >= met.shape[1]).float()


sequence = Term(torch=_sequence, state={"stage": State()})
"""One once the conditions held in order, each at a later step than the one before, zero until then. Its state
``stage`` counts the conditions met so far in the episode, as subtask progress: read it with
``mdp2.term_state("quantities.<name>", "stage")``."""
