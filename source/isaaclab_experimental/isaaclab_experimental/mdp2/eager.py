# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Init hooks: Python functions that fill a term's state once, at startup, with MDP1-style access to the scene.

An init hook, ``Term(init=fn)``, is called as ``fn(ctx, **params)`` on every backend, on the host, before the startup
events. It may synchronize with the host, branch on data, and call any library. It gets Torch views of its data
through a :class:`TermContext`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .compiled import MDP, MDPState
from .config import Entity, FieldSelector, SubsetSelector, term_state
from .plan import TermOp
from .scene import SceneData


@dataclass(frozen=True)
class TermContext:
    """Data that an init hook reads and writes, as Torch views over its worlds."""

    num_worlds: int
    step_dt: float  # Duration of one control step [s]
    inputs: dict[str, torch.Tensor]  # Bound inputs by argument name, read-only
    outputs: dict[str, torch.Tensor]  # Events: bound outputs by argument name, written in place
    state: dict[str, torch.Tensor]  # The term's state by name, written in place
    rng: torch.Tensor | None  # (num_worlds,) the use's random stream, for random terms, see :mod:`.rng`
    # The selectors bound to the inputs, by argument name
    selectors: Mapping[str, FieldSelector | SubsetSelector | Entity]
    scene_data: SceneData  # The scene, whose entity objects the hook may read

    def entity(self, argument: str) -> Any:
        """Return the scene object, such as an articulation, of an entity input or of the entity an input selects."""
        selector = self.selectors[argument]
        return self.scene_data.entity(selector.name if isinstance(selector, Entity) else selector.entity)


def call_init(mdp: MDP, op: TermOp, state: MDPState, hook: Callable) -> None:
    """Call an init hook of an op with a context bound to the MDP state."""
    arrays = state.arrays
    # A narrower world config reads its own copy of the last actions, see TermOp.array_key. Entities are not arrays.
    selectors = {arg: selector for arg, selector in op.inputs.items() if not isinstance(selector, Entity)}
    inputs = {arg: arrays[op.array_key(arg)].torch for arg, selector in selectors.items() if selector not in op.writes}
    outputs = {arg: arrays[selector].torch for arg, selector in selectors.items() if selector in op.writes}

    context = TermContext(
        num_worlds=mdp.num_worlds,
        step_dt=mdp.step_dt,
        inputs=inputs,
        outputs=outputs,
        state={name: arrays[term_state(op.use, name)].torch for name in op.state},
        rng=state.rng.torch[op.stream] if op.random else None,
        selectors=op.inputs,
        scene_data=state.scene_data,
    )
    hook(context, **op.params)
