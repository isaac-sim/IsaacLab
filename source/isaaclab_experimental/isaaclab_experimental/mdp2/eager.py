# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eager terms: Python functions with MDP1-style access to their worlds' data, run outside CUDA graphs.

An eager term, ``Term(eager=fn)``, is called as ``fn(ctx, **params)`` on every backend. It may synchronize with the
host, branch on data, and call any library, so it never runs inside a CUDA graph: it splits the graphs of the
runtime function it belongs to. It gets Torch views of its data through a :class:`TermContext`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .compiled import MDP, MDPState
from .config import FieldSelector, SubsetSelector, term_state
from .plan import TermOp
from .scene import SceneData


@dataclass(frozen=True)
class TermContext:
    """Data that an eager term reads and writes, as Torch views over its worlds."""

    num_worlds: int
    step_dt: float  # Duration of one control step [s]
    inputs: dict[str, torch.Tensor]  # Bound inputs by argument name, read-only
    outputs: dict[str, torch.Tensor]  # Events: bound outputs by argument name, written in place
    state: dict[str, torch.Tensor]  # The term's state by name, written in place
    rng: torch.Tensor | None  # (num_worlds,) the use's random stream, for random terms, see :mod:`.rng`
    mask: torch.Tensor | None  # Events: (num_worlds,) the worlds to act on
    out: torch.Tensor | None  # (num_worlds, *shape) output to fill for every world, except for events
    selectors: Mapping[str, FieldSelector | SubsetSelector]  # The selectors bound to the inputs, by argument name
    scene_data: SceneData  # The scene, whose entity objects eager events may change directly
    cache: dict  # Host objects the use keeps across calls, such as Warp meshes built on its first call

    def entity(self, argument: str) -> Any:
        """Return the scene object, such as an articulation, of the entity that an input selects."""
        return self.scene_data.entity(self.selectors[argument].entity)


def call_eager(
    mdp: MDP, op: TermOp, state: MDPState, mask: torch.Tensor | None = None, hook: Callable | None = None
) -> None:
    """Call an eager term, or another eager ``hook`` of its op, with a context bound to the MDP state.

    An eager term writes its output to the op's scratch buffer.
    """
    arrays = state.arrays
    inputs = {arg: arrays[selector].torch for arg, selector in op.inputs.items() if selector not in op.writes}
    outputs = {arg: arrays[selector].torch for arg, selector in op.inputs.items() if selector in op.writes}

    context = TermContext(
        num_worlds=mdp.num_worlds,
        step_dt=mdp.step_dt,
        inputs=inputs,
        outputs=outputs,
        state={name: arrays[term_state(op.use, name)].torch for name in op.state},
        rng=state.rng.torch[op.stream] if op.random else None,
        mask=mask,
        out=arrays["scratch", op.name].torch if ("scratch", op.name) in arrays else None,
        selectors=op.inputs,
        scene_data=state.scene_data,
        cache=state.caches.setdefault(op.name, {}),
    )

    (op.compute if hook is None else hook)(context, **op.params)
