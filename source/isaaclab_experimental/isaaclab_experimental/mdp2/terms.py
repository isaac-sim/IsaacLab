# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Term descriptors: what a calculation produces, the state it keeps, and how each backend computes it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class State:
    """A ``float32`` buffer a term keeps across steps, with one value per world and element."""

    shape: tuple[str | int, ...] = ("worlds",)  # Resolved like Term.shape
    initial: float = 0.0  # Value at creation, and on reset when the term has no reset hook


@dataclass(frozen=True)
class Implementation:
    """Backend hooks of a term. Hooks take the term's inputs, state, and params as keyword arguments.

    - ``compute(**args)`` returns a tensor batched over worlds (Torch), or ``compute(out, **args)`` writes the
      output into ``out``, ``bool`` for terminations and ``float32`` otherwise (Warp). It must not change state:
      observations are computed again after a reset, and continuing worlds must get the same sample.
    - ``update(**args)`` advances state in place, once per step after ``compute``.
    - ``reset(mask, **args)`` resets the masked worlds' state. Without it, state is set to ``initial``.

    Event terms are ``compute(mask, **args)`` in both backends and write their inputs in place.
    """

    compute: Callable
    update: Callable | None = None
    reset: Callable | None = None


@dataclass(frozen=True)
class Term:
    """One reusable calculation; each use gets its own state. A function means ``Implementation(compute=fn)``."""

    torch: Implementation | Callable | None = None
    warp: Implementation | Callable | None = None
    shape: tuple[str | int, ...] = ("worlds",)  # Output shape; names other than "worlds" come from the inputs
    state: dict[str, State] = field(default_factory=dict)
    random: bool = False  # Pass the use's per-world random stream as ``rng``, see :func:`uniform`

    def __post_init__(self):
        if self.torch is not None and not isinstance(self.torch, Implementation):
            object.__setattr__(self, "torch", Implementation(compute=self.torch))
        if self.warp is not None and not isinstance(self.warp, Implementation):
            object.__setattr__(self, "warp", Implementation(compute=self.warp))
