# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Term descriptors: what a term computes, the state it keeps, and its implementation per backend."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class State:
    """A ``float32`` buffer that a term keeps across steps, with one value per world and element by default."""

    # Resolved like Term.shape. Without "worlds", the worlds share one buffer, which outlives episodes: a hook that
    # launches its own kernels can reduce over the worlds into it, such as a sum with ``wp.atomic_add``.
    shape: tuple[str | int, ...] = ("worlds",)
    initial: float = 0.0  # Value at creation, and on reset when the term has no reset hook
    persistent: bool = False  # Whether it outlives episodes, such as data that the term's ``init`` fills once


@dataclass(frozen=True)
class Implementation:
    """Backend hooks of a term. Hooks take any of the term's inputs, state, params, and ``rng`` by name.

    Fields arrive as arrays of shape ``(num_worlds, ...)``, and subsets as ``int32`` index arrays.

    Torch hooks are batched over worlds:

    - ``compute(**args)`` returns the output. It must not change state: observations are computed again after a
      reset, and continuing worlds must get the same sample.
    - ``update(**args)`` advances state in place, once per step after ``compute``. A hook that writes a field that
      other world configs share, such as a command, also takes ``mask``, the worlds where its op runs (None for
      every world).
    - ``reset(mask, **args)`` resets the masked worlds' state, after the reset events ran and the commands were
      resampled. Without it, state is set to ``initial``.

    Warp hooks are ``wp.func`` s that handle one world, called from the kernels generated for the MDP:

    - ``compute(world, *indices, **args)`` returns one output element, with one index per output dimension after
      ``"worlds"``; ``bool`` for terminations and ``float32`` otherwise.
    - ``update(world, **args)`` and ``reset(world, **args)`` act on one world; ``reset`` runs for reset worlds only.
    - A func gets ``rng`` as its world's ``int32`` seed. It must only write its own world, and may read other
      worlds only where no term writes.

    A Warp hook can instead be a function that launches kernels itself, with the Torch arguments plus ``out`` for
    ``compute``, an array of the output shape to fill for every world, and ``rng`` as the stream of every world.
    It runs between the generated kernels.

    Event and command terms are ``compute(mask, **args)``, or a func ``compute(world, **args)`` run for the selected
    worlds, and write their outputs in place. A command gets its value as ``command``: ``compute`` resamples it and
    writes the command's state, ``update`` may change it every step, and commands have no ``reset`` hook.
    """

    compute: Callable
    update: Callable | None = None
    reset: Callable | None = None


@dataclass(frozen=True)
class Term:
    """A reusable calculation with separate state per use. A function is short for ``Implementation(compute=fn)``."""

    torch: Implementation | Callable | None = None
    warp: Implementation | Callable | None = None
    shape: tuple[str | int, ...] = ("worlds",)  # Output shape; other names are dimensions of inputs or int params
    state: dict[str, State] = field(default_factory=dict)
    random: bool = False  # Pass the use's per-world random stream as ``rng``, see :mod:`.rng`
    eager: Callable | None = None  # ``fn(ctx, **params)`` run eagerly instead of the hooks, see :mod:`.eager`
    init: Callable | None = None  # ``fn(ctx, **params)`` run eagerly once by ``startup``, to fill the state
    capturable: bool = True  # False for Torch or Warp hooks that cannot run inside a CUDA graph

    def __post_init__(self):
        if self.torch is not None and not isinstance(self.torch, Implementation):
            object.__setattr__(self, "torch", Implementation(compute=self.torch))
        if self.warp is not None and not isinstance(self.warp, Implementation):
            object.__setattr__(self, "warp", Implementation(compute=self.warp))

    def __deepcopy__(self, memo: dict) -> Term:
        # Functions and Warp funcs cannot be copied, and a term is immutable, so copies share it.
        return self


def hook_arguments(hook: Callable, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Return the arguments that a Python hook names, or all of them if it takes ``**kwargs``."""
    names = _keyword_names(hook)
    return dict(arguments) if names is None else {name: arguments[name] for name in names if name in arguments}


@functools.cache
def _keyword_names(hook: Callable) -> frozenset[str] | None:
    parameters = inspect.signature(hook).parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return None
    return frozenset(parameter.name for parameter in parameters)
