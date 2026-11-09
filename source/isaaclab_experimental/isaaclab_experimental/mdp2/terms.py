# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Term descriptors: what a calculation produces and how each backend computes it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TensorSpec:
    """Symbolic output shape. Names other than ``"worlds"`` resolve against the input entities."""

    shape: tuple[str | int, ...] = ("worlds",)


@dataclass(frozen=True)
class Implementation:
    """Backend hooks of a term."""

    compute: Callable
    """``compute(**inputs, **params)``, batched over worlds for Torch."""


@dataclass(frozen=True)
class Term:
    """One reusable calculation. A function is shorthand for ``Implementation(compute=fn)``."""

    torch: Implementation | Callable | None = None
    output: TensorSpec = field(default_factory=TensorSpec)

    def __post_init__(self):
        if self.torch is not None and not isinstance(self.torch, Implementation):
            object.__setattr__(self, "torch", Implementation(compute=self.torch))
