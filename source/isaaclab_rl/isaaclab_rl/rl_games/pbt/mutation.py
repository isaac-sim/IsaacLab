# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import random
from collections.abc import Callable
from typing import Any


def mutate_float(x: float, change_min: float = 1.1, change_max: float = 1.5) -> float:
    """Multiply or divide by a random factor in [change_min, change_max]."""
    k = random.uniform(change_min, change_max)
    return x / k if random.random() < 0.5 else x * k


def mutate_discount(x: float, **kwargs) -> float:
    """Conservative change near 1.0 by mutating (1 - x) in [1.1, 1.2]."""
    inv = 1.0 - x
    new_inv = mutate_float(inv, change_min=1.1, change_max=1.2)
    return 1.0 - new_inv


MUTATION_FUNCS: dict[str, Callable[..., Any]] = {
    "mutate_float": mutate_float,
    "mutate_discount": mutate_discount,
}


def mutate(
    params: dict[str, Any],
    mutations: dict[str, str],
    mutation_rate: float,
    change_range: tuple[float, float],
) -> dict[str, Any]:
    cmin, cmax = change_range
    out: dict[str, Any] = {}
    for name, val in params.items():
        fn_name = mutations.get(name)
        # skip if no rule or coin flip says "no"
        if fn_name is None or random.random() > mutation_rate:
            out[name] = val
            continue
        fn = MUTATION_FUNCS.get(fn_name)
        if fn is None:
            raise KeyError(f"Unknown mutation function: {fn_name!r}")
        out[name] = fn(val, change_min=cmin, change_max=cmax)
    return out
