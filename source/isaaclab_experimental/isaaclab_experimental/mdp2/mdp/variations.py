# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Variations: values of the scene that each episode draws, records, and can replay, such as how far a drawer is open.

A variation is a reset event that draws one value per world and writes it to elements of a scene field, and a quantity
that holds the value, recorded at the end of every episode. :func:`variation` returns both as a part of a task, to
compose with the others::

    task = mdp2.compose(
        SCENE,
        EMBODIMENT,
        TASK,
        mdp.variation("drawer", cabinet.joint_pos, cabinet.joints("drawer"), mdp.Uniform(0.0, 0.35)),
    )
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import warp as wp

from ..config import Event, FieldSelector, Quantity, SubsetSelector, TaskCfg, Tunable, term_state
from ..eager import TermContext
from ..rng import uniform
from ..terms import State, Term
from .rewards import value


@dataclass(frozen=True)
class Uniform:
    """Values drawn uniformly from ``[low, high)``. Both are tunable, as ``events.<name>.low`` and ``high``."""

    low: float
    high: float


@dataclass(frozen=True)
class Choice:
    """One of ``values``, drawn with the relative chances ``weights``, even ones if None."""

    values: tuple[float, ...]
    weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Replay:
    """``values`` in the order the resets take them, such as those that an earlier run recorded, from the first again
    once all were taken. Worlds that reset at the same step take them in world order, so the order of the values does
    not depend on the number of worlds. The row each episode took is recorded as ``<name>_row``."""

    values: tuple[float, ...]


def variation(name: str, target: FieldSelector, subset: SubsetSelector, values: Uniform | Choice | Replay) -> TaskCfg:
    """Return a part of a task that draws a value at every reset, writes it to the ``subset`` elements of the
    ``target`` field, such as a drawer's joint position, and records it per episode as the quantity ``name``."""
    inputs = {"ids": subset}
    if isinstance(values, Uniform):
        term, params = uniform_variation, {"low": Tunable(float(values.low)), "high": Tunable(float(values.high))}
    elif isinstance(values, Choice):
        weights = values.weights or (1.0,) * len(values.values)
        bounds = tuple(itertools.accumulate(weight / sum(weights) for weight in weights[:-1]))
        params = {"values": tuple(map(float, values.values)), "chances": bounds}
        params |= {"num_values": len(values.values), "num_bounds": max(len(bounds), 1)}
        term = choice_variation
    else:
        term, params = replay_variation, {"values": tuple(map(float, values.values)), "num_values": len(values.values)}
    event = Event(term, inputs=inputs, outputs={"target": target}, params=params)

    quantities = {name: Quantity(value, inputs={"value": term_state(f"events.{name}", "value")}, record=True)}
    if isinstance(values, Replay):
        row = Quantity(value, inputs={"value": term_state(f"events.{name}", "row")}, record=True)
        quantities[f"{name}_row"] = row
    return TaskCfg(events={name: event}, quantities=quantities)


def _write(target: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor, drawn: torch.Tensor, kept: torch.Tensor):
    """Keep the drawn values of the masked worlds, and write them to the elements ``ids`` of the target."""
    kept.copy_(torch.where(mask, drawn, kept))
    elements = target[:, ids.long()]
    target[:, ids.long()] = torch.where(mask[:, None], kept[:, None].expand_as(elements), elements)


@wp.func
def _uniform(
    world: int,
    target: wp.array2d(dtype=float),
    ids: wp.array(dtype=wp.int32),
    value: wp.array(dtype=float),
    rng: wp.int32,
    low: float,
    high: float,
):
    value[world] = wp.randf(wp.rand_init(rng, 0), low, high)
    for i in range(ids.shape[0]):
        target[world, ids[i]] = value[world]


def _uniform_torch(mask, target, ids, value, rng, low, high):
    # As Warp's randf: the span times a unit draw plus the low bound, fused on CUDA.
    unit = uniform(rng, 0.0, 1.0)
    drawn = torch.addcmul(low, high - low, unit) if unit.is_cuda else low + (high - low) * unit
    _write(target, ids, mask, drawn, value)


uniform_variation = Term(
    torch=_uniform_torch,
    warp=_uniform,
    state={"value": State(persistent=True)},
    random=True,
)
"""Draw a value per world uniformly from ``[low, high)``, write it to the ``ids`` elements of ``target``, and keep it
as the state ``value``."""


@wp.func
def _choice(
    world: int,
    target: wp.array2d(dtype=float),
    ids: wp.array(dtype=wp.int32),
    value: wp.array(dtype=float),
    rng: wp.int32,
    table: wp.array(dtype=float),
    bounds: wp.array(dtype=float),
):
    draw = wp.randf(wp.rand_init(rng, 0))
    index = int(0)
    for k in range(table.shape[0] - 1):
        if draw >= bounds[k]:
            index = k + 1
    value[world] = table[index]
    for i in range(ids.shape[0]):
        target[world, ids[i]] = value[world]


def _choice_torch(mask, target, ids, value, rng, table, bounds):
    draw = uniform(rng, 0.0, 1.0)
    index = (draw[:, None] >= bounds[None, : table.shape[0] - 1]).sum(dim=1)
    _write(target, ids, mask, table[index], value)


def _fill_tables(ctx: TermContext, values: tuple[float, ...] = (), chances: tuple[float, ...] = (), **params) -> None:
    """Fill the tables of the values to draw from, and of the cumulative chances that bound them, once."""
    ctx.state["table"].copy_(torch.tensor(values, dtype=torch.float32))
    if "bounds" in ctx.state and chances:
        ctx.state["bounds"].copy_(torch.tensor(chances, dtype=torch.float32))


choice_variation = Term(
    torch=_choice_torch,
    warp=_choice,
    init=_fill_tables,
    state={
        "value": State(persistent=True),
        "table": State(("num_values",), persistent=True),
        "bounds": State(("num_bounds",), persistent=True),
    },
    random=True,
)
"""Draw one of ``values`` per world with the cumulative ``chances`` between them, write it to the ``ids``
elements of ``target``, and keep it as the state ``value``."""


def _replay_torch(mask, target, ids, value, row, table, taken):
    # The rows of the worlds that reset, in world order, after those that earlier resets took.
    rank = torch.cumsum(mask.long(), dim=0) - 1
    rows = (taken.long() + rank) % table.shape[0]
    row.copy_(torch.where(mask, rows.float(), row))
    _write(target, ids, mask, table[rows], value)
    # Modulo the table, so that the count stays exact in float32.
    taken.copy_((taken + mask.sum()) % table.shape[0])


replay_variation = Term(
    torch=_replay_torch,
    init=_fill_tables,
    state={
        "value": State(persistent=True),
        "row": State(persistent=True),
        "table": State(("num_values",), persistent=True),
        "taken": State((1,), persistent=True),
    },
)
"""Take the next of ``values`` in the order of the resets, write it to the ``ids`` elements of ``target``, and keep it
and its row as the states ``value`` and ``row``."""
