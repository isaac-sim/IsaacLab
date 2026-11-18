# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data: the schema an MDP is built against, and the data views it runs on."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import warp as wp

from isaaclab.utils.warp import ProxyArray

from .config import FieldSelector


@dataclass(frozen=True)
class FieldSchema:
    """Layout of an entity field, an array of shape ``(num_worlds, *dims)``."""

    dims: tuple[str, ...] = ()  # Named dimensions after worlds, e.g. ("joints",)
    dtype: type = wp.float32  # Element type, e.g. wp.vec3f for a position per body
    # Whether events and actions may write it. Derived data, such as body poses that forward kinematics computes, and
    # sensor data are read-only: writes would not reach the simulation.
    writable: bool = False
    # Fields of the same entity that it is computed from, such as body poses from the joint and root state: a reset
    # event that reads it after an earlier one wrote them gets a commit first.
    derived_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntitySchema:
    """Element names and field layouts of one scene entity."""

    names: Mapping[str, tuple[str, ...]]  # Element names per dimension, e.g. {"joints": ("x", "y")}
    fields: Mapping[str, FieldSchema]


@dataclass(frozen=True)
class SceneSchema:
    """Description of what a scene provides to an MDP. The builder resolves selectors against it."""

    num_worlds: int
    step_dt: float  # Duration of one control step [s]
    entities: Mapping[str, EntitySchema]


class SceneData(Protocol):
    """Interface through which an MDP reads and writes scene data. The environment owns the scene."""

    schema: SceneSchema

    def data(self, entity: str, field: str) -> ProxyArray:
        """Return a live view of an entity field, of shape ``(num_worlds, ...)``."""
        ...

    def entity(self, name: str) -> Any:
        """Return the scene object of an entity, such as an articulation, for Python hooks that take it."""
        ...

    def commit(self, mask: wp.array, fields: Collection[FieldSelector]) -> None:
        """Make the simulation see the ``fields`` that events wrote in the masked worlds, and refresh derived data."""
        ...
