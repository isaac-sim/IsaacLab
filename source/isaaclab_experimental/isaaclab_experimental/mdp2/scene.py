# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data: the schema an MDP is built against, and the data views it runs on."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray


@dataclass(frozen=True)
class FieldSchema:
    """Layout of an entity field, an array of shape ``(num_worlds, *dims)``."""

    dims: tuple[str, ...] = ()  # Named dimensions after worlds, e.g. ("joints",)
    dtype: type = wp.float32  # Element type, e.g. wp.vec3f for a position per body


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
        """Return the scene object of an entity, such as an articulation, for eager terms."""
        ...


class PointMassScene:
    """Test scene: one ``"robot"`` whose joints are unit point masses driven by their effort targets."""

    def __init__(
        self,
        num_worlds: int,
        joint_names: tuple[str, ...] = ("x",),
        dt: float = 0.01,
        decimation: int = 2,
        device: str = "cpu",
    ):
        fields = dict.fromkeys(("joint_pos", "joint_vel", "joint_effort_target"), FieldSchema(("joints",)))

        self.schema = SceneSchema(
            num_worlds=num_worlds,
            step_dt=dt * decimation,
            entities={"robot": EntitySchema(names={"joints": joint_names}, fields=fields)},
        )
        self.dt = dt
        self.decimation = decimation

        shape = (num_worlds, len(joint_names))
        self._data = {"robot": {name: ProxyArray(wp.zeros(shape, dtype=wp.float32, device=device)) for name in fields}}

    def data(self, entity: str, field: str) -> ProxyArray:
        return self._data[entity][field]

    def step(self) -> None:
        """Advance one control step: ``decimation`` substeps of semi-implicit Euler."""
        robot = {name: value.torch for name, value in self._data["robot"].items()}
        for _ in range(self.decimation):
            robot["joint_vel"].add_(robot["joint_effort_target"], alpha=self.dt)
            robot["joint_pos"].add_(robot["joint_vel"], alpha=self.dt)

    def reset(self, mask: torch.Tensor) -> None:
        """Zero every field of the worlds selected by a boolean mask."""
        for value in self._data["robot"].values():
            value.torch.masked_fill_(mask[:, None], 0.0)
