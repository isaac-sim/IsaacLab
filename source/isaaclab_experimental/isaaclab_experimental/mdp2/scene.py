# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data: the schema an MDP is built against and the views it runs on."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray


@dataclass(frozen=True)
class EntitySchema:
    """Names and fields of one scene entity."""

    names: Mapping[str, tuple[str, ...]]  # Element names per dimension, e.g. {"joints": ("x", "y")}
    fields: Mapping[str, tuple[str, ...]]  # float32 field dimensions after worlds, e.g. {"joint_pos": ("joints",)}


@dataclass(frozen=True)
class SceneSchema:
    """What a scene provides to an MDP. Builders resolve selectors against it."""

    num_worlds: int
    step_dt: float  # Duration of one control step [s]
    entities: Mapping[str, EntitySchema]


class SceneData(Protocol):
    """Scene data as seen by an MDP. The environment owns the scene itself."""

    schema: SceneSchema

    def data(self, entity: str, field: str) -> ProxyArray:
        """Live view of an entity field, shape ``(num_worlds, ...)``."""
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
        fields = ("joint_pos", "joint_vel", "joint_effort_target")
        self.schema = SceneSchema(
            num_worlds=num_worlds,
            step_dt=dt * decimation,
            entities={"robot": EntitySchema(names={"joints": joint_names}, fields=dict.fromkeys(fields, ("joints",)))},
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
