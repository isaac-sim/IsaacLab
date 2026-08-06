# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``StateLayout``: spatial structure + index mapping of a curriculum's state pool.

Decoupled from the actual reset-state buffer (e.g. terrain's ``task_table``
or factory's :class:`StateBuffer`): this object describes where the pool's
states sit in coordinate space and how items (the units the curriculum
samples among) map onto those states. Signals consume it; consumers
translate item index -> full reset state in their own way.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class StateLayout:
    """Spatial layout + index mapping of a curriculum's state pool.

    Attributes:
        coords: ``[num_states, coord_dim]`` spatial coordinates of pool
            states (``coord_dim=2`` for terrain xy, ``3`` for factory xyz,
            arbitrary positive D allowed).
        spawn_index: ``[num_items]`` long tensor; for each curriculum
            item, the state index used as the spawn endpoint.
        target_index: ``[num_items]`` long tensor of target endpoints,
            or ``None`` when items have no separate target endpoint
            (factory's slot==item case).
        task_partition: Optional ``[num_items]`` long tensor of partition
            keys (e.g. command-type id for terrain, gait id, mechanic
            class). Tasks in different partitions live on independent
            frontier manifolds -- a "walking A→B" and a "flying A→B"
            share spatial endpoints but should not propagate frontier
            into each other. ``FrontierSamplingStrategy`` builds its kNN graph
            *within* each partition so neighbour relations never cross
            partition boundaries. ``None`` means a single global partition.
    """

    coords: torch.Tensor
    spawn_index: torch.Tensor
    target_index: torch.Tensor | None = None
    task_partition: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.coords.ndim != 2:
            raise ValueError(f"coords must have shape [num_states, coord_dim]; got shape {tuple(self.coords.shape)}.")
        if self.spawn_index.ndim != 1:
            raise ValueError(
                f"spawn_index must be 1D with shape [num_items]; got shape {tuple(self.spawn_index.shape)}."
            )
        if self.target_index is not None and self.target_index.shape != self.spawn_index.shape:
            raise ValueError(
                "spawn_index and target_index must have the same shape; "
                f"got {tuple(self.spawn_index.shape)} vs {tuple(self.target_index.shape)}."
            )
        if self.task_partition is not None and self.task_partition.shape != self.spawn_index.shape:
            raise ValueError(
                "spawn_index and task_partition must have the same shape; "
                f"got {tuple(self.spawn_index.shape)} vs {tuple(self.task_partition.shape)}."
            )

    @property
    def num_states(self) -> int:
        """Size of the underlying state pool."""
        return int(self.coords.shape[0])

    @property
    def num_items(self) -> int:
        """Number of items the curriculum picks among."""
        return int(self.spawn_index.shape[0])

    @property
    def coord_dim(self) -> int:
        """Dimensionality of :attr:`coords` (2 for xy, 3 for xyz, ...)."""
        return int(self.coords.shape[1])


@configclass
class StateLayoutCfg:
    """Eval-backed config for building a :class:`StateLayout` from an env."""

    coords_bind: str = MISSING
    """Expression resolving to ``StateLayout.coords``."""

    spawn_index_bind: str = MISSING
    """Expression resolving to ``StateLayout.spawn_index``."""

    target_index_bind: str | None = None
    """Expression resolving to ``StateLayout.target_index``."""

    task_partition_bind: str | None = None
    """Expression resolving to ``StateLayout.task_partition``."""

    def build(self, env: ManagerBasedRLEnv) -> StateLayout:
        """Build the layout from env-owned tensors."""
        return StateLayout(
            coords=eval(self.coords_bind),  # noqa: S307
            spawn_index=eval(self.spawn_index_bind),  # noqa: S307
            target_index=None if self.target_index_bind is None else eval(self.target_index_bind),  # noqa: S307
            task_partition=None if self.task_partition_bind is None else eval(self.task_partition_bind),  # noqa: S307
        )
