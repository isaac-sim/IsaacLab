# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone-context lifecycle, independent of physics and rendering backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from ..sim import SimulationContext
    from .clone_plan import ClonePlan


class ReplicateContext(ABC):
    """Prepare routes before construction, then apply a clone plan to one representation.

    Native resources belong to the simulation registry. Contexts borrow them during dispatch.
    """

    replicate_priority: ClassVar[int] = 0
    """Dispatch order; lower values run first."""

    def __init__(self, sim_context: SimulationContext):
        """Retain the simulation that owns this representation's resources.

        Args:
            sim_context: Simulation owning the clone plan and backend registry.
        """
        self._sim = sim_context

    @staticmethod
    def prepare(sim: SimulationContext, routing: dict[type[ReplicateContext], set[int]]) -> None:
        """Resolve routes and acquire resources before constructing or dispatching contexts.

        The default keeps the declared routes. Overrides may replace routes with a shared context.

        Args:
            sim: Simulation owning the consumers and native resources.
            routing: Mutable mapping of context classes to routed asset prototype indices.
        """

    @abstractmethod
    def replicate(self, plan: ClonePlan, asset_prototype_ids: tuple[int, ...]) -> object:
        """Apply this representation's routed prototypes to its simulation-owned resources.

        Args:
            plan: Replication layout shared by every clone backend.
            asset_prototype_ids: Asset definitions routed to this context.

        Returns:
            A backend-specific result, if any. Dispatch does not consume it.
        """
