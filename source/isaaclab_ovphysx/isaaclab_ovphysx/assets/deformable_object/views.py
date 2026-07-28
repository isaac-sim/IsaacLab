# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin OVPhysX deformable body view."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab_ovphysx.sim.views.ovphysx_view import OvPhysxView
from isaaclab_ovphysx.tensor_types import TensorType


class OvPhysxDeformableBodyView(OvPhysxView):
    """OVPhysX view exposing deformable mesh dimensions.

    The deformable asset selects the volume or surface tensor bindings. This
    subclass adds the mesh-dimension properties required by the common
    deformable-object interface and rejects mixed simulation-node counts, whose
    padded state cannot currently be represented safely.

    Args:
        physx: OVPhysX instance exposing tensor bindings.
        simulation_nodal_position_type: Tensor type used for simulation-node positions.
        simulation_element_indices_type: Tensor type used for simulation-element connectivity.
        collision_element_indices_type: Optional tensor type used for collision-element connectivity.
        **kwargs: Additional arguments forwarded to the generic OVPhysX view.

    Raises:
        ValueError: If the matched deformable bodies have different simulation-node counts.

    """

    def __init__(
        self,
        physx: Any,
        *,
        simulation_nodal_position_type: TensorType,
        simulation_element_indices_type: TensorType,
        collision_element_indices_type: TensorType | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(physx, **kwargs)
        self._simulation_nodal_position_type = simulation_nodal_position_type
        self._simulation_element_indices_type = simulation_element_indices_type
        self._collision_element_indices_type = collision_element_indices_type
        self._max_collision_nodes_per_body: int | None = None
        self._simulation_node_counts_per_body: tuple[int, ...] | None = None

        node_counts = self.simulation_node_counts_per_body
        if any(count != self.max_simulation_nodes_per_body for count in node_counts):
            raise ValueError(
                f"OVPhysX deformable views require uniform simulation-node counts; received {list(node_counts)}."
            )

    @property
    def simulation_node_counts_per_body(self) -> tuple[int, ...]:
        """Simulation-node counts for each deformable body, derived from connectivity."""
        if self._simulation_node_counts_per_body is None:
            element_indices = torch.from_dlpack(self.get_attribute(self._simulation_element_indices_type))
            node_counts = element_indices.flatten(start_dim=1).amax(dim=1) + 1
            self._simulation_node_counts_per_body = tuple(int(count) for count in node_counts.cpu().tolist())
        return self._simulation_node_counts_per_body

    @property
    def max_simulation_nodes_per_body(self) -> int:
        """Maximum number of simulation nodes per deformable body."""
        return int(self.binding_for(self._simulation_nodal_position_type).shape[1])

    @property
    def max_simulation_elements_per_body(self) -> int:
        """Maximum number of simulation elements per deformable body."""
        return int(self.binding_for(self._simulation_element_indices_type).shape[1])

    @property
    def max_collision_elements_per_body(self) -> int:
        """Maximum number of collision elements per deformable body."""
        if self._collision_element_indices_type is None:
            return 0
        return int(self.binding_for(self._collision_element_indices_type).shape[1])

    @property
    def max_collision_nodes_per_body(self) -> int:
        """Maximum number of collision nodes per deformable body."""
        if self._collision_element_indices_type is None:
            return 0
        if self._max_collision_nodes_per_body is None:
            element_indices = self.get_attribute(self._collision_element_indices_type).numpy()
            self._max_collision_nodes_per_body = int(element_indices.max()) + 1 if element_indices.size else 0
        return self._max_collision_nodes_per_body
