# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from newton.selection import ArticulationView

from pxr import UsdGeom

from isaaclab.assets.cable_object.base_cable_object import BaseCableObject
from isaaclab.cloner import queue_usd_replication
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.utils.queries import resolve_matching_prims_from_source
from isaaclab.utils.version import has_kit

from isaaclab_newton.cloner import queue_newton_physics_replication
from isaaclab_newton.physics import NewtonManager as SimulationManager

from .cable_object_data import CableObjectData

if TYPE_CHECKING:
    from isaaclab.assets.cable_object.cable_object_cfg import CableObjectCfg


class CableObject(BaseCableObject):
    """A Newton cable that requires the VBD solver."""

    cfg: CableObjectCfg
    """Configuration instance for the cable object."""

    __backend_name__: str = "newton"
    """The name of the backend for the cable object."""

    def __init__(self, cfg: CableObjectCfg):
        """Initialize the cable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        if has_kit():
            queue_usd_replication(cfg)
        queue_newton_physics_replication(cfg)

    @property
    def data(self) -> CableObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_view.count

    @property
    def num_segments(self) -> int:
        """Number of rigid segments per cable."""
        return self.root_view.link_count + 1

    @property
    def root_view(self) -> ArticulationView:
        """Articulation view for the cable."""
        return self._root_view

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the cable object's internal buffers.

        Args:
            env_ids: Environment indices. Defaults to all instances.
        """
        del env_ids

    def write_data_to_sim(self) -> None:
        """Write buffered commands to the simulation."""

    def update(self, dt: float) -> None:
        """Update the cable object data.

        Args:
            dt: The time step [s].
        """
        self.data.update(dt)

    def _initialize_impl(self) -> None:
        def is_cable_curve(prim) -> bool:
            applied_schemas = prim.GetPrimTypeInfo().GetAppliedAPISchemas()
            return prim.IsA(UsdGeom.BasisCurves) and "PhysicsCurvesDeformableSimAPI" in applied_schemas

        resolve_kwargs = {"predicate": is_cable_curve, "expected_num_matches": 1}
        _, curve_path_expr = resolve_matching_prims_from_source(self.cfg.prim_path, **resolve_kwargs)[0]
        articulation_path_expr = f"{curve_path_expr}_articulation"
        self._root_view = ArticulationView(
            SimulationManager.get_model(),
            articulation_path_expr.replace(".*", "*"),
            verbose=False,
        )
        self._data = CableObjectData(self.root_view, self.device)
        self._physics_ready_handle = SimulationManager.register_callback(
            lambda _: self._data._create_simulation_bindings(),
            PhysicsEvent.PHYSICS_READY,
            name=f"cable_object_rebind_{self.cfg.prim_path}",
        )
        self.update(0.0)

    def _clear_callbacks(self) -> None:
        """Clear all registered callbacks."""
        super()._clear_callbacks()
        if hasattr(self, "_physics_ready_handle") and self._physics_ready_handle is not None:
            self._physics_ready_handle.deregister()
            self._physics_ready_handle = None
