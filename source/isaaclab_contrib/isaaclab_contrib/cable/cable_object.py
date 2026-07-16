# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton cable asset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_newton.cloner import queue_newton_physics_replication
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import JointType
from newton.selection import ArticulationView

from pxr import UsdGeom

from isaaclab.assets import AssetBase
from isaaclab.cloner import queue_usd_replication
from isaaclab.physics import PhysicsEvent
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.queries import matches_path_expr_prefix, resolve_matching_prims_from_source
from isaaclab.utils.version import has_kit

from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg
from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

from .cable_object_data import CableData

if TYPE_CHECKING:
    from isaaclab.assets import CableObjectCfg


class CableObject(AssetBase):
    """One open cable simulated by the uncoupled Newton VBD solver."""

    cfg: CableObjectCfg
    """Configuration instance for the cable."""

    def __init__(self, cfg: CableObjectCfg):
        """Initialize the cable.

        Args:
            cfg: A configuration instance.
        """
        self._initialize_handle = None
        self._invalidate_initialize_handle = None
        self._prim_deletion_handle = None
        self._physics_ready_handle = None
        sim = SimulationContext.instance()
        solver_cfg = getattr(sim.cfg.physics, "solver_cfg", None) if sim is not None else None
        manager_key = (sim.physics_manager.__module__, sim.physics_manager.__qualname__) if sim is not None else None
        expected_manager_key = (NewtonVBDManager.__module__, NewtonVBDManager.__qualname__)
        if (
            manager_key != expected_manager_key
            or not isinstance(solver_cfg, VBDSolverCfg)
            or solver_cfg.integrate_with_external_rigid_solver
        ):
            raise RuntimeError("CableObject requires the uncoupled NewtonVBDManager.")

        super().__init__(cfg)
        self._curve_path_expr, self._num_segments = self._resolve_curve()
        if has_kit():
            queue_usd_replication(cfg)
        queue_newton_physics_replication(cfg)

    @property
    def data(self) -> CableData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_segments(self) -> int:
        """Number of rigid segments per cable."""
        return self._num_segments

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Cables have no internal reset buffers."""
        del env_ids

    def write_data_to_sim(self) -> None:
        """Cables have no buffered commands."""

    def update(self, dt: float) -> None:
        self._data.update(dt)

    def _initialize_impl(self) -> None:
        self._bind_runtime()
        if self._physics_ready_handle is None:
            self._physics_ready_handle = SimulationManager.register_callback(
                self._rebind_runtime_callback,
                PhysicsEvent.PHYSICS_READY,
                name=f"cable_object_rebind_{self.cfg.prim_path}",
            )

    def _rebind_runtime_callback(self, _event) -> None:
        if self._is_initialized:
            self._bind_runtime()

    def _bind_runtime(self) -> None:
        model = SimulationManager.get_model()
        articulation_expr = f"{self._curve_path_expr}_articulation"
        articulation_ids = [
            index
            for index, label in enumerate(model.articulation_label)
            if matches_path_expr_prefix(articulation_expr, label)
        ]
        self._articulation_view = ArticulationView(model, articulation_ids, verbose=False)
        self._validate_articulation(model)

        body_indices = self._find_body_indices(model)
        self._num_instances = body_indices.shape[0]
        self._body_indices = wp.array(body_indices, dtype=wp.int32, device=self.device)
        if hasattr(self, "_data"):
            self._data._bind(self._body_indices)
        else:
            self._data = CableData(self._body_indices, self.device)
        self.update(0.0)

    def _resolve_curve(self) -> tuple[str, int]:
        def is_cable_curve(prim) -> bool:
            applied = prim.GetPrimTypeInfo().GetAppliedAPISchemas()
            return prim.IsA(UsdGeom.BasisCurves) and "PhysicsCurvesDeformableSimAPI" in applied

        try:
            curve_prim, curve_path_expr = resolve_matching_prims_from_source(
                self.cfg.prim_path,
                predicate=is_cable_curve,
                expected_num_matches=1,
            )[0]
        except RuntimeError as error:
            raise ValueError(self._support_message()) from error

        curves = UsdGeom.BasisCurves(curve_prim)
        counts = curves.GetCurveVertexCountsAttr().Get()
        points = curves.GetPointsAttr().Get()
        valid = (
            curves.GetTypeAttr().Get() == UsdGeom.Tokens.linear
            and curves.GetWrapAttr().Get() == UsdGeom.Tokens.nonperiodic
            and counts is not None
            and points is not None
            and len(counts) == 1
            and int(counts[0]) == len(points)
            and len(points) >= 3
        )
        if not valid:
            raise ValueError(self._support_message())
        return curve_path_expr, len(points) - 1

    def _validate_articulation(self, model) -> None:
        view = self._articulation_view
        expected_joints = self._num_segments - 1
        joint_types = view.get_attribute("joint_type", model).numpy()
        valid = (
            view.count_per_world == 1
            and view.count == view.world_count
            and view.joint_count == expected_joints
            and view.joint_dof_count == 2 * expected_joints
            and view.link_count == expected_joints
            and np.all(joint_types == int(JointType.CABLE))
        )
        if not valid:
            raise RuntimeError(self._support_message())

    def _find_body_indices(self, model) -> np.ndarray:
        by_world: dict[int, list[tuple[int, int]]] = {world: [] for world in range(self._articulation_view.world_count)}
        body_world = model.body_world.numpy()
        for body_id, label in enumerate(model.body_label):
            body_root, separator, suffix = label.rpartition("_edge_body_")
            if not separator or not matches_path_expr_prefix(self._curve_path_expr, body_root):
                continue
            if not suffix.isdecimal():
                raise RuntimeError(self._support_message())
            world = int(body_world[body_id])
            if world == -1 and self._articulation_view.world_count == 1:
                world = 0
            if world not in by_world:
                raise RuntimeError(self._support_message())
            by_world[world].append((int(suffix), body_id))

        rows = []
        expected_segments = list(range(self._num_segments))
        for world in range(self._articulation_view.world_count):
            entries = sorted(by_world[world])
            if [segment for segment, _ in entries] != expected_segments:
                raise RuntimeError(self._support_message())
            rows.append([body_id for _, body_id in entries])
        return np.asarray(rows, dtype=np.int32)

    @staticmethod
    def _support_message() -> str:
        return "CableObject supports exactly one open, linear, non-periodic curve with at least two segments."

    def _clear_callbacks(self) -> None:
        super()._clear_callbacks()
        if self._physics_ready_handle is not None:
            self._physics_ready_handle.deregister()
            self._physics_ready_handle = None
