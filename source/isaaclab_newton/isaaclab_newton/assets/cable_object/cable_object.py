# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import GeoType, JointType
from newton.selection import ArticulationView

from pxr import UsdGeom

from isaaclab.assets.cable_object.base_cable_object import BaseCableObject
from isaaclab.cloner import queue_usd_replication
from isaaclab.physics import PhysicsEvent
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.queries import matches_path_expr_prefix, resolve_matching_prims_from_source
from isaaclab.utils.version import has_kit
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.cloner import queue_newton_physics_replication
from isaaclab_newton.physics import NewtonManager as SimulationManager

from .cable_object_data import CableObjectData
from .kernels import set_segment_pose_to_sim_index, set_segment_velocity_to_sim_index

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
        self._initialize_handle = None
        self._invalidate_initialize_handle = None
        self._prim_deletion_handle = None
        sim = SimulationContext.instance()
        if sim is not None and not getattr(sim.physics_manager, "_supports_cable_joints", False):
            raise RuntimeError("CableObject requires the pure Newton VBD solver. Configure VBDSolverCfg.")

        super().__init__(cfg)
        self._curve_path_expr, self._num_authored_segments = self._resolve_curve()
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

    def write_segment_pose_to_sim_index(
        self,
        *,
        segment_pose: torch.Tensor | wp.array(dtype=wp.transformf) | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
    ) -> None:
        """Set segment poses for selected environments.

        Args:
            segment_pose: Actor-frame poses in simulation world frame. The Torch shape is
                (len(env_ids), num_segments, 7), with position ``(x, y, z)`` [m] followed by quaternion
                ``(x, y, z, w)``. The Warp shape is (len(env_ids), num_segments), dtype ``wp.transformf``.
            env_ids: Environment indices. If None, all instances are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(segment_pose, ProxyArray):
            segment_pose = segment_pose.warp
        if isinstance(segment_pose, torch.Tensor):
            segment_pose = segment_pose.contiguous()
        self.assert_shape_and_dtype(segment_pose, (env_ids.shape[0], self.num_segments), wp.transformf, "segment_pose")

        for state in self._iter_states():
            wp.launch(
                set_segment_pose_to_sim_index,
                dim=(env_ids.shape[0], self.num_segments),
                inputs=[
                    segment_pose,
                    env_ids,
                    self.data._sim_bind_root_body_ids,
                    self.data._sim_bind_link_body_ids,
                ],
                outputs=[state.body_q],
                device=self.device,
            )
        SimulationManager.invalidate_body_state(env_ids)
        self.update(0.0)

    def write_segment_velocity_to_sim_index(
        self,
        *,
        segment_velocity: torch.Tensor | wp.array(dtype=wp.spatial_vectorf) | ProxyArray,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None = None,
    ) -> None:
        """Set segment center-of-mass velocities for selected environments.

        Args:
            segment_velocity: Segment center-of-mass velocities in simulation world frame. The Torch shape is
                (len(env_ids), num_segments, 6), with linear ``(x, y, z)`` [m/s] followed by angular
                ``(x, y, z)`` [rad/s] velocity. The Warp shape is (len(env_ids), num_segments), dtype
                ``wp.spatial_vectorf``.
            env_ids: Environment indices. If None, all instances are used.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if isinstance(segment_velocity, ProxyArray):
            segment_velocity = segment_velocity.warp
        if isinstance(segment_velocity, torch.Tensor):
            segment_velocity = segment_velocity.contiguous()
        self.assert_shape_and_dtype(
            segment_velocity, (env_ids.shape[0], self.num_segments), wp.spatial_vectorf, "segment_velocity"
        )

        for state in self._iter_states():
            wp.launch(
                set_segment_velocity_to_sim_index,
                dim=(env_ids.shape[0], self.num_segments),
                inputs=[
                    segment_velocity,
                    env_ids,
                    self.data._sim_bind_root_body_ids,
                    self.data._sim_bind_link_body_ids,
                ],
                outputs=[state.body_qd],
                device=self.device,
            )
        SimulationManager.invalidate_body_state(env_ids)
        self.update(0.0)

    def _resolve_curve(self) -> tuple[str, int]:
        """Resolve and validate the authored cable curve."""

        def is_cable_curve(prim) -> bool:
            applied_schemas = prim.GetPrimTypeInfo().GetAppliedAPISchemas()
            return prim.IsA(UsdGeom.BasisCurves) and "PhysicsCurvesDeformableSimAPI" in applied_schemas

        resolve_kwargs = {"predicate": is_cable_curve, "expected_num_matches": 1}
        curve_prim, curve_path_expr = resolve_matching_prims_from_source(self.cfg.prim_path, **resolve_kwargs)[0]
        curves = UsdGeom.BasisCurves(curve_prim)
        vertex_counts = curves.GetCurveVertexCountsAttr().Get()
        points = curves.GetPointsAttr().Get()
        if (
            vertex_counts is None
            or len(vertex_counts) != 1
            or points is None
            or int(vertex_counts[0]) != len(points)
            or len(points) < 3
            or curves.GetTypeAttr().Get() != UsdGeom.Tokens.linear
            or curves.GetWrapAttr().Get() != UsdGeom.Tokens.nonperiodic
        ):
            raise RuntimeError("CableObject requires one open, linear curve with at least two segments.")

        curve_to_world = UsdGeom.XformCache().GetLocalToWorldTransform(curve_prim)
        points_w = [curve_to_world.Transform(point) for point in points]
        if any(not math.isfinite(coordinate) for point in points_w for coordinate in point) or any(
            math.dist(start, end) <= 1.0e-8 for start, end in zip(points_w, points_w[1:])
        ):
            raise RuntimeError("CableObject requires finite world-space segments longer than 1e-8 m.")
        return curve_path_expr, len(points_w) - 1

    def _initialize_impl(self) -> None:
        curve_path_expr = self._curve_path_expr
        num_segments = self._num_authored_segments

        model = SimulationManager.get_model()
        articulation_path_expr = f"{curve_path_expr}_articulation"
        self._root_view = ArticulationView(
            model,
            articulation_path_expr.replace(".*", "*"),
            verbose=False,
        )
        topology_error = "CableObject requires one standalone, unwelded cable articulation per simulation world."
        expected_joint_count = num_segments - 1
        joint_types = self.root_view.get_attribute("joint_type", model).numpy()
        valid_topology = (
            self.root_view.count_per_world == 1
            and self.root_view.count == self.root_view.world_count
            and self.root_view.joint_count == expected_joint_count
            and self.root_view.joint_dof_count == 2 * expected_joint_count
            and self.root_view.link_count == expected_joint_count
            and self.num_segments == num_segments
            and bool((joint_types == int(JointType.CABLE)).all())
        )
        if not valid_topology:
            raise RuntimeError(topology_error)

        root_body_ids = self.root_view.get_attribute("joint_parent", model)[:, 0, 0].numpy().tolist()
        link_body_ids = self.root_view.get_attribute("joint_child", model)[:, 0].numpy().tolist()
        for root_body_id, link_ids in zip(root_body_ids, link_body_ids, strict=True):
            body_ids = [int(root_body_id), *(int(value) for value in link_ids)]
            if len(body_ids) != num_segments:
                raise RuntimeError(topology_error)
            for segment, body_id in enumerate(body_ids):
                label = model.body_label[body_id]
                if label is None:
                    raise RuntimeError(topology_error)
                body_root, separator, suffix = label.rpartition("_edge_body_")
                if not separator or suffix != str(segment) or not matches_path_expr_prefix(curve_path_expr, body_root):
                    raise RuntimeError(topology_error)

        self._ALL_INDICES = wp.array(list(range(self.num_instances)), dtype=wp.int32, device=self.device)

        self._data = CableObjectData(self.root_view, self.device)
        self._physics_ready_handle = SimulationManager.register_callback(
            lambda _: self._rebind(),
            PhysicsEvent.PHYSICS_READY,
            name=f"cable_object_rebind_{self.cfg.prim_path}",
        )
        self._register_visualization()

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | None) -> wp.array(
        dtype=wp.int32
    ):
        """Resolve environment indices to a Warp array."""
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._ALL_INDICES
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(device=self.device, dtype=torch.int32).contiguous(), dtype=wp.int32)
        if isinstance(env_ids, Sequence):
            return wp.array(list(env_ids), dtype=wp.int32, device=self.device)
        return env_ids

    def _iter_states(self):
        """Yield active Newton states."""
        state_0 = SimulationManager.get_state_0()
        state_1 = SimulationManager.get_state_1()
        yield state_0
        if state_1 is not None and state_1 is not state_0:
            yield state_1

    def _rebind(self) -> None:
        """Rebind simulation arrays after a Newton model rebuild."""
        self._data._create_simulation_bindings()
        self._register_visualization()

    def _register_visualization(self) -> None:
        """Register concrete cable curves for Kit Fabric synchronization."""
        sim = SimulationContext.instance()
        if not has_kit() or sim is None or sim.physics_manager._clone_physics_only:
            return

        model = SimulationManager.get_model()
        root_body_ids = self.data._sim_bind_root_body_ids.numpy().tolist()
        link_body_ids = self.data._sim_bind_link_body_ids.numpy().tolist()
        shape_types = model.shape_type.numpy()
        for root_body_id, link_ids in zip(root_body_ids, link_body_ids, strict=True):
            body_ids = [int(root_body_id), *(int(value) for value in link_ids)]
            shape_ids: list[int] = []
            for body_id in body_ids:
                body_shapes = model.body_shapes[body_id]
                if len(body_shapes) != 1:
                    raise RuntimeError("Cable visualization requires one capsule shape per segment body.")
                shape_id = int(body_shapes[0])
                if int(shape_types[shape_id]) != int(GeoType.CAPSULE):
                    raise RuntimeError("Cable visualization requires capsule segment shapes.")
                shape_ids.append(shape_id)

            root_label = model.body_label[body_ids[0]]
            suffix = "_edge_body_0"
            if root_label is None or not root_label.endswith(suffix):
                raise RuntimeError("Cable visualization requires Newton's default segment labels.")
            SimulationManager.register_cable_visual_prim(
                root_label[: -len(suffix)], body_ids=body_ids, shape_ids=shape_ids
            )

    def _clear_callbacks(self) -> None:
        """Clear all registered callbacks."""
        super()._clear_callbacks()
        if hasattr(self, "_physics_ready_handle") and self._physics_ready_handle is not None:
            self._physics_ready_handle.deregister()
            self._physics_ready_handle = None
