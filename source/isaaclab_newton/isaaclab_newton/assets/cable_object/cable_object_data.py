# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from isaaclab.assets.cable_object.base_cable_object_data import BaseCableObjectData
from isaaclab.assets.physics_properties import UsdAttribute, usd_field
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.physics import NewtonManager as SimulationManager
from isaaclab_newton.physics.contact_data import NewtonContactData

if TYPE_CHECKING:
    from newton.selection import ArticulationView


class CableObjectData(BaseCableObjectData, NewtonContactData):
    """Data container for a Newton cable object."""

    __backend_name__: str = "newton"
    """The name of the backend for the cable object data."""

    def __init__(self, root_view: ArticulationView, device: str) -> None:
        """Initialize the cable object data.

        Args:
            root_view: The cable articulation view.
            device: The device used for processing.
        """
        super().__init__(device)
        self._root_view: ArticulationView = weakref.proxy(root_view)
        self._create_simulation_bindings()
        self._create_buffers()
        self.update(0.0)
        self.default_segment_pose_w = ProxyArray(wp.clone(self._segment_pose_w))
        self.default_segment_velocity_w = ProxyArray(wp.clone(self._segment_velocity_w))

    @property
    def segment_pose_w(self) -> ProxyArray:
        """Segment actor-frame poses in the world frame.

        Shape is (num_instances, num_segments), dtype ``wp.transformf``. Positions are in [m].
        """
        return self._segment_pose_w_ta

    @property
    def segment_velocity_w(self) -> ProxyArray:
        """Segment COM velocities in the world frame.

        Shape is (num_instances, num_segments), dtype ``wp.spatial_vectorf``, with units [m/s, rad/s].
        """
        return self._segment_velocity_w_ta

    def update(self, dt: float) -> None:
        """Update the cable segment state.

        Args:
            dt: The time step [s].
        """
        del dt
        wp.copy(self._segment_pose_w[:, 0], self._sim_bind_root_pose_w)
        wp.copy(self._segment_velocity_w[:, 0], self._sim_bind_root_velocity_w)
        wp.copy(self._segment_pose_w[:, 1:], self._sim_bind_link_pose_w)
        wp.copy(self._segment_velocity_w[:, 1:], self._sim_bind_link_velocity_w)

    def _create_simulation_bindings(self) -> None:
        """Create bindings to Newton simulation data."""
        model = SimulationManager.get_model()
        state = SimulationManager.get_state_0()
        self._num_instances = self._root_view.count
        self._num_segments = self._root_view.link_count + 1
        self._sim_bind_root_body_ids = self._root_view.get_attribute("joint_parent", model)[:, 0, 0].contiguous()
        self._sim_bind_link_body_ids = self._root_view.get_attribute("joint_child", model)[:, 0].contiguous()
        self._sim_bind_root_pose_w = state.body_q[self._sim_bind_root_body_ids]
        self._sim_bind_root_velocity_w = state.body_qd[self._sim_bind_root_body_ids]
        self._sim_bind_link_pose_w = self._root_view.get_link_transforms(state)[:, 0]
        self._sim_bind_link_velocity_w = self._root_view.get_link_velocities(state)[:, 0]

    def _create_buffers(self) -> None:
        """Create cable state buffers."""
        shape = (self._num_instances, self._num_segments)
        self._segment_pose_w = wp.empty(shape, dtype=wp.transformf, device=self.device)
        self._segment_velocity_w = wp.empty(shape, dtype=wp.spatial_vectorf, device=self.device)
        self._segment_pose_w_ta = ProxyArray(self._segment_pose_w)
        self._segment_velocity_w_ta = ProxyArray(self._segment_velocity_w)

    def _shape_values(self, name: str) -> np.ndarray:
        """Select the cable's generated colliders in environment/segment order."""
        return getattr(SimulationManager.get_model(), name).numpy()[self.shape_indices]

    @property
    def shape_indices(self) -> np.ndarray:
        """Native collider identities in environment/segment order."""
        model = SimulationManager.get_model()
        bodies = np.concatenate(
            (self._sim_bind_root_body_ids.numpy().reshape(-1, 1), self._sim_bind_link_body_ids.numpy()), axis=1
        )
        owners = model.shape_body.numpy()
        rows = [np.flatnonzero(np.isin(owners, body_ids)) for body_ids in bodies]
        if len({len(row) for row in rows}) != 1:
            raise NotImplementedError("Cable environments must have matching collider counts.")
        return np.asarray(rows)

    @property
    @usd_field(
        UsdAttribute("newton:export:shape_margin", type_name="float[]"),
        scope="array",
    )
    def shape_margin(self) -> np.ndarray:
        """Outward collision margin [m], shape [num_instances, num_shapes]."""
        return super().shape_margin

    @property
    @usd_field(
        UsdAttribute("newton:export:shape_gap", type_name="float[]"),
        scope="array",
    )
    def shape_gap(self) -> np.ndarray:
        """Contact generation gap [m], shape [num_instances, num_shapes]."""
        return super().shape_gap

    @property
    @usd_field(
        UsdAttribute("newton:export:shape_material_mu", type_name="float[]"),
        scope="array",
    )
    def shape_material_mu(self) -> np.ndarray:
        """Friction coefficient, shape [num_instances, num_shapes]."""
        return super().shape_material_mu

    @property
    @usd_field(
        UsdAttribute("newton:export:shape_material_restitution", type_name="float[]"),
        scope="array",
    )
    def shape_material_restitution(self) -> np.ndarray:
        """Restitution coefficient, shape [num_instances, num_shapes]."""
        return super().shape_material_restitution

    @property
    @usd_field(
        UsdAttribute("newton:export:shape_material_ke", type_name="float[]"),
        scope="array",
    )
    def shape_material_ke(self) -> np.ndarray:
        """Contact stiffness [N/m], shape [num_instances, num_shapes]."""
        return super().shape_material_ke

    @property
    @usd_field(
        UsdAttribute("newton:export:shape_material_kd", type_name="float[]"),
        scope="array",
    )
    def shape_material_kd(self) -> np.ndarray:
        """Contact damping [N*s/m], shape [num_instances, num_shapes]."""
        return super().shape_material_kd

    @property
    @usd_field(
        UsdAttribute("newton:export:shape_material_kf", type_name="float[]"),
        scope="array",
    )
    def shape_material_kf(self) -> np.ndarray:
        """Friction gain [N*s/m], shape [num_instances, num_shapes]."""
        return super().shape_material_kf

    @property
    @usd_field(
        UsdAttribute("newton:export:shape_material_ka", type_name="float[]"),
        scope="array",
    )
    def shape_material_ka(self) -> np.ndarray:
        """Contact adhesion distance [m], shape [num_instances, num_shapes]."""
        return super().shape_material_ka

    @property
    @usd_field(
        UsdAttribute(
            "newton:export:shape_material_mu_torsional",
            type_name="float[]",
        ),
        scope="array",
    )
    def shape_material_mu_torsional(self) -> np.ndarray:
        """Torsional friction [m], shape [num_instances, num_shapes]."""
        return super().shape_material_mu_torsional

    @property
    @usd_field(
        UsdAttribute("newton:export:shape_material_mu_rolling", type_name="float[]"),
        scope="array",
    )
    def shape_material_mu_rolling(self) -> np.ndarray:
        """Rolling friction [m], shape [num_instances, num_shapes]."""
        return super().shape_material_mu_rolling
