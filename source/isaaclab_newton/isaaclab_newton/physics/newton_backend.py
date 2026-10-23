# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton simulation data bound to one finalized model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp
from newton import CollisionPipeline, Contacts, ModelBuilder, State
from newton.sensors import SensorContact, SensorIMU
from newton.solvers import SolverBase

from isaaclab.sim import SimulationContext
from isaaclab.utils.buffers import TimestampedBuffer

from isaaclab_newton.renderers.visual_material import VisualMaterialWriter

from .newton_builder_cfg import NewtonBuilderCfg
from .newton_manager_cfg import NewtonCfg

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from .newton_backend_cfg import NewtonBackendCfg

SiteEntry = tuple[int, None] | tuple[None, list[list[int]]]
"""Resolved site: ``(global_shape_index, None)`` for a global site, or ``(None, per_world_indices)``."""


@dataclass
class NewtonCloneRecord:
    """Native replication outputs, kept across hard resets."""

    world_xforms: list[wp.transform] | None
    source_builders: dict[str, ModelBuilder]
    particle_ranges: dict[str, tuple[int, int]]
    cable_bindings: dict[str, list[int]]
    geometry_batches: list


class NewtonBackend:
    """Everything bound to one finalized Newton model.

    The simulation registry owns the backend; physics, renderers, sensors and visualizers borrow it. Under Newton
    physics the manager also binds the solver, contacts, sensors and actuators to it, and a hard reset closes it and
    finalizes a new one.
    """

    def __init__(self, cfg: NewtonBackendCfg, builder: ModelBuilder | None = None, *, dt: float | None = None):
        """Finalize a builder and allocate the buffers bound to its model.

        Args:
            cfg: Physics settings and device. :class:`NewtonCfg` physics makes this a simulation backend; any other
                physics makes it render-only.
            builder: Builder to finalize. Defaults to the simulation's shared builder.
            dt: Duration of one physics step [s]. Required for a simulation backend.
        """
        if builder is None:
            builder = SimulationContext.instance().get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics_cfg))
        self.model = builder.finalize(device=cfg.device)
        self.model.num_envs = self.model.world_count
        self.device: wp.Device = self.model.device
        self.particle_ranges: dict[str, tuple[int, int]] = {}
        self.deformable_ranges = {
            label: (start, end - start, kind)
            for kind, labels, starts, ends in (
                ("surface", builder.surface_label, builder._surface_particle_start, builder._surface_particle_end),
                ("volume", builder.volume_label, builder._volume_particle_start, builder._volume_particle_end),
            )
            for label, start, end in zip(labels, starts, ends, strict=True)
        }
        physics_cfg = cfg.physics_cfg
        simulation = isinstance(physics_cfg, NewtonCfg)
        soft_contact = physics_cfg.soft_contact_cfg if simulation else None
        if soft_contact is not None:
            self.model.soft_contact_ke = float(soft_contact.soft_contact_ke)
            self.model.soft_contact_kd = float(soft_contact.soft_contact_kd)
            self.model.soft_contact_mu = float(soft_contact.soft_contact_mu)
        self.state_0: State = self.model.state()
        self.geometry_offsets = (
            {} if simulation else {path: bounds[0] for path, bounds in self.deformable_ranges.items()}
        )
        self.bvh_refit = TimestampedBuffer()
        self.cfg: NewtonCfg | None = physics_cfg if simulation else None
        if not simulation:
            self.state_1 = self.control = None
            return
        if dt is None:
            raise ValueError("A simulation NewtonBackend requires the physics step duration dt.")

        self.dt = dt  # [s]
        self.num_substeps = physics_cfg.num_substeps
        self.solver_dt = dt / self.num_substeps  # [s]
        self.state_1: State = self.model.state()
        self.control = self.model.control()
        self.solver: SolverBase | None = None
        self.collision_pipeline: CollisionPipeline | None = None
        self.contacts: Contacts | None = None
        # Newton reserves the final world slot for global entities; Isaac Lab resets local worlds only.
        self.world_mask = wp.zeros(self.model.world_count + 1, dtype=wp.bool, device=self.device)
        self.fk_mask = wp.zeros(self.model.articulation_count, dtype=wp.bool, device=self.device)
        self.kinematics_dirty = False
        self.contact_sensors: dict[tuple, SensorContact] = {}
        self.imu_sensors: list[SensorIMU] = []
        self.actuators: NewtonActuatorAdapter | None = None

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Bind material writes to this resource's native shape-color buffer."""
        return VisualMaterialWriter(self.model, batches)

    def close(self) -> None:
        """Drop native handles after consumers release their bindings."""
        self.model = self.state_0 = self.state_1 = self.control = None
        self.particle_ranges, self.deformable_ranges, self.geometry_offsets = {}, {}, {}
        self.bvh_refit = TimestampedBuffer()
        if self.cfg is not None:
            self.solver = self.collision_pipeline = self.contacts = self.actuators = None
            self.contact_sensors, self.imu_sensors = {}, []
            self.world_mask = self.fk_mask = None
