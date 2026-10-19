# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton simulation data bound to one finalized model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp
from newton import CollisionPipeline, Contacts, Model, ModelBuilder, State
from newton.sensors import SensorContact, SensorIMU
from newton.solvers import SolverBase

from isaaclab.utils.buffers import TimestampedBuffer

from isaaclab_newton.renderers.visual_material import VisualMaterialWriter

from .newton_manager_cfg import NewtonBackendCfg, NewtonBuilderCfg, NewtonCfg

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.physics import PhysicsCfg
    from isaaclab.renderers.base_renderer import VisualMaterialBatch
    from isaaclab.sim import SimulationContext

SiteEntry = tuple[int, None] | tuple[None, list[list[int]]]
"""Resolved site: ``(global_shape_index, None)`` for a global site, or ``(None, per_world_indices)``."""


@dataclass
class NewtonCloneRecord:
    """Native replication outputs, kept across hard resets and consumed when the model is finalized."""

    world_xforms: list[wp.transform] | None
    source_builders: dict[str, ModelBuilder]  # per source, so single-model consumers can finalize one environment
    particle_ranges: dict[str, tuple[int, int]]  # native (start, count) of each particle prim
    cable_bindings: dict[str, list[int]]  # capsule shape indices of each open cable
    geometry_batches: list  # deformable and particle geometry published through scene data


class NewtonBackend:
    """Everything bound to one finalized Newton model.

    Physics, renderers, sensors, and visualizers borrow the backend. Under Newton physics the
    :class:`~isaaclab_newton.physics.NewtonManager` finalizes and owns it; under other physics the simulation registry
    owns a render-only backend (see :func:`get_newton_backend`). A hard reset closes it and finalizes a new one, so no
    solver or sensor outlives the buffers it binds.

    A render-only backend allocates only :attr:`model` and :attr:`state_0`. A simulation backend also allocates the
    second state, control, and reset masks; the manager then constructs the solver and contacts.
    """

    def __init__(
        self,
        model: Model,
        physics_cfg: PhysicsCfg | None = None,
        *,
        dt: float | None = None,
        deformable_ranges: dict[str, tuple[int, int, str]] | None = None,
    ):
        """Bind native buffers to a finalized model.

        Args:
            model: Finalized model.
            physics_cfg: Physics configuration. A :class:`NewtonCfg` makes this a simulation backend; anything else
                makes it render-only.
            dt: Duration of one physics step [s]. Required for a simulation backend.
            deformable_ranges: Native ``(start, count, kind)`` particle range of each deformable mesh, by label.
        """
        self.model = model
        self.deformable_ranges = deformable_ranges or {}
        self.device: wp.Device = model.device
        # Physics settings apply to the model before any state is allocated from it.
        soft_contact = physics_cfg.soft_contact_cfg if isinstance(physics_cfg, NewtonCfg) else None
        if soft_contact is not None:
            model.soft_contact_ke = float(soft_contact.soft_contact_ke)
            model.soft_contact_kd = float(soft_contact.soft_contact_kd)
            model.soft_contact_mu = float(soft_contact.soft_contact_mu)
        self.state_0: State = model.state()
        self.bvh_refit = TimestampedBuffer()

        simulation = isinstance(physics_cfg, NewtonCfg)
        self.cfg: NewtonCfg | None = physics_cfg if simulation else None
        self.geometry_offsets = (
            {} if simulation else {path: bounds[0] for path, bounds in self.deformable_ranges.items()}
        )
        if not simulation:
            self.state_1 = self.control = None
            return
        if dt is None:
            raise ValueError("A simulation NewtonBackend requires the physics step duration dt.")

        self.dt = dt  # [s]
        self.num_substeps = physics_cfg.num_substeps
        self.solver_dt = dt / self.num_substeps  # [s]
        self.state_1: State = model.state()
        self.control = model.control()
        self.solver: SolverBase | None = None  # Newton's solver, constructed by the manager
        self.collision_pipeline: CollisionPipeline | None = None
        self.contacts: Contacts | None = None

        # Newton's final reset-mask slot selects global entities in world -1. Isaac Lab resets local worlds only, so
        # that slot remains false.
        self.world_mask = wp.zeros(model.world_count + 1, dtype=wp.bool, device=self.device)
        self.fk_mask = wp.zeros(model.articulation_count, dtype=wp.bool, device=self.device)
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
        self.deformable_ranges, self.geometry_offsets = {}, {}
        self.bvh_refit = TimestampedBuffer()
        if self.cfg is not None:
            self.solver = self.collision_pipeline = self.contacts = self.actuators = None
            self.contact_sensors, self.imu_sensors = {}, []
            self.world_mask = self.fk_mask = None


def get_newton_builder(sim: SimulationContext) -> ModelBuilder:
    """Return the Newton construction model of a simulation.

    Under Newton physics this is the physics manager's builder. Under other physics it is the render-only builder
    that renderers, visualizers, and ray casters share through the simulation registry.

    Args:
        sim: Simulation context.
    """
    if isinstance(sim.cfg.physics, NewtonCfg):
        return sim.backend_manager.builder
    return sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))


def get_newton_backend(sim: SimulationContext) -> NewtonBackend | None:
    """Return the Newton backend of a simulation.

    Under Newton physics this is the physics manager's active backend, or None before the first reset. Under other
    physics it is the render-only backend, built on first use and shared through the simulation registry.

    Args:
        sim: Simulation context.
    """
    if isinstance(sim.cfg.physics, NewtonCfg):
        return sim.backend_manager.backend
    return sim.get_or_create_backend(
        NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device, builder=get_newton_builder(sim))
    )


def create_newton_backend(cfg: NewtonBackendCfg) -> NewtonBackend:
    """Finalize a shared builder into a render-only backend; the simulation registry's factory.

    Newton physics finalizes its own backend (:meth:`NewtonManager.finalize_backend`).

    Args:
        cfg: Physics configuration, device, and builder.

    Returns:
        The backend.
    """
    model = cfg.builder.finalize(device=cfg.device)
    return NewtonBackend(model, cfg.physics_cfg, deformable_ranges=deformable_ranges(cfg.builder))


def deformable_ranges(builder: ModelBuilder) -> dict[str, tuple[int, int, str]]:
    """Return the native ``(start, count, kind)`` particle range of each deformable mesh of a builder, by label."""
    return {
        label: (start, end - start, kind)
        for kind, labels, starts, ends in (
            ("surface", builder.surface_label, builder._surface_particle_start, builder._surface_particle_end),
            ("volume", builder.volume_label, builder._volume_particle_start, builder._volume_particle_end),
        )
        for label, start, end in zip(labels, starts, ends, strict=True)
    }
