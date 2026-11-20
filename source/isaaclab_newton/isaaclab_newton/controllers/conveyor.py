# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bind authored conveyor surfaces to Newton's upstream force model and solver lifecycle."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from newton.solvers import SolverBase

from isaaclab.cloner.cloner_cfg import DEFAULT_ENV_TEMPLATE
from isaaclab.controllers.conveyor import BaseConveyor
from isaaclab.controllers.conveyor_cfg import ConveyorCfg
from isaaclab.physics import PhysicsEvent
from isaaclab.utils.warp import memory

from isaaclab_newton.physics import MJWarpSolverCfg

from ._conveyor_forces import ConveyorForceModel

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


class Conveyor(BaseConveyor):
    """Newton implementation of :class:`isaaclab.controllers.Conveyor`.

    Register after the simulation context is created and before its first reset.
    Speeds come from the authored surface descriptions. Newton computes traction
    after each solver substep and applies it on the next one. Hard resets rebuild
    the binding before CUDA graph capture; episode resets clear selected-world forces.
    """

    def __init__(self, cfg: ConveyorCfg, num_envs: int, *, sim: SimulationContext) -> None:
        """Bind configured surfaces before the first simulation reset."""
        if not isinstance(sim.cfg.physics.solver_cfg, MJWarpSolverCfg):
            raise NotImplementedError("Conveyors currently require the Newton MJWarp solver.")
        super().__init__(cfg, num_envs, sim=sim)
        self._sim = sim
        self._manager = sim.backend_manager
        self._body_pattern = re.compile(self.cfg.body_pattern)
        self._conveyor: ConveyorForceModel | None = None
        self._manager.request_solver_observables(SolverBase.ObservableKind.CONTACT_F, owner=self)
        self._physics_ready_handle = self._manager.register_callback(
            self._queue_solver_binding, PhysicsEvent.PHYSICS_READY, name="surface_velocity_solver_init"
        )

    def reset(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
        *,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Clear selected-world forces while preserving commanded speed scales.

        A full reset also restarts the startup traction ramp. Native masks and
        same-device indices select worlds without synchronizing with the host.
        """
        if not self.is_initialized:
            raise RuntimeError("Surface velocity is not bound to an initialized Newton solver.")
        mask = memory.resolve_1d_mask(
            ids=env_ids, mask=env_mask, all_mask=self._all_env_mask, scratch_mask=self._selection_mask
        )
        self._conveyor.reset(None if mask is self._all_env_mask else mask)

    def close(self) -> None:
        """Detach lifecycle and graph callbacks before the model is released."""
        self._manager.unregister_solver_init_callback(self._bind_solver)
        self._physics_ready_handle.deregister()
        if self._conveyor is not None:
            self._manager.unregister_state_force_callback(self._conveyor.apply)
            self._manager.unregister_post_solver_substep_callback(self._conveyor.update)
        self._manager.release_solver_observables(self)
        self._conveyor = None
        self._is_initialized = False

    def _queue_solver_binding(self, _event: Any) -> None:
        """Bind after solver allocation on initial construction and every hard reset."""
        self._is_initialized = False
        self._conveyor = None
        self._manager.register_solver_init_callback(self._bind_solver)

    def _bind_solver(self) -> None:
        """Resolve the newly finalized model and register its graph callbacks."""
        self._is_initialized = False
        self._conveyor = None
        backend = self._manager.backend
        model = backend.model
        contacts = backend.contacts
        observables = self._manager.get_solver_observables()
        if contacts is None or observables is None or observables.contact_f is None:
            raise RuntimeError("The conveyor requires solved per-contact force observables.")
        if model.world_count != self.num_envs:
            raise RuntimeError(f"Newton model has {model.world_count} worlds, expected {self.num_envs}.")

        plan = self._sim.get_clone_plan()
        template = DEFAULT_ENV_TEMPLATE if plan is None else plan.env_template
        prim_paths = tuple(
            spec.prim_path.replace("{ENV_REGEX_NS}", template.format(env_id))
            for env_id in range(self.num_envs)
            for spec in self.cfg.surface_specs
        )
        surfaces_per_env = len(self.cfg.surface_specs)
        surface_shapes = [-1] * len(prim_paths)
        shape_world = model.shape_world.numpy()
        for shape_id, label in enumerate(model.shape_label):
            world_id = int(shape_world[shape_id])
            if not 0 <= world_id < self.num_envs:
                continue
            for spec_id in range(surfaces_per_env):
                surface_id = world_id * surfaces_per_env + spec_id
                path = prim_paths[surface_id]
                if label != path and not label.startswith(f"{path}/"):
                    continue
                if surface_shapes[surface_id] >= 0:
                    raise RuntimeError(f"Multiple collision shapes match conveyor surface {path!r}.")
                surface_shapes[surface_id] = shape_id
        missing = [
            path
            for index, (path, shape_id) in enumerate(zip(prim_paths, surface_shapes, strict=True))
            if shape_id < 0 and not self.cfg.surface_specs[index % surfaces_per_env].optional
        ]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} conveyor collision sections: {missing[:8]}.")

        body_is_tracked = np.zeros(model.body_count, dtype=np.int32)
        tracked_counts = np.zeros(self.num_envs, dtype=np.int32)
        body_world = model.body_world.numpy()
        for body_id, label in enumerate(model.body_label):
            if self._body_pattern.search(label) is None:
                continue
            world_id = int(body_world[body_id])
            if not 0 <= world_id < self.num_envs:
                raise RuntimeError(f"Transported body {label!r} belongs to invalid world {world_id}.")
            body_is_tracked[body_id] = 1
            tracked_counts[world_id] += 1
        if np.any(tracked_counts != self.cfg.body_count_per_env):
            raise RuntimeError(
                f"Body pattern {self._body_pattern.pattern!r} expected {self.cfg.body_count_per_env} bodies per world, "
                f"got {tracked_counts.tolist()}."
            )
        conveyor = ConveyorForceModel(model, self._speed_scale.warp, self.cfg.startup_duration_s)
        transforms = model.shape_transform.numpy()
        surface_worlds = []
        for surface_id, shape_id in enumerate(surface_shapes):
            if shape_id < 0:
                continue
            spec = self.cfg.surface_specs[surface_id % surfaces_per_env]
            surface_worlds.append(surface_id // surfaces_per_env)
            values = transforms[shape_id]
            transform = wp.transform(wp.vec3(*values[:3]), wp.quat(*values[3:]))
            direction = wp.normalize(wp.transform_vector(transform, wp.vec3(*spec.direction)))
            speed = float(spec.velocity)
            parameters = {
                "surface_normal": wp.normalize(wp.transform_vector(transform, wp.vec3(*spec.surface_normal))),
                "friction": spec.friction_coefficient,
                "threshold": spec.contact_threshold,
            }
            if spec.curved:
                conveyor.add_pivot_belt(
                    shape_id,
                    wp.transform_point(transform, wp.vec3(*spec.pivot_point)),
                    direction * (speed / float(spec.radius)),
                    **parameters,
                )
            else:
                conveyor.add_constant_belt(shape_id, direction * speed, **parameters)
        conveyor.finalize(contacts, observables, surface_worlds, body_is_tracked)

        self._conveyor = conveyor
        self._manager.register_state_force_callback(conveyor.apply)
        self._manager.register_post_solver_substep_callback(conveyor.update)
        self._is_initialized = True
