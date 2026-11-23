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
from newton import Contacts, State

# The example model and its buffers are tied to the Newton revision in uv.lock.
# Switch to the public runtime API tracked in https://github.com/newton-physics/newton/issues/5501 when available.
from newton.examples.basic.example_basic_conveyor_forces import ConveyorForceModel
from newton.solvers import SolverBase

from isaaclab.cloner.cloner_cfg import DEFAULT_ENV_TEMPLATE
from isaaclab.controllers.conveyor import BaseConveyor
from isaaclab.controllers.conveyor_cfg import ConveyorCfg
from isaaclab.physics import PhysicsEvent
from isaaclab.utils.warp import memory

from isaaclab_newton.physics import MJWarpSolverCfg

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
        if mask is self._all_env_mask:
            self._conveyor.conveyor_body_f.zero_()
            self._elapsed_time.zero_()
            self._conveyor.global_velocity_scale.zero_()
            return
        self._full_reset.fill_(1)
        wp.launch(_check_full_reset, self.num_envs, inputs=[mask], outputs=[self._full_reset], device=self.device)
        model = self._conveyor.model
        wp.launch(
            _clear_selected_body_forces,
            dim=model.body_count,
            inputs=[model.body_world, mask, self._full_reset],
            outputs=[self._conveyor.conveyor_body_f, self._elapsed_time, self._conveyor.global_velocity_scale],
            device=model.device,
        )

    def close(self) -> None:
        """Detach lifecycle and graph callbacks before the model is released."""
        self._manager.unregister_solver_init_callback(self._bind_solver)
        self._physics_ready_handle.deregister()
        self._manager.unregister_state_force_callback(self._apply)
        self._manager.unregister_post_solver_substep_callback(self._update)
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
        conveyor = ConveyorForceModel(model, solver_type="mujoco")
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
        conveyor.finalize(contacts, observables)
        conveyor.set_speed_scale(0.0)

        self._conveyor = conveyor
        self._surface_worlds = wp.array(surface_worlds, dtype=wp.int32, device=model.device)
        self._default_linear_velocity = wp.clone(conveyor.conv_const_vel)
        self._default_angular_velocity = wp.clone(conveyor.conv_pivot_angvel)
        self._body_is_tracked = wp.array(body_is_tracked, dtype=wp.int32, device=model.device)
        self._elapsed_time = wp.zeros(1, dtype=wp.float32, device=model.device)
        self._full_reset = wp.ones(1, dtype=wp.int32, device=model.device)
        self._manager.register_state_force_callback(self._apply)
        self._manager.register_post_solver_substep_callback(self._update)
        self._is_initialized = True

    def _apply(self, state: State) -> None:
        """Apply the wrench from the preceding solve."""
        self._conveyor.apply(state)

    def _update(
        self, contacts: Contacts | None, observables: SolverBase.Observables | None, state: State, dt: float
    ) -> None:
        """Compute the next wrench using Newton's upstream traction model."""
        model = self._conveyor.model
        wp.launch(
            _set_surface_speed,
            dim=self._conveyor.conv_const_vel.shape[0],
            inputs=[
                self._surface_worlds,
                self._speed_scale.warp,
                self._default_linear_velocity,
                self._default_angular_velocity,
            ],
            outputs=[self._conveyor.conv_const_vel, self._conveyor.conv_pivot_angvel],
            device=model.device,
        )
        wp.launch(
            _advance_startup_scale,
            dim=1,
            inputs=[dt, self.cfg.startup_duration_s],
            outputs=[self._elapsed_time, self._conveyor.global_velocity_scale],
            device=model.device,
        )
        self._conveyor.update(contacts, observables, state, dt)
        wp.launch(
            _filter_body_forces,
            dim=model.body_count,
            inputs=[self._body_is_tracked],
            outputs=[self._conveyor.conveyor_body_f],
            device=model.device,
        )


@wp.kernel
def _filter_body_forces(body_is_tracked: wp.array[wp.int32], body_force: wp.array[wp.spatial_vector]):
    body_id = wp.tid()
    if body_is_tracked[body_id] == 0:
        body_force[body_id] = wp.spatial_vector()


@wp.kernel
def _advance_startup_scale(
    dt: wp.float32, duration: wp.float32, elapsed: wp.array[wp.float32], scale: wp.array[wp.float32]
):
    elapsed[0] = elapsed[0] + dt
    scale[0] = wp.min(1.0, elapsed[0] / duration)


@wp.kernel
def _check_full_reset(world_mask: wp.array[wp.bool], full_reset: wp.array[wp.int32]):
    if not world_mask[wp.tid()]:
        wp.atomic_min(full_reset, 0, 0)


@wp.kernel
def _clear_selected_body_forces(
    body_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    full_reset: wp.array[wp.int32],
    body_force: wp.array[wp.spatial_vector],
    elapsed: wp.array[wp.float32],
    scale: wp.array[wp.float32],
):
    body_id = wp.tid()
    if body_id == 0 and full_reset[0] != 0:
        elapsed[0] = 0.0
        scale[0] = 0.0
    world_id = body_world[body_id]
    if world_id >= 0 and world_mask[world_id]:
        body_force[body_id] = wp.spatial_vector()


@wp.kernel
def _set_surface_speed(
    surface_worlds: wp.array[wp.int32],
    speed_scale: wp.array[wp.float32],
    default_linear: wp.array[wp.vec3],
    default_angular: wp.array[wp.vec3],
    linear: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
):
    surface = wp.tid()
    scale = speed_scale[surface_worlds[surface]]
    linear[surface] = default_linear[surface] * scale
    angular[surface] = default_angular[surface] * scale
