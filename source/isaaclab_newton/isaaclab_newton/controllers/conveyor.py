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
from newton.examples.basic.example_basic_conveyor_forces import ConveyorForceModel
from newton.solvers import SolverBase

from isaaclab.controllers.conveyor import BaseConveyor
from isaaclab.controllers.conveyor_cfg import ConveyorCfg, ConveyorSurfaceCfg
from isaaclab.physics import PhysicsEvent
from isaaclab.utils.warp import memory

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


@wp.kernel
def _filter_body_forces(body_is_tracked: wp.array[wp.int32], body_force: wp.array[wp.spatial_vector]):
    body_id = wp.tid()
    if body_is_tracked[body_id] == 0:
        body_force[body_id] = wp.spatial_vector()


@wp.kernel
def _advance_startup_scale(
    dt: wp.float32, duration: wp.float32, elapsed: wp.array[wp.float32], scale: wp.array[wp.float32]
):
    elapsed[0] += dt
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


def _resolve_surface_paths(
    num_envs: int,
    surface_specs: Sequence[ConveyorSurfaceCfg],
    env_path_format: str,
) -> tuple[str, ...]:
    """Resolve conveyor templates to exact environment-major prim paths."""
    if not surface_specs:
        raise ValueError("At least one ConveyorSurfaceCfg is required.")
    namespace = "{ENV_REGEX_NS}"
    templates = tuple(spec.prim_path for spec in surface_specs)
    ordered_templates = sorted(templates)
    if any(
        right == left or right.startswith(f"{left}/") for left, right in zip(ordered_templates, ordered_templates[1:])
    ):
        raise ValueError("Conveyor prim paths must be unique and must not be ancestors of one another.")
    if env_path_format.count("{}") != 1:
        raise ValueError("Conveyor env_path_format must contain exactly one '{}'.")
    env_paths = tuple(env_path_format.format(env_id) for env_id in range(num_envs))
    if any(
        not path.startswith("/") or not all(part.isidentifier() for part in path[1:].split("/")) for path in env_paths
    ):
        raise ValueError("Conveyor env_path_format must produce concrete absolute prim paths.")
    return tuple(template.replace(namespace, env_path) for env_path in env_paths for template in templates)


def _world_vector(transform: wp.transform, vector: tuple[float, float, float]) -> wp.vec3:
    """Rotate and normalize an authored direction or surface normal."""
    rotated = wp.transform_vector(transform, wp.vec3(*vector))
    values = np.asarray([float(rotated[index]) for index in range(3)], dtype=np.float32)
    values /= float(np.linalg.norm(values))
    return wp.vec3(*(float(value) for value in values))


class Conveyor(BaseConveyor):
    """Newton implementation of :class:`isaaclab.controllers.Conveyor`.

    Register after the simulation context is created and before its first reset.
    Speeds come from the authored surface descriptions. Newton computes traction
    after each solver substep and applies it on the next one. Hard resets rebuild
    the binding before CUDA graph capture; episode resets clear selected-world forces.
    """

    def __init__(self, cfg: ConveyorCfg, num_envs: int, *, sim: SimulationContext) -> None:
        """Bind configured surfaces before the first simulation reset."""
        super().__init__(cfg, num_envs, sim=sim)
        self._surface_specs = tuple(self.cfg.surface_specs)
        self._prim_paths = _resolve_surface_paths(num_envs, self._surface_specs, self.cfg.env_path_format)
        self._manager = sim.backend_manager
        self._body_pattern = re.compile(self.cfg.body_pattern)
        if not np.isfinite(self.cfg.startup_duration_s) or self.cfg.startup_duration_s <= 0.0:
            raise ValueError(f"Conveyor startup duration must be positive, got {self.cfg.startup_duration_s}.")
        self._conveyor: ConveyorForceModel | None = None
        self._manager.request_solver_observables(SolverBase.ObservableKind.CONTACT_F)
        self._physics_ready_handle = self._manager.register_callback(
            self._queue_solver_binding, PhysicsEvent.PHYSICS_READY, name="surface_velocity_solver_init"
        )

    def _queue_solver_binding(self, _event: Any) -> None:
        """Bind after solver allocation on initial construction and every hard reset."""
        self._is_initialized = False
        self._manager.register_solver_init_callback(self._bind_solver)

    def _bind_solver(self) -> None:
        """Resolve the newly finalized model and register its graph callbacks."""
        backend = self._manager.backend
        model = backend.model
        contacts = backend.contacts
        observables = self._manager.get_solver_observables()
        if contacts is None or observables is None or observables.contact_f is None:
            raise RuntimeError("The conveyor requires solved per-contact force observables.")
        if model.world_count != self.num_envs:
            raise RuntimeError(f"Newton model has {model.world_count} worlds, expected {self.num_envs}.")

        surfaces_per_env = len(self._surface_specs)
        surface_shapes = [-1] * len(self._prim_paths)
        shape_world = model.shape_world.numpy()
        for shape_id, label in enumerate(model.shape_label):
            world_id = int(shape_world[shape_id])
            if not 0 <= world_id < self.num_envs:
                continue
            for spec_id in range(surfaces_per_env):
                surface_id = world_id * surfaces_per_env + spec_id
                path = self._prim_paths[surface_id]
                if label != path and not label.startswith(f"{path}/"):
                    continue
                if surface_shapes[surface_id] >= 0:
                    raise RuntimeError(f"Multiple collision shapes match conveyor surface {path!r}.")
                surface_shapes[surface_id] = shape_id
        missing = [
            path
            for index, (path, shape_id) in enumerate(zip(self._prim_paths, surface_shapes, strict=True))
            if shape_id < 0 and not self._surface_specs[index % surfaces_per_env].optional
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
        if not np.any(body_is_tracked):
            raise RuntimeError(f"Body pattern {self._body_pattern.pattern!r} matched no Newton bodies.")

        conveyor = ConveyorForceModel(model, solver_type="mujoco")
        transforms = model.shape_transform.numpy()
        surface_worlds = []
        for surface_id, shape_id in enumerate(surface_shapes):
            if shape_id < 0:
                continue
            spec = self._surface_specs[surface_id % surfaces_per_env]
            surface_worlds.append(surface_id // surfaces_per_env)
            values = transforms[shape_id]
            transform = wp.transform(wp.vec3(*values[:3]), wp.quat(*values[3:]))
            direction = np.asarray(_world_vector(transform, spec.direction), dtype=np.float32)
            speed = np.float32(spec.velocity)
            parameters = {
                "surface_normal": _world_vector(transform, spec.surface_normal),
                "friction": spec.friction_coefficient,
                "threshold": spec.contact_threshold,
            }
            if spec.curved:
                conveyor.add_pivot_belt(
                    shape_id,
                    wp.transform_point(transform, wp.vec3(*spec.pivot_point)),
                    wp.vec3(*(direction * (speed / np.float32(spec.radius)))),
                    **parameters,
                )
            else:
                conveyor.add_constant_belt(shape_id, wp.vec3(*(direction * speed)), **parameters)
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

    def _apply(self, state) -> None:
        """Apply the wrench from the preceding solve."""
        self._conveyor.apply(state)

    def _update(self, contacts, observables, state, dt: float) -> None:
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
        if self._conveyor is None:
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
        self._conveyor = None
        self._is_initialized = False
