# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bind authored conveyor surfaces to Newton's upstream force model and solver lifecycle."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from newton.examples.basic.example_basic_conveyor_forces import ConveyorForceModel

from isaaclab.physics import PhysicsEvent

from .surface_velocity import SurfaceVelocitySpec, resolve_surface_velocity_paths


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
def _clear_selected_body_forces(
    body_world: wp.array[wp.int32], world_mask: wp.array[wp.bool], body_force: wp.array[wp.spatial_vector]
):
    body_id = wp.tid()
    world_id = body_world[body_id]
    if world_id >= 0 and world_mask[world_id]:
        body_force[body_id] = wp.spatial_vector()


def _world_vector(transform: wp.transform, vector: tuple[float, float, float]) -> wp.vec3:
    """Rotate and normalize an authored direction or surface normal."""
    rotated = wp.transform_vector(transform, wp.vec3(*vector))
    values = np.asarray([float(rotated[index]) for index in range(3)], dtype=np.float32)
    values /= float(np.linalg.norm(values))
    return wp.vec3(*(float(value) for value in values))


class SurfaceVelocity:
    """Own replicated conveyor bindings from model initialization through graph teardown.

    Register after the simulation context is created and before its first reset.
    Speeds come from the authored surface descriptions. Newton computes traction
    after each solver substep and applies it on the next one. Hard resets rebuild
    the binding before CUDA graph capture; episode resets clear selected-world forces.
    """

    def __init__(
        self,
        num_envs: int,
        surface_specs: Sequence[SurfaceVelocitySpec],
        *,
        body_pattern: str,
        body_count_per_env: int,
        startup_duration_s: float = 1.0,
        env_path_format: str = "/World/envs/env_{}",
    ) -> None:
        """Register model and solver callbacks for the authored surfaces.

        Args:
            num_envs: Number of replicated simulation environments.
            surface_specs: Surface descriptions in within-environment order.
            body_pattern: Regular expression selecting transported bodies.
            body_count_per_env: Expected transported-body count.
            startup_duration_s: Initial traction ramp duration [s].
            env_path_format: Clone-template format resolving environment roots.
        """
        self._surface_specs = tuple(surface_specs)
        self._prim_paths = resolve_surface_velocity_paths(num_envs, self._surface_specs, env_path_format)
        self._body_pattern = re.compile(body_pattern)
        if not np.isfinite(startup_duration_s) or startup_duration_s <= 0.0:
            raise ValueError(f"Conveyor startup duration must be positive, got {startup_duration_s}.")
        self._num_envs = num_envs
        self._body_count_per_env = body_count_per_env
        self._startup_duration_s = startup_duration_s
        self._model = None
        self._conveyor: ConveyorForceModel | None = None
        self._closed = False
        self._model_init_handle = NewtonManager.register_callback(
            self._request_contact_forces, PhysicsEvent.MODEL_INIT, name="surface_velocity_contact_attribute"
        )
        NewtonManager.register_solver_init_callback(self._bind_solver)

    def _request_contact_forces(self, _event: Any) -> None:
        """Request solved contact forces before model finalization."""
        NewtonManager.request_extended_contact_attribute("force")

    def _bind_solver(self, model: Any, contacts: Any) -> None:
        """Resolve the newly finalized model and register its graph callbacks."""
        if contacts is None or contacts.force is None:
            raise RuntimeError("The conveyor requires solved per-contact forces allocated before model finalization.")
        if model.world_count != self._num_envs:
            raise RuntimeError(f"Newton model has {model.world_count} worlds, expected {self._num_envs}.")

        surfaces_per_env = len(self._surface_specs)
        surface_shapes = [-1] * len(self._prim_paths)
        shape_world = model.shape_world.numpy()
        shape_body = model.shape_body.numpy()
        for shape_id, label in enumerate(model.shape_label):
            world_id = int(shape_world[shape_id])
            if not 0 <= world_id < self._num_envs:
                continue
            for spec_id in range(surfaces_per_env):
                surface_id = world_id * surfaces_per_env + spec_id
                path = self._prim_paths[surface_id]
                if label != path and not label.startswith(f"{path}/"):
                    continue
                if shape_body[shape_id] >= 0:
                    raise ValueError(f"Conveyor shape must be static: {label}")
                if surface_shapes[surface_id] >= 0:
                    raise RuntimeError(f"Multiple collision shapes match conveyor surface {path!r}.")
                surface_shapes[surface_id] = shape_id
        missing = [path for path, shape_id in zip(self._prim_paths, surface_shapes, strict=True) if shape_id < 0]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} conveyor collision sections: {missing[:8]}.")

        body_is_tracked = np.zeros(model.body_count, dtype=np.int32)
        tracked_counts = np.zeros(self._num_envs, dtype=np.int32)
        body_world = model.body_world.numpy()
        for body_id, label in enumerate(model.body_label):
            if self._body_pattern.search(label) is None:
                continue
            world_id = int(body_world[body_id])
            if not 0 <= world_id < self._num_envs:
                raise RuntimeError(f"Transported body {label!r} belongs to invalid world {world_id}.")
            body_is_tracked[body_id] = 1
            tracked_counts[world_id] += 1
        if np.any(tracked_counts != self._body_count_per_env):
            raise RuntimeError(
                f"Body pattern {self._body_pattern.pattern!r} expected {self._body_count_per_env} bodies per world, "
                f"got {tracked_counts.tolist()}."
            )
        if not np.any(body_is_tracked):
            raise RuntimeError(f"Body pattern {self._body_pattern.pattern!r} matched no Newton bodies.")

        conveyor = ConveyorForceModel(model, solver_type="mujoco")
        transforms = model.shape_transform.numpy()
        for surface_id, shape_id in enumerate(surface_shapes):
            spec = self._surface_specs[surface_id % surfaces_per_env]
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
        conveyor.finalize(contacts)
        conveyor.set_speed_scale(0.0)

        self._model = model
        self._conveyor = conveyor
        self._body_is_tracked = wp.array(body_is_tracked, dtype=wp.int32, device=model.device)
        self._elapsed_time = wp.zeros(1, dtype=wp.float32, device=model.device)
        self._world_mask_host = np.zeros(self._num_envs, dtype=np.bool_)
        self._world_mask = wp.zeros(self._num_envs, dtype=wp.bool, device=model.device)
        NewtonManager.register_state_force_callback(self._apply)
        NewtonManager.register_post_solver_substep_callback(self._update)

    def _apply(self, state) -> None:
        """Apply the wrench from the preceding solve."""
        self._conveyor.apply(state)

    def _update(self, solver, contacts, state, dt: float) -> None:
        """Compute the next wrench using Newton's upstream traction model."""
        wp.launch(
            _advance_startup_scale,
            dim=1,
            inputs=[dt, self._startup_duration_s],
            outputs=[self._elapsed_time, self._conveyor.global_velocity_scale],
            device=self._model.device,
        )
        self._conveyor.update(solver, contacts, state, dt)
        wp.launch(
            _filter_body_forces,
            dim=self._model.body_count,
            inputs=[self._body_is_tracked],
            outputs=[self._conveyor.conveyor_body_f],
            device=self._model.device,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear selected-world forces; a full reset also restarts the startup ramp."""
        if self._conveyor is None:
            raise RuntimeError("Surface velocity is not bound to an initialized Newton solver.")
        if env_ids is None:
            self._conveyor.conveyor_body_f.zero_()
            self._elapsed_time.zero_()
            self._conveyor.global_velocity_scale.zero_()
            return
        ids = env_ids.detach().cpu().numpy() if isinstance(env_ids, torch.Tensor) else np.asarray(env_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise IndexError(f"Surface reset environment indices must be integers, got {env_ids!r}.")
        ids = ids.reshape(-1)
        if np.any((ids < 0) | (ids >= self._num_envs)):
            raise IndexError(f"Conveyor reset environment indices are out of range: {ids.tolist()}.")
        self._world_mask_host.fill(False)
        self._world_mask_host[ids] = True
        if np.all(self._world_mask_host):
            self.reset()
            return
        self._world_mask.assign(self._world_mask_host)
        wp.launch(
            _clear_selected_body_forces,
            dim=self._model.body_count,
            inputs=[self._model.body_world, self._world_mask],
            outputs=[self._conveyor.conveyor_body_f],
            device=self._model.device,
        )

    def close(self) -> None:
        """Detach lifecycle and graph callbacks before the model is released."""
        if self._closed:
            return
        NewtonManager.unregister_solver_init_callback(self._bind_solver)
        self._model_init_handle.deregister()
        NewtonManager.unregister_state_force_callback(self._apply)
        NewtonManager.unregister_post_solver_substep_callback(self._update)
        self._conveyor = None
        self._model = None
        self._closed = True
