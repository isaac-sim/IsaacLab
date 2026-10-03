# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched surface controls and lifecycle integration for Newton's conveyor force model.

Newton owns contact classification and traction computation. This adapter resolves replicated
USD surfaces, selects transported bodies, and preserves controls across solver rebuilds.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp
from newton.examples.basic.example_basic_conveyor_forces import ConveyorForceModel
from newton.examples.basic.example_basic_conveyor_forces import (
    Vec3Pair as Vec3Pair,
)
from newton.examples.basic.example_basic_conveyor_forces import (
    compute_basis_vectors as compute_basis_vectors,
)
from newton.examples.basic.example_basic_conveyor_forces import (
    compute_point_force as compute_point_force,
)
from newton.examples.basic.example_basic_conveyor_forces import (
    compute_point_impulse as compute_point_impulse,
)

from isaaclab.physics import PhysicsEvent, SurfaceVelocitySpec

from .newton_manager import NewtonManager


@wp.kernel
def _update_belt_velocities(
    direction: wp.array[wp.vec3],
    radius: wp.array[wp.float32],
    effective_velocity: wp.array[wp.float32],
    linear_velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
):
    conveyor_id = wp.tid()
    linear_velocity[conveyor_id] = direction[conveyor_id] * effective_velocity[conveyor_id]
    angular_velocity[conveyor_id] = direction[conveyor_id] * (effective_velocity[conveyor_id] / radius[conveyor_id])


@wp.kernel
def _filter_body_forces(
    body_is_tracked: wp.array[wp.int32],
    body_force: wp.array[wp.spatial_vector],
):
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
def _integrate_encoders(
    dt: wp.float32,
    effective_velocity: wp.array[wp.float32],
    position: wp.array[wp.float32],
):
    conveyor_id = wp.tid()
    position[conveyor_id] += dt * effective_velocity[conveyor_id]


@wp.kernel
def _update_effective_velocities(
    commanded_velocity: wp.array[wp.float32],
    enabled: wp.array[wp.int32],
    effective_velocity: wp.array[wp.float32],
):
    conveyor_id = wp.tid()
    effective_velocity[conveyor_id] = commanded_velocity[conveyor_id] * wp.float32(enabled[conveyor_id])


@wp.kernel
def _gather_float_values(
    source: wp.array[wp.float32],
    indices: wp.array[wp.int32],
    values: wp.array[wp.float32],
):
    output_id = wp.tid()
    values[output_id] = source[indices[output_id]]


@wp.kernel
def _gather_int_values(
    source: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    values: wp.array[wp.int32],
):
    output_id = wp.tid()
    values[output_id] = source[indices[output_id]]


@wp.kernel
def _clear_selected_body_forces(
    body_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    body_force: wp.array[wp.spatial_vector],
):
    body_id = wp.tid()
    world_id = body_world[body_id]
    if world_id >= 0 and world_mask[world_id]:
        body_force[body_id] = wp.spatial_vector()


@wp.kernel
def _clear_selected_encoders(
    conveyor_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    encoder_position: wp.array[wp.float32],
):
    conveyor_id = wp.tid()
    world_id = conveyor_world[conveyor_id]
    if world_mask[world_id]:
        encoder_position[conveyor_id] = 0.0


def _require_buffer_length(name: str, buffer: Any, expected: int) -> None:
    """Reject missing or mis-sized buffers before a Warp launch can access them."""
    actual = len(buffer) if buffer is not None else 0
    if actual != expected:
        raise RuntimeError(f"Conveyor force buffer {name!r} has length {actual}, expected {expected}.")


def _as_numpy(values: Any) -> np.ndarray:
    """Convert supported tensor-like values to a host NumPy array."""
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    if hasattr(values, "numpy") and not isinstance(values, np.ndarray):
        return values.numpy()
    return np.asarray(values)


def _world_vector(transform_values: np.ndarray, local_vector: tuple[float, float, float]) -> wp.vec3:
    """Rotate a local vector into world space and normalize it."""
    transform = wp.transform(
        wp.vec3(*(float(value) for value in transform_values[:3])),
        wp.quat(*(float(value) for value in transform_values[3:])),
    )
    rotated = wp.transform_vector(transform, wp.vec3(*local_vector))
    values = np.asarray([float(rotated[index]) for index in range(3)], dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-8:
        raise ValueError(f"Conveyor direction or surface normal must be non-zero, got {local_vector}.")
    values /= norm
    return wp.vec3(*(float(value) for value in values))


def _world_point(transform_values: np.ndarray, local_point: tuple[float, float, float]) -> wp.vec3:
    """Transform a local point into world space."""
    transform = wp.transform(
        wp.vec3(*(float(value) for value in transform_values[:3])),
        wp.quat(*(float(value) for value in transform_values[3:])),
    )
    point = wp.transform_point(transform, wp.vec3(*local_point))
    return wp.vec3(*(float(point[index]) for index in range(3)))


def _resolve_belt_prim_path(prim_path: str, env_path_format: str, world_id: int) -> str:
    """Resolve one replicated conveyor template to an exact Newton shape label."""
    if prim_path.startswith("{ENV_REGEX_NS}/"):
        return prim_path.format(ENV_REGEX_NS=env_path_format.format(world_id))
    return prim_path


def _shape_belongs_to_prim(shape_label: str, prim_path: str) -> bool:
    """Return whether a Newton collision-shape label is the prim or one of its descendants."""
    return shape_label == prim_path or shape_label.startswith(f"{prim_path.rstrip('/')}/")


def _validate_env_path_format(env_path_format: str) -> None:
    """Validate the concrete per-world path format derived from the scene cloner."""
    if not isinstance(env_path_format, str) or env_path_format.count("{}") != 1:
        raise ValueError(f"Conveyor env_path_format must contain exactly one '{{}}', got {env_path_format!r}.")
    if not env_path_format.startswith("/"):
        raise ValueError(f"Conveyor env_path_format must be absolute, got {env_path_format!r}.")


def _validate_newton_surface_specs(surface_specs: Sequence[SurfaceVelocitySpec]) -> None:
    """Validate Newton-specific requirements before registering lifecycle callbacks."""
    if not surface_specs:
        raise ValueError("At least one surface-velocity specification is required.")
    prim_paths = [spec.prim_path for spec in surface_specs]
    if len(set(prim_paths)) != len(prim_paths):
        raise ValueError(f"Conveyor prim paths must be unique, got {prim_paths}.")
    for index, path in enumerate(prim_paths):
        for other in prim_paths[index + 1 :]:
            if _shape_belongs_to_prim(path, other) or _shape_belongs_to_prim(other, path):
                raise ValueError(f"Conveyor prim paths must not be ancestors of one another: {path!r}, {other!r}.")
    for spec in surface_specs:
        if spec.curved and spec.radius is None:
            raise ValueError(f"Newton requires an explicit positive radius for curved belt {spec.prim_path!r}.")


class SurfaceVelocity:
    """Adapt Newton's MuJoCo conveyor force model to the Isaac Lab lifecycle.

    The driver is created after the simulation context but before its first
    reset. It requests solved contact forces before model finalization, then
    binds model-specific buffers after solver initialization and before CUDA
    graph capture. A hard simulation reset transparently replaces that binding
    with buffers for the re-finalized model. Resolved belt indices use deterministic
    environment-major ordering across those rebuilds.
    """

    def __init__(
        self,
        num_envs: int,
        surface_specs: Sequence[SurfaceVelocitySpec],
        *,
        body_pattern: str,
        body_count_per_env: int | None = None,
        startup_duration_s: float = 1.0,
        env_path_format: str = "/World/envs/env_{}",
    ) -> None:
        """Register the force pipeline for the next Newton model initialization.

        Args:
            num_envs: Number of replicated simulation environments.
            surface_specs: Authored surface descriptions in stable within-environment order.
            body_pattern: Regular expression selecting bodies that receive surface traction.
            body_count_per_env: Expected selected body count per environment, or ``None``.
            startup_duration_s: Duration of the initial traction ramp [s].
            env_path_format: Format string resolving one exact environment root from its integer world index.
        """
        surface_specs = tuple(surface_specs)
        if not all(isinstance(spec, SurfaceVelocitySpec) for spec in surface_specs):
            raise TypeError("Every surface specification must be a SurfaceVelocitySpec.")
        _validate_newton_surface_specs(surface_specs)
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}.")
        if num_envs > 1 and any(not spec.prim_path.startswith("{ENV_REGEX_NS}/") for spec in surface_specs):
            raise ValueError(
                "Replicated conveyor environments require every belt prim_path to start with '{ENV_REGEX_NS}/'."
            )
        if not np.isfinite(startup_duration_s) or startup_duration_s <= 0.0:
            raise ValueError(f"Conveyor startup duration must be positive, got {startup_duration_s}.")
        _validate_env_path_format(env_path_format)
        if not isinstance(body_pattern, str):
            raise ValueError(f"body_pattern must be a regular-expression string, got {body_pattern!r}.")
        try:
            re.compile(body_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid body pattern: {body_pattern!r}.") from exc
        if body_count_per_env is not None and (
            not isinstance(body_count_per_env, int) or isinstance(body_count_per_env, bool) or body_count_per_env < 0
        ):
            raise ValueError("body_count_per_env must be a non-negative integer or None.")
        self._binding: _SurfaceVelocityBinding | None = None
        self._closed = False
        self._num_envs = num_envs
        self._surface_specs = surface_specs
        self._binding_kwargs = {
            "num_envs": num_envs,
            "surface_specs": self._surface_specs,
            "startup_duration_s": startup_duration_s,
            "env_path_format": env_path_format,
            "body_pattern": body_pattern,
            "body_count_per_env": body_count_per_env,
        }
        self._model_init_handle = NewtonManager.register_callback(
            self._request_contact_forces,
            PhysicsEvent.MODEL_INIT,
            name="surface_velocity_contact_attribute",
        )
        try:
            NewtonManager.register_solver_init_callback(self._bind_solver)
        except Exception:
            self._model_init_handle.deregister()
            raise

    def _require_binding(self) -> _SurfaceVelocityBinding:
        """Return the current binding or fail before solver initialization."""
        binding = self._binding
        if binding is None:
            raise RuntimeError("Surface velocity is not bound to an initialized Newton solver.")
        return binding

    @property
    def specs(self) -> tuple[SurfaceVelocitySpec, ...]:
        """Authored surface descriptions in stable within-environment order."""
        return self._surface_specs

    @property
    def surfaces_per_env(self) -> int:
        """Number of authored surfaces in each replicated environment."""
        return len(self._surface_specs)

    @property
    def num_surfaces(self) -> int:
        """Total number of resolved surfaces across all environments."""
        return self._num_envs * self.surfaces_per_env

    @property
    def count(self) -> int:
        """Alias for :attr:`num_surfaces`, matching tensor-view naming."""
        return self.num_surfaces

    @property
    def initialized(self) -> bool:
        """Whether the driver is bound to the active Newton solver."""
        return self._binding is not None

    @property
    def prim_paths(self) -> tuple[str, ...]:
        """Resolved Newton shape labels in environment-major belt order."""
        return self._require_binding().surface_paths

    @property
    def surface_paths(self) -> tuple[str, ...]:
        """Alias for :attr:`prim_paths`."""
        return self.prim_paths

    def set_velocities(self, velocities: Any, indices: Any = None) -> None:
        """Set signed surface speeds, preserving commands while surfaces are disabled."""
        self._require_binding().set_velocities(velocities, indices)

    def get_velocities(self, indices: Any = None, clone: bool = True) -> wp.array:
        """Return effective surface speeds, with disabled surfaces reported as zero."""
        return self._require_binding().get_velocities(indices, clone)

    def get_commanded_velocities(self, indices: Any = None, clone: bool = True) -> wp.array:
        """Return staged surface speeds without applying the enabled mask."""
        return self._require_binding().get_commanded_velocities(indices, clone)

    def set_enabled(self, flags: Any, indices: Any = None) -> None:
        """Enable or disable selected surfaces without discarding their speed commands."""
        self._require_binding().set_enabled(flags, indices)

    def get_enabled(self, indices: Any = None, clone: bool = True) -> wp.array:
        """Return integer enabled flags for selected surfaces."""
        return self._require_binding().get_enabled(indices, clone)

    def get_encoder_positions(self, indices: Any = None, clone: bool = True) -> wp.array:
        """Return physics-rate integrated surface travel distances [m]."""
        return self._require_binding().get_encoder_positions(indices, clone)

    def reset(self, env_ids: Any = None) -> None:
        """Clear stale force and encoder state for selected environments."""
        self._require_binding().reset(env_ids)

    def _request_contact_forces(self, _event: Any) -> None:
        """Request solved per-contact forces before the Newton model is finalized."""
        NewtonManager.request_extended_contact_attribute("force")

    def _bind_solver(self, model: Any, contacts: Any) -> None:
        """Bind graph callbacks and buffers to the newly initialized solver."""
        previous = self._binding
        settings = None
        if previous is not None:
            settings = (
                previous._command_velocity_host.copy(),
                previous._enabled_host.copy(),
            )
            previous.close()
            self._binding = None

        binding = _SurfaceVelocityBinding(model=model, contacts=contacts, **self._binding_kwargs)
        if settings is not None:
            velocities, enabled = settings
            binding.set_velocities(velocities)
            binding.set_enabled(enabled)
        self._binding = binding

    def close(self) -> None:
        """Deregister lifecycle and graph callbacks; repeated calls are safe."""
        if self._closed:
            return
        NewtonManager.unregister_solver_init_callback(self._bind_solver)
        self._model_init_handle.deregister()
        if self._binding is not None:
            self._binding.close()
            self._binding = None
        self._closed = True


class _SurfaceVelocityBinding:
    """Bind replicated surfaces and controls to Newton's conveyor force model."""

    def __init__(
        self,
        model: Any,
        contacts: Any,
        num_envs: int,
        surface_specs: Sequence[SurfaceVelocitySpec],
        *,
        body_pattern: str,
        body_count_per_env: int | None = None,
        startup_duration_s: float = 1.0,
        env_path_format: str = "/World/envs/env_{}",
    ) -> None:
        """Initialize the binding before Newton CUDA graph capture.

        Args:
            model: Finalized Newton model owned by the active solver.
            contacts: Contact buffer owned by the active solver.
            num_envs: Number of replicated simulation environments.
            surface_specs: Authored surface descriptions in stable within-environment order.
            body_pattern: Regular expression selecting bodies that receive surface traction.
            body_count_per_env: Expected selected body count per environment, or ``None``.
            startup_duration_s: Duration of the initial traction ramp [s].
            env_path_format: Format string resolving one exact environment root from its integer world index.
        """
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}.")
        if not np.isfinite(startup_duration_s) or startup_duration_s <= 0.0:
            raise ValueError(f"Conveyor startup duration must be positive, got {startup_duration_s}.")
        _validate_env_path_format(env_path_format)

        self._surface_specs = tuple(surface_specs)
        _validate_newton_surface_specs(self._surface_specs)
        try:
            compiled_body_pattern = re.compile(body_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid body pattern: {body_pattern!r}.") from exc

        if model is None or contacts is None:
            raise RuntimeError("The conveyor driver requires an initialized Newton model and contact buffer.")
        if contacts.force is None:
            raise RuntimeError(
                "Newton did not allocate per-contact force reporting. The conveyor driver must request the "
                "'force' contact attribute before model finalization."
            )
        if model.world_count != num_envs:
            raise RuntimeError(f"Newton model has {model.world_count} worlds, expected {num_envs}.")

        self._model = model
        self._contacts = contacts
        self._device = model.device
        self._num_envs = num_envs
        self._startup_duration_s = startup_duration_s
        self._closed = False
        self._validate_backend_buffers()

        surfaces_per_env = len(self._surface_specs)
        conveyor_count = num_envs * surfaces_per_env
        conveyor_shapes = [-1] * conveyor_count
        direction = [wp.vec3() for _ in range(conveyor_count)]
        pivot_point = [wp.vec3() for _ in range(conveyor_count)]
        radius = [1.0] * conveyor_count
        surface_normal = [wp.vec3() for _ in range(conveyor_count)]
        conveyor_world = [conveyor_id // surfaces_per_env for conveyor_id in range(conveyor_count)]
        surface_paths = [""] * conveyor_count

        shape_body = model.shape_body.numpy()
        shape_world = model.shape_world.numpy()
        shape_transform = model.shape_transform.numpy()
        seen_sections: set[tuple[int, int]] = set()
        for shape_id, label in enumerate(model.shape_label):
            world_id = int(shape_world[shape_id])
            if not 0 <= world_id < num_envs:
                continue
            matching_specs = [
                index
                for index, spec in enumerate(self._surface_specs)
                if _shape_belongs_to_prim(label, _resolve_belt_prim_path(spec.prim_path, env_path_format, world_id))
            ]
            if not matching_specs:
                continue
            if len(matching_specs) > 1:
                raise RuntimeError(f"Conveyor shape {label!r} matches more than one section specification.")
            if int(shape_body[shape_id]) >= 0:
                raise ValueError(f"Conveyor shape must be static: {label}")

            spec_id = matching_specs[0]
            section_key = (world_id, spec_id)
            if section_key in seen_sections:
                raise RuntimeError(
                    f"World {world_id} contains multiple shapes matching conveyor section "
                    f"{self._surface_specs[spec_id].prim_path!r}."
                )
            seen_sections.add(section_key)

            spec = self._surface_specs[spec_id]
            conveyor_id = world_id * surfaces_per_env + spec_id
            conveyor_shapes[conveyor_id] = shape_id
            direction[conveyor_id] = _world_vector(shape_transform[shape_id], spec.direction)
            pivot_point[conveyor_id] = _world_point(shape_transform[shape_id], spec.pivot_point)
            radius[conveyor_id] = 1.0 if spec.radius is None else spec.radius
            surface_normal[conveyor_id] = _world_vector(shape_transform[shape_id], spec.surface_normal)
            surface_paths[conveyor_id] = label

        expected_sections = {
            (world_id, spec_id) for world_id in range(num_envs) for spec_id in range(len(self._surface_specs))
        }
        missing_sections = sorted(expected_sections - seen_sections)
        if missing_sections:
            details = ", ".join(
                f"world {world_id}: {self._surface_specs[spec_id].prim_path}"
                for world_id, spec_id in missing_sections[:8]
            )
            raise RuntimeError(f"Missing {len(missing_sections)} conveyor collision sections ({details}).")

        body_is_tracked = np.zeros(model.body_count, dtype=np.int32)
        tracked_counts = np.zeros(num_envs, dtype=np.int32)
        body_world = model.body_world.numpy()
        for body_id, label in enumerate(model.body_label):
            if compiled_body_pattern.search(label) is None:
                continue
            world_id = int(body_world[body_id])
            if not 0 <= world_id < num_envs:
                raise RuntimeError(f"Transported body {label!r} belongs to invalid world {world_id}.")
            body_is_tracked[body_id] = 1
            tracked_counts[world_id] += 1

        if body_count_per_env is not None:
            bad_worlds = np.flatnonzero(tracked_counts != body_count_per_env)
            if bad_worlds.size:
                details = ", ".join(f"world {world_id}: {tracked_counts[world_id]}" for world_id in bad_worlds[:8])
                raise RuntimeError(
                    f"Body pattern {body_pattern!r} expected {body_count_per_env} bodies per world ({details})."
                )
        if not np.any(body_is_tracked):
            raise RuntimeError(f"Body pattern {body_pattern!r} matched no Newton bodies.")

        self._conveyor = ConveyorForceModel(model, solver_type="mujoco")
        for conveyor_id, shape_id in enumerate(conveyor_shapes):
            spec = self._surface_specs[conveyor_id % surfaces_per_env]
            parameters = {
                "surface_normal": surface_normal[conveyor_id],
                "friction": spec.friction_coefficient,
                "threshold": spec.contact_threshold,
            }
            if spec.curved:
                self._conveyor.add_pivot_belt(shape_id, pivot_point[conveyor_id], wp.vec3(), **parameters)
            else:
                self._conveyor.add_constant_belt(shape_id, wp.vec3(), **parameters)
        self._conveyor.finalize(contacts)

        self._surface_paths = tuple(surface_paths)
        self._body_is_tracked = wp.array(body_is_tracked, dtype=wp.int32, device=self._device)
        self._direction = wp.array(direction, dtype=wp.vec3, device=self._device)
        self._radius = wp.array(radius, dtype=wp.float32, device=self._device)
        self._conveyor_world = wp.array(conveyor_world, dtype=wp.int32, device=self._device)

        authored_velocity = np.asarray([spec.velocity for spec in self._surface_specs], dtype=np.float32)
        authored_enabled = np.asarray([spec.enabled for spec in self._surface_specs], dtype=np.int32)
        self._command_velocity_host = np.tile(authored_velocity, num_envs)
        self._enabled_host = np.tile(authored_enabled, num_envs)
        self._command_velocity = wp.array(self._command_velocity_host, dtype=wp.float32, device=self._device)
        self._enabled = wp.array(self._enabled_host, dtype=wp.int32, device=self._device)
        self._effective_velocity = wp.zeros(conveyor_count, dtype=wp.float32, device=self._device)
        self._encoder_position = wp.zeros(conveyor_count, dtype=wp.float32, device=self._device)
        self._elapsed_time = wp.zeros(1, dtype=wp.float32, device=self._device)
        self._conveyor.set_speed_scale(0.0)

        self._world_mask_host = np.zeros(num_envs, dtype=np.bool_)
        self._world_mask = wp.zeros(num_envs, dtype=wp.bool, device=self._device)
        self._refresh_effective_velocities()

        NewtonManager.register_state_force_callback(self.apply)
        NewtonManager.register_post_solver_substep_callback(self.update)

    @property
    def surface_paths(self) -> tuple[str, ...]:
        """Resolved Newton shape labels in conveyor-index order."""
        return self._surface_paths

    def set_velocities(self, velocities: Any, indices: Any = None) -> None:
        """Set signed surface speeds, preserving commands while surfaces are disabled."""
        selected = self._resolve_indices(indices)
        self._command_velocity_host[selected] = self._broadcast_1d(velocities, len(selected), "velocities")
        self._command_velocity.assign(self._command_velocity_host)
        self._refresh_effective_velocities()

    def get_velocities(self, indices: Any = None, clone: bool = True) -> wp.array:
        """Return effective surface speeds, with disabled surfaces reported as zero."""
        return self._get_device_values(self._effective_velocity, indices, clone)

    def get_commanded_velocities(self, indices: Any = None, clone: bool = True) -> wp.array:
        """Return staged surface speeds without applying the enabled mask."""
        return self._get_device_values(self._command_velocity, indices, clone)

    def set_enabled(self, flags: Any, indices: Any = None) -> None:
        """Enable or disable selected surfaces without discarding their speed commands."""
        selected = self._resolve_indices(indices)
        values = self._broadcast_1d(flags, len(selected), "enabled flags")
        if not np.all(np.isin(values, (0.0, 1.0))):
            raise ValueError(f"Surface enabled flags must contain {len(selected)} boolean values.")
        self._enabled_host[selected] = values.astype(np.int32)
        self._enabled.assign(self._enabled_host)
        self._refresh_effective_velocities()

    def get_enabled(self, indices: Any = None, clone: bool = True) -> wp.array:
        """Return integer enabled flags for selected surfaces."""
        return self._get_device_int_values(self._enabled, indices, clone)

    def get_encoder_positions(self, indices: Any = None, clone: bool = True) -> wp.array:
        """Return physics-rate integrated surface travel distances [m]."""
        return self._get_device_values(self._encoder_position, indices, clone)

    def reset(self, env_ids: Any = None) -> None:
        """Clear stale force and encoder state for selected environments.

        A full reset also restarts the global startup ramp. Partial vectorized
        resets leave other environments' conveyor forces and startup state intact.

        Args:
            env_ids: Environment indices to reset, or ``None`` for every environment.
        """
        if env_ids is None:
            self._conveyor.conveyor_body_f.zero_()
            self._encoder_position.zero_()
            self._elapsed_time.zero_()
            self._conveyor.global_velocity_scale.zero_()
            return

        ids = _as_numpy(env_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise IndexError(f"Surface reset environment indices must be integers, got {env_ids!r}.")
        ids = ids.astype(np.int64, copy=False).reshape(-1)
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
            device=self._device,
        )
        wp.launch(
            _clear_selected_encoders,
            dim=len(self._surface_paths),
            inputs=[self._conveyor_world, self._world_mask],
            outputs=[self._encoder_position],
            device=self._device,
        )

    def close(self) -> None:
        """Deregister Newton callbacks and release references held by the driver."""
        if self._closed:
            return
        NewtonManager.unregister_state_force_callback(self.apply)
        NewtonManager.unregister_post_solver_substep_callback(self.update)
        self._closed = True

    def apply(self, state) -> None:
        """Apply the wrench computed from the preceding physics solve."""
        self._conveyor.apply(state)

    def update(self, solver, contacts, state, dt: float) -> None:
        """Update surface travel and let Newton compute the next conveyor wrench."""
        wp.launch(
            _advance_startup_scale,
            dim=1,
            inputs=[dt, self._startup_duration_s],
            outputs=[self._elapsed_time, self._conveyor.global_velocity_scale],
            device=self._device,
        )
        wp.launch(
            _integrate_encoders,
            dim=len(self._surface_paths),
            inputs=[dt, self._effective_velocity],
            outputs=[self._encoder_position],
            device=self._device,
        )
        self._conveyor.update(solver, contacts, state, dt)
        # Exclude robot links and other bodies outside the transported-body selection.
        wp.launch(
            _filter_body_forces,
            dim=self._model.body_count,
            inputs=[self._body_is_tracked],
            outputs=[self._conveyor.conveyor_body_f],
            device=self._device,
        )

    def _validate_backend_buffers(self) -> None:
        """Validate every fixed-size Newton buffer consumed by conveyor kernels."""
        model = self._model
        contacts = self._contacts
        _require_buffer_length("model.shape_body", model.shape_body, model.shape_count)
        _require_buffer_length("model.shape_world", model.shape_world, model.shape_count)
        _require_buffer_length("model.shape_transform", model.shape_transform, model.shape_count)
        _require_buffer_length("model.body_world", model.body_world, model.body_count)
        _require_buffer_length("model.body_com", model.body_com, model.body_count)
        _require_buffer_length("model.body_inv_mass", model.body_inv_mass, model.body_count)
        _require_buffer_length("model.body_inv_inertia", model.body_inv_inertia, model.body_count)
        _require_buffer_length("contacts.force", contacts.force, contacts.rigid_contact_max)
        _require_buffer_length(
            "contacts.rigid_contact_shape0", contacts.rigid_contact_shape0, contacts.rigid_contact_max
        )
        _require_buffer_length(
            "contacts.rigid_contact_shape1", contacts.rigid_contact_shape1, contacts.rigid_contact_max
        )
        _require_buffer_length(
            "contacts.rigid_contact_normal", contacts.rigid_contact_normal, contacts.rigid_contact_max
        )
        _require_buffer_length(
            "contacts.rigid_contact_point0", contacts.rigid_contact_point0, contacts.rigid_contact_max
        )
        _require_buffer_length(
            "contacts.rigid_contact_point1", contacts.rigid_contact_point1, contacts.rigid_contact_max
        )
        _require_buffer_length(
            "contacts.rigid_contact_offset0", contacts.rigid_contact_offset0, contacts.rigid_contact_max
        )
        _require_buffer_length(
            "contacts.rigid_contact_offset1", contacts.rigid_contact_offset1, contacts.rigid_contact_max
        )
        _require_buffer_length("contacts.rigid_contact_count", contacts.rigid_contact_count, 1)

    def _resolve_indices(self, indices: Any) -> np.ndarray:
        """Normalize and validate a surface index selection."""
        if indices is None:
            return np.arange(len(self._surface_paths), dtype=np.int64)
        if isinstance(indices, slice):
            return np.arange(len(self._surface_paths), dtype=np.int64)[indices]
        selected = _as_numpy(indices)
        if selected.dtype == np.bool_:
            if selected.ndim != 1 or selected.size != len(self._surface_paths):
                raise IndexError(f"Boolean surface indices must have length {len(self._surface_paths)}.")
            return np.flatnonzero(selected).astype(np.int64)
        if not np.issubdtype(selected.dtype, np.integer):
            raise IndexError(f"Surface indices must be integers, got {indices!r}.")
        selected = selected.astype(np.int64, copy=False).reshape(-1)
        if np.any((selected < 0) | (selected >= len(self._surface_paths))):
            raise IndexError(f"Conveyor surface indices are out of range: {selected.tolist()}.")
        return selected

    def _refresh_effective_velocities(self) -> None:
        """Apply the enabled mask at the one device-side command seam."""
        wp.launch(
            _update_effective_velocities,
            dim=len(self._surface_paths),
            inputs=[self._command_velocity, self._enabled],
            outputs=[self._effective_velocity],
            device=self._device,
        )
        wp.launch(
            _update_belt_velocities,
            dim=len(self._surface_paths),
            inputs=[self._direction, self._radius, self._effective_velocity],
            outputs=[self._conveyor.conv_const_vel, self._conveyor.conv_pivot_angvel],
            device=self._device,
        )

    def _get_device_values(self, source: wp.array, indices: Any, clone: bool) -> wp.array:
        """Clone a complete device buffer or gather a selected subset."""
        if indices is None:
            return wp.clone(source) if clone else source
        selected = self._resolve_indices(indices)
        selected_device = wp.array(selected, dtype=wp.int32, device=self._device)
        values = wp.empty(len(selected), dtype=wp.float32, device=self._device)
        if len(selected) > 0:
            wp.launch(
                _gather_float_values,
                dim=len(selected),
                inputs=[source, selected_device],
                outputs=[values],
                device=self._device,
            )
        return values

    def _get_device_int_values(self, source: wp.array, indices: Any, clone: bool) -> wp.array:
        """Clone an integer device buffer or gather a selected subset."""
        if indices is None:
            return wp.clone(source) if clone else source
        selected = self._resolve_indices(indices)
        selected_device = wp.array(selected, dtype=wp.int32, device=self._device)
        values = wp.empty(len(selected), dtype=wp.int32, device=self._device)
        if len(selected) > 0:
            wp.launch(
                _gather_int_values,
                dim=len(selected),
                inputs=[source, selected_device],
                outputs=[values],
                device=self._device,
            )
        return values

    @staticmethod
    def _broadcast_1d(values: Any, count: int, name: str) -> np.ndarray:
        """Broadcast one scalar or validate one value per selected surface."""
        array = np.asarray(_as_numpy(values), dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Conveyor {name} must contain only finite values.")
        if array.ndim == 0 or array.size == 1:
            return np.full(count, float(array.reshape(-1)[0]), dtype=np.float32)
        if array.ndim != 1 or array.size != count:
            raise ValueError(f"Conveyor {name} need one value or {count} values, got shape {array.shape}.")
        return array
