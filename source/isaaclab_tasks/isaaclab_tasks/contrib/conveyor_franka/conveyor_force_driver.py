# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched contact-force conveyor surfaces for Newton physics.

The driver reads solver-reported normal contact forces, computes a Coulomb-limited force that
drives each transported body's contact points toward their conveyor velocity fields, and applies
the resulting body wrenches on the following physics solve.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp
from isaaclab_newton.physics import NewtonManager

from .conveyor_geometry import ConveyorSectionSpec

_VELOCITY_FIELD_TYPE_CONSTANT = 0
_VELOCITY_FIELD_TYPE_PIVOT = 1
_SAME_NORMAL_THRESHOLD = 0.999


@wp.struct
class Vec3Pair:
    """Two orthonormal vectors spanning a contact tangent plane."""

    v0: wp.vec3
    v1: wp.vec3


@wp.func
def compute_basis_vectors(direction: wp.vec3) -> Vec3Pair:
    """Return the reference conveyor's tangent basis for a unit direction."""
    basis = Vec3Pair()
    if wp.abs(direction[1]) <= 0.9999:
        basis.v0 = wp.normalize(wp.vec3(direction[2], 0.0, -direction[0]))
        basis.v1 = wp.vec3(
            direction[1] * basis.v0[2],
            (direction[2] * basis.v0[0]) - (direction[0] * basis.v0[2]),
            -direction[1] * basis.v0[0],
        )
    else:
        basis.v0 = wp.vec3(1.0, 0.0, 0.0)
        basis.v1 = wp.normalize(wp.vec3(0.0, direction[2], -direction[1]))
    return basis


@wp.func
def compute_point_impulse(
    normal: wp.vec3,
    normal_impulse: wp.float32,
    current_vel: wp.vec3,
    target_vel: wp.vec3,
    response_linear: wp.float32,
    inv_inertia_world: wp.mat33,
    center_of_mass_to_point: wp.vec3,
    friction_coefficient: wp.float32,
    mass_splitting_scale: wp.float32,
) -> wp.vec3:
    """Compute a Coulomb-clamped tangential impulse using point effective mass."""
    rel_vel = target_vel - current_vel
    basis = compute_basis_vectors(normal)

    r_cross_t0 = wp.cross(center_of_mass_to_point, basis.v0)
    r_cross_t1 = wp.cross(center_of_mass_to_point, basis.v1)
    k00 = response_linear + wp.dot(r_cross_t0, wp.mul(inv_inertia_world, r_cross_t0))
    k11 = response_linear + wp.dot(r_cross_t1, wp.mul(inv_inertia_world, r_cross_t1))
    k01 = wp.dot(r_cross_t0, wp.mul(inv_inertia_world, r_cross_t1))
    det = (k00 * k11) - (k01 * k01)

    i0 = wp.float32(0.0)
    i1 = wp.float32(0.0)
    if det > 0.0:
        v0 = wp.dot(basis.v0, rel_vel)
        v1 = wp.dot(basis.v1, rel_vel)
        i0 = ((k11 * v0) - (k01 * v1)) * mass_splitting_scale / det
        i1 = ((k00 * v1) - (k01 * v0)) * mass_splitting_scale / det

    friction_impulse_max = normal_impulse * friction_coefficient
    zero_err_magn = wp.sqrt((i0 * i0) + (i1 * i1))
    impulse_magn = wp.min(friction_impulse_max, zero_err_magn)
    if zero_err_magn > 0.0:
        ratio = impulse_magn / zero_err_magn
    else:
        ratio = 0.0
    return (basis.v0 * (i0 * ratio)) + (basis.v1 * (i1 * ratio))


@wp.func
def compute_point_force(
    dt: wp.float32,
    inverse_dt: wp.float32,
    com_world: wp.vec3,
    body_inverse_mass: wp.float32,
    body_inverse_inertia_world: wp.mat33,
    body_linear_velocity: wp.vec3,
    body_angular_velocity: wp.vec3,
    contact_position: wp.vec3,
    contact_normal: wp.vec3,
    contact_force: wp.float32,
    mass_splitting_scale: wp.float32,
    target_vel: wp.vec3,
    friction_coefficient: wp.float32,
) -> wp.spatial_vector:
    """Compute force and torque at one conveyor contact."""
    contact_impulse = contact_force * dt
    center_of_mass_to_point = contact_position - com_world
    current_point_vel = body_linear_velocity + wp.cross(body_angular_velocity, center_of_mass_to_point)

    tangential_impulse = compute_point_impulse(
        contact_normal,
        contact_impulse,
        current_point_vel,
        target_vel,
        body_inverse_mass,
        body_inverse_inertia_world,
        center_of_mass_to_point,
        friction_coefficient,
        mass_splitting_scale,
    )

    force = tangential_impulse * inverse_dt
    torque = wp.cross(center_of_mass_to_point, force)
    return wp.spatial_vector(force, torque)


@wp.struct
class BeltContact:
    """Reduced contact data consumed by the conveyor force kernels."""

    valid: wp.int32
    body: wp.int32
    conveyor: wp.int32
    point: wp.vec3
    normal: wp.vec3
    normal_force: wp.float32
    next_body_contact: wp.int32


@wp.kernel
def _extract_linear_force(spatial_force: wp.array[wp.spatial_vector], force: wp.array[wp.vec3]):
    contact_id = wp.tid()
    force[contact_id] = wp.spatial_top(spatial_force[contact_id])


@wp.kernel
def _classify_contacts(
    contact_count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    normal: wp.array[wp.vec3],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    contact_force: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_conveyor: wp.array[wp.int32],
    body_is_tracked: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    conveyor_surface_normal: wp.array[wp.vec3],
    conveyor_threshold: wp.array[wp.float32],
    contacts_out: wp.array[BeltContact],
    body_contact_head: wp.array[wp.int32],
):
    contact_id = wp.tid()
    result = BeltContact()
    result.valid = 0
    result.next_body_contact = -1

    if contact_id < contact_count[0]:
        contact_shape0 = shape0[contact_id]
        contact_shape1 = shape1[contact_id]
        if contact_shape0 >= 0 and contact_shape1 >= 0:
            conveyor0 = shape_conveyor[contact_shape0]
            conveyor1 = shape_conveyor[contact_shape1]
            contact_normal = normal[contact_id]

            body = wp.int32(-1)
            conveyor = wp.int32(-1)
            local_point = wp.vec3()
            normal_toward_body = wp.vec3()
            if conveyor0 >= 0 and conveyor1 < 0:
                conveyor = conveyor0
                body = shape_body[contact_shape1]
                local_point = point1[contact_id]
                normal_toward_body = contact_normal
            elif conveyor1 >= 0 and conveyor0 < 0:
                conveyor = conveyor1
                body = shape_body[contact_shape0]
                local_point = point0[contact_id]
                normal_toward_body = -contact_normal

            if body >= 0 and conveyor >= 0 and body_is_tracked[body] != 0:
                alignment = wp.dot(normal_toward_body, conveyor_surface_normal[conveyor])
                normal_force = wp.abs(wp.dot(contact_force[contact_id], contact_normal))
                if alignment >= conveyor_threshold[conveyor] and normal_force > 0.0:
                    result.valid = 1
                    result.body = body
                    result.conveyor = conveyor
                    result.point = wp.transform_point(body_q[body], local_point)
                    result.normal = normal_toward_body
                    result.normal_force = normal_force
                    result.next_body_contact = wp.atomic_exch(body_contact_head, body, contact_id)

    contacts_out[contact_id] = result


@wp.kernel
def _prepare_contact_patches(
    contacts: wp.array[BeltContact],
    body_contact_head: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    contact_patch_head: wp.array[wp.int32],
    adjusted_normal_force: wp.array[wp.float32],
    mass_splitting_scale: wp.array[wp.float32],
):
    """Correlate contacts by normal and normalize loads across overlapping sections."""
    body_id = wp.tid()
    reference_point = wp.transform_point(body_q[body_id], body_com[body_id])
    patch_contact_id = body_contact_head[body_id]

    while patch_contact_id >= 0:
        if contact_patch_head[patch_contact_id] < 0:
            patch_contact = contacts[patch_contact_id]
            basis = compute_basis_vectors(patch_contact.normal)
            point_count = wp.int32(0)
            first_conveyor = patch_contact.conveyor
            spans_multiple_conveyors = wp.int32(0)
            patch_force_sum = wp.float32(0.0)
            min0 = wp.float32(1.0e30)
            max0 = wp.float32(-1.0e30)
            min1 = wp.float32(1.0e30)
            max1 = wp.float32(-1.0e30)

            contact_id = body_contact_head[body_id]
            while contact_id >= 0:
                contact = contacts[contact_id]
                if (
                    contact_patch_head[contact_id] < 0
                    and wp.dot(patch_contact.normal, contact.normal) > _SAME_NORMAL_THRESHOLD
                ):
                    contact_patch_head[contact_id] = patch_contact_id
                    point_count += 1
                    if contact.conveyor != first_conveyor:
                        spans_multiple_conveyors = 1
                    patch_force_sum += contact.normal_force

                    delta = contact.point - reference_point
                    projection0 = wp.dot(basis.v0, delta)
                    projection1 = wp.dot(basis.v1, delta)
                    min0 = wp.min(min0, projection0)
                    max0 = wp.max(max0, projection0)
                    min1 = wp.min(min1, projection1)
                    max1 = wp.max(max1, projection1)
                contact_id = contact.next_body_contact

            splitting_scale = 1.0 / wp.float32(point_count)
            if point_count == 1 or spans_multiple_conveyors == 0:
                contact_id = body_contact_head[body_id]
                while contact_id >= 0:
                    contact = contacts[contact_id]
                    if contact_patch_head[contact_id] == patch_contact_id:
                        adjusted_normal_force[contact_id] = contact.normal_force
                        mass_splitting_scale[contact_id] = splitting_scale
                    contact_id = contact.next_body_contact
            else:
                kernel_radius = 0.25 * ((max0 - min0) + (max1 - min1))
                kernel_radius_sqr = kernel_radius * kernel_radius
                if kernel_radius > 0.0:
                    point_force_weight_sum = wp.float32(0.0)
                    contact_id = body_contact_head[body_id]
                    while contact_id >= 0:
                        contact = contacts[contact_id]
                        if contact_patch_head[contact_id] == patch_contact_id:
                            density = wp.float32(1.0)
                            other_contact_id = body_contact_head[body_id]
                            while other_contact_id >= 0:
                                other_contact = contacts[other_contact_id]
                                if (
                                    other_contact_id != contact_id
                                    and contact_patch_head[other_contact_id] == patch_contact_id
                                ):
                                    delta = contact.point - other_contact.point
                                    projected_delta = delta - (
                                        wp.dot(delta, patch_contact.normal) * patch_contact.normal
                                    )
                                    density += wp.exp(-0.5 * wp.length_sq(projected_delta) / kernel_radius_sqr)
                                other_contact_id = other_contact.next_body_contact

                            weight = 1.0 / density
                            adjusted_normal_force[contact_id] = weight
                            mass_splitting_scale[contact_id] = splitting_scale
                            point_force_weight_sum += weight
                        contact_id = contact.next_body_contact

                    force_per_weight = patch_force_sum / point_force_weight_sum
                    contact_id = body_contact_head[body_id]
                    while contact_id >= 0:
                        contact = contacts[contact_id]
                        if contact_patch_head[contact_id] == patch_contact_id:
                            adjusted_normal_force[contact_id] *= force_per_weight
                        contact_id = contact.next_body_contact
                else:
                    adjusted_force = patch_force_sum / wp.float32(point_count)
                    contact_id = body_contact_head[body_id]
                    while contact_id >= 0:
                        contact = contacts[contact_id]
                        if contact_patch_head[contact_id] == patch_contact_id:
                            adjusted_normal_force[contact_id] = adjusted_force
                            mass_splitting_scale[contact_id] = splitting_scale
                        contact_id = contact.next_body_contact

        patch_contact_id = contacts[patch_contact_id].next_body_contact


@wp.kernel
def _accumulate_forces(
    dt: wp.float32,
    contacts: wp.array[BeltContact],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
    adjusted_normal_force: wp.array[wp.float32],
    mass_splitting_scale: wp.array[wp.float32],
    conveyor_field_type: wp.array[wp.int32],
    conveyor_direction: wp.array[wp.vec3],
    conveyor_pivot_point: wp.array[wp.vec3],
    conveyor_radius: wp.array[wp.float32],
    conveyor_effective_velocity: wp.array[wp.float32],
    conveyor_friction: wp.array[wp.float32],
    velocity_scale: wp.array[wp.float32],
    body_force: wp.array[wp.spatial_vector],
):
    contact_id = wp.tid()
    contact = contacts[contact_id]
    if contact.valid == 0:
        return

    splitting_scale = mass_splitting_scale[contact_id]
    if splitting_scale <= 0.0:
        return

    conveyor = contact.conveyor
    effective_velocity = conveyor_effective_velocity[conveyor] * velocity_scale[0]
    if conveyor_field_type[conveyor] == _VELOCITY_FIELD_TYPE_CONSTANT:
        target_velocity = conveyor_direction[conveyor] * effective_velocity
    else:
        angular_velocity = conveyor_direction[conveyor] * (effective_velocity / conveyor_radius[conveyor])
        target_velocity = wp.cross(angular_velocity, contact.point - conveyor_pivot_point[conveyor])

    pose = body_q[contact.body]
    center_of_mass = wp.transform_point(pose, body_com[contact.body])
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    inverse_inertia_world = rotation * body_inv_inertia[contact.body] * wp.transpose(rotation)
    velocity = body_qd[contact.body]

    force = compute_point_force(
        dt,
        1.0 / dt,
        center_of_mass,
        body_inv_mass[contact.body],
        inverse_inertia_world,
        wp.spatial_top(velocity),
        wp.spatial_bottom(velocity),
        contact.point,
        contact.normal,
        adjusted_normal_force[contact_id],
        splitting_scale,
        target_velocity,
        conveyor_friction[conveyor],
    )
    wp.atomic_add(body_force, contact.body, force)


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
def _add_body_force(dst: wp.array[wp.spatial_vector], src: wp.array[wp.spatial_vector]):
    body_id = wp.tid()
    dst[body_id] = dst[body_id] + src[body_id]


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


class ConveyorForceDriver:
    """Run one batched moving-surface force pipeline for a Newton scene."""

    def __init__(
        self,
        num_envs: int,
        surface_specs: Sequence[ConveyorSectionSpec],
        speed: float = 0.35,
        friction: float = 0.5,
        normal_threshold: float = 0.997,
        startup_duration_s: float = 1.0,
        transported_body_pattern: str = r"(?:^|/)Cube_?[0-3](?:/|$)",
        transported_body_count_per_env: int | None = None,
    ) -> None:
        """Initialize the driver after Newton simulation startup.

        Args:
            num_envs: Number of replicated simulation environments.
            surface_specs: Collision sections and matching velocity fields.
            speed: Initial signed conveyor surface speed [m/s].
            friction: Coulomb friction coefficient used to limit traction.
            normal_threshold: Minimum contact-normal alignment in the range [0, 1].
            startup_duration_s: Duration of the initial traction ramp [s].
            transported_body_pattern: Regular expression selecting bodies that receive traction.
            transported_body_count_per_env: Expected selected body count per environment, or ``None``.
        """
        if num_envs <= 0:
            raise ValueError(f"Number of conveyor environments must be positive, got {num_envs}.")
        if not np.isfinite(speed):
            raise ValueError(f"Conveyor speed must be finite, got {speed}.")
        if not np.isfinite(friction) or friction < 0.0:
            raise ValueError(f"Conveyor friction must be non-negative, got {friction}.")
        if not np.isfinite(normal_threshold) or not 0.0 <= normal_threshold <= 1.0:
            raise ValueError(f"Conveyor normal threshold must be in [0, 1], got {normal_threshold}.")
        if not np.isfinite(startup_duration_s) or startup_duration_s <= 0.0:
            raise ValueError(f"Conveyor startup duration must be positive, got {startup_duration_s}.")

        self._surface_specs = tuple(surface_specs)
        self._validate_surface_specs()
        try:
            body_pattern = re.compile(transported_body_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid transported-body pattern: {transported_body_pattern!r}.") from exc

        model = NewtonManager.get_model()
        contacts = NewtonManager.get_contacts()
        if model is None or contacts is None:
            raise RuntimeError("The conveyor driver must be created after Newton simulation initialization.")
        if contacts.force is None:
            raise RuntimeError(
                "Newton did not allocate per-contact force reporting. The scene contact sensor must initialize "
                "before the conveyor driver."
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

        shape_conveyor = [-1] * model.shape_count
        field_type: list[int] = []
        direction: list[wp.vec3] = []
        pivot_point: list[wp.vec3] = []
        radius: list[float] = []
        surface_normal: list[wp.vec3] = []
        conveyor_world: list[int] = []
        surface_paths: list[str] = []

        shape_body = model.shape_body.numpy()
        shape_world = model.shape_world.numpy()
        shape_transform = model.shape_transform.numpy()
        patterns = tuple(re.compile(rf"(?:^|/){re.escape(spec.mesh.name)}(?:/|$)") for spec in self._surface_specs)
        seen_sections: set[tuple[int, int]] = set()
        for shape_id, label in enumerate(model.shape_label):
            matching_specs = [index for index, pattern in enumerate(patterns) if pattern.search(label)]
            if not matching_specs:
                continue
            if len(matching_specs) > 1:
                raise RuntimeError(f"Conveyor shape {label!r} matches more than one section specification.")
            if int(shape_body[shape_id]) >= 0:
                raise ValueError(f"Conveyor shape must be static: {label}")

            spec_id = matching_specs[0]
            world_id = int(shape_world[shape_id])
            if not 0 <= world_id < num_envs:
                raise RuntimeError(f"Conveyor shape {label!r} belongs to invalid world {world_id}.")
            section_key = (world_id, spec_id)
            if section_key in seen_sections:
                raise RuntimeError(
                    f"World {world_id} contains multiple shapes matching conveyor section "
                    f"{self._surface_specs[spec_id].mesh.name!r}."
                )
            seen_sections.add(section_key)

            spec = self._surface_specs[spec_id]
            conveyor_id = len(field_type)
            shape_conveyor[shape_id] = conveyor_id
            field_type.append(
                _VELOCITY_FIELD_TYPE_CONSTANT if spec.velocity_field_type == "constant" else _VELOCITY_FIELD_TYPE_PIVOT
            )
            direction.append(_world_vector(shape_transform[shape_id], spec.direction))
            pivot_point.append(_world_point(shape_transform[shape_id], spec.pivot_point))
            radius.append(1.0 if spec.radius is None else spec.radius)
            surface_normal.append(_world_vector(shape_transform[shape_id], spec.surface_normal))
            conveyor_world.append(world_id)
            surface_paths.append(label)

        expected_sections = {(world_id, spec_id) for world_id in range(num_envs) for spec_id in range(len(patterns))}
        missing_sections = sorted(expected_sections - seen_sections)
        if missing_sections:
            details = ", ".join(
                f"world {world_id}: {self._surface_specs[spec_id].mesh.name}"
                for world_id, spec_id in missing_sections[:8]
            )
            raise RuntimeError(f"Missing {len(missing_sections)} conveyor collision sections ({details}).")

        body_is_tracked = np.zeros(model.body_count, dtype=np.int32)
        tracked_counts = np.zeros(num_envs, dtype=np.int32)
        body_world = model.body_world.numpy()
        for body_id, label in enumerate(model.body_label):
            if body_pattern.search(label) is None:
                continue
            world_id = int(body_world[body_id])
            if not 0 <= world_id < num_envs:
                raise RuntimeError(f"Transported body {label!r} belongs to invalid world {world_id}.")
            body_is_tracked[body_id] = 1
            tracked_counts[world_id] += 1

        if transported_body_count_per_env is not None:
            bad_worlds = np.flatnonzero(tracked_counts != transported_body_count_per_env)
            if bad_worlds.size:
                details = ", ".join(f"world {world_id}: {tracked_counts[world_id]}" for world_id in bad_worlds[:8])
                raise RuntimeError(
                    f"Transported-body pattern {transported_body_pattern!r} expected "
                    f"{transported_body_count_per_env} bodies per world ({details})."
                )
        if not np.any(body_is_tracked):
            raise RuntimeError(f"Transported-body pattern {transported_body_pattern!r} matched no Newton bodies.")

        conveyor_count = len(field_type)
        self._surface_paths = tuple(surface_paths)
        self._shape_conveyor = wp.array(shape_conveyor, dtype=wp.int32, device=self._device)
        self._body_is_tracked = wp.array(body_is_tracked, dtype=wp.int32, device=self._device)
        self._field_type = wp.array(field_type, dtype=wp.int32, device=self._device)
        self._direction = wp.array(direction, dtype=wp.vec3, device=self._device)
        self._pivot_point = wp.array(pivot_point, dtype=wp.vec3, device=self._device)
        self._radius = wp.array(radius, dtype=wp.float32, device=self._device)
        self._surface_normal = wp.array(surface_normal, dtype=wp.vec3, device=self._device)
        self._conveyor_world = wp.array(conveyor_world, dtype=wp.int32, device=self._device)

        self._command_velocity_host = np.full(conveyor_count, speed, dtype=np.float32)
        self._enabled_host = np.ones(conveyor_count, dtype=np.int32)
        self._friction_host = np.full(conveyor_count, friction, dtype=np.float32)
        self._threshold_host = np.full(conveyor_count, normal_threshold, dtype=np.float32)
        self._command_velocity = wp.array(self._command_velocity_host, dtype=wp.float32, device=self._device)
        self._enabled = wp.array(self._enabled_host, dtype=wp.int32, device=self._device)
        self._effective_velocity = wp.zeros(conveyor_count, dtype=wp.float32, device=self._device)
        self._friction = wp.array(self._friction_host, dtype=wp.float32, device=self._device)
        self._threshold = wp.array(self._threshold_host, dtype=wp.float32, device=self._device)
        self._encoder_position = wp.zeros(conveyor_count, dtype=wp.float32, device=self._device)
        self._elapsed_time = wp.zeros(1, dtype=wp.float32, device=self._device)
        self._velocity_scale = wp.zeros(1, dtype=wp.float32, device=self._device)

        contact_capacity = contacts.rigid_contact_max
        self._contact_force = wp.zeros(contact_capacity, dtype=wp.vec3, device=self._device)
        self._belt_contacts = wp.empty(contact_capacity, dtype=BeltContact, device=self._device)
        self._body_contact_head = wp.full(model.body_count, -1, dtype=wp.int32, device=self._device)
        self._contact_patch_head = wp.full(contact_capacity, -1, dtype=wp.int32, device=self._device)
        self._adjusted_normal_force = wp.zeros(contact_capacity, dtype=wp.float32, device=self._device)
        self._mass_splitting_scale = wp.zeros(contact_capacity, dtype=wp.float32, device=self._device)
        self._body_force = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=self._device)
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
        values = self._broadcast_1d(flags, len(selected), "enabled flags").astype(np.bool_)
        self._enabled_host[selected] = values.astype(np.int32)
        self._enabled.assign(self._enabled_host)
        self._refresh_effective_velocities()

    def set_friction_coefficients(self, coefficients: Any, indices: Any = None) -> None:
        """Set Coulomb traction limits for selected surfaces."""
        selected = self._resolve_indices(indices)
        values = self._broadcast_1d(coefficients, len(selected), "friction coefficients")
        if np.any(values < 0.0):
            raise ValueError("Conveyor friction coefficients must be non-negative.")
        self._friction_host[selected] = values
        self._friction.assign(self._friction_host)

    def set_contact_processing_thresholds(self, thresholds: Any, indices: Any = None) -> None:
        """Set minimum contact-normal alignment for selected surfaces."""
        selected = self._resolve_indices(indices)
        values = self._broadcast_1d(thresholds, len(selected), "contact thresholds")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Conveyor contact thresholds must lie in [0, 1].")
        self._threshold_host[selected] = values
        self._threshold.assign(self._threshold_host)

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
            self._body_force.zero_()
            self._encoder_position.zero_()
            self._elapsed_time.zero_()
            self._velocity_scale.zero_()
            return

        ids = np.asarray(_as_numpy(env_ids), dtype=np.int64).reshape(-1)
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
            outputs=[self._body_force],
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
        wp.launch(
            _add_body_force,
            dim=self._model.body_count,
            inputs=[state.body_f, self._body_force],
            device=self._device,
        )

    def update(self, solver, contacts, state, dt: float) -> None:
        """Read solved contacts and compute the next per-solve conveyor wrench."""
        solver.update_contacts(contacts)
        self._body_force.zero_()
        self._body_contact_head.fill_(-1)
        self._contact_patch_head.fill_(-1)
        self._mass_splitting_scale.zero_()
        wp.launch(
            _advance_startup_scale,
            dim=1,
            inputs=[dt, self._startup_duration_s],
            outputs=[self._elapsed_time, self._velocity_scale],
            device=self._device,
        )
        wp.launch(
            _integrate_encoders,
            dim=len(self._surface_paths),
            inputs=[dt, self._effective_velocity],
            outputs=[self._encoder_position],
            device=self._device,
        )
        wp.launch(
            _extract_linear_force,
            dim=self._contacts.rigid_contact_max,
            inputs=[contacts.force, self._contact_force],
            device=self._device,
        )
        wp.launch(
            _classify_contacts,
            dim=self._contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                self._contact_force,
                self._model.shape_body,
                self._shape_conveyor,
                self._body_is_tracked,
                state.body_q,
                self._surface_normal,
                self._threshold,
            ],
            outputs=[self._belt_contacts, self._body_contact_head],
            device=self._device,
        )
        wp.launch(
            _prepare_contact_patches,
            dim=self._model.body_count,
            inputs=[self._belt_contacts, self._body_contact_head, state.body_q, self._model.body_com],
            outputs=[self._contact_patch_head, self._adjusted_normal_force, self._mass_splitting_scale],
            device=self._device,
        )
        wp.launch(
            _accumulate_forces,
            dim=self._contacts.rigid_contact_max,
            inputs=[
                dt,
                self._belt_contacts,
                state.body_q,
                state.body_qd,
                self._model.body_com,
                self._model.body_inv_mass,
                self._model.body_inv_inertia,
                self._adjusted_normal_force,
                self._mass_splitting_scale,
                self._field_type,
                self._direction,
                self._pivot_point,
                self._radius,
                self._effective_velocity,
                self._friction,
                self._velocity_scale,
            ],
            outputs=[self._body_force],
            device=self._device,
        )

    def _validate_surface_specs(self) -> None:
        """Validate structural surface descriptions before resolving Newton shapes."""
        if not self._surface_specs:
            raise ValueError("At least one conveyor surface specification is required.")
        names = [spec.mesh.name for spec in self._surface_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Conveyor surface names must be unique, got {names}.")
        for spec in self._surface_specs:
            if spec.velocity_field_type not in {"constant", "pivot"}:
                raise ValueError(
                    f"Unknown velocity field {spec.velocity_field_type!r} for conveyor surface {spec.mesh.name!r}."
                )
            direction = np.asarray(spec.direction, dtype=np.float64)
            pivot_point = np.asarray(spec.pivot_point, dtype=np.float64)
            surface_normal = np.asarray(spec.surface_normal, dtype=np.float64)
            if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 1.0e-8:
                raise ValueError(f"Conveyor surface {spec.mesh.name!r} needs a non-zero 3-D direction.")
            if pivot_point.shape != (3,) or not np.all(np.isfinite(pivot_point)):
                raise ValueError(f"Conveyor surface {spec.mesh.name!r} needs a 3-D pivot point.")
            if (
                surface_normal.shape != (3,)
                or not np.all(np.isfinite(surface_normal))
                or np.linalg.norm(surface_normal) <= 1.0e-8
            ):
                raise ValueError(f"Conveyor surface {spec.mesh.name!r} needs a non-zero 3-D surface normal.")
            if spec.velocity_field_type == "pivot" and (
                spec.radius is None or not np.isfinite(spec.radius) or spec.radius <= 0.0
            ):
                raise ValueError(f"Pivot conveyor surface {spec.mesh.name!r} needs a positive arc radius.")

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
        _require_buffer_length("contacts.rigid_contact_count", contacts.rigid_contact_count, 1)

    def _resolve_indices(self, indices: Any) -> np.ndarray:
        """Normalize and validate a conveyor index selection."""
        if indices is None:
            return np.arange(len(self._surface_paths), dtype=np.int64)
        selected = np.asarray(_as_numpy(indices), dtype=np.int64).reshape(-1)
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
