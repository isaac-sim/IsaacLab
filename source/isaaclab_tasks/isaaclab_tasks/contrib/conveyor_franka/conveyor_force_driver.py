# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local force model for static conveyor surfaces under Newton physics.

The driver reads solver-reported normal contact forces, computes a Coulomb-limited force that
drives each parcel's contact point toward the belt velocity, and applies that wrench on the
following physics step.
"""

from __future__ import annotations

import re

import warp as wp

from isaaclab_newton.physics import NewtonManager

from .conveyor_geometry import BELT_CENTER_X, BELT_CENTER_Y, BELT_HALF_STRAIGHT, belt_direction

_BELT_LABEL = re.compile(r"Conveyor(Left|Right)Belt")


@wp.struct
class BeltContact:
    """Reduced contact data consumed by the conveyor force kernel."""

    valid: wp.int32
    body: wp.int32
    point: wp.vec3
    normal: wp.vec3
    normal_force: wp.float32
    target_velocity: wp.vec3


@wp.kernel
def _extract_linear_force(spatial_force: wp.array[wp.spatial_vector], force: wp.array[wp.vec3]):
    contact_id = wp.tid()
    force[contact_id] = wp.spatial_top(spatial_force[contact_id])


@wp.func
def _racetrack_velocity(
    point: wp.vec3,
    center: wp.vec3,
    half_straight: wp.float32,
    direction: wp.float32,
    speed: wp.float32,
) -> wp.vec3:
    relative = point - center
    tangent = wp.vec3()
    if relative[0] > half_straight:
        radial = wp.vec3(relative[0] - half_straight, relative[1], 0.0)
        radial_length = wp.length(radial)
        if radial_length > 0.0:
            tangent = wp.vec3(radial[1], -radial[0], 0.0) / radial_length
    elif relative[0] < -half_straight:
        radial = wp.vec3(relative[0] + half_straight, relative[1], 0.0)
        radial_length = wp.length(radial)
        if radial_length > 0.0:
            tangent = wp.vec3(radial[1], -radial[0], 0.0) / radial_length
    elif relative[1] >= 0.0:
        tangent = wp.vec3(1.0, 0.0, 0.0)
    else:
        tangent = wp.vec3(-1.0, 0.0, 0.0)
    return tangent * direction * speed


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
    shape_is_belt: wp.array[wp.int32],
    shape_belt_center: wp.array[wp.vec3],
    shape_belt_direction: wp.array[wp.float32],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    half_straight: wp.float32,
    speed: wp.float32,
    normal_threshold: wp.float32,
    contacts_out: wp.array[BeltContact],
    body_contact_count: wp.array[wp.int32],
):
    contact_id = wp.tid()
    result = BeltContact()
    result.valid = 0

    if contact_id < contact_count[0]:
        contact_shape0 = shape0[contact_id]
        contact_shape1 = shape1[contact_id]
        if contact_shape0 >= 0 and contact_shape1 >= 0:
            belt0 = shape_is_belt[contact_shape0]
            belt1 = shape_is_belt[contact_shape1]
            contact_normal = normal[contact_id]

            body = wp.int32(-1)
            belt_shape = wp.int32(-1)
            local_point = wp.vec3()
            normal_toward_body = wp.vec3()
            if belt0 == 1 and belt1 == 0:
                belt_shape = contact_shape0
                body = shape_body[contact_shape1]
                local_point = point1[contact_id]
                normal_toward_body = contact_normal
            elif belt1 == 1 and belt0 == 0:
                belt_shape = contact_shape1
                body = shape_body[contact_shape0]
                local_point = point0[contact_id]
                normal_toward_body = -contact_normal

            alignment = wp.dot(normal_toward_body, wp.vec3(0.0, 0.0, 1.0))
            normal_force = wp.abs(wp.dot(contact_force[contact_id], contact_normal))
            if body >= 0 and belt_shape >= 0 and alignment >= normal_threshold and normal_force > 0.0:
                result.valid = 1
                result.body = body
                result.point = wp.transform_point(body_q[body], local_point)
                result.normal = normal_toward_body
                result.normal_force = normal_force
                belt_center = wp.transform_point(shape_transform[belt_shape], shape_belt_center[belt_shape])
                result.target_velocity = _racetrack_velocity(
                    result.point,
                    belt_center,
                    half_straight,
                    shape_belt_direction[belt_shape],
                    speed,
                )
                wp.atomic_add(body_contact_count, body, 1)

    contacts_out[contact_id] = result


@wp.kernel
def _accumulate_forces(
    dt: wp.float32,
    friction: wp.float32,
    contacts: wp.array[BeltContact],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    body_contact_count: wp.array[wp.int32],
    body_force: wp.array[wp.spatial_vector],
):
    contact_id = wp.tid()
    contact = contacts[contact_id]
    if contact.valid == 0:
        return

    count = body_contact_count[contact.body]
    inverse_mass = body_inv_mass[contact.body]
    if count <= 0 or inverse_mass <= 0.0:
        return

    pose = body_q[contact.body]
    center_of_mass = wp.transform_point(pose, body_com[contact.body])
    center_to_contact = contact.point - center_of_mass
    velocity = body_qd[contact.body]
    point_velocity = wp.spatial_top(velocity) + wp.cross(wp.spatial_bottom(velocity), center_to_contact)

    velocity_error = contact.target_velocity - point_velocity
    velocity_error = velocity_error - contact.normal * wp.dot(velocity_error, contact.normal)
    desired_force = velocity_error / (inverse_mass * dt * float(count))

    desired_magnitude = wp.length(desired_force)
    max_magnitude = friction * contact.normal_force
    if desired_magnitude > max_magnitude and desired_magnitude > 0.0:
        desired_force = desired_force * (max_magnitude / desired_magnitude)

    torque = wp.cross(center_to_contact, desired_force)
    wp.atomic_add(body_force, contact.body, wp.spatial_vector(desired_force, torque))


@wp.kernel
def _add_body_force(dst: wp.array[wp.spatial_vector], src: wp.array[wp.spatial_vector]):
    body_id = wp.tid()
    dst[body_id] = dst[body_id] + src[body_id]


class ConveyorForceDriver:
    """Convert Newton contact forces into moving-surface forces for the racetrack belts."""

    def __init__(
        self,
        num_envs: int,
        speed: float = 0.35,
        friction: float = 0.5,
        normal_threshold: float = 0.95,
    ) -> None:
        """Initialize the driver after Newton simulation startup.

        Args:
            num_envs: Number of replicated simulation environments.
            speed: Conveyor surface speed [m/s].
            friction: Coulomb friction coefficient used to limit traction.
            normal_threshold: Minimum upward contact-normal alignment.
        """
        model = NewtonManager.get_model()
        contacts = NewtonManager.get_contacts()
        if model is None or contacts is None:
            raise RuntimeError("The conveyor driver must be created after Newton simulation initialization.")
        if contacts.force is None:
            raise RuntimeError(
                "Newton did not allocate per-contact force reporting. The scene contact sensor must initialize "
                "before the conveyor driver."
            )

        self._model = model
        self._contacts = contacts
        self._device = model.device
        self._dt = NewtonManager.get_solver_dt()
        self._speed = speed
        self._friction = friction
        self._normal_threshold = normal_threshold

        shape_is_belt = [0] * model.shape_count
        shape_belt_center = [wp.vec3()] * model.shape_count
        shape_belt_direction = [0.0] * model.shape_count
        shape_body = model.shape_body.numpy()
        matched_shapes = 0
        for shape_id, label in enumerate(model.shape_label):
            match = _BELT_LABEL.search(label)
            if match is None:
                continue
            if int(shape_body[shape_id]) >= 0:
                raise ValueError(f"Conveyor shape must be static: {label}")

            side = match.group(1)
            shape_is_belt[shape_id] = 1
            center_y = BELT_CENTER_Y if side == "Left" else -BELT_CENTER_Y
            shape_belt_center[shape_id] = wp.vec3(BELT_CENTER_X, center_y, 0.0)
            shape_belt_direction[shape_id] = belt_direction(side)
            matched_shapes += 1

        expected_shapes = 2 * num_envs
        if matched_shapes != expected_shapes:
            raise RuntimeError(f"Expected {expected_shapes} conveyor shapes, but matched {matched_shapes}.")

        self._shape_is_belt = wp.array(shape_is_belt, dtype=wp.int32, device=self._device)
        self._shape_belt_center = wp.array(shape_belt_center, dtype=wp.vec3, device=self._device)
        self._shape_belt_direction = wp.array(shape_belt_direction, dtype=wp.float32, device=self._device)
        self._contact_force = wp.zeros(contacts.rigid_contact_max, dtype=wp.vec3, device=self._device)
        self._belt_contacts = wp.empty(contacts.rigid_contact_max, dtype=BeltContact, device=self._device)
        self._body_contact_count = wp.zeros(model.body_count, dtype=wp.int32, device=self._device)
        self._body_force = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=self._device)

        NewtonManager.register_post_actuator_callback(self.apply)

    def clear(self) -> None:
        """Discard forces computed before an environment reset."""
        self._body_force.zero_()
        self._body_contact_count.zero_()

    def apply(self) -> None:
        """Apply the wrench computed from the preceding physics step."""
        state = NewtonManager.get_state_0()
        wp.launch(
            _add_body_force,
            dim=self._model.body_count,
            inputs=[state.body_f, self._body_force],
            device=self._device,
        )

    def update(self) -> None:
        """Read current contact forces and compute the next conveyor wrench."""
        state = NewtonManager.get_state_0()
        self._body_force.zero_()
        self._body_contact_count.zero_()
        wp.launch(
            _extract_linear_force,
            dim=self._contacts.rigid_contact_max,
            inputs=[self._contacts.force, self._contact_force],
            device=self._device,
        )
        wp.launch(
            _classify_contacts,
            dim=self._contacts.rigid_contact_max,
            inputs=[
                self._contacts.rigid_contact_count,
                self._contacts.rigid_contact_shape0,
                self._contacts.rigid_contact_shape1,
                self._contacts.rigid_contact_normal,
                self._contacts.rigid_contact_point0,
                self._contacts.rigid_contact_point1,
                self._contact_force,
                self._model.shape_body,
                self._shape_is_belt,
                self._shape_belt_center,
                self._shape_belt_direction,
                self._model.shape_transform,
                state.body_q,
                BELT_HALF_STRAIGHT,
                self._speed,
                self._normal_threshold,
            ],
            outputs=[self._belt_contacts, self._body_contact_count],
            device=self._device,
        )
        wp.launch(
            _accumulate_forces,
            dim=self._contacts.rigid_contact_max,
            inputs=[
                self._dt,
                self._friction,
                self._belt_contacts,
                state.body_q,
                state.body_qd,
                self._model.body_com,
                self._model.body_inv_mass,
                self._body_contact_count,
            ],
            outputs=[self._body_force],
            device=self._device,
        )
