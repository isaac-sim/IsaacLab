# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers

"""Private conveyor-force runtime derived from Newton's Apache-2.0 example.

Source: newton-physics/newton, commit d31603ff5e8389d1d6b2a5daef6ea2cb074c26ff,
``newton/examples/basic/example_basic_conveyor_forces.py`` (runtime declarations only).
Force and contact-load kernels are unchanged. Lab adds borrowed per-world commands,
startup ramping, tracked-body filtering and selected resets; scene/demo code and the
unused solver selector are omitted.
"""

from __future__ import annotations

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverBase

VELOCITY_FIELD_TYPE_CONSTANT = 0
VELOCITY_FIELD_TYPE_PIVOT = 1


class ConveyorForceModel:
    """Force-based conveyor driver for a Newton :class:`~newton.Model`.

    Register belts (a static shape + a velocity field) with :meth:`add_constant_belt` or
    :meth:`add_pivot_belt`, call :meth:`finalize`, then drive it each substep::

        conveyor.apply(state_0)  # add the conveyor wrench from the last step
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt, observables=observables)
        conveyor.update(contacts, observables, state_1, dt)  # read forces, recompute wrench
    """

    def __init__(self, model: newton.Model, speed_scale: wp.array, startup_duration: float) -> None:
        """Bind a model and borrowed per-world speed commands; duration is in seconds."""
        self.model = model
        self.device = model.device
        self._speed_scale = speed_scale
        self._startup_duration = startup_duration
        self._num_envs = speed_scale.shape[0]
        self._field_type: list[int] = []
        self._const_vel: list[wp.vec3] = []
        self._pivot_point: list[wp.vec3] = []
        self._pivot_angvel: list[wp.vec3] = []
        self._surface_normal: list[wp.vec3] = []
        self._threshold: list[float] = []
        self._friction: list[float] = []
        self._shape_conveyor = [-1] * model.shape_count
        self._finalized = False

    def add_constant_belt(
        self,
        shape_index: int,
        velocity: wp.vec3,
        surface_normal: wp.vec3 | None = None,
        friction: float = 0.5,
        threshold: float = 0.9,
    ) -> int:
        """Register a belt with a constant world-space surface velocity.

        Args:
            shape_index: Index of the static belt shape.
            velocity: Target surface velocity [m/s].
            surface_normal: Belt-facing unit normal in world space. If ``None``, uses +Z.
            friction: Coulomb friction coefficient used by the force model.
            threshold: Minimum dot product between the contact and belt normals.

        Returns:
            Index of the registered belt.
        """
        if surface_normal is None:
            surface_normal = wp.vec3(0.0, 0.0, 1.0)
        return self._add_belt(
            shape_index,
            VELOCITY_FIELD_TYPE_CONSTANT,
            velocity,
            wp.vec3(),
            wp.vec3(),
            surface_normal,
            friction,
            threshold,
        )

    def add_pivot_belt(
        self,
        shape_index: int,
        pivot_point: wp.vec3,
        angular_velocity: wp.vec3,
        surface_normal: wp.vec3 | None = None,
        friction: float = 0.5,
        threshold: float = 0.9,
    ) -> int:
        """Register a belt with a pivoting world-space surface velocity.

        Args:
            shape_index: Index of the static belt shape.
            pivot_point: World-space center of rotation [m].
            angular_velocity: World-space angular velocity [rad/s].
            surface_normal: Belt-facing unit normal in world space. If ``None``, uses +Z.
            friction: Coulomb friction coefficient used by the force model.
            threshold: Minimum dot product between the contact and belt normals.

        Returns:
            Index of the registered belt.
        """
        if surface_normal is None:
            surface_normal = wp.vec3(0.0, 0.0, 1.0)
        return self._add_belt(
            shape_index,
            VELOCITY_FIELD_TYPE_PIVOT,
            wp.vec3(),
            pivot_point,
            angular_velocity,
            surface_normal,
            friction,
            threshold,
        )

    def finalize(
        self,
        contacts: newton.Contacts,
        observables: SolverBase.Observables,
        surface_worlds: list[int],
        body_is_tracked: np.ndarray,
    ) -> None:
        """Allocate device buffers. Call once after registering all belts.

        Args:
            contacts: The :class:`~newton.Contacts` the step loop will populate; used to size the
                per-contact force buffer.
            observables: Contact-force output requested from the solver.
            surface_worlds: World ID for each registered belt, in registration order.
            body_is_tracked: Int32 model-body mask selecting transported bodies.
        """
        if self._finalized:
            raise RuntimeError("ConveyorForceModel is already finalized.")
        if not self._field_type:
            raise RuntimeError("Register at least one belt before finalize().")
        if contacts.rigid_contact_max <= 0:
            raise ValueError("Contacts must have nonzero rigid-contact capacity.")
        if observables is None or observables.contact_f is None:
            raise ValueError("Request SolverBase.ObservableKind.CONTACT_F before finalizing the conveyor force model.")

        d = self.device
        self.conv_field_type = wp.array(self._field_type, dtype=wp.int32, device=d)
        self.conv_const_vel = wp.array(self._const_vel, dtype=wp.vec3, device=d)
        self.conv_pivot_point = wp.array(self._pivot_point, dtype=wp.vec3, device=d)
        self.conv_pivot_angvel = wp.array(self._pivot_angvel, dtype=wp.vec3, device=d)
        self.conv_surface_normal = wp.array(self._surface_normal, dtype=wp.vec3, device=d)
        self.conv_threshold = wp.array(self._threshold, dtype=wp.float32, device=d)
        self.conv_friction = wp.array(self._friction, dtype=wp.float32, device=d)
        self.shape_conveyor = wp.array(self._shape_conveyor, dtype=wp.int32, device=d)
        self.global_velocity_scale = wp.zeros(1, dtype=wp.float32, device=d)

        self.body_contact_count = wp.zeros(self.model.body_count, dtype=wp.int32, device=d)
        self.conveyor_body_f = wp.zeros(self.model.body_count, dtype=wp.spatial_vector, device=d)
        self.belt_contacts = wp.empty(contacts.rigid_contact_max, dtype=BeltContact, device=d)
        self.contact_force_vec = wp.zeros(contacts.rigid_contact_max, dtype=wp.vec3, device=d)
        # Belts with identical velocity fields (e.g. the segments of one turn) share a group.
        groups: dict[tuple, int] = {}
        conv_velocity_group = []
        for field_type, const_vel, pivot_point, pivot_angvel in zip(
            self._field_type, self._const_vel, self._pivot_point, self._pivot_angvel, strict=True
        ):
            key = (field_type, *(round(float(x), 6) for x in (*const_vel, *pivot_point, *pivot_angvel)))
            conv_velocity_group.append(groups.setdefault(key, len(groups)))
        self.conv_velocity_group = wp.array(conv_velocity_group, dtype=wp.int32, device=d)
        self.body_group_min = wp.zeros(self.model.body_count, dtype=wp.int32, device=d)
        self.body_group_max = wp.zeros(self.model.body_count, dtype=wp.int32, device=d)
        self.body_load = wp.zeros(self.model.body_count, dtype=wp.float32, device=d)
        self.body_point_sum = wp.zeros(self.model.body_count, dtype=wp.vec3, device=d)
        self.body_point_cov = wp.zeros(self.model.body_count, dtype=wp.mat33, device=d)
        self.body_load_moment = wp.zeros(self.model.body_count, dtype=wp.vec3, device=d)
        self._surface_worlds = wp.array(surface_worlds, dtype=wp.int32, device=d)
        self._body_is_tracked = wp.array(body_is_tracked, dtype=wp.int32, device=d)
        self._default_linear_velocity = wp.clone(self.conv_const_vel)
        self._default_angular_velocity = wp.clone(self.conv_pivot_angvel)
        self._elapsed_time = wp.zeros(1, dtype=wp.float32, device=d)
        self._full_reset = wp.ones(1, dtype=wp.int32, device=d)
        self._finalized = True

    def reset(self, mask: wp.array | None = None) -> None:
        """Clear selected-world wrenches; a full reset restarts the shared startup ramp."""
        if mask is None:
            self.conveyor_body_f.zero_()
            self._elapsed_time.zero_()
            self.global_velocity_scale.zero_()
            return
        self._full_reset.fill_(1)
        wp.launch(_check_full_reset, self._num_envs, inputs=[mask], outputs=[self._full_reset], device=self.device)
        wp.launch(
            _clear_selected_body_forces,
            dim=self.model.body_count,
            inputs=[self.model.body_world, mask, self._full_reset],
            outputs=[self.conveyor_body_f, self._elapsed_time, self.global_velocity_scale],
            device=self.device,
        )

    def apply(self, state: newton.State) -> None:
        """Add the conveyor wrench computed by the previous :meth:`update` into ``state.body_f``."""
        wp.launch(
            add_spatial,
            dim=self.model.body_count,
            inputs=[state.body_f, self.conveyor_body_f],
            device=self.device,
        )

    def update(
        self, contacts: newton.Contacts, observables: SolverBase.Observables, state: newton.State, dt: float
    ) -> None:
        """Read solved contacts and compute selected-body wrenches for the following substep."""
        wp.launch(
            _set_surface_speed,
            dim=self.conv_const_vel.shape[0],
            inputs=[
                self._surface_worlds,
                self._speed_scale,
                self._default_linear_velocity,
                self._default_angular_velocity,
            ],
            outputs=[self.conv_const_vel, self.conv_pivot_angvel],
            device=self.device,
        )
        wp.launch(
            _advance_startup_scale,
            dim=1,
            inputs=[dt, self._startup_duration],
            outputs=[self._elapsed_time, self.global_velocity_scale],
            device=self.device,
        )
        wp.launch(
            extract_linear,
            dim=contacts.rigid_contact_max,
            inputs=[observables.contact_f, self.contact_force_vec],
            device=self.device,
        )
        self.conveyor_body_f.zero_()
        self.body_contact_count.zero_()
        wp.launch(
            classify_belt_contacts,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                self.contact_force_vec,
                self.model.shape_body,
                self.shape_conveyor,
                state.body_q,
                self.conv_surface_normal,
                self.conv_threshold,
            ],
            outputs=[self.belt_contacts, self.body_contact_count],
            device=self.device,
        )
        self.body_group_min.fill_(np.iinfo(np.int32).max)
        self.body_group_max.fill_(-1)
        self.body_load.zero_()
        self.body_point_sum.zero_()
        self.body_point_cov.zero_()
        self.body_load_moment.zero_()
        wp.launch(
            accumulate_belt_load,
            dim=contacts.rigid_contact_max,
            inputs=[self.belt_contacts, self.conv_velocity_group],
            outputs=[
                self.body_load,
                self.body_point_sum,
                self.body_group_min,
                self.body_group_max,
            ],
            device=self.device,
        )
        wp.launch(
            accumulate_belt_load_moment,
            dim=contacts.rigid_contact_max,
            inputs=[self.belt_contacts, self.body_contact_count, self.body_point_sum],
            outputs=[self.body_point_cov, self.body_load_moment],
            device=self.device,
        )
        wp.launch(
            redistribute_belt_load,
            dim=contacts.rigid_contact_max,
            inputs=[
                self.body_contact_count,
                self.body_load,
                self.body_point_sum,
                self.body_point_cov,
                self.body_load_moment,
                self.body_group_min,
                self.body_group_max,
            ],
            outputs=[self.belt_contacts],
            device=self.device,
        )
        wp.launch(
            accumulate_conveyor_forces,
            dim=contacts.rigid_contact_max,
            inputs=[
                dt,
                self.belt_contacts,
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
                self.body_contact_count,
                self.conv_field_type,
                self.conv_const_vel,
                self.conv_pivot_point,
                self.conv_pivot_angvel,
                self.conv_surface_normal,
                self.conv_threshold,
                self.conv_friction,
                self.global_velocity_scale,
            ],
            outputs=[self.conveyor_body_f],
            device=self.device,
        )
        wp.launch(
            _filter_body_forces,
            dim=self.model.body_count,
            inputs=[self._body_is_tracked],
            outputs=[self.conveyor_body_f],
            device=self.device,
        )

    def _add_belt(
        self, shape_index, field_type, const_vel, pivot_point, pivot_angvel, surface_normal, friction, threshold
    ):
        """Validate and store one belt registration."""
        if self._finalized:
            raise RuntimeError("Cannot register belts after finalize().")
        if shape_index < 0 or shape_index >= self.model.shape_count:
            raise IndexError(f"Belt shape index {shape_index} is out of range.")
        if self._shape_conveyor[shape_index] >= 0:
            raise ValueError(f"Shape {shape_index} is already registered as a belt.")
        if int(self.model.shape_body.numpy()[shape_index]) >= 0:
            raise ValueError("Conveyor belts must be static shapes.")
        if wp.length(surface_normal) == 0.0:
            raise ValueError("surface_normal must be nonzero.")
        if friction < 0.0:
            raise ValueError("friction must be nonnegative.")
        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("threshold must be in [-1, 1].")

        conv = len(self._field_type)
        self._shape_conveyor[shape_index] = conv
        self._field_type.append(field_type)
        self._const_vel.append(const_vel)
        self._pivot_point.append(pivot_point)
        self._pivot_angvel.append(pivot_angvel)
        self._surface_normal.append(wp.normalize(surface_normal))
        self._friction.append(friction)
        self._threshold.append(threshold)
        return conv


@wp.struct
class Vec3Pair:
    """Two orthonormal vectors spanning the plane tangent to a contact normal."""

    v0: wp.vec3
    v1: wp.vec3


@wp.struct
class BeltContact:
    """One belt-to-body contact, reduced to what the force model needs."""

    valid: wp.int32
    body: wp.int32
    conv: wp.int32
    point: wp.vec3
    normal: wp.vec3  # points from belt toward the body
    normal_force: wp.float32


@wp.func
def compute_basis_vectors(direction: wp.vec3) -> Vec3Pair:
    """Two unit vectors orthogonal to normalized ``direction`` and to each other."""
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
    """Coulomb-clamped tangential impulse driving the contact-point velocity toward ``target_vel``."""
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
    """Friction-limited (force, torque about COM) for one contact point, world frame."""
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


@wp.func
def identify_belt_contact(
    i: int,
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    contact_force_vec: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_conveyor: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    conv_surface_normal: wp.array[wp.vec3],
    conv_threshold: wp.array[wp.float32],
) -> BeltContact:
    """Decide whether contact ``i`` is a belt/body contact and extract its inputs for the force model."""
    out = BeltContact()
    out.valid = 0
    if i >= rigid_contact_count[0]:
        return out

    shape0 = rigid_contact_shape0[i]
    shape1 = rigid_contact_shape1[i]
    if shape0 < 0 or shape1 < 0:
        return out

    conv0 = shape_conveyor[shape0]
    conv1 = shape_conveyor[shape1]

    # Exactly one side must be a belt; the other must be a dynamic body.
    normal = rigid_contact_normal[i]  # shape0 -> shape1
    if conv0 >= 0 and conv1 < 0:
        conv = conv0
        body = shape_body[shape1]
        body_point_local = rigid_contact_point1[i] + rigid_contact_offset1[i]
        normal_toward_body = normal
    elif conv1 >= 0 and conv0 < 0:
        conv = conv1
        body = shape_body[shape0]
        body_point_local = rigid_contact_point0[i] + rigid_contact_offset0[i]
        normal_toward_body = -normal
    else:
        return out

    if body < 0:
        return out

    # Reject contacts whose normal is not aligned with the belt surface normal.
    acceptance = wp.dot(normal_toward_body, conv_surface_normal[conv])
    if acceptance < conv_threshold[conv]:
        return out

    # Normal-force magnitude from the reported per-contact force vector.
    nforce = wp.abs(wp.dot(contact_force_vec[i], normal))
    if nforce <= 0.0:
        return out

    out.valid = 1
    out.body = body
    out.conv = conv
    out.point = wp.transform_point(body_q[body], body_point_local)
    out.normal = normal_toward_body
    out.normal_force = nforce
    return out


@wp.kernel
def classify_belt_contacts(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    contact_force_vec: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_conveyor: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    conv_surface_normal: wp.array[wp.vec3],
    conv_threshold: wp.array[wp.float32],
    # output
    belt_contacts: wp.array[BeltContact],
    body_contact_count: wp.array[wp.int32],
):
    """Classify belt contacts and count the accepted contacts per dynamic body."""
    i = wp.tid()
    c = identify_belt_contact(
        i,
        rigid_contact_count,
        rigid_contact_shape0,
        rigid_contact_shape1,
        rigid_contact_normal,
        rigid_contact_point0,
        rigid_contact_point1,
        rigid_contact_offset0,
        rigid_contact_offset1,
        contact_force_vec,
        shape_body,
        shape_conveyor,
        body_q,
        conv_surface_normal,
        conv_threshold,
    )
    belt_contacts[i] = c
    if c.valid == 1:
        wp.atomic_add(body_contact_count, c.body, 1)


@wp.kernel
def accumulate_belt_load(
    belt_contacts: wp.array[BeltContact],
    conv_velocity_group: wp.array[wp.int32],
    # outputs
    body_load: wp.array[wp.float32],
    body_point_sum: wp.array[wp.vec3],
    body_group_min: wp.array[wp.int32],
    body_group_max: wp.array[wp.int32],
):
    """Accumulate each body's total belt normal load, contact centroid sum, and velocity-group range."""
    i = wp.tid()
    c = belt_contacts[i]
    if c.valid == 0:
        return
    wp.atomic_min(body_group_min, c.body, conv_velocity_group[c.conv])
    wp.atomic_max(body_group_max, c.body, conv_velocity_group[c.conv])
    wp.atomic_add(body_load, c.body, c.normal_force)
    wp.atomic_add(body_point_sum, c.body, c.point)


@wp.kernel
def accumulate_belt_load_moment(
    belt_contacts: wp.array[BeltContact],
    body_contact_count: wp.array[wp.int32],
    body_point_sum: wp.array[wp.vec3],
    # outputs
    body_point_cov: wp.array[wp.mat33],
    body_load_moment: wp.array[wp.vec3],
):
    """Accumulate each body's contact-point covariance and first moment of the normal load about the centroid."""
    i = wp.tid()
    c = belt_contacts[i]
    if c.valid == 0:
        return
    d = c.point - body_point_sum[c.body] / float(body_contact_count[c.body])
    wp.atomic_add(body_point_cov, c.body, wp.outer(d, d))
    wp.atomic_add(body_load_moment, c.body, d * c.normal_force)


@wp.kernel
def redistribute_belt_load(
    body_contact_count: wp.array[wp.int32],
    body_load: wp.array[wp.float32],
    body_point_sum: wp.array[wp.vec3],
    body_point_cov: wp.array[wp.mat33],
    body_load_moment: wp.array[wp.vec3],
    body_group_min: wp.array[wp.int32],
    body_group_max: wp.array[wp.int32],
    # in/out
    belt_contacts: wp.array[BeltContact],
):
    """Replace per-contact normal forces with the linear pressure field that has the same
    total force and first moment, for bodies that span belts with different velocity fields.

    A body resting on several coplanar contacts is statically indeterminate: solvers may
    return any split with the right total and moment. Since the drive is bounded per contact
    by friction times normal force, an uneven split would bias the drive between belts.
    """
    i = wp.tid()
    c = belt_contacts[i]
    if c.valid == 0:
        return
    if body_group_min[c.body] == body_group_max[c.body]:
        return
    n = float(body_contact_count[c.body])
    cov = body_point_cov[c.body]
    eps = 1.0e-4 * wp.trace(cov) + 1.0e-9
    g = wp.inverse(cov + wp.identity(n=3, dtype=wp.float32) * eps) * body_load_moment[c.body]
    d = c.point - body_point_sum[c.body] / n
    c.normal_force = wp.max(body_load[c.body] / n + wp.dot(g, d), 0.0)
    belt_contacts[i] = c


@wp.kernel
def accumulate_conveyor_forces(
    dt: wp.float32,
    belt_contacts: wp.array[BeltContact],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
    body_contact_count: wp.array[wp.int32],
    conv_field_type: wp.array[wp.int32],
    conv_const_vel: wp.array[wp.vec3],
    conv_pivot_point: wp.array[wp.vec3],
    conv_pivot_angvel: wp.array[wp.vec3],
    conv_surface_normal: wp.array[wp.vec3],
    conv_threshold: wp.array[wp.float32],
    conv_friction: wp.array[wp.float32],
    global_velocity_scale: wp.array[wp.float32],
    # output
    conveyor_body_f: wp.array[wp.spatial_vector],
):
    """Accumulate Coulomb-limited belt-contact wrenches by body."""
    i = wp.tid()
    c = belt_contacts[i]
    if c.valid == 0:
        return

    count = body_contact_count[c.body]
    if count <= 0:
        return
    mass_splitting_scale = 1.0 / float(count)

    # Target surface velocity from the belt's velocity field, evaluated at the contact point.
    conv = c.conv
    if conv_field_type[conv] == VELOCITY_FIELD_TYPE_CONSTANT:
        target_vel = conv_const_vel[conv]
    else:
        delta = c.point - conv_pivot_point[conv]
        target_vel = wp.cross(conv_pivot_angvel[conv], delta)
    target_vel = target_vel * global_velocity_scale[0]

    # World-space body quantities.
    q = body_q[c.body]
    com_world = wp.transform_point(q, body_com[c.body])
    r = wp.quat_to_matrix(wp.transform_get_rotation(q))
    inv_inertia_world = r * body_inv_inertia[c.body] * wp.transpose(r)
    qd = body_qd[c.body]
    lin_vel = wp.spatial_top(qd)
    ang_vel = wp.spatial_bottom(qd)

    ft = compute_point_force(
        dt,
        1.0 / dt,
        com_world,
        body_inv_mass[c.body],
        inv_inertia_world,
        lin_vel,
        ang_vel,
        c.point,
        c.normal,
        c.normal_force,
        mass_splitting_scale,
        target_vel,
        conv_friction[conv],
    )

    wp.atomic_add(conveyor_body_f, c.body, ft)


@wp.kernel
def add_spatial(dst: wp.array[wp.spatial_vector], src: wp.array[wp.spatial_vector]):
    """Add one spatial-vector array into another."""
    i = wp.tid()
    dst[i] = dst[i] + src[i]


@wp.kernel
def extract_linear(spatial_force: wp.array[wp.spatial_vector], out_vec: wp.array[wp.vec3]):
    """Copy the linear part of a per-contact spatial wrench into a plain force vector."""
    i = wp.tid()
    out_vec[i] = wp.spatial_top(spatial_force[i])


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
