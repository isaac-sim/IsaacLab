# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Warp GPU kernels for the Direct Position-Based Solver for Stiff Rods.

This module implements the constraint projection kernels based on:
Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods"
Computer Graphics Forum, Vol. 37, No. 8

Key constraints implemented:
1. Zero-stretch constraint (inextensibility)
2. Bending constraint (Cosserat model)
3. Twisting constraint (Cosserat model)

The kernels are designed for the XPBD (Extended Position-Based Dynamics)
framework with Newton iterations for improved convergence.
"""

import warp as wp

# ============================================================================
# Quaternion Helper Functions
# ============================================================================


@wp.func
def quat_mul(q1: wp.quatf, q2: wp.quatf) -> wp.quatf:
    """Multiply two quaternions."""
    return q1 * q2


@wp.func
def quat_conjugate(q: wp.quatf) -> wp.quatf:
    """Compute quaternion conjugate."""
    return wp.quat(-q[0], -q[1], -q[2], q[3])


@wp.func
def quat_rotate_vec(q: wp.quatf, v: wp.vec3f) -> wp.vec3f:
    """Rotate vector v by quaternion q."""
    return wp.quat_rotate(q, v)


@wp.func
def quat_rotate_inv_vec(q: wp.quatf, v: wp.vec3f) -> wp.vec3f:
    """Rotate vector v by inverse of quaternion q."""
    return wp.quat_rotate_inv(q, v)


@wp.func
def quat_to_omega(q: wp.quatf, q_prev: wp.quatf, dt: float) -> wp.vec3f:
    """Compute angular velocity from quaternion change.

    ω = 2 * Im(q * q_prev^(-1)) / dt
    """
    dq = quat_mul(q, quat_conjugate(q_prev))
    # Angular velocity = 2 * vector part of dq / dt
    return wp.vec3f(2.0 * dq[0] / dt, 2.0 * dq[1] / dt, 2.0 * dq[2] / dt)


@wp.func
def omega_to_quat_delta(omega: wp.vec3f, dt: float) -> wp.quatf:
    """Convert angular velocity to quaternion increment.

    Δq = [sin(|ω|*dt/2) * ω/|ω|, cos(|ω|*dt/2)]
    """
    angle = wp.length(omega) * dt
    if angle < 1e-8:
        # Small angle approximation
        return wp.quatf(omega[0] * dt * 0.5, omega[1] * dt * 0.5, omega[2] * dt * 0.5, 1.0)

    half_angle = angle * 0.5
    s = wp.sin(half_angle) / (angle / dt)
    return wp.quatf(s * omega[0], s * omega[1], s * omega[2], wp.cos(half_angle))


# ============================================================================
# Cosserat Rod Model Functions
# ============================================================================


@wp.func
def compute_darboux_vector(q1: wp.quatf, q2: wp.quatf, segment_length: float) -> wp.vec3f:
    """Compute the Darboux vector (curvature + twist) between two segments.

    The Darboux vector Ω represents the angular rate of change of the
    material frame along the rod:
        Ω = 2 * Im(q1^(-1) * q2) / L

    where L is the average segment length.

    Args:
        q1: Quaternion of first segment.
        q2: Quaternion of second segment.
        segment_length: Average length of segments.

    Returns:
        Darboux vector in the local frame of q1.
    """
    # Relative rotation: q_rel = q1^(-1) * q2
    q_rel = quat_mul(quat_conjugate(q1), q2)

    # Darboux vector = 2 * vector_part(q_rel) / L
    inv_L = 1.0 / segment_length
    return wp.vec3f(2.0 * q_rel[0] * inv_L, 2.0 * q_rel[1] * inv_L, 2.0 * q_rel[2] * inv_L)


@wp.func
def compute_bending_twist_potential(
    darboux: wp.vec3f, rest_darboux: wp.vec3f, stiffness: wp.vec3f, length: float
) -> float:
    """Compute the Cosserat bending/twisting potential energy.

    E = 0.5 * L * Σ k_i * (Ω_i - Ω_i^0)²

    Args:
        darboux: Current Darboux vector.
        rest_darboux: Rest Darboux vector.
        stiffness: Stiffness coefficients (k_bend_x, k_bend_y, k_twist).
        length: Segment length.

    Returns:
        Potential energy.
    """
    diff = darboux - rest_darboux
    return 0.5 * length * (
        stiffness[0] * diff[0] * diff[0]
        + stiffness[1] * diff[1] * diff[1]
        + stiffness[2] * diff[2] * diff[2]
    )


# ============================================================================
# XPBD Integration Kernels
# ============================================================================


@wp.kernel
def predict_positions_kernel(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    prev_positions: wp.array(dtype=wp.vec3f),
    masses: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.bool),
    gravity: wp.vec3f,
    dt: float,
    damping: float,
):
    """Predict positions using explicit Euler integration.

    x̃ = x + dt * v + dt² * f_ext / m

    This is the prediction step of XPBD before constraint projection.
    """
    idx = wp.tid()

    if fixed[idx]:
        return

    # Store previous position
    prev_positions[idx] = positions[idx]

    # External acceleration (gravity)
    m = masses[idx]
    if m > 1e-10:
        # Apply damping to velocity
        v = velocities[idx] * (1.0 - damping)

        # Predict new position
        positions[idx] = positions[idx] + dt * v + dt * dt * gravity
    # else: infinite mass, don't move


@wp.kernel
def predict_orientations_kernel(
    orientations: wp.array(dtype=wp.quatf),
    angular_velocities: wp.array(dtype=wp.vec3f),
    prev_orientations: wp.array(dtype=wp.quatf),
    fixed: wp.array(dtype=wp.bool),
    dt: float,
    damping: float,
):
    """Predict orientations using explicit Euler integration.

    q̃ = q + dt/2 * [ω, 0] * q

    This is the prediction step for rotational motion.
    """
    idx = wp.tid()

    if fixed[idx]:
        return

    # Store previous orientation
    prev_orientations[idx] = orientations[idx]

    # Apply damping
    omega = angular_velocities[idx] * (1.0 - damping)

    # Predict new orientation using quaternion integration
    dq = omega_to_quat_delta(omega, dt)
    q_new = quat_mul(dq, orientations[idx])

    # Normalize quaternion
    q_len = wp.sqrt(q_new[0] * q_new[0] + q_new[1] * q_new[1] + q_new[2] * q_new[2] + q_new[3] * q_new[3])
    if q_len > 1e-8:
        orientations[idx] = wp.quatf(
            q_new[0] / q_len, q_new[1] / q_len, q_new[2] / q_len, q_new[3] / q_len
        )


@wp.kernel
def update_velocities_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    prev_positions: wp.array(dtype=wp.vec3f),
    prev_orientations: wp.array(dtype=wp.quatf),
    velocities: wp.array(dtype=wp.vec3f),
    angular_velocities: wp.array(dtype=wp.vec3f),
    fixed: wp.array(dtype=wp.bool),
    dt: float,
):
    """Update velocities from position/orientation changes.

    v = (x - x_prev) / dt
    ω = 2 * Im(q * q_prev^(-1)) / dt

    This is the velocity update step after constraint projection.
    """
    idx = wp.tid()

    if fixed[idx]:
        velocities[idx] = wp.vec3f(0.0, 0.0, 0.0)
        angular_velocities[idx] = wp.vec3f(0.0, 0.0, 0.0)
        return

    inv_dt = 1.0 / dt

    # Linear velocity
    velocities[idx] = (positions[idx] - prev_positions[idx]) * inv_dt

    # Angular velocity from quaternion difference
    angular_velocities[idx] = quat_to_omega(orientations[idx], prev_orientations[idx], dt)


# ============================================================================
# Zero-Stretch Constraint Kernel
# ============================================================================


@wp.kernel
def solve_stretch_constraints_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    inv_masses: wp.array(dtype=wp.float32),
    segment_lengths: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),
    lambda_stretch: wp.array(dtype=wp.float32),
    parent_indices: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.bool),
    num_segments: int,
    dt: float,
):
    """Solve zero-stretch (distance) constraints between connected segments.

    Constraint: C = |x_j - x_i| - L = 0

    For XPBD, we solve: (α + ∂C^T M^(-1) ∂C) Δλ = -C - α*λ

    The constraint enforces that segment endpoints match up (inextensibility).
    """
    idx = wp.tid()

    # Skip first segment (has no parent constraint)
    segment_idx = idx + 1  # Constraint idx maps to segment idx+1
    if segment_idx >= num_segments:
        return

    parent_idx = parent_indices[segment_idx]
    if parent_idx < 0:
        return  # No parent, skip

    # Get inverse masses
    w1 = inv_masses[parent_idx]
    w2 = inv_masses[segment_idx]

    # Skip if both are fixed
    if w1 < 1e-10 and w2 < 1e-10:
        return

    # Get current positions
    x1 = positions[parent_idx]
    x2 = positions[segment_idx]

    # Get segment orientations
    q1 = orientations[parent_idx]
    q2 = orientations[segment_idx]

    # Compute attachment points (end of segment 1, start of segment 2)
    L1 = segment_lengths[parent_idx]
    L2 = segment_lengths[segment_idx]

    # Local attachment point at end of parent segment
    local_end = wp.vec3f(L1 * 0.5, 0.0, 0.0)
    # Local attachment point at start of child segment
    local_start = wp.vec3f(-L2 * 0.5, 0.0, 0.0)

    # World space attachment points
    p1 = x1 + quat_rotate_vec(q1, local_end)
    p2 = x2 + quat_rotate_vec(q2, local_start)

    # Constraint vector: p2 - p1 should be zero
    diff = p2 - p1
    dist = wp.length(diff)

    # Skip if already satisfied
    if dist < 1e-8:
        return

    # Constraint direction (normalized)
    n = diff / dist

    # XPBD: compute Δλ
    # For distance constraint: ∂C/∂x1 = -n, ∂C/∂x2 = n
    # Denominator: w1 + w2 + α/dt²
    alpha = compliance[idx]
    w_sum = w1 + w2 + alpha / (dt * dt)

    if w_sum < 1e-10:
        return

    # Constraint value (should be 0 for inextensibility)
    C = dist

    # Delta lambda
    delta_lambda = (-C - alpha * lambda_stretch[idx]) / w_sum

    # Update lambda
    lambda_stretch[idx] = lambda_stretch[idx] + delta_lambda

    # Position corrections
    p_corr = delta_lambda * n

    if not fixed[parent_idx]:
        positions[parent_idx] = positions[parent_idx] - w1 * p_corr
    if not fixed[segment_idx]:
        positions[segment_idx] = positions[segment_idx] + w2 * p_corr


# ============================================================================
# Bending/Twisting Constraint Kernel
# ============================================================================


@wp.kernel
def solve_bend_twist_constraints_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    inv_masses: wp.array(dtype=wp.float32),
    inv_inertias_diag: wp.array(dtype=wp.vec3f),
    segment_lengths: wp.array(dtype=wp.float32),
    rest_darboux: wp.array(dtype=wp.vec3f),
    compliance: wp.array(dtype=wp.vec3f),
    lambda_bend_twist: wp.array(dtype=wp.vec3f),
    parent_indices: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.bool),
    num_segments: int,
    dt: float,
):
    """Solve bending and twisting constraints based on Cosserat model.

    Constraint: C = Ω - Ω^0 = 0

    where Ω is the Darboux vector (curvature + twist) and Ω^0 is the
    rest configuration.

    The Darboux vector encodes:
    - Ω_x, Ω_y: Bending curvatures
    - Ω_z: Twist

    For XPBD, we solve: (α + J^T W J) Δλ = -C - α*λ
    where W is the generalized inverse mass matrix.
    """
    idx = wp.tid()

    # Skip first segment (has no parent constraint)
    segment_idx = idx + 1
    if segment_idx >= num_segments:
        return

    parent_idx = parent_indices[segment_idx]
    if parent_idx < 0:
        return

    # Check if both are fixed
    if fixed[parent_idx] and fixed[segment_idx]:
        return

    # Get orientations
    q1 = orientations[parent_idx]
    q2 = orientations[segment_idx]

    # Compute average segment length
    L = 0.5 * (segment_lengths[parent_idx] + segment_lengths[segment_idx])

    # Compute current Darboux vector
    darboux = compute_darboux_vector(q1, q2, L)

    # Constraint: C = darboux - rest_darboux
    C = darboux - rest_darboux[idx]

    # Get compliance (α)
    alpha = compliance[idx]

    # Get inverse inertias (diagonal approximation)
    # For rotation constraints, we use the angular mass (inertia)
    I1_inv = inv_inertias_diag[parent_idx]
    I2_inv = inv_inertias_diag[segment_idx]

    # Effective inverse mass for each rotation axis
    # The Jacobian for the Darboux constraint is approximately 2/L
    J_scale = 2.0 / L
    w1 = I1_inv * J_scale * J_scale
    w2 = I2_inv * J_scale * J_scale

    # Total effective mass (per axis)
    w_total = w1 + w2

    # Previous lambda
    lambda_prev = lambda_bend_twist[idx]

    # Compute delta lambda for each axis
    dt2_inv = 1.0 / (dt * dt)
    delta_lambda = wp.vec3f(0.0, 0.0, 0.0)

    for i in range(3):
        if wp.abs(w_total[i]) > 1e-10:
            denom = w_total[i] + alpha[i] * dt2_inv
            if wp.abs(denom) > 1e-10:
                delta_lambda_i = (-C[i] - alpha[i] * lambda_prev[i] * dt2_inv) / denom
                if i == 0:
                    delta_lambda = wp.vec3f(delta_lambda_i, delta_lambda[1], delta_lambda[2])
                elif i == 1:
                    delta_lambda = wp.vec3f(delta_lambda[0], delta_lambda_i, delta_lambda[2])
                else:
                    delta_lambda = wp.vec3f(delta_lambda[0], delta_lambda[1], delta_lambda_i)

    # Update lambda
    lambda_bend_twist[idx] = lambda_prev + delta_lambda

    # Compute orientation corrections
    # Δθ = W * J^T * Δλ
    omega_corr = wp.vec3f(
        J_scale * delta_lambda[0], J_scale * delta_lambda[1], J_scale * delta_lambda[2]
    )

    # Apply corrections to parent (rotate by -correction in local frame)
    if not fixed[parent_idx]:
        corr1 = wp.vec3f(
            I1_inv[0] * omega_corr[0], I1_inv[1] * omega_corr[1], I1_inv[2] * omega_corr[2]
        )
        # Convert local correction to world frame and apply
        corr1_world = quat_rotate_vec(q1, corr1)
        dq1 = omega_to_quat_delta(-corr1_world, 1.0)
        q1_new = quat_mul(dq1, q1)
        # Normalize
        q1_len = wp.sqrt(q1_new[0] * q1_new[0] + q1_new[1] * q1_new[1] + q1_new[2] * q1_new[2] + q1_new[3] * q1_new[3])
        if q1_len > 1e-8:
            orientations[parent_idx] = wp.quatf(
                q1_new[0] / q1_len, q1_new[1] / q1_len, q1_new[2] / q1_len, q1_new[3] / q1_len
            )

    # Apply corrections to child (rotate by +correction in local frame)
    if not fixed[segment_idx]:
        corr2 = wp.vec3f(
            I2_inv[0] * omega_corr[0], I2_inv[1] * omega_corr[1], I2_inv[2] * omega_corr[2]
        )
        corr2_world = quat_rotate_vec(q2, corr2)
        dq2 = omega_to_quat_delta(corr2_world, 1.0)
        q2_new = quat_mul(dq2, q2)
        # Normalize
        q2_len = wp.sqrt(q2_new[0] * q2_new[0] + q2_new[1] * q2_new[1] + q2_new[2] * q2_new[2] + q2_new[3] * q2_new[3])
        if q2_len > 1e-8:
            orientations[segment_idx] = wp.quatf(
                q2_new[0] / q2_len, q2_new[1] / q2_len, q2_new[2] / q2_len, q2_new[3] / q2_len
            )


# ============================================================================
# Shear Constraint Kernel (Timoshenko beam formulation)
# ============================================================================


@wp.kernel
def solve_shear_constraints_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    inv_masses: wp.array(dtype=wp.float32),
    segment_lengths: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.vec2f),
    lambda_shear: wp.array(dtype=wp.vec2f),
    parent_indices: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.bool),
    num_segments: int,
    dt: float,
):
    """Solve shear constraints between connected segments.
    
    Shear constraint ensures the tangent vector aligns with the segment
    connection direction, preventing unphysical lateral deformation.
    
    This implements a Timoshenko beam formulation which is more accurate
    for thick rods and catheters.
    
    Constraint: C = (q^(-1) * d - [1,0,0])_yz = 0
    where d is the normalized direction between segment centers.
    """
    idx = wp.tid()
    
    segment_idx = idx + 1
    if segment_idx >= num_segments:
        return
    
    parent_idx = parent_indices[segment_idx]
    if parent_idx < 0:
        return
    
    # Skip if both are fixed
    if fixed[parent_idx] and fixed[segment_idx]:
        return
    
    # Get positions
    x1 = positions[parent_idx]
    x2 = positions[segment_idx]
    
    # Direction between centers
    diff = x2 - x1
    dist = wp.length(diff)
    if dist < 1e-8:
        return
    d = diff / dist
    
    # Get parent orientation
    q1 = orientations[parent_idx]
    
    # Local tangent should point in +x direction
    # Current tangent in local frame
    d_local = quat_rotate_inv_vec(q1, d)
    
    # Shear constraint: y and z components of d_local should be zero
    C_y = d_local[1]
    C_z = d_local[2]
    
    # Get compliance
    alpha = compliance[idx]
    
    # Inverse masses
    w1 = inv_masses[parent_idx]
    w2 = inv_masses[segment_idx]
    w_sum = w1 + w2
    
    if w_sum < 1e-10:
        return
    
    dt2_inv = 1.0 / (dt * dt)
    lambda_prev = lambda_shear[idx]
    
    # Solve for Y shear
    denom_y = w_sum + alpha[0] * dt2_inv
    delta_lambda_y = 0.0
    if wp.abs(denom_y) > 1e-10:
        delta_lambda_y = (-C_y - alpha[0] * lambda_prev[0] * dt2_inv) / denom_y
    
    # Solve for Z shear
    denom_z = w_sum + alpha[1] * dt2_inv
    delta_lambda_z = 0.0
    if wp.abs(denom_z) > 1e-10:
        delta_lambda_z = (-C_z - alpha[1] * lambda_prev[1] * dt2_inv) / denom_z
    
    # Update lambda
    lambda_shear[idx] = wp.vec2f(lambda_prev[0] + delta_lambda_y, lambda_prev[1] + delta_lambda_z)
    
    # Position corrections (perpendicular to tangent)
    # Get local Y and Z axes in world frame
    y_axis = quat_rotate_vec(q1, wp.vec3f(0.0, 1.0, 0.0))
    z_axis = quat_rotate_vec(q1, wp.vec3f(0.0, 0.0, 1.0))
    
    p_corr = delta_lambda_y * y_axis + delta_lambda_z * z_axis
    
    if not fixed[parent_idx]:
        positions[parent_idx] = positions[parent_idx] - w1 * p_corr
    if not fixed[segment_idx]:
        positions[segment_idx] = positions[segment_idx] + w2 * p_corr


# ============================================================================
# Friction Constraint Kernels
# ============================================================================


@wp.kernel
def apply_coulomb_friction_kernel(
    positions: wp.array(dtype=wp.vec3f),
    prev_positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    contact_normals: wp.array(dtype=wp.vec3f),
    contact_depths: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.bool),
    static_friction: float,
    dynamic_friction: float,
    stiction_velocity: float,
    dt: float,
):
    """Apply Coulomb friction at contact points.
    
    Implements static/dynamic friction model:
    - If sliding velocity < stiction_velocity: static friction (μ_s)
    - Otherwise: dynamic friction (μ_d)
    
    Friction force opposes tangential motion and is proportional to
    normal force (penetration depth).
    """
    idx = wp.tid()
    
    if fixed[idx]:
        return
    
    # Check if in contact
    depth = contact_depths[idx]
    if depth <= 0.0:
        return
    
    normal = contact_normals[idx]
    
    # Compute tangential velocity
    v = velocities[idx]
    v_normal = wp.dot(v, normal) * normal
    v_tangent = v - v_normal
    v_tangent_mag = wp.length(v_tangent)
    
    if v_tangent_mag < 1e-8:
        return
    
    # Choose friction coefficient based on velocity
    mu = static_friction
    if v_tangent_mag > stiction_velocity:
        mu = dynamic_friction
    
    # Friction impulse magnitude (proportional to normal force ~ depth)
    friction_impulse_mag = mu * depth * 1000.0  # Scale factor for contact stiffness
    
    # Clamp to not reverse velocity
    max_impulse = v_tangent_mag * dt
    if friction_impulse_mag > max_impulse:
        friction_impulse_mag = max_impulse
    
    # Apply friction in opposite direction of tangential velocity
    tangent_dir = v_tangent / v_tangent_mag
    friction_corr = friction_impulse_mag * tangent_dir * dt
    
    positions[idx] = positions[idx] - friction_corr


@wp.kernel
def apply_viscous_friction_kernel(
    velocities: wp.array(dtype=wp.vec3f),
    angular_velocities: wp.array(dtype=wp.vec3f),
    contact_depths: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.bool),
    viscous_coefficient: float,
    dt: float,
):
    """Apply viscous friction at contact points.
    
    Viscous friction: F = -c * v
    Provides velocity-dependent damping at contacts.
    """
    idx = wp.tid()
    
    if fixed[idx]:
        return
    
    # Check if in contact
    depth = contact_depths[idx]
    if depth <= 0.0:
        return
    
    # Apply viscous damping proportional to contact depth
    damping = viscous_coefficient * depth * 100.0  # Scale factor
    damping = wp.min(damping, 0.99)  # Clamp to prevent energy gain
    
    velocities[idx] = velocities[idx] * (1.0 - damping)
    angular_velocities[idx] = angular_velocities[idx] * (1.0 - damping)


# ============================================================================
# Mesh Collision Kernel (BVH-accelerated)
# ============================================================================


@wp.kernel
def solve_mesh_collision_kernel(
    positions: wp.array(dtype=wp.vec3f),
    velocities: wp.array(dtype=wp.vec3f),
    radii: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.bool),
    mesh: wp.uint64,  # Mesh ID for BVH queries
    contact_normals: wp.array(dtype=wp.vec3f),
    contact_depths: wp.array(dtype=wp.float32),
    restitution: float,
    collision_radius: float,
):
    """Solve collision constraints with triangle mesh using BVH.
    
    Uses Warp's BVH-accelerated mesh queries to find closest point
    on mesh surface and apply position correction.
    """
    idx = wp.tid()
    
    if fixed[idx]:
        contact_depths[idx] = 0.0
        return
    
    pos = positions[idx]
    radius = radii[idx] + collision_radius
    
    # Query closest point on mesh
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    
    # Use mesh query to find closest point
    max_dist = radius * 2.0
    
    # Query mesh for closest point
    success = wp.mesh_query_point(mesh, pos, max_dist, sign, face_index, face_u, face_v)
    
    if not success:
        contact_depths[idx] = 0.0
        contact_normals[idx] = wp.vec3f(0.0, 1.0, 0.0)
        return
    
    # Get closest point on mesh
    closest = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
    
    # Compute penetration
    diff = pos - closest
    dist = wp.length(diff)
    
    if dist < 1e-8:
        # Use face normal if too close
        contact_normals[idx] = wp.mesh_eval_face_normal(mesh, face_index)
        contact_depths[idx] = radius
        positions[idx] = closest + contact_normals[idx] * radius
        return
    
    normal = diff / dist
    
    # Inside mesh check (sign < 0 means inside)
    if sign < 0.0:
        # We're inside the mesh, push out
        penetration = radius + dist
        contact_depths[idx] = penetration
        contact_normals[idx] = -normal  # Flip normal to point outward
        positions[idx] = closest - normal * radius
    elif dist < radius:
        # Outside but penetrating
        penetration = radius - dist
        contact_depths[idx] = penetration
        contact_normals[idx] = normal
        positions[idx] = closest + normal * radius
    else:
        # No penetration
        contact_depths[idx] = 0.0
        contact_normals[idx] = normal


# ============================================================================
# Direct Solver Kernels (Linear-Time Tree Algorithm)
# ============================================================================


@wp.kernel
def compute_constraint_residuals_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    segment_lengths: wp.array(dtype=wp.float32),
    rest_darboux: wp.array(dtype=wp.vec3f),
    parent_indices: wp.array(dtype=wp.int32),
    stretch_residual: wp.array(dtype=wp.vec3f),
    bend_twist_residual: wp.array(dtype=wp.vec3f),
    num_segments: int,
):
    """Compute constraint residuals for all constraints.

    This is the first step of the direct solver: evaluate all constraints.
    """
    idx = wp.tid()

    segment_idx = idx + 1
    if segment_idx >= num_segments:
        return

    parent_idx = parent_indices[segment_idx]
    if parent_idx < 0:
        return

    # Get positions and orientations
    x1 = positions[parent_idx]
    x2 = positions[segment_idx]
    q1 = orientations[parent_idx]
    q2 = orientations[segment_idx]

    L1 = segment_lengths[parent_idx]
    L2 = segment_lengths[segment_idx]

    # Compute attachment points
    local_end = wp.vec3f(L1 * 0.5, 0.0, 0.0)
    local_start = wp.vec3f(-L2 * 0.5, 0.0, 0.0)
    p1 = x1 + quat_rotate_vec(q1, local_end)
    p2 = x2 + quat_rotate_vec(q2, local_start)

    # Stretch residual (should be zero vector)
    stretch_residual[idx] = p2 - p1

    # Bend/twist residual
    L = 0.5 * (L1 + L2)
    darboux = compute_darboux_vector(q1, q2, L)
    bend_twist_residual[idx] = darboux - rest_darboux[idx]


@wp.kernel
def reset_lambda_kernel(
    lambda_stretch: wp.array(dtype=wp.float32),
    lambda_bend_twist: wp.array(dtype=wp.vec3f),
):
    """Reset Lagrange multipliers at the start of each time step."""
    idx = wp.tid()
    lambda_stretch[idx] = 0.0
    lambda_bend_twist[idx] = wp.vec3f(0.0, 0.0, 0.0)


@wp.kernel
def reset_lambda_with_shear_kernel(
    lambda_stretch: wp.array(dtype=wp.float32),
    lambda_shear: wp.array(dtype=wp.vec2f),
    lambda_bend_twist: wp.array(dtype=wp.vec3f),
):
    """Reset all Lagrange multipliers including shear."""
    idx = wp.tid()
    lambda_stretch[idx] = 0.0
    lambda_shear[idx] = wp.vec2f(0.0, 0.0)
    lambda_bend_twist[idx] = wp.vec3f(0.0, 0.0, 0.0)


# ============================================================================
# Collision Constraint Kernels
# ============================================================================


@wp.kernel
def solve_ground_collision_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    segment_lengths: wp.array(dtype=wp.float32),
    radii: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.bool),
    ground_height: float,
    restitution: float,
):
    """Solve collision constraints with ground plane.

    Simple ground plane collision at y = ground_height.
    """
    idx = wp.tid()

    if fixed[idx]:
        return

    # Get segment position and radius
    pos = positions[idx]
    radius = radii[idx]

    # Check penetration with ground
    penetration = ground_height + radius - pos[1]

    if penetration > 0.0:
        # Push out of ground
        positions[idx] = wp.vec3f(pos[0], ground_height + radius, pos[2])


@wp.kernel
def solve_self_collision_kernel(
    positions: wp.array(dtype=wp.vec3f),
    radii: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    fixed: wp.array(dtype=wp.bool),
    num_segments: int,
    collision_margin: float,
):
    """Solve self-collision constraints between non-adjacent segments.

    Uses a simple sphere-sphere collision model.
    """
    idx = wp.tid()

    if fixed[idx]:
        return

    # Check against all other non-adjacent segments
    for j in range(num_segments):
        # Skip self and adjacent segments
        if j == idx or j == idx - 1 or j == idx + 1:
            continue

        if fixed[j]:
            continue

        # Compute distance between segment centers
        diff = positions[j] - positions[idx]
        dist = wp.length(diff)

        # Combined radius with margin
        r_sum = radii[idx] + radii[j] + collision_margin

        # Check overlap
        if dist < r_sum and dist > 1e-8:
            # Penetration depth
            penetration = r_sum - dist

            # Normal direction
            n = diff / dist

            # Inverse mass ratio
            w1 = inv_masses[idx]
            w2 = inv_masses[j]
            w_sum = w1 + w2

            if w_sum > 1e-10:
                # Position correction
                corr = penetration * n / w_sum

                positions[idx] = positions[idx] - w1 * corr
                positions[j] = positions[j] + w2 * corr


# ============================================================================
# Utility Kernels
# ============================================================================


@wp.kernel
def compute_total_energy_kernel(
    positions: wp.array(dtype=wp.vec3f),
    orientations: wp.array(dtype=wp.quatf),
    velocities: wp.array(dtype=wp.vec3f),
    angular_velocities: wp.array(dtype=wp.vec3f),
    masses: wp.array(dtype=wp.float32),
    inertias_diag: wp.array(dtype=wp.vec3f),
    segment_lengths: wp.array(dtype=wp.float32),
    rest_darboux: wp.array(dtype=wp.vec3f),
    bending_stiffness: wp.array(dtype=wp.vec2f),
    torsion_stiffness: wp.array(dtype=wp.float32),
    parent_indices: wp.array(dtype=wp.int32),
    gravity: wp.vec3f,
    kinetic_energy: wp.array(dtype=wp.float32),
    potential_energy: wp.array(dtype=wp.float32),
    num_segments: int,
):
    """Compute total kinetic and potential energy of the rod.

    Used for debugging and validation.
    """
    idx = wp.tid()

    if idx >= num_segments:
        return

    m = masses[idx]
    v = velocities[idx]
    omega = angular_velocities[idx]
    I_diag = inertias_diag[idx]

    # Kinetic energy: 0.5 * m * |v|² + 0.5 * ω^T * I * ω
    ke = 0.5 * m * wp.dot(v, v)
    ke = ke + 0.5 * (I_diag[0] * omega[0] * omega[0] + I_diag[1] * omega[1] * omega[1] + I_diag[2] * omega[2] * omega[2])

    # Gravitational potential energy: m * g * h
    pe_gravity = -m * wp.dot(gravity, positions[idx])

    # Elastic potential energy (bending/twisting) - only for non-root segments
    pe_elastic = 0.0
    if idx > 0:
        parent_idx = parent_indices[idx]
        if parent_idx >= 0:
            constraint_idx = idx - 1
            q1 = orientations[parent_idx]
            q2 = orientations[idx]
            L = 0.5 * (segment_lengths[parent_idx] + segment_lengths[idx])

            darboux = compute_darboux_vector(q1, q2, L)
            diff = darboux - rest_darboux[constraint_idx]

            # Bending energy
            k_bend = bending_stiffness[constraint_idx]
            pe_bend_x = 0.5 * L * k_bend[0] * diff[0] * diff[0]
            pe_bend_y = 0.5 * L * k_bend[1] * diff[1] * diff[1]
            pe_elastic = pe_elastic + pe_bend_x + pe_bend_y

            # Torsion energy
            k_twist = torsion_stiffness[constraint_idx]
            pe_twist = 0.5 * L * k_twist * diff[2] * diff[2]
            pe_elastic = pe_elastic + pe_twist

    kinetic_energy[idx] = ke
    potential_energy[idx] = pe_gravity + pe_elastic


@wp.kernel
def normalize_quaternions_kernel(
    orientations: wp.array(dtype=wp.quatf),
):
    """Normalize all quaternions to ensure unit length."""
    idx = wp.tid()
    q = orientations[idx]
    q_len = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if q_len > 1e-8:
        orientations[idx] = wp.quatf(q[0] / q_len, q[1] / q_len, q[2] / q_len, q[3] / q_len)

