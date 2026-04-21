# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab re-exports of ovphysx TensorType with short backward-compat aliases.

Import TensorType directly for new code:
    from ovphysx.types import TensorType

Or use the module-level short aliases (existing code pattern):
    import isaaclab_ovphysx.tensor_types as TT
    TT.DOF_STIFFNESS  # resolves to TensorType.ARTICULATION_DOF_STIFFNESS

ovphysx.types is pure Python with zero native dependencies, so this module is
always safe to import regardless of USD state or native library loading.
"""

from ovphysx.types import TensorType  # noqa: F401 — re-exported for new code

_TT = TensorType  # shorter reference for alias block

# Short aliases -- existing code using ``TT.DOF_STIFFNESS`` etc. continues to work.
# All values are IntEnum members (== plain ints) of TensorType.

# fmt: off  -- aligned columns are intentional; do not reformat

"""
Root state (GPU)
"""

ROOT_POSE = _TT.ARTICULATION_ROOT_POSE
"""Root pose of each articulation instance.

Shape is ``[N, 7]``, dtype ``float32`` (px, py, pz, qx, qy, qz, qw).
"""

ROOT_VELOCITY = _TT.ARTICULATION_ROOT_VELOCITY
"""Root velocity of each articulation instance.

Shape is ``[N, 6]``, dtype ``float32`` (vx, vy, vz, wx, wy, wz).
"""

"""
Link (body) state (GPU)
"""

LINK_POSE = _TT.ARTICULATION_LINK_POSE
"""Pose of every link (body) in each articulation instance.

Shape is ``[N, L, 7]``, dtype ``float32``.
"""

LINK_VELOCITY = _TT.ARTICULATION_LINK_VELOCITY
"""Velocity of every link (body) in each articulation instance.

Shape is ``[N, L, 6]``, dtype ``float32``.
"""

LINK_ACCELERATION = _TT.ARTICULATION_LINK_ACCELERATION
"""Acceleration of every link (body) in each articulation instance.

Shape is ``[N, L, 6]``, dtype ``float32``.
"""

"""
DOF state (GPU)
"""

DOF_POSITION = _TT.ARTICULATION_DOF_POSITION
"""DOF (joint) positions.

Shape is ``[N, D]``, dtype ``float32`` [m or rad].
"""

DOF_VELOCITY = _TT.ARTICULATION_DOF_VELOCITY
"""DOF (joint) velocities.

Shape is ``[N, D]``, dtype ``float32`` [m/s or rad/s].
"""

"""
DOF command targets (GPU, write-only)
"""

DOF_POSITION_TARGET = _TT.ARTICULATION_DOF_POSITION_TARGET
"""DOF position targets for the PD controller.

Shape is ``[N, D]``, dtype ``float32``.
"""

DOF_VELOCITY_TARGET = _TT.ARTICULATION_DOF_VELOCITY_TARGET
"""DOF velocity targets for the PD controller.

Shape is ``[N, D]``, dtype ``float32``.
"""

DOF_ACTUATION_FORCE = _TT.ARTICULATION_DOF_ACTUATION_FORCE
"""DOF actuation (effort) forces applied directly.

Shape is ``[N, D]``, dtype ``float32`` [N or N·m].
"""

"""
DOF properties (CPU)
"""

DOF_STIFFNESS = _TT.ARTICULATION_DOF_STIFFNESS
"""DOF stiffness (spring constant for PD controller).

Shape is ``[N, D]``, dtype ``float32``.
"""

DOF_DAMPING = _TT.ARTICULATION_DOF_DAMPING
"""DOF damping (damper constant for PD controller).

Shape is ``[N, D]``, dtype ``float32``.
"""

DOF_LIMIT = _TT.ARTICULATION_DOF_LIMIT
"""DOF position limits (lower, upper).

Shape is ``[N, D, 2]``, dtype ``float32``.
"""

DOF_MAX_VELOCITY = _TT.ARTICULATION_DOF_MAX_VELOCITY
"""DOF maximum velocity.

Shape is ``[N, D]``, dtype ``float32``.
"""

DOF_MAX_FORCE = _TT.ARTICULATION_DOF_MAX_FORCE
"""DOF maximum force.

Shape is ``[N, D]``, dtype ``float32``.
"""

DOF_ARMATURE = _TT.ARTICULATION_DOF_ARMATURE
"""DOF armature (added inertia on the diagonal of the joint-space mass matrix).

Shape is ``[N, D]``, dtype ``float32``.
"""

DOF_FRICTION_PROPERTIES = _TT.ARTICULATION_DOF_FRICTION_PROPERTIES
"""DOF friction properties (static, dynamic, viscous).

Shape is ``[N, D, 3]``, dtype ``float32``.
"""

"""
External wrench (GPU, write-only)
"""

LINK_WRENCH = _TT.ARTICULATION_LINK_WRENCH
"""External wrench applied to each link.

Shape is ``[N, L, 9]``, dtype ``float32`` (fx, fy, fz, tx, ty, tz, px, py, pz).
"""

"""
Body properties (CPU)
"""

BODY_MASS = _TT.ARTICULATION_BODY_MASS
"""Mass of each body (link).

Shape is ``[N, L]``, dtype ``float32`` [kg].
"""

BODY_COM_POSE = _TT.ARTICULATION_BODY_COM_POSE
"""Center-of-mass pose of each body in local frame.

Shape is ``[N, L, 7]``, dtype ``float32``.
"""

BODY_INERTIA = _TT.ARTICULATION_BODY_INERTIA
"""Inertia tensor of each body.

Shape is ``[N, L, 9]``, dtype ``float32`` [kg·m^2].
"""

BODY_INV_MASS = _TT.ARTICULATION_BODY_INV_MASS
"""Inverse mass of each body.

Shape is ``[N, L]``, dtype ``float32``.
"""

BODY_INV_INERTIA = _TT.ARTICULATION_BODY_INV_INERTIA
"""Inverse inertia tensor of each body.

Shape is ``[N, L, 9]``, dtype ``float32``.
"""

"""
Dynamics tensors (GPU)
"""

JACOBIAN = _TT.ARTICULATION_JACOBIAN
"""Jacobian matrix of each articulation instance.

Shape is ``[N, L, 6, D+6]``, dtype ``float32``.
"""

MASS_MATRIX = _TT.ARTICULATION_MASS_MATRIX
"""Generalized mass (inertia) matrix.

Shape is ``[N, D+6, D+6]``, dtype ``float32``.
"""

CORIOLIS = _TT.ARTICULATION_CORIOLIS_AND_CENTRIFUGAL_FORCE
"""Coriolis and centrifugal force vector.

Shape is ``[N, D]``, dtype ``float32``.
"""

GRAVITY_FORCE = _TT.ARTICULATION_GRAVITY_FORCE
"""Generalized gravity force vector.

Shape is ``[N, D]``, dtype ``float32``.
"""

"""
Joint force feedback (GPU)
"""

LINK_INCOMING_JOINT_FORCE = _TT.ARTICULATION_LINK_INCOMING_JOINT_FORCE
"""Incoming joint force (constraint force) on each link.

Shape is ``[N, L, 6]``, dtype ``float32``.
"""

DOF_PROJECTED_JOINT_FORCE = _TT.ARTICULATION_DOF_PROJECTED_JOINT_FORCE
"""DOF-projected joint force.

Shape is ``[N, D]``, dtype ``float32``.
"""

"""
Fixed tendon properties (CPU)
"""

FIXED_TENDON_STIFFNESS = _TT.ARTICULATION_FIXED_TENDON_STIFFNESS
"""Stiffness of each fixed tendon.

Shape is ``[N, T_fix]``, dtype ``float32``.
"""

FIXED_TENDON_DAMPING = _TT.ARTICULATION_FIXED_TENDON_DAMPING
"""Damping of each fixed tendon.

Shape is ``[N, T_fix]``, dtype ``float32``.
"""

FIXED_TENDON_LIMIT_STIFFNESS = _TT.ARTICULATION_FIXED_TENDON_LIMIT_STIFFNESS
"""Limit stiffness of each fixed tendon.

Shape is ``[N, T_fix]``, dtype ``float32``.
"""

FIXED_TENDON_LIMIT = _TT.ARTICULATION_FIXED_TENDON_LIMIT
"""Position limits of each fixed tendon (lower, upper).

Shape is ``[N, T_fix, 2]``, dtype ``float32``.
"""

FIXED_TENDON_REST_LENGTH = _TT.ARTICULATION_FIXED_TENDON_REST_LENGTH
"""Rest length of each fixed tendon.

Shape is ``[N, T_fix]``, dtype ``float32``.
"""

FIXED_TENDON_OFFSET = _TT.ARTICULATION_FIXED_TENDON_OFFSET
"""Offset of each fixed tendon.

Shape is ``[N, T_fix]``, dtype ``float32``.
"""

"""
Spatial tendon properties (CPU)
"""

SPATIAL_TENDON_STIFFNESS = _TT.ARTICULATION_SPATIAL_TENDON_STIFFNESS
"""Stiffness of each spatial tendon.

Shape is ``[N, T_spa]``, dtype ``float32``.
"""

SPATIAL_TENDON_DAMPING = _TT.ARTICULATION_SPATIAL_TENDON_DAMPING
"""Damping of each spatial tendon.

Shape is ``[N, T_spa]``, dtype ``float32``.
"""

SPATIAL_TENDON_LIMIT_STIFFNESS = _TT.ARTICULATION_SPATIAL_TENDON_LIMIT_STIFFNESS
"""Limit stiffness of each spatial tendon.

Shape is ``[N, T_spa]``, dtype ``float32``.
"""

SPATIAL_TENDON_OFFSET = _TT.ARTICULATION_SPATIAL_TENDON_OFFSET
"""Offset of each spatial tendon.

Shape is ``[N, T_spa]``, dtype ``float32``.
"""

# fmt: on
# DOF/body property tensor types are CPU-resident even in GPU simulations.
# Write helpers check this set to route data through CPU, not self._device.
_CPU_ONLY_TYPES: frozenset[TensorType] = frozenset(
    {
        DOF_STIFFNESS,
        DOF_DAMPING,
        DOF_LIMIT,
        DOF_MAX_VELOCITY,
        DOF_MAX_FORCE,
        DOF_ARMATURE,
        DOF_FRICTION_PROPERTIES,
        BODY_MASS,
        BODY_COM_POSE,
        BODY_INERTIA,
        BODY_INV_MASS,
        BODY_INV_INERTIA,
        FIXED_TENDON_STIFFNESS,
        FIXED_TENDON_DAMPING,
        FIXED_TENDON_LIMIT_STIFFNESS,
        FIXED_TENDON_LIMIT,
        FIXED_TENDON_REST_LENGTH,
        FIXED_TENDON_OFFSET,
        SPATIAL_TENDON_STIFFNESS,
        SPATIAL_TENDON_DAMPING,
        SPATIAL_TENDON_LIMIT_STIFFNESS,
        SPATIAL_TENDON_OFFSET,
    }
)
