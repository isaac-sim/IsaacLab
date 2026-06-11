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

from isaaclab_ovphysx._runtime import import_ovphysx

TensorType = import_ovphysx("ovphysx.types").TensorType

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
Rigid-body TensorTypes

Shapes assume N = number of rigid actor instances matched by the binding
pattern. Components and units are stated per alias below.
"""

RIGID_BODY_POSE = _TT.RIGID_BODY_POSE
"""Rigid actor root transform — read/write, GPU. Shape ``(N, 7)``,
components ``(px, py, pz, qx, qy, qz, qw)`` [m, dimensionless]."""

RIGID_BODY_VELOCITY = _TT.RIGID_BODY_VELOCITY
"""Rigid actor root spatial velocity — read/write, GPU. Shape ``(N, 6)``,
components ``(vx, vy, vz, wx, wy, wz)`` [m/s, rad/s]."""

RIGID_BODY_WRENCH = _TT.RIGID_BODY_WRENCH
"""External wrench applied at a world-frame point — write-only, GPU.
Shape ``(N, 9)``, components ``(fx, fy, fz, tx, ty, tz, px, py, pz)``
[N, N·m, m]. Cleared after each sim step (instantaneous semantics)."""

RIGID_BODY_MASS = _TT.RIGID_BODY_MASS
"""Rigid actor mass — read/write, CPU. Shape ``(N,)`` [kg]."""

RIGID_BODY_COM_POSE = _TT.RIGID_BODY_COM_POSE
"""Center-of-mass pose in actor-link frame — read/write, CPU. Shape
``(N, 7)``, components ``(px, py, pz, qx, qy, qz, qw)`` [m, dimensionless]."""

RIGID_BODY_INERTIA = _TT.RIGID_BODY_INERTIA
"""Rigid actor inertia tensor in COM frame — read/write, CPU. Shape
``(N, 9)``, row-major flatten of the 3×3 inertia matrix
``(Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz)`` [kg·m²]."""

# These three aliases are pending an upcoming ovphysx wheel update.
# When the wheel ships them, the corresponding ``hasattr`` checks below
# in IsaacLab consumers will start returning True and the bindings will
# become usable; until then, ``isaaclab_ovphysx.tensor_types`` simply
# does not expose the alias.
try:
    RIGID_BODY_ACCELERATION = _TT.RIGID_BODY_ACCELERATION
    """Rigid actor spatial acceleration — read-only, GPU. Shape ``(N, 6)``,
    components ``(ax, ay, az, αx, αy, αz)`` [m/s², rad/s²]."""
except AttributeError:
    pass

try:
    RIGID_BODY_INV_MASS = _TT.RIGID_BODY_INV_MASS
    """Rigid actor inverse mass — read-only, CPU. Shape ``(N,)`` [1/kg].
    Zero indicates an immovable actor."""
except AttributeError:
    pass

try:
    RIGID_BODY_INV_INERTIA = _TT.RIGID_BODY_INV_INERTIA
    """Rigid actor inverse inertia tensor in COM frame — read-only, CPU.
    Shape ``(N, 9)``, row-major flatten of the 3×3 matrix [1/(kg·m²)].
    Zero rows indicate locked rotational DOFs."""
except AttributeError:
    pass

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
#
# Tendon tensor types are NOT in this set: PhysX exposes tendons on the
# simulation device (its ``set_fixed_tendon_properties`` takes ``data.warp``
# without a ``device="cpu"`` clone, unlike ``set_dof_stiffnesses``), and the
# OVPhysX wheel mirrors that — tendon bindings are GPU-resident on a GPU sim.
_CPU_ONLY_TYPES_CANDIDATES: tuple = (
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
    # Rigid-body CPU-only entries (always available)
    RIGID_BODY_MASS,
    RIGID_BODY_COM_POSE,
    RIGID_BODY_INERTIA,
)
# Optional rigid-body CPU entries: only included when the wheel exposes them.
_RIGID_BODY_OPTIONAL_CPU: tuple = tuple(
    globals()[name] for name in ("RIGID_BODY_INV_MASS", "RIGID_BODY_INV_INERTIA") if name in globals()
)
_CPU_ONLY_TYPES: frozenset[TensorType] = frozenset(_CPU_ONLY_TYPES_CANDIDATES + _RIGID_BODY_OPTIONAL_CPU)
