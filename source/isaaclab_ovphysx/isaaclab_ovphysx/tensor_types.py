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

# --- Root state (GPU) ---
ROOT_POSE                    = _TT.ARTICULATION_ROOT_POSE              # [N, 7]  float32  (px,py,pz,qx,qy,qz,qw)
ROOT_VELOCITY                = _TT.ARTICULATION_ROOT_VELOCITY          # [N, 6]  float32  (vx,vy,vz,wx,wy,wz)

# --- Link (body) state (GPU) ---
LINK_POSE                    = _TT.ARTICULATION_LINK_POSE              # [N, L, 7]  float32
LINK_VELOCITY                = _TT.ARTICULATION_LINK_VELOCITY          # [N, L, 6]  float32
LINK_ACCELERATION            = _TT.ARTICULATION_LINK_ACCELERATION      # [N, L, 6]  float32

# --- DOF state (GPU) ---
DOF_POSITION                 = _TT.ARTICULATION_DOF_POSITION           # [N, D]  float32  [m or rad]
DOF_VELOCITY                 = _TT.ARTICULATION_DOF_VELOCITY           # [N, D]  float32  [m/s or rad/s]

# --- DOF command targets (GPU, write-only) ---
DOF_POSITION_TARGET          = _TT.ARTICULATION_DOF_POSITION_TARGET    # [N, D]  float32
DOF_VELOCITY_TARGET          = _TT.ARTICULATION_DOF_VELOCITY_TARGET    # [N, D]  float32
DOF_ACTUATION_FORCE          = _TT.ARTICULATION_DOF_ACTUATION_FORCE    # [N, D]  float32  [N or N*m]

# --- DOF properties (CPU) ---
DOF_STIFFNESS                = _TT.ARTICULATION_DOF_STIFFNESS          # [N, D]  float32
DOF_DAMPING                  = _TT.ARTICULATION_DOF_DAMPING            # [N, D]  float32
DOF_LIMIT                    = _TT.ARTICULATION_DOF_LIMIT              # [N, D, 2]  float32  [lower, upper]
DOF_MAX_VELOCITY             = _TT.ARTICULATION_DOF_MAX_VELOCITY       # [N, D]  float32
DOF_MAX_FORCE                = _TT.ARTICULATION_DOF_MAX_FORCE          # [N, D]  float32
DOF_ARMATURE                 = _TT.ARTICULATION_DOF_ARMATURE           # [N, D]  float32
DOF_FRICTION_PROPERTIES      = _TT.ARTICULATION_DOF_FRICTION_PROPERTIES  # [N, D, 3]  float32  (static, dynamic, viscous)

# --- External wrench (GPU, write-only) ---
LINK_WRENCH                  = _TT.ARTICULATION_LINK_WRENCH            # [N, L, 9]  float32  (fx,fy,fz,tx,ty,tz,px,py,pz)

# --- Body properties (CPU) ---
BODY_MASS                    = _TT.ARTICULATION_BODY_MASS              # [N, L]  float32  [kg]
BODY_COM_POSE                = _TT.ARTICULATION_BODY_COM_POSE          # [N, L, 7]  float32
BODY_INERTIA                 = _TT.ARTICULATION_BODY_INERTIA           # [N, L, 9]  float32  [kg*m^2]
BODY_INV_MASS                = _TT.ARTICULATION_BODY_INV_MASS          # [N, L]  float32
BODY_INV_INERTIA             = _TT.ARTICULATION_BODY_INV_INERTIA       # [N, L, 9]  float32

# --- Dynamics tensors (GPU) ---
JACOBIAN                     = _TT.ARTICULATION_JACOBIAN               # [N, L, 6, D+6]  float32
MASS_MATRIX                  = _TT.ARTICULATION_MASS_MATRIX            # [N, D+6, D+6]  float32
CORIOLIS                     = _TT.ARTICULATION_CORIOLIS_AND_CENTRIFUGAL_FORCE  # [N, D]  float32
GRAVITY_FORCE                = _TT.ARTICULATION_GRAVITY_FORCE          # [N, D]  float32

# --- Joint force feedback (GPU) ---
LINK_INCOMING_JOINT_FORCE    = _TT.ARTICULATION_LINK_INCOMING_JOINT_FORCE  # [N, L, 6]  float32
DOF_PROJECTED_JOINT_FORCE    = _TT.ARTICULATION_DOF_PROJECTED_JOINT_FORCE  # [N, D]  float32

# --- Fixed tendon properties (CPU) ---
FIXED_TENDON_STIFFNESS       = _TT.ARTICULATION_FIXED_TENDON_STIFFNESS       # [N, T_fix]  float32
FIXED_TENDON_DAMPING         = _TT.ARTICULATION_FIXED_TENDON_DAMPING         # [N, T_fix]  float32
FIXED_TENDON_LIMIT_STIFFNESS = _TT.ARTICULATION_FIXED_TENDON_LIMIT_STIFFNESS # [N, T_fix]  float32
FIXED_TENDON_LIMIT           = _TT.ARTICULATION_FIXED_TENDON_LIMIT           # [N, T_fix, 2]  float32
FIXED_TENDON_REST_LENGTH     = _TT.ARTICULATION_FIXED_TENDON_REST_LENGTH     # [N, T_fix]  float32
FIXED_TENDON_OFFSET          = _TT.ARTICULATION_FIXED_TENDON_OFFSET          # [N, T_fix]  float32

# --- Spatial tendon properties (CPU) ---
SPATIAL_TENDON_STIFFNESS     = _TT.ARTICULATION_SPATIAL_TENDON_STIFFNESS       # [N, T_spa]  float32
SPATIAL_TENDON_DAMPING       = _TT.ARTICULATION_SPATIAL_TENDON_DAMPING         # [N, T_spa]  float32
SPATIAL_TENDON_LIMIT_STIFFNESS = _TT.ARTICULATION_SPATIAL_TENDON_LIMIT_STIFFNESS  # [N, T_spa]  float32
SPATIAL_TENDON_OFFSET        = _TT.ARTICULATION_SPATIAL_TENDON_OFFSET          # [N, T_spa]  float32

# DOF/body property tensor types are CPU-resident even in GPU simulations.
# Write helpers check this set to route data through CPU, not self._device.
_CPU_ONLY_TYPES: frozenset[TensorType] = frozenset({
    DOF_STIFFNESS, DOF_DAMPING, DOF_LIMIT, DOF_MAX_VELOCITY, DOF_MAX_FORCE,
    DOF_ARMATURE, DOF_FRICTION_PROPERTIES,
    BODY_MASS, BODY_COM_POSE, BODY_INERTIA, BODY_INV_MASS, BODY_INV_INERTIA,
    FIXED_TENDON_STIFFNESS, FIXED_TENDON_DAMPING, FIXED_TENDON_LIMIT_STIFFNESS,
    FIXED_TENDON_LIMIT, FIXED_TENDON_REST_LENGTH, FIXED_TENDON_OFFSET,
    SPATIAL_TENDON_STIFFNESS, SPATIAL_TENDON_DAMPING,
    SPATIAL_TENDON_LIMIT_STIFFNESS, SPATIAL_TENDON_OFFSET,
})
