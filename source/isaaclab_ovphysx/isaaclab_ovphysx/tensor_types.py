# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ovphysx tensor type constants mirroring ovphysx_types.h.

We avoid ``import ovphysx`` at module level because it triggers USD version
checks that conflict with IsaacSim's bundled pxr. These integer values are
stable across ovphysx releases and match the C enum in
``omni/ovphysx/include/ovphysx/ovphysx_types.h``.
"""

# -- Articulation root state (read/write) --
ROOT_POSE = 10  # [N, 7]
ROOT_VELOCITY = 11  # [N, 6]

# -- Articulation link state (read-only except wrench) --
LINK_POSE = 20  # [N, L, 7]
LINK_VELOCITY = 21  # [N, L, 6]
LINK_ACCELERATION = 22  # [N, L, 6] (read-only)

# -- Articulation DOF state (read/write) --
DOF_POSITION = 30  # [N, D]
DOF_VELOCITY = 31  # [N, D]
DOF_POSITION_TARGET = 32  # [N, D]
DOF_VELOCITY_TARGET = 33  # [N, D]
DOF_ACTUATION_FORCE = 34  # [N, D]

# -- Articulation DOF properties (read/write, CPU-side in GPU mode) --
DOF_STIFFNESS = 35  # [N, D]
DOF_DAMPING = 36  # [N, D]
DOF_LIMIT = 37  # [N, D, 2]
DOF_MAX_VELOCITY = 38  # [N, D]
DOF_MAX_FORCE = 39  # [N, D]
DOF_ARMATURE = 40  # [N, D]
DOF_FRICTION_PROPERTIES = 41  # [N, D, 3] (static, dynamic, viscous)

# -- External wrenches (write-only) --
LINK_WRENCH = 52  # [N, L, 9] layout: [fx, fy, fz, tx, ty, tz, px, py, pz] world frame

# -- Articulation body properties (read/write unless noted) --
BODY_MASS = 60  # [N, L]
BODY_COM_POSE = 61  # [N, L, 7]
BODY_INERTIA = 62  # [N, L, 9]
BODY_INV_MASS = 63  # [N, L] (read-only)
BODY_INV_INERTIA = 64  # [N, L, 9] (read-only)

# -- Articulation dynamics queries (read-only) --
JACOBIAN = 70  # [N, R, C]
MASS_MATRIX = 71  # [N, M, M]
CORIOLIS = 72  # [N, M]
GRAVITY_FORCE = 73  # [N, M]
LINK_INCOMING_JOINT_FORCE = 74  # [N, L, 6]
DOF_PROJECTED_JOINT_FORCE = 75  # [N, D] (read-only)

# -- Fixed tendon properties (read/write, CPU-side) --
FIXED_TENDON_STIFFNESS = 80  # [N, T]
FIXED_TENDON_DAMPING = 81  # [N, T]
FIXED_TENDON_LIMIT_STIFFNESS = 82  # [N, T]
FIXED_TENDON_LIMIT = 83  # [N, T, 2]
FIXED_TENDON_REST_LENGTH = 84  # [N, T]
FIXED_TENDON_OFFSET = 85  # [N, T]

# -- Spatial tendon properties (read/write, CPU-side) --
SPATIAL_TENDON_STIFFNESS = 90  # [N, T]
SPATIAL_TENDON_DAMPING = 91  # [N, T]
SPATIAL_TENDON_LIMIT_STIFFNESS = 92  # [N, T]
SPATIAL_TENDON_OFFSET = 93  # [N, T]
