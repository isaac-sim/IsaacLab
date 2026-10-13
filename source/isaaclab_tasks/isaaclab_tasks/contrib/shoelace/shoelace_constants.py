# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime asset paths and physical defaults shared with offline authoring."""

from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "data"

# Runtime assets; texture paths are resolved relative to each USD.
SHOELACE_ASSET = ASSET_DIR / "shoelace.usda"  # Base asset
SHOELACE_ASSETS = (
    SHOELACE_ASSET,
    *(
        ASSET_DIR / name
        for name in (
            "canvas_sand.usda",
            "canvas_rust.usda",
            "suede_sage.usda",
            "suede_burgundy.usda",
            "knit_ocean.usda",
            "knit_lavender.usda",
        )
    ),
)

# Control defaults.
TCP_OFFSET = (0.0, 0.0, 0.1034)
ARM_ACTION_SCALE = 0.005  # [m]
ARM_ROTATION_ACTION_SCALE = 0.01  # [rad]
GRIPPER_OPEN_POSITION = 0.01
GRIPPER_CLOSED_POSITION = 0.001
GRIPPER_STIFFNESS = 8000.0
CONTACT_OBSERVATION_HISTORY_LENGTH = 3

# Success criteria.
TAIL_SUCCESS_OUTWARD_DISTANCE = 0.09
TAIL_SUCCESS_X_SEPARATION = 2.0 * TAIL_SUCCESS_OUTWARD_DISTANCE
THROAT_RADIUS = 0.025
MAXIMUM_THROAT_SEGMENTS_PER_ARM = 15

# Solver defaults.
NEWTON_NUM_SUBSTEPS = 10
NEWTON_COLLISION_DECIMATION = 2
VBD_ITERATIONS = 12
ADMM_ITERATIONS = 2
ADMM_RHO = 500.0
ADMM_BAUMGARTE = 0.5
ADMM_CONTACT_MATCHING = "latest"

# Initial poses.
LEFT_ROBOT_POSITION = (-0.525248, 0.023338, -0.089901)
RIGHT_ROBOT_POSITION = (0.507963, -0.001890, -0.088518)
LEFT_ARM_JOINT_POSITIONS = {
    "panda_joint1": 0.267436,
    "panda_joint2": -0.276844,
    "panda_joint3": -0.509809,
    "panda_joint4": -2.681384,
    "panda_joint5": 0.871062,
    "panda_joint6": 2.543250,
    "panda_joint7": -0.079776,
}
RIGHT_ARM_JOINT_POSITIONS = {
    "panda_joint1": -0.379171,
    "panda_joint2": -0.306599,
    "panda_joint3": 0.461994,
    "panda_joint4": -2.726323,
    "panda_joint5": -0.930591,
    "panda_joint6": 2.645580,
    "panda_joint7": 1.577059,
}

# Shared Franka-validated cable material and contact defaults.
CABLE_INERTIA_REGULARIZATION = 1.0e-6  # Additive isotropic proxy inertia [kg*m^2].
STRETCH_STIFFNESS = 1.0e7
STRETCH_DAMPING = 2.0e2
BEND_STIFFNESS = 5.0
BEND_DAMPING = 1.0
CONTACT_GAP = 1.0e-4
CONTACT_DISTANCE_CAP = 2.0e-3
CONTACT_KE = 1.0e6
CONTACT_KD = 0.0
FINGER_MU = 40.0
VBD_CONTACT_BUFFER = 256
CONTACTS_PER_ENV = 512
TRIANGLE_PAIRS_PER_ENV = 8192
MIN_TRIANGLE_PAIRS = 1_000_000

TAIL_BEND_STIFFNESS = 100.0
TAIL_BEND_DAMPING = 2.0
TAIL_STIFF_CORE_LENGTH = 0.050
TAIL_STIFF_TRANSITION_LENGTH = 0.012
