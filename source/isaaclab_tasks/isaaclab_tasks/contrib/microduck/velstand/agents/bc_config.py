# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-teacher imitation settings for VelStand."""

ANCHOR_YAW_REMAP = {
    "lin": (0.0, 0.1, 0.2, 0.3, 0.4),
    "targets": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    "pos": (
        (0.75, 0.89, 1.0, 1.06, 1.11, 1.17, 1.25, 1.38, 1.67, 1.95),
        (0.8, 0.85, 0.89, 0.94, 0.99, 1.12, 1.27, 1.39, 1.67, 1.87),
        (0.28, 0.45, 0.55, 0.68, 0.86, 1.06, 1.26, 1.44, 1.55, 1.85),
        (0.21, 0.32, 0.47, 0.68, 0.86, 1.05, 1.23, 1.39, 1.51, 1.77),
        (0.19, 0.39, 0.62, 0.72, 0.82, 0.97, 1.25, 1.37, 1.51, 1.72),
    ),
    "neg": (
        (0.9, 1.01, 1.07, 1.13, 1.19, 1.31, 1.46, 1.7, 1.94, 2.16),
        (0.71, 0.81, 0.89, 0.96, 1.08, 1.29, 1.49, 1.67, 1.88, 2.08),
        (0.25, 0.49, 0.67, 0.8, 1.04, 1.17, 1.41, 1.58, 1.81, 1.9),
        (0.23, 0.38, 0.6, 0.77, 0.92, 1.1, 1.33, 1.49, 1.62, 1.83),
        (0.17, 0.3, 0.44, 0.64, 0.85, 1.02, 1.15, 1.42, 1.6, 1.73),
    ),
}


def default_bc_cfg() -> dict:
    """Return independent mutable settings for the source two-teacher pass."""
    return {
        "coef": 1.0,
        "learning_rate": 3e-4,
        "gate_tilt_deg": 35.0,
        "epochs": 5,
        "mini_batches": 4,
        "min_mini_batch": 512,
        "min_samples": 64,
        "gravity_slice": (3, 6),
        "twist_slice": (48, 51),
        "anchor_coef": 1.0,
        "anchor_tilt_deg": 25.0,
        "body_slice": (55, 61),
        "unanchor_turn_in_place": False,
        "anchor_yaw_remap": ANCHOR_YAW_REMAP,
        "body_active_axes": (2, 3, 4),
        "body_no_teacher_axes": (2,),
        "expert_mask_cols": tuple(range(6, 48)),
        "expert_mask_tilt_deg": 75.0,
    }
