# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory: per-joint effort clamp for the Newton backend.

PR #5400's ``BaseArticulationData`` accessors cover OSC Jacobian + mass
matrix; only the effort clamp stays here because Newton's ``joint_f``
write doesn't honour ``effort_limit_sim`` like PhysX's drive does.
"""

from __future__ import annotations

import torch

# Franka FR3 per-joint torque limits [N·m] (datasheet, symmetric).
FRANKA_FR3_EFFORT_LIMITS: tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)


def clamp_to_effort_limits(
    dof_torque: torch.Tensor, limits: tuple[float, ...] = FRANKA_FR3_EFFORT_LIMITS
) -> torch.Tensor:
    """Clamp arm-DOF torques in-place; trailing DOFs (e.g. gripper) untouched."""
    n = len(limits)
    lim = torch.as_tensor(limits, device=dof_torque.device, dtype=dof_torque.dtype)
    dof_torque[..., :n] = torch.clamp(dof_torque[..., :n], min=-lim, max=lim)
    return dof_torque
