# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory: Newton-specific control helpers.

Backend-agnostic OSC kinematics + dynamics (Jacobian, mass matrix) now flow
through :class:`BaseArticulationData` accessors added in PR #5400
(``body_link_jacobian_w``, ``mass_matrix``); the only Factory-specific Newton
glue still needed is the per-joint effort clamp, since Newton's direct
``joint_f`` write does not enforce ``effort_limit_sim`` the way PhysX's
articulation drive does.
"""

from __future__ import annotations

import torch

# Franka FR3 per-joint torque limits [N·m] (datasheet symmetric).
FRANKA_FR3_EFFORT_LIMITS: tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)


def clamp_to_effort_limits(
    dof_torque: torch.Tensor, limits: tuple[float, ...] = FRANKA_FR3_EFFORT_LIMITS
) -> torch.Tensor:
    """Per-joint elementwise torque clamp [N·m].

    PhysX's articulation drive enforces ``effort_limit_sim`` automatically.
    Newton does not on direct ``joint_f`` writes, so the Newton path applies
    the clamp explicitly here. The first ``len(limits)`` columns of
    ``dof_torque`` are clamped in-place against the symmetric limits;
    remaining columns (e.g. gripper DOFs) are left untouched.

    Args:
        dof_torque: ``(num_envs, num_dofs)`` torch tensor on any device.
        limits: Per-DOF symmetric clamp values, one per arm DOF. Defaults
            to :data:`FRANKA_FR3_EFFORT_LIMITS`.

    Returns:
        The same ``dof_torque`` tensor with arm columns clamped in-place.
    """
    n = len(limits)
    lim = torch.as_tensor(limits, device=dof_torque.device, dtype=dof_torque.dtype)
    dof_torque[..., :n] = torch.clamp(dof_torque[..., :n], min=-lim, max=lim)
    return dof_torque
