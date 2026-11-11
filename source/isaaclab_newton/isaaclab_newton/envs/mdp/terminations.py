# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms that read the Newton physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def solver_fault(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the environments whose solver produced a body pose or velocity that is not finite.

    See :meth:`~isaaclab_newton.physics.NewtonManager.get_non_finite_worlds`. Resetting such an environment
    restores a valid state while the other environments keep running.
    """
    return wp.to_torch(env.sim.backend_manager.get_non_finite_worlds())
