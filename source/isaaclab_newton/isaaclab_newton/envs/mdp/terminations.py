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
    """Terminate worlds with nonfinite Newton state or persistent solver buffers."""
    return wp.to_torch(env.sim.backend_manager.get_non_finite_worlds())
