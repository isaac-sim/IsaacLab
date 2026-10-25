# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of learned generalized-coordinate Newton dynamics."""

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg


@configclass
class NeRDSolverCfg(NewtonSolverCfg):
    """Advance one recorded control period with a native NeRD checkpoint.

    Use one physics substep and one environment decimation step. The checkpoint declares the
    frame duration, modeled joints, control type and contact features. Numerical teacher settings
    belong to the recording task, and do not change the learned frame duration.
    """

    class_type: str = "{DIR}.nerd_manager:NewtonNeRDManager"
    solver_type: str = "nerd"
    checkpoint: str = ""
    """Native checkpoint written by ``scripts/learning/nerd/train.py``."""

    compile: bool = True
    """Compile model inference before the native manager captures its physics CUDA graph."""
