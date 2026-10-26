# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of learned generalized-coordinate Newton dynamics."""

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg


@configclass
class NeRDSolverCfg(NewtonSolverCfg):
    """Advance one physics step from applied joint efforts with a native NeRD checkpoint.

    Use one physics substep. Policy decimation repeats actuator evaluation and learned physics;
    collection records each physics step separately. The checkpoint declares the frame duration,
    modeled joints and contact features. Actuator commands, state and effort limits remain owned
    by the actuator. Solver-side motor drives are unsupported.
    """

    class_type: str = "{DIR}.nerd_manager:NewtonNeRDManager"
    solver_type: str = "nerd"
    checkpoint: str = ""
    """Native checkpoint written by ``scripts/learning/nerd/train.py``."""

    compile: bool = True
    """Compile model inference before the native manager captures its physics CUDA graph."""
