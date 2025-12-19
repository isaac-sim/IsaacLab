# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .solvers_cfg import MJWarpSolverCfg, NewtonSolverCfg


@configclass
class NewtonCfg:
    """Configuration for Newton-related parameters.

    These parameters are used to configure the Newton physics simulation.
    """

    num_substeps: int = 1
    """Number of substeps to use for the solver."""

    debug_mode: bool = False
    """Whether to enable debug mode for the solver."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graphing when simulating.

    If set to False, the simulation performance will be severely degraded.
    """

    newton_viewer_update_frequency: int = 1
    """Frequency of updates to the Newton viewer."""

    newton_viewer_camera_pos: tuple[float, float, float] = (10.0, 0.0, 3.0)
    """Position of the camera in the Newton viewer."""

    visualizer_train_mode: bool = True
    """Whether the visualizer is in training mode (True) or play mode (False)."""

    solver_cfg: NewtonSolverCfg = MJWarpSolverCfg()
