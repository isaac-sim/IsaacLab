# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of Newton's Steel solver."""

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg


@configclass
class SteelSolverCfg(NewtonSolverCfg):
    class_type: type | str = "{DIR}.steel_solver:SteelSolver"

    solver_type: str = "steel"

    solve_mode: Literal["matrix_free", "split"] = "matrix_free"
    """Constraint solve; ``matrix_free`` requires CUDA."""

    iterations: int = 8
    """PGS sweeps per simulation step."""

    baumgarte: float = 0.05
    """Fraction of contact and limit position error corrected per step."""

    update_mass_matrix_interval: int = 1
    """Simulation steps between mass-matrix factorizations."""

    use_parallel_streams: bool = False
    """Run each articulation-size group on its own CUDA stream; also enables the fused mass paths."""

    enable_joint_limits: bool = False
    """Whether to enforce joint position limits."""

    joint_limit_activation_gap: float = 0.2
    """Distance from a joint limit at which to allocate its row [m or rad, depending on joint type]."""

    enable_joint_velocity_limits: bool = False
    """Whether to enforce joint velocity limits."""

    velocity_limit_activation_fraction: float = 0.7
    """Fraction of a velocity limit at which to allocate its row."""

    articulation_constraint_max: int = 32
    """Maximum articulated constraint rows per world."""

    free_body_constraint_max: int = 512
    """Maximum free-body constraint rows per world."""

    friction_anchor_beta: float = 0.0
    """Zero selects point friction instead of persistent friction patches."""

    contact_friction_gap_max: float | None = None
    """Largest contact gap for friction rows [m]; ``None`` uses Newton's default."""

    contact_friction_position_iterations: int = -1
    """Position iterations used for contact friction anchoring."""

    filter_fixed_root_contacts: bool = False
    """Drop contact pairs between static shapes and bodies welded to the world by a zero-DOF root joint."""

    contact_torsion: bool = True
    """Whether to construct contact torsion rows from the authored torsion radius."""

    contact_torsion_device: bool = False
    """Build contact torsion rows on CUDA instead of the capacity-limited host path."""

    parallel_tree: bool = False
    """Traverse independent branches of each articulation tree in parallel; measure before enabling."""
