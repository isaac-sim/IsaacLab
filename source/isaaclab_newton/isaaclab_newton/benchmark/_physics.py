# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact Newton solver selection for micro-benchmark entrypoints."""

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg


def create_microbenchmark_physics_cfg(physics_variant: str) -> NewtonCfg:
    """Create the exact Newton physics configuration selected by the CLI.

    Args:
        physics_variant: Exact Newton solver variant.

    Returns:
        Newton configuration for the selected solver.

    Raises:
        ValueError: If the exact variant is unsupported.
    """
    solver_types = {
        "newton_mjwarp": MJWarpSolverCfg,
        "newton_kamino": KaminoSolverCfg,
    }
    try:
        solver_type = solver_types[physics_variant]
    except KeyError as exc:
        available = ", ".join(solver_types)
        raise ValueError(f"Unsupported Newton physics variant '{physics_variant}'. Available: {available}.") from exc
    return NewtonCfg(solver_cfg=solver_type())
