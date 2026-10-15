# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse MPM capacity settings derived from the final environment configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.physics import MPMSolverCfg

from ..env_cfg.physics_cfg import MPM_ENTRY

if TYPE_CHECKING:
    from ..mpm_env_cfg import G1MovingPatchEnvCfg

SPARSE_MPM_MIN_LOWER_NODES_PER_WORLD = 1 << 6
SPARSE_MPM_MIN_UPPER_NODES_PER_WORLD = 1
SPARSE_MPM_MIN_TOTAL_UPPER_NODE_COUNT = 1 << 5


def get_mpm_solver_cfg(cfg: G1MovingPatchEnvCfg) -> MPMSolverCfg:
    """Return the MPM solver configuration held by the coupled physics configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        The MPM solver configuration.

    Raises:
        ValueError: If the physics configuration does not hold exactly one MPM entry.
    """
    assert cfg.sim.physics is not None
    entries = [entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == MPM_ENTRY]  # type: ignore
    if len(entries) != 1 or not isinstance(entries[0].solver_cfg, MPMSolverCfg):
        raise ValueError(f"Expected one {MPM_ENTRY!r} MPMSolverCfg entry, found {len(entries)}.")
    return entries[0].solver_cfg


def configure_sparse_mpm_capacities(cfg: G1MovingPatchEnvCfg) -> None:
    """Size sparse storage from the final particle grid and world count.

    With no explicit cell limit, reserve one cell per particle. The unpadded
    index grid inserts one voxel per particle and deduplicates occupied voxels,
    so this bounds cell use even if the particles separate. Node reservations
    remain explicit; leaf capacity is a minimum for the cell reservation.

    The environment config calls this during validation, after final terrain sampling
    and command-line ``--num_envs`` overrides are available.

    Args:
        cfg: Environment configuration; its MPM solver capacities are updated in place.

    Raises:
        ValueError: If a per-world capacity is not a positive integer or the capacity hierarchy
            ``upper <= lower <= leaf <= active`` is violated.
    """
    per_world = {
        "active cells": cfg.mpm_active_cell_count_per_world,
        "leaf nodes": cfg.mpm_leaf_node_count_per_world,
        "lower nodes": cfg.mpm_lower_node_count_per_world,
        "upper nodes": cfg.mpm_upper_node_count_per_world,
    }
    for name, capacity in per_world.items():
        if name == "active cells" and capacity is None:
            continue
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"Sparse MPM {name} capacity per world must be a positive integer, got {capacity!r}.")
    if per_world["active cells"] is None:
        spawn = cfg.scene.sand.spawn
        # Match IsaacLab MPMGridCfg sampling, including optional boundary endpoints.
        extent = np.asarray(spawn.upper) - np.asarray(spawn.lower)
        dimensions = np.maximum(np.ceil(spawn.particles_per_cell * extent / spawn.voxel_size), 1).astype(np.int64)
        if spawn.particle_placement == "boundary":
            dimensions += 1
        per_world["active cells"] = max(int(np.prod(dimensions)), per_world["leaf nodes"])

    if not (
        per_world["upper nodes"] <= per_world["lower nodes"] <= per_world["leaf nodes"] <= per_world["active cells"]
    ):
        raise ValueError(
            "Sparse MPM per-world capacity hierarchy must satisfy "
            "upper nodes <= lower nodes <= leaf nodes <= active cells."
        )
    if per_world["lower nodes"] < SPARSE_MPM_MIN_LOWER_NODES_PER_WORLD:
        raise ValueError(
            "G1 MPM sparse lower-node capacity is below the runtime-topology-derived safety floor of "
            f"{SPARSE_MPM_MIN_LOWER_NODES_PER_WORLD} nodes per world."
        )
    if per_world["upper nodes"] < SPARSE_MPM_MIN_UPPER_NODES_PER_WORLD:
        raise ValueError(
            "G1 MPM sparse upper-node capacity is below the runtime-topology-derived safety floor of "
            f"{SPARSE_MPM_MIN_UPPER_NODES_PER_WORLD} nodes per world."
        )

    world_count = max(1, int(cfg.scene.num_envs))
    solver_cfg = get_mpm_solver_cfg(cfg)
    solver_cfg.max_active_cell_count = per_world["active cells"] * world_count
    solver_cfg.max_leaf_node_count = per_world["leaf nodes"] * world_count
    solver_cfg.max_lower_node_count = per_world["lower nodes"] * world_count
    solver_cfg.max_upper_node_count = max(
        SPARSE_MPM_MIN_TOTAL_UPPER_NODE_COUNT,
        per_world["upper nodes"] * world_count,
    )
