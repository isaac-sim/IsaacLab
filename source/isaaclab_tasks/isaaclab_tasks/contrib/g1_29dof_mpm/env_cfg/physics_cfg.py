# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled MJWarp + implicit-MPM physics for the G1 granular locomotion task.

The rigid entry integrates the articulation with MJWarp, the MPM entry integrates the granular
bed, and a lagged proxy mapping hands the feet to the MPM solver as colliders. Only the ankle
roll links are proxied: they are the sole bodies expected to touch the bed, and every proxied
body costs an extra collider in the MPM solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from .scene_cfg import MPM_VOXEL_SIZE

if TYPE_CHECKING:
    from ..g1_mpm_env_cfg import G1MPMEnvCfg

RIGID_ENTRY = "robot"
"""Name of the MJWarp coupler entry."""

MPM_ENTRY = "sand"
"""Name of the implicit-MPM coupler entry."""

FOOT_PROXY_BODIES = [r"/World/envs/env_.*/Robot/.*ankle_roll_link"]
"""Rigid bodies handed to the MPM solver as colliders."""

DEFAULT_PROXY_MASS_SCALE = 1.0
"""Scale applied to the foot proxy masses in the MPM solver."""

SPARSE_MPM_MIN_LOWER_NODES_PER_WORLD = 1 << 6
SPARSE_MPM_MIN_UPPER_NODES_PER_WORLD = 1
SPARSE_MPM_MIN_TOTAL_UPPER_NODE_COUNT = 1 << 5


def g1_mpm_physics_cfg(proxy_mass_scale: float = DEFAULT_PROXY_MASS_SCALE, *, rough_terrain: bool = False) -> NewtonCfg:
    """Build the coupled MJWarp/MPM physics configuration.

    Args:
        proxy_mass_scale: Effective-mass scale of the foot proxies seen by the MPM solver.
        rough_terrain: Select the terrain spawner's MPM support body instead of flat retaining boxes.

    Returns:
        The Newton physics configuration.
    """
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name=RIGID_ENTRY,
                    solver_cfg=MJWarpSolverCfg(
                        use_mujoco_contacts=False,
                        integrator="implicitfast",
                        cone="pyramidal",
                        impratio=1.0,
                        nconmax=300,
                        njmax=1000,
                        iterations=100,
                    ),
                    bodies=[r"/World/envs/env_.*/Robot"],
                    # picks up the hidden static pan floor, which has no rigid body to name
                    include_static_shapes=True,
                    # Refine the articulation integration between coupled exchanges.
                    substeps=2,
                ),
                CouplerEntryCfg(
                    name=MPM_ENTRY,
                    solver_cfg=MPMSolverCfg(
                        voxel_size=MPM_VOXEL_SIZE,
                        grid_type="sparse",
                        grid_padding=0,
                        strain_basis="P0",
                        transfer_scheme="apic",
                        max_iterations=25,
                        tolerance=1.0e-4,
                        warmstart_mode="auto",
                        velocity_basis="Q1",
                        collider_basis="pic27",
                        collider_velocity_mode="forward",
                        solver="auto",
                        critical_fraction=0.0,
                        separate_worlds=True,
                        project_outside_colliders=False,
                    ),
                    bodies=[r"/World/envs/env_.*/Sand/MPMBedFloor"]
                    if rough_terrain
                    else [
                        r"/World/envs/env_.*/MPMBedFloor",
                        r"/World/envs/env_.*/MPMBedWallFront",
                        r"/World/envs/env_.*/MPMApproachBank",
                        r"/World/envs/env_.*/MPMBedWallLeft",
                        r"/World/envs/env_.*/MPMBedWallRight",
                    ],
                    all_particles=True,
                    include_static_shapes=False,
                    include_child_joints=False,
                    substeps=1,
                    in_place=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source=RIGID_ENTRY,
                    destination=MPM_ENTRY,
                    bodies=FOOT_PROXY_BODIES,  # type: ignore
                    mode="lagged",
                    mass_scale=proxy_mass_scale,
                    collision_pipeline=None,
                )
            ],
            iterations=1,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        # Newton's default shape stiffness (ke 2.5e3, kd 1e2) is two orders of magnitude too
        # compliant for a 32 kg humanoid: `MJWarpSolverCfg.use_mujoco_contacts=False` routes it
        # through `convert_solref`, so the rigid approach platform answers a footfall like a
        # mattress. These are the values the rigid and soft-contact G1 tasks walk on.
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        num_substeps=1,
        use_cuda_graph=True,
    )


def get_mpm_solver_cfg(cfg: G1MPMEnvCfg) -> MPMSolverCfg:
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


def configure_sparse_mpm_capacities(cfg: G1MPMEnvCfg) -> None:
    """Size sparse storage from the final particle grid and world count.

    With no explicit cell limit, reserve one cell per particle, with leaf capacity
    as a minimum. The unpadded index grid inserts one voxel per particle and
    deduplicates occupied voxels, so this bounds cell use even if particles separate.

    Command-line ``--num_envs`` overrides are applied after config construction, so the
    environment calls this once more just before simulation creation.

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
        # Match MPMGridCfg sampling, including optional boundary endpoints.
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
