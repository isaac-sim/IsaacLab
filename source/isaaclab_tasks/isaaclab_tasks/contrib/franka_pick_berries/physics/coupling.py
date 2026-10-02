# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled Newton physics: the Franka on MuJoCo-Warp and the berry tissue on explicit MPM
(:mod:`.grasp_explicit_mpm`, the default) or implicit MPM (:mod:`.grasp_implicit_mpm`)."""

import newton
import warp as wp
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonMPMManager,
)

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from . import grasp_explicit_mpm, grasp_implicit_mpm
from .grasp_explicit_mpm import GraspExplicitMPMSolverCfg, SolverGraspExplicitMPM, explicit_solver_config
from .grasp_implicit_mpm import GraspImplicitMPMSolverCfg, SolverGraspImplicitMPM

ARM_ENTRY = "arm"
TISSUE_ENTRY = "tissue"

CELLS_PER_BERRY = 4096
"""Active grid cells the implicit solver reserves per berry: room for the tissue, its halo and the gripper's
disturbance. CUDA graph capture needs a fixed capacity, and the cost of each step grows with it."""

# The MPM sees the finger links and the (kinematic) table as colliders; the rest of the arm never reaches the berry.
_FINGERS = r"/World/envs/env_.*/Robot/panda_(left|right)finger"
_FINGER = {side: rf"/World/envs/env_.*/Robot/panda_{side}finger" for side in ("left", "right")}
_TABLE = r"/World/envs/env_.*/Table"


def _arm_entry(substeps: int) -> CouplerEntryCfg:
    return CouplerEntryCfg(
        name=ARM_ENTRY,
        solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, integrator="implicitfast", njmax=512, nconmax=256),
        bodies=[r"/World/envs/env_.*/Robot", _TABLE],
        substeps=substeps,
    )


def berry_physics_cfg(solver: str = "explicit") -> NewtonCfg:
    """Return the coupled arm and tissue physics configuration; fit it to the berries with
    :func:`configure_tissue_solver`.

    Args:
        solver: ``"explicit"`` for the explicit MLS-MPM with frictional finger pads
            (:class:`.grasp_explicit_mpm.SolverGraspExplicitMPM`), or ``"implicit"`` for Newton's implicit MPM with
            clamped grasping (:class:`.grasp_implicit_mpm.SolverGraspImplicitMPM`).
    """
    if solver == "implicit":
        # One implicit solve per coupled substep (240 Hz); a single solve per 120 Hz physics step diverges.
        tissue = CouplerEntryCfg(
            name=TISSUE_ENTRY,
            solver_cfg=GraspImplicitMPMSolverCfg(gripping_bodies=_FINGERS),
            all_particles=True,
            include_static_shapes=True,
            substeps=1,
            in_place=True,
        )
        # The arm holds the fingers rigidly; at 1 the light proxies would be pushed away by the tissue and the pads
        # would feel soft. Any value from 100 up gives the same grip.
        proxies = CouplerProxyMappingCfg(
            source=ARM_ENTRY,
            destination=TISSUE_ENTRY,
            bodies=[_FINGERS, _TABLE],
            mass_scale=100.0,
            mode="staggered",
            collision_pipeline=None,
        )
        arm_substeps, coupled_substeps = 2, 2
    elif solver == "explicit":
        # The explicit solver substeps internally and exchanges the pad reaction once per 120 Hz physics step; it
        # handles the table analytically, so only the fingers are proxies.
        tissue = CouplerEntryCfg(
            name=TISSUE_ENTRY,
            solver_cfg=GraspExplicitMPMSolverCfg(),
            all_particles=True,
            substeps=1,
            in_place=True,
        )
        proxies = CouplerProxyMappingCfg(
            source=ARM_ENTRY, destination=TISSUE_ENTRY, bodies=[_FINGERS], mode="staggered", collision_pipeline=None
        )
        arm_substeps, coupled_substeps = 4, 1
    else:
        raise ValueError(f"Unknown tissue solver: {solver!r}")
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(entries=[_arm_entry(arm_substeps), tissue], proxies=[proxies], iterations=1),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        num_substeps=coupled_substeps,
        use_cuda_graph=True,
        # The task draws with its own viewer, which the default (None) does not detect.
        load_visual_shapes=True,
    )


def configure_tissue_solver(physics: NewtonCfg, specs, background: str) -> None:
    """Fit the tissue solver of a configuration from :func:`berry_physics_cfg` to the berries.

    The implicit solver gets the grid spacing [m] of the coarsest berry, which keeps at least the required particles
    per cell for all of them, and active-cell room for each; the explicit solver gets its grid, rates, contacts and
    fields.
    """
    for entry in physics.solver_cfg.entries:
        if entry.name == TISSUE_ENTRY:
            if isinstance(entry.solver_cfg, GraspExplicitMPMSolverCfg):
                entry.solver_cfg.solver_config = explicit_solver_config(specs, background, _FINGER)
            else:
                entry.solver_cfg.voxel_size = max(spec.voxel_size for spec in specs)
                entry.solver_cfg.max_active_cell_count = CELLS_PER_BERRY * len(specs)
            return
    raise ValueError(f"No {TISSUE_ENTRY!r} entry in the physics configuration")


def coupled_solver():
    """Return the coupled solver of the arm and tissue entries."""
    return NewtonManager._solver


def tissue_solver():
    """Return the tissue sub-solver with its own state and model, whose per-particle arrays are separate copies.

    The coupled solver steps a view of the model, so the tissue's stress, velocity gradient and material arrays live
    here and not on the manager's state and model.
    """
    coupled = coupled_solver()
    solver = coupled.solver(TISSUE_ENTRY)
    return solver, coupled.entry_state(TISSUE_ENTRY), solver.model


def bind_tissues(tissues) -> None:
    """Set each berry's material on the tissue solver, and read its damage from the solver.

    Each berry is its own velocity field of the explicit solver, in frictional contact with the others; the implicit
    solver shares one field.
    """
    solver = tissue_solver()[0]
    module = grasp_explicit_mpm if isinstance(solver, SolverGraspExplicitMPM) else grasp_implicit_mpm
    for field, tissue in enumerate(tissues):
        solver.set_tissue(
            tissue.particle_start, tissue.proxy["interface"], field, module.tissue_material(tissue.profile)
        )
        tissue.bind_solver(solver)


def set_grasping(grasping: wp.array) -> None:
    """Clamp the tissue that the fingers press to them, or release it, with the implicit solver.

    Args:
        grasping: One-element ``int32`` flag on the simulation device, nonzero to grasp. It is copied on the device,
            without synchronizing with the host.
    """
    solver = tissue_solver()[0]
    if isinstance(solver, SolverGraspImplicitMPM):
        wp.copy(solver.grasping, grasping)


def reset_tissue() -> None:
    """Reset the tissue solver's deformation and damage after the scene restored the particles."""
    if isinstance(tissue_solver()[0], SolverGraspExplicitMPM):
        # The coupled reset passes the restored particles to the entry, which restores its own solver state.
        NewtonManager._solver.reset(NewtonManager.get_state_0(), flags=newton.StateFlags.PARTICLE)
    else:
        # There is one environment, so an unmasked reset selects exactly that world.
        NewtonMPMManager.reset_solver_state(flags=newton.StateFlags.PARTICLE)
