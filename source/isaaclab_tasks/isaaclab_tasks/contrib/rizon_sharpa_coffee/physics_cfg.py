# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coffee coupling configuration, importable before simulator startup."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg


@configclass
class CoffeeMPMSolverCfg(MPMSolverCfg):
    """Native MPM with optional passive rendering grains, disabled for training."""

    class_type: str = "isaaclab_tasks.contrib.rizon_sharpa_coffee.physics:CoffeeMPMManager"
    solver: str | tuple[str, ...] | None = None
    """Pressure solver; ``None`` selects the final wet or dry task profile."""
    max_iterations: int | None = None
    """Pressure budget; ``None`` selects the final wet or dry task profile."""
    grains_per_particle: int = 0
    """Number of visual samples per physical particle; zero disables their allocation."""


@configclass
class CoffeeCouplerCfg(CouplerProxyCfg):
    """Task-local MJWarp/MPM coupling with mounted and contact-grasp modes."""

    class_type: type | str = "{DIR}.physics:NewtonCoffeeCouplerManager"


def make_grasp_solver_cfg() -> MJWarpSolverCfg:
    """Rigid solver profile shared by the free-grasp screen and coffee scene."""
    return MJWarpSolverCfg(
        use_mujoco_contacts=True,
        integrator="implicitfast",
        cone="elliptic",
        impratio=20.0,
        enable_multiccd=True,
        iterations=100,
        ls_iterations=30,
        njmax=512,
        nconmax=128,
    )


def configure_contact_grasp(physics: NewtonCfg, *, rigid_substeps: int = 2, mass_scale: float = 20.0) -> None:
    """Give the free spoon to MJWarp and return MPM reaction impulses to it.

    Args:
        physics: Task coupling configuration, modified before model creation.
        rigid_substeps: Rigid integration steps per coupling tick.
        mass_scale: Finite MPM proxy inertia multiplier; the rigid mass is unchanged.
    """
    coupler = physics.solver_cfg
    rigid, media = coupler.entries
    rigid.solver_cfg = make_grasp_solver_cfg()
    rigid.bodies = [r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/Spoon"]
    rigid.substeps = rigid_substeps
    media.bodies = []
    coupler.proxies = [
        CouplerProxyMappingCfg(
            source="robot",
            destination="media",
            bodies=[r"/World/envs/env_.*/Spoon"],
            # MPM's free-spoon inertia omits support from the finger contacts.
            # A finite virtual inertia prevents the particle solve from yielding
            # through the held tool. The source spoon remains 25 g and receives
            # the reaction impulses; opening the hand still releases it.
            mass_scale=mass_scale,
            mode="staggered",
            collision_pipeline=None,
        )
    ]


def make_physics_cfg(voxel_size: float, *, water: bool | None = False) -> NewtonCfg:
    """Configure coupling; ``water=None`` defers scene defaults until after overrides."""
    return NewtonCfg(
        solver_cfg=CoffeeCouplerCfg(
            entries=[
                CouplerEntryCfg(
                    name="robot",
                    solver_cfg=MJWarpSolverCfg(
                        use_mujoco_contacts=False, integrator="implicitfast", njmax=1024, nconmax=128
                    ),
                    bodies=[r"/World/envs/env_.*/Robot"],
                ),
                CouplerEntryCfg(
                    name="media",
                    solver_cfg=CoffeeMPMSolverCfg(
                        voxel_size=voxel_size,
                        # configure_scene reserves a particle-count-based budget
                        # unless the caller supplies a positive capacity.
                        max_active_cell_count=-1,
                        max_lower_node_count=256,
                        max_upper_node_count=32,
                        solver=None if water is None else "gs" if water else "auto",
                        # Tall liquid fills need a converged pressure solve;
                        # conserved particle mass alone does not preserve level.
                        max_iterations=None if water is None else 500 if water else 100,
                        # Resolve contacts at particle sites and pressure within
                        # each cell. Hard projection can permanently collapse
                        # powder onto the concave tool's surface.
                        collider_basis="pic27",
                        strain_basis="P1d",
                        collider_velocity_mode="backward",
                        # Native background-air regularization bounds nearly empty grid nodes.
                        air_drag=1.0,
                        separate_worlds=True,
                    ),
                    bodies=[r"/World/envs/env_.*/Spoon"],
                    all_particles=True,
                    include_static_shapes=True,
                    in_place=True,
                ),
            ],
            proxies=[],
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        num_substeps=1,
        use_cuda_graph=True,
    )
