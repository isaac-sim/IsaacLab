# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none


"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
from isaaclab_newton.assets import RigidObject
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, VBDSolverCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_contrib.custom_coupling import CoupledMJWarpVBDSolverCfg
from isaaclab_contrib.deformable import DeformableObject


def _make_coupled_cfg(coupling_mode: str) -> SimulationCfg:
    """Create a simulation config for the manual coupling manager."""
    solver_cfg = CoupledMJWarpVBDSolverCfg(
        rigid_solver_cfg=MJWarpSolverCfg(
            njmax=40,
            nconmax=20,
            ls_iterations=20,
            integrator="implicitfast",
        ),
        soft_solver_cfg=VBDSolverCfg(
            iterations=3,
            integrate_with_external_rigid_solver=True,
        ),
        coupling_mode=coupling_mode,
    )
    return SimulationCfg(
        dt=1.0 / 60.0,
        physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=5, use_cuda_graph=True),
    )


def _coupled_sim_context(cfg: SimulationCfg, device="cuda:0"):
    """Helper to create a coupled solver simulation context."""
    cfg.device = device
    return build_simulation_context(device=device, sim_cfg=cfg, auto_add_lighting=True)


def generate_lateral_rigid_and_deformable_cubes(
    rigid_cube_pos: tuple = (0.0, 0.0, 1.0),
    deformable_cube_pos: tuple = (-0.16, 0.0, 1.0),
) -> tuple[RigidObject, DeformableObject]:
    """Create rigid and deformable cubes for lateral contact."""
    sim_utils.create_prim("/World/env_0", "Xform", translation=(0.0, 0.0, 0.0))

    rigid_cube_cfg = RigidObjectCfg(
        prim_path="/World/env_[^/]+/rigid_cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.05),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=rigid_cube_pos),
    )
    deformable_cube_cfg = DeformableObjectCfg(
        prim_path="/World/env_[^/]+/deformable_cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.08, 0.08, 0.08),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=1000.0,
                k_mu=1e5,
                k_lambda=1e5,
                particle_radius=0.005,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=deformable_cube_pos),
    )
    plan = clone_plan_from_env_0(
        CloneCfg(clone_template="/World/env_{}"), (rigid_cube_cfg, deformable_cube_cfg), 1, 0.0
    )
    rigid_cube, deformable_cube = RigidObject(rigid_cube_cfg), DeformableObject(deformable_cube_cfg)
    replicate(plan)
    return rigid_cube, deformable_cube


def _run_lateral_rigid_cube_response(coupling_mode: str) -> tuple[float, float]:
    """Run a compact lateral contact scene and return the rigid and deformable cube X displacements."""
    with _coupled_sim_context(_make_coupled_cfg(coupling_mode)) as sim:
        sim._app_control_on_stop_handle = None
        rigid_cube, deformable_cube = generate_lateral_rigid_and_deformable_cubes()
        sim.reset()

        initial_rigid_x = rigid_cube.data.root_pos_w.torch[0, 0].item()
        initial_deformable_x = deformable_cube.data.root_pos_w.torch[0, 0].item()
        nodal_vel = torch.zeros_like(deformable_cube.data.nodal_vel_w.torch)
        nodal_vel[..., 0] = 2.0
        deformable_cube.write_nodal_velocity_to_sim_index(nodal_vel)

        for _ in range(60):
            sim.step()
            rigid_cube.update(sim.cfg.dt)
            deformable_cube.update(sim.cfg.dt)

        return (
            rigid_cube.data.root_pos_w.torch[0, 0].item() - initial_rigid_x,
            deformable_cube.data.root_pos_w.torch[0, 0].item() - initial_deformable_x,
        )


def test_two_way_coupling_applies_reaction_to_rigid_body():
    """Test that two-way coupling laterally pushes a rigid body, while rigid contact blocks the deformable."""
    one_way_dx, one_way_deformable_dx = _run_lateral_rigid_cube_response("one_way")
    two_way_dx, _ = _run_lateral_rigid_cube_response("two_way")

    assert abs(one_way_dx) < 1e-2
    # Rigid-to-deformable contact stops the deformable (about 2 m of free flight otherwise).
    assert one_way_deformable_dx < 0.1
    assert two_way_dx > one_way_dx + 1e-2
