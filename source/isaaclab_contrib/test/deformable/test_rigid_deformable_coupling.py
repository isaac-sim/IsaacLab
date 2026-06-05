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

import pytest
import torch
from isaaclab_newton.assets import Articulation, RigidObject
from isaaclab_newton.physics import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_contrib.deformable import (
    CoupledFeatherstoneVBDSolverCfg,
    CoupledMJWarpVBDSolverCfg,
    DeformableObject,
    VBDSolverCfg,
)

from isaaclab_assets import FRANKA_PANDA_CFG  # isort:skip


def _make_coupled_cfg(coupling_mode: str, rigid_solver: str = "mjwarp") -> SimulationCfg:
    """Create a simulation config for a coupled rigid-deformable solver."""
    if rigid_solver == "mjwarp":
        solver_cfg = CoupledMJWarpVBDSolverCfg(
            rigid_solver_cfg=MJWarpSolverCfg(
                njmax=40,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            soft_solver_cfg=VBDSolverCfg(
                iterations=3,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            coupling_mode=coupling_mode,
        )
    elif rigid_solver == "featherstone":
        solver_cfg = CoupledFeatherstoneVBDSolverCfg(
            rigid_solver_cfg=FeatherstoneSolverCfg(),
            soft_solver_cfg=VBDSolverCfg(
                iterations=3,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            coupling_mode=coupling_mode,
        )
    else:
        raise ValueError(f"Unknown rigid solver: {rigid_solver}")

    return SimulationCfg(
        dt=1.0 / 60.0,
        physics=NewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=5,
            use_cuda_graph=True,
        ),
    )


def _coupled_sim_context(cfg: SimulationCfg, device="cuda:0"):
    """Helper to create a coupled solver simulation context."""
    cfg.device = device
    return build_simulation_context(device=device, sim_cfg=cfg, auto_add_lighting=True)


@pytest.fixture
def sim(request):
    """Create a coupled solver simulation context.

    Defaults to one-way coupling. Tests can parametrize this fixture with
    ``"two_way"`` when both coupling paths should be exercised.
    """
    param = getattr(request, "param", "one_way")
    if isinstance(param, tuple):
        rigid_solver, coupling_mode = param
    else:
        rigid_solver, coupling_mode = "mjwarp", param
    with _coupled_sim_context(_make_coupled_cfg(coupling_mode, rigid_solver)) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


def generate_robot_and_two_cubes(
    colliding_cube_pos: tuple = (0.3, 0.0, 1.0),
    free_cube_pos: tuple = (2.0, 0.0, 1.0),
) -> tuple[Articulation, DeformableObject, DeformableObject]:
    """Generate a scene with one Franka robot and two deformable cubes.

    A single env contains a robot and two cube objects at different positions.
    One cube is placed above the robot arm (will collide), the other is placed
    far away (falls freely).

    Args:
        colliding_cube_pos: Position of the cube above the robot arm.
        free_cube_pos: Position of the cube that falls freely.

    Returns:
        Tuple of (robot, colliding_cube, free_cube).
    """
    sim_utils.create_prim("/World/env_0", "Xform", translation=(0.0, 0.0, 0.0))

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/env_.*/Robot")
    robot = Articulation(robot_cfg)

    colliding_cube = DeformableObject(
        cfg=DeformableObjectCfg(
            prim_path="/World/env_.*/cube_collide",
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.05, 0.05, 0.05),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
                physics_material=NewtonDeformableBodyMaterialCfg(
                    density=500.0,
                    k_mu=1e5,
                    k_lambda=1e5,
                    particle_radius=0.005,
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=colliding_cube_pos),
        )
    )

    free_cube = DeformableObject(
        cfg=DeformableObjectCfg(
            prim_path="/World/env_.*/cube_free",
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.05, 0.05, 0.05),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
                physics_material=NewtonDeformableBodyMaterialCfg(
                    density=500.0,
                    k_mu=1e4,
                    k_lambda=1e4,
                    particle_radius=0.005,
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=free_cube_pos),
        )
    )

    return robot, colliding_cube, free_cube


def generate_lateral_rigid_and_deformable_cubes(
    rigid_cube_pos: tuple = (0.0, 0.0, 1.0),
    deformable_cube_pos: tuple = (-0.16, 0.0, 1.0),
) -> tuple[RigidObject, DeformableObject]:
    """Generate rigid and deformable cubes arranged for lateral contact.

    Args:
        rigid_cube_pos: Initial position of the rigid cube.
        deformable_cube_pos: Initial position of the deformable cube.

    Returns:
        Tuple of (rigid cube, deformable cube).
    """
    sim_utils.create_prim("/World/env_0", "Xform", translation=(0.0, 0.0, 0.0))

    rigid_cube = RigidObject(
        cfg=RigidObjectCfg(
            prim_path="/World/env_.*/rigid_cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=rigid_cube_pos),
        )
    )

    deformable_cube = DeformableObject(
        cfg=DeformableObjectCfg(
            prim_path="/World/env_.*/deformable_cube",
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
    )

    return rigid_cube, deformable_cube


@pytest.mark.parametrize(
    "sim",
    [("featherstone", "kinematic")],
    indirect=True,
    ids=["featherstone_kinematic"],
)
def test_smoke_featherstone_kinematic(sim):
    """Smoke test: Featherstone + VBD kinematic coupling initializes and steps."""
    robot, colliding_cube, free_cube = generate_robot_and_two_cubes()
    sim.reset()

    assert robot.is_initialized
    assert colliding_cube.is_initialized
    assert free_cube.is_initialized

    initial_z_collide = colliding_cube.data.root_pos_w.torch[0, 2].item()
    initial_z_free = free_cube.data.root_pos_w.torch[0, 2].item()

    for _ in range(10):
        sim.step()
        robot.update(sim.cfg.dt)
        colliding_cube.update(sim.cfg.dt)
        free_cube.update(sim.cfg.dt)

    assert colliding_cube.data.root_pos_w.torch[0, 2].item() < initial_z_collide - 0.01
    assert free_cube.data.root_pos_w.torch[0, 2].item() < initial_z_free - 0.01


def _run_lateral_rigid_cube_response(coupling_mode: str) -> float:
    """Run a compact lateral contact scene and return rigid cube X displacement."""
    with _coupled_sim_context(_make_coupled_cfg(coupling_mode)) as sim:
        sim._app_control_on_stop_handle = None
        rigid_cube, deformable_cube = generate_lateral_rigid_and_deformable_cubes()
        sim.reset()

        initial_rigid_x = rigid_cube.data.root_pos_w.torch[0, 0].item()
        nodal_vel = torch.zeros_like(deformable_cube.data.nodal_vel_w.torch)
        nodal_vel[..., 0] = 2.0
        deformable_cube.write_nodal_velocity_to_sim_index(nodal_vel)

        for _ in range(60):
            sim.step()
            rigid_cube.update(sim.cfg.dt)
            deformable_cube.update(sim.cfg.dt)

        return rigid_cube.data.root_pos_w.torch[0, 0].item() - initial_rigid_x


def test_two_way_coupling_applies_reaction_to_rigid_body():
    """Test that two-way coupling laterally pushes a rigid body."""
    one_way_dx = _run_lateral_rigid_cube_response("one_way")
    two_way_dx = _run_lateral_rigid_cube_response("two_way")

    assert abs(one_way_dx) < 1e-2
    assert two_way_dx > one_way_dx + 1e-2


def test_deformable_deflected_by_rigid_contact(sim):
    """Test that a cube falling onto the robot is deflected horizontally.

    Two cubes start at the same height (Z=1.0m). Cube 0 (env 0) is placed
    above the robot arm at X=0.3m. Cube 1 (env 1) is shifted +2m in X away
    from the robot so it falls freely.

    Expected timeline (dt=1/60s):

    - Steps 0-15: Both cubes in free-fall, identical Z trajectories.
      No horizontal motion. Z drops from ~1.0m to ~0.65m.
    - Step ~20: Cube 0 hits the robot arm at Z~0.54m. Contact deflects it
      horizontally (X starts increasing). Cube 1 is still in free-fall.
    - Step ~28: Cube 1 reaches the ground (Z~0.005m) and stops. Zero
      horizontal displacement.
    - Steps 20-40: Cube 0 continues falling while sliding off the arm,
      gaining horizontal velocity. Reaches the ground around step 40.
    - Steps 40-70: Cube 0 slides along the ground, decelerating due to
      friction. Settles around X~1.0m.
    - Steps 70+: Both cubes at rest on the ground.

    The test asserts that cube 0 has a significantly larger horizontal
    displacement than cube 1 (which should be ~zero).
    """
    robot, colliding_cube, free_cube = generate_robot_and_two_cubes(
        colliding_cube_pos=(0.3, 0.0, 1.0),  # above robot arm
        free_cube_pos=(2.0, 0.0, 1.0),  # far from robot, same height
    )
    sim.reset()

    initial_xy_collide = colliding_cube.data.root_pos_w.torch[0, :2].clone()
    initial_xy_free = free_cube.data.root_pos_w.torch[0, :2].clone()

    # Free-fall from 1.0m takes sqrt(2*1.0/9.81) ~ 0.45s ~ 27 steps at dt=1/60.
    # Collision with the robot arm happens around step 20 (Z~0.5m).
    # 120 steps (2s) gives ample time for collision, bounce, and settling.
    for _ in range(120):
        sim.step()
        robot.update(sim.cfg.dt)
        colliding_cube.update(sim.cfg.dt)
        free_cube.update(sim.cfg.dt)

    final_xy_collide = colliding_cube.data.root_pos_w.torch[0, :2]
    final_xy_free = free_cube.data.root_pos_w.torch[0, :2]

    displacement_collide = (final_xy_collide - initial_xy_collide).norm().item()
    displacement_free = (final_xy_free - initial_xy_free).norm().item()

    # Colliding cube should be deflected; free cube should fall straight
    assert displacement_collide > displacement_free + 0.01, (
        f"Colliding cube should be deflected more than free cube: "
        f"collide={displacement_collide:.4f}, free={displacement_free:.4f}"
    )
