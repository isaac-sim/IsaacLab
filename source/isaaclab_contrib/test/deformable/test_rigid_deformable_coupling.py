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
import warp as wp

from isaaclab_contrib.deformable import CoupledSolverCfg, DeformableObject, VBDSolverCfg, register_hooks
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_assets import FRANKA_PANDA_CFG  # isort:skip

COUPLED_ONE_WAY_CFG = SimulationCfg(
    dt=1.0 / 60.0,
    physics=NewtonCfg(
        solver_cfg=CoupledSolverCfg(
            rigid_solver_cfg=MJWarpSolverCfg(
                njmax=40,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                ls_parallel=False,
                integrator="implicitfast",
            ),
            vbd_cfg=VBDSolverCfg(
                iterations=3,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            soft_contact_margin=0.01,
            coupling_mode="one_way",
        ),
        num_substeps=5,
        use_cuda_graph=True,
    ),
)

COUPLED_TWO_WAY_CFG = SimulationCfg(
    dt=1.0 / 60.0,
    physics=NewtonCfg(
        solver_cfg=CoupledSolverCfg(
            rigid_solver_cfg=MJWarpSolverCfg(
                njmax=40,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                ls_parallel=False,
                integrator="implicitfast",
            ),
            vbd_cfg=VBDSolverCfg(
                iterations=3,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            soft_contact_margin=0.01,
            coupling_mode="two_way",
        ),
        num_substeps=5,
        use_cuda_graph=True,
    ),
)


def _coupled_sim_context(cfg: SimulationCfg, device="cuda:0"):
    """Helper to create a coupled solver simulation context."""
    cfg.device = device
    return build_simulation_context(device=device, sim_cfg=cfg, auto_add_lighting=True)


@pytest.fixture
def sim_one_way():
    """Create a one-way coupled solver simulation context."""
    register_hooks()
    with _coupled_sim_context(COUPLED_ONE_WAY_CFG) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.fixture
def sim_two_way():
    """Create a two-way coupled solver simulation context."""
    register_hooks()
    with _coupled_sim_context(COUPLED_TWO_WAY_CFG) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


def generate_robot_and_two_cubes(
    colliding_cube_pos: tuple = (0.3, 0.0, 1.0),
    free_cube_pos: tuple = (2.0, 0.0, 1.0),
    device: str = "cuda:0",
) -> tuple[Articulation, DeformableObject, DeformableObject]:
    """Generate a scene with one Franka robot and two deformable cubes.

    A single env contains a robot and two cube objects at different positions.
    One cube is placed above the robot arm (will collide), the other is placed
    far away (falls freely).

    Args:
        colliding_cube_pos: Position of the cube above the robot arm.
        free_cube_pos: Position of the cube that falls freely.
        device: Device to use.

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
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(
                    density=500.0,
                    youngs_modulus=2.5e5,
                    poissons_ratio=0.25,
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
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
                physics_material=sim_utils.DeformableBodyMaterialCfg(
                    density=500.0,
                    youngs_modulus=2.5e4,
                    poissons_ratio=0.25,
                    particle_radius=0.005,
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=free_cube_pos),
        )
    )

    return robot, colliding_cube, free_cube


def test_smoke_one_way(sim_one_way):
    """Smoke test: coupled solver (one-way) initializes and steps without crash."""
    robot, colliding_cube, free_cube = generate_robot_and_two_cubes()
    sim_one_way.reset()

    assert robot.is_initialized
    assert colliding_cube.is_initialized
    assert free_cube.is_initialized

    initial_z_collide = wp.to_torch(colliding_cube.data.root_pos_w)[0, 2].item()
    initial_z_free = wp.to_torch(free_cube.data.root_pos_w)[0, 2].item()

    for _ in range(10):
        sim_one_way.step()
        robot.update(sim_one_way.cfg.dt)
        colliding_cube.update(sim_one_way.cfg.dt)
        free_cube.update(sim_one_way.cfg.dt)

    # Both cubes should have fallen under gravity
    assert wp.to_torch(colliding_cube.data.root_pos_w)[0, 2].item() < initial_z_collide - 0.01
    assert wp.to_torch(free_cube.data.root_pos_w)[0, 2].item() < initial_z_free - 0.01


def test_smoke_two_way(sim_two_way):
    """Smoke test: coupled solver (two-way) initializes and steps without crash."""
    robot, colliding_cube, free_cube = generate_robot_and_two_cubes()
    sim_two_way.reset()

    assert robot.is_initialized
    assert colliding_cube.is_initialized
    assert free_cube.is_initialized

    initial_z_collide = wp.to_torch(colliding_cube.data.root_pos_w)[0, 2].item()
    initial_z_free = wp.to_torch(free_cube.data.root_pos_w)[0, 2].item()

    for _ in range(10):
        sim_two_way.step()
        robot.update(sim_two_way.cfg.dt)
        colliding_cube.update(sim_two_way.cfg.dt)
        free_cube.update(sim_two_way.cfg.dt)

    # Both cubes should have fallen under gravity
    assert wp.to_torch(colliding_cube.data.root_pos_w)[0, 2].item() < initial_z_collide - 0.01
    assert wp.to_torch(free_cube.data.root_pos_w)[0, 2].item() < initial_z_free - 0.01


def test_deformable_deflected_by_rigid_contact(sim_one_way):
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
    sim_one_way.reset()

    initial_xy_collide = wp.to_torch(colliding_cube.data.root_pos_w)[0, :2].clone()
    initial_xy_free = wp.to_torch(free_cube.data.root_pos_w)[0, :2].clone()

    # Free-fall from 1.0m takes sqrt(2*1.0/9.81) ~ 0.45s ~ 27 steps at dt=1/60.
    # Collision with the robot arm happens around step 20 (Z~0.5m).
    # 120 steps (2s) gives ample time for collision, bounce, and settling.
    for _ in range(120):
        sim_one_way.step()
        robot.update(sim_one_way.cfg.dt)
        colliding_cube.update(sim_one_way.cfg.dt)
        free_cube.update(sim_one_way.cfg.dt)

    final_xy_collide = wp.to_torch(colliding_cube.data.root_pos_w)[0, :2]
    final_xy_free = wp.to_torch(free_cube.data.root_pos_w)[0, :2]

    displacement_collide = (final_xy_collide - initial_xy_collide).norm().item()
    displacement_free = (final_xy_free - initial_xy_free).norm().item()

    # Colliding cube should be deflected; free cube should fall straight
    assert displacement_collide > displacement_free + 0.01, (
        f"Colliding cube should be deflected more than free cube: "
        f"collide={displacement_collide:.4f}, free={displacement_free:.4f}"
    )
