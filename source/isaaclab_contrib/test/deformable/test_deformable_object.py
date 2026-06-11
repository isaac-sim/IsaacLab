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
import warp as wp
from flaky import flaky
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg

NEWTON_VBD_CFG = SimulationCfg(
    physics=NewtonCfg(
        solver_cfg=VBDSolverCfg(iterations=3),
        num_substeps=2,
    ),
)


def _newton_sim_context(device="cuda:0", gravity_enabled=True):
    """Helper to create a Newton VBD simulation context."""
    NEWTON_VBD_CFG.device = device
    NEWTON_VBD_CFG.gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    return build_simulation_context(device=device, sim_cfg=NEWTON_VBD_CFG, auto_add_lighting=True)


def generate_cubes_scene(
    num_cubes: int = 1,
    height: float = 1.0,
    device: str = "cuda:0",
) -> DeformableObject:
    """Generate a scene with deformable tet-mesh cubes.

    Args:
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        device: Device to use for the simulation.

    Returns:
        The deformable object representing the cubes.
    """
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cubes)]).to(device)
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=origin)

    cube_object_cfg = DeformableObjectCfg(
        prim_path="/World/env_.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.1),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=500.0,
                k_mu=1e4,
                k_lambda=1e4,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, height),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    cube_object = DeformableObject(cfg=cube_object_cfg)
    return cube_object


def generate_cloth_scene(
    num_cloths: int = 1,
    height: float = 1.0,
    device: str = "cuda:0",
) -> DeformableObject:
    """Generate a scene with surface deformable cloth squares.

    Args:
        num_cloths: Number of cloths to generate.
        height: Height of the cloths.
        device: Device to use for the simulation.

    Returns:
        The deformable object representing the cloths.
    """
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cloths)]).to(device)
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=origin)

    cloth_object_cfg = DeformableObjectCfg(
        prim_path="/World/env_.*/Cloth",
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            resolution=(3, 3),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(density=0.02, particle_radius=0.005),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, height),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    return DeformableObject(cfg=cloth_object_cfg)


def generate_cuboid_and_cylinder_scene(height: float = 1.0) -> tuple[DeformableObject, DeformableObject]:
    """Generate two independent deformable assets with different mesh shapes."""
    sim_utils.create_prim("/World/env_0", "Xform", translation=(0.0, 0.0, 0.0))

    cuboid_cfg = DeformableObjectCfg(
        prim_path="/World/env_.*/Cuboid",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.16, 0.08, 0.12),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=500.0,
                k_mu=1e4,
                k_lambda=1e4,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cylinder_cfg = DeformableObjectCfg(
        prim_path="/World/env_.*/Cylinder",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.06,
            height=0.14,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=500.0,
                k_mu=1e4,
                k_lambda=1e4,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, height + 0.2)),
    )
    return DeformableObject(cfg=cuboid_cfg), DeformableObject(cfg=cylinder_cfg)


@pytest.fixture
def sim():
    """Create Newton VBD simulation context."""
    with _newton_sim_context() as sim:
        sim._app_control_on_stop_handle = None
        yield sim


def test_initialization(sim):
    """Test initialization of Newton deformable objects."""
    num_cubes = 2
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    sim.reset()

    assert cube_object.is_initialized
    assert cube_object.num_instances == num_cubes
    assert cube_object.max_sim_vertices_per_body > 0

    particles_per_body = cube_object.max_sim_vertices_per_body

    # nodal_state_w: (N, V, 6)
    nodal_state = cube_object.data.nodal_state_w.torch
    assert nodal_state.shape == (num_cubes, particles_per_body, 6)

    # nodal_pos_w: (N, V, 3)
    nodal_pos = cube_object.data.nodal_pos_w.torch
    assert nodal_pos.shape == (num_cubes, particles_per_body, 3)

    # nodal_vel_w: (N, V, 3)
    nodal_vel = cube_object.data.nodal_vel_w.torch
    assert nodal_vel.shape == (num_cubes, particles_per_body, 3)

    # root_pos_w: (N, 3)
    root_pos = cube_object.data.root_pos_w.torch
    assert root_pos.shape == (num_cubes, 3)

    # root_vel_w: (N, 3)
    root_vel = cube_object.data.root_vel_w.torch
    assert root_vel.shape == (num_cubes, 3)


def test_surface_initialization_and_freefall(sim):
    """Test initialization and stepping for surface deformable objects."""
    num_cloths = 2
    cloth_object = generate_cloth_scene(num_cloths=num_cloths, height=5.0)

    sim.reset()

    assert cloth_object.is_initialized
    assert cloth_object.num_instances == num_cloths
    assert cloth_object._deformable_type == "surface"
    assert cloth_object.max_sim_vertices_per_body > 0
    assert cloth_object.data.nodal_pos_w.torch.shape == (
        num_cloths,
        cloth_object.max_sim_vertices_per_body,
        3,
    )

    initial_root_z = cloth_object.data.root_pos_w.torch[:, 2].clone()
    for _ in range(5):
        sim.step()
        cloth_object.update(sim.cfg.dt)

    assert torch.all(cloth_object.data.root_pos_w.torch[:, 2] < initial_root_z)


def test_set_nodal_state(sim):
    """Test setting the state of the deformable object."""
    num_cubes = 2
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    sim.reset()

    for state_type_to_randomize in ["nodal_pos_w", "nodal_vel_w"]:
        state_dict = {
            "nodal_pos_w": torch.zeros_like(cube_object.data.nodal_pos_w.torch),
            "nodal_vel_w": torch.zeros_like(cube_object.data.nodal_vel_w.torch),
        }

        for _ in range(5):
            state_dict[state_type_to_randomize] = torch.randn(
                num_cubes, cube_object.max_sim_vertices_per_body, 3, device=sim.device
            )

            for _ in range(5):
                nodal_state = torch.cat(
                    [
                        state_dict["nodal_pos_w"],
                        state_dict["nodal_vel_w"],
                    ],
                    dim=-1,
                )
                cube_object.write_nodal_state_to_sim_index(nodal_state)

                torch.testing.assert_close(cube_object.data.nodal_state_w.torch, nodal_state, rtol=1e-5, atol=1e-5)

                sim.step()
                cube_object.update(sim.cfg.dt)


def test_write_partial_env_ids(sim):
    """Test writing to a subset of environments using env_ids."""
    num_cubes = 2
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    sim.reset()

    particles_per_body = cube_object.max_sim_vertices_per_body
    default_pos = cube_object.data.nodal_pos_w.torch.clone()

    # Write new positions only for env 0
    new_pos = torch.randn(1, particles_per_body, 3, device=sim.device)
    cube_object.write_nodal_pos_to_sim_index(new_pos, env_ids=torch.tensor([0], device=sim.device))
    cube_object.update(sim.cfg.dt)

    read_pos = cube_object.data.nodal_pos_w.torch

    # env 0 should have new positions
    torch.testing.assert_close(read_pos[0], new_pos[0], rtol=1e-5, atol=1e-5)

    # other envs should be unchanged
    torch.testing.assert_close(read_pos[1:], default_pos[1:], rtol=1e-5, atol=1e-5)


def test_write_partial_velocity_env_ids(sim):
    """Test writing nodal velocities to a subset of environments."""
    num_cubes = 4
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    sim.reset()

    particles_per_body = cube_object.max_sim_vertices_per_body
    default_vel = cube_object.data.nodal_vel_w.torch.clone()

    env_ids = torch.tensor([1], device=sim.device)
    new_vel = torch.full((1, particles_per_body, 3), 0.25, device=sim.device)
    new_vel[..., 2] = 1.0
    cube_object.write_nodal_velocity_to_sim_index(new_vel, env_ids=env_ids)
    cube_object.update(sim.cfg.dt)

    read_vel = cube_object.data.nodal_vel_w.torch
    torch.testing.assert_close(read_vel[1], new_vel[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cube_object.data.root_vel_w.torch[1], new_vel[0].mean(dim=0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_vel[0], default_vel[0], rtol=1e-5, atol=1e-5)
    if num_cubes > 2:
        torch.testing.assert_close(read_vel[2:], default_vel[2:], rtol=1e-5, atol=1e-5)


def test_full_data_writes_selected_env(sim):
    """Test full-sized write buffers with selected environment ids."""
    num_cubes = 3
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    sim.reset()

    particles_per_body = cube_object.max_sim_vertices_per_body
    env_ids = torch.tensor([1], device=sim.device)

    default_pos = cube_object.data.nodal_pos_w.torch.clone()
    full_pos = default_pos + torch.linspace(0.1, 0.3, num_cubes, device=sim.device).view(num_cubes, 1, 1)
    cube_object.write_nodal_pos_to_sim_index(full_pos, env_ids=env_ids, full_data=True)
    cube_object.update(sim.cfg.dt)

    read_pos = cube_object.data.nodal_pos_w.torch
    torch.testing.assert_close(read_pos[1], full_pos[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_pos[0], default_pos[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_pos[2], default_pos[2], rtol=1e-5, atol=1e-5)

    default_vel = cube_object.data.nodal_vel_w.torch.clone()
    full_vel = torch.zeros(num_cubes, particles_per_body, 3, device=sim.device)
    full_vel[0, :, 0] = 0.5
    full_vel[1, :, 1] = 0.75
    full_vel[2, :, 2] = 1.0
    cube_object.write_nodal_velocity_to_sim_index(full_vel, env_ids=env_ids, full_data=True)
    cube_object.update(sim.cfg.dt)

    read_vel = cube_object.data.nodal_vel_w.torch
    torch.testing.assert_close(read_vel[1], full_vel[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_vel[0], default_vel[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_vel[2], default_vel[2], rtol=1e-5, atol=1e-5)

    default_targets = cube_object.data.nodal_kinematic_target.torch.clone()
    full_targets = torch.zeros(num_cubes, particles_per_body, 4, device=sim.device)
    full_targets[..., :3] = cube_object.data.default_nodal_state_w.torch[..., :3]
    full_targets[..., 3] = 1.0
    full_targets[1, :, :3] += torch.tensor([0.0, 0.0, 0.1], device=sim.device)
    full_targets[1, :, 3] = 0.0
    cube_object.write_nodal_kinematic_target_to_sim_index(full_targets, env_ids=env_ids, full_data=True)

    read_targets = cube_object.data.nodal_kinematic_target.torch
    torch.testing.assert_close(read_targets[1], full_targets[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_targets[0], default_targets[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_targets[2], default_targets[2], rtol=1e-5, atol=1e-5)


def test_kinematic_target_partial_env_ids_with_warp_input(sim):
    """Test indexed kinematic target writes with device-native input arrays."""
    num_cubes = 3
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    sim.reset()

    particles_per_body = cube_object.max_sim_vertices_per_body
    env_ids = torch.tensor([2], device=sim.device)

    default_targets = cube_object.data.nodal_kinematic_target.torch.clone()
    targets = torch.zeros(1, particles_per_body, 4, device=sim.device)
    targets[0, :, :3] = cube_object.data.default_nodal_state_w.torch[2, :, :3]
    targets[0, :, :3] += torch.tensor([0.0, 0.0, 0.1], device=sim.device)
    targets[0, :, 3] = 0.0

    cube_object.write_nodal_kinematic_target_to_sim_index(
        wp.from_torch(targets.contiguous(), dtype=wp.vec4f), env_ids=env_ids
    )

    read_targets = cube_object.data.nodal_kinematic_target.torch
    torch.testing.assert_close(read_targets[2], targets[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_targets[0], default_targets[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_targets[1], default_targets[1], rtol=1e-5, atol=1e-5)


def test_mask_writes_selected_env(sim):
    """Test full-sized write buffers with selected environment masks."""
    num_cubes = 3
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    sim.reset()

    env_mask = wp.array([False, True, False], dtype=wp.bool, device=sim.device)

    default_state = cube_object.data.nodal_state_w.torch.clone()
    full_state = default_state.clone()
    full_state[:, :, :3] += torch.tensor([10.0, 10.0, 10.0], device=sim.device)
    full_state[:, :, 3:] = 10.0
    full_state[1, :, :3] = default_state[1, :, :3] + torch.tensor([0.1, 0.2, 0.3], device=sim.device)
    full_state[1, :, 3:] = torch.tensor([0.4, 0.5, 0.6], device=sim.device)
    cube_object.write_nodal_state_to_sim_mask(full_state, env_mask=env_mask)
    cube_object.update(sim.cfg.dt)

    read_state = cube_object.data.nodal_state_w.torch
    torch.testing.assert_close(read_state[1], full_state[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_state[0], default_state[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_state[2], default_state[2], rtol=1e-5, atol=1e-5)

    pos_before = cube_object.data.nodal_pos_w.torch.clone()
    full_pos = pos_before + torch.tensor([5.0, 5.0, 5.0], device=sim.device)
    full_pos[1] = pos_before[1] + torch.tensor([0.0, -0.1, 0.2], device=sim.device)
    cube_object.write_nodal_pos_to_sim_mask(full_pos, env_mask=env_mask)
    cube_object.update(sim.cfg.dt)

    read_pos = cube_object.data.nodal_pos_w.torch
    torch.testing.assert_close(read_pos[1], full_pos[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_pos[0], pos_before[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_pos[2], pos_before[2], rtol=1e-5, atol=1e-5)

    vel_before = cube_object.data.nodal_vel_w.torch.clone()
    full_vel = torch.full_like(vel_before, 7.0)
    full_vel[1] = torch.tensor([0.7, 0.8, 0.9], device=sim.device)
    cube_object.write_nodal_velocity_to_sim_mask(full_vel, env_mask=env_mask)
    cube_object.update(sim.cfg.dt)

    read_vel = cube_object.data.nodal_vel_w.torch
    torch.testing.assert_close(read_vel[1], full_vel[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_vel[0], vel_before[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_vel[2], vel_before[2], rtol=1e-5, atol=1e-5)

    default_targets = cube_object.data.nodal_kinematic_target.torch.clone()
    full_targets = default_targets.clone()
    full_targets[:, :, :3] = 3.0
    full_targets[:, :, 3] = 0.0
    full_targets[1, :, :3] = read_pos[1] + torch.tensor([0.0, 0.0, 0.1], device=sim.device)
    cube_object.write_nodal_kinematic_target_to_sim_mask(full_targets, env_mask=env_mask)

    read_targets = cube_object.data.nodal_kinematic_target.torch
    torch.testing.assert_close(read_targets[1], full_targets[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_targets[0], default_targets[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(read_targets[2], default_targets[2], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "num_cubes, randomize_pos, randomize_rot",
    [
        (1, False, False),
        (1, True, False),
        (1, False, True),
        (2, True, True),
    ],
)
@flaky(max_runs=3, min_passes=1)
def test_set_nodal_state_with_applied_transform(num_cubes, randomize_pos, randomize_rot):
    """Test setting the state of the deformable object with applied transform.

    Applies random position/rotation transforms to the default nodal state,
    writes it to simulation, steps with no gravity, and verifies the mean
    nodal position (root_pos_w) matches the expected transformed centroid.
    """
    cfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=3),
            num_substeps=2,
        ),
    )
    cfg.device = "cuda:0"
    cfg.gravity = (0.0, 0.0, 0.0)

    with build_simulation_context(device="cuda:0", sim_cfg=cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object = generate_cubes_scene(num_cubes=num_cubes, height=5.0)
        sim.reset()

        for _ in range(5):
            nodal_state = cube_object.data.default_nodal_state_w.torch.clone()
            mean_nodal_pos_default = nodal_state[..., :3].mean(dim=1)

            if randomize_pos:
                pos_w = 0.5 * torch.rand(cube_object.num_instances, 3, device=sim.device)
                pos_w[:, 2] += 0.5
            else:
                pos_w = None
            if randomize_rot:
                quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            else:
                quat_w = None

            # transform_nodal_pos: center, rotate, translate, un-center
            nodal_pos = nodal_state[..., :3]
            mean_pos = nodal_pos.mean(dim=1, keepdim=True)
            centered = nodal_pos - mean_pos
            nodal_state[..., :3] = math_utils.transform_points(centered, pos_w, quat_w) + mean_pos
            mean_nodal_pos_init = nodal_state[..., :3].mean(dim=1)

            if pos_w is None:
                torch.testing.assert_close(mean_nodal_pos_init, mean_nodal_pos_default, rtol=1e-5, atol=1e-5)
            else:
                torch.testing.assert_close(mean_nodal_pos_init, mean_nodal_pos_default + pos_w, rtol=1e-5, atol=1e-5)

            cube_object.write_nodal_state_to_sim_index(nodal_state)

            for _ in range(50):
                sim.step()
                cube_object.update(sim.cfg.dt)

            torch.testing.assert_close(cube_object.data.root_pos_w.torch, mean_nodal_pos_init, rtol=1e-4, atol=1e-4)


def test_freefall_analytical(sim):
    """Test that one step of free-fall matches the inertia target prediction.

    VBD computes an inertia target per substep (h = sub_dt)::

        v_new = v + g * h
        x_new = x + v_new * h

    then optimizes elastic + contact potentials around it. For free-fall
    (no contacts, negligible elastic forces over one step), the final
    position equals the inertia target.

    Starting from rest (v_0 = 0) with N substeps of h = dt/N:

        substep 1: v_1 = g*h,       dx_1 = g*h^2         (1 * g*h^2)
        substep 2: v_2 = 2*g*h,     dx_2 = 2*g*h^2       (2 * g*h^2)
        ...
        substep k: v_k = k*g*h,     dx_k = k*g*h^2       (k * g*h^2)

        total dz = g*h^2 * (1 + 2 + ... + N) = g*h^2 * N*(N+1)/2

    """
    g = -9.81
    dt = 1.0 / 60.0
    num_substeps = 2
    sub_dt = dt / num_substeps
    expected_dz = g * sub_dt**2 * num_substeps * (num_substeps + 1) / 2

    cube_object = generate_cubes_scene(num_cubes=1, height=5.0)
    sim.reset()

    x0 = cube_object.data.nodal_pos_w.torch.clone()
    sim.step()
    cube_object.update(sim.cfg.dt)
    x1 = cube_object.data.nodal_pos_w.torch

    dz = x1[..., 2] - x0[..., 2]
    # Every vertex should have the same Z displacement under uniform gravity
    torch.testing.assert_close(dz, torch.full_like(dz, expected_dz), rtol=1e-2, atol=1e-5)


def test_nodal_pos_reads_current_state_after_odd_substep_swap():
    """Test deformable reads use the current Newton state after state swapping."""
    cfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=3),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )
    cfg.device = "cuda:0"
    cfg.gravity = (0.0, 0.0, -9.81)

    with build_simulation_context(device="cuda:0", sim_cfg=cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object = generate_cubes_scene(num_cubes=1, height=5.0)

        sim.reset()

        initial_pos = cube_object.data.nodal_pos_w.torch.clone()
        sim.step()
        cube_object.update(sim.cfg.dt)

        stepped_pos = cube_object.data.nodal_pos_w.torch
        assert torch.all(stepped_pos[..., 2] < initial_pos[..., 2])


def test_set_kinematic_targets(sim):
    """Test setting kinematic targets for the deformable object.

    Env 0 is kinematically constrained (all vertices pinned at default positions,
    flag=0). Other envs are free (flag=1) and fall under gravity. After several
    steps, env 0 should stay in place while the others have fallen.
    """
    num_cubes = 4
    cube_object = generate_cubes_scene(num_cubes=num_cubes, height=5.0)

    sim.reset()

    particles_per_body = cube_object.max_sim_vertices_per_body
    default_state = cube_object.data.default_nodal_state_w.torch
    default_pos = default_state[..., :3].clone()

    # Build kinematic target buffer: (N, V, 4) = [x, y, z, flag]
    nodal_kinematic_targets = torch.zeros(num_cubes, particles_per_body, 4, device=sim.device)

    for _ in range(5):
        # Restore default state
        cube_object.write_nodal_state_to_sim_index(default_state)

        # Env 0: pin all vertices at default positions (flag=0 = kinematic)
        nodal_kinematic_targets[0, :, :3] = default_pos[0]
        nodal_kinematic_targets[0, :, 3] = 0.0

        # Other envs: free (flag=1)
        nodal_kinematic_targets[1:, :, :3] = default_pos[1:]
        nodal_kinematic_targets[1:, :, 3] = 1.0

        cube_object.write_nodal_kinematic_target_to_sim_index(nodal_kinematic_targets)

        for _ in range(20):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

            # Env 0 should stay at default position (kinematically constrained)
            torch.testing.assert_close(
                cube_object.data.nodal_pos_w.torch[0],
                default_pos[0],
                rtol=1e-5,
                atol=1e-5,
            )

        # Other envs should have fallen
        final_root_z = cube_object.data.root_pos_w.torch[1:, 2]
        default_root_z = default_pos[1:, :, 2].mean(dim=1)
        assert torch.all(final_root_z < default_root_z)


def test_kinematic_target_release_restores_free_motion(sim):
    """Test that a pinned deformable falls again after kinematic targets are released."""
    cube_object = generate_cubes_scene(num_cubes=1, height=5.0)

    sim.reset()

    particles_per_body = cube_object.max_sim_vertices_per_body
    default_state = cube_object.data.default_nodal_state_w.torch
    default_pos = default_state[..., :3].clone()

    targets = torch.zeros(1, particles_per_body, 4, device=sim.device)
    targets[0, :, :3] = default_pos[0]
    targets[0, :, 3] = 0.0
    cube_object.write_nodal_kinematic_target_to_sim_index(targets)

    for _ in range(5):
        cube_object.write_data_to_sim()
        sim.step()
        cube_object.update(sim.cfg.dt)

    torch.testing.assert_close(cube_object.data.nodal_pos_w.torch[0], default_pos[0], rtol=1e-5, atol=1e-5)

    targets[0, :, 3] = 1.0
    cube_object.write_nodal_kinematic_target_to_sim_index(targets)

    for _ in range(20):
        cube_object.write_data_to_sim()
        sim.step()
        cube_object.update(sim.cfg.dt)

    assert cube_object.data.root_pos_w.torch[0, 2] < default_pos[0, :, 2].mean()


def test_multiple_deformable_assets_do_not_alias(sim):
    """Test independent writes for two different deformable assets in one scene."""
    cuboid, cylinder = generate_cuboid_and_cylinder_scene(height=2.0)

    sim.reset()

    cuboid_default = cuboid.data.nodal_pos_w.torch.clone()
    cylinder_default = cylinder.data.nodal_pos_w.torch.clone()

    cuboid_pos = cuboid_default + torch.tensor([0.15, -0.05, 0.1], device=sim.device)
    cuboid.write_nodal_pos_to_sim_index(cuboid_pos)
    cuboid.update(sim.cfg.dt)
    cylinder.update(sim.cfg.dt)

    torch.testing.assert_close(cuboid.data.nodal_pos_w.torch, cuboid_pos, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cylinder.data.nodal_pos_w.torch, cylinder_default, rtol=1e-5, atol=1e-5)
    assert cuboid._recorded_particle_offsets != cylinder._recorded_particle_offsets


def test_rebind_after_sim_reset(sim):
    """Test that deformable write paths remain valid after a simulation reset."""
    cube_object = generate_cubes_scene(num_cubes=1, height=2.0)

    sim.reset()

    initial_pos = cube_object.data.default_nodal_state_w.torch[..., :3].clone()
    first_pos = initial_pos + torch.tensor([0.1, 0.0, 0.0], device=sim.device)
    cube_object.write_nodal_pos_to_sim_index(first_pos)
    cube_object.update(sim.cfg.dt)
    torch.testing.assert_close(cube_object.data.nodal_pos_w.torch, first_pos, rtol=1e-5, atol=1e-5)

    sim.reset()

    second_pos = initial_pos + torch.tensor([0.0, -0.1, 0.2], device=sim.device)
    cube_object.write_nodal_pos_to_sim_index(second_pos)
    cube_object.update(sim.cfg.dt)
    torch.testing.assert_close(cube_object.data.nodal_pos_w.torch, second_pos, rtol=1e-5, atol=1e-5)


def test_write_shape_validation(sim):
    """Test public write APIs reject wrong tensor shapes."""
    cube_object = generate_cubes_scene(num_cubes=2)

    sim.reset()

    particles_per_body = cube_object.max_sim_vertices_per_body
    wrong_pos = torch.zeros(particles_per_body, 3, device=sim.device)
    wrong_vel = torch.zeros(1, particles_per_body, 2, device=sim.device)
    wrong_targets = torch.zeros(2, particles_per_body, 3, device=sim.device)

    with pytest.raises(AssertionError, match="Shape mismatch"):
        cube_object.write_nodal_pos_to_sim_index(wrong_pos)
    with pytest.raises(AssertionError, match="Shape mismatch"):
        cube_object.write_nodal_velocity_to_sim_index(wrong_vel, env_ids=torch.tensor([0], device=sim.device))
    with pytest.raises(AssertionError, match="Shape mismatch"):
        cube_object.write_nodal_kinematic_target_to_sim_index(wrong_targets)
