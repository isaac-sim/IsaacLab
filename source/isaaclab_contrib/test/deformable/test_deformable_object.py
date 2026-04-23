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

from isaaclab_contrib.deformable import DeformableObject, VBDSolverCfg, register_hooks
from isaaclab_newton.physics import NewtonCfg

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

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
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                density=500.0,
                youngs_modulus=2.5e4,
                poissons_ratio=0.25,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, height),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    cube_object = DeformableObject(cfg=cube_object_cfg)
    return cube_object


@pytest.fixture
def sim():
    """Create Newton VBD simulation context."""
    register_hooks()
    with _newton_sim_context() as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.parametrize("num_cubes", [1, 2])
def test_initialization(sim, num_cubes):
    """Test initialization of Newton deformable objects."""
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    sim.reset()

    assert cube_object.is_initialized
    assert cube_object.num_instances == num_cubes
    assert cube_object.max_sim_vertices_per_body > 0

    particles_per_body = cube_object.max_sim_vertices_per_body

    # nodal_state_w: (N, V, 6)
    nodal_state = wp.to_torch(cube_object.data.nodal_state_w)
    assert nodal_state.shape == (num_cubes, particles_per_body, 6)

    # nodal_pos_w: (N, V, 3)
    nodal_pos = wp.to_torch(cube_object.data.nodal_pos_w)
    assert nodal_pos.shape == (num_cubes, particles_per_body, 3)

    # nodal_vel_w: (N, V, 3)
    nodal_vel = wp.to_torch(cube_object.data.nodal_vel_w)
    assert nodal_vel.shape == (num_cubes, particles_per_body, 3)

    # root_pos_w: (N, 3)
    root_pos = wp.to_torch(cube_object.data.root_pos_w)
    assert root_pos.shape == (num_cubes, 3)

    # root_vel_w: (N, 3)
    root_vel = wp.to_torch(cube_object.data.root_vel_w)
    assert root_vel.shape == (num_cubes, 3)


@pytest.mark.parametrize("num_cubes", [1, 2])
def test_set_nodal_state(sim, num_cubes):
    """Test setting the state of the deformable object."""
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    sim.reset()

    for state_type_to_randomize in ["nodal_pos_w", "nodal_vel_w"]:
        state_dict = {
            "nodal_pos_w": torch.zeros_like(wp.to_torch(cube_object.data.nodal_pos_w)),
            "nodal_vel_w": torch.zeros_like(wp.to_torch(cube_object.data.nodal_vel_w)),
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

                torch.testing.assert_close(
                    wp.to_torch(cube_object.data.nodal_state_w), nodal_state, rtol=1e-5, atol=1e-5
                )

                sim.step()
                cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [2, 4])
def test_write_partial_env_ids(sim, num_cubes):
    """Test writing to a subset of environments using env_ids."""
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    sim.reset()

    particles_per_body = cube_object.max_sim_vertices_per_body
    default_pos = wp.to_torch(cube_object.data.nodal_pos_w).clone()

    # Write new positions only for env 0
    new_pos = torch.randn(1, particles_per_body, 3, device=sim.device)
    cube_object.write_nodal_pos_to_sim_index(new_pos, env_ids=torch.tensor([0], device=sim.device))
    cube_object.update(sim.cfg.dt)

    read_pos = wp.to_torch(cube_object.data.nodal_pos_w)

    # env 0 should have new positions
    torch.testing.assert_close(read_pos[0], new_pos[0], rtol=1e-5, atol=1e-5)

    # other envs should be unchanged
    torch.testing.assert_close(read_pos[1:], default_pos[1:], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("randomize_pos", [True, False])
@pytest.mark.parametrize("randomize_rot", [True, False])
@flaky(max_runs=3, min_passes=1)
def test_set_nodal_state_with_applied_transform(num_cubes, randomize_pos, randomize_rot):
    """Test setting the state of the deformable object with applied transform.

    Applies random position/rotation transforms to the default nodal state,
    writes it to simulation, steps with no gravity, and verifies the mean
    nodal position (root_pos_w) matches the expected transformed centroid.
    """
    register_hooks()
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
            nodal_state = wp.to_torch(cube_object.data.default_nodal_state_w).clone()
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

            torch.testing.assert_close(
                wp.to_torch(cube_object.data.root_pos_w), mean_nodal_pos_init, rtol=1e-4, atol=1e-4
            )


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

    x0 = wp.to_torch(cube_object.data.nodal_pos_w).clone()
    sim.step()
    cube_object.update(sim.cfg.dt)
    x1 = wp.to_torch(cube_object.data.nodal_pos_w)

    dz = x1[..., 2] - x0[..., 2]
    # Every vertex should have the same Z displacement under uniform gravity
    torch.testing.assert_close(dz, torch.full_like(dz, expected_dz), rtol=1e-2, atol=1e-5)


@pytest.mark.parametrize("num_cubes", [2, 4])
def test_set_kinematic_targets(sim, num_cubes):
    """Test setting kinematic targets for the deformable object.

    Env 0 is kinematically constrained (all vertices pinned at default positions,
    flag=0). Other envs are free (flag=1) and fall under gravity. After several
    steps, env 0 should stay in place while the others have fallen.
    """
    cube_object = generate_cubes_scene(num_cubes=num_cubes, height=5.0)

    sim.reset()

    particles_per_body = cube_object.max_sim_vertices_per_body
    default_state = wp.to_torch(cube_object.data.default_nodal_state_w)
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
                wp.to_torch(cube_object.data.nodal_pos_w)[0],
                default_pos[0],
                rtol=1e-5,
                atol=1e-5,
            )

        # Other envs should have fallen
        final_root_z = wp.to_torch(cube_object.data.root_pos_w)[1:, 2]
        default_root_z = default_pos[1:, :, 2].mean(dim=1)
        assert torch.all(final_root_z < default_root_z)
