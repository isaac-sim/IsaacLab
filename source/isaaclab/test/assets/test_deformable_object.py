# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import ctypes
import torch

import carb
import isaacsim.core.utils.prims as prim_utils
import pytest
from flaky import flaky

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import build_simulation_context


def generate_cubes_scene(
    num_cubes: int = 1,
    height: float = 1.0,
    initial_rot: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    has_api: bool = True,
    material_path: str | None = "material",
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
) -> DeformableObject:
    """Generate a scene with the provided number of cubes.

    Args:
        num_cubes: Number of cubes to generate.
        height: Height of the cubes. Default is 1.0.
        initial_rot: Initial rotation of the cubes. Default is (1.0, 0.0, 0.0, 0.0).
        has_api: Whether the cubes have a deformable body API on them.
        material_path: Path to the material file. If None, no material is added. Default is "material",
            which is path relative to the spawned object prim path.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.

    Returns:
        The deformable object representing the cubes.

    """
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cubes)]).to(device)
    # Create Top-level Xforms, one for each cube
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

    # Resolve spawn configuration
    if has_api:
        spawn_cfg = sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
        # Add physics material if provided
        if material_path is not None:
            spawn_cfg.physics_material = sim_utils.DeformableBodyMaterialCfg()
            spawn_cfg.physics_material_path = material_path
        else:
            spawn_cfg.physics_material = None
    else:
        # since no deformable body properties defined, this is just a static collider
        spawn_cfg = sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    # Create deformable object
    cube_object_cfg = DeformableObjectCfg(
        prim_path="/World/Table_.*/Object",
        spawn=spawn_cfg,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height), rot=initial_rot),
    )
    cube_object = DeformableObject(cfg=cube_object_cfg)

    return cube_object


@pytest.fixture
def sim():
    """Create simulation context."""
    with build_simulation_context(auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("material_path", [None, "/World/SoftMaterial", "material"])
def test_initialization(sim, num_cubes, material_path):
    """Test initialization for prim with deformable body API at the provided prim path."""
    cube_object = generate_cubes_scene(num_cubes=num_cubes, material_path=material_path)

    # Check that boundedness of deformable object is correct
    assert ctypes.c_long.from_address(id(cube_object)).value == 1

    # Play sim
    sim.reset()

    # Check if object is initialized
    assert cube_object.is_initialized

    # Check correct number of cubes
    assert cube_object.num_instances == num_cubes
    assert cube_object.root_physx_view.count == num_cubes

    # Check correct number of materials in the view
    if material_path:
        if material_path.startswith("/"):
            assert cube_object.material_physx_view.count == 1
        else:
            assert cube_object.material_physx_view.count == num_cubes
    else:
        assert cube_object.material_physx_view is None

    # Check buffers that exist and have correct shapes
    assert cube_object.data.nodal_state_w.shape == (num_cubes, cube_object.max_sim_vertices_per_body, 6)
    assert cube_object.data.nodal_kinematic_target.shape == (num_cubes, cube_object.max_sim_vertices_per_body, 4)
    assert cube_object.data.root_pos_w.shape == (num_cubes, 3)
    assert cube_object.data.root_vel_w.shape == (num_cubes, 3)

    # Simulate physics
    for _ in range(2):
        sim.step()
        cube_object.update(sim.cfg.dt)

    # Check sim data
    assert cube_object.data.sim_element_quat_w.shape == (num_cubes, cube_object.max_sim_elements_per_body, 4)
    assert cube_object.data.sim_element_deform_gradient_w.shape == (
        num_cubes,
        cube_object.max_sim_elements_per_body,
        3,
        3,
    )
    assert cube_object.data.sim_element_stress_w.shape == (num_cubes, cube_object.max_sim_elements_per_body, 3, 3)
    assert cube_object.data.collision_element_quat_w.shape == (
        num_cubes,
        cube_object.max_collision_elements_per_body,
        4,
    )
    assert cube_object.data.collision_element_deform_gradient_w.shape == (
        num_cubes,
        cube_object.max_collision_elements_per_body,
        3,
        3,
    )
    assert cube_object.data.collision_element_stress_w.shape == (
        num_cubes,
        cube_object.max_collision_elements_per_body,
        3,
        3,
    )


@pytest.mark.isaacsim_ci
def test_initialization_on_device_cpu():
    """Test that initialization fails with deformable body API on the CPU."""
    with build_simulation_context(device="cpu", auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object = generate_cubes_scene(num_cubes=5, device="cpu")

        # Check that boundedness of deformable object is correct
        assert ctypes.c_long.from_address(id(cube_object)).value == 1

        # Play sim
        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.isaacsim_ci
def test_initialization_with_kinematic_enabled(sim, num_cubes):
    """Test that initialization for prim with kinematic flag enabled."""
    cube_object = generate_cubes_scene(num_cubes=num_cubes, kinematic_enabled=True)

    # Check that boundedness of deformable object is correct
    assert ctypes.c_long.from_address(id(cube_object)).value == 1

    # Play sim
    sim.reset()

    # Check if object is initialized
    assert cube_object.is_initialized

    # Check buffers that exist and have correct shapes
    assert cube_object.data.root_pos_w.shape == (num_cubes, 3)
    assert cube_object.data.root_vel_w.shape == (num_cubes, 3)

    # Simulate physics
    for _ in range(2):
        sim.step()
        cube_object.update(sim.cfg.dt)
        default_nodal_state_w = cube_object.data.default_nodal_state_w.clone()
        torch.testing.assert_close(cube_object.data.nodal_state_w, default_nodal_state_w)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.isaacsim_ci
def test_initialization_with_no_deformable_body(sim, num_cubes):
    """Test that initialization fails when no deformable body is found at the provided prim path."""
    cube_object = generate_cubes_scene(num_cubes=num_cubes, has_api=False)

    # Check that boundedness of deformable object is correct
    assert ctypes.c_long.from_address(id(cube_object)).value == 1

    # Play sim
    with pytest.raises(RuntimeError):
        sim.reset()


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.isaacsim_ci
def test_set_nodal_state(sim, num_cubes):
    """Test setting the state of the deformable object."""
    cube_object = generate_cubes_scene(num_cubes=num_cubes)

    # Play the simulator
    sim.reset()

    for state_type_to_randomize in ["nodal_pos_w", "nodal_vel_w"]:
        state_dict = {
            "nodal_pos_w": torch.zeros_like(cube_object.data.nodal_pos_w),
            "nodal_vel_w": torch.zeros_like(cube_object.data.nodal_vel_w),
        }

        for _ in range(5):
            cube_object.reset()

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
                cube_object.write_nodal_state_to_sim(nodal_state)

                torch.testing.assert_close(cube_object.data.nodal_state_w, nodal_state, rtol=1e-5, atol=1e-5)

                sim.step()
                cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("randomize_pos", [True, False])
@pytest.mark.parametrize("randomize_rot", [True, False])
@flaky(max_runs=3, min_passes=1)
@pytest.mark.isaacsim_ci
def test_set_nodal_state_with_applied_transform(sim, num_cubes, randomize_pos, randomize_rot):
    """Test setting the state of the deformable object with applied transform."""
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    # Create a new simulation context with gravity disabled
    with build_simulation_context(auto_add_lighting=True, gravity_enabled=False) as sim:
        sim._app_control_on_stop_handle = None
        cube_object = generate_cubes_scene(num_cubes=num_cubes)
        sim.reset()

        for _ in range(5):
            nodal_state = cube_object.data.default_nodal_state_w.clone()
            mean_nodal_pos_default = nodal_state[..., :3].mean(dim=1)

            if randomize_pos:
                pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device)
                pos_w[:, 2] += 0.5
            else:
                pos_w = None
            if randomize_rot:
                quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            else:
                quat_w = None

            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
            mean_nodal_pos_init = nodal_state[..., :3].mean(dim=1)

            if pos_w is None:
                torch.testing.assert_close(mean_nodal_pos_init, mean_nodal_pos_default, rtol=1e-5, atol=1e-5)
            else:
                torch.testing.assert_close(mean_nodal_pos_init, mean_nodal_pos_default + pos_w, rtol=1e-5, atol=1e-5)

            cube_object.write_nodal_state_to_sim(nodal_state)
            cube_object.reset()

            for _ in range(50):
                sim.step()
                cube_object.update(sim.cfg.dt)

            torch.testing.assert_close(cube_object.data.root_pos_w, mean_nodal_pos_init, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_cubes", [2, 4])
@pytest.mark.isaacsim_ci
def test_set_kinematic_targets(sim, num_cubes):
    """Test setting kinematic targets for the deformable object."""
    cube_object = generate_cubes_scene(num_cubes=num_cubes, height=1.0)

    sim.reset()

    nodal_kinematic_targets = cube_object.root_physx_view.get_sim_kinematic_targets().clone()

    for _ in range(5):
        cube_object.write_nodal_state_to_sim(cube_object.data.default_nodal_state_w)

        default_root_pos = cube_object.data.default_nodal_state_w.mean(dim=1)

        cube_object.reset()

        nodal_kinematic_targets[1:, :, 3] = 1.0
        nodal_kinematic_targets[0, :, 3] = 0.0
        nodal_kinematic_targets[0, :, :3] = cube_object.data.default_nodal_state_w[0, :, :3]
        cube_object.write_nodal_kinematic_target_to_sim(
            nodal_kinematic_targets[0], env_ids=torch.tensor([0], device=sim.device)
        )

        for _ in range(20):
            sim.step()
            cube_object.update(sim.cfg.dt)

            torch.testing.assert_close(
                cube_object.data.nodal_pos_w[0], nodal_kinematic_targets[0, :, :3], rtol=1e-5, atol=1e-5
            )
            root_pos_w = cube_object.data.root_pos_w
            assert torch.all(root_pos_w[1:, 2] < default_root_pos[1:, 2])
