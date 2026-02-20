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

import ctypes
from typing import Literal

import pytest
import torch
import warp as wp
from flaky import flaky
from isaaclab_physx.assets import RigidObject

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim.spawners import materials
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import (
    combine_frame_transforms,
    default_orientation,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_rotate,
    random_orientation,
)


def generate_cubes_scene(
    num_cubes: int = 1,
    height=1.0,
    api: Literal["none", "rigid_body", "articulation_root"] = "rigid_body",
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
) -> tuple[RigidObject, torch.Tensor]:
    """Generate a scene with the provided number of cubes.

    Args:
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        api: The type of API that the cubes should have.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.

    Returns:
        A tuple containing the rigid object representing the cubes and the origins of the cubes.

    """
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cubes)]).to(device)
    # Create Top-level Xforms, one for each cube
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

    # Resolve spawn configuration
    if api == "none":
        # since no rigid body properties defined, this is just a static collider
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    elif api == "rigid_body":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    elif api == "articulation_root":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Tests/RigidObject/Cube/dex_cube_instanceable_with_articulation_root.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
        )
    else:
        raise ValueError(f"Unknown api: {api}")

    # Create rigid object
    cube_object_cfg = RigidObjectCfg(
        prim_path="/World/Table_.*/Object",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cube_object = RigidObject(cfg=cube_object_cfg)

    return cube_object, origins


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization(num_cubes, device):
    """Test initialization for prim with rigid body API at the provided prim path."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Check that boundedness of rigid object is correct
        assert ctypes.c_long.from_address(id(cube_object)).value == 1

        # Play sim
        sim.reset()

        # Check if object is initialized
        assert cube_object.is_initialized
        assert len(cube_object.body_names) == 1

        # Check buffers that exists and have correct shapes
        assert wp.to_torch(cube_object.data.root_pos_w).shape == (num_cubes, 3)
        assert wp.to_torch(cube_object.data.root_quat_w).shape == (num_cubes, 4)
        assert wp.to_torch(cube_object.data.body_mass).shape == (num_cubes, 1)
        assert wp.to_torch(cube_object.data.body_inertia).shape == (num_cubes, 9)

        # Simulate physics
        for _ in range(2):
            # perform rendering
            sim.step()
            # update object
            cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_kinematic_enabled(num_cubes, device):
    """Test that initialization for prim with kinematic flag enabled."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Generate cubes scene
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, kinematic_enabled=True, device=device)

        # Check that boundedness of rigid object is correct
        assert ctypes.c_long.from_address(id(cube_object)).value == 1

        # Play sim
        sim.reset()

        # Check if object is initialized
        assert cube_object.is_initialized
        assert len(cube_object.body_names) == 1

        # Check buffers that exists and have correct shapes
        assert wp.to_torch(cube_object.data.root_pos_w).shape == (num_cubes, 3)
        assert wp.to_torch(cube_object.data.root_quat_w).shape == (num_cubes, 4)

        # Simulate physics
        for _ in range(2):
            # perform rendering
            sim.step()
            # update object
            cube_object.update(sim.cfg.dt)
            # check that the object is kinematic
            default_root_pose = wp.to_torch(cube_object.data.default_root_pose).clone()
            default_root_vel = wp.to_torch(cube_object.data.default_root_vel).clone()
            default_root_pose[:, :3] += origins
            torch.testing.assert_close(wp.to_torch(cube_object.data.root_link_pose_w), default_root_pose)
            torch.testing.assert_close(wp.to_torch(cube_object.data.root_com_vel_w), default_root_vel)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_no_rigid_body(num_cubes, device):
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, api="none", device=device)

        # Check that boundedness of rigid object is correct
        assert ctypes.c_long.from_address(id(cube_object)).value == 1

        # Play sim
        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_initialization_with_articulation_root(num_cubes, device):
    """Test that initialization fails when an articulation root is found at the provided prim path."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, api="articulation_root", device=device)

        # Check that boundedness of rigid object is correct
        assert ctypes.c_long.from_address(id(cube_object)).value == 1

        # Play sim
        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_buffer(device):
    """Test if external force buffer correctly updates in the force value is zero case.

    In this test, we apply a non-zero force, then a zero force, then finally a non-zero force
    to an object. We check if the force buffer is properly updated at each step.
    """

    # Generate cubes scene
    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, origins = generate_cubes_scene(num_cubes=1, device=device)

        # play the simulator
        sim.reset()

        # find bodies to apply the force
        body_ids, body_names = cube_object.find_bodies(".*")

        # reset object
        cube_object.reset()

        # perform simulation
        for step in range(5):
            # initiate force tensor
            external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)

            if step == 0 or step == 3:
                # set a non-zero force
                force = 1
            else:
                # set a zero force
                force = 0

            # set force value
            external_wrench_b[:, :, 0] = force
            external_wrench_b[:, :, 3] = force

            # apply force
            cube_object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                body_ids=body_ids,
            )

            # check if the cube's force and torque buffers are correctly updated
            for i in range(cube_object.num_instances):
                assert wp.to_torch(cube_object._permanent_wrench_composer.composed_force)[i, 0, 0].item() == force
                assert wp.to_torch(cube_object._permanent_wrench_composer.composed_torque)[i, 0, 0].item() == force

            # Check if the instantaneous wrench is correctly added to the permanent wrench
            cube_object.permanent_wrench_composer.add_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                body_ids=body_ids,
            )

            # apply action to the object
            cube_object.write_data_to_sim()

            # perform step
            sim.step()

            # update buffers
            cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [2, 4])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_external_force_on_single_body(num_cubes, device):
    """Test application of external force on the base of the object.

    In this test, we apply a force equal to the weight of an object on the base of
    one of the objects. We check that the object does not move. For the other object,
    we do not apply any force and check that it falls down.

    We validate that this works when we apply the force in the global frame and in the local frame.
    """
    # Generate cubes scene
    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Play the simulator
        sim.reset()

        # Find bodies to apply the force
        body_ids, body_names = cube_object.find_bodies(".*")

        # Sample a force equal to the weight of the object
        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
        # Every 2nd cube should have a force applied to it
        external_wrench_b[0::2, :, 2] = 9.81 * wp.to_torch(cube_object.root_view.get_masses())[0]

        # Now we are ready!
        for i in range(5):
            # reset root state
            root_pose = wp.to_torch(cube_object.data.default_root_pose).clone()
            root_vel = wp.to_torch(cube_object.data.default_root_vel).clone()

            # need to shift the position of the cubes otherwise they will be on top of each other
            root_pose[:, :3] = origins
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)

            # reset object
            cube_object.reset()

            is_global = False
            if i % 2 == 0:
                is_global = True
                positions = wp.to_torch(cube_object.data.body_com_pos_w)[:, body_ids, :3]
            else:
                positions = None

            # apply force
            cube_object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                positions=positions,
                body_ids=body_ids,
                is_global=is_global,
            )
            # perform simulation
            for _ in range(5):
                # apply action to the object
                cube_object.write_data_to_sim()

                # perform step
                sim.step()

                # update buffers
                cube_object.update(sim.cfg.dt)

            # First object should still be at the same Z position (1.0)
            torch.testing.assert_close(
                wp.to_torch(cube_object.data.root_pos_w)[0::2, 2], torch.ones(num_cubes // 2, device=sim.device)
            )
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(wp.to_torch(cube_object.data.root_pos_w)[1::2, 2] < 1.0)


@pytest.mark.parametrize("num_cubes", [2, 4])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_external_force_on_single_body_at_position(num_cubes, device):
    """Test application of external force on the base of the object at a specific position.

    In this test, we apply a force equal to the weight of an object on the base of
    one of the objects at 1m in the Y direction, we check that the object rotates around it's X axis.
    For the other object, we do not apply any force and check that it falls down.

    We validate that this works when we apply the force in the global frame and in the local frame.
    """
    # Generate cubes scene
    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Play the simulator
        sim.reset()

        # Find bodies to apply the force
        body_ids, body_names = cube_object.find_bodies(".*")

        # Sample a force equal to the weight of the object
        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
        external_wrench_positions_b = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        # Every 2nd cube should have a force applied to it
        external_wrench_b[0::2, :, 2] = 500.0
        external_wrench_positions_b[0::2, :, 1] = 1.0

        # Desired force and torque
        desired_force = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        desired_force[0::2, :, 2] = 1000.0
        desired_torque = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        desired_torque[0::2, :, 0] = 1000.0
        # Now we are ready!
        for i in range(5):
            # reset root state
            root_pose = wp.to_torch(cube_object.data.default_root_pose).clone()
            root_vel = wp.to_torch(cube_object.data.default_root_vel).clone()

            # need to shift the position of the cubes otherwise they will be on top of each other
            root_pose[:, :3] = origins
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)

            # reset object
            cube_object.reset()

            is_global = False
            if i % 2 == 0:
                is_global = True
                body_com_pos_w = wp.to_torch(cube_object.data.body_com_pos_w)[:, body_ids, :3]
                external_wrench_positions_b[..., 0] = 0.0
                external_wrench_positions_b[..., 1] = 1.0
                external_wrench_positions_b[..., 2] = 0.0
                external_wrench_positions_b += body_com_pos_w
            else:
                external_wrench_positions_b[..., 0] = 0.0
                external_wrench_positions_b[..., 1] = 1.0
                external_wrench_positions_b[..., 2] = 0.0

            # apply force
            cube_object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                positions=external_wrench_positions_b,
                body_ids=body_ids,
                is_global=is_global,
            )
            cube_object.permanent_wrench_composer.add_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
                positions=external_wrench_positions_b,
                body_ids=body_ids,
                is_global=is_global,
            )
            torch.testing.assert_close(
                wp.to_torch(cube_object._permanent_wrench_composer.composed_force)[:, 0, :],
                desired_force[:, 0, :],
                rtol=1e-6,
                atol=1e-7,
            )
            torch.testing.assert_close(
                wp.to_torch(cube_object._permanent_wrench_composer.composed_torque)[:, 0, :],
                desired_torque[:, 0, :],
                rtol=1e-6,
                atol=1e-7,
            )
            # perform simulation
            for _ in range(5):
                # apply action to the object
                cube_object.write_data_to_sim()

                # perform step
                sim.step()

                # update buffers
                cube_object.update(sim.cfg.dt)

            # The first object should be rotating around it's X axis
            assert torch.all(torch.abs(wp.to_torch(cube_object.data.root_ang_vel_b)[0::2, 0]) > 0.1)
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(wp.to_torch(cube_object.data.root_pos_w)[1::2, 2] < 1.0)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_set_rigid_object_state(num_cubes, device):
    """Test setting the state of the rigid object.

    In this test, we set the state of the rigid object to a random state and check
    that the object is in that state after simulation. We set gravity to zero as
    we don't want any external forces acting on the object to ensure state remains static.
    """
    # Turn off gravity for this test as we don't want any external forces acting on the object
    # to ensure state remains static
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Play the simulator
        sim.reset()

        state_types = ["root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"]

        # Set each state type individually as they are dependent on each other
        for state_type_to_randomize in state_types:
            state_dict = {
                "root_pos_w": torch.zeros_like(wp.to_torch(cube_object.data.root_pos_w), device=sim.device),
                "root_quat_w": default_orientation(num=num_cubes, device=sim.device),
                "root_lin_vel_w": torch.zeros_like(wp.to_torch(cube_object.data.root_lin_vel_w), device=sim.device),
                "root_ang_vel_w": torch.zeros_like(wp.to_torch(cube_object.data.root_ang_vel_w), device=sim.device),
            }

            # Now we are ready!
            for _ in range(5):
                # reset object
                cube_object.reset()

                # Set random state
                if state_type_to_randomize == "root_quat_w":
                    state_dict[state_type_to_randomize] = random_orientation(num=num_cubes, device=sim.device)
                else:
                    state_dict[state_type_to_randomize] = torch.randn(num_cubes, 3, device=sim.device)

                # perform simulation
                for _ in range(5):
                    root_pose = torch.cat(
                        [state_dict["root_pos_w"], state_dict["root_quat_w"]],
                        dim=-1,
                    )
                    root_vel = torch.cat(
                        [state_dict["root_lin_vel_w"], state_dict["root_ang_vel_w"]],
                        dim=-1,
                    )
                    # reset root state
                    cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
                    cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)

                    sim.step()

                    # assert that set root quantities are equal to the ones set in the state_dict
                    for key, expected_value in state_dict.items():
                        value = wp.to_torch(getattr(cube_object.data, key))
                        torch.testing.assert_close(value, expected_value, rtol=1e-3, atol=1e-3)

                    cube_object.update(sim.cfg.dt)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_reset_rigid_object(num_cubes, device):
    """Test resetting the state of the rigid object."""
    with build_simulation_context(device=device, gravity_enabled=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Play the simulator
        sim.reset()

        for i in range(5):
            # perform rendering
            sim.step()

            # update object
            cube_object.update(sim.cfg.dt)

            # Move the object to a random position
            root_pose = wp.to_torch(cube_object.data.default_root_pose).clone()
            root_pose[:, :3] = torch.randn(num_cubes, 3, device=sim.device)

            # Random orientation
            root_pose[:, 3:7] = random_orientation(num=num_cubes, device=sim.device)
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = wp.to_torch(cube_object.data.default_root_vel).clone()
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)

            if i % 2 == 0:
                # reset object
                cube_object.reset()

                # Reset should zero external forces and torques
                assert not cube_object._instantaneous_wrench_composer.active
                assert not cube_object._permanent_wrench_composer.active
                assert torch.count_nonzero(wp.to_torch(cube_object._instantaneous_wrench_composer.composed_force)) == 0
                assert torch.count_nonzero(wp.to_torch(cube_object._instantaneous_wrench_composer.composed_torque)) == 0
                assert torch.count_nonzero(wp.to_torch(cube_object._permanent_wrench_composer.composed_force)) == 0
                assert torch.count_nonzero(wp.to_torch(cube_object._permanent_wrench_composer.composed_torque)) == 0


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_set_material_properties(num_cubes, device):
    """Test getting and setting material properties of rigid object."""
    with build_simulation_context(
        device=device, gravity_enabled=True, add_ground_plane=True, auto_add_lighting=True
    ) as sim:
        sim._app_control_on_stop_handle = None
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Play sim
        sim.reset()

        # Set material properties
        static_friction = torch.FloatTensor(num_cubes, 1).uniform_(0.4, 0.8)
        dynamic_friction = torch.FloatTensor(num_cubes, 1).uniform_(0.4, 0.8)
        restitution = torch.FloatTensor(num_cubes, 1).uniform_(0.0, 0.2)

        materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

        indices = torch.tensor(range(num_cubes), dtype=torch.int32)
        # Add friction to cube
        cube_object.root_view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32), wp.from_torch(indices, dtype=wp.int32)
        )

        # Simulate physics
        # perform rendering
        sim.step()
        # update object
        cube_object.update(sim.cfg.dt)

        # Get material properties
        materials_to_check = wp.to_torch(cube_object.root_view.get_material_properties())

        # Check if material properties are set correctly
        torch.testing.assert_close(materials_to_check.reshape(num_cubes, 3), materials)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_no_friction(num_cubes, device):
    """Test that a rigid object with no friction will maintain it's velocity when sliding across a plane."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)

        # Create ground plane with no friction
        cfg = sim_utils.GroundPlaneCfg(
            physics_material=materials.RigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.0,
            )
        )
        cfg.func("/World/GroundPlane", cfg)

        # Play sim
        sim.reset()

        # Set material friction properties to be all zero
        static_friction = torch.zeros(num_cubes, 1)
        dynamic_friction = torch.zeros(num_cubes, 1)
        restitution = torch.FloatTensor(num_cubes, 1).uniform_(0.0, 0.2)

        cube_object_materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)
        indices = torch.tensor(range(num_cubes), dtype=torch.int32)

        cube_object.root_view.set_material_properties(
            wp.from_torch(cube_object_materials, dtype=wp.float32), wp.from_torch(indices, dtype=wp.int32)
        )

        # Set initial velocity
        # Initial velocity in X to get the block moving
        initial_velocity = torch.zeros((num_cubes, 6), device=sim.cfg.device)
        initial_velocity[:, 0] = 0.1

        cube_object.write_root_velocity_to_sim_index(root_velocity=initial_velocity)

        # Simulate physics
        for _ in range(5):
            # perform rendering
            sim.step()
            # update object
            cube_object.update(sim.cfg.dt)

            # Non-deterministic when on GPU, so we use different tolerances
            if device == "cuda:0":
                tolerance = 1e-2
            else:
                tolerance = 1e-5

            torch.testing.assert_close(
                wp.to_torch(cube_object.data.root_lin_vel_w), initial_velocity[:, :3], rtol=1e-5, atol=tolerance
            )


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_with_static_friction(num_cubes, device):
    """Test that static friction applied to rigid object works as expected.

    This test works by applying a force to the object and checking if the object moves or not based on the
    mu (coefficient of static friction) value set for the object. We set the static friction to be non-zero and
    apply a force to the object. When the force applied is below mu, the object should not move. When the force
    applied is above mu, the object should move.
    """
    with build_simulation_context(device=device, dt=0.01, add_ground_plane=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=0.03125, device=device)

        # Create ground plane
        static_friction_coefficient = 0.5
        cfg = sim_utils.GroundPlaneCfg(
            physics_material=materials.RigidBodyMaterialCfg(
                static_friction=static_friction_coefficient,
                dynamic_friction=static_friction_coefficient,  # This shouldn't be required but is due to a bug in PhysX
            )
        )
        cfg.func("/World/GroundPlane", cfg)

        # Play sim
        sim.reset()

        # Set static friction to be non-zero
        # Dynamic friction also needs to be zero due to a bug in PhysX
        static_friction = torch.Tensor([[static_friction_coefficient]] * num_cubes)
        dynamic_friction = torch.Tensor([[static_friction_coefficient]] * num_cubes)
        restitution = torch.zeros(num_cubes, 1)

        cube_object_materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

        indices = torch.tensor(range(num_cubes), dtype=torch.int32)

        # Add friction to cube
        cube_object.root_view.set_material_properties(
            wp.from_torch(cube_object_materials, dtype=wp.float32), wp.from_torch(indices, dtype=wp.int32)
        )

        # let everything settle
        for _ in range(100):
            sim.step()
            cube_object.update(sim.cfg.dt)
        cube_object.write_root_velocity_to_sim_index(root_velocity=torch.zeros((num_cubes, 6), device=sim.device))
        cube_mass = wp.to_torch(cube_object.root_view.get_masses())
        gravity_magnitude = abs(sim.cfg.gravity[2])
        # 2 cases: force applied is below and above mu
        # below mu: block should not move as the force applied is <= mu
        # above mu: block should move as the force applied is > mu
        for force in "below_mu", "above_mu":
            # set initial velocity to zero
            cube_object.write_root_velocity_to_sim_index(root_velocity=torch.zeros((num_cubes, 6), device=sim.device))

            external_wrench_b = torch.zeros((num_cubes, 1, 6), device=sim.device)
            if force == "below_mu":
                external_wrench_b[..., 0] = static_friction_coefficient * cube_mass * gravity_magnitude * 0.99
            else:
                external_wrench_b[..., 0] = static_friction_coefficient * cube_mass * gravity_magnitude * 1.01

            cube_object.permanent_wrench_composer.set_forces_and_torques_index(
                forces=external_wrench_b[..., :3],
                torques=external_wrench_b[..., 3:],
            )

            # Get root state
            initial_root_pos = wp.to_torch(cube_object.data.root_pos_w).clone()
            # Simulate physics
            for _ in range(200):
                # apply the wrench
                cube_object.write_data_to_sim()
                sim.step()
                # update object
                cube_object.update(sim.cfg.dt)
                if force == "below_mu":
                    # Assert that the block has not moved
                    torch.testing.assert_close(
                        wp.to_torch(cube_object.data.root_pos_w), initial_root_pos, rtol=2e-3, atol=2e-3
                    )
            if force == "above_mu":
                assert (wp.to_torch(cube_object.data.root_pos_w)[..., 0] - initial_root_pos[..., 0] > 0.02).all()


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_with_restitution(num_cubes, device):
    """Test that restitution when applied to rigid object works as expected.

    This test works by dropping a block from a height and checking if the block bounces or not based on the
    restitution value set for the object. We set the restitution to be non-zero and drop the block from a height.
    When the restitution is 0, the block should not bounce. When the restitution is between 0 and 1, the block
    should bounce with less energy.
    """
    for expected_collision_type in "partially_elastic", "inelastic":
        with build_simulation_context(device=device, add_ground_plane=False, auto_add_lighting=True) as sim:
            sim._app_control_on_stop_handle = None
            cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=1.0, device=device)

            # Set static friction to be non-zero
            if expected_collision_type == "inelastic":
                restitution_coefficient = 0.0
            elif expected_collision_type == "partially_elastic":
                restitution_coefficient = 0.5

            # Create ground plane such that has a restitution of 1.0 (perfectly elastic collision)
            cfg = sim_utils.GroundPlaneCfg(
                physics_material=materials.RigidBodyMaterialCfg(
                    restitution=restitution_coefficient,
                )
            )
            cfg.func("/World/GroundPlane", cfg)

            indices = torch.tensor(range(num_cubes), dtype=torch.int32)

            # Play sim
            sim.reset()

            root_pose = torch.zeros(num_cubes, 7, device=sim.device)
            root_pose[:, 3] = 1.0  # To make orientation a quaternion
            for i in range(num_cubes):
                root_pose[i, 1] = 1.0 * i
            root_pose[:, 2] = 1.0  # Set an initial drop height
            root_vel = torch.zeros(num_cubes, 6, device=sim.device)
            root_vel[:, 2] = -1.0  # Set an initial downward velocity

            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            cube_object.write_root_velocity_to_sim_index(root_velocity=root_vel)

            static_friction = torch.zeros(num_cubes, 1)
            dynamic_friction = torch.zeros(num_cubes, 1)
            restitution = torch.Tensor([[restitution_coefficient]] * num_cubes)

            cube_object_materials = torch.cat([static_friction, dynamic_friction, restitution], dim=-1)

            # Add restitution to cube
            cube_object.root_view.set_material_properties(
                wp.from_torch(cube_object_materials, dtype=wp.float32), wp.from_torch(indices, dtype=wp.int32)
            )

            curr_z_velocity = wp.to_torch(cube_object.data.root_lin_vel_w)[:, 2].clone()

            for _ in range(100):
                sim.step()

                # update object
                cube_object.update(sim.cfg.dt)
                curr_z_velocity = wp.to_torch(cube_object.data.root_lin_vel_w)[:, 2].clone()

                if expected_collision_type == "inelastic":
                    # assert that the block has not bounced by checking that the z velocity is less than or equal to 0
                    assert (curr_z_velocity <= 0.0).all()

                if torch.all(curr_z_velocity <= 0.0):
                    # Still in the air
                    prev_z_velocity = curr_z_velocity
                else:
                    # collision has happened, exit the for loop
                    break

            if expected_collision_type == "partially_elastic":
                # Assert that the block has lost some energy by checking that the z velocity is less
                assert torch.all(torch.le(abs(curr_z_velocity), abs(prev_z_velocity)))
                assert (curr_z_velocity > 0.0).all()


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_rigid_body_set_mass(num_cubes, device):
    """Test getting and setting mass of rigid object."""
    with build_simulation_context(
        device=device, gravity_enabled=False, add_ground_plane=True, auto_add_lighting=True
    ) as sim:
        sim._app_control_on_stop_handle = None
        # Create a scene with random cubes
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=1.0, device=device)

        # Play sim
        sim.reset()

        # Get masses before increasing
        original_masses = wp.to_torch(cube_object.root_view.get_masses())

        assert original_masses.shape == (num_cubes, 1)

        # Randomize mass of the object
        masses = original_masses + torch.FloatTensor(num_cubes, 1).uniform_(4, 8)

        indices = torch.tensor(range(num_cubes), dtype=torch.int32)

        # Add friction to cube
        cube_object.root_view.set_masses(
            wp.from_torch(masses, dtype=wp.float32), wp.from_torch(indices, dtype=wp.int32)
        )

        torch.testing.assert_close(wp.to_torch(cube_object.root_view.get_masses()), masses)

        # Simulate physics
        # perform rendering
        sim.step()
        # update object
        cube_object.update(sim.cfg.dt)

        masses_to_check = wp.to_torch(cube_object.root_view.get_masses())

        # Check if mass is set correctly
        torch.testing.assert_close(masses, masses_to_check)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("gravity_enabled", [True, False])
@pytest.mark.isaacsim_ci
def test_gravity_vec_w(num_cubes, device, gravity_enabled):
    """Test that gravity vector direction is set correctly for the rigid object."""
    with build_simulation_context(device=device, gravity_enabled=gravity_enabled) as sim:
        sim._app_control_on_stop_handle = None
        # Create a scene with random cubes
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)

        # Obtain gravity direction
        if gravity_enabled:
            gravity_dir = (0.0, 0.0, -1.0)
        else:
            gravity_dir = (0.0, 0.0, 0.0)

        # Play sim
        sim.reset()

        # Check that gravity is set correctly
        assert wp.to_torch(cube_object.data.GRAVITY_VEC_W)[0, 0] == gravity_dir[0]
        assert wp.to_torch(cube_object.data.GRAVITY_VEC_W)[0, 1] == gravity_dir[1]
        assert wp.to_torch(cube_object.data.GRAVITY_VEC_W)[0, 2] == gravity_dir[2]

        # Simulate physics
        for _ in range(2):
            # perform rendering
            sim.step()
            # update object
            cube_object.update(sim.cfg.dt)

            # Expected gravity value is the acceleration of the body
            gravity = torch.zeros(num_cubes, 1, 6, device=device)
            if gravity_enabled:
                gravity[:, :, 2] = -9.81
            # Check the body accelerations are correct
            torch.testing.assert_close(wp.to_torch(cube_object.data.body_acc_w), gravity)


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.isaacsim_ci
@flaky(max_runs=3, min_passes=1)
def test_body_root_state_properties(num_cubes, device, with_offset):
    """Test the root_com_state_w, root_link_state_w, body_com_state_w, and body_link_state_w properties."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32)

        # Play sim
        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        if with_offset:
            offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        else:
            offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        com = wp.to_torch(cube_object.root_view.get_coms())
        com[..., :3] = offset.to("cpu")
        cube_object.root_view.set_coms(wp.from_torch(com, dtype=wp.float32), wp.from_torch(env_idx, dtype=wp.int32))

        # check ceter of mass has been set
        torch.testing.assert_close(wp.to_torch(cube_object.root_view.get_coms()), com)

        # random z spin velocity
        spin_twist = torch.zeros(6, device=device)
        spin_twist[5] = torch.randn(1, device=device)

        # Simulate physics
        for _ in range(100):
            # spin the object around Z axis (com)
            cube_object.write_root_velocity_to_sim_index(root_velocity=spin_twist.repeat(num_cubes, 1))
            # perform rendering
            sim.step()
            # update object
            cube_object.update(sim.cfg.dt)

            # get state properties
            root_link_pose_w = wp.to_torch(cube_object.data.root_link_pose_w)
            root_link_vel_w = wp.to_torch(cube_object.data.root_link_vel_w)
            root_com_pose_w = wp.to_torch(cube_object.data.root_com_pose_w)
            root_com_vel_w = wp.to_torch(cube_object.data.root_com_vel_w)
            body_link_pose_w = wp.to_torch(cube_object.data.body_link_pose_w)
            body_link_vel_w = wp.to_torch(cube_object.data.body_link_vel_w)
            body_com_pose_w = wp.to_torch(cube_object.data.body_com_pose_w)
            body_com_vel_w = wp.to_torch(cube_object.data.body_com_vel_w)

            # if offset is [0,0,0] all root_state_%_w will match and all body_%_w will match
            if not with_offset:
                torch.testing.assert_close(root_link_pose_w, root_com_pose_w)
                torch.testing.assert_close(root_com_vel_w, root_link_vel_w)
                torch.testing.assert_close(root_link_pose_w, root_link_pose_w)
                torch.testing.assert_close(root_com_vel_w, root_link_vel_w)
                torch.testing.assert_close(body_link_pose_w, body_com_pose_w)
                torch.testing.assert_close(body_com_vel_w, body_link_vel_w)
                torch.testing.assert_close(body_link_pose_w, body_link_pose_w)
                torch.testing.assert_close(body_com_vel_w, body_link_vel_w)
            else:
                # cubes are spinning around center of mass
                # position will not match
                # center of mass position will be constant (i.e. spinning around com)
                torch.testing.assert_close(env_pos + offset, root_com_pose_w[..., :3])
                torch.testing.assert_close(env_pos + offset, body_com_pose_w[..., :3].squeeze(-2))
                # link position will be moving but should stay constant away from center of mass
                root_link_state_pos_rel_com = quat_apply_inverse(
                    root_link_pose_w[..., 3:],
                    root_link_pose_w[..., :3] - root_com_pose_w[..., :3],
                )
                torch.testing.assert_close(-offset, root_link_state_pos_rel_com)
                body_link_state_pos_rel_com = quat_apply_inverse(
                    body_link_pose_w[..., 3:],
                    body_link_pose_w[..., :3] - body_com_pose_w[..., :3],
                )
                torch.testing.assert_close(-offset, body_link_state_pos_rel_com.squeeze(-2))

                # orientation of com will be a constant rotation from link orientation
                com_quat_b = wp.to_torch(cube_object.data.body_com_quat_b)
                com_quat_w = quat_mul(body_link_pose_w[..., 3:], com_quat_b)
                torch.testing.assert_close(com_quat_w, body_com_pose_w[..., 3:])
                torch.testing.assert_close(com_quat_w.squeeze(-2), root_com_pose_w[..., 3:])

                # orientation of link will match root state will always match
                torch.testing.assert_close(root_link_pose_w[..., 3:], root_link_pose_w[..., 3:])
                torch.testing.assert_close(body_link_pose_w[..., 3:], body_link_pose_w[..., 3:])

                # lin_vel will not match
                # center of mass vel will be constant (i.e. spinning around com)
                torch.testing.assert_close(torch.zeros_like(root_com_vel_w[..., :3]), root_com_vel_w[..., :3])
                torch.testing.assert_close(torch.zeros_like(body_com_vel_w[..., :3]), body_com_vel_w[..., :3])
                # link frame will be moving, and should be equal to input angular velocity cross offset
                lin_vel_rel_root_gt = quat_apply_inverse(root_link_pose_w[..., 3:], root_link_vel_w[..., :3])
                lin_vel_rel_body_gt = quat_apply_inverse(body_link_pose_w[..., 3:], body_link_vel_w[..., :3])
                lin_vel_rel_gt = torch.linalg.cross(spin_twist.repeat(num_cubes, 1)[..., 3:], -offset)
                torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_root_gt, atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_body_gt.squeeze(-2), atol=1e-4, rtol=1e-4)

                # ang_vel will always match
                torch.testing.assert_close(root_com_vel_w[..., 3:], root_com_vel_w[..., 3:])
                torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
                torch.testing.assert_close(body_com_vel_w[..., 3:], body_com_vel_w[..., 3:])
                torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("state_location", ["com", "link"])
@pytest.mark.isaacsim_ci
def test_write_root_state(num_cubes, device, with_offset, state_location):
    """Test the setters for root_state using both the link frame and center of mass as reference frame."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32)

        # Play sim
        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        if with_offset:
            offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        else:
            offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        com = wp.to_torch(cube_object.root_view.get_coms())
        com[..., :3] = offset.to("cpu")
        cube_object.root_view.set_coms(wp.from_torch(com, dtype=wp.float32), wp.from_torch(env_idx, dtype=wp.int32))

        # check center of mass has been set
        torch.testing.assert_close(wp.to_torch(cube_object.root_view.get_coms()), com)

        rand_state = torch.zeros(num_cubes, 13, device=device)
        rand_state[..., :7] = wp.to_torch(cube_object.data.default_root_pose)
        rand_state[..., :3] += env_pos
        # make quaternion a unit vector
        rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

        env_idx = env_idx.to(device)
        for i in range(10):
            # perform step
            sim.step()
            # update buffers
            cube_object.update(sim.cfg.dt)

            if state_location == "com":
                if i % 2 == 0:
                    cube_object.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7])
                    cube_object.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
                else:
                    cube_object.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_idx)
                    cube_object.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_idx)
            elif state_location == "link":
                if i % 2 == 0:
                    cube_object.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7])
                    cube_object.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
                else:
                    cube_object.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_idx)
                    cube_object.write_root_link_velocity_to_sim_index(
                        root_velocity=rand_state[..., 7:], env_ids=env_idx
                    )

            if state_location == "com":
                torch.testing.assert_close(rand_state[..., :7], wp.to_torch(cube_object.data.root_com_pose_w))
                torch.testing.assert_close(rand_state[..., 7:], wp.to_torch(cube_object.data.root_com_vel_w))
            elif state_location == "link":
                torch.testing.assert_close(rand_state[..., :7], wp.to_torch(cube_object.data.root_link_pose_w))
                torch.testing.assert_close(rand_state[..., 7:], wp.to_torch(cube_object.data.root_link_vel_w))


@pytest.mark.parametrize("num_cubes", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True])
@pytest.mark.parametrize("state_location", ["com", "link", "root"])
@pytest.mark.isaacsim_ci
def test_write_state_functions_data_consistency(num_cubes, device, with_offset, state_location):
    """Test the setters for root_state using both the link frame and center of mass as reference frame."""
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32)

        # Play sim
        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        if with_offset:
            offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
        else:
            offset = torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        com = wp.to_torch(cube_object.root_view.get_coms())
        com[..., :3] = offset.to("cpu")
        cube_object.root_view.set_coms(wp.from_torch(com, dtype=wp.float32), wp.from_torch(env_idx, dtype=wp.int32))

        # check ceter of mass has been set
        torch.testing.assert_close(wp.to_torch(cube_object.root_view.get_coms()), com)

        rand_state = torch.rand(num_cubes, 13, device=device)
        rand_state[..., :3] += env_pos
        # make quaternion a unit vector
        rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

        env_idx = env_idx.to(device)

        # perform step
        sim.step()
        # update buffers
        cube_object.update(sim.cfg.dt)

        if state_location == "com":
            cube_object.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7])
            cube_object.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
        elif state_location == "link":
            cube_object.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7])
            cube_object.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
        elif state_location == "root":
            cube_object.write_root_pose_to_sim_index(root_pose=rand_state[..., :7])
            cube_object.write_root_velocity_to_sim_index(root_velocity=rand_state[..., 7:])

        if state_location == "com":
            root_com_pose_w = wp.to_torch(cube_object.data.root_com_pose_w)
            root_com_vel_w = wp.to_torch(cube_object.data.root_com_vel_w)
            body_com_pose_b = wp.to_torch(cube_object.data.body_com_pose_b)
            expected_root_link_pos, expected_root_link_quat = combine_frame_transforms(
                root_com_pose_w[:, :3],
                root_com_pose_w[:, 3:],
                quat_rotate(quat_inv(body_com_pose_b[:, 0, 3:7]), -body_com_pose_b[:, 0, :3]),
                quat_inv(body_com_pose_b[:, 0, 3:7]),
            )
            expected_root_link_pose = torch.cat((expected_root_link_pos, expected_root_link_quat), dim=1)
            root_link_pose_w = wp.to_torch(cube_object.data.root_link_pose_w)
            root_link_vel_w = wp.to_torch(cube_object.data.root_link_vel_w)
            # test both root_pose and root_link successfully updated when root_com updates
            torch.testing.assert_close(expected_root_link_pose, root_link_pose_w)
            # skip lin_vel because it differs from link frame, this should be fine because we are only checking
            # if velocity update is triggered, which can be determined by comparing angular velocity
            torch.testing.assert_close(root_com_vel_w[:, 3:], root_link_vel_w[:, 3:])
            torch.testing.assert_close(expected_root_link_pose, root_link_pose_w)
            torch.testing.assert_close(root_com_vel_w[:, 3:], wp.to_torch(cube_object.data.root_com_vel_w)[:, 3:])
        elif state_location == "link":
            root_link_pose_w = wp.to_torch(cube_object.data.root_link_pose_w)
            root_link_vel_w = wp.to_torch(cube_object.data.root_link_vel_w)
            body_com_pose_b = wp.to_torch(cube_object.data.body_com_pose_b)
            expected_com_pos, expected_com_quat = combine_frame_transforms(
                root_link_pose_w[:, :3],
                root_link_pose_w[:, 3:],
                body_com_pose_b[:, 0, :3],
                body_com_pose_b[:, 0, 3:7],
            )
            expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1)
            root_com_pose_w = wp.to_torch(cube_object.data.root_com_pose_w)
            root_com_vel_w = wp.to_torch(cube_object.data.root_com_vel_w)
            # test both root_pose and root_com successfully updated when root_link updates
            torch.testing.assert_close(expected_com_pose, root_com_pose_w)
            # skip lin_vel because it differs from link frame, this should be fine because we are only checking
            # if velocity update is triggered, which can be determined by comparing angular velocity
            torch.testing.assert_close(root_link_vel_w[:, 3:], root_com_vel_w[:, 3:])
            torch.testing.assert_close(root_link_pose_w, wp.to_torch(cube_object.data.root_link_pose_w))
            torch.testing.assert_close(root_link_vel_w[:, 3:], wp.to_torch(cube_object.data.root_com_vel_w)[:, 3:])
        elif state_location == "root":
            root_link_pose_w = wp.to_torch(cube_object.data.root_link_pose_w)
            root_com_vel_w = wp.to_torch(cube_object.data.root_com_vel_w)
            body_com_pose_b = wp.to_torch(cube_object.data.body_com_pose_b)
            expected_com_pos, expected_com_quat = combine_frame_transforms(
                root_link_pose_w[:, :3],
                root_link_pose_w[:, 3:],
                body_com_pose_b[:, 0, :3],
                body_com_pose_b[:, 0, 3:7],
            )
            expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1)
            root_com_pose_w = wp.to_torch(cube_object.data.root_com_pose_w)
            root_link_vel_w = wp.to_torch(cube_object.data.root_link_vel_w)
            # test both root_com and root_link successfully updated when root_pose updates
            torch.testing.assert_close(expected_com_pose, root_com_pose_w)
            torch.testing.assert_close(root_com_vel_w, wp.to_torch(cube_object.data.root_com_vel_w))
            torch.testing.assert_close(root_link_pose_w, wp.to_torch(cube_object.data.root_link_pose_w))
            torch.testing.assert_close(root_com_vel_w[:, 3:], root_link_vel_w[:, 3:])


@pytest.mark.isaacsim_ci
def test_warmup_attach_stage_not_called_for_cpu():
    """Regression test: attach_stage() must not be called for CPU in _warmup_and_create_views().

    Bug (commit 0ba9c5cb3b): ``PhysxManager._warmup_and_create_views()`` called
    ``_physx_sim.attach_stage()`` unconditionally before ``force_load_physics_from_usd()``.
    These are two alternative initialization patterns; combining them causes
    double-initialization that corrupts the CPU MBP broadphase, producing
    non-deterministic collision failures (objects passing through surfaces).

    Fix: guard ``attach_stage()`` with ``if is_gpu:`` — it is only required by the
    GPU pipeline, which needs explicit stage attachment before the physics load step.
    The CPU pipeline attaches implicitly via ``force_load_physics_from_usd()``.

    This test verifies the guard is in place by monkeypatching ``attach_stage`` on
    the PhysX simulation interface and asserting it is *not* called during CPU warmup.
    The simulation test itself (1 cube falling onto a ground plane) is intentionally
    omitted here because the MBP corruption is non-deterministic and depends on scene
    complexity (multiple dynamic actors on a mesh collider), making it unreliable as a
    unit test assertion.
    """
    from unittest.mock import MagicMock

    from isaaclab_physx.physics import PhysxManager

    with build_simulation_context(device="cpu", add_ground_plane=True, dt=0.01, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        generate_cubes_scene(num_cubes=1, height=1.0, device="cpu")

        # IPhysxSimulation is a C++ binding whose attributes are read-only, so we cannot
        # assign to _physx_sim.attach_stage directly.  Instead, replace the class-level
        # reference with a MagicMock that wraps the real object so all other calls still
        # work, then restore it in the finally block.
        original_physx_sim = PhysxManager._physx_sim
        spy = MagicMock(wraps=original_physx_sim)
        PhysxManager._physx_sim = spy
        try:
            sim.reset()
        finally:
            PhysxManager._physx_sim = original_physx_sim

        assert spy.attach_stage.call_count == 0, (
            f"attach_stage() was called {spy.attach_stage.call_count} time(s) during CPU warmup. "
            f"This indicates the CPU MBP broadphase double-initialization regression is present: "
            f"attach_stage() + force_load_physics_from_usd() must not be combined for CPU."
        )
