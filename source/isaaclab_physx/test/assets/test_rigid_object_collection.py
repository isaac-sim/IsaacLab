# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none


"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from isaaclab.test.utils import resolve_test_sim_device, test_devices

# launch omniverse app
simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

"""Rest everything follows."""

import sys
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import RigidObjectCollection
from isaaclab_physx.physics import IsaacEvents, PhysxManager

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    combine_frame_transforms,
    default_orientation,
    quat_apply_inverse,
    quat_mul,
    random_orientation,
)


def generate_cubes_scene(
    num_envs: int = 1,
    num_cubes: int = 1,
    height=1.0,
    has_api: bool = True,
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
) -> tuple[RigidObjectCollection, torch.Tensor]:
    """Generate a scene with the provided number of cubes.

    Args:
        num_envs: Number of envs to generate.
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        has_api: Whether the cubes have a rigid body API on them.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.

    Returns:
        A tuple containing the rigid object representing the cubes and the origins of the cubes.

    """
    origins = torch.tensor([(i * 3.0, 0, height) for i in range(num_envs)]).to(device)
    # Create Top-level Xforms, one for each cube
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

    # Resolve spawn configuration
    if has_api:
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic_enabled),
        )
    else:
        # since no rigid body properties defined, this is just a static collider
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        )

    # create the rigid object configs
    cube_config_dict = {}
    for i in range(num_cubes):
        cube_object_cfg = RigidObjectCfg(
            prim_path=f"/World/Table_[^/]*/Object_{i}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3 * i, height)),
        )
        cube_config_dict[f"cube_{i}"] = cube_object_cfg
    # create the rigid object collection
    cube_object_collection_cfg = RigidObjectCollectionCfg(rigid_objects=cube_config_dict)
    cube_object_colection = RigidObjectCollection(cfg=cube_object_collection_cfg)

    return cube_object_colection, origins


@pytest.fixture
def sim(request):
    """Create simulation context with the specified device."""
    device = request.getfixturevalue("device")
    if "gravity_enabled" in request.fixturenames:
        gravity_enabled = request.getfixturevalue("gravity_enabled")
    else:
        gravity_enabled = True  # default to gravity enabled
    with build_simulation_context(device=device, auto_add_lighting=True, gravity_enabled=gravity_enabled) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.parametrize(("num_envs", "num_cubes"), [(1, 1), (2, 3)])
@pytest.mark.parametrize("device", test_devices())
def test_initialization(sim, num_envs, num_cubes, device):
    """Test initialization for prim with rigid body API at the provided prim path."""
    object_collection, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(object_collection) < 10

    # Play sim
    sim.reset()

    # Check if object is initialized
    assert object_collection.is_initialized
    assert len(object_collection.body_names) == num_cubes

    # Check buffers that exist and have correct shapes
    assert object_collection.data.body_link_pos_w.torch.shape == (num_envs, num_cubes, 3)
    assert object_collection.data.body_link_quat_w.torch.shape == (num_envs, num_cubes, 4)
    assert object_collection.data.body_mass.torch.shape == (num_envs, num_cubes)
    assert object_collection.data.body_inertia.torch.shape == (num_envs, num_cubes, 9)

    # Simulate physics
    for _ in range(2):
        sim.step()
        object_collection.update(sim.cfg.dt)


@pytest.mark.parametrize("event_as_dict", [False, True])
@pytest.mark.parametrize("device", test_devices())
def test_prim_deletion_event_invalidates_collection(sim, event_as_dict, device):
    """A matching PhysX deletion event invalidates the collection and releases callbacks."""
    object_collection, _ = generate_cubes_scene(num_envs=1, num_cubes=1, device=device)
    sim.reset()

    def make_event(prim_path):
        payload = {"prim_path": prim_path}
        return payload if event_as_dict else SimpleNamespace(payload=payload)

    object_collection._on_prim_deletion(make_event("/World/Other"))
    assert object_collection.is_initialized
    assert object_collection.root_view is not None

    object_collection._on_prim_deletion(make_event("/World/Table_0/Object_0"))

    assert not object_collection.is_initialized
    assert object_collection.root_view is None
    assert object_collection._initialize_handle is None
    assert object_collection._invalidate_initialize_handle is None
    assert object_collection._prim_deletion_handle is None


@pytest.mark.parametrize("device", test_devices())
def test_prim_deletion_message_bus_invalidates_collection(sim, device):
    """The PhysX message bus passes a deletion event that invalidates the collection."""
    object_collection, _ = generate_cubes_scene(num_envs=1, num_cubes=1, device=device)
    sim.reset()

    PhysxManager._message_bus.dispatch_event(
        IsaacEvents.PRIM_DELETION.value, payload={"prim_path": "/World/Table_0/Object_0"}
    )
    PhysxManager.raise_callback_exception_if_any()

    assert not object_collection.is_initialized
    assert object_collection.root_view is None
    assert object_collection._initialize_handle is None
    assert object_collection._invalidate_initialize_handle is None
    assert object_collection._prim_deletion_handle is None


@pytest.mark.parametrize("device", test_devices())
def test_prim_deletion_dict_root_invalidates_collection(sim, device):
    """A dictionary root-deletion payload invalidates the collection."""
    object_collection, _ = generate_cubes_scene(num_envs=1, num_cubes=1, device=device)
    sim.reset()

    object_collection._on_prim_deletion({"prim_path": "/"})

    assert not object_collection.is_initialized
    assert object_collection.root_view is None
    assert object_collection._initialize_handle is None
    assert object_collection._invalidate_initialize_handle is None
    assert object_collection._prim_deletion_handle is None


@pytest.mark.parametrize("device", test_devices())
def test_prim_deletion_clears_callbacks_when_invalidation_fails(sim, device, monkeypatch):
    """A matching deletion releases callbacks when collection invalidation fails."""
    object_collection, _ = generate_cubes_scene(num_envs=1, num_cubes=1, device=device)
    sim.reset()

    def raise_invalidation_error(_event):
        raise RuntimeError("invalidation failed")

    monkeypatch.setattr(object_collection, "_invalidate_initialize_callback", raise_invalidation_error)

    with pytest.raises(RuntimeError, match="invalidation failed"):
        object_collection._on_prim_deletion({"prim_path": "/World/Table_0/Object_0"})

    assert object_collection._initialize_handle is None
    assert object_collection._invalidate_initialize_handle is None
    assert object_collection._prim_deletion_handle is None


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [False])
def test_subset_write_reaches_selected_view_entry(sim, device, gravity_enabled):
    """A write to one (env, body) cell must move only that object in the simulation."""
    object_collection, _ = generate_cubes_scene(num_envs=2, num_cubes=3, device=device)

    # Play sim
    sim.reset()
    sim.step()
    object_collection.update(sim.cfg.dt)

    initial_pose = object_collection.data.body_link_pose_w.torch.clone()
    new_pose = initial_pose[1:2, 2:3].clone()
    new_pose[..., 2] += 0.5
    object_collection.write_body_link_pose_to_sim_index(body_poses=new_pose, env_ids=[1], body_ids=[2])

    # Read the pose back from the simulation so that a wrong view index cannot hide in the data buffer
    sim.step()
    object_collection.update(sim.cfg.dt)

    expected_pose = initial_pose.clone()
    expected_pose[1, 2] = new_pose[0, 0]
    torch.testing.assert_close(object_collection.data.body_link_pose_w.torch, expected_pose)


@pytest.mark.parametrize("num_envs", [2])
@pytest.mark.parametrize("num_cubes", [3])
@pytest.mark.parametrize("device", test_devices())
def test_initialization_with_kinematic_enabled(sim, num_envs, num_cubes, device):
    """Test that initialization for prim with kinematic flag enabled."""
    object_collection, origins = generate_cubes_scene(
        num_envs=num_envs, num_cubes=num_cubes, kinematic_enabled=True, device=device
    )

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(object_collection) < 10

    # Play sim
    sim.reset()

    # Check if object is initialized
    assert object_collection.is_initialized
    assert len(object_collection.body_names) == num_cubes

    # Check buffers that exist and have correct shapes
    assert object_collection.data.body_link_pos_w.torch.shape == (num_envs, num_cubes, 3)
    assert object_collection.data.body_link_quat_w.torch.shape == (num_envs, num_cubes, 4)

    # Simulate physics
    for _ in range(2):
        sim.step()
        object_collection.update(sim.cfg.dt)
        # check that the object is kinematic
        default_body_pose = object_collection.data.default_body_pose.torch.clone()
        default_body_vel = object_collection.data.default_body_vel.torch.clone()
        default_body_pose[..., :3] += origins.unsqueeze(1)
        torch.testing.assert_close(object_collection.data.body_link_pose_w.torch, default_body_pose)
        torch.testing.assert_close(object_collection.data.body_link_vel_w.torch, default_body_vel)


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_initialization_with_no_rigid_body(sim, num_cubes, device):
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    object_collection, _ = generate_cubes_scene(num_cubes=num_cubes, has_api=False, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(object_collection) < 10

    # Play sim
    with pytest.raises(RuntimeError):
        sim.reset()


@pytest.mark.parametrize("num_envs", [2])
@pytest.mark.parametrize("num_cubes", [4])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_single_body(sim, num_envs, num_cubes, device):
    """Test application of external force on the base of the object."""
    object_collection, origins = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
    sim.reset()

    # find objects to apply the force
    object_ids, object_names = object_collection.find_bodies(".*")

    # Sample a force equal to the weight of the object
    external_wrench_b = torch.zeros(object_collection.num_instances, len(object_ids), 6, device=sim.device)
    # Every 2nd cube should have a force applied to it
    external_wrench_b[:, 0::2, 2] = 9.81 * object_collection.data.body_mass.torch[:, 0::2]

    for i in range(5):
        # reset object state
        body_pose = object_collection.data.default_body_pose.torch.clone()
        body_vel = object_collection.data.default_body_vel.torch.clone()
        # need to shift the position of the cubes otherwise they will be on top of each other
        body_pose[..., :2] += origins.unsqueeze(1)[..., :2]
        object_collection.write_body_link_pose_to_sim_index(body_poses=body_pose)
        object_collection.write_body_com_velocity_to_sim_index(body_velocities=body_vel)
        # reset object
        object_collection.reset()

        is_global = False
        if i % 2 == 0:
            positions = object_collection.data.body_link_pos_w.torch[:, object_ids, :3]
            is_global = True
        else:
            positions = None

        # apply force
        object_collection.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=positions,
            body_ids=object_ids,
            env_ids=None,
            is_global=is_global,
        )
        for _ in range(10):
            # write data to sim
            object_collection.write_data_to_sim()
            # step sim
            sim.step()
            # update object collection
            object_collection.update(sim.cfg.dt)

        # First object should still be at the same Z position (1.0)
        torch.testing.assert_close(
            object_collection.data.body_link_pos_w.torch[:, 0::2, 2],
            torch.ones_like(object_collection.data.body_link_pos_w.torch[:, 0::2, 2]),
        )
        # Second object should have fallen, so it's Z height should be less than initial height of 1.0
        assert torch.all(object_collection.data.body_link_pos_w.torch[:, 1::2, 2] < 1.0)


@pytest.mark.parametrize("num_envs", [2])
@pytest.mark.parametrize("num_cubes", [4])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_single_body_at_position(sim, num_envs, num_cubes, device):
    """Test application of external force on the base of the object at a specific position.

    In this test, we apply a force equal to the weight of an object on the base of
    one of the objects at 1m in the Y direction, we check that the object rotates around it's X axis.
    For the other object, we do not apply any force and check that it falls down.
    """
    object_collection, origins = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
    sim.reset()

    # find objects to apply the force
    object_ids, object_names = object_collection.find_bodies(".*")

    # Sample a force equal to the weight of the object
    external_wrench_b = torch.zeros(object_collection.num_instances, len(object_ids), 6, device=sim.device)
    external_wrench_positions_b = torch.zeros(object_collection.num_instances, len(object_ids), 3, device=sim.device)
    # Every 2nd cube should have a force applied to it
    external_wrench_b[:, 0::2, 2] = 500.0
    external_wrench_positions_b[:, 0::2, 1] = 1.0

    # Desired force and torque
    for i in range(5):
        # reset object state
        body_pose = object_collection.data.default_body_pose.torch.clone()
        body_vel = object_collection.data.default_body_vel.torch.clone()
        # need to shift the position of the cubes otherwise they will be on top of each other
        body_pose[..., :2] += origins.unsqueeze(1)[..., :2]
        object_collection.write_body_link_pose_to_sim_index(body_poses=body_pose)
        object_collection.write_body_com_velocity_to_sim_index(body_velocities=body_vel)
        # reset object
        object_collection.reset()

        is_global = False
        if i % 2 == 0:
            body_com_pos_w = object_collection.data.body_link_pos_w.torch[:, object_ids, :3]
            external_wrench_positions_b[..., 0] = 0.0
            external_wrench_positions_b[..., 1] = 1.0
            external_wrench_positions_b[..., 2] = 0.0
            external_wrench_positions_b += body_com_pos_w
            is_global = True
        else:
            external_wrench_positions_b[..., 0] = 0.0
            external_wrench_positions_b[..., 1] = 1.0
            external_wrench_positions_b[..., 2] = 0.0

        # apply force
        object_collection.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            body_ids=object_ids,
            env_ids=None,
            is_global=is_global,
        )
        object_collection.permanent_wrench_composer.add_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            body_ids=object_ids,
            is_global=is_global,
        )

        for _ in range(10):
            # write data to sim
            object_collection.write_data_to_sim()
            # step sim
            sim.step()
            # update object collection
            object_collection.update(sim.cfg.dt)

        # First object should be rotating around it's X axis
        assert torch.all(object_collection.data.body_com_ang_vel_b.torch[:, 0::2, 0] > 0.1)
        # Second object should have fallen, so it's Z height should be less than initial height of 1.0
        assert torch.all(object_collection.data.body_link_pos_w.torch[:, 1::2, 2] < 1.0)


@pytest.mark.parametrize("num_envs", [3])
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [False])
def test_set_object_state(sim, num_envs, num_cubes, device, gravity_enabled):
    """Test setting the state of the object.

    .. note::
        Turn off gravity for this test as we don't want any external forces acting on the object
        to ensure state remains static
    """
    object_collection, origins = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
    sim.reset()

    state_types = ["body_link_pos_w", "body_link_quat_w", "body_com_lin_vel_w", "body_com_ang_vel_w"]

    # Set each state type individually as they are dependent on each other
    for state_type_to_randomize in state_types:
        state_dict = {
            "body_link_pos_w": torch.zeros_like(object_collection.data.body_link_pos_w.torch, device=sim.device),
            "body_link_quat_w": default_orientation(num=num_cubes * num_envs, device=sim.device).view(
                num_envs, num_cubes, 4
            ),
            "body_com_lin_vel_w": torch.zeros_like(object_collection.data.body_com_lin_vel_w.torch, device=sim.device),
            "body_com_ang_vel_w": torch.zeros_like(object_collection.data.body_com_ang_vel_w.torch, device=sim.device),
        }

        for _ in range(5):
            # reset object
            object_collection.reset()

            # Set random state
            if state_type_to_randomize == "body_link_quat_w":
                state_dict[state_type_to_randomize] = random_orientation(
                    num=num_cubes * num_envs, device=sim.device
                ).view(num_envs, num_cubes, 4)
            else:
                state_dict[state_type_to_randomize] = torch.randn(num_envs, num_cubes, 3, device=sim.device)
                # make sure objects do not overlap
                if state_type_to_randomize == "body_link_pos_w":
                    state_dict[state_type_to_randomize][..., :2] += origins.unsqueeze(1)[..., :2]

            # perform simulation
            for _ in range(5):
                body_pose = torch.cat(
                    [state_dict["body_link_pos_w"], state_dict["body_link_quat_w"]],
                    dim=-1,
                )
                body_vel = torch.cat(
                    [state_dict["body_com_lin_vel_w"], state_dict["body_com_ang_vel_w"]],
                    dim=-1,
                )
                # reset object state
                object_collection.write_body_link_pose_to_sim_index(body_poses=body_pose)
                object_collection.write_body_com_velocity_to_sim_index(body_velocities=body_vel)
                sim.step()

                # assert that set object quantities are equal to the ones set in the state_dict
                for key, expected_value in state_dict.items():
                    value = getattr(object_collection.data, key).torch
                    torch.testing.assert_close(value, expected_value, rtol=1e-5, atol=1e-5)

                object_collection.update(sim.cfg.dt)


@pytest.mark.parametrize("num_envs", [4])
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("gravity_enabled", [False])
def test_object_state_properties(sim, num_envs, num_cubes, device, with_offset, gravity_enabled):
    """Test the object_com_state_w and object_link_state_w properties."""
    cube_object, env_pos = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, height=0.0, device=device)
    view_ids = torch.tensor([x for x in range(num_cubes * num_envs)], dtype=torch.int32)

    sim.reset()

    # check if cube_object is initialized
    assert cube_object.is_initialized

    # change center of mass offset from link frame
    offset = (
        torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
        if with_offset
        else torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
    )

    com = wp.to_torch(cube_object.reshape_view_to_data_2d(cube_object.root_view.get_coms().view(wp.transformf)))
    com[..., :3] = offset.to("cpu")
    cube_object.root_view.set_coms(
        cube_object.reshape_data_to_view_2d(wp.from_torch(com.clone(), dtype=wp.transformf)).view(wp.float32),
        wp.from_torch(view_ids, dtype=wp.int32),
    )

    # check center of mass has been set
    torch.testing.assert_close(
        wp.to_torch(cube_object.reshape_view_to_data_2d(cube_object.root_view.get_coms().view(wp.transformf))), com
    )

    # random z spin velocity
    spin_twist = torch.zeros(6, device=device)
    spin_twist[5] = torch.randn(1, device=device)

    # initial spawn point
    init_com = cube_object.data.body_com_pose_w.torch[..., :3]

    for i in range(10):
        # spin the object around Z axis (com)
        cube_object.write_body_com_velocity_to_sim_index(body_velocities=spin_twist.repeat(num_envs, num_cubes, 1))
        sim.step()
        cube_object.update(sim.cfg.dt)

        # get state properties
        object_link_pose_w = cube_object.data.body_link_pose_w.torch
        object_link_vel_w = cube_object.data.body_link_vel_w.torch
        object_com_pose_w = cube_object.data.body_com_pose_w.torch
        object_com_vel_w = cube_object.data.body_com_vel_w.torch

        # if offset is [0,0,0] all object_state_%_w will match and all body_%_w will match
        if not with_offset:
            torch.testing.assert_close(object_link_pose_w, object_com_pose_w)
            torch.testing.assert_close(object_com_vel_w, object_link_vel_w)
        else:
            # cubes are spinning around center of mass
            # position will not match
            # center of mass position will be constant (i.e. spinning around com)
            torch.testing.assert_close(init_com, object_com_pose_w[..., :3])

            # link position will be moving but should stay constant away from center of mass
            object_link_state_pos_rel_com = quat_apply_inverse(
                object_link_pose_w[..., 3:],
                object_link_pose_w[..., :3] - object_com_pose_w[..., :3],
            )

            torch.testing.assert_close(-offset, object_link_state_pos_rel_com)

            # orientation of com will be a constant rotation from link orientation
            com_quat_b = cube_object.data.body_com_quat_b.torch
            com_quat_w = quat_mul(object_link_pose_w[..., 3:], com_quat_b)
            torch.testing.assert_close(com_quat_w, object_com_pose_w[..., 3:])

            # lin_vel will not match
            # center of mass vel will be constant (i.e. spinning around com)
            torch.testing.assert_close(
                torch.zeros_like(object_com_vel_w[..., :3]),
                object_com_vel_w[..., :3],
            )

            # link frame will be moving, and should be equal to input angular velocity cross offset
            lin_vel_rel_object_gt = quat_apply_inverse(object_link_pose_w[..., 3:], object_link_vel_w[..., :3])
            lin_vel_rel_gt = torch.linalg.cross(spin_twist.repeat(num_envs, num_cubes, 1)[..., 3:], -offset)
            torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_object_gt, atol=1e-4, rtol=1e-3)

            # ang_vel will always match
            torch.testing.assert_close(object_com_vel_w[..., 3:], object_link_vel_w[..., 3:])


@pytest.mark.parametrize("num_envs", [3])
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("state_location", ["com", "link", "root"])
@pytest.mark.parametrize("gravity_enabled", [False])
def test_write_object_state(sim, num_envs, num_cubes, device, with_offset, state_location, gravity_enabled):
    """Test the body state setters in the link, center-of-mass, and mixed root frames.

    Every write must round-trip through the written frame and keep the other frame consistent with the
    center-of-mass offset.
    """
    # Create a scene with random cubes
    cube_object, env_pos = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, height=0.0, device=device)
    view_ids = torch.tensor([x for x in range(num_cubes * num_envs)], dtype=torch.int32)
    env_ids = torch.tensor([x for x in range(num_envs)], dtype=torch.int32)
    object_ids = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32)

    sim.reset()

    # Check if cube_object is initialized
    assert cube_object.is_initialized

    # change center of mass offset from link frame
    offset = (
        torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
        if with_offset
        else torch.tensor([0.0, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)
    )

    com = wp.to_torch(cube_object.reshape_view_to_data_2d(cube_object.root_view.get_coms().view(wp.transformf)))
    com[..., :3] = offset.to("cpu")
    cube_object.root_view.set_coms(
        cube_object.reshape_data_to_view_2d(wp.from_torch(com.clone(), dtype=wp.transformf)).view(wp.float32),
        wp.from_torch(view_ids, dtype=wp.int32),
    )
    # check center of mass has been set
    torch.testing.assert_close(
        wp.to_torch(cube_object.reshape_view_to_data_2d(cube_object.root_view.get_coms().view(wp.transformf))), com
    )

    # random pose and velocity so frame conversions see a non-trivial rotation
    rand_state = torch.rand(num_envs, num_cubes, 13, device=device)
    rand_state[..., :3] += cube_object.data.body_link_pos_w.torch
    # make quaternion a unit vector
    rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

    env_ids = env_ids.to(device)
    object_ids = object_ids.to(device)
    for i in range(10):
        sim.step()
        cube_object.update(sim.cfg.dt)

        ids = {} if i % 2 == 0 else {"env_ids": env_ids, "body_ids": object_ids}
        if state_location == "com":
            cube_object.write_body_com_pose_to_sim_index(body_poses=rand_state[..., :7], **ids)
            cube_object.write_body_com_velocity_to_sim_index(body_velocities=rand_state[..., 7:], **ids)
        elif state_location == "link":
            cube_object.write_body_link_pose_to_sim_index(body_poses=rand_state[..., :7], **ids)
            cube_object.write_body_link_velocity_to_sim_index(body_velocities=rand_state[..., 7:], **ids)
        elif state_location == "root":
            cube_object.write_body_link_pose_to_sim_index(body_poses=rand_state[..., :7], **ids)
            cube_object.write_body_com_velocity_to_sim_index(body_velocities=rand_state[..., 7:], **ids)

        if state_location == "com":
            torch.testing.assert_close(rand_state[..., :7], cube_object.data.body_com_pose_w.torch)
            torch.testing.assert_close(rand_state[..., 7:], cube_object.data.body_com_vel_w.torch)
        elif state_location == "link":
            torch.testing.assert_close(rand_state[..., :7], cube_object.data.body_link_pose_w.torch)
            torch.testing.assert_close(rand_state[..., 7:], cube_object.data.body_link_vel_w.torch)
        elif state_location == "root":
            torch.testing.assert_close(rand_state[..., :7], cube_object.data.body_link_pose_w.torch)
            torch.testing.assert_close(rand_state[..., 7:], cube_object.data.body_com_vel_w.torch)

        # the frame that was not written must follow through the center-of-mass offset
        link_pose_w = cube_object.data.body_link_pose_w.torch
        body_com_pose_b = cube_object.data.body_com_pose_b.torch
        expected_com_pos, expected_com_quat = combine_frame_transforms(
            link_pose_w[..., :3].reshape(-1, 3),
            link_pose_w[..., 3:].reshape(-1, 4),
            body_com_pose_b[..., :3].reshape(-1, 3),
            body_com_pose_b[..., 3:].reshape(-1, 4),
        )
        expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1).view(num_envs, num_cubes, 7)
        torch.testing.assert_close(expected_com_pose, cube_object.data.body_com_pose_w.torch)
        torch.testing.assert_close(
            cube_object.data.body_com_vel_w.torch[..., 3:], cube_object.data.body_link_vel_w.torch[..., 3:]
        )


@pytest.mark.parametrize("num_envs", [3])
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_reset_object_collection(sim, num_envs, num_cubes, device):
    """Test that reset clears the external wrenches of the selected environments only."""
    object_collection, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
    sim.reset()
    sim.step()
    object_collection.update(sim.cfg.dt)

    composers = (object_collection.instantaneous_wrench_composer, object_collection.permanent_wrench_composer)
    ones = torch.ones((num_envs, num_cubes, 3), device=device)
    object_collection.permanent_wrench_composer.set_forces_and_torques_index(forces=ones, torques=ones)
    object_collection.instantaneous_wrench_composer.add_forces_and_torques_index(forces=ones, torques=ones)

    # A partial reset clears only the selected environment
    object_collection.reset(env_ids=torch.tensor([0], device=device))
    for composer in composers:
        assert composer.active
        for buffer in (composer.out_force_b.torch, composer.out_torque_b.torch):
            assert torch.count_nonzero(buffer[0]) == 0
            assert torch.count_nonzero(buffer[1:]) == buffer[1:].numel()

    # A full reset clears every environment
    object_collection.reset()
    for composer in composers:
        assert torch.count_nonzero(composer.out_force_b.torch) == 0
        assert torch.count_nonzero(composer.out_torque_b.torch) == 0


@pytest.mark.parametrize("num_envs", [3])
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [True, False])
def test_gravity_vec_w(sim, num_envs, num_cubes, device, gravity_enabled):
    """Test that gravity vector direction is set correctly for the rigid object."""
    object_collection, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)

    # Obtain gravity direction
    gravity_dir = (0.0, 0.0, -1.0) if gravity_enabled else (0.0, 0.0, 0.0)

    sim.reset()

    # Check if gravity vector is set correctly
    gravity_vec = object_collection.data.GRAVITY_VEC_W.torch
    assert gravity_vec[0, 0, 0] == gravity_dir[0]
    assert gravity_vec[0, 0, 1] == gravity_dir[1]
    assert gravity_vec[0, 0, 2] == gravity_dir[2]

    # Perform simulation
    for _ in range(2):
        sim.step()
        object_collection.update(sim.cfg.dt)

        # Expected gravity value is the acceleration of the body
        gravity = torch.zeros(num_envs, num_cubes, 6, device=device)
        if gravity_enabled:
            gravity[..., 2] = -9.81

        # Check the body accelerations are correct
        torch.testing.assert_close(object_collection.data.body_com_acc_w.torch, gravity)
