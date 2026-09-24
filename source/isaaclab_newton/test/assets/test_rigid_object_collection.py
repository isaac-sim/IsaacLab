# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none


"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from isaaclab.test.utils import DeviceScope, resolve_test_sim_device, test_devices

# launch omniverse app
simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

"""Rest everything follows."""

import sys

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.assets import RigidObjectCollection
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_rotate,
    random_orientation,
    subtract_frame_transforms,
)

NEWTON_SIM_CFG = SimulationCfg(
    physics=NewtonCfg(
        solver_cfg=MJWarpSolverCfg(),
    ),
)


def _newton_sim_context(device, gravity_enabled=True, dt=None, **kwargs):
    """Helper to create a Newton simulation context with the correct device.

    When sim_cfg is provided to build_simulation_context, the device, gravity_enabled, and dt
    kwargs are ignored. This helper applies them to the shared NEWTON_SIM_CFG before calling.
    """
    NEWTON_SIM_CFG.device = device
    NEWTON_SIM_CFG.gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    if dt is not None:
        NEWTON_SIM_CFG.dt = dt
    return build_simulation_context(device=device, sim_cfg=NEWTON_SIM_CFG, **kwargs)


def generate_cubes_scene(
    num_envs: int = 1,
    num_cubes: int = 1,
    height=1.0,
    has_api: bool = True,
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
    spawn_unrelated_sibling: bool = False,
) -> tuple[RigidObjectCollection, torch.Tensor]:
    """Generate a scene with the provided number of cubes.

    Args:
        num_envs: Number of envs to generate.
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        has_api: Whether the cubes have a rigid body API on them.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.
        spawn_unrelated_sibling: Whether to spawn a rigid body outside the collection in each environment.

    Returns:
        A tuple containing the rigid object collection representing the cubes and the origins of the cubes.

    """
    origins = torch.tensor([(i * 3.0, 0, height) for i in range(num_envs)]).to(device)
    # Create Top-level Xforms, one for each cube
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=origin)

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
            prim_path=f"/World/Env_[^/]*/Object_{i}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3 * i, height)),
        )
        cube_config_dict[f"cube_{i}"] = cube_object_cfg
    if spawn_unrelated_sibling:
        spawn_cfg.func(
            "/World/Env_[^/]*/UnrelatedObject",
            spawn_cfg,
            translation=(0.0, -3.0, height),
        )
    # create the rigid object collection
    cube_object_collection_cfg = RigidObjectCollectionCfg(rigid_objects=cube_config_dict)
    cube_object_collection = RigidObjectCollection(cfg=cube_object_collection_cfg)

    return cube_object_collection, origins


@pytest.mark.parametrize(("num_envs", "num_cubes", "spawn_unrelated_sibling"), [(1, 1, False), (2, 3, True)])
@pytest.mark.parametrize("device", test_devices())
def test_initialization(num_envs, num_cubes, spawn_unrelated_sibling, device):
    """Test initialization for prim with rigid body API at the provided prim path.

    With an unrelated rigid body next to the cubes in each environment, the collection view must still
    select only its configured rigid objects.
    """
    with _newton_sim_context(device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        object_collection, _ = generate_cubes_scene(
            num_envs=num_envs,
            num_cubes=num_cubes,
            device=device,
            spawn_unrelated_sibling=spawn_unrelated_sibling,
        )

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(object_collection) < 10

        # Play sim
        sim.reset()

        # Check if object is initialized
        assert object_collection.is_initialized
        assert object_collection.num_instances == num_envs
        assert object_collection.root_view.count == num_envs * num_cubes
        assert len(object_collection.body_names) == num_cubes

        # Check buffers that exist and have correct shapes
        assert object_collection.data.default_body_pose.torch.shape == (num_envs, num_cubes, 7)
        assert object_collection.data.body_link_pos_w.torch.shape == (num_envs, num_cubes, 3)
        assert object_collection.data.body_link_quat_w.torch.shape == (num_envs, num_cubes, 4)
        assert object_collection.data.body_mass.torch.shape == (num_envs, num_cubes)
        assert object_collection.data.body_inertia.torch.shape == (num_envs, num_cubes, 9)


@pytest.mark.parametrize("device", test_devices())
def test_set_body_inertial_properties_updates_inverses(device):
    """Masked inertial-property writes update only selected Newton inverse entries."""
    num_envs = 2
    num_cubes = 3
    with _newton_sim_context(device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        for env_index in range(num_envs):
            sim_utils.create_prim(f"/World/Env_{env_index}", "Xform", translation=(float(env_index), 0.0, 1.0))
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=PhysxRigidBodyCfg(disable_gravity=True),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        )
        object_collection = RigidObjectCollection(
            RigidObjectCollectionCfg(
                rigid_objects={
                    f"cube_{body_index}": RigidObjectCfg(
                        prim_path=f"/World/Env_[^/]*/Object_{body_index}",
                        spawn=spawn_cfg,
                        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, float(body_index), 0.0)),
                    )
                    for body_index in range(num_cubes)
                }
            )
        )
        sim.reset()

        env_mask = wp.array([True, False], dtype=wp.bool, device=device)
        body_mask = wp.array([False, True, False], dtype=wp.bool, device=device)
        selected = (0, 1)

        raw_model_inv_mass = object_collection.root_view.get_attribute("body_inv_mass", SimulationManager.get_model())[
            :, :, 0
        ]
        assert object_collection.data._sim_bind_body_inv_mass.ptr == raw_model_inv_mass.ptr
        model_inv_mass = object_collection.data._sim_bind_body_inv_mass
        original_inv_mass = wp.to_torch(model_inv_mass).clone()
        masses = object_collection.data.body_mass.torch.clone()
        masses[selected] = 4.0
        object_collection.set_masses_mask(masses=masses, env_mask=env_mask, body_mask=body_mask)

        updated_inv_mass = wp.to_torch(model_inv_mass).clone()
        torch.testing.assert_close(updated_inv_mass[selected], masses[selected].reciprocal())
        unselected = torch.ones_like(updated_inv_mass, dtype=torch.bool)
        unselected[selected] = False
        torch.testing.assert_close(updated_inv_mass[unselected], original_inv_mass[unselected])

        raw_model_inv_inertia = object_collection.root_view.get_attribute(
            "body_inv_inertia", SimulationManager.get_model()
        )[:, :, 0]
        assert object_collection.data._sim_bind_body_inv_inertia.ptr == raw_model_inv_inertia.ptr
        model_inv_inertia = object_collection.data._sim_bind_body_inv_inertia
        original_inv_inertia = wp.to_torch(model_inv_inertia).clone()
        inertias = object_collection.data.body_inertia.torch.clone()
        inertia_matrix = torch.diag(torch.tensor([2.0, 3.0, 5.0], device=device))
        inertias[selected] = inertia_matrix.reshape(9)
        object_collection.set_inertias_mask(inertias=inertias, env_mask=env_mask, body_mask=body_mask)

        updated_inv_inertia = wp.to_torch(model_inv_inertia)
        torch.testing.assert_close(updated_inv_inertia[selected], torch.linalg.inv(inertia_matrix))
        torch.testing.assert_close(updated_inv_inertia[unselected], original_inv_inertia[unselected])
        torch.testing.assert_close(wp.to_torch(model_inv_mass), updated_inv_mass)


def test_initialization_with_no_rigid_body():
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with _newton_sim_context("cpu", auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        object_collection, _ = generate_cubes_scene(num_cubes=2, has_api=False, device="cpu")

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(object_collection) < 10

        # Play sim
        with pytest.raises(RuntimeError, match="Expected 1 prims at"):
            sim.reset()


@pytest.mark.parametrize("num_envs", [2])
@pytest.mark.parametrize("num_cubes", [4])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_single_body(num_envs, num_cubes, device):
    """Test application of external force on the base of the object.

    In the first phase, a force equal to the weight of every 2nd object keeps it in place while the others
    fall. In the second phase, the force is applied at 1m in the Y direction and the object must rotate
    around its X axis. Both phases run in the global and the local frame, and resetting the collection
    must clear the wrench applied in the previous iteration.
    """
    with _newton_sim_context(device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        object_collection, origins = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
        sim.reset()

        # find objects to apply the force
        object_ids, object_names = object_collection.find_bodies(".*")

        def reset_objects():
            # reset object state; shift the cubes to their origins so they are not on top of each other
            body_pose = object_collection.data.default_body_pose.torch.clone()
            body_vel = object_collection.data.default_body_vel.torch.clone()
            body_pose[..., :2] += origins.unsqueeze(1)[..., :2]
            object_collection.write_body_link_pose_to_sim_index(body_poses=body_pose)
            object_collection.write_body_com_velocity_to_sim_index(body_velocities=body_vel)
            object_collection.reset()

            # Reset should zero external forces and torques
            assert torch.count_nonzero(object_collection.instantaneous_wrench_composer.out_force_b.torch) == 0
            assert torch.count_nonzero(object_collection.instantaneous_wrench_composer.out_torque_b.torch) == 0
            assert torch.count_nonzero(object_collection.permanent_wrench_composer.out_force_b.torch) == 0
            assert torch.count_nonzero(object_collection.permanent_wrench_composer.out_torque_b.torch) == 0

        def simulate():
            for _ in range(10):
                object_collection.write_data_to_sim()
                sim.step()
                object_collection.update(sim.cfg.dt)

        # Sample a force equal to the weight of the object
        external_wrench_b = torch.zeros(object_collection.num_instances, len(object_ids), 6, device=sim.device)
        # Every 2nd cube should have a force applied to it
        external_wrench_b[:, 0::2, 2] = 9.81 * object_collection.data.body_mass.torch[:, 0::2]

        for i in range(2):
            reset_objects()

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
            simulate()

            # First object should still be at the same Z position (1.0)
            torch.testing.assert_close(
                object_collection.data.body_link_pos_w.torch[:, 0::2, 2],
                torch.ones_like(object_collection.data.body_link_pos_w.torch[:, 0::2, 2]),
            )
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(object_collection.data.body_link_pos_w.torch[:, 1::2, 2] < 1.0)

        # Apply a force at 1m in the Y direction on every 2nd cube
        external_wrench_b = torch.zeros(object_collection.num_instances, len(object_ids), 6, device=sim.device)
        external_wrench_positions_b = torch.zeros(
            object_collection.num_instances, len(object_ids), 3, device=sim.device
        )
        external_wrench_b[:, 0::2, 2] = 50.0
        external_wrench_positions_b[:, 0::2, 1] = 1.0

        for i in range(2):
            reset_objects()

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
            simulate()

            # First object should be rotating around it's X axis
            assert torch.all(object_collection.data.body_com_ang_vel_b.torch[:, 0::2, 0] > 0.1)
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(object_collection.data.body_link_pos_w.torch[:, 1::2, 2] < 1.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Newton's ArticulationView assumes each body's shapes are contiguous, but the collection's model interleaves"
        " them per environment (visuals of all cubes, then collisions), so the shape binding misaddresses shapes."
    ),
)
@pytest.mark.parametrize("num_envs", [3])
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_set_material_properties(num_envs, num_cubes, device):
    """Material writes through the collection's view bindings reach the Newton model shapes of its bodies."""
    with _newton_sim_context(device, add_ground_plane=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        object_collection, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
        sim.reset()

        # Resolve the collection's shapes from the flat Newton model, independent of the view binding.
        model = SimulationManager.get_model()
        shape_body = model.shape_body.numpy()
        collection_shapes = np.flatnonzero(shape_body >= 0)
        other_shapes = np.flatnonzero(shape_body < 0)
        assert len(collection_shapes) > 0 and len(other_shapes) > 0
        original_mu = model.shape_material_mu.numpy().copy()
        original_restitution = model.shape_material_restitution.numpy().copy()

        # Write friction/restitution in place through the view-level bindings
        friction_binding = object_collection._root_view.get_attribute("shape_material_mu", model)
        restitution_binding = object_collection._root_view.get_attribute("shape_material_restitution", model)
        wp.to_torch(friction_binding).fill_(0.55)
        wp.to_torch(restitution_binding).fill_(0.15)
        SimulationManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

        # Perform simulation
        sim.step()
        object_collection.update(sim.cfg.dt)

        # Every shape of the collection's bodies is updated, and no other shape is touched.
        mu = model.shape_material_mu.numpy()
        restitution = model.shape_material_restitution.numpy()
        np.testing.assert_allclose(mu[collection_shapes], 0.55)
        np.testing.assert_allclose(restitution[collection_shapes], 0.15)
        np.testing.assert_array_equal(mu[other_shapes], original_mu[other_shapes])
        np.testing.assert_array_equal(restitution[other_shapes], original_restitution[other_shapes])


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("num_envs", [3])
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_gravity_vec_w_tracks_model_gravity(num_envs, num_cubes, device):
    """Per-env mutations to Newton's ``model.gravity`` reach ``GRAVITY_VEC_W`` and ``projected_gravity_b``.

    Regression for the pre-fix snapshot: ``GRAVITY_VEC_W`` used to be env 0's
    gravity broadcast to every env and body, hiding per-env gravity
    randomization (e.g. :class:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`).
    """
    with _newton_sim_context(device, gravity_enabled=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        object_collection, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, device=device)
        sim.reset()

        # Check if gravity vector is set correctly
        torch.testing.assert_close(
            object_collection.data.GRAVITY_VEC_W.torch[0], torch.tensor((0.0, 0.0, -9.81), device=device)
        )

        # The free-falling cubes accelerate with gravity and keep their identity orientation.
        for _ in range(2):
            sim.step()
            object_collection.update(sim.cfg.dt)
        gravity = torch.zeros(num_envs, num_cubes, 6, device=device)
        gravity[..., 2] = -9.81
        torch.testing.assert_close(object_collection.data.body_com_acc_w.torch, gravity)

        # GRAVITY_VEC_W must share storage with Newton's per-env gravity array.
        model = SimulationManager.get_model()
        model_gravity_arr = model.gravity[: model.world_count]
        global_gravity = wp.to_torch(model.gravity)[-1].clone()
        assert object_collection.data.GRAVITY_VEC_W.warp.ptr == model_gravity_arr.ptr
        assert object_collection.data.GRAVITY_VEC_W.shape == (num_envs,)

        # Mutate model.gravity per-env in place, as randomize_physics_scene_gravity does.
        new_gravity = torch.tensor(
            [[0.1 * (i + 1), 0.2 * (i + 1), -3.0 - float(i)] for i in range(num_envs)],
            device=device,
            dtype=torch.float32,
        )
        wp.to_torch(model_gravity_arr).copy_(new_gravity)
        SimulationManager.add_model_change(ModelFlags.MODEL_PROPERTIES)
        torch.testing.assert_close(wp.to_torch(model.gravity)[-1], global_gravity)

        # Recompute the lazily-cached projected_gravity_b without sim.step: bodies stay
        # at identity orientation, so each env's unit gravity broadcasts across its bodies.
        object_collection.update(sim.cfg.dt)
        expected_per_env = torch.nn.functional.normalize(new_gravity, dim=-1)
        expected = expected_per_env.unsqueeze(1).expand(-1, num_cubes, -1).contiguous()
        torch.testing.assert_close(object_collection.data.projected_gravity_b.torch, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_envs", [4])
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_object_state_properties(num_envs, num_cubes, device):
    """Test the object_com_state_w and object_link_state_w properties."""
    with _newton_sim_context(device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, env_pos = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, height=0.0, device=device)

        sim.reset()

        # check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)

        # Set center of mass offset via Newton API (position only, shape (E, B, 3))
        cube_object.set_coms_index(coms=wp.from_torch(offset, dtype=wp.vec3f))
        # Flush the model change immediately so it takes effect before the next step
        with wp.ScopedDevice(device):
            SimulationManager._solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        # check center of mass has been set
        torch.testing.assert_close(cube_object.data.body_com_pos_b.torch, offset)

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

            _tol = dict(atol=2e-3, rtol=2e-3)
            # cubes are spinning around center of mass
            # position will not match
            # center of mass position will be constant (i.e. spinning around com)
            torch.testing.assert_close(init_com, object_com_pose_w[..., :3], **_tol)

            # link position will be moving but should stay constant away from center of mass
            object_link_state_pos_rel_com = quat_apply_inverse(
                object_link_pose_w[..., 3:],
                object_link_pose_w[..., :3] - object_com_pose_w[..., :3],
            )

            torch.testing.assert_close(-offset, object_link_state_pos_rel_com, **_tol)

            # orientation of com will be a constant rotation from link orientation
            com_quat_b = cube_object.data.body_com_quat_b.torch
            com_quat_w = quat_mul(object_link_pose_w[..., 3:], com_quat_b)
            torch.testing.assert_close(com_quat_w, object_com_pose_w[..., 3:], **_tol)

            # lin_vel will not match
            # center of mass vel will be constant (i.e. spinning around com)
            torch.testing.assert_close(
                torch.zeros_like(object_com_vel_w[..., :3]),
                object_com_vel_w[..., :3],
                **_tol,
            )

            # link frame will be moving, and should be equal to input angular velocity cross offset
            lin_vel_rel_object_gt = quat_apply_inverse(object_link_pose_w[..., 3:], object_link_vel_w[..., :3])
            lin_vel_rel_gt = torch.linalg.cross(spin_twist.repeat(num_envs, num_cubes, 1)[..., 3:], -offset)
            torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_object_gt, **_tol)

            # ang_vel will always match
            torch.testing.assert_close(object_com_vel_w[..., 3:], object_link_vel_w[..., 3:])


@pytest.mark.parametrize("num_envs", [3])
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("state_location", ["com", "link", "root"])
def test_write_object_state(num_envs, num_cubes, device, state_location):
    """Test the object state setters in the center-of-mass frame, the link frame, and the default root frames.

    A write must be readable in the written frame and refresh the derived frame without a sim step, and the
    written state must persist into the solver across a step.
    """
    with _newton_sim_context(device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, height=0.0, device=device)
        env_ids = torch.tensor([x for x in range(num_envs)], dtype=torch.int32, device=device)
        object_ids = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32, device=device)

        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)

        # Set center of mass offset via Newton API (position only, shape (E, B, 3))
        cube_object.set_coms_index(coms=wp.from_torch(offset, dtype=wp.vec3f))
        # Flush the model change immediately so it takes effect before the next step
        with wp.ScopedDevice(device):
            SimulationManager._solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        # check center of mass has been set
        torch.testing.assert_close(cube_object.data.body_com_pos_b.torch, offset)

        for i in range(2):
            sim.step()
            cube_object.update(sim.cfg.dt)

            body_link_pose_w = cube_object.data.body_link_pose_w.torch
            body_com_pose_w = cube_object.data.body_com_pose_w.torch
            object_link_to_com_pos, object_link_to_com_quat = subtract_frame_transforms(
                body_link_pose_w[..., :3].view(-1, 3),
                body_link_pose_w[..., 3:7].view(-1, 4),
                body_com_pose_w[..., :3].view(-1, 3),
                body_com_pose_w[..., 3:7].view(-1, 4),
            )

            # A target state distinct from the current one in position, orientation, and velocity.
            target_pose = torch.cat(
                [
                    body_link_pose_w[..., :3] + 0.3 * torch.rand(num_envs, num_cubes, 3, device=device),
                    random_orientation(num_envs * num_cubes, device).view(num_envs, num_cubes, 4),
                ],
                dim=-1,
            )
            target_vel = torch.randn(num_envs, num_cubes, 6, device=device)

            # Alternate between the default and explicit environment and body selectors.
            selectors = {} if i == 0 else {"env_ids": env_ids, "body_ids": object_ids}
            if state_location == "com":
                cube_object.write_body_com_pose_to_sim_index(body_poses=target_pose, **selectors)
                cube_object.write_body_com_velocity_to_sim_index(body_velocities=target_vel, **selectors)
            elif state_location == "link":
                cube_object.write_body_link_pose_to_sim_index(body_poses=target_pose, **selectors)
                cube_object.write_body_link_velocity_to_sim_index(body_velocities=target_vel, **selectors)
            elif state_location == "root":
                cube_object.write_body_link_pose_to_sim_index(body_poses=target_pose, **selectors)
                cube_object.write_body_com_velocity_to_sim_index(body_velocities=target_vel, **selectors)

            link_pose_w = cube_object.data.body_link_pose_w.torch
            link_vel_w = cube_object.data.body_link_vel_w.torch
            com_pose_w = cube_object.data.body_com_pose_w.torch
            com_vel_w = cube_object.data.body_com_vel_w.torch
            if state_location == "com":
                torch.testing.assert_close(target_pose, com_pose_w)
                torch.testing.assert_close(target_vel, com_vel_w)
                # the com pose was written, so the derived link pose must be refreshed
                expected_link_pos, expected_link_quat = combine_frame_transforms(
                    com_pose_w[..., :3].view(-1, 3),
                    com_pose_w[..., 3:].view(-1, 4),
                    quat_rotate(quat_inv(object_link_to_com_quat), -object_link_to_com_pos),
                    quat_inv(object_link_to_com_quat),
                )
                expected_link_pose = torch.cat((expected_link_pos, expected_link_quat), dim=1).view(num_envs, -1, 7)
                torch.testing.assert_close(expected_link_pose, link_pose_w)
            else:
                torch.testing.assert_close(target_pose, link_pose_w)
                written_vel_w = link_vel_w if state_location == "link" else com_vel_w
                torch.testing.assert_close(target_vel, written_vel_w)
                # the link pose was written, so the derived com pose must be refreshed
                expected_com_pos, expected_com_quat = combine_frame_transforms(
                    link_pose_w[..., :3].view(-1, 3),
                    link_pose_w[..., 3:].view(-1, 4),
                    object_link_to_com_pos,
                    object_link_to_com_quat,
                )
                expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1).view(num_envs, -1, 7)
                torch.testing.assert_close(expected_com_pose, com_pose_w)
            # skip lin_vel because it differs between the frames; angular velocity is frame-independent
            # and only matches when the derived velocity was actually refreshed after the write
            torch.testing.assert_close(com_vel_w[..., 3:], link_vel_w[..., 3:])

        # The written state persists into the solver: with gravity off, one step only integrates the
        # written velocity.
        written_pose_w = (com_pose_w if state_location == "com" else link_pose_w).clone()
        sim.step()
        cube_object.update(sim.cfg.dt)
        pose_w = cube_object.data.body_com_pose_w if state_location == "com" else cube_object.data.body_link_pose_w
        torch.testing.assert_close(pose_w.torch, written_pose_w, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.isaacsim_ci
def test_body_pose_write_marks_fk_reset_mask(device):
    """Regression: ``write_body_{link,com}_pose_to_sim_{index,mask}`` must mark FK dirty.

    For a collection, ``_sim_bind_body_link_pose_w`` is bound directly to the simulator's root-transforms
    buffer, so the property read is not what becomes stale — the simulator's internal ``body_q`` used by
    collision detection is. The write methods must therefore call :meth:`SimulationManager.invalidate_fk`
    so downstream consumers re-run forward kinematics before the next step. Without the fix,
    ``_fk_reset_mask`` remains unset after an explicit pose write. The buffer-aliasing invariant is
    also pinned: a refactor that decouples ``_sim_bind_body_link_pose_w`` from the write target would
    silently make the property stale, so we check the post-write pose matches the written value.
    """

    def _fk_reset_mask_dirty() -> bool:
        assert SimulationManager._fk_reset_mask is not None
        return bool(wp.to_torch(SimulationManager._fk_reset_mask).any().item())

    num_envs = 2
    num_cubes = 2
    with _newton_sim_context(device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes, height=0.5, device=device)

        sim.reset()
        assert cube_object.is_initialized

        sim.step()
        cube_object.update(sim.cfg.dt)

        for writer in ("link_index", "link_mask", "com_index", "com_mask"):
            # Clear the dirty flag so we can observe that the write sets it.
            SimulationManager.forward()
            assert not _fk_reset_mask_dirty()

            pre_write_pose = wp.to_torch(cube_object.data.body_link_pose_w).clone()

            target_pose = wp.to_torch(cube_object.data.body_link_pose_w).clone()
            target_pose[..., 0] += 10.0
            target_pose[..., 1] += 5.0
            target_pose[..., 2] += 2.0

            if writer == "link_index":
                cube_object.write_body_link_pose_to_sim_index(body_poses=target_pose)
            elif writer == "link_mask":
                cube_object.write_body_link_pose_to_sim_mask(body_poses=target_pose)
            elif writer == "com_index":
                cube_object.write_body_com_pose_to_sim_index(body_poses=target_pose)
            elif writer == "com_mask":
                cube_object.write_body_com_pose_to_sim_mask(body_poses=target_pose)

            assert _fk_reset_mask_dirty(), f"{writer} pose write must call SimulationManager.invalidate_fk()"

            # body_link_pose_w must reflect the write immediately — its underlying buffer is the write
            # target. A regression that moves this property to a separate cached buffer (mirroring the
            # single-object case) would silently break this invariant.
            body_link = wp.to_torch(cube_object.data.body_link_pose_w)
            assert not torch.allclose(body_link[..., :3], pre_write_pose[..., :3], rtol=1e-4, atol=1e-4), (
                f"body_link_pose_w still aliases the pre-write pose after {writer}; the buffer was not written"
            )
            torch.testing.assert_close(body_link[..., :3], target_pose[..., :3], rtol=1e-4, atol=1e-4)
