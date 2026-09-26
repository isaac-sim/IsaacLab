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
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import torch
import warp as wp
from flaky import flaky
from isaaclab_newton.assets import RigidObject
from isaaclab_newton.envs.mdp import randomize_world_gravity
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_rotate,
    random_orientation,
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
    num_cubes: int = 1,
    height=1.0,
    api: Literal["none", "rigid_body", "articulation_root"] = "rigid_body",
    kinematic_enabled: bool = False,
    device: str = "cuda:0",
    add_ground_plane: bool = False,
) -> tuple[RigidObject, torch.Tensor]:
    """Generate a scene with the provided number of cubes.

    Args:
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        api: The type of API that the cubes should have.
        kinematic_enabled: Whether the cubes are kinematic.
        device: Device to use for the simulation.
        add_ground_plane: Whether the simulation context authored a shared ground plane.

    Returns:
        A tuple containing the rigid object representing the cubes and the origins of the cubes.

    """
    origins = np.asarray([(i * 1.0, 0, height) for i in range(num_cubes)], dtype=np.float32)
    sim_utils.create_prim("/World/Env_0", "Xform", translation=origins[0])

    # Resolve spawn configuration
    if api == "none":
        # since no rigid body properties defined, this is just a static collider
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        )
    elif api == "rigid_body":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic_enabled),
        )
    elif api == "articulation_root":
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Tests/RigidObject/Cube/dex_cube_instanceable_with_articulation_root.usd",
            # Only tune existing bodies; do not create one on this invalid articulation fixture.
            rigid_props={"(/.*)?": [sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic_enabled)]},
        )
    else:
        raise ValueError(f"Unknown api: {api}")

    # Create rigid object
    cube_object_cfg = RigidObjectCfg(
        prim_path="/World/Env_[^/]*/Object",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cfgs = [cube_object_cfg]
    if add_ground_plane:
        cfgs.append(AssetBaseCfg(prim_path="/World/defaultGroundPlane"))
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), cfgs, num_cubes, 1.0, positions=origins)
    cube_object = RigidObject(cfg=cube_object_cfg)

    return cube_object, torch.as_tensor(origins, device=device)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("api", ["none", "articulation_root"])
def test_initialization_rejects_non_rigid_body_prims(api):
    """Initialization fails when the prim path has no rigid body API or carries an articulation root."""
    with _newton_sim_context("cpu", auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, api=api, device="cpu")

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(cube_object) < 10

        replicate(sim.get_clone_plan())
        with pytest.raises(RuntimeError, match="Expected 1 prims at"):
            sim.reset()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("num_cubes", [4])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_single_body(num_cubes, device):
    """Test initialization and external forces on the base of the object.

    In the first phase, we apply a force equal to the weight of an object on the base of
    one of the objects. We check that the object does not move. For the other object,
    we do not apply any force and check that it falls down. In the second phase, the force
    is applied at 1m in the Y direction and the object must rotate around its X axis.

    We validate that this works when we apply the force in the global frame and in the local frame,
    and that resetting the object clears the wrench applied in the previous iteration.
    """
    # Generate cubes scene
    with _newton_sim_context(device, add_ground_plane=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, origins = generate_cubes_scene(num_cubes=num_cubes, device=device, add_ground_plane=True)

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(cube_object) < 10

        # Play the simulator
        replicate(sim.get_clone_plan())
        sim.reset()

        # Check if object is initialized
        assert cube_object.is_initialized
        assert len(cube_object.body_names) == 1

        # Check buffers that exists and have correct shapes
        assert cube_object.data.root_pos_w.torch.shape == (num_cubes, 3)
        assert cube_object.data.root_quat_w.torch.shape == (num_cubes, 4)
        assert cube_object.data.body_mass.torch.shape == (num_cubes, 1)
        assert cube_object.data.body_inertia.torch.shape == (num_cubes, 1, 9)

        # Find bodies to apply the force
        body_ids, body_names = cube_object.find_bodies(".*")

        def reset_cubes():
            # reset root state; shift the cubes to their origins so they are not on top of each other
            root_pose = cube_object.data.default_root_pose.torch.clone()
            root_pose[:, :3] = origins
            cube_object.write_root_pose_to_sim_index(root_pose=root_pose)
            cube_object.write_root_velocity_to_sim_index(root_velocity=cube_object.data.default_root_vel.torch.clone())
            cube_object.reset()

            # Reset should zero external forces and torques
            assert not cube_object.instantaneous_wrench_composer.active
            assert not cube_object.permanent_wrench_composer.active
            assert torch.count_nonzero(cube_object.instantaneous_wrench_composer.out_force_b.torch) == 0
            assert torch.count_nonzero(cube_object.instantaneous_wrench_composer.out_torque_b.torch) == 0
            assert torch.count_nonzero(cube_object.permanent_wrench_composer.out_force_b.torch) == 0
            assert torch.count_nonzero(cube_object.permanent_wrench_composer.out_torque_b.torch) == 0

        def simulate():
            for _ in range(5):
                cube_object.write_data_to_sim()
                sim.step()
                cube_object.update(sim.cfg.dt)

        # Sample a force equal to the weight of the object
        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
        # Every 2nd cube should have a force applied to it
        external_wrench_b[0::2, :, 2] = 9.81 * cube_object.data.body_mass.torch[0]

        for i in range(2):
            reset_cubes()

            is_global = False
            if i % 2 == 0:
                is_global = True
                positions = cube_object.data.body_com_pos_w.torch[:, body_ids, :3]
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
            simulate()

            # First object should still be at the same Z position (1.0)
            torch.testing.assert_close(
                cube_object.data.root_pos_w.torch[0::2, 2], torch.ones(num_cubes // 2, device=sim.device)
            )
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(cube_object.data.root_pos_w.torch[1::2, 2] < 1.0)

        # Apply a force at 1m in the Y direction on every 2nd cube
        external_wrench_b = torch.zeros(cube_object.num_instances, len(body_ids), 6, device=sim.device)
        external_wrench_positions_b = torch.zeros(cube_object.num_instances, len(body_ids), 3, device=sim.device)
        external_wrench_b[0::2, :, 2] = 50.0
        external_wrench_positions_b[0::2, :, 1] = 1.0

        for i in range(2):
            reset_cubes()

            is_global = False
            if i % 2 == 0:
                is_global = True
                body_com_pos_w = cube_object.data.body_com_pos_w.torch[:, body_ids, :3]
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
            simulate()

            # The first object should be rotating around it's X axis
            assert torch.all(torch.abs(cube_object.data.root_ang_vel_b.torch[0::2, 0]) > 0.1)
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(cube_object.data.root_pos_w.torch[1::2, 2] < 1.0)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_rigid_body_set_material_properties(num_cubes, device):
    """Material randomization writes friction and restitution into the Newton model shapes of the selected envs."""
    with _newton_sim_context(device, gravity_enabled=True, add_ground_plane=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Generate cubes scene
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device, add_ground_plane=True)

        # Play sim
        replicate(sim.get_clone_plan())
        sim.reset()

        # Resolve each cube's shapes from the flat Newton model, independent of the asset's view binding.
        model = SimulationManager.get_model()
        body_world = model.body_world.numpy()
        shape_body = model.shape_body.numpy()
        cube_shapes = [
            np.flatnonzero(np.isin(shape_body, np.flatnonzero(body_world == index))) for index in range(num_cubes)
        ]
        assert all(len(shapes) > 0 for shapes in cube_shapes)
        original_mu = model.shape_material_mu.numpy().copy()
        original_restitution = model.shape_material_restitution.numpy().copy()

        # Randomize the materials of the last cube through the event term, with degenerate ranges.
        env = SimpleNamespace(scene={"cube": cube_object}, sim=sim, device=device, num_envs=num_cubes)
        params = {
            "static_friction_range": (0.55, 0.55),
            "dynamic_friction_range": (0.55, 0.55),
            "restitution_range": (0.15, 0.15),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("cube"),
        }
        term = randomize_rigid_body_material(
            EventTermCfg(func=randomize_rigid_body_material, mode="startup", params=params), env
        )
        term(env, torch.tensor([num_cubes - 1], device=device), **params)

        # Simulate physics
        sim.step()
        cube_object.update(sim.cfg.dt)

        mu = model.shape_material_mu.numpy()
        restitution = model.shape_material_restitution.numpy()
        np.testing.assert_allclose(mu[cube_shapes[-1]], 0.55)
        np.testing.assert_allclose(restitution[cube_shapes[-1]], 0.15)
        # Shapes of the other cubes are untouched.
        for shapes in cube_shapes[:-1]:
            np.testing.assert_array_equal(mu[shapes], original_mu[shapes])
            np.testing.assert_array_equal(restitution[shapes], original_restitution[shapes])


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_rigid_body_set_mass(num_cubes, device):
    """Test that selected mass writes update inverse mass and inertia across static transitions."""
    with _newton_sim_context(device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        origins = np.asarray([(float(index), 0.0, 1.0) for index in range(num_cubes)], dtype=np.float32)
        sim_utils.create_prim("/World/Env_0", "Xform", translation=origins[0])
        cfg = RigidObjectCfg(
            prim_path="/World/Env_[^/]*/Object",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=PhysxRigidBodyCfg(disable_gravity=True),
                mass_props=sim_utils.MassCfg(mass=1.0),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            ),
        )
        clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), (cfg,), num_cubes, 1.0, positions=origins)
        cube_object = RigidObject(cfg)

        # Play sim
        replicate(sim.get_clone_plan())
        sim.reset()

        # Get masses before updating one environment.
        original_masses = cube_object.data.body_mass.torch.clone()
        raw_model_inv_mass = cube_object.root_view.get_attribute("body_inv_mass", SimulationManager.get_model())[:, 0]
        raw_model_inv_inertia = cube_object.root_view.get_attribute("body_inv_inertia", SimulationManager.get_model())[
            :, 0
        ]
        assert cube_object.data._sim_bind_body_inv_mass.ptr == raw_model_inv_mass.ptr
        assert cube_object.data._sim_bind_body_inv_inertia.ptr == raw_model_inv_inertia.ptr
        model_inv_mass = cube_object.data._sim_bind_body_inv_mass
        model_inv_inertia = cube_object.data._sim_bind_body_inv_inertia
        original_inv_mass = wp.to_torch(model_inv_mass).clone()
        original_inv_inertia = wp.to_torch(model_inv_inertia).clone()

        assert original_masses.shape == (num_cubes, 1)

        env_ids = torch.tensor([1], dtype=torch.int32, device=device)
        body_ids = torch.tensor([0], dtype=torch.int32, device=device)

        # A positive-to-zero transition makes the selected body static.
        zero_mass = torch.zeros(1, 1, device=device)
        cube_object.set_masses_index(masses=zero_mass, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(wp.to_torch(model_inv_mass)[1], torch.zeros_like(original_inv_mass[1]))
        torch.testing.assert_close(wp.to_torch(model_inv_inertia)[1], torch.zeros_like(original_inv_inertia[1]))
        torch.testing.assert_close(wp.to_torch(model_inv_mass)[0], original_inv_mass[0])
        torch.testing.assert_close(wp.to_torch(model_inv_inertia)[0], original_inv_inertia[0])

        # Inertia writes keep inverse mass unchanged and respect the body's current static state.
        wp.to_torch(model_inv_inertia)[1].copy_(torch.eye(3, device=device).reshape(1, 3, 3))
        inertia_matrix = torch.diag(torch.tensor([2.0, 3.0, 4.0], device=device))
        inertias = inertia_matrix.reshape(1, 1, 9)
        cube_object.set_inertias_index(inertias=inertias, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(wp.to_torch(model_inv_mass)[1], torch.zeros_like(original_inv_mass[1]))
        torch.testing.assert_close(wp.to_torch(model_inv_inertia)[1], torch.zeros_like(original_inv_inertia[1]))

        # A zero-to-positive transition restores both inverse arrays from current primary data.
        masses = original_masses[env_ids][:, body_ids] + 4.0
        cube_object.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(cube_object.data.body_mass.torch[env_ids][:, body_ids], masses)
        torch.testing.assert_close(wp.to_torch(model_inv_mass)[env_ids][:, body_ids], masses.reciprocal())
        torch.testing.assert_close(
            wp.to_torch(model_inv_inertia)[env_ids][:, body_ids],
            torch.linalg.inv(inertia_matrix).reshape(1, 1, 3, 3),
        )

        # Simulate physics
        # perform rendering
        sim.step()
        # update object
        cube_object.update(sim.cfg.dt)

        masses_to_check = cube_object.data.body_mass.torch[env_ids][:, body_ids]

        # Check if mass is set correctly
        torch.testing.assert_close(masses, masses_to_check)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("num_cubes", [3])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_gravity_vec_w_tracks_model_gravity(num_cubes, device):
    """Per-env mutations to Newton's ``model.gravity`` reach ``GRAVITY_VEC_W`` and ``projected_gravity_b``.

    Regression for the pre-fix snapshot: ``GRAVITY_VEC_W`` used to be env 0's
    gravity broadcast to every env, hiding per-env gravity randomization (e.g.
    :class:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`).
    """
    with _newton_sim_context(device, gravity_enabled=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, device=device)
        replicate(sim.get_clone_plan())
        sim.reset()

        # Check that gravity is set correctly
        torch.testing.assert_close(
            cube_object.data.GRAVITY_VEC_W.torch[0], torch.tensor((0.0, 0.0, -9.81), device=device)
        )

        # The free-falling cubes accelerate with gravity and keep their identity orientation.
        for _ in range(2):
            sim.step()
            cube_object.update(sim.cfg.dt)
        gravity = torch.zeros(num_cubes, 1, 6, device=device)
        gravity[:, :, 2] = -9.81
        torch.testing.assert_close(cube_object.data.body_acc_w.torch, gravity)

        # GRAVITY_VEC_W must share storage with Newton's per-env gravity array.
        model = SimulationManager.get_model()
        model_gravity_arr = model.gravity[: model.world_count]
        global_gravity = wp.to_torch(model.gravity)[-1].clone()
        assert cube_object.data.GRAVITY_VEC_W.warp.ptr == model_gravity_arr.ptr
        assert cube_object.data.GRAVITY_VEC_W.shape == (num_cubes,)

        # Exercise the public event and verify the asset sees its per-world writes.
        new_gravity = torch.tensor(
            [[0.1 * (i + 1), 0.2 * (i + 1), -3.0 - float(i)] for i in range(num_cubes)],
            device=device,
            dtype=torch.float32,
        )
        env = SimpleNamespace(sim=sim, device=device, num_envs=num_cubes)
        params = {"gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "operation": "abs"}
        event = randomize_world_gravity(EventTermCfg(func=randomize_world_gravity, params=params), env)
        for row, values in enumerate(new_gravity.tolist()):
            event(env, torch.tensor([row], device=device), (values, values), operation="abs")

        # Live view: new per-env values are visible immediately, no invalidation step.
        torch.testing.assert_close(cube_object.data.GRAVITY_VEC_W.torch, new_gravity)
        torch.testing.assert_close(wp.to_torch(model.gravity)[-1], global_gravity)

        # Recompute the lazily-cached projected_gravity_b without sim.step, so cube
        # orientation stays at identity and the projection equals unit-direction gravity.
        cube_object.update(sim.cfg.dt)
        expected = torch.nn.functional.normalize(new_gravity, dim=-1)
        torch.testing.assert_close(cube_object.data.projected_gravity_b.torch, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@flaky(max_runs=3, min_passes=1)
def test_body_root_state_properties(num_cubes, device):
    """Test the root_com_state_w, root_link_state_w, body_com_state_w, and body_link_state_w properties."""
    with _newton_sim_context(device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)

        # Play sim
        replicate(sim.get_clone_plan())
        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        # Set center of mass offset via Newton API (position only, no quaternion)
        com_pos = offset.unsqueeze(1)  # (N, 1, 3)
        cube_object.set_coms_index(coms=wp.from_torch(com_pos, dtype=wp.vec3f))
        with wp.ScopedDevice(device):
            SimulationManager._solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        # check center of mass has been set
        torch.testing.assert_close(cube_object.data.body_com_pos_b.torch.squeeze(1), offset)

        # random z spin velocity (bounded to keep numerical drift within the position tolerance below)
        spin_twist = torch.zeros(6, device=device)
        spin_twist[5] = 0.5 * torch.randn(1, device=device).clamp(-1.0, 1.0)

        # Simulate physics
        for _ in range(100):
            # spin the object around Z axis (com)
            cube_object.write_root_velocity_to_sim_index(root_velocity=spin_twist.repeat(num_cubes, 1))
            # perform rendering
            sim.step()
            # update object
            cube_object.update(sim.cfg.dt)

            # get state properties
            root_link_pose_w = cube_object.data.root_link_pose_w.torch
            root_link_vel_w = cube_object.data.root_link_vel_w.torch
            root_com_pose_w = cube_object.data.root_com_pose_w.torch
            root_com_vel_w = cube_object.data.root_com_vel_w.torch
            body_link_pose_w = cube_object.data.body_link_pose_w.torch
            body_link_vel_w = cube_object.data.body_link_vel_w.torch
            body_com_pose_w = cube_object.data.body_com_pose_w.torch
            body_com_vel_w = cube_object.data.body_com_vel_w.torch

            # cubes are spinning around center of mass
            # position will not match
            # center of mass position will be constant (i.e. spinning around com)
            _tol = dict(atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(env_pos + offset, root_com_pose_w[..., :3], **_tol)
            torch.testing.assert_close(env_pos + offset, body_com_pose_w[..., :3].squeeze(-2), **_tol)
            # link position will be moving but should stay constant away from center of mass
            root_link_state_pos_rel_com = quat_apply_inverse(
                root_link_pose_w[..., 3:],
                root_link_pose_w[..., :3] - root_com_pose_w[..., :3],
            )
            torch.testing.assert_close(-offset, root_link_state_pos_rel_com, **_tol)
            body_link_state_pos_rel_com = quat_apply_inverse(
                body_link_pose_w[..., 3:],
                body_link_pose_w[..., :3] - body_com_pose_w[..., :3],
            )
            torch.testing.assert_close(-offset, body_link_state_pos_rel_com.squeeze(-2), **_tol)

            # orientation of com will be a constant rotation from link orientation
            com_quat_b = cube_object.data.body_com_quat_b.torch
            com_quat_w = quat_mul(body_link_pose_w[..., 3:], com_quat_b)
            torch.testing.assert_close(com_quat_w, body_com_pose_w[..., 3:], **_tol)
            torch.testing.assert_close(com_quat_w.squeeze(-2), root_com_pose_w[..., 3:], **_tol)

            # root and body link orientations describe the same rigid body
            torch.testing.assert_close(root_link_pose_w[..., 3:], body_link_pose_w[..., 3:].squeeze(-2), **_tol)

            # lin_vel will not match
            # center of mass vel will be constant (i.e. spinning around com)
            torch.testing.assert_close(torch.zeros_like(root_com_vel_w[..., :3]), root_com_vel_w[..., :3], **_tol)
            torch.testing.assert_close(torch.zeros_like(body_com_vel_w[..., :3]), body_com_vel_w[..., :3], **_tol)
            # link frame will be moving, and should be equal to input angular velocity cross offset
            lin_vel_rel_root_gt = quat_apply_inverse(root_link_pose_w[..., 3:], root_link_vel_w[..., :3])
            lin_vel_rel_body_gt = quat_apply_inverse(body_link_pose_w[..., 3:], body_link_vel_w[..., :3])
            lin_vel_rel_gt = torch.linalg.cross(spin_twist.repeat(num_cubes, 1)[..., 3:], -offset)
            torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_root_gt, **_tol)
            torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_body_gt.squeeze(-2), **_tol)

            # ang_vel will always match
            torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
            torch.testing.assert_close(root_com_vel_w[..., 3:], body_com_vel_w[..., 3:].squeeze(-2))
            torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    ("num_cubes", "state_location"), [(2, "com"), (2, "link"), (2, "root"), (1, "root")]
)  # num_cubes=1 covers single-instance initialization
@pytest.mark.parametrize("device", test_devices())
def test_write_root_state(num_cubes, device, state_location):
    """Test the root state setters in the center-of-mass frame, the link frame, and the default root frames.

    A write must be readable in the written frame and refresh the derived frame and the body-frame caches
    without a sim step, and the written state must persist into the solver across a step.
    """
    with _newton_sim_context(device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Create a scene with random cubes
        cube_object, env_pos = generate_cubes_scene(num_cubes=num_cubes, height=0.0, device=device)
        env_idx = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32, device=device)

        # Play sim
        replicate(sim.get_clone_plan())
        sim.reset()

        # Check if cube_object is initialized
        assert cube_object.is_initialized

        # change center of mass offset from link frame
        offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)

        # Set center of mass offset via Newton API (position only)
        com_pos = offset.unsqueeze(1)  # (N, 1, 3)
        cube_object.set_coms_index(coms=wp.from_torch(com_pos, dtype=wp.vec3f))
        with wp.ScopedDevice(device):
            SimulationManager._solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        # check center of mass has been set
        torch.testing.assert_close(cube_object.data.body_com_pos_b.torch.squeeze(1), offset)

        for i in range(2):
            # perform step
            sim.step()
            # update buffers
            cube_object.update(sim.cfg.dt)

            # A target state distinct from the current one in position, orientation, and velocity.
            target_pose = torch.cat(
                [env_pos + 0.3 * torch.rand(num_cubes, 3, device=device), random_orientation(num_cubes, device)],
                dim=-1,
            )
            target_vel = torch.randn(num_cubes, 6, device=device)

            # Prime the lazily-derived caches at the current sim timestamp. Without this they would
            # recompute on first access after the write regardless of invalidation; priming them makes a
            # missing reset_pose/reset_velocity observable as a stale read in the assertions below.
            _ = cube_object.data.root_link_pose_w.torch
            _ = cube_object.data.root_com_pose_w.torch
            _ = cube_object.data.root_link_vel_w.torch
            _ = cube_object.data.root_com_vel_w.torch

            # Alternate between the default and explicit environment selectors.
            env_ids = None if i == 0 else env_idx
            if state_location == "com":
                cube_object.write_root_com_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
                cube_object.write_root_com_velocity_to_sim_index(root_velocity=target_vel, env_ids=env_ids)
            elif state_location == "link":
                cube_object.write_root_link_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
                cube_object.write_root_link_velocity_to_sim_index(root_velocity=target_vel, env_ids=env_ids)
            elif state_location == "root":
                cube_object.write_root_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
                cube_object.write_root_velocity_to_sim_index(root_velocity=target_vel, env_ids=env_ids)

            # Snapshot the body-frame caches *before* reading the root-frame caches: touching a
            # root cache lazily recomputes the shared buffer and would mask a stale body cache.
            # The body-frame caches must already reflect the write on their own (regression:
            # body_com_pose_w returned the pre-write buffer after a link-frame pose write).
            body_link_pose_w = cube_object.data.body_link_pose_w.torch.squeeze(1).clone()
            body_com_pose_w = cube_object.data.body_com_pose_w.torch.squeeze(1).clone()
            body_com_vel_w = cube_object.data.body_com_vel_w.torch.squeeze(1).clone()

            root_link_pose_w = cube_object.data.root_link_pose_w.torch
            root_com_pose_w = cube_object.data.root_com_pose_w.torch
            root_link_vel_w = cube_object.data.root_link_vel_w.torch
            root_com_vel_w = cube_object.data.root_com_vel_w.torch
            body_com_pose_b = cube_object.data.body_com_pose_b.torch
            if state_location == "com":
                torch.testing.assert_close(target_pose, root_com_pose_w)
                torch.testing.assert_close(target_vel, root_com_vel_w)
                # the com pose was written, so the derived link pose must be refreshed
                expected_root_link_pos, expected_root_link_quat = combine_frame_transforms(
                    root_com_pose_w[:, :3],
                    root_com_pose_w[:, 3:],
                    quat_rotate(quat_inv(body_com_pose_b[:, 0, 3:7]), -body_com_pose_b[:, 0, :3]),
                    quat_inv(body_com_pose_b[:, 0, 3:7]),
                )
                expected_root_link_pose = torch.cat((expected_root_link_pos, expected_root_link_quat), dim=1)
                torch.testing.assert_close(expected_root_link_pose, root_link_pose_w)
            else:
                torch.testing.assert_close(target_pose, root_link_pose_w)
                # the root velocity is the center-of-mass velocity
                written_vel_w = root_link_vel_w if state_location == "link" else root_com_vel_w
                torch.testing.assert_close(target_vel, written_vel_w)
                # the link pose was written, so the derived com pose must be refreshed
                expected_com_pos, expected_com_quat = combine_frame_transforms(
                    root_link_pose_w[:, :3],
                    root_link_pose_w[:, 3:],
                    body_com_pose_b[:, 0, :3],
                    body_com_pose_b[:, 0, 3:7],
                )
                expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1)
                torch.testing.assert_close(expected_com_pose, root_com_pose_w)
            # skip lin_vel because it differs between the frames; angular velocity is frame-independent
            # and only matches when the derived velocity was actually refreshed after the write
            torch.testing.assert_close(root_com_vel_w[:, 3:], root_link_vel_w[:, 3:])

            # For a single-body rigid object the body-frame caches are exactly the root-frame
            # caches reshaped, so they must stay consistent after a write without a sim step.
            torch.testing.assert_close(root_link_pose_w, body_link_pose_w)
            torch.testing.assert_close(root_com_pose_w, body_com_pose_w)
            torch.testing.assert_close(root_link_vel_w, cube_object.data.body_link_vel_w.torch.squeeze(1))
            torch.testing.assert_close(root_com_vel_w, body_com_vel_w)

        # The written state persists into the solver: with gravity off, one step only integrates the
        # written velocity.
        written_pose_w = (root_com_pose_w if state_location == "com" else root_link_pose_w).clone()
        written_com_vel_w = root_com_vel_w.clone()
        sim.step()
        cube_object.update(sim.cfg.dt)
        pose_w = cube_object.data.root_com_pose_w if state_location == "com" else cube_object.data.root_link_pose_w
        torch.testing.assert_close(pose_w.torch, written_pose_w, rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(cube_object.data.root_com_vel_w.torch, written_com_vel_w, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.isaacsim_ci
def test_body_link_pose_w_fresh_after_root_pose_write(device):
    """Regression: ``body_link_pose_w`` must reflect a freshly written root pose without an intervening sim step.

    After ``write_root_{link,com}_pose_to_sim_{index,mask}``, the cached ``_sim_bind_body_link_pose_w``
    (Newton ``body_q``) is stale until forward kinematics is re-evaluated. The getter must call
    :meth:`SimulationManager.forward` so the returned tensor matches the written pose. Without the fix,
    the getter returns the pre-write value. The write must also dirty the simulator-side
    ``_fk_reset_mask`` so collision queries (which read ``body_q`` directly, not via the property)
    re-run FK before the next step.
    """

    def _fk_reset_mask_dirty() -> bool:
        assert SimulationManager._fk_reset_mask is not None
        return bool(wp.to_torch(SimulationManager._fk_reset_mask).any().item())

    num_cubes = 2
    with _newton_sim_context(device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=num_cubes, height=0.5, device=device)

        replicate(sim.get_clone_plan())
        sim.reset()
        assert cube_object.is_initialized

        # Step once so that _sim_timestamp > 0 and caches are primed.
        sim.step()
        cube_object.update(sim.cfg.dt)

        for writer in ("link_index", "link_mask", "com_index", "com_mask"):
            # Prime the body_link_pose_w cache with the current pose.
            pre_write_pose = wp.to_torch(cube_object.data.body_link_pose_w).clone().view(num_cubes, 7)

            # Clear the dirty flag so we can observe that the write sets it.
            SimulationManager.forward()
            assert not _fk_reset_mask_dirty()

            # Build a target pose clearly distinct from the current one in both translation and orientation.
            # Quaternion in (x, y, z, w) for 90° about z: [0, 0, sin(pi/4), cos(pi/4)] = [0, 0, sqrt(0.5), sqrt(0.5)].
            target_pose = wp.to_torch(cube_object.data.root_link_pose_w).clone()
            target_pose[..., 0] += 10.0
            target_pose[..., 1] += 5.0
            target_pose[..., 2] += 2.0
            sqrt_half = 0.7071067811865476
            target_pose[..., 3] = 0.0
            target_pose[..., 4] = 0.0
            target_pose[..., 5] = sqrt_half
            target_pose[..., 6] = sqrt_half

            if writer == "link_index":
                cube_object.write_root_link_pose_to_sim_index(root_pose=target_pose)
            elif writer == "link_mask":
                cube_object.write_root_link_pose_to_sim_mask(root_pose=target_pose)
            elif writer == "com_index":
                cube_object.write_root_com_pose_to_sim_index(root_pose=target_pose)
            elif writer == "com_mask":
                cube_object.write_root_com_pose_to_sim_mask(root_pose=target_pose)

            # The simulator-side dirty flag must be set before any property read clears it via forward().
            assert _fk_reset_mask_dirty(), f"{writer} pose write must call SimulationManager.invalidate_fk()"

            # Read without stepping: getter must trigger forward kinematics and return the fresh pose.
            body_link = wp.to_torch(cube_object.data.body_link_pose_w).view(num_cubes, 7)
            # Defeat alias accidents: the property must not still return the pre-write value.
            assert not torch.allclose(body_link[..., :3], pre_write_pose[..., :3], rtol=1e-4, atol=1e-4), (
                f"body_link_pose_w returned the pre-write cached pose after {writer}; forward() was not invoked"
            )
            # Translation must match the write.
            torch.testing.assert_close(body_link[..., :3], target_pose[..., :3], rtol=1e-4, atol=1e-4)
            # Orientation: compare via |q1 · q2| ≈ 1 to account for the q ≡ -q double cover.
            quat_dot = torch.abs((body_link[..., 3:7] * target_pose[..., 3:7]).sum(dim=-1))
            torch.testing.assert_close(quat_dot, torch.ones_like(quat_dot), rtol=1e-4, atol=1e-4)
