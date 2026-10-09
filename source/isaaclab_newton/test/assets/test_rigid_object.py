# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Kitless real-Newton rigid-object and rigid-object-collection coverage on one persistent two-environment CPU scene.

Every environment holds a rigid-object cube and a three-cube collection, all locally spawned shapes. The cube
also stands beside the collection as an unrelated rigid body, so the collection view must select exactly its
configured bodies. Scene gravity is off; a test that needs gravity applies it to every world for its own duration.
Configurations that fail initialization and the single-instance cases build their own scenes. Only one simulation
context can be alive, so those tests come first and fail if selected after a shared-scene test.
"""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.assets import RigidObject, RigidObjectCollection
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import ModelFlags
from newton_test_utils import env_origins, local_usd, newton_sim_cfg, spawn_assets, world_gravity

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs.mdp import randomize_physics_scene_gravity
from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationContext, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_rotate,
    random_orientation,
    subtract_frame_transforms,
)

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

_GRAVITY = (0.0, 0.0, -9.81)


def _cube_cfg(height: float = 1.0) -> RigidObjectCfg:
    """Return a dynamic unit-mass cube with a collider."""
    return RigidObjectCfg(
        prim_path="/World/Env_[^/]*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -3.0, height)),
    )


_NUM_CUBES = 3


def _cube_spawn_cfg(rigid: bool = True) -> sim_utils.CuboidCfg:
    """Return a unit-mass cube, or a static collider without a rigid body."""
    if not rigid:
        # since no rigid body properties defined, this is just a static collider
        return sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1), collision_props=sim_utils.UsdPhysicsCollisionCfg())
    return sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        mass_props=sim_utils.MassCfg(mass=1.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )


def _collection_cfg(
    num_cubes: int, height: float = 1.0, rigid: bool = True, *, object_paths: tuple[str, ...] | None = None
) -> RigidObjectCollectionCfg:
    """Return a collection of cubes spaced 3 m apart along y."""
    if object_paths is None:
        object_paths = tuple(f"Object_{i}" for i in range(num_cubes))
    return RigidObjectCollectionCfg(
        rigid_objects={
            f"cube_{i}": RigidObjectCfg(
                prim_path=f"/World/Env_[^/]*/{path}",
                spawn=_cube_spawn_cfg(rigid),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3.0 * i, height)),
            )
            for i, path in enumerate(object_paths)
        }
    )


##
# Own scenes. These tests run before the shared scene exists: only one simulation context can be alive.
##


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("api", ["none", "articulation"])
def test_initialization_rejects_non_rigid_body_prims(api: str) -> None:
    """Initialization fails unless the prim path holds exactly one rigid body.

    A static collider has no rigid body, and an articulation rooted on a rigid body has several.
    """
    if api == "none":
        # since no rigid body properties defined, this is just a static collider
        spawn_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1), collision_props=sim_utils.UsdPhysicsCollisionCfg())
    else:
        spawn_cfg = local_usd("rigid_body_with_articulation_root.usda")
    cube_cfg = RigidObjectCfg(prim_path="/World/Env_[^/]*/Object", spawn=spawn_cfg)
    with build_simulation_context(sim_cfg=newton_sim_cfg("cpu")) as sim:
        cube_object = spawn_assets({"cube": cube_cfg}, num_envs=1)["cube"]

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(cube_object) < 10

        with pytest.raises(RuntimeError, match="Expected 1 prims at"):
            sim.reset()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_single_instance_initialization_and_root_state_write(device: str) -> None:
    """A single rigid object initializes with singleton buffers and reads a root state write back without a step."""
    with build_simulation_context(sim_cfg=newton_sim_cfg(device)) as sim:
        cube_object = spawn_assets({"cube": _cube_cfg()}, num_envs=1)["cube"]
        sim.reset()
        assert cube_object.is_initialized
        assert cube_object.num_instances == 1
        assert cube_object.data.root_pos_w.torch.shape == (1, 3)
        assert cube_object.data.body_mass.torch.shape == (1, 1)
        assert cube_object.data.body_inertia.torch.shape == (1, 1, 9)

        target_pose = torch.cat(
            [torch.tensor([[0.3, -0.2, 1.5]], device=device), random_orientation(1, device)], dim=-1
        )
        target_vel = torch.randn(1, 6, device=device)
        cube_object.write_root_pose_to_sim_index(root_pose=target_pose)
        cube_object.write_root_velocity_to_sim_index(root_velocity=target_vel)
        torch.testing.assert_close(cube_object.data.root_link_pose_w.torch, target_pose)
        torch.testing.assert_close(cube_object.data.root_com_vel_w.torch, target_vel)
        torch.testing.assert_close(cube_object.data.body_link_pose_w.torch.squeeze(1), target_pose)


def test_collection_initialization_with_no_rigid_body() -> None:
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with build_simulation_context(sim_cfg=newton_sim_cfg("cpu")) as sim:
        object_collection = spawn_assets({"collection": _collection_cfg(2, rigid=False)}, num_envs=1)["collection"]

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(object_collection) < 10

        with pytest.raises(RuntimeError, match="Expected 1 prims at"):
            sim.reset()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_collection_single_instance_initialization(device: str) -> None:
    """A singleton collection keeps working after repeated hard resets without accumulating callbacks."""
    with build_simulation_context(sim_cfg=newton_sim_cfg(device)) as sim:
        object_collection = spawn_assets({"collection": _collection_cfg(1)}, num_envs=1)["collection"]
        sim.reset()
        assert object_collection.is_initialized
        assert object_collection.num_instances == 1
        assert object_collection.root_view.count == 1
        assert len(object_collection.body_names) == 1
        assert object_collection.data.default_body_pose.torch.shape == (1, 1, 7)
        assert object_collection.data.body_link_pos_w.torch.shape == (1, 1, 3)
        assert object_collection.data.body_link_quat_w.torch.shape == (1, 1, 4)
        assert object_collection.data.body_mass.torch.shape == (1, 1)
        assert object_collection.data.body_inertia.torch.shape == (1, 1, 9)

        callback_count = len(SimulationManager._callbacks)
        for _ in range(2):
            old_data = object_collection.data
            sim.reset()
            assert object_collection.data is not old_data
            assert len(SimulationManager._callbacks) == callback_count

            pose = object_collection.data.default_body_pose.torch.clone()
            velocity = torch.zeros((1, 1, 6), device=device)
            velocity[..., 0] = 0.5
            object_collection.write_body_pose_to_sim_index(body_poses=pose)
            object_collection.write_body_velocity_to_sim_index(body_velocities=velocity)
            sim.step(render=False)
            object_collection.update(sim.cfg.dt)
            pose[..., 0] += 0.5 * sim.cfg.dt
            torch.testing.assert_close(object_collection.data.body_link_pose_w.torch, pose)


##
# Shared scene.
##


@dataclass
class _Scene:
    """Rigid-object cubes and cube collections that share one real Newton model and solver lifecycle."""

    sim: SimulationContext
    cube: RigidObject
    collection: RigidObjectCollection
    origins: torch.Tensor
    device: str
    inertial: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    """Configured masses [kg], center-of-mass positions [m], and inertias [kg·m^2] per asset, restored before
    each test."""

    def step(self, num_steps: int = 1) -> None:
        """Step ``num_steps`` times, writing the assets' commands before and updating them after each step."""
        for _ in range(num_steps):
            self.cube.write_data_to_sim()
            self.collection.write_data_to_sim()
            self.sim.step()
            self.cube.update(self.sim.cfg.dt)
            self.collection.update(self.sim.cfg.dt)

    def rest(self) -> None:
        """Put every cube at rest at its configured pose."""
        root_pose = self.cube.data.default_root_pose.torch.clone()
        root_pose[:, :3] += self.origins
        self.cube.write_root_pose_to_sim_index(root_pose=root_pose)
        self.cube.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros_like(self.cube.data.default_root_vel.torch)
        )
        self.cube.reset()
        body_pose = self.collection.data.default_body_pose.torch.clone()
        body_pose[..., :3] += self.origins.unsqueeze(1)
        self.collection.write_body_link_pose_to_sim_index(body_poses=body_pose)
        self.collection.write_body_com_velocity_to_sim_index(
            body_velocities=torch.zeros_like(self.collection.data.default_body_vel.torch)
        )
        self.collection.reset()


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CPU))
def shared_scene(request: pytest.FixtureRequest) -> Iterator[_Scene]:
    """Initialize the cubes and the collections once for this module."""
    device = request.param
    with build_simulation_context(sim_cfg=newton_sim_cfg(device)) as sim:
        assets = spawn_assets(
            {
                "cube": _cube_cfg(),
                "collection": _collection_cfg(
                    _NUM_CUBES, object_paths=("Object_0", "RobotLeft/Body", "RobotRight/Body")
                ),
            }
        )
        sim.reset()
        yield _Scene(
            sim=sim,
            cube=assets["cube"],
            collection=assets["collection"],
            origins=env_origins(device),
            device=device,
            inertial={
                name: (
                    asset.data.body_mass.torch.clone(),
                    asset.data.body_com_pos_b.torch.clone(),
                    asset.data.body_inertia.torch.clone(),
                )
                for name, asset in assets.items()
            },
        )


@pytest.fixture
def scene(shared_scene: _Scene) -> _Scene:
    """Hand each test the shared scene at rest with the configured inertial properties."""
    for asset, (masses, coms, inertias) in (
        (shared_scene.cube, shared_scene.inertial["cube"]),
        (shared_scene.collection, shared_scene.inertial["collection"]),
    ):
        asset.set_masses_index(masses=masses)
        asset.set_coms_index(coms=coms)
        asset.set_inertias_index(inertias=inertias)
    shared_scene.rest()
    return shared_scene


@pytest.mark.isaacsim_ci
def test_external_force_on_single_body(scene: _Scene) -> None:
    """A permanent wrench reaches the solver in the local and the global frame, and a reset clears it.

    Cube 0 receives a force equal to its weight and hovers while cube 1 falls. Cube 0 then receives a force set
    and added 1 m off its center along y and must turn about x while cube 1 keeps falling.
    """
    cube = scene.cube
    composer = cube.permanent_wrench_composer
    zeros = torch.zeros(cube.num_instances, 1, 3, device=scene.device)
    weight, lift, lever = zeros.clone(), zeros.clone(), zeros.clone()
    weight[0, :, 2] = 9.81 * cube.data.body_mass.torch[0]
    lift[0, :, 2] = 50.0
    lever[..., 1] = 1.0

    with world_gravity(_GRAVITY):
        for is_global in (True, False):
            for forces, offset, writes in ((weight, zeros, 1), (lift, lever, 2)):
                scene.rest()
                for wrench_composer in (composer, cube.instantaneous_wrench_composer):
                    assert not wrench_composer.active
                    assert torch.count_nonzero(wrench_composer.out_force_b.torch) == 0
                    assert torch.count_nonzero(wrench_composer.out_torque_b.torch) == 0
                positions = offset + cube.data.body_com_pos_w.torch if is_global else offset
                for write in (composer.set_forces_and_torques_index, composer.add_forces_and_torques_index)[:writes]:
                    write(forces=forces, torques=zeros, positions=positions, is_global=is_global)
                scene.step(5)

                if forces is weight:
                    torch.testing.assert_close(cube.data.root_pos_w.torch[0, 2], torch.tensor(1.0, device=scene.device))
                else:
                    assert cube.data.root_ang_vel_b.torch[0, 0].abs() > 0.1
                assert cube.data.root_pos_w.torch[1, 2] < 1.0


@pytest.mark.isaacsim_ci
def test_rigid_body_set_material_properties(scene: _Scene) -> None:
    """Material randomization writes friction and restitution into the Newton model shapes of the selected envs."""
    cube_object = scene.cube
    num_cubes = cube_object.num_instances
    device = scene.device

    # Resolve each cube's shapes from the flat Newton model, independent of the asset's view binding.
    model = SimulationManager.get_model()
    body_world = model.body_world.numpy()
    shape_body = model.shape_body.numpy()
    is_cube = np.asarray([label.endswith("/Object") for label in model.body_label])
    cube_shapes = [
        np.flatnonzero(np.isin(shape_body, np.flatnonzero(is_cube & (body_world == index))))
        for index in range(num_cubes)
    ]
    assert all(len(shapes) > 0 for shapes in cube_shapes)
    original_mu = model.shape_material_mu.numpy().copy()
    original_restitution = model.shape_material_restitution.numpy().copy()

    # Randomize the materials of the last cube through the event term, with degenerate ranges.
    env = SimpleNamespace(scene={"cube": cube_object}, sim=scene.sim, device=device, num_envs=num_cubes)
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
    scene.step()

    mu = model.shape_material_mu.numpy()
    restitution = model.shape_material_restitution.numpy()
    np.testing.assert_allclose(mu[cube_shapes[-1]], 0.55)
    np.testing.assert_allclose(restitution[cube_shapes[-1]], 0.15)
    # Shapes of the other cubes are untouched.
    for shapes in cube_shapes[:-1]:
        np.testing.assert_array_equal(mu[shapes], original_mu[shapes])
        np.testing.assert_array_equal(restitution[shapes], original_restitution[shapes])


@pytest.mark.isaacsim_ci
def test_rigid_body_set_mass(scene: _Scene) -> None:
    """Test that selected mass writes update inverse mass and inertia across static transitions."""
    cube_object = scene.cube
    num_cubes = cube_object.num_instances
    device = scene.device

    # Get masses before updating one environment.
    original_masses = cube_object.data.body_mass.torch.clone()
    raw_model_inv_mass = cube_object.root_view.get_attribute("body_inv_mass", SimulationManager.get_model())[:, 0]
    raw_model_inv_inertia = cube_object.root_view.get_attribute("body_inv_inertia", SimulationManager.get_model())[:, 0]
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
    scene.step()

    # Check if mass is set correctly
    torch.testing.assert_close(masses, cube_object.data.body_mass.torch[env_ids][:, body_ids])


@pytest.mark.isaacsim_ci
def test_gravity_vec_w_tracks_model_gravity(scene: _Scene) -> None:
    """Per-env mutations to Newton's ``model.gravity`` reach ``GRAVITY_VEC_W`` and ``projected_gravity_b``.

    Regression for the pre-fix snapshot: ``GRAVITY_VEC_W`` used to be env 0's
    gravity broadcast to every env, hiding per-env gravity randomization (e.g.
    :class:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`).
    """
    cube_object = scene.cube
    num_cubes = cube_object.num_instances
    device = scene.device

    with world_gravity(_GRAVITY):
        # Check that gravity is set correctly
        torch.testing.assert_close(cube_object.data.GRAVITY_VEC_W.torch[0], torch.tensor(_GRAVITY, device=device))

        # The free-falling cubes accelerate with gravity and keep their identity orientation.
        scene.step(2)
        gravity = torch.zeros(num_cubes, 1, 6, device=device)
        gravity[:, :, 2] = -9.81
        torch.testing.assert_close(cube_object.data.body_acc_w.torch, gravity)

        # One update may span several physics steps, e.g. once per env step when Newton owns decimation.
        scene.sim.step()
        scene.sim.step()
        cube_object.update(2 * scene.sim.cfg.dt)
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
        env = SimpleNamespace(sim=scene.sim, device=device, num_envs=num_cubes)
        params = {"gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "operation": "abs"}
        event = randomize_physics_scene_gravity(EventTermCfg(func=randomize_physics_scene_gravity, params=params), env)
        for row, values in enumerate(new_gravity.tolist()):
            event(env, torch.tensor([row], device=device), (values, values), operation="abs")

        # Live view: new per-env values are visible immediately, no invalidation step.
        torch.testing.assert_close(cube_object.data.GRAVITY_VEC_W.torch, new_gravity)
        torch.testing.assert_close(wp.to_torch(model.gravity)[-1], global_gravity)

        # Recompute the lazily-cached projected_gravity_b without sim.step, so cube
        # orientation stays at identity and the projection equals unit-direction gravity.
        scene.rest()
        cube_object.update(scene.sim.cfg.dt)
        expected = torch.nn.functional.normalize(new_gravity, dim=-1)
        torch.testing.assert_close(cube_object.data.projected_gravity_b.torch, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.isaacsim_ci
def test_body_root_state_properties(scene: _Scene) -> None:
    """Test the root_com_state_w, root_link_state_w, body_com_state_w, and body_link_state_w properties."""
    cube_object = scene.cube
    num_cubes = cube_object.num_instances
    device = scene.device
    env_pos = scene.origins + cube_object.data.default_root_pose.torch[:, :3]

    # change center of mass offset from link frame
    offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
    cube_object.set_coms_index(coms=wp.from_torch(offset.unsqueeze(1), dtype=wp.vec3f))

    # check center of mass has been set
    torch.testing.assert_close(cube_object.data.body_com_pos_b.torch.squeeze(1), offset)

    # spin about z, fast enough for the link-frame velocity to discriminate
    spin_twist = torch.zeros(6, device=device)
    spin_twist[5] = 2.0

    # Simulate physics
    for _ in range(20):
        # spin the object around Z axis (com)
        cube_object.write_root_velocity_to_sim_index(root_velocity=spin_twist.repeat(num_cubes, 1))
        scene.step()

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
def test_write_root_state(scene: _Scene) -> None:
    """Test the root state setters in the center-of-mass frame, the link frame, and the default root frames.

    A write must be readable in the written frame and refresh the derived frame and the body-frame caches
    without a sim step, and the written state must persist into the solver across a step.
    """
    cube_object = scene.cube
    num_cubes = cube_object.num_instances
    device = scene.device
    env_pos = scene.origins
    env_idx = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32, device=device)

    # change center of mass offset from link frame
    offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_cubes, 1)
    cube_object.set_coms_index(coms=wp.from_torch(offset.unsqueeze(1), dtype=wp.vec3f))

    # check center of mass has been set
    torch.testing.assert_close(cube_object.data.body_com_pos_b.torch.squeeze(1), offset)

    for state_location in ("com", "link", "root"):
        for i in range(2):
            # perform step
            scene.step()

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
            # The body-frame caches must reflect the write on their own, including ``body_com_pose_w``
            # after a link-frame pose write and ``body_link_vel_w`` after a center-of-mass velocity write.
            body_link_pose_w = cube_object.data.body_link_pose_w.torch.squeeze(1).clone()
            body_com_pose_w = cube_object.data.body_com_pose_w.torch.squeeze(1).clone()
            body_link_vel_w = cube_object.data.body_link_vel_w.torch.squeeze(1).clone()
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
            torch.testing.assert_close(root_link_vel_w, body_link_vel_w)
            torch.testing.assert_close(root_com_vel_w, body_com_vel_w)

        # The written state persists into the solver: with gravity off, one step only integrates the
        # written velocity.
        written_pose_w = (root_com_pose_w if state_location == "com" else root_link_pose_w).clone()
        written_com_vel_w = root_com_vel_w.clone()
        scene.step()
        pose_w = cube_object.data.root_com_pose_w if state_location == "com" else cube_object.data.root_link_pose_w
        torch.testing.assert_close(pose_w.torch, written_pose_w, rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(cube_object.data.root_com_vel_w.torch, written_com_vel_w, rtol=1e-1, atol=1e-1)


@pytest.mark.isaacsim_ci
def test_body_link_pose_w_fresh_after_root_pose_write(scene: _Scene) -> None:
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

    cube_object = scene.cube
    num_cubes = cube_object.num_instances

    # Step once so that _sim_timestamp > 0 and caches are primed.
    scene.step()

    for writer in ("link_pose_to_sim_index", "link_pose_to_sim_mask", "com_pose_to_sim_index", "com_pose_to_sim_mask"):
        # Prime the body_link_pose_w cache with the current pose.
        pre_write_pose = cube_object.data.body_link_pose_w.torch.clone().view(num_cubes, 7)

        # Clear the dirty flag so we can observe that the write sets it.
        SimulationManager.forward()
        assert not _fk_reset_mask_dirty()

        # A target pose distinct from the current one in translation and in orientation (90 deg about z).
        target_pose = cube_object.data.root_link_pose_w.torch.clone()
        target_pose[:, :3] += torch.tensor([10.0, 5.0, 2.0], device=target_pose.device)
        target_pose[:, 3:] = torch.tensor([0.0, 0.0, 0.5**0.5, 0.5**0.5], device=target_pose.device)
        getattr(cube_object, f"write_root_{writer}")(root_pose=target_pose)

        # The simulator-side dirty flag must be set before any property read clears it via forward().
        assert _fk_reset_mask_dirty(), f"{writer} pose write must call SimulationManager.invalidate_fk()"

        # Read without stepping: getter must trigger forward kinematics and return the fresh pose.
        body_link = cube_object.data.body_link_pose_w.torch.view(num_cubes, 7)
        assert not torch.allclose(body_link[..., :3], pre_write_pose[..., :3], rtol=1e-4, atol=1e-4), (
            f"body_link_pose_w returned the pre-write cached pose after {writer}; forward() was not invoked"
        )
        torch.testing.assert_close(body_link[..., :3], target_pose[..., :3], rtol=1e-4, atol=1e-4)
        # Orientation: compare via |q1 · q2| ≈ 1 to account for the q ≡ -q double cover.
        quat_dot = torch.abs((body_link[..., 3:7] * target_pose[..., 3:7]).sum(dim=-1))
        torch.testing.assert_close(quat_dot, torch.ones_like(quat_dot), rtol=1e-4, atol=1e-4)


##
# Rigid-object collection.
##


def test_collection_initialization(scene: _Scene) -> None:
    """Test initialization for prims with rigid body API at the provided prim paths.

    With the rigid-object cube next to the collection in each environment, the collection view must still
    select only its configured rigid objects.
    """
    object_collection = scene.collection
    num_envs = object_collection.num_instances

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(object_collection) < 10

    # Check if object is initialized
    assert object_collection.is_initialized
    assert scene.cube.is_initialized
    assert object_collection.num_instances == 2
    assert object_collection.root_view.count == num_envs * _NUM_CUBES
    assert object_collection.body_names == [f"cube_{i}" for i in range(_NUM_CUBES)]

    # Check buffers that exist and have correct shapes
    assert object_collection.data.default_body_pose.torch.shape == (num_envs, _NUM_CUBES, 7)
    assert object_collection.data.body_link_pos_w.torch.shape == (num_envs, _NUM_CUBES, 3)
    assert object_collection.data.body_link_quat_w.torch.shape == (num_envs, _NUM_CUBES, 4)
    assert object_collection.data.body_mass.torch.shape == (num_envs, _NUM_CUBES)
    assert object_collection.data.body_inertia.torch.shape == (num_envs, _NUM_CUBES, 9)

    # The selected bodies are the configured cubes, at their configured places, and not the rigid-object cube.
    expected_pos = object_collection.data.default_body_pose.torch[..., :3] + scene.origins.unsqueeze(1)
    torch.testing.assert_close(object_collection.data.body_link_pos_w.torch, expected_pos)


def test_set_body_inertial_properties_updates_inverses(scene: _Scene) -> None:
    """Masked inertial-property writes update only selected Newton inverse entries."""
    object_collection = scene.collection
    device = scene.device
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


def test_collection_external_force_on_single_body(scene: _Scene) -> None:
    """A permanent wrench reaches the solver in the local and the global frame, and a reset clears it.

    Every other cube receives a force equal to its weight and hovers while the rest fall. Those cubes then
    receive a force set and added 1 m off their centers along y and must turn about x.
    """
    collection = scene.collection
    composer = collection.permanent_wrench_composer
    zeros = torch.zeros(collection.num_instances, collection.num_bodies, 3, device=scene.device)
    weight, lift, lever = zeros.clone(), zeros.clone(), zeros.clone()
    weight[:, 0::2, 2] = 9.81 * collection.data.body_mass.torch[:, 0::2]
    lift[:, 0::2, 2] = 50.0
    lever[..., 1] = 1.0

    with world_gravity(_GRAVITY):
        for is_global in (True, False):
            for forces, offset, writes in ((weight, zeros, 1), (lift, lever, 2)):
                scene.rest()
                for wrench_composer in (composer, collection.instantaneous_wrench_composer):
                    assert torch.count_nonzero(wrench_composer.out_force_b.torch) == 0
                    assert torch.count_nonzero(wrench_composer.out_torque_b.torch) == 0
                positions = offset + collection.data.body_link_pos_w.torch if is_global else offset
                for write in (composer.set_forces_and_torques_index, composer.add_forces_and_torques_index)[:writes]:
                    write(forces=forces, torques=zeros, positions=positions, is_global=is_global)
                scene.step(10)

                height = collection.data.body_link_pos_w.torch[..., 2]
                if forces is weight:
                    torch.testing.assert_close(height[:, 0::2], torch.ones_like(height[:, 0::2]))
                else:
                    assert torch.all(collection.data.body_com_ang_vel_b.torch[:, 0::2, 0] > 0.1)
                assert torch.all(height[:, 1::2] < 1.0)


@pytest.mark.isaacsim_ci
def test_collection_gravity_vec_w_tracks_model_gravity(scene: _Scene) -> None:
    """Per-env mutations to Newton's ``model.gravity`` reach ``GRAVITY_VEC_W`` and ``projected_gravity_b``.

    Regression for the pre-fix snapshot: ``GRAVITY_VEC_W`` used to be env 0's
    gravity broadcast to every env and body, hiding per-env gravity
    randomization (e.g. :class:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`).
    """
    object_collection = scene.collection
    num_envs = object_collection.num_instances
    num_cubes = object_collection.num_bodies
    device = scene.device

    with world_gravity(_GRAVITY):
        # Check if gravity vector is set correctly
        torch.testing.assert_close(object_collection.data.GRAVITY_VEC_W.torch[0], torch.tensor(_GRAVITY, device=device))

        # The free-falling cubes accelerate with gravity and keep their identity orientation.
        scene.step(2)
        gravity = torch.zeros(num_envs, num_cubes, 6, device=device)
        gravity[..., 2] = -9.81
        torch.testing.assert_close(object_collection.data.body_com_acc_w.torch, gravity)

        # An environment update may span multiple physics steps under Newton decimation.
        scene.sim.step()
        scene.sim.step()
        object_collection.update(2 * scene.sim.cfg.dt)
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
        object_collection.update(scene.sim.cfg.dt)
        expected_per_env = torch.nn.functional.normalize(new_gravity, dim=-1)
        expected = expected_per_env.unsqueeze(1).expand(-1, num_cubes, -1).contiguous()
        torch.testing.assert_close(object_collection.data.projected_gravity_b.torch, expected, atol=1e-5, rtol=1e-5)


def test_object_state_properties(scene: _Scene) -> None:
    """Test the object_com_state_w and object_link_state_w properties."""
    cube_object = scene.collection
    num_envs = cube_object.num_instances
    num_cubes = cube_object.num_bodies
    device = scene.device

    # change center of mass offset from link frame
    offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)

    # Set center of mass offset via Newton API (position only, shape (E, B, 3))
    cube_object.set_coms_index(coms=wp.from_torch(offset, dtype=wp.vec3f))

    # check center of mass has been set
    torch.testing.assert_close(cube_object.data.body_com_pos_b.torch, offset)

    # spin about z, fast enough for the link-frame velocity to discriminate
    spin_twist = torch.zeros(6, device=device)
    spin_twist[5] = 2.0

    # initial spawn point
    init_com = cube_object.data.body_com_pose_w.torch[..., :3]

    for _ in range(10):
        # spin the object around Z axis (com)
        cube_object.write_body_com_velocity_to_sim_index(body_velocities=spin_twist.repeat(num_envs, num_cubes, 1))
        scene.step()

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
        torch.testing.assert_close(torch.zeros_like(object_com_vel_w[..., :3]), object_com_vel_w[..., :3], **_tol)

        # link frame will be moving, and should be equal to input angular velocity cross offset
        lin_vel_rel_object_gt = quat_apply_inverse(object_link_pose_w[..., 3:], object_link_vel_w[..., :3])
        lin_vel_rel_gt = torch.linalg.cross(spin_twist.repeat(num_envs, num_cubes, 1)[..., 3:], -offset)
        torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_object_gt, **_tol)

        # ang_vel will always match
        torch.testing.assert_close(object_com_vel_w[..., 3:], object_link_vel_w[..., 3:])


def test_write_object_state(scene: _Scene) -> None:
    """Test the object state setters in the center-of-mass frame, the link frame, and the default root frames.

    A write must be readable in the written frame and refresh the derived frame without a sim step, and the
    written state must persist into the solver across a step.
    """
    cube_object = scene.collection
    num_envs = cube_object.num_instances
    num_cubes = cube_object.num_bodies
    device = scene.device
    env_ids = torch.tensor([x for x in range(num_envs)], dtype=torch.int32, device=device)
    object_ids = torch.tensor([x for x in range(num_cubes)], dtype=torch.int32, device=device)

    # change center of mass offset from link frame
    offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(num_envs, num_cubes, 1)

    # Set center of mass offset via Newton API (position only, shape (E, B, 3))
    cube_object.set_coms_index(coms=wp.from_torch(offset, dtype=wp.vec3f))

    # check center of mass has been set
    torch.testing.assert_close(cube_object.data.body_com_pos_b.torch, offset)

    for state_location in ("com", "link", "root"):
        for i in range(2):
            scene.step()

            body_link_pose_w = cube_object.data.body_link_pose_w.torch
            body_com_pose_w = cube_object.data.body_com_pose_w.torch
            object_link_to_com_pos, object_link_to_com_quat = subtract_frame_transforms(
                body_link_pose_w[..., :3].reshape(-1, 3),
                body_link_pose_w[..., 3:7].reshape(-1, 4),
                body_com_pose_w[..., :3].reshape(-1, 3),
                body_com_pose_w[..., 3:7].reshape(-1, 4),
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
                    com_pose_w[..., :3].reshape(-1, 3),
                    com_pose_w[..., 3:].reshape(-1, 4),
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
                    link_pose_w[..., :3].reshape(-1, 3),
                    link_pose_w[..., 3:].reshape(-1, 4),
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
        written_com_vel_w = com_vel_w.clone()
        scene.step()
        pose_w = cube_object.data.body_com_pose_w if state_location == "com" else cube_object.data.body_link_pose_w
        torch.testing.assert_close(pose_w.torch, written_pose_w, rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(cube_object.data.body_com_vel_w.torch, written_com_vel_w, rtol=1e-1, atol=1e-1)


@pytest.mark.isaacsim_ci
def test_body_pose_write_marks_fk_reset_mask(scene: _Scene) -> None:
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

    cube_object = scene.collection
    scene.step()

    for writer in ("link_pose_to_sim_index", "link_pose_to_sim_mask", "com_pose_to_sim_index", "com_pose_to_sim_mask"):
        # Clear the dirty flag so we can observe that the write sets it.
        SimulationManager.forward()
        assert not _fk_reset_mask_dirty()

        pre_write_pose = cube_object.data.body_link_pose_w.torch.clone()
        target_pose = pre_write_pose.clone()
        target_pose[..., :3] += torch.tensor([10.0, 5.0, 2.0], device=target_pose.device)
        getattr(cube_object, f"write_body_{writer}")(body_poses=target_pose)

        assert _fk_reset_mask_dirty(), f"{writer} pose write must call SimulationManager.invalidate_fk()"

        # body_link_pose_w must reflect the write immediately — its underlying buffer is the write
        # target. A regression that moves this property to a separate cached buffer (mirroring the
        # single-object case) would silently break this invariant.
        body_link = cube_object.data.body_link_pose_w.torch
        assert not torch.allclose(body_link[..., :3], pre_write_pose[..., :3], rtol=1e-4, atol=1e-4), (
            f"body_link_pose_w still aliases the pre-write pose after {writer}; the buffer was not written"
        )
        torch.testing.assert_close(body_link[..., :3], target_pose[..., :3], rtol=1e-4, atol=1e-4)
