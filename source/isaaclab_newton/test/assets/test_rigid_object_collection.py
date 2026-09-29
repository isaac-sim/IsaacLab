# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Kitless real-Newton rigid-object-collection coverage on one persistent scene per device.

Every environment holds three locally spawned cubes in the collection and an unrelated rigid body beside them,
so the collection view must select exactly its configured bodies. Scene gravity is off; a test that needs gravity
applies it to every world for its own duration. Configurations that fail initialization and the single-instance
case build their own scenes.
"""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import warp as wp
from isaaclab_newton.assets import RigidObject, RigidObjectCollection
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim import SimulationContext, build_simulation_context
from isaaclab.test.utils import test_devices
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    quat_rotate,
    random_orientation,
    subtract_frame_transforms,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from articulation_test_utils import env_origins, newton_sim_cfg, spawn_assets, world_gravity  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

_GRAVITY = (0.0, 0.0, -9.81)
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


def _collection_cfg(num_cubes: int, height: float = 1.0, rigid: bool = True) -> RigidObjectCollectionCfg:
    """Return a collection of cubes spaced 3 m apart along y."""
    return RigidObjectCollectionCfg(
        rigid_objects={
            f"cube_{i}": RigidObjectCfg(
                prim_path=f"/World/Env_[^/]*/Object_{i}",
                spawn=_cube_spawn_cfg(rigid),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3.0 * i, height)),
            )
            for i in range(num_cubes)
        }
    )


##
# Own scenes. These tests run before the shared scene exists: only one simulation context can be alive.
##


def test_initialization_with_no_rigid_body() -> None:
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with build_simulation_context(sim_cfg=newton_sim_cfg("cpu")) as sim:
        object_collection = spawn_assets({"collection": _collection_cfg(2, rigid=False)}, num_envs=1)["collection"]

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(object_collection) < 10

        with pytest.raises(RuntimeError, match="Expected 1 prims at"):
            sim.reset()


@pytest.mark.parametrize("device", test_devices())
def test_single_instance_initialization(device: str) -> None:
    """A single-environment, single-body collection initializes with singleton buffers."""
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


##
# Shared scene.
##


@dataclass
class _Scene:
    """Cube collections that share one real Newton model and solver lifecycle."""

    sim: SimulationContext
    collection: RigidObjectCollection
    sibling: RigidObject
    origins: torch.Tensor
    device: str

    def step(self, num_steps: int = 1) -> None:
        for _ in range(num_steps):
            self.collection.write_data_to_sim()
            self.sim.step()
            self.collection.update(self.sim.cfg.dt)

    def rest(self) -> None:
        """Put the cubes at rest at their configured poses."""
        body_pose = self.collection.data.default_body_pose.torch.clone()
        body_pose[..., :3] += self.origins.unsqueeze(1)
        self.collection.write_body_link_pose_to_sim_index(body_poses=body_pose)
        self.collection.write_body_com_velocity_to_sim_index(
            body_velocities=torch.zeros_like(self.collection.data.default_body_vel.torch)
        )
        self.collection.reset()


@pytest.fixture(scope="module", params=test_devices())
def shared_scene(request):
    """Initialize the collection and its unrelated sibling bodies once per device for this module."""
    device = request.param
    sibling_cfg = RigidObjectCfg(
        prim_path="/World/Env_[^/]*/UnrelatedObject",
        spawn=_cube_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -3.0, 1.0)),
    )
    with build_simulation_context(sim_cfg=newton_sim_cfg(device)) as sim:
        assets = spawn_assets({"collection": _collection_cfg(_NUM_CUBES), "sibling": sibling_cfg})
        sim.reset()
        collection = assets["collection"]
        scene = _Scene(
            sim=sim, collection=collection, sibling=assets["sibling"], origins=env_origins(device), device=device
        )
        initial_properties = (
            collection.data.body_mass.torch.clone(),
            collection.data.body_com_pos_b.torch.clone(),
            collection.data.body_inertia.torch.clone(),
        )
        yield scene, initial_properties


@pytest.fixture
def scene(shared_scene) -> _Scene:
    """Hand each test the shared scene at rest with the configured inertial properties."""
    scene, (masses, coms, inertias) = shared_scene
    scene.collection.set_masses_index(masses=masses)
    scene.collection.set_coms_index(coms=coms)
    scene.collection.set_inertias_index(inertias=inertias)
    scene.rest()
    return scene


def test_initialization(scene: _Scene) -> None:
    """Test initialization for prims with rigid body API at the provided prim paths.

    With an unrelated rigid body next to the cubes in each environment, the collection view must still
    select only its configured rigid objects.
    """
    object_collection = scene.collection
    num_envs = object_collection.num_instances

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(object_collection) < 10

    # Check if object is initialized
    assert object_collection.is_initialized
    assert scene.sibling.is_initialized
    assert object_collection.num_instances == 2
    assert object_collection.root_view.count == num_envs * _NUM_CUBES
    assert object_collection.body_names == [f"cube_{i}" for i in range(_NUM_CUBES)]

    # Check buffers that exist and have correct shapes
    assert object_collection.data.default_body_pose.torch.shape == (num_envs, _NUM_CUBES, 7)
    assert object_collection.data.body_link_pos_w.torch.shape == (num_envs, _NUM_CUBES, 3)
    assert object_collection.data.body_link_quat_w.torch.shape == (num_envs, _NUM_CUBES, 4)
    assert object_collection.data.body_mass.torch.shape == (num_envs, _NUM_CUBES)
    assert object_collection.data.body_inertia.torch.shape == (num_envs, _NUM_CUBES, 9)

    # The selected bodies are the configured cubes, at their configured places, and not the sibling.
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


def test_external_force_on_single_body(scene: _Scene) -> None:
    """Test application of external force on the base of the object.

    In the first phase, a force equal to the weight of every 2nd object keeps it in place while the others
    fall. In the second phase, the force is applied at 1m in the Y direction and the object must rotate
    around its X axis. Both phases run in the global and the local frame, and resetting the collection
    must clear the wrench applied in the previous iteration.
    """
    object_collection = scene.collection
    device = scene.device

    # find objects to apply the force
    object_ids, object_names = object_collection.find_bodies(".*")

    def reset_objects():
        scene.rest()

        # Reset should zero external forces and torques
        assert torch.count_nonzero(object_collection.instantaneous_wrench_composer.out_force_b.torch) == 0
        assert torch.count_nonzero(object_collection.instantaneous_wrench_composer.out_torque_b.torch) == 0
        assert torch.count_nonzero(object_collection.permanent_wrench_composer.out_force_b.torch) == 0
        assert torch.count_nonzero(object_collection.permanent_wrench_composer.out_torque_b.torch) == 0

    with world_gravity(_GRAVITY):
        # Sample a force equal to the weight of the object
        external_wrench_b = torch.zeros(object_collection.num_instances, len(object_ids), 6, device=device)
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
            scene.step(10)

            # First object should still be at the same Z position (1.0)
            torch.testing.assert_close(
                object_collection.data.body_link_pos_w.torch[:, 0::2, 2],
                torch.ones_like(object_collection.data.body_link_pos_w.torch[:, 0::2, 2]),
            )
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(object_collection.data.body_link_pos_w.torch[:, 1::2, 2] < 1.0)

        # Apply a force at 1m in the Y direction on every 2nd cube
        external_wrench_b = torch.zeros(object_collection.num_instances, len(object_ids), 6, device=device)
        external_wrench_positions_b = torch.zeros(object_collection.num_instances, len(object_ids), 3, device=device)
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
            scene.step(10)

            # First object should be rotating around it's X axis
            assert torch.all(object_collection.data.body_com_ang_vel_b.torch[:, 0::2, 0] > 0.1)
            # Second object should have fallen, so it's Z height should be less than initial height of 1.0
            assert torch.all(object_collection.data.body_link_pos_w.torch[:, 1::2, 2] < 1.0)


def test_gravity_vec_w_tracks_model_gravity(scene: _Scene) -> None:
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

    # random z spin velocity
    spin_twist = torch.zeros(6, device=device)
    spin_twist[5] = torch.randn(1, device=device)

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
