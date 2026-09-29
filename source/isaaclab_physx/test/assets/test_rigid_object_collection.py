# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real PhysX rigid-object-collection coverage.

Most checks run against one module-scoped scene of locally authored cubes. PhysX stores collection bodies in
body-major view order, so writes select non-sorted environment and body subsets and read the view back. Tests
that need their own simulation context are defined first: pytest runs them before the composite scene is
created, and a new simulation context would replace the composite stage.
"""

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from isaaclab.test.utils import launch_test_simulation, test_devices

launch_test_simulation()

import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import RigidObjectCollection
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim import SimulationContext, build_simulation_context
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_apply_inverse, quat_mul


def generate_cubes_scene(
    num_envs: int = 1,
    num_cubes: int = 1,
    height=1.0,
    has_api: bool = True,
    kinematic_enabled: bool = False,
    disable_gravity: bool = False,
    root: str = "/World",
    y_offset: float = 0.0,
) -> tuple[RigidObjectCollection, torch.Tensor]:
    """Generate a collection of local 1 kg cubes, 3 m apart along y within each environment.

    Args:
        num_envs: Number of envs to generate.
        num_cubes: Number of cubes to generate.
        height: Height of the cubes.
        has_api: Whether the cubes have a rigid body API on them.
        kinematic_enabled: Whether the cubes are kinematic.
        disable_gravity: Whether the cubes ignore gravity.
        root: Prim path under which the environments are created.
        y_offset: Offset of the environments along y [m].

    Returns:
        A tuple containing the rigid object collection and the origins of the environments.
    """
    origins = torch.tensor([(i * 3.0, y_offset, height) for i in range(num_envs)])
    # Create Top-level Xforms, one for each cube
    for i, origin in enumerate(origins.tolist()):
        sim_utils.create_prim(f"{root}/Table_{i}", "Xform", translation=origin)

    # Resolve spawn configuration
    if has_api:
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=[
                sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic_enabled),
                PhysxRigidBodyCfg(disable_gravity=disable_gravity),
            ],
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        )
    else:
        # since no rigid body properties defined, this is just a static collider
        spawn_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1), collision_props=sim_utils.UsdPhysicsCollisionCfg())

    # create the rigid object configs
    cube_config_dict = {}
    for i in range(num_cubes):
        cube_config_dict[f"cube_{i}"] = RigidObjectCfg(
            prim_path=f"{root}/Table_[^/]*/Object_{i}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3 * i, height)),
        )
    # create the rigid object collection
    return RigidObjectCollection(cfg=RigidObjectCollectionCfg(rigid_objects=cube_config_dict)), origins


def _yaw_quat(angle: float) -> tuple[float, float, float, float]:
    """Return a unit quaternion ``(x, y, z, w)`` for a rotation of ``angle`` [rad] about the world z axis."""
    return (0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle))


@pytest.fixture
def sim(request) -> Iterator[SimulationContext]:
    """Create a function-scoped simulation context for tests that own their scene."""
    device = request.getfixturevalue("device")
    gravity_enabled = request.getfixturevalue("gravity_enabled") if "gravity_enabled" in request.fixturenames else True
    with build_simulation_context(device=device, auto_add_lighting=True, gravity_enabled=gravity_enabled) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


##
# Tests that own their simulation context. Keep them above the composite scene.
##


@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
def test_initialization_with_no_rigid_body(sim, num_cubes, device) -> None:
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    object_collection, _ = generate_cubes_scene(num_cubes=num_cubes, has_api=False)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(object_collection) < 10

    # Play sim
    with pytest.raises(RuntimeError):
        sim.reset()


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [False])
def test_subset_write_reaches_selected_view_entry(sim, device, gravity_enabled) -> None:
    """A write to one (env, body) cell must move only that object in the simulation."""
    object_collection, _ = generate_cubes_scene(num_envs=2, num_cubes=3)

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


@pytest.mark.parametrize("device", test_devices())
def test_inertial_property_subset_writes_reach_selected_view_entries(sim, device) -> None:
    """COM and inertia writes to a subset of (env, body) cells must reach only those PhysX view entries."""
    num_envs, num_cubes = 2, 3
    object_collection, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes)
    sim.reset()

    def read_view(values: wp.array, data_dim: int) -> torch.Tensor:
        # The view is body-major, (num_cubes * num_envs, data_dim); return it as (num_envs, num_cubes, data_dim)
        return wp.to_torch(values).reshape(num_cubes, num_envs, data_dim).transpose(0, 1).cpu()

    # Write in non-sorted env order so that a wrong env/body to view index mapping cannot pass
    env_ids = [1, 0]
    body_ids = [2]

    initial_coms = read_view(object_collection.root_view.get_coms().view(wp.float32), 7)
    coms = initial_coms[env_ids][:, body_ids].clone()
    coms[0, 0, :3] = torch.tensor([0.02, 0.03, 0.04])
    coms[1, 0, :3] = torch.tensor([-0.01, 0.01, 0.02])
    object_collection.set_coms_index(coms=coms.to(device), env_ids=env_ids, body_ids=body_ids)
    expected_coms = initial_coms.clone()
    expected_coms[env_ids, 2] = coms[:, 0]
    torch.testing.assert_close(read_view(object_collection.root_view.get_coms().view(wp.float32), 7), expected_coms)
    torch.testing.assert_close(object_collection.data.body_com_pose_b.torch.cpu(), expected_coms)

    initial_inertias = read_view(object_collection.root_view.get_inertias(), 9)
    inertias = initial_inertias[env_ids][:, body_ids].clone()
    inertias[0, 0, [0, 4, 8]] *= 1.5
    inertias[1, 0, [0, 4, 8]] *= 2.0
    object_collection.set_inertias_index(inertias=inertias.to(device), env_ids=env_ids, body_ids=body_ids)
    expected_inertias = initial_inertias.clone()
    expected_inertias[env_ids, 2] = inertias[:, 0]
    torch.testing.assert_close(read_view(object_collection.root_view.get_inertias(), 9), expected_inertias)
    torch.testing.assert_close(object_collection.data.body_inertia.torch.cpu(), expected_inertias)


@pytest.mark.parametrize("num_envs", [3])
@pytest.mark.parametrize("num_cubes", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [True, False])
def test_gravity_vec_w(sim, num_envs, num_cubes, device, gravity_enabled) -> None:
    """Test that gravity vector direction is set correctly for the rigid object."""
    object_collection, _ = generate_cubes_scene(num_envs=num_envs, num_cubes=num_cubes)

    gravity_dir = (0.0, 0.0, -1.0) if gravity_enabled else (0.0, 0.0, 0.0)

    sim.reset()

    gravity_vec = object_collection.data.GRAVITY_VEC_W.torch
    assert gravity_vec[0, 0, 0] == gravity_dir[0]
    assert gravity_vec[0, 0, 1] == gravity_dir[1]
    assert gravity_vec[0, 0, 2] == gravity_dir[2]

    for _ in range(2):
        sim.step()
        object_collection.update(sim.cfg.dt)

        # Expected gravity value is the acceleration of the body
        gravity = torch.zeros(num_envs, num_cubes, 6, device=device)
        if gravity_enabled:
            gravity[..., 2] = -9.81

        torch.testing.assert_close(object_collection.data.body_com_acc_w.torch, gravity)


##
# Composite scene shared by the remaining tests.
##

_NUM_ENVS = 2
_NUM_CUBES = 3


@dataclass
class _CollectionScene:
    """Rigid-object collections that share one real PhysX lifecycle under gravity."""

    sim: SimulationContext
    device: str
    cubes: RigidObjectCollection
    """Two environments of three dynamic cubes that ignore gravity."""
    kinematic: RigidObjectCollection
    """One environment of one kinematic cube."""
    origins: dict[str, torch.Tensor]
    """Environment origins keyed by collection name."""
    refcounts: dict[str, int]
    """Reference count of each collection right after construction."""

    def step(self, num_steps: int = 1) -> None:
        """Write, step, and update every collection."""
        for _ in range(num_steps):
            for collection in (self.cubes, self.kinematic):
                collection.write_data_to_sim()
            self.sim.step()
            for collection in (self.cubes, self.kinematic):
                collection.update(self.sim.cfg.dt)

    def place_cubes_at_rest(self, body_poses: torch.Tensor) -> None:
        """Teleport the dynamic cubes to ``body_poses`` at rest and clear their external wrenches."""
        self.cubes.write_body_link_pose_to_sim_index(body_poses=body_poses)
        self.cubes.write_body_com_velocity_to_sim_index(
            body_velocities=torch.zeros((_NUM_ENVS, _NUM_CUBES, 6), device=self.device)
        )
        self.cubes.permanent_wrench_composer.reset()
        self.cubes.instantaneous_wrench_composer.reset()


@pytest.fixture(scope="module", params=test_devices())
def collection_scene(request) -> Iterator[_CollectionScene]:
    """Initialize the composite collection scene once per device."""
    device = request.param
    with build_simulation_context(device=device, gravity_enabled=True) as sim:
        sim._app_control_on_stop_handle = None
        cubes, cube_origins = generate_cubes_scene(
            num_envs=_NUM_ENVS, num_cubes=_NUM_CUBES, disable_gravity=True, root="/World/Cubes"
        )
        kinematic, kinematic_origins = generate_cubes_scene(
            num_envs=1, num_cubes=1, kinematic_enabled=True, root="/World/Kinematic", y_offset=12.0
        )
        refcounts = {"cubes": sys.getrefcount(cubes), "kinematic": sys.getrefcount(kinematic)}
        sim.reset()
        yield _CollectionScene(
            sim=sim,
            device=device,
            cubes=cubes,
            kinematic=kinematic,
            origins={"cubes": cube_origins.to(device), "kinematic": kinematic_origins.to(device)},
            refcounts=refcounts,
        )


def _rest_poses(scene: _CollectionScene, yaw: float = 0.0) -> torch.Tensor:
    """Return the default body poses of the dynamic cubes at their environment origins with a given yaw."""
    poses = scene.cubes.data.default_body_pose.torch.clone()
    poses[..., :2] += scene.origins["cubes"].unsqueeze(1)[..., :2]
    poses[..., 3:] = torch.tensor(_yaw_quat(yaw), device=scene.device)
    return poses


def test_collection_initialization(collection_scene: _CollectionScene) -> None:
    """Initialize local collections, including a single-cube one; kinematic cubes hold their pose under gravity."""
    scene = collection_scene
    for name, collection, num_envs, num_cubes in (
        ("cubes", scene.cubes, _NUM_ENVS, _NUM_CUBES),
        ("kinematic", scene.kinematic, 1, 1),
    ):
        # Check that the framework doesn't hold excessive strong references.
        assert scene.refcounts[name] < 10
        assert collection.is_initialized
        assert collection.num_instances == num_envs
        assert len(collection.body_names) == num_cubes
        assert collection.data.body_link_pos_w.torch.shape == (num_envs, num_cubes, 3)
        assert collection.data.body_link_quat_w.torch.shape == (num_envs, num_cubes, 4)
        assert collection.data.body_mass.torch.shape == (num_envs, num_cubes)
        assert collection.data.body_inertia.torch.shape == (num_envs, num_cubes, 9)

    kinematic = scene.kinematic
    for _ in range(2):
        scene.step()
        default_body_pose = kinematic.data.default_body_pose.torch.clone()
        default_body_vel = kinematic.data.default_body_vel.torch.clone()
        default_body_pose[..., :3] += scene.origins["kinematic"].unsqueeze(1)
        torch.testing.assert_close(kinematic.data.body_link_pose_w.torch, default_body_pose)
        torch.testing.assert_close(kinematic.data.body_link_vel_w.torch, default_body_vel)


def test_collection_state_writes(collection_scene: _CollectionScene) -> None:
    """Body state writes reach the selected body-major view entries in the frame they are given in."""
    scene = collection_scene
    device = scene.device
    collection = scene.cubes
    scene.place_cubes_at_rest(_rest_poses(scene))

    # A non-sorted partial write reaches only the selected view entries.
    env_ids = torch.tensor([1, 0], dtype=torch.int32, device=device)
    body_ids = torch.tensor([2], dtype=torch.int32, device=device)
    initial_pose = collection.data.body_link_pose_w.torch.clone()
    initial_velocity = collection.data.body_com_vel_w.torch.clone()
    target_pose = initial_pose[env_ids][:, body_ids].clone()
    target_pose[0, 0, :3] += torch.tensor([0.2, 0.3, 0.4], device=device)
    target_pose[1, 0, :3] += torch.tensor([-0.1, -0.2, 0.1], device=device)
    target_pose[..., 3:] = torch.tensor(_yaw_quat(0.3), device=device)
    target_velocity = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], [[-0.6, -0.5, -0.4, -0.3, -0.2, -0.1]]])
    target_velocity = target_velocity.to(device)
    collection.write_body_link_pose_to_sim_index(body_poses=target_pose, env_ids=env_ids, body_ids=body_ids)
    collection.write_body_com_velocity_to_sim_index(body_velocities=target_velocity, env_ids=env_ids, body_ids=body_ids)
    expected_pose = initial_pose.clone()
    expected_pose[env_ids[:, None], body_ids] = target_pose
    expected_velocity = initial_velocity.clone()
    expected_velocity[env_ids[:, None], body_ids] = target_velocity
    torch.testing.assert_close(collection.data.body_link_pose_w.torch, expected_pose)
    torch.testing.assert_close(collection.data.body_com_vel_w.torch, expected_velocity)
    for raw, expected in (
        (collection.root_view.get_transforms(), expected_pose),
        (collection.root_view.get_velocities(), expected_velocity),
    ):
        raw_by_env = wp.to_torch(raw).to(device).reshape(_NUM_CUBES, _NUM_ENVS, -1).transpose(0, 1)
        torch.testing.assert_close(raw_by_env, expected)

    # With a center-of-mass offset, every writer round-trips through its own frame.
    scene.place_cubes_at_rest(_rest_poses(scene))
    offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(_NUM_ENVS, _NUM_CUBES, 1)
    coms = collection.data.body_com_pose_b.torch.clone()
    coms[..., :3] = offset
    coms[..., 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    collection.set_coms_index(coms=coms)
    # random pose and velocity so frame conversions see a non-trivial rotation
    rand_state = torch.rand(_NUM_ENVS, _NUM_CUBES, 13, device=device)
    rand_state[..., :3] += collection.data.body_link_pos_w.torch
    rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)
    all_ids = {"env_ids": torch.arange(_NUM_ENVS, device=device), "body_ids": torch.arange(_NUM_CUBES, device=device)}
    for state_location in ("com", "link", "root"):
        for ids in ({}, all_ids):
            if state_location == "com":
                collection.write_body_com_pose_to_sim_index(body_poses=rand_state[..., :7], **ids)
                collection.write_body_com_velocity_to_sim_index(body_velocities=rand_state[..., 7:], **ids)
                torch.testing.assert_close(rand_state[..., :7], collection.data.body_com_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], collection.data.body_com_vel_w.torch)
            elif state_location == "link":
                collection.write_body_link_pose_to_sim_index(body_poses=rand_state[..., :7], **ids)
                collection.write_body_link_velocity_to_sim_index(body_velocities=rand_state[..., 7:], **ids)
                torch.testing.assert_close(rand_state[..., :7], collection.data.body_link_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], collection.data.body_link_vel_w.torch)
                # PhysX holds the center-of-mass velocity: the link velocity plus the angular velocity crossed
                # with the world-frame center-of-mass offset.
                com_offset_w = quat_apply(rand_state[..., 3:7], offset)
                expected_com_velocity = rand_state[..., 7:].clone()
                expected_com_velocity[..., :3] += torch.linalg.cross(rand_state[..., 10:], com_offset_w, dim=-1)
                raw_velocity = wp.to_torch(collection.root_view.get_velocities()).to(device)
                raw_velocity = raw_velocity.reshape(_NUM_CUBES, _NUM_ENVS, 6).transpose(0, 1)
                torch.testing.assert_close(raw_velocity, expected_com_velocity)
            else:
                collection.write_body_link_pose_to_sim_index(body_poses=rand_state[..., :7], **ids)
                collection.write_body_com_velocity_to_sim_index(body_velocities=rand_state[..., 7:], **ids)
                torch.testing.assert_close(rand_state[..., :7], collection.data.body_link_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], collection.data.body_com_vel_w.torch)

            # the frame that was not written must follow through the center-of-mass offset
            link_pose_w = collection.data.body_link_pose_w.torch
            body_com_pose_b = collection.data.body_com_pose_b.torch
            expected_com_pos, expected_com_quat = combine_frame_transforms(
                link_pose_w[..., :3].reshape(-1, 3),
                link_pose_w[..., 3:].reshape(-1, 4),
                body_com_pose_b[..., :3].reshape(-1, 3),
                body_com_pose_b[..., 3:].reshape(-1, 4),
            )
            expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1).view(_NUM_ENVS, _NUM_CUBES, 7)
            torch.testing.assert_close(expected_com_pose, collection.data.body_com_pose_w.torch)
            torch.testing.assert_close(
                collection.data.body_com_vel_w.torch[..., 3:], collection.data.body_link_vel_w.torch[..., 3:]
            )
            # Move away from the written state so that the next pass writes it again.
            scene.step()

    # Spinning about the center of mass keeps it in place while the link frame orbits it.
    scene.place_cubes_at_rest(_rest_poses(scene))
    spin_twist = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.3], device=device).repeat(_NUM_ENVS, _NUM_CUBES, 1)
    init_com = collection.data.body_com_pose_w.torch[..., :3].clone()
    for _ in range(10):
        collection.write_body_com_velocity_to_sim_index(body_velocities=spin_twist)
        scene.step()
        object_link_pose_w = collection.data.body_link_pose_w.torch
        object_link_vel_w = collection.data.body_link_vel_w.torch
        object_com_pose_w = collection.data.body_com_pose_w.torch
        object_com_vel_w = collection.data.body_com_vel_w.torch
        # center of mass position will be constant (i.e. spinning around com)
        torch.testing.assert_close(init_com, object_com_pose_w[..., :3])
        # link position will be moving but should stay constant away from center of mass
        object_link_state_pos_rel_com = quat_apply_inverse(
            object_link_pose_w[..., 3:], object_link_pose_w[..., :3] - object_com_pose_w[..., :3]
        )
        torch.testing.assert_close(-offset, object_link_state_pos_rel_com)
        # orientation of com will be a constant rotation from link orientation
        com_quat_w = quat_mul(object_link_pose_w[..., 3:], collection.data.body_com_quat_b.torch)
        torch.testing.assert_close(com_quat_w, object_com_pose_w[..., 3:])
        # center of mass is at rest while the link frame moves with angular velocity cross offset
        torch.testing.assert_close(torch.zeros_like(object_com_vel_w[..., :3]), object_com_vel_w[..., :3])
        lin_vel_rel_object_gt = quat_apply_inverse(object_link_pose_w[..., 3:], object_link_vel_w[..., :3])
        lin_vel_rel_gt = torch.linalg.cross(spin_twist[..., 3:], -offset)
        torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_object_gt, atol=1e-4, rtol=1e-3)
        # ang_vel will always match
        torch.testing.assert_close(object_com_vel_w[..., 3:], object_link_vel_w[..., 3:])


def test_collection_masses_reach_selected_view_entries(collection_scene: _CollectionScene) -> None:
    """Mass writes to a non-sorted subset reach only the selected body-major view entries."""
    scene = collection_scene
    device = scene.device
    collection = scene.cubes
    env_ids = torch.tensor([1, 0], dtype=torch.int32, device=device)
    body_ids = torch.tensor([1], dtype=torch.int32, device=device)
    expected = collection.data.body_mass.torch.clone()
    masses = torch.tensor([[5.0], [7.0]], device=device)
    expected[env_ids[:, None], body_ids] = masses
    collection.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(collection.data.body_mass.torch, expected)
    raw_mass = wp.to_torch(collection.root_view.get_masses()).to(device).reshape(_NUM_CUBES, _NUM_ENVS).T
    torch.testing.assert_close(raw_mass, expected)
    # Restore uniform masses for the wrench checks.
    collection.set_masses_index(masses=torch.ones((_NUM_ENVS, _NUM_CUBES), device=device))


def test_collection_wrench_delivery_and_reset(collection_scene: _CollectionScene) -> None:
    """External wrenches act on the selected bodies in the frame they are given in; reset clears them."""
    scene = collection_scene
    device = scene.device
    collection = scene.cubes
    rest_poses = _rest_poses(scene, yaw=0.5 * math.pi)
    # Give every body a 1 kg mass and the center of mass and inertia of the first body so that their responses are
    # comparable.
    collection.set_masses_index(masses=torch.ones((_NUM_ENVS, _NUM_CUBES), device=device))
    coms = collection.data.body_com_pose_b.torch[:1, :1]
    collection.set_coms_index(coms=coms.expand(_NUM_ENVS, _NUM_CUBES, 7).contiguous())
    inertias = collection.data.body_inertia.torch[:1, :1]
    collection.set_inertias_index(inertias=inertias.expand(_NUM_ENVS, _NUM_CUBES, 9).contiguous())

    # A body-frame force on bodies 0 and 2 accelerates only those bodies, along the rotated force direction; the
    # same force given in the world frame produces the same response.
    object_ids, _ = collection.find_bodies(".*")
    for local_wrench, global_wrench, response, is_linear in (
        (
            {"forces": [[6.0, 0.0, 0.0]]},
            {"forces": [[0.0, 6.0, 0.0]]},
            lambda: collection.data.body_com_lin_vel_w,
            True,
        ),
        (
            {"forces": [[0.0, 0.0, 6.0]], "positions": [[0.0, 0.1, 0.0]]},
            {"forces": [[0.0, 0.0, 6.0]], "positions": [[-0.1, 0.0, 0.0]]},
            lambda: collection.data.body_com_ang_vel_b,
            False,
        ),
    ):
        scene.place_cubes_at_rest(rest_poses)
        for env_id, wrench, is_global in ((1, local_wrench, False), (0, global_wrench, True)):
            wrench = {key: torch.tensor(value, device=device).expand(1, 2, 3).clone() for key, value in wrench.items()}
            if is_global and "positions" in wrench:
                # World-frame positions add the rotated body-frame lever arm to the center of mass.
                wrench["positions"] += collection.data.body_com_pos_w.torch[env_id : env_id + 1, 0::2]
            collection.permanent_wrench_composer.set_forces_and_torques_index(
                body_ids=object_ids[0::2], env_ids=[env_id], is_global=is_global, **wrench
            )
        scene.step()
        response_value = response().torch
        torch.testing.assert_close(response_value[0], response_value[1], atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(response_value[:, 1], torch.zeros_like(response_value[:, 1]), atol=1e-5, rtol=0)
        if is_linear:
            assert torch.all(response_value[:, 0::2, 1] > 1e-2), response_value
            # The permanent force persists across steps, so a second step doubles the velocity.
            first_step_value = response_value.clone()
            scene.step()
            torch.testing.assert_close(response().torch, 2.0 * first_step_value, atol=1e-4, rtol=1e-3)
    # An upward force 0.1 m along the body y-axis rolls the cube about its x-axis.
    assert torch.all(collection.data.body_com_ang_vel_b.torch[:, 0::2, 0] > 0.1)
    heading = quat_apply(
        collection.data.body_link_quat_w.torch, torch.tensor([1.0, 0.0, 0.0], device=device).expand(2, 3, 3)
    )
    assert torch.all(heading[..., 1] > 0.99)

    # A partial reset clears the external wrenches of the selected environments only.
    composers = (collection.instantaneous_wrench_composer, collection.permanent_wrench_composer)
    ones = torch.ones((_NUM_ENVS, _NUM_CUBES, 3), device=device)
    collection.permanent_wrench_composer.set_forces_and_torques_index(forces=ones, torques=ones)
    collection.instantaneous_wrench_composer.add_forces_and_torques_index(forces=ones, torques=ones)
    collection.reset(env_ids=torch.tensor([0], device=device))
    for composer in composers:
        assert composer.active
        for buffer in (composer.out_force_b.torch, composer.out_torque_b.torch):
            assert torch.count_nonzero(buffer[0]) == 0
            assert torch.count_nonzero(buffer[1:]) == buffer[1:].numel()

    # A full reset clears every environment
    collection.reset()
    for composer in composers:
        assert torch.count_nonzero(composer.out_force_b.torch) == 0
        assert torch.count_nonzero(composer.out_torque_b.torch) == 0
