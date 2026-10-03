# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real PhysX rigid-object and rigid-object-collection coverage.

Most checks run against one module-scoped scene of locally authored cubes, holding both asset families. Each asset
holds two environments so that partial writes can target one environment and prove that the other is preserved in
the real PhysX state; PhysX stores collection bodies in body-major view order, so collection writes select
non-sorted environment and body subsets and read the view back. Tests that need their own simulation context are
defined first: pytest runs them before the composite scene is created, and a new simulation context would replace
the composite stage.
"""

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

launch_test_simulation(physics="isaacsim_physx")

import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.assets import RigidObject, RigidObjectCollection
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import omni.kit.app
import omni.physx
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.sim.schemas import apply_articulation_root_properties
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_apply_inverse, quat_mul

_NUM_ENVS = 2
_NUM_CUBES = 3
"""Cubes per environment of the dynamic collection."""


def _cube_spawn(*, kinematic: bool = False, disable_gravity: bool = False) -> sim_utils.CuboidCfg:
    """Create the spawn configuration of a local 1 kg collision cube."""
    return sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic),
            PhysxRigidBodyCfg(disable_gravity=disable_gravity),
        ],
        mass_props=sim_utils.MassCfg(mass=1.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )


_STATIC_COLLIDER = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1), collision_props=sim_utils.UsdPhysicsCollisionCfg())
"""A cube without rigid body properties, which PhysX treats as a static collider."""


def _cube_cfg(prim_path: str, **spawn_kwargs) -> RigidObjectCfg:
    """Create a local cube one meter above its environment origin."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=_cube_spawn(**spawn_kwargs),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def _spawn_envs(root: str, y_offset: float = 0.0) -> torch.Tensor:
    """Create two environment prims 2 m apart and return their origins."""
    origins = torch.tensor([(2.0 * index, y_offset, 0.0) for index in range(_NUM_ENVS)])
    for index, origin in enumerate(origins.tolist()):
        sim_utils.create_prim(f"{root}/Env_{index}", "Xform", translation=origin)
    return origins


def _collection(
    root: str, num_envs: int, num_cubes: int, spawn: sim_utils.CuboidCfg, y_offset: float = 0.0
) -> tuple[RigidObjectCollection, torch.Tensor]:
    """Create a collection of cubes 3 m apart along y within environments 3 m apart along x, 1 m above ground.

    Returns:
        The collection and the origins of its environments.
    """
    origins = torch.tensor([(3.0 * index, y_offset, 1.0) for index in range(num_envs)])
    for index, origin in enumerate(origins.tolist()):
        sim_utils.create_prim(f"{root}/Table_{index}", "Xform", translation=origin)
    rigid_objects = {
        f"cube_{index}": RigidObjectCfg(
            prim_path=f"{root}/Table_[^/]*/Object_{index}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 3.0 * index, 1.0)),
        )
        for index in range(num_cubes)
    }
    return RigidObjectCollection(cfg=RigidObjectCollectionCfg(rigid_objects=rigid_objects)), origins


def _yaw_quat(angle: float) -> tuple[float, float, float, float]:
    """Return a unit quaternion ``(x, y, z, w)`` for a rotation of ``angle`` [rad] about the world z axis."""
    return (0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle))


@pytest.fixture
def sim(device: str) -> Iterator[SimulationContext]:
    """Create a function-scoped simulation context for tests that own their scene."""
    # A new context would replace the stage of a live composite scene, so these tests must run before it.
    assert SimulationContext.instance() is None, "define tests that own a simulation above the composite scene"
    with build_simulation_context(device=device, auto_add_lighting=True, sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01)) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


##
# Tests that own their simulation context. Keep them above the composite scene.
##


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="The first CPU simulation after a failed initialization reads zero poses"
)
def test_collection_without_rigid_bodies_fails_without_leaking_into_the_next_cpu_simulation() -> None:
    """A collection of static colliders fails to initialize; a kinematic cube then reports its spawn pose in the next
    CPU simulation.

    The failure only reproduces before any CUDA simulation ran in the process, so this test comes first.
    """
    with build_simulation_context(device="cpu", sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01)) as sim:
        sim._app_control_on_stop_handle = None
        invalid_collection, _ = _collection("/World", 1, 2, _STATIC_COLLIDER)
        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(invalid_collection) < 10
        with pytest.raises(RuntimeError):
            sim.reset()
    with build_simulation_context(device="cpu", sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01)) as sim:
        sim._app_control_on_stop_handle = None
        kinematic, origins = _collection("/World", 1, 1, _cube_spawn(kinematic=True))
        sim.reset()
        torch.testing.assert_close(kinematic.data.body_link_pos_w.torch[:, 0], origins + torch.tensor([0.0, 0.0, 1.0]))


@pytest.mark.parametrize(
    "api",
    [
        "none",
        pytest.param(
            "articulation_root",
            marks=pytest.mark.xfail(
                strict=True,
                raises=pytest.fail.Exception,
                reason="The PhysX rigid object does not reject an enabled articulation root.",
            ),
        ),
    ],
)
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.isaacsim_ci
def test_initialization_rejects_invalid_rigid_body(sim, device, api: Literal["none", "articulation_root"]) -> None:
    """Initialization fails without a rigid body and when the rigid body is an articulation root."""
    _spawn_envs("/World")
    if api == "none":
        # Without rigid body properties the cube is a static collider.
        cube_object = RigidObject(cfg=RigidObjectCfg(prim_path="/World/Env_[^/]*/Object", spawn=_STATIC_COLLIDER))
    else:
        cube_object = RigidObject(cfg=_cube_cfg("/World/Env_[^/]*/Object"))
        # An enabled articulation root on the rigid body turns the body into an articulation link.
        assert apply_articulation_root_properties(
            "/World/Env_[^/]*/Object",
            [PhysxArticulationCfg(articulation_enabled=True)],
            stage=sim.stage,
            create_if_missing=True,
        )
        for index in range(_NUM_ENVS):
            prim = sim.stage.GetPrimAtPath(f"/World/Env_{index}/Object")
            assert prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            assert prim.GetAttribute("physxArticulation:articulationEnabled").Get()

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(cube_object) < 10

    with pytest.raises(RuntimeError):
        sim.reset()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("device", test_devices())
def test_warmup_loads_physics_once(sim, device) -> None:
    """Attach on GPU or force-load on CPU, without destroying and rebuilding native objects."""
    sim_utils.create_prim("/World/Table_0", "Xform", translation=(0.0, 0.0, 1.0))
    cube = RigidObject(cfg=_cube_cfg("/World/Table_[^/]*/Object"))

    # Wrap read-only native interfaces through their accessors; real calls still execute.
    physx_spy = MagicMock(wraps=omni.physx.get_physx_interface())
    physx_sim_spy = MagicMock(wraps=omni.physx.get_physx_simulation_interface())
    with (
        patch("omni.physx.get_physx_interface", return_value=physx_spy),
        patch("omni.physx.get_physx_simulation_interface", return_value=physx_sim_spy),
    ):
        sim.reset()

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    assert extension_manager.is_extension_enabled("omni.physics.physx"), (
        "The omni.physics.physx bridge must register PhysX with the unified physics API."
    )
    assert physx_sim_spy.attach_stage.call_count == int(device.startswith("cuda"))
    assert physx_spy.force_load_physics_from_usd.call_count == int(device == "cpu")
    assert cube.is_initialized and cube.num_instances == 1
    initial = cube.data.root_pos_w.torch.clone()
    for _ in range(10):
        sim.step(render=False)
        cube.update(sim.get_physics_dt())
    assert torch.all(cube.data.root_pos_w.torch[:, 2] < initial[:, 2])


##
# Composite scene shared by the remaining tests.
##


@dataclass
class _Scene:
    """Rigid objects and collections that share one real PhysX lifecycle under gravity."""

    sim: SimulationContext
    device: str
    cubes: RigidObject
    """Dynamic cubes that ignore gravity."""
    kinematic: RigidObject
    """Kinematic cubes."""
    falling: RigidObject
    """Dynamic cubes under gravity."""
    collection: RigidObjectCollection
    """Two environments of three dynamic cubes that ignore gravity."""
    kinematic_collection: RigidObjectCollection
    """One environment of one kinematic cube."""
    falling_collection: RigidObjectCollection
    """Two environments of two dynamic cubes under gravity."""
    origins: dict[str, torch.Tensor]
    """Environment origins keyed by asset name."""
    refcounts: dict[str, int]
    """Reference count of each asset right after construction."""

    @property
    def assets(self) -> tuple[RigidObject | RigidObjectCollection, ...]:
        return (
            self.cubes,
            self.kinematic,
            self.falling,
            self.collection,
            self.kinematic_collection,
            self.falling_collection,
        )

    def step(self, num_steps: int = 1) -> None:
        """Write, step, and update every asset."""
        for _ in range(num_steps):
            for asset in self.assets:
                asset.write_data_to_sim()
            self.sim.step()
            for asset in self.assets:
                asset.update(self.sim.cfg.dt)

    def place_cubes_at_rest(self, root_pose: torch.Tensor) -> None:
        """Teleport the dynamic cubes to ``root_pose`` at rest and clear their external wrenches."""
        self.cubes.write_root_link_pose_to_sim_index(root_pose=root_pose)
        self.cubes.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((_NUM_ENVS, 6), device=self.device))
        self.cubes.reset()

    def place_collection_at_rest(self, yaw: float = 0.0) -> torch.Tensor:
        """Teleport the dynamic collection to its default poses with a given yaw, at rest and without wrenches.

        Returns:
            The body poses the collection was placed at.
        """
        poses = self.collection.data.default_body_pose.torch.clone()
        poses[..., :2] += self.origins["collection"].unsqueeze(1)[..., :2]
        poses[..., 3:] = torch.tensor(_yaw_quat(yaw), device=self.device)
        self.collection.write_body_link_pose_to_sim_index(body_poses=poses)
        self.collection.write_body_com_velocity_to_sim_index(
            body_velocities=torch.zeros((_NUM_ENVS, _NUM_CUBES, 6), device=self.device)
        )
        self.collection.permanent_wrench_composer.reset()
        self.collection.instantaneous_wrench_composer.reset()
        return poses


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CUDA))
def scene(request) -> Iterator[_Scene]:
    """Initialize the composite scene once."""
    device = request.param
    with build_simulation_context(device=device, sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01)) as sim:
        sim._app_control_on_stop_handle = None
        origins = {
            "cubes": _spawn_envs("/World/Cubes"),
            "kinematic": _spawn_envs("/World/Kinematic", 3.0),
            "falling": _spawn_envs("/World/Falling", 6.0),
        }
        assets = {
            "cubes": RigidObject(cfg=_cube_cfg("/World/Cubes/Env_[^/]*/Object", disable_gravity=True)),
            "kinematic": RigidObject(cfg=_cube_cfg("/World/Kinematic/Env_[^/]*/Object", kinematic=True)),
            "falling": RigidObject(cfg=_cube_cfg("/World/Falling/Env_[^/]*/Object")),
        }
        for name, num_envs, num_cubes, spawn, y_offset in (
            ("collection", _NUM_ENVS, _NUM_CUBES, _cube_spawn(disable_gravity=True), 10.0),
            ("kinematic_collection", 1, 1, _cube_spawn(kinematic=True), 22.0),
            ("falling_collection", _NUM_ENVS, 2, _cube_spawn(), 26.0),
        ):
            assets[name], origins[name] = _collection(f"/World/{name}", num_envs, num_cubes, spawn, y_offset)
        refcounts = {name: sys.getrefcount(asset) for name, asset in assets.items()}
        sim.reset()
        yield _Scene(
            sim=sim,
            device=device,
            origins={name: value.to(device) for name, value in origins.items()},
            refcounts=refcounts,
            **assets,
        )


@pytest.mark.isaacsim_ci
def test_initialization(scene: _Scene) -> None:
    """Initialize local rigid objects and collections, including a single-cube collection; under gravity, kinematic
    cubes hold their default pose and dynamic cubes accelerate downward on every step."""
    for name, asset in zip(
        ("cubes", "kinematic", "collection", "kinematic_collection"), scene.assets[:2] + scene.assets[3:5]
    ):
        # Check that the framework doesn't hold excessive strong references.
        assert scene.refcounts[name] < 10, name
        assert asset.is_initialized, name
    for rigid_object in (scene.cubes, scene.kinematic):
        assert rigid_object.num_instances == _NUM_ENVS
        assert len(rigid_object.body_names) == 1
        assert rigid_object.data.root_pos_w.torch.shape == (_NUM_ENVS, 3)
        assert rigid_object.data.root_quat_w.torch.shape == (_NUM_ENVS, 4)
        assert rigid_object.data.body_mass.torch.shape == (_NUM_ENVS, 1)
        assert rigid_object.data.body_inertia.torch.shape == (_NUM_ENVS, 1, 9)
        torch.testing.assert_close(
            rigid_object.data.GRAVITY_VEC_W.torch, torch.tensor([[0.0, 0.0, -1.0]] * _NUM_ENVS, device=scene.device)
        )
    for collection, num_envs, num_cubes in (
        (scene.collection, _NUM_ENVS, _NUM_CUBES),
        (scene.kinematic_collection, 1, 1),
    ):
        assert collection.num_instances == num_envs
        assert len(collection.body_names) == num_cubes
        assert collection.data.body_link_pos_w.torch.shape == (num_envs, num_cubes, 3)
        assert collection.data.body_link_quat_w.torch.shape == (num_envs, num_cubes, 4)
        assert collection.data.body_mass.torch.shape == (num_envs, num_cubes)
        assert collection.data.body_inertia.torch.shape == (num_envs, num_cubes, 9)
        torch.testing.assert_close(
            collection.data.GRAVITY_VEC_W.torch[..., 2], torch.full((num_envs, num_cubes), -1.0, device=scene.device)
        )

    gravity_acceleration = torch.tensor([0.0, 0.0, -9.81, 0.0, 0.0, 0.0], device=scene.device)
    kinematic, kinematic_collection = scene.kinematic, scene.kinematic_collection
    for _ in range(2):
        scene.step()
        torch.testing.assert_close(scene.falling.data.body_acc_w.torch, gravity_acceleration.expand(_NUM_ENVS, 1, 6))
        torch.testing.assert_close(
            scene.falling_collection.data.body_com_acc_w.torch, gravity_acceleration.expand(_NUM_ENVS, 2, 6)
        )
        default_root_pose = kinematic.data.default_root_pose.torch.clone()
        default_root_pose[:, :3] += scene.origins["kinematic"]
        torch.testing.assert_close(kinematic.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(kinematic.data.root_com_vel_w.torch, kinematic.data.default_root_vel.torch)
        default_body_pose = kinematic_collection.data.default_body_pose.torch.clone()
        default_body_pose[..., :3] += scene.origins["kinematic_collection"].unsqueeze(1)
        torch.testing.assert_close(kinematic_collection.data.body_link_pose_w.torch, default_body_pose)
        torch.testing.assert_close(
            kinematic_collection.data.body_link_vel_w.torch, kinematic_collection.data.default_body_vel.torch
        )


@pytest.mark.isaacsim_ci
def test_rigid_object_inertial_properties(scene: _Scene) -> None:
    """Mass, center-of-mass, and inertia writes reach only the selected PhysX entries and survive a step."""
    device = scene.device
    cube_object = scene.cubes

    # Full-data mass writes reach the view and survive a simulation step.
    original_masses = wp.to_torch(cube_object.root_view.get_masses()).clone()
    assert original_masses.shape == (_NUM_ENVS, 1)
    masses = original_masses + torch.tensor([[4.5], [6.5]])
    cube_object.set_masses_index(masses=masses.to(device))
    torch.testing.assert_close(cube_object.data.body_mass.torch, masses.to(device))
    torch.testing.assert_close(wp.to_torch(cube_object.root_view.get_masses()), masses)
    scene.step()
    torch.testing.assert_close(cube_object.data.body_mass.torch, masses.to(device))
    torch.testing.assert_close(wp.to_torch(cube_object.root_view.get_masses()), masses)

    # Partial writes with an int64 selector change environment 1 only.
    env_ids = torch.tensor([1], dtype=torch.int64, device=device)
    body_ids = torch.tensor([0], dtype=torch.int32, device=device)
    expected = {
        "body_mass": cube_object.data.body_mass.torch.clone(),
        "body_com_pose_b": cube_object.data.body_com_pose_b.torch.clone(),
        "body_inertia": cube_object.data.body_inertia.torch.clone(),
    }
    expected["body_mass"][1, 0] = 3.0
    expected["body_com_pose_b"][1, 0, :3] = torch.tensor([0.03, -0.02, 0.01], device=device)
    expected["body_inertia"][1, 0, [0, 4, 8]] *= torch.tensor([1.2, 1.3, 1.4], device=device)
    cube_object.set_masses_index(masses=expected["body_mass"][1:], env_ids=env_ids, body_ids=body_ids)
    cube_object.set_coms_index(coms=expected["body_com_pose_b"][1:], env_ids=env_ids, body_ids=body_ids)
    cube_object.set_inertias_index(inertias=expected["body_inertia"][1:], env_ids=env_ids, body_ids=body_ids)
    for name, data, raw in (
        ("body_mass", cube_object.data.body_mass, cube_object.root_view.get_masses()),
        ("body_com_pose_b", cube_object.data.body_com_pose_b, cube_object.root_view.get_coms().view(wp.float32)),
        ("body_inertia", cube_object.data.body_inertia, cube_object.root_view.get_inertias()),
    ):
        torch.testing.assert_close(data.torch, expected[name])
        torch.testing.assert_close(wp.to_torch(raw).to(device).reshape(expected[name].shape), expected[name])


@pytest.mark.isaacsim_ci
def test_rigid_object_root_state_writes(scene: _Scene) -> None:
    """Root writes round-trip through the written frame and keep the other frame consistent with the offset."""
    device = scene.device
    cube_object = scene.cubes
    rest_pose = torch.cat(
        (scene.origins["cubes"], torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(_NUM_ENVS, 1)), -1
    )
    rest_pose[:, 2] += 1.0
    scene.place_cubes_at_rest(rest_pose)

    # A partial pose and velocity write reaches only the selected root.
    initial_pose = cube_object.data.root_link_pose_w.torch.clone()
    initial_velocity = cube_object.data.root_link_vel_w.torch.clone()
    initial_com_velocity = cube_object.data.root_com_vel_w.torch.clone()
    target_pose = initial_pose[1:].clone()
    target_pose[:, :3] += torch.tensor([0.25, -0.1, 0.3], device=device)
    target_pose[:, 3:] = torch.tensor(_yaw_quat(0.4), device=device)
    target_velocity = torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.0, 0.1]], device=device)
    cube_object.write_root_link_pose_to_sim_index(root_pose=target_pose, env_ids=[1])
    cube_object.write_root_link_velocity_to_sim_index(root_velocity=target_velocity, env_ids=[1])
    torch.testing.assert_close(cube_object.data.root_link_pose_w.torch[1:], target_pose)
    torch.testing.assert_close(cube_object.data.root_link_vel_w.torch[1:], target_velocity)
    torch.testing.assert_close(cube_object.data.root_link_pose_w.torch[:1], initial_pose[:1])
    torch.testing.assert_close(cube_object.data.root_link_vel_w.torch[:1], initial_velocity[:1])
    raw_pose = wp.to_torch(cube_object.root_view.get_transforms()).to(device)
    torch.testing.assert_close(raw_pose, torch.cat((initial_pose[:1], target_pose)))
    # PhysX holds the center-of-mass velocity: the link velocity plus the angular velocity crossed with the
    # world-frame center-of-mass offset.
    com_offset_w = quat_apply(target_pose[:, 3:], cube_object.data.body_com_pos_b.torch[1:, 0])
    target_com_velocity = target_velocity.clone()
    target_com_velocity[:, :3] += torch.linalg.cross(target_velocity[:, 3:], com_offset_w)
    raw_velocity = wp.to_torch(cube_object.root_view.get_velocities()).to(device)
    torch.testing.assert_close(raw_velocity, torch.cat((initial_com_velocity[:1], target_com_velocity)))

    # With a center-of-mass offset, every writer round-trips through its own frame.
    offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(_NUM_ENVS, 1)
    coms = cube_object.data.body_com_pose_b.torch.clone()
    coms[:, 0, :3] = offset
    coms[:, 0, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    cube_object.set_coms_index(coms=coms)
    # random pose and velocity so frame conversions see a non-trivial rotation
    rand_state = torch.rand(_NUM_ENVS, 13, device=device)
    rand_state[..., :3] += rest_pose[:, :3]
    rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)
    env_idx = torch.arange(_NUM_ENVS, dtype=torch.int32, device=device)
    for state_location in ("com", "link", "root"):
        for env_ids in (None, env_idx):
            if state_location == "com":
                cube_object.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_ids)
                cube_object.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_ids)
                torch.testing.assert_close(rand_state[..., :7], cube_object.data.root_com_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], cube_object.data.root_com_vel_w.torch)
            elif state_location == "link":
                cube_object.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_ids)
                cube_object.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_ids)
                torch.testing.assert_close(rand_state[..., :7], cube_object.data.root_link_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], cube_object.data.root_link_vel_w.torch)
            else:
                cube_object.write_root_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_ids)
                cube_object.write_root_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_ids)
                torch.testing.assert_close(rand_state[..., :7], cube_object.data.root_link_pose_w.torch)
                torch.testing.assert_close(rand_state[..., 7:], cube_object.data.root_com_vel_w.torch)

            # the frame that was not written must follow through the center-of-mass offset
            root_link_pose_w = cube_object.data.root_link_pose_w.torch
            body_com_pose_b = cube_object.data.body_com_pose_b.torch
            expected_com_pos, expected_com_quat = combine_frame_transforms(
                root_link_pose_w[:, :3],
                root_link_pose_w[:, 3:],
                body_com_pose_b[:, 0, :3],
                body_com_pose_b[:, 0, 3:7],
            )
            torch.testing.assert_close(
                torch.cat((expected_com_pos, expected_com_quat), dim=1), cube_object.data.root_com_pose_w.torch
            )
            torch.testing.assert_close(
                cube_object.data.root_com_vel_w.torch[:, 3:], cube_object.data.root_link_vel_w.torch[:, 3:]
            )
            # Move away from the written state so that the next pass writes it again.
            scene.step()

    # Spinning about the center of mass keeps it in place while the link frame orbits it.
    scene.place_cubes_at_rest(rest_pose)
    spin_twist = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.3], device=device).repeat(_NUM_ENVS, 1)
    initial_com_pos = cube_object.data.root_com_pose_w.torch[:, :3].clone()
    for _ in range(10):
        cube_object.write_root_com_velocity_to_sim_index(root_velocity=spin_twist)
        scene.step()
        root_link_pose_w = cube_object.data.root_link_pose_w.torch
        root_link_vel_w = cube_object.data.root_link_vel_w.torch
        root_com_pose_w = cube_object.data.root_com_pose_w.torch
        root_com_vel_w = cube_object.data.root_com_vel_w.torch
        body_link_pose_w = cube_object.data.body_link_pose_w.torch
        body_com_pose_w = cube_object.data.body_com_pose_w.torch
        # center of mass position will be constant (i.e. spinning around com)
        torch.testing.assert_close(root_com_pose_w[:, :3], initial_com_pos)
        torch.testing.assert_close(body_com_pose_w[:, 0, :3], initial_com_pos)
        # link position will be moving but should stay constant away from center of mass
        torch.testing.assert_close(
            -offset, quat_apply_inverse(root_link_pose_w[:, 3:], root_link_pose_w[:, :3] - root_com_pose_w[:, :3])
        )
        # orientation of com will be a constant rotation from link orientation
        com_quat_w = quat_mul(body_link_pose_w[..., 3:], cube_object.data.body_com_quat_b.torch)
        torch.testing.assert_close(com_quat_w, body_com_pose_w[..., 3:])
        torch.testing.assert_close(com_quat_w[:, 0], root_com_pose_w[:, 3:])
        # center of mass is at rest while the link frame moves with angular velocity cross offset
        torch.testing.assert_close(torch.zeros_like(root_com_vel_w[:, :3]), root_com_vel_w[:, :3])
        lin_vel_rel_root = quat_apply_inverse(root_link_pose_w[:, 3:], root_link_vel_w[:, :3])
        lin_vel_rel_gt = torch.linalg.cross(spin_twist[:, 3:], -offset)
        torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_root, atol=1e-4, rtol=1e-4)
        # ang_vel will always match
        torch.testing.assert_close(root_com_vel_w[:, 3:], root_link_vel_w[:, 3:])


@pytest.mark.isaacsim_ci
def test_rigid_object_wrench_delivery_and_reset(scene: _Scene) -> None:
    """External wrenches act in the frame they are given in on the selected root; reset clears them."""
    device = scene.device
    cube_object = scene.cubes
    rest_pose = torch.cat(
        (scene.origins["cubes"], torch.tensor(_yaw_quat(0.5 * math.pi), device=device).expand(_NUM_ENVS, 4)), -1
    )
    rest_pose[:, 2] += 1.0
    # Give both environments the inertial properties of a uniform 1 kg cube so that their responses are comparable.
    cube_object.set_masses_index(masses=torch.ones((_NUM_ENVS, 1), device=device))
    coms = torch.zeros((_NUM_ENVS, 1, 7), device=device)
    coms[..., 6] = 1.0
    cube_object.set_coms_index(coms=coms)
    inertias = torch.zeros((_NUM_ENVS, 1, 9), device=device)
    inertias[..., [0, 4, 8]] = 1.0 * (0.2**2 + 0.2**2) / 12.0
    cube_object.set_inertias_index(inertias=inertias)

    # A body-frame force on environment 1 accelerates only that cube, along the rotated force direction.
    scene.place_cubes_at_rest(rest_pose)
    cube_object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([[[6.0, 0.0, 0.0]]], device=device), env_ids=[1]
    )
    scene.step()
    velocity = cube_object.data.root_com_lin_vel_w.torch
    assert velocity[1, 1] > 1e-2, velocity
    torch.testing.assert_close(velocity[0], torch.zeros(3, device=device), atol=1e-5, rtol=0.0)

    # A world-frame force, and a force at a position, match their body-frame counterparts.
    for local_wrench, global_wrench, response in (
        ({"forces": [[[6.0, 0.0, 0.0]]]}, {"forces": [[[0.0, 6.0, 0.0]]]}, lambda: cube_object.data.root_com_lin_vel_w),
        (
            {"forces": [[[0.0, 0.0, 6.0]]], "positions": [[[0.0, 0.1, 0.0]]]},
            {"forces": [[[0.0, 0.0, 6.0]]], "positions": [[[-0.1, 0.0, 0.0]]]},
            lambda: cube_object.data.root_com_ang_vel_b,
        ),
    ):
        scene.place_cubes_at_rest(rest_pose)
        local_wrench = {key: torch.tensor(value, device=device) for key, value in local_wrench.items()}
        global_wrench = {key: torch.tensor(value, device=device) for key, value in global_wrench.items()}
        if "positions" in global_wrench:
            # World-frame positions add the rotated body-frame lever arm to the center of mass.
            global_wrench["positions"] = global_wrench["positions"] + cube_object.data.body_com_pos_w.torch[:1]
        cube_object.permanent_wrench_composer.set_forces_and_torques_index(env_ids=[1], **local_wrench)
        cube_object.permanent_wrench_composer.set_forces_and_torques_index(env_ids=[0], is_global=True, **global_wrench)
        scene.step()
        response_value = response().torch
        torch.testing.assert_close(response_value[0], response_value[1], atol=1e-4, rtol=1e-3)
    # An upward force 0.1 m along the body y-axis rolls the cube about its x-axis.
    assert torch.all(cube_object.data.root_com_ang_vel_b.torch[:, 0] > 0.1)
    assert torch.all(
        quat_apply(
            cube_object.data.root_link_quat_w.torch, torch.tensor([1.0, 0.0, 0.0], device=device).repeat(_NUM_ENVS, 1)
        )[:, 1]
        > 0.99
    )

    # A partial reset clears only the selected environment; a full reset deactivates the composers.
    composers = (cube_object.instantaneous_wrench_composer, cube_object.permanent_wrench_composer)
    ones = torch.ones((_NUM_ENVS, 1, 3), device=device)
    cube_object.permanent_wrench_composer.set_forces_and_torques_index(forces=ones, torques=ones)
    cube_object.instantaneous_wrench_composer.add_forces_and_torques_index(forces=ones, torques=ones)
    cube_object.reset(env_ids=torch.tensor([0], device=device))
    for composer in composers:
        assert composer.active
        for buffer in (composer.out_force_b.torch, composer.out_torque_b.torch):
            assert torch.count_nonzero(buffer[0]) == 0
            assert torch.count_nonzero(buffer[1:]) == buffer[1:].numel()
    cube_object.reset()
    for composer in composers:
        assert not composer.active
        assert torch.count_nonzero(composer.out_force_b.torch) == 0
        assert torch.count_nonzero(composer.out_torque_b.torch) == 0


def test_collection_state_writes(scene: _Scene) -> None:
    """Body state writes reach the selected body-major view entries in the frame they are given in."""
    device = scene.device
    collection = scene.collection
    scene.place_collection_at_rest()

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
    scene.place_collection_at_rest()
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
    scene.place_collection_at_rest()
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


def test_collection_inertial_properties_reach_selected_view_entries(scene: _Scene) -> None:
    """Mass, center-of-mass, and inertia writes to a non-sorted subset reach only the selected body-major entries."""
    device = scene.device
    collection = scene.collection
    env_ids = torch.tensor([1, 0], dtype=torch.int32, device=device)
    body_ids = torch.tensor([1], dtype=torch.int32, device=device)
    initial = {
        "masses": collection.data.body_mass.torch.clone(),
        "coms": collection.data.body_com_pose_b.torch.clone(),
        "inertias": collection.data.body_inertia.torch.clone(),
    }
    values = {name: value[env_ids][:, body_ids].clone() for name, value in initial.items()}
    values["masses"][:, 0] = torch.tensor([5.0, 7.0], device=device)
    values["coms"][:, 0, :3] = torch.tensor([[0.02, 0.03, 0.04], [-0.01, 0.01, 0.02]], device=device)
    values["inertias"][0, 0, [0, 4, 8]] *= 1.5
    values["inertias"][1, 0, [0, 4, 8]] *= 2.0
    collection.set_masses_index(masses=values["masses"], env_ids=env_ids, body_ids=body_ids)
    collection.set_coms_index(coms=values["coms"], env_ids=env_ids, body_ids=body_ids)
    collection.set_inertias_index(inertias=values["inertias"], env_ids=env_ids, body_ids=body_ids)
    for name, data, raw in (
        ("masses", collection.data.body_mass, collection.root_view.get_masses()),
        ("coms", collection.data.body_com_pose_b, collection.root_view.get_coms().view(wp.float32)),
        ("inertias", collection.data.body_inertia, collection.root_view.get_inertias()),
    ):
        expected = initial[name].clone()
        expected[env_ids[:, None], body_ids] = values[name]
        torch.testing.assert_close(data.torch, expected)
        # The view is body-major: (num_cubes * num_envs, ...).
        raw = wp.to_torch(raw).to(device).reshape(_NUM_CUBES, _NUM_ENVS, -1).transpose(0, 1)
        torch.testing.assert_close(raw, expected.reshape(_NUM_ENVS, _NUM_CUBES, -1))
    # Restore the initial properties for the wrench checks.
    collection.set_masses_index(masses=initial["masses"])
    collection.set_coms_index(coms=initial["coms"])
    collection.set_inertias_index(inertias=initial["inertias"])


def test_collection_wrench_delivery_and_reset(scene: _Scene) -> None:
    """External wrenches act on the selected bodies in the frame they are given in; reset clears them."""
    device = scene.device
    collection = scene.collection
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
        scene.place_collection_at_rest(yaw=0.5 * math.pi)
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
