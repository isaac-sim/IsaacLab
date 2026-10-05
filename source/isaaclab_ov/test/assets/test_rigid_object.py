# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Real OVPhysX rigid-object and rigid-object-collection coverage on one module-scoped scene per device.

Each device builds one scene of locally authored cuboid pairs and ``N=2, B=3`` collections, resets it once,
and keeps it alive for every test that uses it. Each asset belongs to one test, so those tests do not depend
on each other's order. The CPU scene covers host-resident property bindings; the CUDA scene covers their
pinned-host staging.

Initialization failures need their own scenes. A second simulation context cannot start while a shared
scene is alive, so these tests must run before any shared-scene test in the same session; they are
defined first and pytest runs them before it creates the module-scoped scenes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
import warp as wp

from pxr import UsdPhysics

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import RigidObject, RigidObjectCollection  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab import cloner  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context  # noqa: E402
from isaaclab.test.utils import DeviceScope, test_devices  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, quat_mul  # noqa: E402

pytestmark = pytest.mark.integration

_NUM_CUBES = 2
_NUM_ENVS, _NUM_BODIES = 2, 3


def _spawn_cubes(name: str, y_offset: float, **rigid_props) -> RigidObject:
    """Author one pair of local cuboids."""
    for index in range(_NUM_CUBES):
        sim_utils.create_prim(f"/World/{name}/Env_{index}", "Xform", translation=(2.0 * index, y_offset, 0.0))
    return RigidObject(
        RigidObjectCfg(
            prim_path=f"/World/{name}/Env_[^/]*/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(**rigid_props),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
    )


def _spawn_static_colliders() -> RigidObject:
    """Author a collider without a rigid body at the rigid-object path."""
    for index in range(_NUM_CUBES):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(2.0 * index, 0.0, 0.0))
    return RigidObject(
        RigidObjectCfg(
            prim_path="/World/Env_[^/]*/Cube",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1), collision_props=sim_utils.UsdPhysicsCollisionCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
    )


def _spawn_collection(name: str, y_offset: float, spawn: sim_utils.SpawnerCfg | None = None) -> RigidObjectCollection:
    """Author one canonical ``N=2, B=3`` local collection."""
    if spawn is None:
        spawn = sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    for env_index in range(_NUM_ENVS):
        sim_utils.create_prim(f"/World/{name}/Env_{env_index}", "Xform", translation=(3.0 * env_index, y_offset, 0.0))
    return RigidObjectCollection(
        RigidObjectCollectionCfg(
            rigid_objects={
                body_name: RigidObjectCfg(
                    prim_path=f"/World/{name}/Env_[^/]*/Object_{body_index}",
                    spawn=spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, float(body_index), 1.0)),
                )
                for body_index, body_name in enumerate(("left", "middle", "right"))
            }
        )
    )


@configclass
class HeterogeneousRigidSceneCfg(InteractiveSceneCfg):
    """Two object variants dropping onto an independently cloned support."""

    support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.2),
            rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionBaseCfg(),
        ),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2)), sim_utils.SphereCfg(radius=0.1)],
            rigid_props=sim_utils.RigidBodyBaseCfg(),
            collision_props=sim_utils.CollisionBaseCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.4)),
    )


_CPU, _CUDA = test_devices(DeviceScope.CPU), test_devices(DeviceScope.CUDA)
# CUDA-only scene tests keep the CPU row as a skip so that pytest groups every test of one scene device together
# and builds each module-scoped scene once.
_CUDA_SCENES = [
    device if device in _CUDA else pytest.param(device, marks=pytest.mark.skip(reason="CUDA-only path"))
    for device in test_devices()
]


@pytest.mark.parametrize(
    ("device", "support_cloning", "filter_collisions"),
    # CUDA replays native clones, so it covers every cloning path. CPU serializes the full stage, one path.
    [(device, "native", False) for device in _CPU]
    + [
        (device, *row)
        for device in _CUDA
        for row in (("native", False), ("usd_and_native", True), ("usd", False), ("usd_nested", True))
    ],
)
def test_heterogeneous_clone_contacts_and_indexed_state(device, support_cloning, filter_collisions):
    """Variants contact their own support, and indexed writes reach the environment-ordered bodies.

    Collision filtering is an independent scene setting, so it alternates across the cloning paths.
    """
    with build_simulation_context(
        device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device, dt=1.0 / 120.0)
    ) as sim:
        cfg = HeterogeneousRigidSceneCfg(num_envs=6, env_spacing=2.0, filter_collisions=filter_collisions)
        if support_cloning != "native":
            cfg.clone_cfg.clone_template = "/Scenes/World_{}"
            cfg.support.cloning_contexts = ("isaaclab.cloner:UsdReplicateContext",)
            if support_cloning == "usd_and_native":
                cfg.support.cloning_contexts += ("isaaclab_ov.cloner:OvPhysxReplicateContext",)
        if support_cloning == "usd_nested":
            cfg.support.prim_path = "{ENV_REGEX_NS}/Assembly/Support"
            cfg.object.spawn.assets_cfg = cfg.object.spawn.assets_cfg[:1]
            spawn = sim_utils.SpawnerCfg(func=lambda path, cfg, **kwargs: sim.stage.DefinePrim(path, "Xform"))
            cfg.assembly = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Assembly", spawn=spawn, cloning_contexts=None)
            cfg.clone_cfg.clone_combinations = [cloner.InclusionSet(assets=names) for names in (["assembly"], [])]
        scene = InteractiveScene(cfg)
        sim.reset()
        obj = scene["object"]
        expected_paths = [f"{cfg.clone_cfg.clone_template.format(i)}/Object" for i in range(scene.num_envs)]
        assert obj.root_view.prim_paths == expected_paths
        for _ in range(60):
            sim.step()
            scene.update(sim.get_physics_dt())
        positions = obj.data.root_pos_w.torch
        torch.testing.assert_close(positions[:, 2], torch.full_like(positions[:, 2], 0.2), atol=0.02, rtol=0.0)
        torch.testing.assert_close(positions[:, :2], scene.env_origins[:, :2], atol=0.02, rtol=0.0)

        # Lift one resting body; an independent exact-path binding detects a write to the wrong body.
        selected = torch.tensor([3], device=device)
        pose = obj.data.root_pose_w.torch[selected].clone()
        pose[:, 2] = 3.0
        obj.write_root_pose_to_sim(pose, env_ids=selected)
        sim.step()
        scene.update(sim.get_physics_dt())
        binding = OvPhysxManager.get_physx_instance().create_tensor_binding(
            prim_paths=expected_paths, tensor_type=TT.RIGID_BODY_POSE
        )
        try:
            actual = torch.empty(binding.shape, device=device)
            binding.read(actual)
        finally:
            binding.destroy()
        expected_height = torch.full((scene.num_envs,), 0.2, device=device)
        expected_height[selected] = 3.0
        torch.testing.assert_close(actual[:, 2], expected_height, atol=0.02, rtol=0.0)
        torch.testing.assert_close(obj.data.root_pos_w.torch, actual[:, :3])


@pytest.mark.parametrize("device", _CUDA)
def test_heterogeneous_clone_collision_isolation(device):
    """Collision groups isolate overlapping environments, including both retained sources."""
    with build_simulation_context(
        device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device, dt=1.0 / 120.0)
    ) as sim:
        # Without GPU environment-ID filtering, 2048 overlapping worlds exhaust broadphase pairs.
        scene = InteractiveScene(HeterogeneousRigidSceneCfg(num_envs=2048, env_spacing=0.0))
        sim.reset()
        for _ in range(60):
            sim.step()
            scene.update(sim.get_physics_dt())
        positions = scene["object"].data.root_pos_w.torch
        expected = scene.env_origins.clone()
        expected[:, 2] = 0.2
        torch.testing.assert_close(positions, expected, atol=0.02, rtol=0.0)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_initialization_with_no_rigid_body(device: str) -> None:
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with build_simulation_context(device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device)) as sim:
        # Keep the asset alive: only a live asset initializes, and fails, on reset.
        cube_object = _spawn_static_colliders()
        with pytest.raises(RuntimeError, match="Expected 1 prims"):
            sim.reset()
        assert not cube_object.is_initialized


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="OVPhysX RigidObject has no ArticulationRootAPI guard; the rigid body initializes normally.",
)
def test_initialization_with_articulation_root(device: str) -> None:
    """Test that initialization fails when an articulation root is found at the provided prim path."""
    with build_simulation_context(device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device)) as sim:
        # Mark each rigid body as an articulation root; the asset must stay alive for reset.
        _cube_object = _spawn_cubes("Rooted", y_offset=0.0)
        stage = sim_utils.get_current_stage()
        for index in range(_NUM_CUBES):
            UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(f"/World/Rooted/Env_{index}/Cube"))
        with pytest.raises(RuntimeError):
            sim.reset()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_collection_initialization_with_no_rigid_body(device: str) -> None:
    """Test that collection initialization fails when no rigid body is found at a body prim path."""
    with build_simulation_context(
        device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device, gravity=(0.0, 0.0, 0.0))
    ) as sim:
        # Keep the asset alive: only a live asset initializes, and fails, on reset.
        object_collection = _spawn_collection(
            "Static",
            y_offset=0.0,
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1), collision_props=sim_utils.UsdPhysicsCollisionCfg()),
        )
        with pytest.raises(RuntimeError, match="Expected 1 prims"):
            sim.reset()
        assert not object_collection.is_initialized


@dataclass
class _RigidScene:
    """Cuboid pairs and collections that share one real OVPhysX lifecycle."""

    sim: SimulationContext
    device: str
    dynamic: RigidObject
    kinematic: RigidObject
    spinning: RigidObject
    fused: RigidObjectCollection
    stepped: RigidObjectCollection
    wrenched: RigidObjectCollection


@pytest.fixture(scope="module")
def scene(request: pytest.FixtureRequest) -> Iterator[_RigidScene]:
    """Initialize every cuboid pair and collection for one device once for this module."""
    device = request.param
    with build_simulation_context(device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device)) as sim:
        dynamic = _spawn_cubes("Dynamic", y_offset=0.0, disable_gravity=True)
        kinematic = _spawn_cubes("Kinematic", y_offset=2.0, kinematic_enabled=True)
        spinning = _spawn_cubes("Spinning", y_offset=4.0, disable_gravity=True)
        # Collection bodies disable gravity and span three units along y.
        fused = _spawn_collection("Fused", y_offset=7.0)
        stepped = _spawn_collection("Stepped", y_offset=11.0)
        wrenched = _spawn_collection("Wrenched", y_offset=15.0)
        sim.reset()
        yield _RigidScene(sim, device, dynamic, kinematic, spinning, fused, stepped, wrenched)


@pytest.mark.parametrize("scene", test_devices(), indirect=True)
def test_rigid_object_real_ovphysx_seams(scene: _RigidScene) -> None:
    """Prove partial state, inertial properties, one real wrench delivery, and the wrench reset."""
    rigid_object, sim, device = scene.dynamic, scene.sim, scene.device
    assert rigid_object.is_initialized
    assert rigid_object.num_instances == _NUM_CUBES
    assert rigid_object.data.body_mass.torch.shape == (_NUM_CUBES, 1)
    assert rigid_object.data.body_com_pose_b.torch.shape == (_NUM_CUBES, 1, 7)
    assert rigid_object.data.body_inertia.torch.shape == (_NUM_CUBES, 1, 9)

    initial_pose = rigid_object.data.root_link_pose_w.torch.clone()
    target_pose = initial_pose[1:2].clone()
    target_pose[0, :3] += torch.tensor([0.25, -0.1, 0.2], device=device)
    rigid_object.write_root_link_pose_to_sim_index(root_pose=target_pose, env_ids=[1])
    torch.testing.assert_close(rigid_object.data.root_link_pose_w.torch[1:2], target_pose)
    torch.testing.assert_close(rigid_object.data.root_link_pose_w.torch[0:1], initial_pose[0:1])
    raw_pose = wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_POSE)).to(device)
    torch.testing.assert_close(raw_pose, torch.cat((initial_pose[0:1], target_pose)))

    raw_mass_before = wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_MASS)).clone()
    rigid_object.set_masses_index(masses=wp.array([[3.0]], dtype=wp.float32, device=device), env_ids=[1])
    expected_raw_mass = raw_mass_before.clone()
    expected_raw_mass[1] = 3.0
    torch.testing.assert_close(wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_MASS)), expected_raw_mass)

    raw_com_before = wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_COM_POSE)).clone()
    coms = rigid_object.data.body_com_pose_b.torch[1:2].clone()
    coms[0, 0, :3] = torch.tensor([0.01, -0.02, 0.03], device=device)
    rigid_object.set_coms_index(coms=wp.from_torch(coms, dtype=wp.transformf), env_ids=[1])
    expected_raw_com = raw_com_before.clone()
    expected_raw_com[1] = coms[0, 0].cpu()
    torch.testing.assert_close(
        wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_COM_POSE)), expected_raw_com
    )

    raw_inertia_before = wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_INERTIA)).clone()
    inertias = rigid_object.data.body_inertia.torch[1:2].clone()
    inertias[0, 0, 0] *= 1.5
    rigid_object.set_inertias_index(inertias=wp.from_torch(inertias, dtype=wp.float32), env_ids=[1])
    expected_raw_inertia = raw_inertia_before.clone()
    expected_raw_inertia[1] = inertias[0, 0].cpu()
    torch.testing.assert_close(
        wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_INERTIA)), expected_raw_inertia
    )
    torch.testing.assert_close(rigid_object.data.body_mass.torch[:, 0], torch.tensor([1.0, 3.0], device=device))
    torch.testing.assert_close(rigid_object.data.body_com_pose_b.torch[1:2], coms)
    torch.testing.assert_close(rigid_object.data.body_inertia.torch[1:2], inertias)

    initial_velocity = rigid_object.data.root_com_vel_w.torch.clone()
    forces = torch.zeros((_NUM_CUBES, 1, 3), device=device)
    forces[1, 0, 0] = 20.0
    rigid_object.instantaneous_wrench_composer.set_forces_and_torques_index(forces=forces, is_global=True)
    rigid_object.write_data_to_sim()
    sim.step()
    rigid_object.update(sim.cfg.dt)

    assert rigid_object.data.root_com_vel_w.torch[1, 0] > initial_velocity[1, 0]
    torch.testing.assert_close(rigid_object.data.root_com_vel_w.torch[0], initial_velocity[0], atol=1e-6, rtol=0.0)

    # Reset clears both wrench composers.
    ones = torch.ones((_NUM_CUBES, 1, 3), device=device)
    rigid_object.permanent_wrench_composer.set_forces_and_torques_index(forces=ones, torques=ones)
    rigid_object.instantaneous_wrench_composer.add_forces_and_torques_index(forces=ones, torques=ones)
    rigid_object.reset()
    for composer in (rigid_object.instantaneous_wrench_composer, rigid_object.permanent_wrench_composer):
        assert not composer.active
        assert torch.count_nonzero(composer.composed_force.torch) == 0
        assert torch.count_nonzero(composer.composed_torque.torch) == 0


@pytest.mark.parametrize("scene", test_devices(), indirect=True)
def test_initialization_with_kinematic_enabled(scene: _RigidScene) -> None:
    """Test that kinematic bodies publish transforms and hold their pose under gravity."""
    cube_object, sim = scene.kinematic, scene.sim

    # SDP bindings must be ready on CPU and GPU before any asset pose read.
    provider = sim.get_scene_data_provider()
    names = ("Dynamic", "Kinematic", "Spinning")
    expected_paths = {f"/World/{name}/Env_{i}/Cube" for name in names for i in (0, 1)}
    expected_paths |= {
        f"/World/{name}/Env_{i}/Object_{body}"
        for name in ("Fused", "Stepped", "Wrenched")
        for i in (0, 1)
        for body in (0, 1, 2)
    }
    assert provider.transform_count == len(expected_paths)
    assert set(provider.backend.transform_paths) == expected_paths

    assert cube_object.is_initialized
    assert len(cube_object.body_names) == 1
    default_root_pose = cube_object.data.default_root_pose.torch.clone()
    default_root_pose[:, :3] += torch.tensor([[0.0, 2.0, 0.0], [2.0, 2.0, 0.0]], device=scene.device)
    default_root_vel = cube_object.data.default_root_vel.torch.clone()
    for _ in range(2):
        sim.step()
        cube_object.update(sim.cfg.dt)
        torch.testing.assert_close(cube_object.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(cube_object.data.root_com_vel_w.torch, default_root_vel)


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_body_root_state_properties(scene: _RigidScene) -> None:
    """Test the root and body link and COM states of cubes spinning about an offset center of mass."""
    cube_object, sim, device = scene.spinning, scene.sim, scene.device
    env_pos = cube_object.data.root_link_pos_w.torch.clone()

    # Offset the center of mass along the link x-axis.
    offset = torch.tensor([0.1, 0.0, 0.0], device=device).repeat(_NUM_CUBES, 1)
    com = cube_object.data.body_com_pose_b.torch.clone()  # shape (N, 1, 7)
    com[..., :3] = offset.unsqueeze(1)
    cube_object.set_coms_index(coms=wp.from_torch(com, dtype=wp.transformf))
    torch.testing.assert_close(cube_object.data.body_com_pose_b.torch, com)

    spin_twist = torch.zeros(6, device=device)
    spin_twist[5] = 2.0

    for _ in range(10):
        # Keep spinning about the z-axis through the center of mass.
        cube_object.write_root_velocity_to_sim_index(root_velocity=spin_twist.repeat(_NUM_CUBES, 1))
        sim.step()
        cube_object.update(sim.cfg.dt)

        root_link_pose_w = cube_object.data.root_link_pose_w.torch
        root_link_vel_w = cube_object.data.root_link_vel_w.torch
        root_com_pose_w = cube_object.data.root_com_pose_w.torch
        root_com_vel_w = cube_object.data.root_com_vel_w.torch
        body_link_pose_w = cube_object.data.body_link_pose_w.torch
        body_link_vel_w = cube_object.data.body_link_vel_w.torch
        body_com_pose_w = cube_object.data.body_com_pose_w.torch
        body_com_vel_w = cube_object.data.body_com_vel_w.torch

        # The center of mass stays fixed while the cube spins about it.
        torch.testing.assert_close(env_pos + offset, root_com_pose_w[..., :3])
        torch.testing.assert_close(env_pos + offset, body_com_pose_w[..., :3].squeeze(-2))
        # The link origin stays at the negated COM offset in the link frame.
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

        # The COM orientation is a constant rotation of the link orientation.
        com_quat_b = cube_object.data.body_com_quat_b.torch
        com_quat_w = quat_mul(body_link_pose_w[..., 3:], com_quat_b)
        torch.testing.assert_close(com_quat_w, body_com_pose_w[..., 3:])
        torch.testing.assert_close(com_quat_w.squeeze(-2), root_com_pose_w[..., 3:])

        # The center of mass does not translate.
        torch.testing.assert_close(torch.zeros_like(root_com_vel_w[..., :3]), root_com_vel_w[..., :3])
        torch.testing.assert_close(torch.zeros_like(body_com_vel_w[..., :3]), body_com_vel_w[..., :3])
        # The link velocity adds the rotation of the offset to the COM velocity.
        lin_vel_rel_root_gt = quat_apply_inverse(root_link_pose_w[..., 3:], root_link_vel_w[..., :3])
        lin_vel_rel_body_gt = quat_apply_inverse(body_link_pose_w[..., 3:], body_link_vel_w[..., :3])
        com_lin_vel_rel_gt = quat_apply_inverse(root_link_pose_w[..., 3:], root_com_vel_w[..., :3])
        com_ang_vel_rel_gt = quat_apply_inverse(root_link_pose_w[..., 3:], root_com_vel_w[..., 3:])
        lin_vel_rel_gt = com_lin_vel_rel_gt + torch.linalg.cross(com_ang_vel_rel_gt, -offset)
        torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_root_gt, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(lin_vel_rel_gt, lin_vel_rel_body_gt.squeeze(-2), atol=1e-4, rtol=1e-4)

        # Link and COM frames share the angular velocity.
        torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
        torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])


@pytest.mark.parametrize("scene", test_devices(), indirect=True)
def test_rigid_object_collection_real_ovphysx_seams(scene: _RigidScene) -> None:
    """Prove fused remapping through partial state, inertial, and material writes."""
    collection, device = scene.fused, scene.device
    assert collection.is_initialized
    assert collection.num_instances == _NUM_ENVS
    assert collection.body_names == ["left", "middle", "right"]
    assert collection.data.body_mass.torch.shape == (_NUM_ENVS, _NUM_BODIES)

    env_ids = torch.tensor([1, 0], dtype=torch.int32, device=device)
    body_ids = torch.tensor([2, 0], dtype=torch.int32, device=device)
    initial_pose = collection.data.body_link_pose_w.torch.clone()
    target_pose = initial_pose[env_ids][:, body_ids].clone()
    target_pose[0, 0, :3] += torch.tensor([0.2, 0.3, 0.4], device=device)
    target_pose[1, 1, :3] += torch.tensor([-0.1, -0.2, 0.1], device=device)
    collection.write_body_link_pose_to_sim_index(body_poses=target_pose, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(collection.data.body_link_pose_w.torch[env_ids][:, body_ids], target_pose)
    torch.testing.assert_close(collection.data.body_link_pose_w.torch[:, 1], initial_pose[:, 1])

    # The fused OVPhysX bindings are body-major; the public data is environment-major.
    cpu_env_ids, cpu_body_ids = env_ids.cpu(), body_ids.cpu()
    raw_pose = wp.to_torch(collection.root_view.get_attribute(TT.RIGID_BODY_POSE))
    expected_pose = initial_pose.clone()
    expected_pose[env_ids[:, None].long(), body_ids[None, :].long()] = target_pose
    torch.testing.assert_close(raw_pose.reshape(_NUM_BODIES, _NUM_ENVS, 7).transpose(0, 1).to(device), expected_pose)
    initial_mass = collection.data.body_mass.torch.clone()
    raw_mass_before = wp.to_torch(collection.root_view.get_attribute(TT.BODY_MASS)).reshape(3, 2).T.clone()
    masses = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    collection.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
    expected_raw_mass = raw_mass_before.clone()
    expected_raw_mass[cpu_env_ids[:, None], cpu_body_ids[None, :]] = masses.cpu()
    raw_mass = wp.to_torch(collection.root_view.get_attribute(TT.BODY_MASS)).reshape(3, 2).T
    torch.testing.assert_close(raw_mass, expected_raw_mass)
    torch.testing.assert_close(collection.data.body_mass.torch[env_ids][:, body_ids], masses)
    torch.testing.assert_close(collection.data.body_mass.torch[:, 1], initial_mass[:, 1])

    raw_com_before = (
        wp.to_torch(collection.root_view.get_attribute(TT.BODY_COM_POSE)).reshape(3, 2, 7).transpose(0, 1).clone()
    )
    coms = collection.data.body_com_pose_b.torch[env_ids][:, body_ids].clone()
    coms[..., :3] = torch.tensor(
        [[[0.01, 0.02, 0.03], [-0.01, 0.03, 0.02]], [[0.02, -0.01, 0.01], [0.03, 0.01, -0.02]]], device=device
    )
    collection.set_coms_index(coms=coms, env_ids=env_ids, body_ids=body_ids)
    expected_raw_com = raw_com_before.clone()
    expected_raw_com[cpu_env_ids[:, None], cpu_body_ids[None, :]] = coms.cpu()
    raw_com = wp.to_torch(collection.root_view.get_attribute(TT.BODY_COM_POSE)).reshape(3, 2, 7).transpose(0, 1)
    torch.testing.assert_close(raw_com, expected_raw_com)
    torch.testing.assert_close(collection.data.body_com_pose_b.torch[env_ids][:, body_ids], coms)

    raw_inertia_before = (
        wp.to_torch(collection.root_view.get_attribute(TT.BODY_INERTIA)).reshape(3, 2, 9).transpose(0, 1).clone()
    )
    inertias = collection.data.body_inertia.torch[env_ids][:, body_ids].clone()
    inertias[..., 0] *= 1.2
    inertias[..., 4] *= 1.3
    collection.set_inertias_index(inertias=inertias, env_ids=env_ids, body_ids=body_ids)
    expected_raw_inertia = raw_inertia_before.clone()
    expected_raw_inertia[cpu_env_ids[:, None], cpu_body_ids[None, :]] = inertias.cpu()
    raw_inertia = wp.to_torch(collection.root_view.get_attribute(TT.BODY_INERTIA)).reshape(3, 2, 9).transpose(0, 1)
    torch.testing.assert_close(raw_inertia, expected_raw_inertia)
    torch.testing.assert_close(collection.data.body_inertia.torch[env_ids][:, body_ids], inertias)

    materials = torch.tensor(
        [
            [[0.9, 0.4, 0.1], [0.8, 0.3, 0.2], [0.7, 0.2, 0.3]],
            [[0.6, 0.1, 0.4], [0.5, 0.2, 0.1], [0.4, 0.3, 0.2]],
        ]
    )
    fused_materials = collection.reshape_data_to_view_3d(wp.from_torch(materials, dtype=wp.float32), 3, device="cpu")
    collection.root_view.set_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION, fused_materials)
    raw_materials = wp.to_torch(collection.root_view.get_attribute(TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION))
    torch.testing.assert_close(raw_materials.reshape(3, 2, 3).transpose(0, 1), materials)


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_wrench_reaches_only_the_selected_collection_body(scene: _RigidScene) -> None:
    """Deliver an external force through the fused body-major binding to the selected body only, then reset it.

    Environment 1, body 1 has different flat indices in body-major and environment-major layouts.
    """
    collection, sim, device = scene.wrenched, scene.sim, scene.device
    velocity_before = collection.data.body_com_vel_w.torch.clone()
    collection.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([[[20.0, 0.0, 0.0]]], device=device),
        torques=torch.zeros((1, 1, 3), device=device),
        env_ids=torch.tensor([1], dtype=torch.int32, device=device),
        body_ids=torch.tensor([1], dtype=torch.int32, device=device),
    )
    collection.write_data_to_sim()
    sim.step()
    collection.update(sim.cfg.dt)

    velocity = collection.data.body_com_vel_w.torch
    assert velocity[1, 1, 0] > velocity_before[1, 1, 0] + 1e-3
    unselected = torch.ones((_NUM_ENVS, _NUM_BODIES), dtype=torch.bool, device=device)
    unselected[1, 1] = False
    torch.testing.assert_close(velocity[unselected], velocity_before[unselected], atol=1e-6, rtol=0.0)

    # Reset clears both wrench composers.
    ones = torch.ones((_NUM_ENVS, _NUM_BODIES, 3), device=device)
    collection.instantaneous_wrench_composer.add_forces_and_torques_index(forces=ones, torques=ones)
    collection.reset()
    for composer in (collection.instantaneous_wrench_composer, collection.permanent_wrench_composer):
        assert not composer.active
        assert torch.count_nonzero(composer.composed_force.torch) == 0
        assert torch.count_nonzero(composer.composed_torque.torch) == 0


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "OVPhysX RigidObjectCollection partial body pose writes push whole environment rows from a pose buffer "
        "that is not refreshed first, so unselected bodies of a selected environment are rewound to stale poses."
    ),
)
def test_partial_body_pose_write_preserves_unselected_bodies_after_steps(scene: _RigidScene) -> None:
    """A partial (environment, body) pose write must not rewind the unselected bodies of that environment."""
    collection, sim, device = scene.stepped, scene.sim, scene.device

    def read_backend_poses() -> torch.Tensor:
        raw = wp.to_torch(collection.root_view.get_attribute(TT.RIGID_BODY_POSE))
        return raw.reshape(_NUM_BODIES, _NUM_ENVS, 7).transpose(0, 1).to(device)

    # Read the public poses once, then let a force move every body for a few steps.
    collection.data.body_link_pose_w
    collection.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([10.0, 0.0, 0.0], device=device).repeat(_NUM_ENVS, _NUM_BODIES, 1),
        torques=torch.zeros((_NUM_ENVS, _NUM_BODIES, 3), device=device),
    )
    for _ in range(5):
        collection.write_data_to_sim()
        sim.step()
        collection.update(sim.cfg.dt)
    collection.reset()

    expected = read_backend_poses().clone()
    body_pose = expected[1:, 1:2].clone()
    body_pose[..., 2] += 0.5
    expected[1, 1] = body_pose[0, 0]
    collection.write_body_link_pose_to_sim_index(
        body_poses=body_pose,
        env_ids=torch.tensor([1], dtype=torch.int32, device=device),
        body_ids=torch.tensor([1], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(read_backend_poses(), expected)
