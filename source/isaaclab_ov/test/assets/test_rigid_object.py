# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Real OVPhysX rigid-object coverage on one module-scoped scene per device.

Each device builds one scene of locally authored cuboid pairs, resets it once, and keeps it alive for
every test that uses it. Each pair belongs to one test, so those tests do not depend on each other's
order.

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
from isaaclab_ov.assets import RigidObject  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab import cloner  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context  # noqa: E402
from isaaclab.test.utils import DeviceScope, test_devices  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, quat_mul  # noqa: E402

pytestmark = pytest.mark.integration

_NUM_CUBES = 2


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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(
    ("support_cloning", "filter_collisions"),
    [("native", False), ("usd_and_native", True), ("usd", False), ("usd_nested", True)],
)
def test_heterogeneous_clone_contacts(device, support_cloning, filter_collisions):
    """Sources and clones from different variants contact their own support.

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
        for _ in range(240):
            sim.step()
            scene.update(sim.get_physics_dt())
        positions = scene["object"].data.root_pos_w.torch
        torch.testing.assert_close(positions[:, 2], torch.full_like(positions[:, 2], 0.2), atol=0.02, rtol=0.0)
        torch.testing.assert_close(positions[:, :2], scene.env_origins[:, :2], atol=0.02, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_heterogeneous_clone_collision_isolation(device):
    """Collision groups isolate overlapping environments, including both retained sources."""
    with build_simulation_context(
        device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device, dt=1.0 / 120.0)
    ) as sim:
        # Without GPU environment-ID filtering, 2048 overlapping worlds exhaust broadphase pairs.
        num_envs = 2048 if device.startswith("cuda") else 6
        scene = InteractiveScene(HeterogeneousRigidSceneCfg(num_envs=num_envs, env_spacing=0.0))
        sim.reset()
        for _ in range(240):
            sim.step()
            scene.update(sim.get_physics_dt())
        positions = scene["object"].data.root_pos_w.torch
        expected = scene.env_origins.clone()
        expected[:, 2] = 0.2
        torch.testing.assert_close(positions, expected, atol=0.02, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_heterogeneous_clone_indexed_state(device):
    """Indexed reads and writes use environment order across six interleaved variants."""
    with build_simulation_context(
        device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device, gravity=(0.0, 0.0, 0.0))
    ) as sim:
        scene = InteractiveScene(HeterogeneousRigidSceneCfg(num_envs=6, env_spacing=2.0))
        sim.reset()
        obj = scene["object"]
        expected_paths = [f"/World/envs/env_{i}/Object" for i in range(scene.num_envs)]
        assert obj.root_view.prim_paths == expected_paths
        torch.testing.assert_close(obj.data.root_pos_w.torch[:, :2], scene.env_origins[:, :2])
        selected = torch.tensor([3], device=device)
        pose = obj.data.root_pose_w.torch[selected].clone()
        pose[:, 2] = 3.0
        obj.write_root_pose_to_sim(pose, env_ids=selected)
        sim.step()
        scene.update(sim.get_physics_dt())
        # An independent exact-path binding detects a write to the wrong physical object.
        binding = OvPhysxManager.get_physx_instance().create_tensor_binding(
            prim_paths=expected_paths, tensor_type=TT.RIGID_BODY_POSE
        )
        try:
            actual = torch.empty(binding.shape, device=device)
            binding.read(actual)
            expected_height = torch.ones(scene.num_envs, device=device)
            expected_height[selected] = 3.0
            torch.testing.assert_close(actual[:, 2], expected_height)
            torch.testing.assert_close(obj.data.root_pos_w.torch, actual[:, :3])
        finally:
            binding.destroy()


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


@dataclass
class _RigidScene:
    """Cuboid pairs that share one real OVPhysX lifecycle."""

    sim: SimulationContext
    device: str
    dynamic: RigidObject
    kinematic: RigidObject
    spinning: RigidObject


@pytest.fixture(scope="module")
def scene(request: pytest.FixtureRequest) -> Iterator[_RigidScene]:
    """Initialize every cuboid pair for one device once for this module."""
    device = request.param
    with build_simulation_context(device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device)) as sim:
        dynamic = _spawn_cubes("Dynamic", y_offset=0.0, disable_gravity=True)
        kinematic = _spawn_cubes("Kinematic", y_offset=2.0, kinematic_enabled=True)
        spinning = _spawn_cubes("Spinning", y_offset=4.0, disable_gravity=True)
        sim.reset()
        yield _RigidScene(sim=sim, device=device, dynamic=dynamic, kinematic=kinematic, spinning=spinning)


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


@pytest.mark.parametrize("scene", test_devices(DeviceScope.CUDA), indirect=True)
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
