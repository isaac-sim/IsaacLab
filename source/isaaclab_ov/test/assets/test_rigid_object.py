# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Real OVPhysX rigid-object coverage on one module-scoped scene per device.

Each device builds one scene of locally authored cuboid pairs, resets it once, and keeps it alive for
every test that uses it. Each pair belongs to one test, so the tests do not depend on each other's
order. Initialization failures need their own scenes and therefore build them before any shared scene.
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
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context  # noqa: E402
from isaaclab.test.utils import DeviceScope, test_devices  # noqa: E402

pytestmark = pytest.mark.integration

_NUM_CUBES = 2


def _sim_context(device: str):
    """Build a local OVPhysX context from an in-memory USD stage."""
    return build_simulation_context(device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device))


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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_initialization_with_no_rigid_body(device):
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with _sim_context(device) as sim:
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
def test_initialization_with_articulation_root(device):
    """Test that initialization fails when an articulation root is found at the provided prim path."""
    with _sim_context(device) as sim:
        # Generate cubes and mark each rigid body as an articulation root. Keep the asset alive for reset.
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
    resettable: RigidObject


@pytest.fixture(scope="module")
def scene(request: pytest.FixtureRequest) -> Iterator[_RigidScene]:
    """Initialize every cuboid pair for one device once for this module."""
    device = request.param
    with _sim_context(device) as sim:
        dynamic = _spawn_cubes("Dynamic", y_offset=0.0, disable_gravity=True)
        kinematic = _spawn_cubes("Kinematic", y_offset=2.0, kinematic_enabled=True)
        resettable = _spawn_cubes("Resettable", y_offset=4.0, disable_gravity=True)
        sim.reset()
        yield _RigidScene(sim=sim, device=device, dynamic=dynamic, kinematic=kinematic, resettable=resettable)


@pytest.mark.parametrize("scene", test_devices(), indirect=True)
def test_rigid_object_real_ovphysx_seams(scene: _RigidScene) -> None:
    """Prove partial state, inertial properties, and one real wrench delivery."""
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


@pytest.mark.parametrize("scene", test_devices(), indirect=True)
def test_initialization_with_kinematic_enabled(scene: _RigidScene) -> None:
    """Test that kinematic bodies publish transforms and hold their pose under gravity."""
    cube_object, sim = scene.kinematic, scene.sim

    # SDP bindings must be ready on CPU and GPU before any asset pose read.
    provider = sim.get_scene_data_provider()
    expected_paths = {f"/World/{name}/Env_{i}/Cube" for name in ("Dynamic", "Kinematic", "Resettable") for i in (0, 1)}
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
def test_reset_clears_active_wrench_composers(scene: _RigidScene) -> None:
    """Test that resetting the rigid object clears both active wrench composers."""
    cube_object, device = scene.resettable, scene.device

    # Make both wrench composers active so the reset has something to clear.
    cube_object.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.ones((_NUM_CUBES, 1, 3), device=device),
        torques=torch.ones((_NUM_CUBES, 1, 3), device=device),
    )
    cube_object.instantaneous_wrench_composer.add_forces_and_torques_index(
        forces=torch.ones((_NUM_CUBES, 1, 3), device=device),
        torques=torch.ones((_NUM_CUBES, 1, 3), device=device),
    )
    assert cube_object._instantaneous_wrench_composer.active
    assert cube_object._permanent_wrench_composer.active

    cube_object.reset()

    # Reset should zero external forces and torques
    assert not cube_object._instantaneous_wrench_composer.active
    assert not cube_object._permanent_wrench_composer.active
    assert torch.count_nonzero(cube_object._instantaneous_wrench_composer.composed_force.torch) == 0
    assert torch.count_nonzero(cube_object._instantaneous_wrench_composer.composed_torque.torch) == 0
    assert torch.count_nonzero(cube_object._permanent_wrench_composer.composed_force.torch) == 0
    assert torch.count_nonzero(cube_object._permanent_wrench_composer.composed_torque.torch) == 0
