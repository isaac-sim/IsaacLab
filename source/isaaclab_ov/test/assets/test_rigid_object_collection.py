# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Real OVPhysX rigid-object-collection coverage on one module-scoped scene per device.

Each device builds one scene of locally authored ``N=2, B=3`` collections, resets it once, and keeps it
alive for every test that uses it. Each collection belongs to one test, so the tests do not depend on each
other's order. Initialization failures need their own scenes and therefore build them before any shared scene.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass

import pytest
import torch
import warp as wp

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import RigidObjectCollection  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context  # noqa: E402
from isaaclab.test.utils import DeviceScope, test_devices  # noqa: E402

pytestmark = pytest.mark.integration

_NUM_ENVS, _NUM_BODIES = 2, 3


def _sim_context(device: str) -> AbstractContextManager[SimulationContext]:
    """Build a local OVPhysX context from an in-memory USD stage."""
    return build_simulation_context(
        device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device, gravity=(0.0, 0.0, 0.0))
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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_initialization_with_no_rigid_body(device: str) -> None:
    """Test that initialization fails when no rigid body is found at the provided prim path."""
    with _sim_context(device) as sim:
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
class _CollectionScene:
    """Collections that share one real OVPhysX lifecycle."""

    sim: SimulationContext
    device: str
    fused: RigidObjectCollection
    resettable: RigidObjectCollection
    stepped: RigidObjectCollection
    wrenched: RigidObjectCollection


@pytest.fixture(scope="module")
def scene(request: pytest.FixtureRequest) -> Iterator[_CollectionScene]:
    """Initialize every collection for one device once for this module."""
    device = request.param
    with _sim_context(device) as sim:
        fused = _spawn_collection("Fused", y_offset=0.0)
        resettable = _spawn_collection("Resettable", y_offset=4.0)
        stepped = _spawn_collection("Stepped", y_offset=8.0)
        wrenched = _spawn_collection("Wrenched", y_offset=12.0)
        sim.reset()
        yield _CollectionScene(
            sim=sim, device=device, fused=fused, resettable=resettable, stepped=stepped, wrenched=wrenched
        )


@pytest.mark.parametrize("scene", test_devices(), indirect=True)
def test_rigid_object_collection_real_ovphysx_seams(scene: _CollectionScene) -> None:
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


@pytest.mark.parametrize("scene", test_devices(DeviceScope.CUDA), indirect=True)
def test_reset_clears_active_wrench_composers(scene: _CollectionScene) -> None:
    """Test that resetting the collection clears both active wrench composers."""
    object_collection, device = scene.resettable, scene.device

    # Make both wrench composers active so the reset has something to clear.
    object_collection.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.ones((_NUM_ENVS, _NUM_BODIES, 3), device=device),
        torques=torch.ones((_NUM_ENVS, _NUM_BODIES, 3), device=device),
    )
    object_collection.instantaneous_wrench_composer.add_forces_and_torques_index(
        forces=torch.ones((_NUM_ENVS, _NUM_BODIES, 3), device=device),
        torques=torch.ones((_NUM_ENVS, _NUM_BODIES, 3), device=device),
    )
    assert object_collection._instantaneous_wrench_composer.active
    assert object_collection._permanent_wrench_composer.active

    object_collection.reset()

    # Reset should zero external forces and torques
    assert not object_collection._instantaneous_wrench_composer.active
    assert not object_collection._permanent_wrench_composer.active
    assert torch.count_nonzero(object_collection._instantaneous_wrench_composer.composed_force.torch) == 0
    assert torch.count_nonzero(object_collection._instantaneous_wrench_composer.composed_torque.torch) == 0
    assert torch.count_nonzero(object_collection._permanent_wrench_composer.composed_force.torch) == 0
    assert torch.count_nonzero(object_collection._permanent_wrench_composer.composed_torque.torch) == 0


@pytest.mark.parametrize("scene", test_devices(DeviceScope.CUDA), indirect=True)
def test_wrench_reaches_only_the_selected_collection_body(scene: _CollectionScene) -> None:
    """Deliver an external force through the fused body-major binding to the selected body only.

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
    collection.reset()

    velocity = collection.data.body_com_vel_w.torch
    assert velocity[1, 1, 0] > velocity_before[1, 1, 0] + 1e-3
    unselected = torch.ones((_NUM_ENVS, _NUM_BODIES), dtype=torch.bool, device=device)
    unselected[1, 1] = False
    torch.testing.assert_close(velocity[unselected], velocity_before[unselected], atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("scene", test_devices(DeviceScope.CUDA), indirect=True)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "OVPhysX RigidObjectCollection partial body pose writes push whole environment rows from a pose buffer "
        "that is not refreshed first, so unselected bodies of a selected environment are rewound to stale poses."
    ),
)
def test_partial_body_pose_write_preserves_unselected_bodies_after_steps(scene: _CollectionScene) -> None:
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
