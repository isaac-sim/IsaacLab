# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("pxr")
pytest.importorskip("omni.physics.tensors")


def test_physics_and_scene_data_share_one_native_view(monkeypatch):
    """Warmup creates one shared native view, and stop releases it before a fresh warmup."""
    from isaaclab_physx.physics import PhysxCfg, physx_manager

    from isaaclab.physics import PhysicsManager
    from isaaclab.sim import SimulationContext
    from isaaclab.sim.utils import stage

    manager = physx_manager.PhysxManager
    sim = object.__new__(SimulationContext)
    sim._backend_registry = {}
    sim.cfg = SimpleNamespace(dt=0.01, physics=PhysxCfg())
    scene_data = physx_manager.PhysxSceneDataBackend()
    publication = scene_data.transforms
    views = [Mock(), Mock()]
    create_view = Mock(side_effect=views)
    monkeypatch.setattr(PhysicsManager, "_sim", sim)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    monkeypatch.setattr(manager, "_backend", None)
    monkeypatch.setattr(manager, "_scene_data_backend", scene_data)
    monkeypatch.setattr(manager, "_warmup_needed", True)
    monkeypatch.setattr(manager, "_view_created", False)
    monkeypatch.setattr(manager, "_event_bus", Mock())
    monkeypatch.setattr(manager, "dispatch_event", Mock())
    monkeypatch.setattr(manager, "views", {})
    monkeypatch.setattr(stage, "get_current_stage_id", lambda: 1)
    monkeypatch.setattr(physx_manager.omni.physx, "get_physx_interface", Mock(return_value=Mock()))
    monkeypatch.setattr(physx_manager.omni.physx, "get_physx_simulation_interface", Mock(return_value=Mock()))
    monkeypatch.setattr(physx_manager.omni.physics.tensors, "create_simulation_view", create_view)

    assert manager.get_physics_sim_view() is scene_data.get_rigid_body_view() is None
    manager._warmup_and_create_views()
    manager._warmup_and_create_views()
    create_view.assert_called_once_with("warp", stage_id=1)
    resource = sim.get_or_create_backend(physx_manager.PhysxBackend, 1, cfg=sim.cfg.physics.copy())
    assert manager.get_physics_sim_view() is resource.simulation_view is views[0]
    assert scene_data._backend is resource

    scene_data._rigid_body_view = object()
    manager._on_stop(None)
    manager._on_stop(None)
    resource.clear()
    views[0].invalidate.assert_called_once_with()
    assert manager.get_physics_sim_view() is scene_data.get_rigid_body_view() is None
    assert scene_data.transforms is publication
    assert publication.transforms is None
    assert not sim._backend_registry

    manager._warmup_and_create_views()
    assert create_view.call_count == 2
    assert manager.get_physics_sim_view() is views[1]
    assert scene_data._backend is not resource
    manager._on_stop(None)
    views[1].invalidate.assert_called_once_with()


@pytest.mark.parametrize("joint_has_rigid_body_api", [False, True])
def test_rigid_body_view_uses_exact_path_for_joint_name_collision(monkeypatch, joint_has_rigid_body_api):
    """Joint names must keep same-named rigid bodies out of wildcard views."""
    from isaaclab_physx.physics import physx_manager
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    body_prim = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/robot0_forearm").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    unique_body_prim = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/torso").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(unique_body_prim)
    joint_prim = UsdPhysics.FixedJoint.Define(stage, "/World/envs/env_0/Robot/joints/robot0_forearm").GetPrim()
    if joint_has_rigid_body_api:
        UsdPhysics.RigidBodyAPI.Apply(joint_prim)

    captured_paths = []

    class _SimulationView:
        def create_rigid_body_view(self, body_paths):
            captured_paths.extend(body_paths)
            return SimpleNamespace(prim_paths=body_paths)

    monkeypatch.setattr(
        physx_manager.omni.usd,
        "get_context",
        lambda: SimpleNamespace(get_stage=lambda: stage),
    )

    backend = PhysxSceneDataBackend()
    backend._backend = SimpleNamespace(simulation_view=_SimulationView())
    backend.get_rigid_body_view()

    assert captured_paths == [
        "/World/envs/env_*/Robot/torso",
        "/World/envs/env_0/Robot/robot0_forearm",
    ]


def test_discover_deformable_geometry_publishes_discovered_roots(monkeypatch):
    """PhysX deformable views may report child meshes; geometry_paths must be roots."""
    from isaaclab_physx.physics import physx_manager
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    from isaaclab.scene_data.deformable_discovery import DeformableStageEntry

    class _FakeDeformableView:
        _backend = object()
        max_simulation_nodes_per_body = 8
        prim_paths = [
            "/World/envs/env_0/Deformable/sim_mesh",
            "/World/envs/env_1/Deformable/sim_mesh",
        ]

    class _SimulationView:
        def create_volume_deformable_body_view(self, patterns):
            return None

        def create_surface_deformable_body_view(self, patterns):
            return _FakeDeformableView()

    monkeypatch.setattr(
        physx_manager.omni.usd,
        "get_context",
        lambda: SimpleNamespace(get_stage=lambda: object()),
    )
    monkeypatch.setattr(
        physx_manager,
        "discover_deformables_on_stage",
        lambda stage: [
            DeformableStageEntry(
                root_path="/World/envs/env_0/Deformable",
                sim_mesh_path="/World/envs/env_0/Deformable/sim_mesh",
                vis_mesh_path="/World/envs/env_0/Deformable/vis_mesh",
                deformable_type="surface",
                vertex_count=4,
                vis_vertex_count=4,
            ),
            DeformableStageEntry(
                root_path="/World/envs/env_1/Deformable",
                sim_mesh_path="/World/envs/env_1/Deformable/sim_mesh",
                vis_mesh_path="/World/envs/env_1/Deformable/vis_mesh",
                deformable_type="surface",
                vertex_count=4,
                vis_vertex_count=4,
            ),
        ],
    )

    backend = PhysxSceneDataBackend()
    assert backend.geometry_paths == []
    backend._backend = SimpleNamespace(simulation_view=_SimulationView())
    backend._discover_deformable_geometry()

    assert backend.geometry_paths == [
        "/World/envs/env_0/Deformable",
        "/World/envs/env_1/Deformable",
    ]
    assert backend.geometry_counts == [4, 4]
