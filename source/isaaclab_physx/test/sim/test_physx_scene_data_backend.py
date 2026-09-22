# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import warp as wp

pytest.importorskip("pxr")
pytest.importorskip("omni.physics.tensors")


@pytest.mark.parametrize("operation", ["step", "forward"])
def test_pose_publication_refreshes_after_physics_but_reuses_clean_reads(monkeypatch, operation):
    """SDP borrows native poses once per dirty generation and completes pending joint writes."""
    from isaaclab_physx.physics import physx_manager

    from isaaclab.physics import PhysicsManager
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    manager = physx_manager.PhysxManager
    fabric = Mock()
    monkeypatch.setattr(manager, "_fabric", fabric)
    backend = physx_manager.PhysxSceneDataBackend()
    transforms = wp.zeros(1, dtype=wp.transformf, device="cpu")
    view = Mock(count=1, get_transforms=Mock(return_value=transforms))
    backend._rigid_body_view = view
    sim_view = Mock()
    monkeypatch.setattr(manager, "backend", SimpleNamespace(simulation_view=sim_view))
    monkeypatch.setattr(manager, "_scene_data_backend", backend)
    monkeypatch.setattr(manager, "_kinematics_dirty", False)
    monkeypatch.setattr(manager, "_anim_recorder", None)
    monkeypatch.setattr(
        PhysicsManager, "_sim", SimpleNamespace(stage=object(), cfg=SimpleNamespace(dt=0.01), is_playing=lambda: True)
    )
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    monkeypatch.setattr(physx_manager.omni.physx, "get_physx_simulation_interface", Mock(return_value=Mock()))
    provider = SceneDataProvider(backend)
    provider._fabric_output = SceneDataFormat.FabricMatrix44(matrices=object())
    provider._fabric_selection = Mock(PrepareForReuse=Mock(return_value=False))
    monkeypatch.setattr(PhysicsManager._sim, "get_scene_data_provider", lambda: provider, raising=False)
    assert backend.fabric is fabric
    provider._prepare_fabric(object(), "cpu")
    provider.request_transforms(SceneDataFormat.FabricMatrix44)
    provider.request_transforms(SceneDataFormat.FabricMatrix44)
    fabric.force_update.assert_called_once_with(0.0, 0.0)
    view.get_transforms.assert_not_called()
    assert backend.transforms_dirty
    assert provider.request_transforms(SceneDataFormat.Transform).transforms.ptr == transforms.ptr
    matrices = provider.request_transforms(SceneDataFormat.Matrix44)
    view.get_transforms.assert_called_once_with()

    transforms.fill_(wp.transformf(wp.vec3f(1, 2, 3), wp.quat_identity()))
    getattr(manager, operation)()
    manager.pre_render()
    manager.pre_render()
    assert sim_view.update_articulations_kinematic.call_count == int(operation == "forward")
    assert provider.request_transforms(SceneDataFormat.Matrix44) is matrices
    np.testing.assert_array_equal(matrices.matrices.numpy()[0, :3, 3], [1, 2, 3])
    assert view.get_transforms.call_count == 2
    provider.request_transforms(SceneDataFormat.FabricMatrix44)
    provider.request_transforms(SceneDataFormat.FabricMatrix44)
    assert fabric.force_update.call_count == 2

    manager.invalidate_transforms(kinematics=True)
    assert backend.transforms_dirty and backend.fabric_dirty
    provider.request_transforms(SceneDataFormat.FabricMatrix44)
    provider.request_transforms(SceneDataFormat.FabricMatrix44)
    assert sim_view.update_articulations_kinematic.call_count == 1 + int(operation == "forward")
    assert fabric.force_update.call_count == 3
    assert backend.transforms_dirty and not backend.fabric_dirty
    provider.request_transforms(SceneDataFormat.Transform)
    assert view.get_transforms.call_count == 3
    assert not backend.transforms_dirty


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
    backend.backend = SimpleNamespace(simulation_view=_SimulationView())
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
    backend.backend = SimpleNamespace(simulation_view=_SimulationView())
    backend._discover_deformable_geometry()

    assert backend.geometry_paths == [
        "/World/envs/env_0/Deformable",
        "/World/envs/env_1/Deformable",
    ]
    assert backend.geometry_counts == [4, 4]
