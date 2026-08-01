# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest

pytest.importorskip("pxr")
pytest.importorskip("omni.physics.tensors")


def test_scene_data_rigid_body_view_skips_joint_prims_with_rigid_body_api(monkeypatch):
    """Joint prims must not be passed to PhysX tensor rigid-body views."""
    from isaaclab_physx.physics import physx_manager
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    body_prim = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/robot0_forearm").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    joint_prim = UsdPhysics.FixedJoint.Define(stage, "/World/envs/env_0/Robot/joints/robot0_forearm").GetPrim()
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
    backend.simulation_view = _SimulationView()
    backend.get_rigid_body_view()

    assert captured_paths == ["/World/envs/env_*/Robot/robot0_forearm"]


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
    backend.simulation_view = _SimulationView()
    backend._discover_deformable_geometry()

    assert backend.geometry_paths == [
        "/World/envs/env_0/Deformable",
        "/World/envs/env_1/Deformable",
    ]
    assert backend.geometry_counts == [4, 4]
