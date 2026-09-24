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
    monkeypatch.setattr(backend, "get_rigid_body_view", Mock(wraps=backend.get_rigid_body_view))
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
    fabric_matrices = wp.zeros(1, dtype=wp.mat44d, device="cpu")
    backend._fabric_selection = SimpleNamespace(
        PrepareForReuse=Mock(return_value=False),
        __fabric_arrays_interface__={
            "version": 1,
            "device": "cpu",
            "attribs": {
                "omni:fabric:worldMatrix": {
                    "type": (True, "f8", 16, 0, "matrix"),
                    "access": 1,
                    "pointers": [fabric_matrices.ptr],
                    "counts": [1],
                }
            },
        },
    )
    assert provider.get_transforms(SceneDataFormat.FabricMatrix44())
    version = backend.transforms_version
    assert provider.get_transforms(SceneDataFormat.FabricMatrix44())
    fabric.force_update.assert_called_once_with(0.0, 0.0)
    backend.get_rigid_body_view.assert_not_called()
    view.get_transforms.assert_not_called()
    assert backend.transforms_version == version
    native = SceneDataFormat.Transform()
    assert provider.get_transforms(native)
    assert native.transforms.ptr == transforms.ptr
    output = SceneDataFormat.Matrix44()
    assert provider.get_transforms(output)
    matrices = output.matrices
    view.get_transforms.assert_called_once_with()

    transforms.fill_(wp.transformf(wp.vec3f(1, 2, 3), wp.quat_identity()))
    getattr(manager, operation)()
    assert backend.transforms_version > version
    version = backend.transforms_version
    manager.pre_render()
    manager.pre_render()
    assert sim_view.update_articulations_kinematic.call_count == int(operation == "forward")
    assert provider.get_transforms(output)
    assert output.matrices is matrices
    np.testing.assert_array_equal(matrices.numpy()[0, :3, 3], [1, 2, 3])
    assert view.get_transforms.call_count == 2
    provider.get_transforms(SceneDataFormat.FabricMatrix44())
    provider.get_transforms(SceneDataFormat.FabricMatrix44())
    assert fabric.force_update.call_count == 2

    transforms.fill_(wp.transformf(wp.vec3f(4, 5, 6), wp.quat_identity()))
    manager.invalidate_transforms(kinematics=True)
    assert backend.transforms_version > version
    version = backend.transforms_version
    provider.get_transforms(SceneDataFormat.FabricMatrix44())
    provider.get_transforms(SceneDataFormat.FabricMatrix44())
    assert sim_view.update_articulations_kinematic.call_count == 1 + int(operation == "forward")
    assert fabric.force_update.call_count == 3
    assert backend.transforms_version == version
    provider.get_transforms(native)
    assert provider.get_transforms(output)
    np.testing.assert_array_equal(output.matrices.numpy()[0, :3, 3], [4, 5, 6])
    assert view.get_transforms.call_count == 3
    assert backend.transforms_version == version

    points = wp.zeros(2, dtype=wp.vec3f, device="cpu")
    pointers = wp.array([points.ptr], dtype=wp.uint64, device="cpu")
    lengths = wp.array([len(points)], dtype=wp.uint64, device="cpu")
    backend._fabric_points_selection = SimpleNamespace(
        PrepareForReuse=Mock(return_value=False),
        __fabric_arrays_interface__={
            "version": 1,
            "device": "cpu",
            "attribs": {
                "points": {
                    "type": (True, "f4", 3, 1, "vector"),
                    "access": 1,
                    "pointers": [pointers.ptr],
                    "counts": [1],
                    "array_lengths": [lengths.ptr],
                }
            },
        },
    )
    geometry = provider.get_geometry_points(output_format=SceneDataFormat.FabricPoints)
    assert geometry._cls is SceneDataFormat.FabricPoints and geometry.points is not None
    backend.geometry_version += 1
    assert provider.get_geometry_points(output_format=SceneDataFormat.FabricPoints) is geometry
    assert provider.get_transforms(SceneDataFormat.FabricMatrix44())
    assert fabric.force_update.call_count == 4
    assert view.get_transforms.call_count == 3
    backend.clear()
    assert backend.transforms_version > version


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


@pytest.mark.parametrize("capacity", [3, 8])
def test_deformable_geometry_uses_declared_counts_and_native_order(monkeypatch, capacity):
    """Declared unpadded counts survive native reordering and mesh paths."""
    from isaaclab_physx.physics import physx_manager

    from isaaclab.physics import PhysicsManager
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
    from isaaclab.scene_data.deformable_discovery import DeformableStageEntry

    values = {"/Clones/slot_2/Asset": 2.0, "/Clones/slot_9/Asset": 9.0, "/Shared": 100.0}
    counts = {"/Clones/slot_2/Asset": 4, "/Clones/slot_9/Asset": 4, "/Shared": 2}
    entries = [
        DeformableStageEntry(
            path, path + "/sim", path + "/vis", "volume" if path == "/Shared" else "surface", count, count
        )
        for path, count in counts.items()
    ]
    bound_paths = []
    native_points, reads = {}, []

    def create_view(paths):
        bound_paths.extend(paths)
        paths = list(reversed(paths))
        nodal = np.full((len(paths), capacity, 3), -999.0, dtype=np.float32)
        for index, path in enumerate(paths):
            nodal[index, : counts[path]] = values[path]
        points = wp.array(nodal, dtype=wp.float32, device="cpu")
        for path in paths:
            native_points[path + "/vis"] = points

        def read():
            reads.append(points.ptr)
            return points

        return SimpleNamespace(
            count=len(paths),
            prim_paths=[path + "/sim" for path in paths],
            max_simulation_nodes_per_body=capacity,
            get_simulation_nodal_positions=read,
        )

    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    monkeypatch.setattr(
        physx_manager.omni.usd, "get_context", lambda: pytest.fail("Geometry bindings must not fetch a stage.")
    )
    backend = physx_manager.PhysxSceneDataBackend()
    backend.backend = SimpleNamespace(
        simulation_view=SimpleNamespace(
            create_volume_deformable_body_view=create_view, create_surface_deformable_body_view=create_view
        )
    )
    if capacity < 4:
        with pytest.raises(RuntimeError, match="node capacity"):
            backend._setup_deformable_geometry(entries)
        return
    backend._setup_deformable_geometry(entries)

    assert set(bound_paths) == counts.keys()
    batches = backend.get_geometry_batches()
    assert [ranges for _, ranges in batches] == [
        {"/Shared/vis": (0, 2)},
        {"/Clones/slot_9/Asset/vis": (0, 4), "/Clones/slot_2/Asset/vis": (capacity, 4)},
    ]
    provider = SceneDataProvider(backend)
    visual = provider.get_geometry_points()
    for publication, ranges in batches:
        assert publication._cls is SceneDataFormat.Points
        for path, (offset, count) in ranges.items():
            assert publication.points.ptr == native_points[path].ptr
            assert visual[path].ptr == publication.points.ptr + offset * wp.types.type_size_in_bytes(wp.vec3f)
            np.testing.assert_array_equal(visual[path].numpy(), np.full((count, 3), values[path[:-4]]))
    read_count = len(reads)
    assert provider.get_geometry_points() is visual
    assert len(reads) == read_count
    backend.geometry_version += 1
    backend.get_geometry_batches()
    assert len(reads) == read_count + len(batches)
    backend.clear()
    assert backend.get_geometry_batches() == []
