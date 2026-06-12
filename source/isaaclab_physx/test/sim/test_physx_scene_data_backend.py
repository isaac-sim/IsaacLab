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
        _backend = object()

        def create_rigid_body_view(self, body_paths):
            captured_paths.extend(body_paths)
            return SimpleNamespace(_backend=object(), prim_paths=body_paths)

    monkeypatch.setattr(
        physx_manager.omni.usd,
        "get_context",
        lambda: SimpleNamespace(get_stage=lambda: stage),
    )

    backend = PhysxSceneDataBackend()
    backend.simulation_view = _SimulationView()
    backend.get_rigid_body_view()

    assert captured_paths == ["/World/envs/env_*/Robot/robot0_forearm"]


def test_scene_data_backend_discards_invalid_cached_rigid_body_view():
    """Cached PhysX rigid-body views can be invalidated by timeline lifecycle events."""
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    backend = PhysxSceneDataBackend()
    backend._rigid_body_view = SimpleNamespace(_backend=None)

    assert backend.get_rigid_body_view() is None
    assert backend.transform_paths == []
    assert backend._rigid_body_view is None


def test_scene_data_backend_discards_invalid_simulation_view():
    """Invalid PhysX simulation views should not be reused to create rigid-body views."""
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    class _InvalidSimulationView:
        _backend = None

        def create_rigid_body_view(self, body_paths):
            raise AssertionError("invalid simulation view must not be used")

    backend = PhysxSceneDataBackend()
    backend.simulation_view = _InvalidSimulationView()

    assert backend.get_rigid_body_view() is None
    assert backend.transform_paths == []
    assert backend.simulation_view is None


def test_scene_data_backend_discards_invalid_created_rigid_body_view(monkeypatch):
    """PhysX may return an invalid rigid-body view when its simulation view is stale."""
    from isaaclab_physx.physics import physx_manager
    from isaaclab_physx.physics.physx_manager import PhysxSceneDataBackend

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    body_prim = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/torso").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body_prim)

    class _SimulationView:
        _backend = object()

        def create_rigid_body_view(self, body_paths):
            return SimpleNamespace(_backend=None)

    monkeypatch.setattr(
        physx_manager.omni.usd,
        "get_context",
        lambda: SimpleNamespace(get_stage=lambda: stage),
    )

    backend = PhysxSceneDataBackend()
    backend.simulation_view = _SimulationView()

    assert backend.transform_paths == []
    assert backend._rigid_body_view is None


def test_invalidate_views_clears_scene_data_backend_simulation_view(monkeypatch):
    """PhysxManager invalidation must also clear the scene-data backend view cache."""
    from isaaclab_physx.physics.physx_manager import PhysxManager, PhysxSceneDataBackend

    class _View:
        _backend = object()

        def __init__(self):
            self.invalidated = False

        def invalidate(self):
            self.invalidated = True

    view = _View()
    view_warp = _View()
    backend = PhysxSceneDataBackend()
    backend.simulation_view = view

    monkeypatch.setattr(PhysxManager, "_view", view)
    monkeypatch.setattr(PhysxManager, "_view_warp", view_warp)
    monkeypatch.setattr(PhysxManager, "_view_created", True)
    monkeypatch.setattr(PhysxManager, "_scene_data_backend", backend)

    PhysxManager._invalidate_views()

    assert view.invalidated
    assert view_warp.invalidated
    assert backend.simulation_view is None
    assert PhysxManager._view is None
    assert PhysxManager._view_warp is None
    assert not PhysxManager._view_created
