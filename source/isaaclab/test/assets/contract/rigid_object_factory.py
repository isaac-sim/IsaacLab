# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Mocked rigid-object and rigid-object-collection backends for the contract tests.

Each factory bypasses ``__init__`` and sets the attributes that ``_initialize_impl`` would create, so the real asset
and data classes run against mocked views. They return the asset and the raw backend view or binding set.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg
from isaaclab.utils.wrench_composer import WrenchComposer

from .backends import AVAILABLE, install_physx_recording_setters, patch_ovphysx_manager, patch_physx_manager

if "physx" in AVAILABLE:
    import isaaclab_physx.assets as physx_assets
    from isaaclab_physx.test.fixtures.views import MockRigidBodyViewWarp

if "newton" in AVAILABLE:
    import isaaclab_newton.assets as newton_assets
    import isaaclab_newton.assets.rigid_object.rigid_object_data as newton_rigid_object_data
    import isaaclab_newton.assets.rigid_object_collection.rigid_object_collection as newton_collection
    import isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data as newton_collection_data
    from isaaclab_newton.test.fixtures.views import MockNewtonArticulationView, MockNewtonCollectionView

if "ovphysx" in AVAILABLE:
    import isaaclab_ov.assets as ovphysx_assets
    from isaaclab_ov.test.fixtures.views import MockOvPhysxBindingSet

_PHYSX_STORAGE = {
    "set_transforms": "_transforms",
    "set_velocities": "_velocities",
    "set_masses": "_masses",
    "set_coms": "_coms",
    "set_inertias": "_inertias",
}


def _finish_shell(asset, *, supports_world_at_com: bool = True) -> None:
    """Attach wrench composers and clear the callback handles that ``__del__`` releases."""
    asset._instantaneous_wrench_composer = WrenchComposer(asset, supports_world_at_com=supports_world_at_com)
    asset._permanent_wrench_composer = WrenchComposer(asset, supports_world_at_com=supports_world_at_com)
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None
    asset._debug_vis_handle = None


def _indices(count: int, device: str) -> wp.array:
    return wp.array(np.arange(count, dtype=np.int32), device=device)


def _newton_manager(num_instances: int, device: str) -> MagicMock:
    """Return a NewtonManager double with a gravity-carrying model."""
    model = MagicMock(world_count=num_instances)
    model.gravity = wp.array(np.tile([[0.0, 0.0, -9.81]], (num_instances + 1, 1)), dtype=wp.vec3f, device=device)
    manager = MagicMock()
    manager.get_model.return_value = model
    manager.get_state_0.return_value = manager.get_state_1.return_value = MagicMock()
    return manager


def _physx_rigid_object(num_instances: int, device: str, monkeypatch: pytest.MonkeyPatch):
    patch_physx_manager(monkeypatch)
    view = MockRigidBodyViewWarp(count=num_instances, device=device)
    view.set_random_mock_data()
    install_physx_recording_setters(view, _PHYSX_STORAGE)

    obj = object.__new__(physx_assets.RigidObject)
    obj.cfg = RigidObjectCfg(prim_path="/World/Object")
    obj._root_view = view
    obj._device = device
    obj._data = physx_assets.RigidObjectData(view, device)
    obj._data.body_names = ["body_0"]
    _finish_shell(obj)
    obj._ALL_INDICES = _indices(num_instances, device)
    obj._ALL_BODY_INDICES = _indices(1, device)
    obj._root_link_pose_w_f32 = None
    obj._root_com_vel_w_f32 = None
    # Pinned CPU staging buffers for PhysX TensorAPI writes.
    pinned = wp.is_cuda_available()
    obj._sim_env_ids = wp.empty(num_instances, dtype=wp.int32, device=device)
    obj._sim_env_ids_views = {}
    obj._cpu_env_ids_all = _indices(num_instances, "cpu")
    obj._cpu_env_ids = wp.empty(num_instances, dtype=wp.int32, device="cpu", pinned=pinned)
    obj._cpu_env_ids_views = {}
    obj._cpu_body_mass = wp.zeros((num_instances, 1), dtype=wp.float32, device="cpu")
    obj._cpu_body_coms = wp.zeros((num_instances, 1, 7), dtype=wp.float32, device="cpu")
    obj._cpu_body_inertia = wp.zeros((num_instances, 1, 9), dtype=wp.float32, device="cpu")
    return obj, view


def _newton_rigid_object(num_instances: int, device: str, monkeypatch: pytest.MonkeyPatch):
    view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=1,
        num_joints=0,
        device=device,
        is_fixed_base=False,
        joint_names=[],
        body_names=["body_0"],
    )
    view.set_random_mock_data()
    monkeypatch.setattr(newton_rigid_object_data, "SimulationManager", _newton_manager(num_instances, device))

    obj = object.__new__(newton_assets.RigidObject)
    obj.cfg = RigidObjectCfg(prim_path="/World/Object")
    obj._root_view = view
    obj._device = device
    obj._data = newton_assets.RigidObjectData(view, device)
    _finish_shell(obj, supports_world_at_com=False)
    obj._ALL_INDICES = _indices(num_instances, device)
    obj._ALL_BODY_INDICES = _indices(1, device)
    obj._ALL_ENV_MASK = wp.ones((num_instances,), dtype=wp.bool, device=device)
    obj._ALL_BODY_MASK = wp.ones((1,), dtype=wp.bool, device=device)
    return obj, view


def _ovphysx_rigid_object(num_instances: int, device: str, monkeypatch: pytest.MonkeyPatch):
    patch_ovphysx_manager(monkeypatch)
    bindings = MockOvPhysxBindingSet(
        num_instances=num_instances, num_joints=0, num_bodies=1, body_names=["body_0"], asset_kind="rigid_object"
    )
    bindings.set_random_data()

    obj = object.__new__(ovphysx_assets.RigidObject)
    obj.cfg = RigidObjectCfg(prim_path="/World/Object")
    obj._device = device
    obj._ovphysx = MagicMock()
    obj._root_view = bindings.view
    obj._bindings = bindings.bindings
    obj._num_instances = num_instances
    obj._num_bodies = 1
    obj._body_names = ["body_0"]
    obj._data = ovphysx_assets.RigidObjectData(bindings.view, device)
    obj._data.num_instances = num_instances
    obj._data.num_bodies = 1
    obj._data._is_primed = True
    obj._create_buffers()
    _finish_shell(obj)
    return obj, bindings


def _collection_cfg(body_names: list[str]) -> RigidObjectCollectionCfg:
    return RigidObjectCollectionCfg(
        rigid_objects={name: RigidObjectCfg(prim_path=f"/World/{name}") for name in body_names}
    )


def _physx_collection(num_instances: int, num_bodies: int, device: str, monkeypatch: pytest.MonkeyPatch):
    patch_physx_manager(monkeypatch)
    body_names = [f"object_{i}" for i in range(num_bodies)]
    # PhysX collection views are body-major: one view entry per (body, environment).
    num_view_ids = num_instances * num_bodies
    view = MockRigidBodyViewWarp(count=num_view_ids, device=device)
    view.set_random_mock_data()
    install_physx_recording_setters(view, _PHYSX_STORAGE)

    collection = object.__new__(physx_assets.RigidObjectCollection)
    collection.cfg = _collection_cfg(body_names)
    collection._root_view = view
    collection._device = device
    collection._num_bodies = num_bodies
    collection._num_instances = num_instances
    collection._body_names_list = body_names
    collection._data = physx_assets.RigidObjectCollectionData(view, num_bodies, device)
    collection._data.body_names = body_names
    _finish_shell(collection)
    collection._ALL_ENV_INDICES = _indices(num_instances, device)
    collection._ALL_BODY_INDICES = _indices(num_bodies, device)
    collection._ALL_VIEW_INDICES = _indices(num_view_ids, device)
    pinned = wp.is_cuda_available()
    collection._sim_view_ids = wp.empty(num_view_ids, dtype=wp.int32, device=device)
    collection._sim_view_ids_views = {}
    collection._cpu_all_view_ids = wp.empty(num_view_ids, dtype=wp.int32, device="cpu", pinned=pinned)
    wp.copy(collection._cpu_all_view_ids, collection._ALL_VIEW_INDICES)
    collection._cpu_view_ids = wp.empty(num_view_ids, dtype=wp.int32, device="cpu", pinned=pinned)
    collection._cpu_view_ids_views = {}
    return collection, view


def _newton_collection(num_instances: int, num_bodies: int, device: str, monkeypatch: pytest.MonkeyPatch):
    body_names = [f"object_{i}" for i in range(num_bodies)]
    view = MockNewtonCollectionView(num_envs=num_instances, num_bodies=num_bodies, device=device, body_names=body_names)
    view.set_random_mock_data()
    manager = _newton_manager(num_instances, device)
    monkeypatch.setattr(newton_collection_data, "SimulationManager", manager)
    monkeypatch.setattr(newton_collection, "SimulationManager", manager)

    collection = object.__new__(newton_assets.RigidObjectCollection)
    collection.cfg = _collection_cfg(body_names)
    collection._root_view = view
    collection._device = device
    collection._num_bodies = num_bodies
    collection._num_instances = num_instances
    collection._body_names_list = body_names
    collection._data = newton_assets.RigidObjectCollectionData(view, num_bodies, device)
    collection._data.body_names = body_names
    _finish_shell(collection, supports_world_at_com=False)
    collection._ALL_ENV_INDICES = _indices(num_instances, device)
    collection._ALL_BODY_INDICES = _indices(num_bodies, device)
    collection._ALL_ENV_MASK = wp.ones((num_instances,), dtype=wp.bool, device=device)
    collection._ALL_BODY_MASK = wp.ones((num_bodies,), dtype=wp.bool, device=device)
    return collection, view


def _ovphysx_collection(num_instances: int, num_bodies: int, device: str, monkeypatch: pytest.MonkeyPatch):
    patch_ovphysx_manager(monkeypatch)
    body_names = [f"object_{i}" for i in range(num_bodies)]
    # Articulation-mode bindings without joints give the (N, B, ...) tensors of a collection.
    bindings = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=0,
        num_bodies=num_bodies,
        body_names=body_names,
        asset_kind="articulation",
    )
    bindings.set_random_data()

    collection = object.__new__(ovphysx_assets.RigidObjectCollection)
    collection.cfg = _collection_cfg(body_names)
    collection._device = device
    collection._ovphysx = MagicMock()
    collection._root_view = bindings.view
    collection._bindings = bindings.bindings
    collection._num_instances = num_instances
    collection._num_bodies = num_bodies
    collection._body_names_list = body_names
    collection._data = ovphysx_assets.RigidObjectCollectionData(bindings.view, num_bodies, device)
    collection._data.num_instances = num_instances
    collection._data.num_bodies = num_bodies
    collection._data._is_primed = True
    collection._create_buffers()
    _finish_shell(collection)
    return collection, bindings


def get_rigid_object(backend: str, num_instances: int = 2, device: str = "cpu", *, monkeypatch: pytest.MonkeyPatch):
    """Create a mocked single-body rigid object of the given backend."""
    factory = {"physx": _physx_rigid_object, "newton": _newton_rigid_object, "ovphysx": _ovphysx_rigid_object}
    return factory[backend](num_instances, device, monkeypatch)


def get_rigid_object_collection(
    backend: str, num_instances: int = 2, num_bodies: int = 3, device: str = "cpu", *, monkeypatch: pytest.MonkeyPatch
):
    """Create a mocked rigid-object collection of the given backend."""
    factory = {"physx": _physx_collection, "newton": _newton_collection, "ovphysx": _ovphysx_collection}
    return factory[backend](num_instances, num_bodies, device, monkeypatch)
