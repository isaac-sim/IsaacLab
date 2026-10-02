# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Shared mocked rigid-object-collection backend factories for the contract tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg
from isaaclab.utils.wrench_composer import WrenchComposer

from .capabilities import available_backends
from .mock_backends import install_physx_recording_setters, patch_ovphysx_manager, patch_physx_manager

BACKENDS = available_backends()

if "physx" in BACKENDS:
    from isaaclab_physx.assets.rigid_object_collection.rigid_object_collection import (
        RigidObjectCollection as PhysXRigidObjectCollection,
    )
    from isaaclab_physx.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData as PhysXRigidObjectCollectionData,
    )
    from isaaclab_physx.test.fixtures.views import MockRigidBodyViewWarp as PhysXMockRigidBodyViewWarp

if "newton" in BACKENDS:
    from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection import (
        RigidObjectCollection as NewtonRigidObjectCollection,
    )
    from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData as NewtonRigidObjectCollectionData,
    )
    from isaaclab_newton.test.fixtures.views import MockNewtonCollectionView as NewtonMockCollectionView

if "ovphysx" in BACKENDS:
    from isaaclab_ov.assets.rigid_object_collection.rigid_object_collection import (
        RigidObjectCollection as OvPhysxRigidObjectCollection,
    )
    from isaaclab_ov.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData as OvPhysxRigidObjectCollectionData,
    )
    from isaaclab_ov.test.fixtures.views import MockOvPhysxBindingSet

_PHYSX_RIGID_BODY_STORAGE = {
    "set_transforms": "_transforms",
    "set_velocities": "_velocities",
    "set_masses": "_masses",
    "set_coms": "_coms",
    "set_inertias": "_inertias",
}


def create_physx_rigid_object_collection(
    num_instances: int = 2, num_bodies: int = 3, device: str = "cuda:0", *, monkeypatch: pytest.MonkeyPatch
):
    """Create a test RigidObjectCollection instance with mocked dependencies."""
    patch_physx_manager(monkeypatch=monkeypatch)
    collection = object.__new__(PhysXRigidObjectCollection)

    rigid_objects = {f"object_{i}": RigidObjectCfg(prim_path=f"/World/Object_{i}") for i in range(num_bodies)}
    collection.cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

    # View count = num_instances * num_bodies (body-major view order)
    mock_view = PhysXMockRigidBodyViewWarp(
        count=num_instances * num_bodies,
        device=device,
    )
    mock_view.set_random_mock_data()
    install_physx_recording_setters(mock_view, _PHYSX_RIGID_BODY_STORAGE)

    collection._root_view = mock_view
    collection._device = device
    collection._num_bodies = num_bodies
    collection._num_instances = num_instances
    collection._body_names_list = [f"object_{i}" for i in range(num_bodies)]

    # Create RigidObjectCollectionData instance
    data = PhysXRigidObjectCollectionData(mock_view, num_bodies, device)
    collection._data = data
    data.body_names = [f"object_{i}" for i in range(num_bodies)]

    # Create wrench composers
    mock_inst_wrench = WrenchComposer(collection, supports_world_at_com=True)
    mock_perm_wrench = WrenchComposer(collection, supports_world_at_com=True)
    collection._instantaneous_wrench_composer = mock_inst_wrench
    collection._permanent_wrench_composer = mock_perm_wrench

    # Prevent __del__ / _clear_callbacks from raising AttributeError
    collection._initialize_handle = None
    collection._invalidate_initialize_handle = None
    collection._prim_deletion_handle = None
    collection._debug_vis_handle = None

    # Set up index arrays
    collection._ALL_ENV_INDICES = wp.array(np.arange(num_instances, dtype=np.int32), device=device)
    collection._ALL_BODY_INDICES = wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    num_view_ids = num_instances * num_bodies
    all_view_ids = wp.array(np.arange(num_view_ids, dtype=np.int32), device=device)
    collection._ALL_VIEW_INDICES = all_view_ids
    collection._sim_view_ids = wp.empty(num_view_ids, dtype=wp.int32, device=device)
    collection._sim_view_ids_views = {}
    cpu_all_view_ids = wp.empty(num_view_ids, dtype=wp.int32, device="cpu", pinned=wp.is_cuda_available())
    wp.copy(cpu_all_view_ids, all_view_ids)
    collection._cpu_all_view_ids = cpu_all_view_ids
    collection._cpu_view_ids = wp.empty(num_view_ids, dtype=wp.int32, device="cpu", pinned=wp.is_cuda_available())
    collection._cpu_view_ids_views = {}

    return collection, mock_view


def create_newton_rigid_object_collection(
    num_instances: int = 2, num_bodies: int = 3, device: str = "cuda:0", *, monkeypatch: pytest.MonkeyPatch
):
    """Create a test Newton RigidObjectCollection instance with mocked dependencies."""
    import isaaclab_newton.assets.rigid_object_collection.rigid_object_collection as newton_coll_module
    import isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data as newton_data_module

    body_names = [f"object_{i}" for i in range(num_bodies)]

    # Create collection-specific mock view with (N, B) root shapes
    mock_view = NewtonMockCollectionView(
        num_envs=num_instances,
        num_bodies=num_bodies,
        device=device,
        body_names=body_names,
    )
    mock_view.set_random_mock_data()

    # Mock NewtonManager (aliased as SimulationManager in Newton modules)
    mock_model = MagicMock()
    mock_model.world_count = num_instances
    mock_model.gravity = wp.array(
        np.tile(np.array([[0.0, 0.0, -9.81]], dtype=np.float32), (num_instances + 1, 1)),
        dtype=wp.vec3f,
        device=device,
    )
    mock_state = MagicMock()
    mock_control = MagicMock()

    mock_manager = MagicMock()
    mock_manager.get_model.return_value = mock_model
    mock_manager.get_state_0.return_value = mock_state
    mock_manager.get_state_1.return_value = mock_state
    mock_manager.get_control.return_value = mock_control

    # Patch SimulationManager in both data and collection modules until the test finishes.
    monkeypatch.setattr(newton_data_module, "SimulationManager", mock_manager, raising=False)
    monkeypatch.setattr(newton_coll_module, "SimulationManager", mock_manager, raising=False)
    data = NewtonRigidObjectCollectionData(mock_view, num_bodies, device)

    # Create collection shell (bypass __init__)
    collection = object.__new__(NewtonRigidObjectCollection)

    rigid_objects = {f"object_{i}": RigidObjectCfg(prim_path=f"/World/Object_{i}") for i in range(num_bodies)}
    collection.cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

    collection._root_view = mock_view
    collection._device = device
    collection._num_bodies = num_bodies
    collection._num_instances = num_instances
    collection._body_names_list = body_names
    collection._data = data
    data.body_names = body_names

    # Wrench composers (Newton-specific)
    mock_inst_wrench = WrenchComposer(collection)
    mock_perm_wrench = WrenchComposer(collection)
    collection._instantaneous_wrench_composer = mock_inst_wrench
    collection._permanent_wrench_composer = mock_perm_wrench

    # Prevent __del__ / _clear_callbacks from raising AttributeError
    collection._initialize_handle = None
    collection._invalidate_initialize_handle = None
    collection._prim_deletion_handle = None
    collection._debug_vis_handle = None

    # Index arrays (warp)
    collection._ALL_ENV_INDICES = wp.array(np.arange(num_instances, dtype=np.int32), device=device)
    collection._ALL_BODY_INDICES = wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    collection._ALL_ENV_MASK = wp.ones((num_instances,), dtype=wp.bool, device=device)
    collection._ALL_BODY_MASK = wp.ones((num_bodies,), dtype=wp.bool, device=device)

    return collection, mock_view


def create_ovphysx_rigid_object_collection(
    num_instances: int = 2, num_bodies: int = 3, device: str = "cuda:0", *, monkeypatch: pytest.MonkeyPatch
):
    """Create a test OVPhysX RigidObjectCollection instance with mocked tensor bindings."""
    patch_ovphysx_manager(monkeypatch=monkeypatch)
    body_names = [f"object_{i}" for i in range(num_bodies)]

    collection = object.__new__(OvPhysxRigidObjectCollection)

    rigid_objects = {f"object_{i}": RigidObjectCfg(prim_path=f"/World/Object_{i}") for i in range(num_bodies)}
    collection.cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

    # Use articulation-mode bindings with num_joints=0 to get (N, B, ...) shaped tensors.
    mock_bindings = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=0,
        num_bodies=num_bodies,
        body_names=body_names,
        asset_kind="articulation",
    )
    mock_bindings.set_random_data()

    collection._device = device
    collection._ovphysx = MagicMock()
    collection._root_view = mock_bindings.view
    collection._bindings = mock_bindings.bindings
    collection._num_instances = num_instances
    collection._num_bodies = num_bodies
    collection._body_names_list = body_names

    # Create RigidObjectCollectionData
    data = OvPhysxRigidObjectCollectionData(mock_bindings.view, num_bodies, device)
    data.num_instances = num_instances
    data.num_bodies = num_bodies
    data._is_primed = True
    collection._data = data

    # Allocate the buffers that RigidObjectCollection normally allocates in _initialize_impl.
    collection._create_buffers()

    # Use production wrench composers for interface coverage.
    mock_inst_wrench = WrenchComposer(collection, supports_world_at_com=True)
    mock_perm_wrench = WrenchComposer(collection, supports_world_at_com=True)
    collection._instantaneous_wrench_composer = mock_inst_wrench
    collection._permanent_wrench_composer = mock_perm_wrench

    # Prevent __del__ / _clear_callbacks from raising
    collection._initialize_handle = None
    collection._invalidate_initialize_handle = None
    collection._prim_deletion_handle = None
    collection._debug_vis_handle = None

    return collection, mock_bindings


def get_rigid_object_collection(
    backend: str,
    num_instances: int = 2,
    num_bodies: int = 3,
    device: str = "cuda:0",
    *,
    monkeypatch: pytest.MonkeyPatch,
):
    if backend == "physx":
        return create_physx_rigid_object_collection(num_instances, num_bodies, device, monkeypatch=monkeypatch)
    elif backend == "ovphysx":
        return create_ovphysx_rigid_object_collection(num_instances, num_bodies, device, monkeypatch=monkeypatch)
    elif backend == "newton":
        return create_newton_rigid_object_collection(num_instances, num_bodies, device, monkeypatch=monkeypatch)
    else:
        raise ValueError(f"Invalid backend: {backend}")
