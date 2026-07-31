# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Shared mocked rigid-object-collection backend factories for interface tests."""

from unittest.mock import MagicMock

from _iface_test_boot import simulation_app

import numpy as np
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg
from isaaclab.test.mock_interfaces.utils import MockWrenchComposer

BACKENDS = ["Mock"]  # Mock backend is always available.

try:
    from isaaclab_physx.assets.rigid_object_collection.rigid_object_collection import (
        RigidObjectCollection as PhysXRigidObjectCollection,
    )
    from isaaclab_physx.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData as PhysXRigidObjectCollectionData,
    )
    from isaaclab_physx.physics import PhysxManager as SimulationManager
    from isaaclab_physx.test.mock_interfaces.views import MockRigidBodyViewWarp as PhysXMockRigidBodyViewWarp
except ImportError:
    pass
else:
    # PhysX data classes need gravity even though interface tests do not create a physics scene.
    _mock_physics_sim_view = MagicMock()
    _mock_physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)
    SimulationManager.get_physics_sim_view = MagicMock(return_value=_mock_physics_sim_view)

    BACKENDS.append("physx")

try:
    from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection import (
        RigidObjectCollection as NewtonRigidObjectCollection,
    )
    from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData as NewtonRigidObjectCollectionData,
    )
    from isaaclab_newton.test.mock_interfaces.mock_newton import MockWrenchComposer as NewtonMockWrenchComposer
    from isaaclab_newton.test.mock_interfaces.views import MockNewtonCollectionView as NewtonMockCollectionView
except ImportError:
    pass
else:
    BACKENDS.append("newton")

try:
    import ovphysx  # noqa: F401

    from isaaclab_ovphysx.assets.rigid_object_collection.rigid_object_collection import (
        RigidObjectCollection as OvPhysxRigidObjectCollection,
    )
    from isaaclab_ovphysx.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData as OvPhysxRigidObjectCollectionData,
    )
    from isaaclab_ovphysx.test.mock_interfaces.views import MockOvPhysxBindingSet
except ImportError:
    pass
else:
    if hasattr(OvPhysxRigidObjectCollection, "_create_buffers"):
        BACKENDS.append("ovphysx")


def create_physx_rigid_object_collection(
    num_instances: int = 2,
    num_bodies: int = 3,
    device: str = "cuda:0",
):
    """Create a test RigidObjectCollection instance with mocked dependencies."""
    collection = object.__new__(PhysXRigidObjectCollection)

    rigid_objects = {f"object_{i}": RigidObjectCfg(prim_path=f"/World/Object_{i}") for i in range(num_bodies)}
    collection.cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

    # View count = num_instances * num_bodies (body-major view order)
    mock_view = PhysXMockRigidBodyViewWarp(
        count=num_instances * num_bodies,
        device=device,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    object.__setattr__(collection, "_root_view", mock_view)
    object.__setattr__(collection, "_device", device)
    object.__setattr__(collection, "_num_bodies", num_bodies)
    object.__setattr__(collection, "_num_instances", num_instances)
    object.__setattr__(collection, "_body_names_list", [f"object_{i}" for i in range(num_bodies)])

    # Create RigidObjectCollectionData instance
    data = PhysXRigidObjectCollectionData(mock_view, num_bodies, device)
    object.__setattr__(collection, "_data", data)
    data.body_names = [f"object_{i}" for i in range(num_bodies)]

    # Create mock wrench composers
    mock_inst_wrench = MockWrenchComposer(collection)
    mock_perm_wrench = MockWrenchComposer(collection)
    object.__setattr__(collection, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(collection, "_permanent_wrench_composer", mock_perm_wrench)

    # Prevent __del__ / _clear_callbacks from raising AttributeError
    object.__setattr__(collection, "_initialize_handle", None)
    object.__setattr__(collection, "_invalidate_initialize_handle", None)
    object.__setattr__(collection, "_prim_deletion_handle", None)
    object.__setattr__(collection, "_debug_vis_handle", None)

    # Set up index arrays
    object.__setattr__(
        collection, "_ALL_ENV_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device)
    )
    object.__setattr__(collection, "_ALL_BODY_INDICES", wp.array(np.arange(num_bodies, dtype=np.int32), device=device))
    num_view_ids = num_instances * num_bodies
    all_view_ids = wp.array(np.arange(num_view_ids, dtype=np.int32), device=device)
    object.__setattr__(collection, "_ALL_VIEW_INDICES", all_view_ids)
    object.__setattr__(collection, "_sim_view_ids", wp.empty(num_view_ids, dtype=wp.int32, device=device))
    object.__setattr__(collection, "_sim_view_ids_views", {})
    cpu_all_view_ids = wp.empty(num_view_ids, dtype=wp.int32, device="cpu", pinned=True)
    wp.copy(cpu_all_view_ids, all_view_ids)
    object.__setattr__(collection, "_cpu_all_view_ids", cpu_all_view_ids)
    object.__setattr__(collection, "_cpu_view_ids", wp.empty(num_view_ids, dtype=wp.int32, device="cpu", pinned=True))
    object.__setattr__(collection, "_cpu_view_ids_views", {})

    return collection, mock_view


def create_newton_rigid_object_collection(
    num_instances: int = 2,
    num_bodies: int = 3,
    device: str = "cuda:0",
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
    mock_view._noop_setters = True

    # Mock NewtonManager (aliased as SimulationManager in Newton modules)
    mock_model = MagicMock()
    mock_model.gravity = wp.array(np.array([[0.0, 0.0, -9.81]], dtype=np.float32), dtype=wp.vec3f, device=device)
    mock_state = MagicMock()
    mock_control = MagicMock()

    mock_manager = MagicMock()
    mock_manager.get_model.return_value = mock_model
    mock_manager.get_state_0.return_value = mock_state
    mock_manager.get_state_1.return_value = mock_state
    mock_manager.get_control.return_value = mock_control

    # Patch SimulationManager in both data and collection modules
    original_data_manager = newton_data_module.SimulationManager
    original_coll_manager = newton_coll_module.SimulationManager
    newton_data_module.SimulationManager = mock_manager
    newton_coll_module.SimulationManager = mock_manager

    try:
        data = NewtonRigidObjectCollectionData(mock_view, num_bodies, device)
    finally:
        newton_data_module.SimulationManager = original_data_manager
        newton_coll_module.SimulationManager = original_coll_manager

    # Create collection shell (bypass __init__)
    collection = object.__new__(NewtonRigidObjectCollection)

    rigid_objects = {f"object_{i}": RigidObjectCfg(prim_path=f"/World/Object_{i}") for i in range(num_bodies)}
    collection.cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

    object.__setattr__(collection, "_root_view", mock_view)
    object.__setattr__(collection, "_device", device)
    object.__setattr__(collection, "_num_bodies", num_bodies)
    object.__setattr__(collection, "_num_instances", num_instances)
    object.__setattr__(collection, "_body_names_list", body_names)
    object.__setattr__(collection, "_data", data)
    data.body_names = body_names

    # Mock wrench composers (Newton-specific)
    mock_inst_wrench = NewtonMockWrenchComposer(collection)
    mock_perm_wrench = NewtonMockWrenchComposer(collection)
    object.__setattr__(collection, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(collection, "_permanent_wrench_composer", mock_perm_wrench)

    # Prevent __del__ / _clear_callbacks from raising AttributeError
    object.__setattr__(collection, "_initialize_handle", None)
    object.__setattr__(collection, "_invalidate_initialize_handle", None)
    object.__setattr__(collection, "_prim_deletion_handle", None)
    object.__setattr__(collection, "_debug_vis_handle", None)

    # Index arrays (warp)
    object.__setattr__(
        collection, "_ALL_ENV_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device)
    )
    object.__setattr__(collection, "_ALL_BODY_INDICES", wp.array(np.arange(num_bodies, dtype=np.int32), device=device))
    object.__setattr__(collection, "_ALL_ENV_MASK", wp.ones((num_instances,), dtype=wp.bool, device=device))
    object.__setattr__(collection, "_ALL_BODY_MASK", wp.ones((num_bodies,), dtype=wp.bool, device=device))

    return collection, mock_view


def create_ovphysx_rigid_object_collection(
    num_instances: int = 2,
    num_bodies: int = 3,
    device: str = "cuda:0",
):
    """Create a test OVPhysX RigidObjectCollection instance with mocked tensor bindings."""
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

    object.__setattr__(collection, "_device", device)
    object.__setattr__(collection, "_ovphysx", MagicMock())
    object.__setattr__(collection, "_root_view", mock_bindings.view)
    object.__setattr__(collection, "_bindings", mock_bindings.bindings)
    object.__setattr__(collection, "_num_instances", num_instances)
    object.__setattr__(collection, "_num_bodies", num_bodies)
    object.__setattr__(collection, "_body_names_list", body_names)

    # Create RigidObjectCollectionData
    data = OvPhysxRigidObjectCollectionData(mock_bindings.view, num_bodies, device)
    data.num_instances = num_instances
    data.num_bodies = num_bodies
    data._is_primed = True
    object.__setattr__(collection, "_data", data)

    # Allocate the buffers that RigidObjectCollection normally allocates in _initialize_impl.
    collection._create_buffers()

    # Replace the real wrench composers with mocks for iface coverage.
    mock_inst_wrench = MockWrenchComposer(collection)
    mock_perm_wrench = MockWrenchComposer(collection)
    object.__setattr__(collection, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(collection, "_permanent_wrench_composer", mock_perm_wrench)

    # Prevent __del__ / _clear_callbacks from raising
    object.__setattr__(collection, "_initialize_handle", None)
    object.__setattr__(collection, "_invalidate_initialize_handle", None)
    object.__setattr__(collection, "_prim_deletion_handle", None)
    object.__setattr__(collection, "_debug_vis_handle", None)

    return collection, mock_bindings


def create_mock_rigid_object_collection(
    num_instances: int = 2,
    num_bodies: int = 3,
    device: str = "cuda:0",
):
    from isaaclab.test.mock_interfaces.assets.mock_rigid_object_collection import MockRigidObjectCollection

    obj = MockRigidObjectCollection(
        num_instances=num_instances,
        num_bodies=num_bodies,
        device=device,
    )
    return obj, None


def get_rigid_object_collection(
    backend: str,
    num_instances: int = 2,
    num_bodies: int = 3,
    device: str = "cuda:0",
):
    if backend == "physx":
        return create_physx_rigid_object_collection(num_instances, num_bodies, device)
    elif backend == "ovphysx":
        return create_ovphysx_rigid_object_collection(num_instances, num_bodies, device)
    elif backend == "newton":
        return create_newton_rigid_object_collection(num_instances, num_bodies, device)
    elif backend.lower() == "mock":
        return create_mock_rigid_object_collection(num_instances, num_bodies, device)
    else:
        raise ValueError(f"Invalid backend: {backend}")
