# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Shared mocked rigid-object backend factories for interface tests."""

from unittest.mock import MagicMock

from _iface_test_boot import simulation_app

import numpy as np
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.test.mock_interfaces.utils import MockWrenchComposer

BACKENDS = ["Mock"]  # Mock backend is always available.

try:
    from isaaclab_physx.assets.rigid_object.rigid_object import RigidObject as PhysXRigidObject
    from isaaclab_physx.assets.rigid_object.rigid_object_data import RigidObjectData as PhysXRigidObjectData
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
    from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject as NewtonRigidObject
    from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData as NewtonRigidObjectData
    from isaaclab_newton.test.mock_interfaces.views import MockNewtonArticulationView as NewtonMockArticulationView
except ImportError:
    pass
else:
    BACKENDS.append("newton")

try:
    import ovphysx  # noqa: F401

    from isaaclab_ovphysx.assets.rigid_object.rigid_object import RigidObject as OvPhysxRigidObject
    from isaaclab_ovphysx.assets.rigid_object.rigid_object_data import RigidObjectData as OvPhysxRigidObjectData
    from isaaclab_ovphysx.test.mock_interfaces.views import MockOvPhysxBindingSet
except ImportError:
    pass
else:
    BACKENDS.append("ovphysx")


def create_physx_rigid_object(
    num_instances: int = 2,
    device: str = "cuda:0",
):
    """Create a test RigidObject instance with mocked dependencies."""
    body_names = ["body_0"]

    rigid_object = object.__new__(PhysXRigidObject)

    rigid_object.cfg = RigidObjectCfg(prim_path="/World/Object")

    # Create PhysX mock view
    mock_view = PhysXMockRigidBodyViewWarp(
        count=num_instances,
        device=device,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    object.__setattr__(rigid_object, "_root_view", mock_view)
    object.__setattr__(rigid_object, "_device", device)

    # Create RigidObjectData instance (SimulationManager already mocked at module level)
    data = PhysXRigidObjectData(mock_view, device)
    object.__setattr__(rigid_object, "_data", data)

    # Set body names on data
    data.body_names = body_names

    # Create mock wrench composers
    mock_inst_wrench = MockWrenchComposer(rigid_object)
    mock_perm_wrench = MockWrenchComposer(rigid_object)
    object.__setattr__(rigid_object, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(rigid_object, "_permanent_wrench_composer", mock_perm_wrench)

    # Prevent __del__ / _clear_callbacks from raising AttributeError
    object.__setattr__(rigid_object, "_initialize_handle", None)
    object.__setattr__(rigid_object, "_invalidate_initialize_handle", None)
    object.__setattr__(rigid_object, "_prim_deletion_handle", None)
    object.__setattr__(rigid_object, "_debug_vis_handle", None)

    # Set up index arrays (warp arrays for rigid object)
    object.__setattr__(rigid_object, "_ALL_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device))
    object.__setattr__(rigid_object, "_ALL_BODY_INDICES", wp.array(np.array([0], dtype=np.int32), device=device))

    # Cached .view(wp.float32) wrappers
    object.__setattr__(rigid_object, "_root_link_pose_w_f32", None)
    object.__setattr__(rigid_object, "_root_com_vel_w_f32", None)
    object.__setattr__(rigid_object, "_inst_wrench_force_f32", None)
    object.__setattr__(rigid_object, "_inst_wrench_torque_f32", None)
    object.__setattr__(rigid_object, "_perm_wrench_force_f32", None)
    object.__setattr__(rigid_object, "_perm_wrench_torque_f32", None)

    # Pre-allocated pinned CPU buffers for PhysX TensorAPI writes
    N, B = num_instances, 1  # rigid object has 1 body
    object.__setattr__(rigid_object, "_sim_env_ids", wp.empty(N, dtype=wp.int32, device=device))
    object.__setattr__(rigid_object, "_sim_env_ids_views", {})
    cpu_env_ids = wp.array(np.arange(N, dtype=np.int32), device="cpu")
    object.__setattr__(rigid_object, "_cpu_env_ids_all", cpu_env_ids)
    object.__setattr__(rigid_object, "_cpu_env_ids", wp.empty(N, dtype=wp.int32, device="cpu", pinned=True))
    object.__setattr__(rigid_object, "_cpu_env_ids_views", {})
    object.__setattr__(rigid_object, "_cpu_body_mass", wp.zeros((N, B), dtype=wp.float32, device="cpu"))
    object.__setattr__(rigid_object, "_cpu_body_coms", wp.zeros((N, B, 7), dtype=wp.float32, device="cpu"))
    object.__setattr__(rigid_object, "_cpu_body_inertia", wp.zeros((N, B, 9), dtype=wp.float32, device="cpu"))

    return rigid_object, mock_view


def create_newton_rigid_object(
    num_instances: int = 2,
    device: str = "cuda:0",
):
    """Create a test Newton RigidObject instance with mocked dependencies."""
    import isaaclab_newton.assets.rigid_object.rigid_object_data as newton_data_module

    body_names = ["body_0"]

    # Create Newton mock view (uses ArticulationView with num_bodies=1 for rigid objects)
    mock_view = NewtonMockArticulationView(
        num_instances=num_instances,
        num_bodies=1,
        num_joints=0,
        device=device,
        is_fixed_base=False,
        joint_names=[],
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

    # Patch SimulationManager in the Newton data module
    original_sim_manager = newton_data_module.SimulationManager
    newton_data_module.SimulationManager = mock_manager

    try:
        data = NewtonRigidObjectData(mock_view, device)
    finally:
        newton_data_module.SimulationManager = original_sim_manager

    # Create RigidObject shell (bypass __init__)
    rigid_object = object.__new__(NewtonRigidObject)

    rigid_object.cfg = RigidObjectCfg(prim_path="/World/Object")

    object.__setattr__(rigid_object, "_root_view", mock_view)
    object.__setattr__(rigid_object, "_device", device)
    object.__setattr__(rigid_object, "_data", data)

    # Mock wrench composers
    mock_inst_wrench = MockWrenchComposer(rigid_object)
    mock_perm_wrench = MockWrenchComposer(rigid_object)
    object.__setattr__(rigid_object, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(rigid_object, "_permanent_wrench_composer", mock_perm_wrench)

    # Prevent __del__ / _clear_callbacks from raising AttributeError
    object.__setattr__(rigid_object, "_initialize_handle", None)
    object.__setattr__(rigid_object, "_invalidate_initialize_handle", None)
    object.__setattr__(rigid_object, "_prim_deletion_handle", None)
    object.__setattr__(rigid_object, "_debug_vis_handle", None)

    # Newton uses wp.array for indices
    object.__setattr__(rigid_object, "_ALL_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device))
    object.__setattr__(rigid_object, "_ALL_BODY_INDICES", wp.array(np.array([0], dtype=np.int32), device=device))

    # Newton uses wp.bool masks
    object.__setattr__(rigid_object, "_ALL_ENV_MASK", wp.ones((num_instances,), dtype=wp.bool, device=device))
    object.__setattr__(rigid_object, "_ALL_BODY_MASK", wp.ones((1,), dtype=wp.bool, device=device))

    return rigid_object, mock_view


def create_ovphysx_rigid_object(
    num_instances: int = 2,
    device: str = "cuda:0",
):
    """Create a test OvPhysX RigidObject instance with mocked tensor bindings."""
    body_names = ["base_link"]

    obj = object.__new__(OvPhysxRigidObject)

    obj.cfg = RigidObjectCfg(prim_path="/World/object")

    # Create mock binding set
    mock_bindings = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=0,
        num_bodies=1,
        body_names=body_names,
        asset_kind="rigid_object",
    )
    mock_bindings.set_random_data()

    object.__setattr__(obj, "_device", device)
    object.__setattr__(obj, "_ovphysx", MagicMock())
    object.__setattr__(obj, "_root_view", mock_bindings.view)
    object.__setattr__(obj, "_bindings", mock_bindings.bindings)
    object.__setattr__(obj, "_num_instances", num_instances)
    object.__setattr__(obj, "_num_bodies", 1)
    object.__setattr__(obj, "_body_names", body_names)

    # Create RigidObjectData
    data = OvPhysxRigidObjectData(mock_bindings.view, device)
    data.num_instances = num_instances
    data.num_bodies = 1
    data._is_primed = True
    object.__setattr__(obj, "_data", data)

    # Build the buffers RigidObject normally allocates in _initialize_impl
    # (_ALL_INDICES, _ALL_*_MASK, pinned CPU staging buffers, wrench buf).
    # _create_buffers also instantiates real WrenchComposers; those get
    # replaced with mocks just below.
    obj._create_buffers()

    # Replace the real wrench composers with mocks for iface coverage.
    mock_inst_wrench = MockWrenchComposer(obj)
    mock_perm_wrench = MockWrenchComposer(obj)
    object.__setattr__(obj, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(obj, "_permanent_wrench_composer", mock_perm_wrench)

    # Prevent __del__ / _clear_callbacks from raising
    object.__setattr__(obj, "_initialize_handle", None)
    object.__setattr__(obj, "_invalidate_initialize_handle", None)
    object.__setattr__(obj, "_prim_deletion_handle", None)
    object.__setattr__(obj, "_debug_vis_handle", None)

    return obj, mock_bindings


def create_mock_rigid_object(
    num_instances: int = 2,
    device: str = "cuda:0",
):
    from isaaclab.test.mock_interfaces.assets.mock_rigid_object import MockRigidObject

    obj = MockRigidObject(
        num_instances=num_instances,
        device=device,
    )
    return obj, None  # No view for mock backend


def get_rigid_object(
    backend: str,
    num_instances: int = 2,
    device: str = "cuda:0",
):
    if backend == "physx":
        return create_physx_rigid_object(num_instances, device)
    elif backend == "ovphysx":
        return create_ovphysx_rigid_object(num_instances, device)
    elif backend == "newton":
        return create_newton_rigid_object(num_instances, device)
    elif backend.lower() == "mock":
        return create_mock_rigid_object(num_instances, device)
    else:
        raise ValueError(f"Invalid backend: {backend}")
