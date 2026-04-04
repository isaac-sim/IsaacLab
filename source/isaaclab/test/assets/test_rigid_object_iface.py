# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""
Checks that the rigid object interfaces are consistent across backends, and are providing the exact same data as what
the base rigid object class advertises. All rigid object interfaces need to comply with the same interface contract.

The setup is a bit convoluted so that we can run these tests without requiring Isaac Sim or GPU simulation.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

HEADLESS = True

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.test.mock_interfaces.utils import MockWrenchComposer

# Mock SimulationManager.get_physics_sim_view() to return a mock object with gravity
# This is needed because the Data classes call SimulationManager.get_physics_sim_view().get_gravity()
# but there's no actual physics scene when running unit tests
_mock_physics_sim_view = MagicMock()
_mock_physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)

from isaacsim.core.simulation_manager import SimulationManager

SimulationManager.get_physics_sim_view = MagicMock(return_value=_mock_physics_sim_view)

"""
Check which backends are available.
"""

BACKENDS = ["Mock"]  # Mock backend is always available.

try:
    from isaaclab_physx.assets.rigid_object.rigid_object import RigidObject as PhysXRigidObject
    from isaaclab_physx.assets.rigid_object.rigid_object_data import RigidObjectData as PhysXRigidObjectData
    from isaaclab_physx.test.mock_interfaces.views import MockRigidBodyViewWarp as PhysXMockRigidBodyViewWarp

    BACKENDS.append("physx")
except ImportError:
    pass

try:
    from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject as NewtonRigidObject
    from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData as NewtonRigidObjectData
    from isaaclab_newton.test.mock_interfaces.views import MockNewtonArticulationView as NewtonMockArticulationView

    BACKENDS.append("newton")
except ImportError:
    pass


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
    elif backend == "newton":
        return create_newton_rigid_object(num_instances, device)
    elif backend.lower() == "mock":
        return create_mock_rigid_object(num_instances, device)
    else:
        raise ValueError(f"Invalid backend: {backend}")


@pytest.fixture
def rigid_object_iface(request):
    backend = request.getfixturevalue("backend")
    num_instances = request.getfixturevalue("num_instances")
    device = request.getfixturevalue("device")
    return get_rigid_object(backend, num_instances, device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_wp_array(arr, *, expected_shape: tuple, expected_dtype: type, name: str):
    """Assert that `arr` is a wp.array with the expected shape and dtype."""
    assert isinstance(arr, wp.array), f"{name}: expected wp.array, got {type(arr)}"
    assert arr.shape == expected_shape, f"{name}: expected shape {expected_shape}, got {arr.shape}"
    assert arr.dtype == expected_dtype, f"{name}: expected dtype {expected_dtype}, got {arr.dtype}"


# Common parametrize decorators
_backends = pytest.mark.parametrize("backend", BACKENDS, indirect=False)

_default_dims = pytest.mark.parametrize("num_instances", [1, 2, 100])

_default_devices = pytest.mark.parametrize("device", ["cuda:0", "cpu"])


# ---------------------------------------------------------------------------
# Tests: RigidObject properties
# ---------------------------------------------------------------------------


class TestRigidObjectProperties:
    """Test that rigid object properties return the correct types/values."""

    @_backends
    @_default_dims
    @_default_devices
    def test_num_instances(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        assert obj.num_instances == num_instances

    @_backends
    @_default_dims
    @_default_devices
    def test_num_bodies(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        assert obj.num_bodies == 1

    @_backends
    @_default_dims
    @_default_devices
    def test_body_names(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        names = obj.body_names
        assert isinstance(names, list)
        assert len(names) == 1
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_devices
    def test_data_returns_rigid_object_data(self, backend, num_instances, device, rigid_object_iface):
        from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData

        obj, _ = rigid_object_iface
        assert isinstance(obj.data, BaseRigidObjectData)


# ---------------------------------------------------------------------------
# Tests: RigidObject finder methods
# ---------------------------------------------------------------------------


class TestRigidObjectFinders:
    """Test that finder methods return (list[int], list[str]) tuples."""

    @_backends
    @_default_dims
    @_default_devices
    def test_find_bodies_all(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        indices, names = obj.find_bodies(".*")
        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == 1
        assert len(names) == 1
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_devices
    def test_find_bodies_single(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        first_body = obj.body_names[0]
        indices, names = obj.find_bodies(first_body)
        assert indices == [0]
        assert names == [first_body]


# ---------------------------------------------------------------------------
# Tests: RigidObjectData root state properties
# ---------------------------------------------------------------------------


class TestRigidObjectDataRootState:
    """Test data properties for root rigid body state."""

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_pose_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_link_pose_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="root_link_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_link_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="root_link_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_pose_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_com_pose_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="root_com_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_com_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="root_com_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_pos_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_link_pos_w, expected_shape=(num_instances,), expected_dtype=wp.vec3f, name="root_link_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_quat_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_link_quat_w, expected_shape=(num_instances,), expected_dtype=wp.quatf, name="root_link_quat_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_lin_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_link_lin_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_ang_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_link_ang_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_ang_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_pos_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_com_pos_w, expected_shape=(num_instances,), expected_dtype=wp.vec3f, name="root_com_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_quat_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_com_quat_w, expected_shape=(num_instances,), expected_dtype=wp.quatf, name="root_com_quat_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_lin_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_com_lin_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_ang_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_com_ang_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_ang_vel_w",
        )


# ---------------------------------------------------------------------------
# Tests: RigidObjectData derived properties
# ---------------------------------------------------------------------------


class TestRigidObjectDataDerivedProperties:
    """Test derived/computed data properties."""

    @_backends
    @_default_dims
    @_default_devices
    def test_projected_gravity_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.projected_gravity_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="projected_gravity_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_heading_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.heading_w, expected_shape=(num_instances,), expected_dtype=wp.float32, name="heading_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_lin_vel_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_link_lin_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_lin_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_ang_vel_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_link_ang_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_ang_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_lin_vel_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_com_lin_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_lin_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_ang_vel_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.root_com_ang_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_ang_vel_b",
        )


# ---------------------------------------------------------------------------
# Tests: RigidObjectData body state properties
# ---------------------------------------------------------------------------


class TestRigidObjectDataBodyState:
    """Test data properties for all body states."""

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_pose_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_link_pose_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.transformf,
            name="body_link_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_link_vel_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.spatial_vectorf,
            name="body_link_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pose_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_com_pose_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.transformf,
            name="body_com_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_com_vel_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.spatial_vectorf,
            name="body_com_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_acc_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_com_acc_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.spatial_vectorf,
            name="body_com_acc_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pose_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_com_pose_b,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.transformf,
            name="body_com_pose_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_mass(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_mass, expected_shape=(num_instances, 1), expected_dtype=wp.float32, name="body_mass"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_inertia(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_inertia, expected_shape=(num_instances, 1, 9), expected_dtype=wp.float32, name="body_inertia"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_pos_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_link_pos_w, expected_shape=(num_instances, 1), expected_dtype=wp.vec3f, name="body_link_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_quat_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_link_quat_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.quatf,
            name="body_link_quat_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_lin_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_link_lin_vel_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.vec3f,
            name="body_link_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_ang_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_link_ang_vel_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.vec3f,
            name="body_link_ang_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pos_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_com_pos_w, expected_shape=(num_instances, 1), expected_dtype=wp.vec3f, name="body_com_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_quat_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_com_quat_w, expected_shape=(num_instances, 1), expected_dtype=wp.quatf, name="body_com_quat_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pos_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_com_pos_b, expected_shape=(num_instances, 1), expected_dtype=wp.vec3f, name="body_com_pos_b"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_quat_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.body_com_quat_b, expected_shape=(num_instances, 1), expected_dtype=wp.quatf, name="body_com_quat_b"
        )


# ---------------------------------------------------------------------------
# Tests: RigidObjectData defaults
# ---------------------------------------------------------------------------


class TestRigidObjectDataDefaults:
    """Test default state properties."""

    @_backends
    @_default_dims
    @_default_devices
    def test_default_root_pose(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.default_root_pose,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="default_root_pose",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_default_root_vel(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_wp_array(
            obj.data.default_root_vel,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="default_root_vel",
        )


# ---------------------------------------------------------------------------
# Writer/setter test helpers
# ---------------------------------------------------------------------------

# Map warp structured dtypes to their torch trailing dimension size.
_WP_DTYPE_TO_TRAILING = {
    wp.transformf: 7,
    wp.spatial_vectorf: 6,
    wp.vec2f: 2,
    wp.float32: 0,  # no trailing dimension
}


def _make_data_torch(shape: tuple, device: str, wp_dtype=wp.float32) -> torch.Tensor:
    """Create valid torch test data for a given warp dtype."""
    trailing = _WP_DTYPE_TO_TRAILING[wp_dtype]
    if trailing:
        full_shape = (*shape, trailing)
    else:
        full_shape = shape
    data = torch.zeros(full_shape, device=device, dtype=torch.float32)
    if wp_dtype == wp.transformf:
        data[..., 6] = 1.0  # identity quat w
    elif wp_dtype == wp.vec2f:
        data[..., 0] = -1.0
        data[..., 1] = 1.0
    elif wp_dtype == wp.float32:
        data.fill_(1.0)
    return data


def _make_data_warp(shape: tuple, device: str, wp_dtype=wp.float32) -> wp.array:
    """Create valid warp test data for a given warp dtype."""
    t = _make_data_torch(shape, device, wp_dtype)
    if wp_dtype == wp.float32:
        return wp.from_torch(t, dtype=wp.float32)
    return wp.from_torch(t.contiguous(), dtype=wp_dtype)


def _make_bad_data_torch(shape: tuple, device: str, wp_dtype=wp.float32) -> torch.Tensor:
    """Create torch data with wrong leading shape for negative testing."""
    bad_shape = (shape[0] + 1,) + shape[1:]
    return _make_data_torch(bad_shape, device, wp_dtype)


def _make_bad_data_warp(shape: tuple, device: str, wp_dtype=wp.float32) -> wp.array:
    """Create warp data with wrong leading shape for negative testing."""
    bad_shape = (shape[0] + 1,) + shape[1:]
    return _make_data_warp(bad_shape, device, wp_dtype)


def _make_env_mask(num_instances: int, device: str, partial: bool) -> wp.array | None:
    """Create an env_mask: None for all envs, or a partial bool mask."""
    if not partial:
        return None
    mask_np = np.zeros(num_instances, dtype=bool)
    mask_np[0] = True
    return wp.array(mask_np, dtype=wp.bool, device=device)


def _make_env_ids(device: str, subset: bool) -> torch.Tensor | None:
    """Create env_ids: None for all envs, or [0] for a subset."""
    if not subset:
        return None
    return torch.tensor([0], dtype=torch.int32, device=device)


def _make_item_mask(total: int, selected: list[int], device: str) -> wp.array:
    """Create a bool warp mask with True at `selected` indices, False elsewhere."""
    mask_np = np.zeros(total, dtype=bool)
    for i in selected:
        mask_np[i] = True
    return wp.array(mask_np, dtype=wp.bool, device=device)


# ---------------------------------------------------------------------------
# Tests: Root writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

_ROOT_POSE_METHODS = ["root_pose", "root_link_pose", "root_com_pose"]
_ROOT_VEL_METHODS = ["root_velocity", "root_link_velocity", "root_com_velocity"]


class TestRigidObjectWritersRoot:
    """Test root pose/velocity writers with all input combinations."""

    # -- index variants --

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
    def test_write_root_pose_to_sim_index(self, backend, num_instances, device, rigid_object_iface, method_suffix):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_index")

        # torch, all envs
        method(root_pose=_make_data_torch((num_instances,), device, wp.transformf))
        # torch, subset
        method(root_pose=_make_data_torch((1,), device, wp.transformf), env_ids=_make_env_ids(device, True))
        # warp, all envs
        method(root_pose=_make_data_warp((num_instances,), device, wp.transformf))
        # warp, subset
        method(root_pose=_make_data_warp((1,), device, wp.transformf), env_ids=_make_env_ids(device, True))
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_torch((num_instances,), device, wp.transformf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_warp((num_instances,), device, wp.transformf))

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_VEL_METHODS)
    def test_write_root_velocity_to_sim_index(self, backend, num_instances, device, rigid_object_iface, method_suffix):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_index")

        # torch, all envs
        method(root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf))
        # torch, subset
        method(root_velocity=_make_data_torch((1,), device, wp.spatial_vectorf), env_ids=_make_env_ids(device, True))
        # warp, all envs
        method(root_velocity=_make_data_warp((num_instances,), device, wp.spatial_vectorf))
        # warp, subset
        method(root_velocity=_make_data_warp((1,), device, wp.spatial_vectorf), env_ids=_make_env_ids(device, True))
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_torch((num_instances,), device, wp.spatial_vectorf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_warp((num_instances,), device, wp.spatial_vectorf))

    # -- mask variants --

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
    def test_write_root_pose_to_sim_mask(self, backend, num_instances, device, rigid_object_iface, method_suffix):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_mask")

        # torch, no mask (all)
        method(root_pose=_make_data_torch((num_instances,), device, wp.transformf))
        # torch, partial mask
        method(
            root_pose=_make_data_torch((num_instances,), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # warp, no mask
        method(root_pose=_make_data_warp((num_instances,), device, wp.transformf))
        # warp, partial mask
        method(
            root_pose=_make_data_warp((num_instances,), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_torch((num_instances,), device, wp.transformf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_warp((num_instances,), device, wp.transformf))

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_VEL_METHODS)
    def test_write_root_velocity_to_sim_mask(self, backend, num_instances, device, rigid_object_iface, method_suffix):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_mask")

        # torch, no mask
        method(root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf))
        # torch, partial mask
        method(
            root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # warp, no mask
        method(root_velocity=_make_data_warp((num_instances,), device, wp.spatial_vectorf))
        # warp, partial mask
        method(
            root_velocity=_make_data_warp((num_instances,), device, wp.spatial_vectorf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_torch((num_instances,), device, wp.spatial_vectorf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_warp((num_instances,), device, wp.spatial_vectorf))


# ---------------------------------------------------------------------------
# Tests: Body writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

# (method_name, kwarg_name, wp_dtype, trailing_dim)
_BODY_METHODS = [
    ("set_masses", "masses", wp.float32, 0),
    ("set_coms", "coms", wp.transformf, 7),
    ("set_inertias", "inertias", wp.float32, 9),
]


class TestRigidObjectWritersBody:
    """Test body property writers/setters with all input combinations."""

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, trailing",
        _BODY_METHODS,
        ids=[m[0] for m in _BODY_METHODS],
    )
    def test_body_writer_index(
        self, backend, num_instances, device, rigid_object_iface, method_base, kwarg, wp_dtype, trailing
    ):
        if backend == "newton" and method_base == "set_coms":
            pytest.xfail("Newton set_coms expects vec3f (position only), not transformf (pose)")
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        num_bodies = 1
        method = getattr(obj, f"{method_base}_index")

        def _torch_shape(n_envs, n_bods):
            if trailing:
                return (n_envs, n_bods, trailing)
            return (n_envs, n_bods)

        def _make_torch(n_envs, n_bods):
            shape = _torch_shape(n_envs, n_bods)
            data = torch.ones(shape, device=device, dtype=torch.float32)
            if wp_dtype == wp.transformf:
                data[..., :3] = 0.0
                data[..., 3:6] = 0.0
                data[..., 6] = 1.0
            return data

        def _make_warp(n_envs, n_bods):
            t = _make_torch(n_envs, n_bods)
            if wp_dtype == wp.transformf:
                return wp.from_torch(t.contiguous(), dtype=wp.transformf)
            return wp.from_torch(t.contiguous(), dtype=wp.float32)

        sub_b = 1  # rigid object always has 1 body
        sub_body_ids = [0]

        # torch, all envs + all bodies
        method(**{kwarg: _make_torch(num_instances, num_bodies)})
        # torch, subset
        method(
            **{
                kwarg: _make_torch(1, sub_b),
                "body_ids": sub_body_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all bodies
        method(**{kwarg: _make_warp(num_instances, num_bodies)})
        # warp, subset
        method(
            **{
                kwarg: _make_warp(1, sub_b),
                "body_ids": sub_body_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # negative: bad torch shape (extra env)
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_torch(num_instances + 1, num_bodies)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_warp(num_instances + 1, num_bodies)})

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, trailing",
        _BODY_METHODS,
        ids=[m[0] for m in _BODY_METHODS],
    )
    def test_body_writer_mask(
        self, backend, num_instances, device, rigid_object_iface, method_base, kwarg, wp_dtype, trailing
    ):
        if backend == "newton" and method_base == "set_coms":
            pytest.xfail("Newton set_coms expects vec3f (position only), not transformf (pose)")
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        num_bodies = 1
        method = getattr(obj, f"{method_base}_mask")

        def _torch_shape(n_envs, n_bods):
            if trailing:
                return (n_envs, n_bods, trailing)
            return (n_envs, n_bods)

        def _make_torch(n_envs, n_bods):
            shape = _torch_shape(n_envs, n_bods)
            data = torch.ones(shape, device=device, dtype=torch.float32)
            if wp_dtype == wp.transformf:
                data[..., :3] = 0.0
                data[..., 3:6] = 0.0
                data[..., 6] = 1.0
            return data

        def _make_warp(n_envs, n_bods):
            t = _make_torch(n_envs, n_bods)
            if wp_dtype == wp.transformf:
                return wp.from_torch(t.contiguous(), dtype=wp.transformf)
            return wp.from_torch(t.contiguous(), dtype=wp.float32)

        sub_body_sel = [0]

        # torch, no mask
        method(**{kwarg: _make_torch(num_instances, num_bodies)})
        # torch, partial env_mask + body_mask
        method(
            **{
                kwarg: _make_torch(num_instances, num_bodies),
                "body_mask": _make_item_mask(num_bodies, sub_body_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # warp, no mask
        method(**{kwarg: _make_warp(num_instances, num_bodies)})
        # warp, partial env_mask + body_mask
        method(
            **{
                kwarg: _make_warp(num_instances, num_bodies),
                "body_mask": _make_item_mask(num_bodies, sub_body_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_torch(num_instances + 1, num_bodies)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_warp(num_instances + 1, num_bodies)})


# ---------------------------------------------------------------------------
# Tests: Alias/shorthand properties
# ---------------------------------------------------------------------------


class TestRigidObjectDataAliases:
    """Test that alias properties return the same shape/dtype as their canonical counterparts."""

    @_backends
    @_default_dims
    @_default_devices
    def test_root_aliases(self, backend, num_instances, device, rigid_object_iface):
        """root_pose_w == root_link_pose_w, root_vel_w == root_com_vel_w, etc."""
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        d = obj.data

        assert d.root_pose_w.shape == d.root_link_pose_w.shape
        assert d.root_pose_w.dtype == d.root_link_pose_w.dtype
        assert d.root_pos_w.shape == d.root_link_pos_w.shape
        assert d.root_quat_w.shape == d.root_link_quat_w.shape

        assert d.root_vel_w.shape == d.root_com_vel_w.shape
        assert d.root_vel_w.dtype == d.root_com_vel_w.dtype
        assert d.root_lin_vel_w.shape == d.root_com_lin_vel_w.shape
        assert d.root_ang_vel_w.shape == d.root_com_ang_vel_w.shape

    @_backends
    @_default_dims
    @_default_devices
    def test_body_aliases(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        d = obj.data

        assert d.body_pose_w.shape == d.body_link_pose_w.shape
        assert d.body_pos_w.shape == d.body_link_pos_w.shape
        assert d.body_quat_w.shape == d.body_link_quat_w.shape
        assert d.body_vel_w.shape == d.body_com_vel_w.shape
        assert d.body_lin_vel_w.shape == d.body_com_lin_vel_w.shape
        assert d.body_ang_vel_w.shape == d.body_com_ang_vel_w.shape
