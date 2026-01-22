# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RigidObject class using mocked dependencies.

This module provides unit tests for the RigidObject class that bypass the heavy
initialization process (`_initialize_impl`) which requires a USD stage and real
simulation infrastructure.

The key technique is to:
1. Create the RigidObject object without calling __init__ using object.__new__
2. Manually set up the required internal state with mock objects
3. Test individual methods in isolation

This allows testing the mathematical operations and return values without
requiring full simulation integration.
"""

from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

import pytest
import warp as wp

# Import mock classes from common test utilities
from common.mock_newton import MockNewtonArticulationView, MockNewtonModel
from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject
from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab_newton.kernels import vec13f

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

# TODO: Move these functions to the test utils so they can't be changed in the future.
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_inv

# Initialize Warp
wp.init()


##
# Test Factory - Creates RigidObject instances without full initialization
##


def create_test_rigid_object(
    num_instances: int = 2,
    device: str = "cuda:0",
    body_names: list[str] | None = None,
) -> tuple[RigidObject, MockNewtonArticulationView, MagicMock]:
    """Create a test RigidObject instance with mocked dependencies.

    This factory bypasses _initialize_impl and manually sets up the internal state,
    allowing unit testing of individual methods without requiring USD/simulation.

    Args:
        num_instances: Number of environment instances.
        device: Device to use ("cpu" or "cuda:0").
        body_names: Custom body names. Defaults to ["body_0"].

    Returns:
        A tuple of (rigid_object, mock_view, mock_newton_manager).
    """
    # Hardcoded values since RigidObject has no joints and a single body.
    num_joints = 0
    num_bodies = 1
    is_fixed_base = False
    joint_names = []

    # Generate default names if not provided
    if body_names is None:
        body_names = [f"body_{i}" for i in range(num_bodies)]

    # Create the RigidObject without calling __init__
    rigid_object = object.__new__(RigidObject)

    # Set up the configuration
    rigid_object.cfg = RigidObjectCfg(
        prim_path="/World/Object",
    )

    # Set up the mock view with all parameters
    mock_view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=num_joints,
        device=device,
        is_fixed_base=is_fixed_base,
        joint_names=joint_names,
        body_names=body_names,
    )
    mock_view.set_mock_data()

    # Set the view on the rigid object (using object.__setattr__ to bypass type checking)
    object.__setattr__(rigid_object, "_root_view", mock_view)
    object.__setattr__(rigid_object, "_device", device)

    # Create mock NewtonManager
    mock_newton_manager = MagicMock()
    mock_model = MockNewtonModel()
    mock_state = MagicMock()
    mock_control = MagicMock()
    mock_newton_manager.get_model.return_value = mock_model
    mock_newton_manager.get_state_0.return_value = mock_state
    mock_newton_manager.get_control.return_value = mock_control
    mock_newton_manager.get_dt.return_value = 0.01

    # Create RigidObjectData with the mock view
    with patch("isaaclab_newton.assets.rigid_object.rigid_object_data.NewtonManager", mock_newton_manager):
        data = RigidObjectData(mock_view, device)
        object.__setattr__(rigid_object, "_data", data)

    # Call _create_buffers() to initialize temp buffers and wrench composers
    rigid_object._create_buffers()

    return rigid_object, mock_view, mock_newton_manager


##
# Test Fixtures
##


@pytest.fixture
def mock_newton_manager():
    """Create mock NewtonManager with necessary methods."""
    mock_model = MockNewtonModel()
    mock_state = MagicMock()
    mock_control = MagicMock()

    # Patch where NewtonManager is used (in the rigid object module)
    with patch("isaaclab_newton.assets.rigid_object.rigid_object.NewtonManager") as MockManager:
        MockManager.get_model.return_value = mock_model
        MockManager.get_state_0.return_value = mock_state
        MockManager.get_control.return_value = mock_control
        MockManager.get_dt.return_value = 0.01
        yield MockManager


@pytest.fixture
def test_rigid_object():
    """Create a test rigid object with default parameters."""
    rigid_object, mock_view, mock_manager = create_test_rigid_object()
    yield rigid_object, mock_view, mock_manager


##
# Test Cases -- Properties
##


class TestProperties:
    """Tests for RigidObject properties.

    Tests the following properties:
    - data
    - num_instances
    - num_bodies
    - body_names
    """

    @pytest.mark.parametrize("num_instances", [1, 2, 4])
    def test_num_instances(self, num_instances: int):
        """Test the num_instances property returns correct count."""
        rigid_object, _, _ = create_test_rigid_object(num_instances=num_instances)
        assert rigid_object.num_instances == num_instances

    def test_num_bodies(self):
        """Test the num_bodies property returns correct count."""
        rigid_object, _, _ = create_test_rigid_object()
        assert rigid_object.num_bodies == 1

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def testdata_property(self, device: str):
        """Test that data property returns RigidObjectData instance."""
        rigid_object, _, _ = create_test_rigid_object(device=device)
        assert isinstance(rigid_object.data, RigidObjectData)

    def test_body_names(self):
        """Test that body_names returns the correct names."""
        custom_names = ["base"]
        rigid_object, _, _ = create_test_rigid_object(
            body_names=custom_names,
        )
        assert rigid_object.body_names == custom_names


##
# Test Cases -- Reset
##


class TestReset:
    """Tests for reset method."""

    def test_reset(self):
        """Test that reset method works properly."""
        rigid_object, _, _ = create_test_rigid_object()
        rigid_object.set_external_force_and_torque(
            forces=torch.ones(rigid_object.num_instances, rigid_object.num_bodies, 3),
            torques=torch.ones(rigid_object.num_instances, rigid_object.num_bodies, 3),
            env_ids=slice(None),
            body_ids=slice(None),
            body_mask=None,
            env_mask=None,
            is_global=False,
        )
        assert wp.to_torch(rigid_object.permanent_wrench_composer.composed_force).allclose(
            torch.ones_like(wp.to_torch(rigid_object.permanent_wrench_composer.composed_force))
        )
        rigid_object.reset()
        assert wp.to_torch(rigid_object.permanent_wrench_composer.composed_force).allclose(
            torch.zeros_like(wp.to_torch(rigid_object.permanent_wrench_composer.composed_force))
        )


##
# Test Cases -- Write Data to Sim. Skipped, this is mostly an integration test.
##


##
# Test Cases -- Update
##


class TestUpdate:
    """Tests for update method."""

    def test_update(self):
        """Test that update method updates the simulation timestamp properly."""
        rigid_object, _, _ = create_test_rigid_object()
        rigid_object.update(dt=0.01)
        assert rigid_object.data._sim_timestamp == 0.01


##
# Test Cases -- Finders
##


class TestFinders:
    """Tests for finder methods."""

    @pytest.mark.parametrize(
        "body_names",
        [["body_0"], ["body_3"], ["body_1"], "body_.*"],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies(self, body_names: list[str], device: str):
        """Test that find_bodies method works properly."""
        rigid_object, _, _ = create_test_rigid_object(device=device)
        if body_names == ["body_0"]:
            mask, names, indices = rigid_object.find_bodies(body_names)
            mask_ref = torch.zeros((1,), dtype=torch.bool, device=device)
            mask_ref[0] = True
            assert names == ["body_0"]
            assert indices == [0]
            assert wp.to_torch(mask).allclose(mask_ref)
        elif body_names == ["body_3"] or body_names == ["body_1"]:
            with pytest.raises(ValueError):
                rigid_object.find_bodies(body_names)
        elif body_names == "body_.*":
            mask, names, indices = rigid_object.find_bodies(body_names)
            mask_ref = torch.ones((1,), dtype=torch.bool, device=device)
            assert names == ["body_0"]
            assert indices == [0]
            assert wp.to_torch(mask).allclose(mask_ref)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_body_with_preserve_order(self, device: str):
        """Test that find_bodies method works properly with preserve_order."""
        rigid_object, _, _ = create_test_rigid_object(device=device)
        mask, names, indices = rigid_object.find_bodies(["body_0"], preserve_order=True)
        assert names == ["body_0"]
        assert indices == [0]
        mask_ref = torch.zeros((1,), dtype=torch.bool, device=device)
        mask_ref[0] = True
        assert wp.to_torch(mask).allclose(mask_ref)


##
# Test Cases -- State Writers
##


class TestStateWriters:
    """Tests for state writing methods."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0], torch.tensor([0, 1, 2], dtype=torch.int32)])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_state_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_state_to_sim method works properly."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                rigid_object.write_root_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_state_w).allclose(data, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                data = torch.rand((len(env_ids), 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                rigid_object.write_root_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_state_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                rigid_object.write_root_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_state_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)
            else:
                # Update envs 0, 1, 2
                data = torch.rand((3, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                env_ids = env_ids.to(device=device)
                rigid_object.write_root_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_state_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_state_to_sim_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_state_to_sim method works properly."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, rigid_object.num_bodies, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                rigid_object.write_root_state_to_sim(wp.from_torch(data, dtype=vec13f))
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_state_w).allclose(data, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                data = torch.rand((len(env_ids), 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = torch.ones((num_instances, 13), device=device)
                mask_warp = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                mask_warp[env_ids] = True
                data_warp[env_ids] = data
                data_warp = wp.from_torch(data_warp, dtype=vec13f)
                mask_warp = wp.from_torch(mask_warp, dtype=wp.bool)
                # Write to simulation
                rigid_object.write_root_state_to_sim(data_warp, env_mask=mask_warp)
                # Check results
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_state_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)
            else:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = wp.from_torch(data.clone(), dtype=vec13f)
                mask_warp = wp.ones((num_instances,), dtype=wp.bool, device=device)
                rigid_object.write_root_state_to_sim(data_warp, env_mask=mask_warp)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_state_w).allclose(data, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0], torch.tensor([0, 1, 2], dtype=torch.int32)])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_state_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_state_to_sim method works properly."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                rigid_object.write_root_com_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_com_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                # Update selected envs
                data = torch.rand((len(env_ids), 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                rigid_object.write_root_com_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_com_vel_w)[env_ids].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids].allclose(
                    data[:, :7], atol=1e-6, rtol=1e-6
                )
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                rigid_object.write_root_com_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_com_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            else:
                # Update envs 0, 1, 2
                data = torch.rand((3, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                env_ids = env_ids.to(device=device)
                rigid_object.write_root_com_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_com_vel_w)[env_ids].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids].allclose(
                    data[:, :7], atol=1e-6, rtol=1e-6
                )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_state_to_sim_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_state_to_sim method works properly with warp arrays."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                rigid_object.write_root_com_state_to_sim(wp.from_torch(data, dtype=vec13f))
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_com_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                data = torch.rand((len(env_ids), 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = torch.ones((num_instances, 13), device=device)
                mask_warp = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                mask_warp[env_ids] = True
                data_warp[env_ids] = data
                data_warp = wp.from_torch(data_warp, dtype=vec13f)
                mask_warp = wp.from_torch(mask_warp, dtype=wp.bool)
                # Write to simulation
                rigid_object.write_root_com_state_to_sim(data_warp, env_mask=mask_warp)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_com_vel_w)[env_ids].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids].allclose(
                    data[:, :7], atol=1e-6, rtol=1e-6
                )
            else:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = wp.from_torch(data.clone(), dtype=vec13f)
                mask_warp = wp.ones((num_instances,), dtype=wp.bool, device=device)
                # Generate reference data
                rigid_object.write_root_com_state_to_sim(data_warp, env_mask=mask_warp)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_com_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_state_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_state_to_sim method works properly."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                rigid_object.write_root_link_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_link_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                # Update selected envs
                data = torch.rand((len(env_ids), 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                rigid_object.write_root_link_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(rigid_object.data.root_link_pose_w)[env_ids].allclose(
                    data[:, :7], atol=1e-6, rtol=1e-6
                )
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                rigid_object.write_root_link_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_link_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            else:
                # Update envs 0, 1, 2
                data = torch.rand((3, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                env_ids = env_ids.to(device=device)
                rigid_object.write_root_link_state_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(rigid_object.data.root_link_pose_w)[env_ids].allclose(
                    data[:, :7], atol=1e-6, rtol=1e-6
                )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_state_to_sim_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_state_to_sim method works properly with warp arrays."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                rigid_object.write_root_link_state_to_sim(wp.from_torch(data, dtype=vec13f))
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_link_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                data = torch.rand((len(env_ids), 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = torch.ones((num_instances, 13), device=device)
                mask_warp = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                mask_warp[env_ids] = True
                data_warp[env_ids] = data
                data_warp = wp.from_torch(data_warp, dtype=vec13f)
                mask_warp = wp.from_torch(mask_warp, dtype=wp.bool)
                # Generate reference data
                data_ref = torch.zeros((num_instances, 13), device=device)
                data_ref[env_ids] = data
                # Write to simulation
                rigid_object.write_root_link_state_to_sim(data_warp, env_mask=mask_warp)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids, :].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(rigid_object.data.root_link_pose_w)[env_ids].allclose(
                    data[:, :7], atol=1e-6, rtol=1e-6
                )
            else:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = wp.from_torch(data.clone(), dtype=vec13f)
                mask_warp = wp.ones((num_instances,), dtype=wp.bool, device=device)
                # Generate reference data
                rigid_object.write_root_link_state_to_sim(data_warp, env_mask=mask_warp)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_link_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)


class TestVelocityWriters:
    """Tests for velocity writing methods.

    Tests methods like:
    - write_root_link_velocity_to_sim
    - write_root_com_velocity_to_sim
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_velocity_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_velocity_to_sim method works properly."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]

        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                # Write to simulation
                rigid_object.write_root_link_velocity_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(rigid_object.data.root_link_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(rigid_object.data.root_link_quat_w)
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[:, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(root_com_velocity, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                # Update selected envs
                data = torch.rand((len(env_ids), 6), device=device)
                # Write to simulation
                rigid_object.write_root_link_velocity_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(rigid_object.data.root_link_quat_w)[env_ids]
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[env_ids, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(rigid_object.data.root_com_vel_w)[env_ids, :].allclose(
                    root_com_velocity, atol=1e-6, rtol=1e-6
                )
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                rigid_object.write_root_link_velocity_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(rigid_object.data.root_link_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(rigid_object.data.root_link_quat_w)
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[:, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(root_com_velocity, atol=1e-6, rtol=1e-6)
            else:
                # Update envs 0, 1, 2
                data = torch.rand((3, 6), device=device)
                env_ids = env_ids.to(device=device)
                rigid_object.write_root_link_velocity_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(rigid_object.data.root_link_quat_w)[env_ids]
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[env_ids, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(rigid_object.data.root_com_vel_w)[env_ids, :].allclose(
                    root_com_velocity, atol=1e-6, rtol=1e-6
                )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_velocity_to_sim_with_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_velocity_to_sim method works properly with warp arrays."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]

        # Set a non-zero body CoM offset
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                rigid_object.write_root_link_velocity_to_sim(wp.from_torch(data, dtype=wp.spatial_vectorf))
                assert wp.to_torch(rigid_object.data.root_link_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(rigid_object.data.root_link_quat_w)
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[:, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(root_com_velocity, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                data = torch.rand((len(env_ids), 6), device=device)
                # Generate warp data
                data_warp = torch.ones((num_instances, 6), device=device)
                mask_warp = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                mask_warp[env_ids] = True
                data_warp[env_ids] = data
                data_warp = wp.from_torch(data_warp, dtype=wp.spatial_vectorf)
                mask_warp = wp.from_torch(mask_warp, dtype=wp.bool)
                # Write to simulation
                rigid_object.write_root_link_velocity_to_sim(data_warp, env_mask=mask_warp)
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(rigid_object.data.root_link_quat_w)[env_ids]
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[env_ids, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(rigid_object.data.root_com_vel_w)[env_ids, :].allclose(
                    root_com_velocity, atol=1e-6, rtol=1e-6
                )
            else:
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                # Generate warp data
                data_warp = wp.from_torch(data.clone(), dtype=wp.spatial_vectorf)
                mask_warp = wp.ones((num_instances,), dtype=wp.bool, device=device)
                # Generate reference data
                rigid_object.write_root_link_velocity_to_sim(data_warp, env_mask=mask_warp)
                assert wp.to_torch(rigid_object.data.root_link_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(rigid_object.data.root_link_quat_w)
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[:, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(root_com_velocity, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_velocity_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_state_to_sim method works properly."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]

        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                # Write to simulation
                rigid_object.write_root_com_velocity_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(rigid_object.data.root_link_vel_w)[:, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                # Update selected envs
                data = torch.rand((len(env_ids), 6), device=device)
                # Write to simulation
                rigid_object.write_root_com_velocity_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(rigid_object.data.root_com_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                rigid_object.write_root_com_velocity_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(rigid_object.data.root_link_vel_w)[:, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
            else:
                # Update envs 0, 1, 2
                data = torch.rand((3, 6), device=device)
                env_ids = env_ids.to(device=device)
                rigid_object.write_root_com_velocity_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(rigid_object.data.root_com_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_velocity_to_sim_with_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_velocity_to_sim method works properly with warp arrays."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]

        # Set a non-zero body CoM offset
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                rigid_object.write_root_com_velocity_to_sim(wp.from_torch(data, dtype=wp.spatial_vectorf))
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(rigid_object.data.root_link_vel_w)[:, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                data = torch.rand((len(env_ids), 6), device=device)
                # Generate warp data
                data_warp = torch.ones((num_instances, 6), device=device)
                mask_warp = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                mask_warp[env_ids] = True
                data_warp[env_ids] = data
                data_warp = wp.from_torch(data_warp, dtype=wp.spatial_vectorf)
                mask_warp = wp.from_torch(mask_warp, dtype=wp.bool)
                # Write to simulation
                rigid_object.write_root_com_velocity_to_sim(data_warp, env_mask=mask_warp)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(rigid_object.data.root_com_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
            else:
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                # Generate warp data
                data_warp = wp.from_torch(data.clone(), dtype=wp.spatial_vectorf)
                mask_warp = wp.ones((num_instances,), dtype=wp.bool, device=device)
                # Generate reference data
                rigid_object.write_root_com_velocity_to_sim(data_warp, env_mask=mask_warp)
                assert rigid_object.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(rigid_object.data.root_link_vel_w)[:, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_link_vel_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(rigid_object.data.root_com_vel_w).allclose(data, atol=1e-6, rtol=1e-6)


class TestPoseWriters:
    """Tests for pose writing methods.

    Tests methods like:
    - write_root_link_pose_to_sim
    - write_root_com_pose_to_sim
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_pose_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_pose_to_sim method works properly."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the pose transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if env_ids is None:
                # Update all envs
                data = torch.rand((num_instances, 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                rigid_object.write_root_link_pose_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_pose_w).allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(rigid_object.data.root_com_pose_w)[:, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                # Update selected envs
                data = torch.rand((len(env_ids), 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                rigid_object.write_root_link_pose_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_pose_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                rigid_object.write_root_link_pose_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_pose_w).allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(rigid_object.data.root_com_pose_w)[:, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
            else:
                # Update envs 0, 1, 2
                data = torch.rand((3, 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                env_ids = env_ids.to(device=device)
                rigid_object.write_root_link_pose_to_sim(data, env_ids=env_ids)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_pose_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_pose_to_sim_with_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_pose_to_sim method works properly with warp arrays."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if (env_ids is None) or (isinstance(env_ids, slice)):
                data = torch.rand((num_instances, 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Update all envs
                rigid_object.write_root_link_pose_to_sim(wp.from_torch(data, dtype=wp.transformf))
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_pose_w).allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(rigid_object.data.root_com_pose_w)[:, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
            else:
                data = torch.rand((len(env_ids), 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = torch.ones((num_instances, 7), device=device)
                mask_warp = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                mask_warp[env_ids] = True
                data_warp[env_ids] = data
                data_warp = wp.from_torch(data_warp, dtype=wp.transformf)
                mask_warp = wp.from_torch(mask_warp, dtype=wp.bool)
                # Write to simulation
                rigid_object.write_root_link_pose_to_sim(data_warp, env_mask=mask_warp)
                assert rigid_object.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(rigid_object.data.root_link_pose_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_pose_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_pose_to_sim method works properly."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if (env_ids is None) or (isinstance(env_ids, slice)):
                data = torch.rand((num_instances, 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                rigid_object.write_root_com_pose_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(rigid_object.data.root_com_pose_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[:, 0, :]
                com_quat_b = wp.to_torch(rigid_object.data.body_com_quat_b)[:, 0, :]
                # transform input CoM pose to link frame
                root_link_pos, root_link_quat = combine_frame_transforms(
                    data[..., :3],
                    data[..., 3:7],
                    quat_apply(quat_inv(com_quat_b), -com_pos_b),
                    quat_inv(com_quat_b),
                )
                root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)
                assert wp.to_torch(rigid_object.data.root_link_pose_w).allclose(root_link_pose, atol=1e-6, rtol=1e-6)
            else:
                if isinstance(env_ids, torch.Tensor):
                    env_ids = env_ids.to(device=device)
                # Update selected envs
                data = torch.rand((len(env_ids), 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                rigid_object.write_root_com_pose_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[env_ids, 0, :]
                com_quat_b = wp.to_torch(rigid_object.data.body_com_quat_b)[env_ids, 0, :]
                # transform input CoM pose to link frame
                root_link_pos, root_link_quat = combine_frame_transforms(
                    data[..., :3],
                    data[..., 3:7],
                    quat_apply(quat_inv(com_quat_b), -com_pos_b),
                    quat_inv(com_quat_b),
                )
                root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)
                assert wp.to_torch(rigid_object.data.root_link_pose_w)[env_ids, :].allclose(
                    root_link_pose, atol=1e-6, rtol=1e-6
                )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_pose_to_sim_with_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_pose_to_sim method works properly with warp arrays."""
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, 1, 3)
        root_transforms = torch.rand((num_instances, 7), device=device)
        root_transforms[:, 3:7] = torch.nn.functional.normalize(root_transforms[:, 3:7], p=2.0, dim=-1)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_transforms, dtype=wp.transformf),
            body_com_pos=wp.from_torch(body_comdata.clone(), dtype=wp.vec3f),
        )
        for _ in range(5):
            if (env_ids is None) or (isinstance(env_ids, slice)):
                data = torch.rand((num_instances, 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                rigid_object.write_root_com_pose_to_sim(wp.from_torch(data, dtype=wp.transformf))
                assert wp.to_torch(rigid_object.data.root_com_pose_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[:, 0, :]
                com_quat_b = wp.to_torch(rigid_object.data.body_com_quat_b)[:, 0, :]
                # transform input CoM pose to link frame
                root_link_pos, root_link_quat = combine_frame_transforms(
                    data[..., :3],
                    data[..., 3:7],
                    quat_apply(quat_inv(com_quat_b), -com_pos_b),
                    quat_inv(com_quat_b),
                )
                root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)
                assert wp.to_torch(rigid_object.data.root_link_pose_w).allclose(root_link_pose, atol=1e-6, rtol=1e-6)
            else:
                data = torch.rand((len(env_ids), 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = torch.ones((num_instances, 7), device=device)
                mask_warp = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                mask_warp[env_ids] = True
                data_warp[env_ids] = data
                data_warp = wp.from_torch(data_warp, dtype=wp.transformf)
                mask_warp = wp.from_torch(mask_warp, dtype=wp.bool)
                # Write to simulation
                rigid_object.write_root_com_pose_to_sim(data_warp, env_mask=mask_warp)
                assert wp.to_torch(rigid_object.data.root_com_pose_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                com_pos_b = wp.to_torch(rigid_object.data.body_com_pos_b)[env_ids, 0, :]
                com_quat_b = wp.to_torch(rigid_object.data.body_com_quat_b)[env_ids, 0, :]
                # transform input CoM pose to link frame
                root_link_pos, root_link_quat = combine_frame_transforms(
                    data[..., :3],
                    data[..., 3:7],
                    quat_apply(quat_inv(com_quat_b), -com_pos_b),
                    quat_inv(com_quat_b),
                )
                root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)
                assert wp.to_torch(rigid_object.data.root_link_pose_w)[env_ids, :].allclose(
                    root_link_pose, atol=1e-6, rtol=1e-6
                )


##
# Test Cases - Setters.
##


class TestSettersBodiesMassCoMInertia:
    """Tests for setter methods that set body mass, center of mass, and inertia.

    Tests methods:
    - set_masses
    - set_coms
    - set_inertias
    """

    def generic_test_property_writer_torch(
        self,
        device: str,
        env_ids,
        body_ids,
        num_instances: int,
        writer_function_name: str,
        property_name: str,
        dtype: type = wp.float32,
    ):
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if (body_ids is not None) and (not isinstance(body_ids, slice)):
            body_ids = [0]

        writer_function = getattr(rigid_object, writer_function_name)
        if dtype == wp.float32:
            ndims = tuple()
        elif dtype == wp.vec3f:
            ndims = (3,)
        elif dtype == wp.mat33f:
            ndims = (
                3,
                3,
            )
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        for _ in range(5):
            if (env_ids is None) or (isinstance(env_ids, slice)):
                if (body_ids is None) or (isinstance(body_ids, slice)):
                    # All envs and joints
                    data1 = torch.rand((num_instances, 1, *ndims), device=device)
                    writer_function(data1, env_ids=env_ids, body_ids=body_ids)
                    property_data = getattr(rigid_object.data, property_name)
                    assert wp.to_torch(property_data).allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected bodies
                    data1 = torch.rand((num_instances, len(body_ids), *ndims), device=device)
                    data_ref = torch.zeros((num_instances, 1, *ndims), device=device)
                    data_ref[:, body_ids] = data1
                    writer_function(data1, env_ids=env_ids, body_ids=body_ids)
                    property_data = getattr(rigid_object.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
            else:
                if (body_ids is None) or (isinstance(body_ids, slice)):
                    # Selected envs and all bodies
                    data1 = torch.rand((len(env_ids), 1, *ndims), device=device)
                    data_ref = torch.zeros((num_instances, 1, *ndims), device=device)
                    data_ref[env_ids, :] = data1
                    writer_function(data1, env_ids=env_ids, body_ids=body_ids)
                    property_data = getattr(rigid_object.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and bodies
                    data1 = torch.rand((len(env_ids), len(body_ids), *ndims), device=device)
                    writer_function(data1, env_ids=env_ids, body_ids=body_ids)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    data_ref = torch.zeros((num_instances, 1, *ndims), device=device)
                    data_ref[env_ids_, body_ids] = data1
                    property_data = getattr(rigid_object.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)

    def generic_test_property_writer_warp(
        self,
        device: str,
        env_ids,
        body_ids,
        num_instances: int,
        writer_function_name: str,
        property_name: str,
        dtype: type = wp.float32,
    ):
        rigid_object, mock_view, _ = create_test_rigid_object(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        body_ids = [0]

        writer_function = getattr(rigid_object, writer_function_name)
        if dtype == wp.float32:
            ndims = tuple()
        elif dtype == wp.vec3f:
            ndims = (3,)
        elif dtype == wp.mat33f:
            ndims = (
                3,
                3,
            )
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        for _ in range(5):
            if env_ids is None:
                if body_ids is None:
                    # All envs and joints
                    data1 = torch.rand((num_instances, 1, *ndims), device=device)
                    data1_warp = wp.from_torch(data1, dtype=dtype)
                    writer_function(data1_warp, env_mask=None, body_mask=None)
                    property_data = getattr(rigid_object.data, property_name)
                    assert wp.to_torch(property_data).allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    data1 = torch.rand((num_instances, 1, *ndims), device=device)
                    data1_warp = torch.ones((num_instances, 1, *ndims), device=device)
                    data1_warp[:] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=dtype)
                    body_mask = torch.zeros((1,), dtype=torch.bool, device=device)
                    body_mask[0] = True
                    body_mask = wp.from_torch(body_mask, dtype=wp.bool)
                    data_ref = torch.zeros((num_instances, 1, *ndims), device=device)
                    data_ref[:] = data1
                    writer_function(data1_warp, env_mask=None, body_mask=body_mask)
                    property_data = getattr(rigid_object.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
            else:
                if body_ids is None:
                    # Selected envs and all joints
                    data1 = torch.rand((1, 1, *ndims), device=device)
                    data1_warp = torch.ones((num_instances, 1, *ndims), device=device)
                    data1_warp[env_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=dtype)
                    data_ref = torch.zeros((num_instances, 1, *ndims), device=device)
                    data_ref[env_ids, :] = data1
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    writer_function(data1_warp, env_mask=env_mask, body_mask=None)
                    property_data = getattr(rigid_object.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    data1 = torch.rand((1, 1, *ndims), device=device)
                    data1_warp = torch.ones((num_instances, 1, *ndims), device=device)
                    data1_warp[env_ids_, 0] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=dtype)
                    data_ref = torch.zeros((num_instances, 1, *ndims), device=device)
                    data_ref[env_ids_, 0] = data1
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    body_mask = torch.zeros((1,), dtype=torch.bool, device=device)
                    body_mask[0] = True
                    body_mask = wp.from_torch(body_mask, dtype=wp.bool)
                    writer_function(data1_warp, env_mask=env_mask, body_mask=body_mask)
                    property_data = getattr(rigid_object.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_masses_to_sim_torch(self, device: str, env_ids, body_ids, num_instances: int):
        self.generic_test_property_writer_torch(
            device, env_ids, body_ids, num_instances, "set_masses", "body_mass", dtype=wp.float32
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_masses_to_sim_warp(self, device: str, env_ids, body_ids, num_instances: int):
        self.generic_test_property_writer_warp(
            device, env_ids, body_ids, num_instances, "set_masses", "body_mass", dtype=wp.float32
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_coms_to_sim_torch(self, device: str, env_ids, body_ids, num_instances: int):
        self.generic_test_property_writer_torch(
            device, env_ids, body_ids, num_instances, "set_coms", "body_com_pos_b", dtype=wp.vec3f
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_coms_to_sim_warp(self, device: str, env_ids, body_ids, num_instances: int):
        self.generic_test_property_writer_warp(
            device, env_ids, body_ids, num_instances, "set_coms", "body_com_pos_b", dtype=wp.vec3f
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_inertias_to_sim_torch(self, device: str, env_ids, body_ids, num_instances: int):
        self.generic_test_property_writer_torch(
            device, env_ids, body_ids, num_instances, "set_inertias", "body_inertia", dtype=wp.mat33f
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_inertias_to_sim_warp(self, device: str, env_ids, body_ids, num_instances: int):
        self.generic_test_property_writer_warp(
            device, env_ids, body_ids, num_instances, "set_inertias", "body_inertia", dtype=wp.mat33f
        )


# TODO: Implement these tests once the Wrench Composers made it to main IsaacLab.
class TestSettersExternalWrench:
    """Tests for setter methods that set external wrench.

    Tests methods:
    - set_external_force_and_torque
    """

    @pytest.mark.skip(reason="Not implemented")
    def test_external_force_and_torque_to_sim_torch(
        self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int
    ):
        raise NotImplementedError()

    @pytest.mark.skip(reason="Not implemented")
    def test_external_force_and_torque_to_sim_warp(
        self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int
    ):
        raise NotImplementedError()


class TestCreateBuffers:
    """Tests for _create_buffers method.

    Tests that the buffers are created correctly:
    - _ALL_INDICES tensor contains correct indices for varying number of environments
    - soft_joint_pos_limits are correctly computed based on soft_joint_pos_limit_factor
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("num_instances", [1, 2, 4, 10, 100])
    def test_create_buffers_all_indices(self, device: str, num_instances: int):
        """Test that _ALL_INDICES contains correct indices for varying number of environments."""
        rigid_object, mock_view, _ = create_test_rigid_object(
            num_instances=num_instances,
            device=device,
        )

        # Call _create_buffers
        rigid_object._create_buffers()

        # Verify _ALL_INDICES
        expected_indices = torch.arange(num_instances, dtype=torch.long, device=device)
        assert rigid_object._ALL_INDICES.shape == (num_instances,)
        assert rigid_object._ALL_INDICES.dtype == torch.long
        assert rigid_object._ALL_INDICES.device.type == device.split(":")[0]
        torch.testing.assert_close(rigid_object._ALL_INDICES, expected_indices)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_create_buffers_single_environment(self, device: str):
        """Test _create_buffers with a single environment."""
        num_instances = 1
        rigid_object, mock_view, _ = create_test_rigid_object(
            num_instances=num_instances,
            device=device,
        )

        # Call _create_buffers
        rigid_object._create_buffers()

        # Verify _ALL_INDICES has single element
        assert rigid_object._ALL_INDICES.shape == (1,)
        assert rigid_object._ALL_INDICES[0].item() == 0

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_create_buffers_large_number_of_environments(self, device: str):
        """Test _create_buffers with a large number of environments."""
        num_instances = 1024
        rigid_object, mock_view, _ = create_test_rigid_object(
            num_instances=num_instances,
            device=device,
        )

        # Call _create_buffers
        rigid_object._create_buffers()

        # Verify _ALL_INDICES
        expected_indices = torch.arange(num_instances, dtype=torch.long, device=device)
        assert rigid_object._ALL_INDICES.shape == (num_instances,)
        torch.testing.assert_close(rigid_object._ALL_INDICES, expected_indices)


class TestProcessCfg:
    """Tests for _process_cfg method.

    Tests that the configuration processing correctly:
    - Uses quaternion in (x, y, z, w) format for default root pose
    - Sets default root velocity from lin_vel and ang_vel
    - Sets default joint positions from joint_pos dict with pattern matching
    - Sets default joint velocities from joint_vel dict with pattern matching
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_default_root_pose(self, device: str):
        """Test that _process_cfg correctly converts quaternion format for root pose."""
        num_instances = 2
        rigid_object, mock_view, _ = create_test_rigid_object(
            num_instances=num_instances,
            device=device,
        )

        # Set up init_state with specific position and rotation
        # Rotation is in (x, y, z, w) format in the config
        rigid_object.cfg.init_state.pos = (1.0, 2.0, 3.0)
        rigid_object.cfg.init_state.rot = (0.0, 0.707, 0.0, 0.707)  # x, y, z, w

        # Call _process_cfg
        rigid_object._process_cfg()

        # Verify the default root pose
        # Expected: position (1, 2, 3) + quaternion in (x, y, z, w) = (0, 0.707, 0, 0.707)
        expected_pose = torch.tensor(
            [[1.0, 2.0, 3.0, 0.0, 0.707, 0.0, 0.707]] * num_instances,
            device=device,
        )
        result = wp.to_torch(rigid_object.data.default_root_pose)
        assert result.allclose(expected_pose, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_default_root_velocity(self, device: str):
        """Test that _process_cfg correctly sets default root velocity."""
        num_instances = 2
        rigid_object, mock_view, _ = create_test_rigid_object(
            num_instances=num_instances,
            device=device,
        )

        # Set up init_state with specific velocities
        rigid_object.cfg.init_state.lin_vel = (1.0, 2.0, 3.0)
        rigid_object.cfg.init_state.ang_vel = (0.1, 0.2, 0.3)

        # Call _process_cfg
        rigid_object._process_cfg()

        # Verify the default root velocity
        # Expected: lin_vel + ang_vel = (1, 2, 3, 0.1, 0.2, 0.3)
        expected_vel = torch.tensor(
            [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]] * num_instances,
            device=device,
        )
        result = wp.to_torch(rigid_object.data.default_root_vel)
        assert result.allclose(expected_vel, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_identity_quaternion(self, device: str):
        """Test that _process_cfg correctly handles identity quaternion."""
        num_instances = 2
        rigid_object, mock_view, _ = create_test_rigid_object(
            num_instances=num_instances,
            device=device,
        )

        # Set up init_state with identity quaternion (x=0, y=0, z=0, w=1)
        rigid_object.cfg.init_state.pos = (0.0, 0.0, 0.0)
        rigid_object.cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # Identity: x, y, z, w

        # Call _process_cfg
        rigid_object._process_cfg()

        # Verify the default root pose
        # Expected: position (0, 0, 0) + quaternion in (x, y, z, w) = (0, 0, 0, 1)
        expected_pose = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * num_instances,
            device=device,
        )
        result = wp.to_torch(rigid_object.data.default_root_pose)
        assert result.allclose(expected_pose, atol=1e-5, rtol=1e-5)


##
# Main
##

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
