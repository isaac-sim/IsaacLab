# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Articulation class using mocked dependencies.

This module provides unit tests for the Articulation class that bypass the heavy
initialization process (`_initialize_impl`) which requires a USD stage and real
simulation infrastructure.

The key technique is to:
1. Create the Articulation object without calling __init__ using object.__new__
2. Manually set up the required internal state with mock objects
3. Test individual methods in isolation

This allows testing the mathematical operations and return values without
requiring full simulation integration.
"""

from __future__ import annotations
from enum import nonmember

import torch
from unittest.mock import MagicMock, patch

import pytest
import warp as wp
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.actuators import ActuatorBaseCfg

from isaaclab.utils.math import quat_apply, quat_inv, combine_frame_transforms

from isaaclab_newton.kernels import vec13f

# Import mock classes from shared module
from .mock_interface import MockNewtonArticulationView, MockNewtonModel

# Initialize Warp
wp.init()


##
# Test Factory - Creates Articulation instances without full initialization
##


def create_test_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
    is_fixed_base: bool = False,
    joint_names: list[str] | None = None,
    body_names: list[str] | None = None,
    soft_joint_pos_limit_factor: float = 1.0,
) -> tuple[Articulation, MockNewtonArticulationView, MagicMock]:
    """Create a test Articulation instance with mocked dependencies.

    This factory bypasses _initialize_impl and manually sets up the internal state,
    allowing unit testing of individual methods without requiring USD/simulation.

    Args:
        num_instances: Number of environment instances.
        num_joints: Number of joints in the articulation.
        num_bodies: Number of bodies in the articulation.
        device: Device to use ("cpu" or "cuda:0").
        is_fixed_base: Whether the articulation is fixed-base.
        joint_names: Custom joint names. Defaults to ["joint_0", "joint_1", ...].
        body_names: Custom body names. Defaults to ["body_0", "body_1", ...].
        soft_joint_pos_limit_factor: Soft joint position limit factor.

    Returns:
        A tuple of (articulation, mock_view, mock_newton_manager).
    """
    # Generate default names if not provided
    if joint_names is None:
        joint_names = [f"joint_{i}" for i in range(num_joints)]
    if body_names is None:
        body_names = [f"body_{i}" for i in range(num_bodies)]

    # Create the Articulation without calling __init__
    articulation = object.__new__(Articulation)

    # Set up the configuration
    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
        actuators={},
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

    # Set the view on the articulation (using object.__setattr__ to bypass type checking)
    object.__setattr__(articulation, "_root_view", mock_view)
    object.__setattr__(articulation, "_device", device)

    # Create mock NewtonManager
    mock_newton_manager = MagicMock()
    mock_model = MockNewtonModel()
    mock_state = MagicMock()
    mock_control = MagicMock()
    mock_newton_manager.get_model.return_value = mock_model
    mock_newton_manager.get_state_0.return_value = mock_state
    mock_newton_manager.get_control.return_value = mock_control
    mock_newton_manager.get_dt.return_value = 0.01

    # Create ArticulationData with the mock view
    with patch("isaaclab_newton.assets.articulation.articulation_data.NewtonManager", mock_newton_manager):
        data = ArticulationData(mock_view, device)
        object.__setattr__(articulation, "_data", data)

    return articulation, mock_view, mock_newton_manager


##
# Test Fixtures
##


@pytest.fixture
def mock_newton_manager():
    """Create mock NewtonManager with necessary methods."""
    mock_model = MockNewtonModel()
    mock_state = MagicMock()
    mock_control = MagicMock()

    # Patch where NewtonManager is used (in the articulation module)
    with patch("isaaclab_newton.assets.articulation.articulation.NewtonManager") as MockManager:
        MockManager.get_model.return_value = mock_model
        MockManager.get_state_0.return_value = mock_state
        MockManager.get_control.return_value = mock_control
        MockManager.get_dt.return_value = 0.01
        yield MockManager


@pytest.fixture
def test_articulation():
    """Create a test articulation with default parameters."""
    articulation, mock_view, mock_manager = create_test_articulation()
    yield articulation, mock_view, mock_manager


##
# Test Cases -- Properties
##


class TestProperties:
    """Tests for Articulation properties.

    Tests the following properties:
    - data
    - num_instances
    - is_fixed_base
    - num_joints
    - num_fixed_tendons
    - num_spatial_tendons
    - num_bodies
    - joint_names
    - body_names
    """

    @pytest.mark.parametrize("num_instances", [1, 2, 4])
    def test_num_instances(self, num_instances: int):
        """Test the num_instances property returns correct count."""
        articulation, _, _ = create_test_articulation(num_instances=num_instances)
        assert articulation.num_instances == num_instances

    @pytest.mark.parametrize("num_joints", [1, 6])
    def test_num_joints(self, num_joints: int):
        """Test the num_joints property returns correct count."""
        articulation, _, _ = create_test_articulation(num_joints=num_joints)
        assert articulation.num_joints == num_joints

    @pytest.mark.parametrize("num_bodies", [1, 7])
    def test_num_bodies(self, num_bodies: int):
        """Test the num_bodies property returns correct count."""
        articulation, _, _ = create_test_articulation(num_bodies=num_bodies)
        assert articulation.num_bodies == num_bodies

    @pytest.mark.parametrize("is_fixed_base", [True, False])
    def test_is_fixed_base(self, is_fixed_base: bool):
        """Test the is_fixed_base property."""
        articulation, _, _ = create_test_articulation(is_fixed_base=is_fixed_base)
        assert articulation.is_fixed_base == is_fixed_base

    # TODO: Update when tendons are supported in Newton.
    def test_num_fixed_tendons(self):
        """Test that num_fixed_tendons returns 0 (not supported in Newton)."""
        articulation, _, _ = create_test_articulation()
        # Always returns 0 because fixed tendons are not supported in Newton.
        assert articulation.num_fixed_tendons == 0

    # TODO: Update when tendons are supported in Newton.
    def test_num_spatial_tendons(self):
        """Test that num_spatial_tendons returns 0 (not supported in Newton)."""
        articulation, _, _ = create_test_articulation()
        # Always returns 0 because spatial tendons are not supported in Newton.
        assert articulation.num_spatial_tendons == 0

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def testdata_property(self, device: str):
        """Test that data property returns ArticulationData instance."""
        articulation, _, _ = create_test_articulation(device=device)
        assert isinstance(articulation.data, ArticulationData)

    def test_joint_names(self):
        """Test that joint_names returns the correct names."""
        custom_names = ["shoulder", "elbow", "wrist"]
        articulation, _, _ = create_test_articulation(
            num_joints=3,
            joint_names=custom_names,
        )
        assert articulation.joint_names == custom_names

    def test_body_names(self):
        """Test that body_names returns the correct names."""
        custom_names = ["base", "link1", "link2", "end_effector"]
        articulation, _, _ = create_test_articulation(
            num_bodies=4,
            body_names=custom_names,
        )
        assert articulation.body_names == custom_names

##
# Test Cases -- Reset
##

class TestReset:
    """Tests for reset method."""
    def test_reset(self):
        """Test that reset method works properly."""
        articulation, _, _ = create_test_articulation()
        articulation.set_external_force_and_torque(
            forces=torch.ones(articulation.num_instances, articulation.num_bodies, 3),
            torques=torch.ones(articulation.num_instances, articulation.num_bodies, 3),
            env_ids=slice(None),
            body_ids=slice(None),
            body_mask=None,
            env_mask=None,
            is_global=False,
        )
        assert wp.to_torch(articulation.data._sim_bind_body_external_wrench).allclose(torch.ones_like(wp.to_torch(articulation.data._sim_bind_body_external_wrench)))
        articulation.reset()
        assert wp.to_torch(articulation.data._sim_bind_body_external_wrench).allclose(torch.zeros_like(wp.to_torch(articulation.data._sim_bind_body_external_wrench)))

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
        articulation, _, _ = create_test_articulation()
        articulation.update(dt=0.01)
        assert articulation.data._sim_timestamp == 0.01

##
# Test Cases -- Finders
##

class TestFinders:
    """Tests for finder methods."""
    
    @pytest.mark.parametrize("body_names", [["body_0", "body_1", "body_2"], ["body_3", "body_4", "body_5"], ["body_1", "body_3", "body_5"], "body_.*"])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies(self, body_names: list[str], device: str):
        """Test that find_bodies method works properly."""
        articulation, _, _ = create_test_articulation(device=device)
        mask, names, indices = articulation.find_bodies(body_names)
        if body_names == ["body_0", "body_1", "body_2"]:
            mask_ref = torch.zeros((7,), dtype=torch.bool, device=device)
            mask_ref[:3] = True
            assert names == ["body_0", "body_1", "body_2"]
            assert indices == [0, 1, 2]
            assert wp.to_torch(mask).allclose(mask_ref)
        elif body_names == ["body_3", "body_4", "body_5"]:
            mask_ref = torch.zeros((7,), dtype=torch.bool, device=device)
            mask_ref[3:6] = True
            assert names == ["body_3", "body_4", "body_5"]
            assert indices == [3, 4, 5]
            assert wp.to_torch(mask).allclose(mask_ref)
        elif body_names == ["body_1", "body_3", "body_5"]:
            mask_ref = torch.zeros((7,), dtype=torch.bool, device=device)
            mask_ref[1] = True
            mask_ref[3] = True
            mask_ref[5] = True
            assert names == ["body_1", "body_3", "body_5"]
            assert indices == [1, 3, 5]
            assert wp.to_torch(mask).allclose(mask_ref)
        elif body_names == "body_.*":
            mask_ref = torch.ones((7,), dtype=torch.bool, device=device)
            assert names == ["body_0", "body_1", "body_2", "body_3", "body_4", "body_5", "body_6"]
            assert indices == [0, 1, 2, 3, 4, 5, 6]
            assert wp.to_torch(mask).allclose(mask_ref)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_body_with_preserve_order(self, device: str):
        """Test that find_bodies method works properly with preserve_order."""
        articulation, _, _ = create_test_articulation(device=device)
        mask, names, indices = articulation.find_bodies(["body_5", "body_3", "body_1"], preserve_order=True)
        assert names == ["body_5", "body_3", "body_1"]
        assert indices == [5, 3, 1]
        mask_ref = torch.zeros((7,), dtype=torch.bool, device=device)
        mask_ref[1] = True
        mask_ref[3] = True
        mask_ref[5] = True
        assert wp.to_torch(mask).allclose(mask_ref)

    @pytest.mark.parametrize("joint_names", [["joint_0", "joint_1", "joint_2"], ["joint_3", "joint_4", "joint_5"], ["joint_1", "joint_3", "joint_5"], "joint_.*"])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints(self, joint_names: list[str], device: str):
        """Test that find_joints method works properly."""
        articulation, _, _ = create_test_articulation(device=device)
        mask, names, indices = articulation.find_joints(joint_names)
        if joint_names == ["joint_0", "joint_1", "joint_2"]:
            mask_ref = torch.zeros((6,), dtype=torch.bool, device=device)
            mask_ref[:3] = True
            assert names == ["joint_0", "joint_1", "joint_2"]
            assert indices == [0, 1, 2]
            assert wp.to_torch(mask).allclose(mask_ref)
        elif joint_names == ["joint_3", "joint_4", "joint_5"]:
            mask_ref = torch.zeros((6,), dtype=torch.bool, device=device)
            mask_ref[3:6] = True
            assert names == ["joint_3", "joint_4", "joint_5"]
            assert indices == [3, 4, 5]
            assert wp.to_torch(mask).allclose(mask_ref)
        elif joint_names == ["joint_1", "joint_3", "joint_5"]:
            mask_ref = torch.zeros((6,), dtype=torch.bool, device=device)
            mask_ref[1] = True
            mask_ref[3] = True
            mask_ref[5] = True
            assert names == ["joint_1", "joint_3", "joint_5"]
            assert indices == [1, 3, 5]
            assert wp.to_torch(mask).allclose(mask_ref)
        elif joint_names == "joint_.*":
            mask_ref = torch.ones((6,), dtype=torch.bool, device=device)
            assert names == ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
            assert indices == [0, 1, 2, 3, 4, 5]
            assert wp.to_torch(mask).allclose(mask_ref)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_with_preserve_order(self, device: str):
        """Test that find_joints method works properly with preserve_order."""
        articulation, _, _ = create_test_articulation(device=device)
        mask, names, indices = articulation.find_joints(["joint_5", "joint_3", "joint_1"], preserve_order=True)
        assert names == ["joint_5", "joint_3", "joint_1"]
        assert indices == [5, 3, 1]
        mask_ref = torch.zeros((6,), dtype=torch.bool, device=device)
        mask_ref[1] = True
        mask_ref[3] = True
        mask_ref[5] = True
        assert wp.to_torch(mask).allclose(mask_ref)

    # TODO: Update when tendons are supported in Newton.
    def test_find_fixed_tendons(self):
        """Test that find_fixed_tendons method works properly."""
        articulation, _, _ = create_test_articulation()
        with pytest.raises(NotImplementedError):
            articulation.find_fixed_tendons(["tendon_0", "tendon_1", "tendon_2"])

    # TODO: Update when tendons are supported in Newton.
    def test_find_spatial_tendons(self):
        """Test that find_spatial_tendons method works properly."""
        articulation, _, _ = create_test_articulation()
        with pytest.raises(NotImplementedError):
            articulation.find_spatial_tendons(["tendon_0", "tendon_1", "tendon_2"])


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
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_state_w).allclose(data, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                data = torch.rand((len(env_ids), 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                articulation.write_root_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_state_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                articulation.write_root_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_state_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)
            else: 
                # Update envs 0, 1, 2
                data = torch.rand((3, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                env_ids = env_ids.to(device=device)
                articulation.write_root_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_state_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_state_to_sim_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_state_to_sim method works properly."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_state_to_sim(wp.from_torch(data, dtype=vec13f))
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_state_w).allclose(data, atol=1e-6, rtol=1e-6)
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
                articulation.write_root_state_to_sim(data_warp, env_mask=mask_warp)
                # Check results
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_state_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)
            else:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = wp.from_torch(data.clone(), dtype=vec13f)
                mask_warp = wp.ones((num_instances,), dtype=wp.bool, device=device)
                articulation.write_root_state_to_sim(data_warp, env_mask=mask_warp)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_state_w).allclose(data, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0], torch.tensor([0, 1, 2], dtype=torch.int32)])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_state_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_state_to_sim method works properly."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_com_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                # Update selected envs
                data = torch.rand((len(env_ids), 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                articulation.write_root_com_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids].allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids].allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                articulation.write_root_com_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            else: 
                # Update envs 0, 1, 2
                data = torch.rand((3, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                env_ids = env_ids.to(device=device)
                articulation.write_root_com_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids].allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids].allclose(data[:, :7], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_state_to_sim_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_state_to_sim method works properly with warp arrays."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_com_state_to_sim(wp.from_torch(data, dtype=vec13f))
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
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
                articulation.write_root_com_state_to_sim(data_warp, env_mask=mask_warp)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids].allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids].allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            else:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = wp.from_torch(data.clone(), dtype=vec13f)
                mask_warp = wp.ones((num_instances,), dtype=wp.bool, device=device)
                # Generate reference data
                articulation.write_root_com_state_to_sim(data_warp, env_mask=mask_warp)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_state_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_state_to_sim method works properly."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_link_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_link_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                # Update selected envs
                data = torch.rand((len(env_ids), 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                articulation.write_root_link_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids].allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids].allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                articulation.write_root_link_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_link_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            else: 
                # Update envs 0, 1, 2
                data = torch.rand((3, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                env_ids = env_ids.to(device=device)
                articulation.write_root_link_state_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids].allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids].allclose(data[:, :7], atol=1e-6, rtol=1e-6)
    
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_state_to_sim_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_state_to_sim method works properly with warp arrays."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_link_state_to_sim(wp.from_torch(data, dtype=vec13f))
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_link_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)
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
                articulation.write_root_link_state_to_sim(data_warp, env_mask=mask_warp)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, :].allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids].allclose(data[:, :7], atol=1e-6, rtol=1e-6)
            else:
                # Update all envs
                data = torch.rand((num_instances, 13), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Generate warp data
                data_warp = wp.from_torch(data.clone(), dtype=vec13f)
                mask_warp = wp.ones((num_instances,), dtype=wp.bool, device=device)
                # Generate reference data
                articulation.write_root_link_state_to_sim(data_warp, env_mask=mask_warp)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_vel_w).allclose(data[:, 7:13], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_link_pose_w).allclose(data[:, :7], atol=1e-6, rtol=1e-6)

class TestVelocityWriters:
    """Tests for velocity writing methods.

    Tests methods like:
    - write_root_link_velocity_to_sim
    - write_root_com_velocity_to_sim
    """
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_state_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_state_to_sim method works properly."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]

        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_link_velocity_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(articulation.data.root_link_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(articulation.data.root_link_quat_w)
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[:, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(root_com_velocity, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                # Update selected envs
                data = torch.rand((len(env_ids), 6), device=device)
                # Write to simulation
                articulation.write_root_link_velocity_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(articulation.data.root_link_quat_w)[env_ids]
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[env_ids, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids, :].allclose(root_com_velocity, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                articulation.write_root_link_velocity_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(articulation.data.root_link_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(articulation.data.root_link_quat_w)
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[:, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(root_com_velocity, atol=1e-6, rtol=1e-6)
            else: 
                # Update envs 0, 1, 2
                data = torch.rand((3, 6), device=device)
                env_ids = env_ids.to(device=device)
                articulation.write_root_link_velocity_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(articulation.data.root_link_quat_w)[env_ids]
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[env_ids, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids, :].allclose(root_com_velocity, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_velocity_to_sim_with_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_velocity_to_sim method works properly with warp arrays."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]

        # Set a non-zero body CoM offset
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_link_velocity_to_sim(wp.from_torch(data, dtype=wp.spatial_vectorf))
                assert wp.to_torch(articulation.data.root_link_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(articulation.data.root_link_quat_w)
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[:, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(root_com_velocity, atol=1e-6, rtol=1e-6)
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
                articulation.write_root_link_velocity_to_sim(data_warp, env_mask=mask_warp)
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(articulation.data.root_link_quat_w)[env_ids]
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[env_ids, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids, :].allclose(root_com_velocity, atol=1e-6, rtol=1e-6)
            else:
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                # Generate warp data
                data_warp = wp.from_torch(data.clone(), dtype=wp.spatial_vectorf)
                mask_warp = wp.ones((num_instances,), dtype=wp.bool, device=device)
                # Generate reference data
                articulation.write_root_link_velocity_to_sim(data_warp, env_mask=mask_warp)
                assert wp.to_torch(articulation.data.root_link_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                quat = wp.to_torch(articulation.data.root_link_quat_w)
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[:, 0, :]
                # transform input velocity to center of mass frame
                root_com_velocity = data.clone()
                root_com_velocity[:, :3] += torch.linalg.cross(
                    root_com_velocity[:, 3:], quat_apply(quat, com_pos_b), dim=-1
                )
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(root_com_velocity, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_state_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_state_to_sim method works properly."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]

        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_com_velocity_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(articulation.data.root_link_vel_w)[:, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_link_vel_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                # Update selected envs
                data = torch.rand((len(env_ids), 6), device=device)
                # Write to simulation
                articulation.write_root_com_velocity_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(articulation.data.root_link_vel_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                articulation.write_root_com_velocity_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(articulation.data.root_link_vel_w)[:, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_link_vel_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
            else: 
                # Update envs 0, 1, 2
                data = torch.rand((3, 6), device=device)
                env_ids = env_ids.to(device=device)
                articulation.write_root_com_velocity_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(articulation.data.root_link_vel_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_velocity_to_sim_with_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_velocity_to_sim method works properly with warp arrays."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]

        # Set a non-zero body CoM offset
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_com_velocity_to_sim(wp.from_torch(data, dtype=wp.spatial_vectorf))
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(articulation.data.root_link_vel_w)[:, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_link_vel_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(data, atol=1e-6, rtol=1e-6)
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
                articulation.write_root_com_velocity_to_sim(data_warp, env_mask=mask_warp)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(articulation.data.root_link_vel_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
            else:
                # Update all envs
                data = torch.rand((num_instances, 6), device=device)
                # Generate warp data
                data_warp = wp.from_torch(data.clone(), dtype=wp.spatial_vectorf)
                mask_warp = wp.ones((num_instances,), dtype=wp.bool, device=device)
                # Generate reference data
                articulation.write_root_com_velocity_to_sim(data_warp, env_mask=mask_warp)
                assert articulation.data._root_link_vel_w.timestamp == -1.0
                assert torch.all(wp.to_torch(articulation.data.root_link_vel_w)[:, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_link_vel_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
                assert wp.to_torch(articulation.data.root_com_vel_w).allclose(data, atol=1e-6, rtol=1e-6)


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
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the pose transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_link_pose_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_pose_w).allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(articulation.data.root_com_pose_w)[:, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_com_pose_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, list):
                # Update selected envs
                data = torch.rand((len(env_ids), 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                articulation.write_root_link_pose_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(articulation.data.root_com_pose_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
            elif isinstance(env_ids, slice):
                # Update all envs
                data = torch.rand((num_instances, 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                articulation.write_root_link_pose_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_pose_w).allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(articulation.data.root_com_pose_w)[:, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_com_pose_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
            else: 
                # Update envs 0, 1, 2
                data = torch.rand((3, 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                env_ids = env_ids.to(device=device)
                articulation.write_root_link_pose_to_sim(data, env_ids=env_ids)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(articulation.data.root_com_pose_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_link_pose_to_sim_with_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_link_pose_to_sim method works properly with warp arrays."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_link_pose_to_sim(wp.from_torch(data, dtype=wp.transformf))
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_pose_w).allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(articulation.data.root_com_pose_w)[:, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_com_pose_w)[:, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)
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
                articulation.write_root_link_pose_to_sim(data_warp, env_mask=mask_warp)
                assert articulation.data._root_com_pose_w.timestamp == -1.0
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                assert torch.all(wp.to_torch(articulation.data.root_com_pose_w)[env_ids, :3] != data[:, :3])
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids, 3:].allclose(data[:, 3:], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_pose_to_sim_torch(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_pose_to_sim method works properly."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset to test the velocity transformation
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_com_pose_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(articulation.data.root_com_pose_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[: , 0, :]
                com_quat_b = wp.to_torch(articulation.data.body_com_quat_b)[:, 0, :]
                # transform input CoM pose to link frame
                root_link_pos, root_link_quat = combine_frame_transforms(
                    data[..., :3],
                    data[..., 3:7],
                    quat_apply(quat_inv(com_quat_b), -com_pos_b),
                    quat_inv(com_quat_b),
                )
                root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)
                assert wp.to_torch(articulation.data.root_link_pose_w).allclose(root_link_pose, atol=1e-6, rtol=1e-6)
            else:
                if isinstance(env_ids, torch.Tensor):
                    env_ids = env_ids.to(device=device)
                # Update selected envs
                data = torch.rand((len(env_ids), 7), device=device)
                data[:, 3:7] = torch.nn.functional.normalize(data[:, 3:7], p=2.0, dim=-1)
                # Write to simulation
                articulation.write_root_com_pose_to_sim(data, env_ids=env_ids)
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[env_ids, 0, :]
                com_quat_b = wp.to_torch(articulation.data.body_com_quat_b)[env_ids, 0, :]
                # transform input CoM pose to link frame
                root_link_pos, root_link_quat = combine_frame_transforms(
                    data[..., :3],
                    data[..., 3:7],
                    quat_apply(quat_inv(com_quat_b), -com_pos_b),
                    quat_inv(com_quat_b),
                )
                root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids, :].allclose(root_link_pose, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_root_com_pose_to_sim_with_warp(self, device: str, env_ids, num_instances: int):
        """Test that write_root_com_pose_to_sim method works properly with warp arrays."""
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        # Set a non-zero body CoM offset
        body_com_offset = torch.tensor([0.1, 0.01, 0.05], device=device)
        body_comdata = body_com_offset.unsqueeze(0).unsqueeze(0).expand(num_instances, articulation.num_bodies, 3)
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
                articulation.write_root_com_pose_to_sim(wp.from_torch(data, dtype=wp.transformf))
                assert wp.to_torch(articulation.data.root_com_pose_w).allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[: , 0, :]
                com_quat_b = wp.to_torch(articulation.data.body_com_quat_b)[:, 0, :]
                # transform input CoM pose to link frame
                root_link_pos, root_link_quat = combine_frame_transforms(
                    data[..., :3],
                    data[..., 3:7],
                    quat_apply(quat_inv(com_quat_b), -com_pos_b),
                    quat_inv(com_quat_b),
                )
                root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)
                assert wp.to_torch(articulation.data.root_link_pose_w).allclose(root_link_pose, atol=1e-6, rtol=1e-6)
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
                articulation.write_root_com_pose_to_sim(data_warp, env_mask=mask_warp)
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids].allclose(data, atol=1e-6, rtol=1e-6)
                # get CoM pose in link frame
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[env_ids, 0, :]
                com_quat_b = wp.to_torch(articulation.data.body_com_quat_b)[env_ids, 0, :]
                # transform input CoM pose to link frame
                root_link_pos, root_link_quat = combine_frame_transforms(
                    data[..., :3],
                    data[..., 3:7],
                    quat_apply(quat_inv(com_quat_b), -com_pos_b),
                    quat_inv(com_quat_b),
                )
                root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids, :].allclose(root_link_pose, atol=1e-6, rtol=1e-6)

class TestJointState:
    """Tests for joint state writing methods.

    Tests methods:
    - write_joint_state_to_sim
    - write_joint_position_to_sim
    - write_joint_velocity_to_sim
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_state_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_joints=num_joints, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_joints == 1:
            if (joint_ids is not None) and (not isinstance(joint_ids, slice)):
                joint_ids = [0]
        
        for _ in range(5):
            if (env_ids is None) or (isinstance(env_ids, slice)):
                if (joint_ids is None) or (isinstance(joint_ids, slice)):
                    # All envs and joints
                    data1 = torch.rand((num_instances, num_joints), device=device)
                    data2 = torch.rand((num_instances, num_joints), device=device)
                    articulation.write_joint_state_to_sim(data1, data2, env_ids=env_ids, joint_ids=joint_ids)
                    assert wp.to_torch(articulation.data.joint_pos).allclose(data1, atol=1e-6, rtol=1e-6)
                    assert wp.to_torch(articulation.data.joint_vel).allclose(data2, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    data1 = torch.rand((num_instances, len(joint_ids)), device=device)
                    data2 = torch.rand((num_instances, len(joint_ids)), device=device)
                    articulation.write_joint_state_to_sim(data1, data2, env_ids=env_ids, joint_ids=joint_ids)
                    assert wp.to_torch(articulation.data.joint_pos)[:, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)
                    assert wp.to_torch(articulation.data.joint_vel)[:, joint_ids].allclose(data2, atol=1e-6, rtol=1e-6)
            else:
                if (joint_ids is None) or (isinstance(joint_ids, slice)):
                    # Selected envs and all joints
                    data1 = torch.rand((len(env_ids), num_joints), device=device)
                    data2 = torch.rand((len(env_ids), num_joints), device=device)
                    articulation.write_joint_state_to_sim(data1, data2, env_ids=env_ids, joint_ids=joint_ids)
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids, :].allclose(data1, atol=1e-6, rtol=1e-6)
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids, :].allclose(data2, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    data1 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    data2 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    articulation.write_joint_state_to_sim(data1, data2, env_ids=env_ids, joint_ids=joint_ids)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids_, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids_, joint_ids].allclose(data2, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_state_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_joints=num_joints, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_joints == 1:
            if (joint_ids is not None) and (not isinstance(joint_ids, slice)):
                joint_ids = [0]
        
        for _ in range(5):
            if env_ids is None:
                if joint_ids is None:
                    # All envs and joints
                    data1 = torch.rand((num_instances, num_joints), device=device)
                    data2 = torch.rand((num_instances, num_joints), device=device)
                    articulation.write_joint_state_to_sim(
                        wp.from_torch(data1, dtype=wp.float32),
                        wp.from_torch(data2, dtype=wp.float32),
                        env_mask=None,
                        joint_mask=None)
                    assert wp.to_torch(articulation.data.joint_pos).allclose(data1, atol=1e-6, rtol=1e-6)
                    assert wp.to_torch(articulation.data.joint_vel).allclose(data2, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    data1 = torch.rand((num_instances, len(joint_ids)), device=device)
                    data2 = torch.rand((num_instances, len(joint_ids)), device=device)
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[:, joint_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    data2_warp = torch.ones((num_instances, num_joints), device=device)
                    data2_warp[:, joint_ids] = data2
                    data2_warp = wp.from_torch(data2_warp, dtype=wp.float32)
                    joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=device)
                    joint_mask[joint_ids] = True
                    joint_mask = wp.from_torch(joint_mask, dtype=wp.bool)
                    articulation.write_joint_state_to_sim(
                        data1_warp,
                        data2_warp,
                        env_mask=None,
                        joint_mask=joint_mask)
                    assert wp.to_torch(articulation.data.joint_pos)[:, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)
                    assert wp.to_torch(articulation.data.joint_vel)[:, joint_ids].allclose(data2, atol=1e-6, rtol=1e-6)
            else:
                if joint_ids is None:
                    # Selected envs and all joints
                    data1 = torch.rand((len(env_ids), num_joints), device=device)
                    data2 = torch.rand((len(env_ids), num_joints), device=device)
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[env_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    data2_warp = torch.ones((num_instances, num_joints), device=device)
                    data2_warp[env_ids] = data2
                    data2_warp = wp.from_torch(data2_warp, dtype=wp.float32)
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    articulation.write_joint_state_to_sim(
                        wp.from_torch(data1, dtype=wp.float32),
                        wp.from_torch(data2, dtype=wp.float32),
                        env_mask=env_mask,
                        joint_mask=None)
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids, :].allclose(data1, atol=1e-6, rtol=1e-6)
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids, :].allclose(data2, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    data1 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    data2 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[env_ids_, joint_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    data2_warp = torch.ones((num_instances, num_joints), device=device)
                    data2_warp[env_ids_, joint_ids] = data2
                    data2_warp = wp.from_torch(data2_warp, dtype=wp.float32)
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=device)
                    joint_mask[joint_ids] = True
                    joint_mask = wp.from_torch(joint_mask, dtype=wp.bool)
                    articulation.write_joint_state_to_sim(data1_warp, data2_warp, env_mask=env_mask, joint_mask=joint_mask)
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids_, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids_, joint_ids].allclose(data2, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_position_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_joints=num_joints, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_joints == 1:
            if (joint_ids is not None) and (not isinstance(joint_ids, slice)):
                joint_ids = [0]
        
        for _ in range(5):
            if (env_ids is None) or (isinstance(env_ids, slice)):
                if (joint_ids is None) or (isinstance(joint_ids, slice)):
                    # All envs and joints
                    data1 = torch.rand((num_instances, num_joints), device=device)
                    articulation.write_joint_position_to_sim(data1, env_ids=env_ids, joint_ids=joint_ids)
                    assert wp.to_torch(articulation.data.joint_pos).allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    data1 = torch.rand((num_instances, len(joint_ids)), device=device)
                    articulation.write_joint_position_to_sim(data1, env_ids=env_ids, joint_ids=joint_ids)
                    assert wp.to_torch(articulation.data.joint_pos)[:, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)
            else:
                if (joint_ids is None) or (isinstance(joint_ids, slice)):
                    # Selected envs and all joints
                    data1 = torch.rand((len(env_ids), num_joints), device=device)
                    articulation.write_joint_position_to_sim(data1, env_ids=env_ids, joint_ids=joint_ids)
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids, :].allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    data1 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    articulation.write_joint_position_to_sim(data1, env_ids=env_ids, joint_ids=joint_ids)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids_, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_position_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_joints=num_joints, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_joints == 1:
            if (joint_ids is not None) and (not isinstance(joint_ids, slice)):
                joint_ids = [0]
        
        for _ in range(5):
            if env_ids is None:
                if joint_ids is None:
                    # All envs and joints
                    data1 = torch.rand((num_instances, num_joints), device=device)
                    articulation.write_joint_position_to_sim(
                        wp.from_torch(data1, dtype=wp.float32),
                        env_mask=None,
                        joint_mask=None)
                    assert wp.to_torch(articulation.data.joint_pos).allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    data1 = torch.rand((num_instances, len(joint_ids)), device=device)
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[:, joint_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=device)
                    joint_mask[joint_ids] = True
                    joint_mask = wp.from_torch(joint_mask, dtype=wp.bool)
                    articulation.write_joint_position_to_sim(
                        data1_warp,
                        env_mask=None,
                        joint_mask=joint_mask)
                    assert wp.to_torch(articulation.data.joint_pos)[:, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)
            else:
                if joint_ids is None:
                    # Selected envs and all joints
                    data1 = torch.rand((len(env_ids), num_joints), device=device)
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[env_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    articulation.write_joint_position_to_sim(
                        data1_warp,
                        env_mask=env_mask,
                        joint_mask=None)
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids, :].allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    data1 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[env_ids_, joint_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=device)
                    joint_mask[joint_ids] = True
                    joint_mask = wp.from_torch(joint_mask, dtype=wp.bool)
                    articulation.write_joint_position_to_sim(
                        data1_warp,
                        env_mask=env_mask,
                        joint_mask=joint_mask)
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids_, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_velocity_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_joints=num_joints, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_joints == 1:
            if (joint_ids is not None) and (not isinstance(joint_ids, slice)):
                joint_ids = [0]
        
        for _ in range(5):
            if (env_ids is None) or (isinstance(env_ids, slice)):
                if (joint_ids is None) or (isinstance(joint_ids, slice)):
                    # All envs and joints
                    data1 = torch.rand((num_instances, num_joints), device=device)
                    articulation.write_joint_velocity_to_sim(data1, env_ids=env_ids, joint_ids=joint_ids)
                    assert wp.to_torch(articulation.data.joint_vel).allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    data1 = torch.rand((num_instances, len(joint_ids)), device=device)
                    articulation.write_joint_velocity_to_sim(data1, env_ids=env_ids, joint_ids=joint_ids)
                    assert wp.to_torch(articulation.data.joint_vel)[:, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)
            else:
                if (joint_ids is None) or (isinstance(joint_ids, slice)):
                    # Selected envs and all joints
                    data1 = torch.rand((len(env_ids), num_joints), device=device)
                    articulation.write_joint_velocity_to_sim(data1, env_ids=env_ids, joint_ids=joint_ids)
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids, :].allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    data1 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    articulation.write_joint_velocity_to_sim(data1, env_ids=env_ids, joint_ids=joint_ids)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids_, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_velocity_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_joints=num_joints, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_joints == 1:
            if (joint_ids is not None) and (not isinstance(joint_ids, slice)):
                joint_ids = [0]
        
        for _ in range(5):
            if env_ids is None:
                if joint_ids is None:
                    # All envs and joints
                    data1 = torch.rand((num_instances, num_joints), device=device)
                    articulation.write_joint_velocity_to_sim(
                        wp.from_torch(data1, dtype=wp.float32),
                        env_mask=None,
                        joint_mask=None)
                    assert wp.to_torch(articulation.data.joint_vel).allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    data1 = torch.rand((num_instances, len(joint_ids)), device=device)
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[:, joint_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=device)
                    joint_mask[joint_ids] = True
                    joint_mask = wp.from_torch(joint_mask, dtype=wp.bool)
                    articulation.write_joint_velocity_to_sim(
                        data1_warp,
                        env_mask=None,
                        joint_mask=joint_mask)
                    assert wp.to_torch(articulation.data.joint_vel)[:, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)
            else:
                if joint_ids is None:
                    # Selected envs and all joints
                    data1 = torch.rand((len(env_ids), num_joints), device=device)
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[env_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    articulation.write_joint_velocity_to_sim(
                        data1_warp,
                        env_mask=env_mask,
                        joint_mask=None)
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids, :].allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    data1 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[env_ids_, joint_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=device)
                    joint_mask[joint_ids] = True
                    joint_mask = wp.from_torch(joint_mask, dtype=wp.bool)
                    articulation.write_joint_velocity_to_sim(
                        data1_warp,
                        env_mask=env_mask,
                        joint_mask=joint_mask)
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids_, joint_ids].allclose(data1, atol=1e-6, rtol=1e-6)
##
# Test Cases -- Simulation Parameters Writers.
##


class TestWriteJointPropertiesToSim:
    """Tests for writing joint properties to the simulation.
    
    Tests methods:
    - write_joint_stiffness_to_sim
    - write_joint_damping_to_sim
    - write_joint_position_limit_to_sim
    - write_joint_velocity_limit_to_sim
    - write_joint_effort_limit_to_sim
    - write_joint_armature_to_sim
    - write_joint_friction_coefficient_to_sim
    - write_joint_dynamic_friction_coefficient_to_sim
    - write_joint_joint_friction_to_sim
    - write_joint_limits_to_sim
    """

    def generic_test_property_writer_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int, writer_function_name: str, property_name: str):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_joints=num_joints, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_joints == 1:
            if (joint_ids is not None) and (not isinstance(joint_ids, slice)):
                joint_ids = [0]

        writer_function = getattr(articulation, writer_function_name)
        
        for i in range(5):
            if (env_ids is None) or (isinstance(env_ids, slice)):
                if (joint_ids is None) or (isinstance(joint_ids, slice)):
                    # All envs and joints
                    if i % 2 == 0:
                        data1 = torch.rand((num_instances, num_joints), device=device)
                    else:
                        data1 = float(i)
                    writer_function(data1, env_ids=env_ids, joint_ids=joint_ids)
                    property_data = getattr(articulation.data, property_name)
                    if i % 2 == 0:
                        assert wp.to_torch(property_data).allclose(data1, atol=1e-6, rtol=1e-6)
                    else:
                        assert wp.to_torch(property_data).allclose(data1 * torch.ones((num_instances, num_joints), device=device), atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    if i % 2 == 0:
                        data1 = torch.rand((num_instances, len(joint_ids)), device=device)
                    else:
                        data1 = float(i)
                    data_ref = torch.zeros((num_instances, num_joints), device=device)
                    data_ref[:, joint_ids] = data1
                    writer_function(data1, env_ids=env_ids, joint_ids=joint_ids)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
            else:
                if (joint_ids is None) or (isinstance(joint_ids, slice)):
                    # Selected envs and all joints
                    if i % 2 == 0:
                        data1 = torch.rand((len(env_ids), num_joints), device=device)
                    else:
                        data1 = float(i)
                    data_ref = torch.zeros((num_instances, num_joints), device=device)
                    data_ref[env_ids, :] = data1
                    writer_function(data1, env_ids=env_ids, joint_ids=joint_ids)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    if i % 2 == 0:
                        data1 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    else:
                        data1 = float(i)
                    writer_function(data1, env_ids=env_ids, joint_ids=joint_ids)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    data_ref = torch.zeros((num_instances, num_joints), device=device)
                    data_ref[env_ids_, joint_ids] = data1
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)

    def generic_test_property_writer_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int, writer_function_name: str, property_name: str):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_joints=num_joints, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_joints == 1:
            if (joint_ids is not None) and (not isinstance(joint_ids, slice)):
                joint_ids = [0]

        writer_function = getattr(articulation, writer_function_name)
        
        for i in range(5):
            if env_ids is None:
                if joint_ids is None:
                    # All envs and joints
                    if i % 2 == 0:
                        data1 = torch.rand((num_instances, num_joints), device=device)
                        data1_warp = wp.from_torch(data1, dtype=wp.float32)
                    else:
                        data1 = float(i)
                        data1_warp = data1
                    writer_function(
                        data1_warp,
                        env_mask=None,
                        joint_mask=None)
                    property_data = getattr(articulation.data, property_name)
                    if i % 2 == 0:
                        assert wp.to_torch(property_data).allclose(data1, atol=1e-6, rtol=1e-6)
                    else:
                        assert wp.to_torch(property_data).allclose(data1 * torch.ones((num_instances, num_joints), device=device), atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    if i % 2 == 0:
                        data1 = torch.rand((num_instances, len(joint_ids)), device=device)
                        data1_warp = torch.ones((num_instances, num_joints), device=device)
                        data1_warp[:, joint_ids] = data1
                        data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    else:
                        data1 = float(i)
                        data1_warp = data1
                    joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=device)
                    joint_mask[joint_ids] = True
                    joint_mask = wp.from_torch(joint_mask, dtype=wp.bool)
                    data_ref = torch.zeros((num_instances, num_joints), device=device)
                    data_ref[:, joint_ids] = data1
                    writer_function(
                        data1_warp,
                        env_mask=None,
                        joint_mask=joint_mask)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
            else:
                if joint_ids is None:
                    # Selected envs and all joints
                    if i % 2 == 0:
                        data1 = torch.rand((len(env_ids), num_joints), device=device)
                        data1_warp = torch.ones((num_instances, num_joints), device=device)
                        data1_warp[env_ids] = data1
                        data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    else:
                        data1 = float(i)
                        data1_warp = data1
                    data_ref = torch.zeros((num_instances, num_joints), device=device)
                    data_ref[env_ids, :] = data1
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    writer_function(
                        data1_warp,
                        env_mask=env_mask,
                        joint_mask=None)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    if i % 2 == 0:
                        data1 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                        data1_warp = torch.ones((num_instances, num_joints), device=device)
                        data1_warp[env_ids_, joint_ids] = data1
                        data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    else:
                        data1 = float(i)
                        data1_warp = data1
                    data_ref = torch.zeros((num_instances, num_joints), device=device)
                    data_ref[env_ids_, joint_ids] = data1
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=device)
                    joint_mask[joint_ids] = True
                    joint_mask = wp.from_torch(joint_mask, dtype=wp.bool)
                    writer_function(
                        data1_warp,
                        env_mask=env_mask,
                        joint_mask=joint_mask)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)

    def generic_test_property_writer_torch_dual(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int, writer_function_name: str, property_name: str):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_joints=num_joints, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_joints == 1:
            if (joint_ids is not None) and (not isinstance(joint_ids, slice)):
                joint_ids = [0]

        writer_function = getattr(articulation, writer_function_name)
        
        for _ in range(5):
            if (env_ids is None) or (isinstance(env_ids, slice)):
                if (joint_ids is None) or (isinstance(joint_ids, slice)):
                    # All envs and joints
                    data1 = torch.rand((num_instances, num_joints), device=device)
                    data2 = torch.rand((num_instances, num_joints), device=device)
                    writer_function(data1, data2, env_ids=env_ids, joint_ids=joint_ids)
                    data = torch.cat([data1.unsqueeze(-1), data2.unsqueeze(-1)], dim=-1)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    data1 = torch.rand((num_instances, len(joint_ids)), device=device)
                    data2 = torch.rand((num_instances, len(joint_ids)), device=device)
                    writer_function(data1, data2, env_ids=env_ids, joint_ids=joint_ids)
                    data = torch.cat([data1.unsqueeze(-1), data2.unsqueeze(-1)], dim=-1)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data)[:, joint_ids].allclose(data, atol=1e-6, rtol=1e-6)
            else:
                if (joint_ids is None) or (isinstance(joint_ids, slice)):
                    # Selected envs and all joints
                    data1 = torch.rand((len(env_ids), num_joints), device=device)
                    data2 = torch.rand((len(env_ids), num_joints), device=device)
                    writer_function(data1, data2, env_ids=env_ids, joint_ids=joint_ids)
                    data = torch.cat([data1.unsqueeze(-1), data2.unsqueeze(-1)], dim=-1)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    data1 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    data2 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    writer_function(data1, data2, env_ids=env_ids, joint_ids=joint_ids)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    property_data = getattr(articulation.data, property_name)
                    data = torch.cat([data1.unsqueeze(-1), data2.unsqueeze(-1)], dim=-1)
                    assert wp.to_torch(property_data)[env_ids_, joint_ids].allclose(data, atol=1e-6, rtol=1e-6)

    def generic_test_property_writer_warp_dual(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int, writer_function_name: str, property_name: str):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_joints=num_joints, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_joints == 1:
            if (joint_ids is not None) and (not isinstance(joint_ids, slice)):
                joint_ids = [0]

        writer_function = getattr(articulation, writer_function_name)
        
        for _ in range(5):
            if env_ids is None:
                if joint_ids is None:
                    # All envs and joints
                    data1 = torch.rand((num_instances, num_joints), device=device)
                    data2 = torch.rand((num_instances, num_joints), device=device)
                    writer_function(
                        wp.from_torch(data1, dtype=wp.float32),
                        wp.from_torch(data2, dtype=wp.float32),
                        env_mask=None,
                        joint_mask=None)
                    data = torch.cat([data1.unsqueeze(-1), data2.unsqueeze(-1)], dim=-1)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data, atol=2e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    data1 = torch.rand((num_instances, len(joint_ids)), device=device)
                    data2 = torch.rand((num_instances, len(joint_ids)), device=device)
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[:, joint_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    data2_warp = torch.ones((num_instances, num_joints), device=device)
                    data2_warp[:, joint_ids] = data2
                    data2_warp = wp.from_torch(data2_warp, dtype=wp.float32)
                    joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=device)
                    joint_mask[joint_ids] = True
                    joint_mask = wp.from_torch(joint_mask, dtype=wp.bool)
                    writer_function(
                        data1_warp,
                        data2_warp,
                        env_mask=None,
                        joint_mask=joint_mask,
                    )
                    data = torch.cat([data1.unsqueeze(-1), data2.unsqueeze(-1)], dim=-1)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data)[:, joint_ids].allclose(data, atol=1e-6, rtol=1e-6)
            else:
                if joint_ids is None:
                    # Selected envs and all joints
                    data1 = torch.rand((len(env_ids), num_joints), device=device)
                    data2 = torch.rand((len(env_ids), num_joints), device=device)
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[env_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    data2_warp = torch.ones((num_instances, num_joints), device=device)
                    data2_warp[env_ids] = data2
                    data2_warp = wp.from_torch(data2_warp, dtype=wp.float32)
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    writer_function(
                        data1_warp,
                        data2_warp,
                        env_mask=env_mask,
                        joint_mask=None,
                    )
                    data = torch.cat([data1.unsqueeze(-1), data2.unsqueeze(-1)], dim=-1)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data)[env_ids, :].allclose(data, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    data1 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    data2 = torch.rand((len(env_ids), len(joint_ids)), device=device)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    data1_warp = torch.ones((num_instances, num_joints), device=device)
                    data1_warp[env_ids_, joint_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=wp.float32)
                    data2_warp = torch.ones((num_instances, num_joints), device=device)
                    data2_warp[env_ids_, joint_ids] = data2
                    data2_warp = wp.from_torch(data2_warp, dtype=wp.float32)
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    joint_mask = torch.zeros((num_joints,), dtype=torch.bool, device=device)
                    joint_mask[joint_ids] = True
                    joint_mask = wp.from_torch(joint_mask, dtype=wp.bool)
                    writer_function(
                        data1_warp,
                        data2_warp,
                        env_mask=env_mask,
                        joint_mask=joint_mask)
                    data = torch.cat([data1.unsqueeze(-1), data2.unsqueeze(-1)], dim=-1)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data)[env_ids_, joint_ids].allclose(data, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_stiffness_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_torch(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_stiffness_to_sim", "joint_stiffness")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_stiffness_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_warp(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_stiffness_to_sim", "joint_stiffness")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_damping_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_torch(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_damping_to_sim", "joint_damping")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_damping_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_warp(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_damping_to_sim", "joint_damping")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_velocity_limit_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_torch(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_velocity_limit_to_sim", "joint_vel_limits")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_velocity_limit_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_warp(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_velocity_limit_to_sim", "joint_vel_limits")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_effort_limit_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_torch(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_effort_limit_to_sim", "joint_effort_limits")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_effort_limit_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_warp(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_effort_limit_to_sim", "joint_effort_limits")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_armature_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_torch(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_armature_to_sim", "joint_armature")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_armature_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_warp(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_armature_to_sim", "joint_armature")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_friction_coefficient_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_torch(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_friction_coefficient_to_sim", "joint_friction_coeff")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_friction_coefficient_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_warp(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_friction_coefficient_to_sim", "joint_friction_coeff")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_dynamic_friction_coefficient_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_torch(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_dynamic_friction_coefficient_to_sim", "joint_dynamic_friction_coeff")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_dynamic_friction_coefficient_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_warp(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_dynamic_friction_coefficient_to_sim", "joint_dynamic_friction_coeff")

    # TODO: Remove once the deprecated function is removed.
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_friction_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_torch(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_friction_to_sim", "joint_friction_coeff")

    # TODO: Remove once the deprecated function is removed.
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_friction_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_warp(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_friction_to_sim", "joint_friction_coeff")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_position_limit_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_torch_dual(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_position_limit_to_sim", "joint_pos_limits")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_position_limit_to_sim_warp_dual(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_warp_dual(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_position_limit_to_sim", "joint_pos_limits")

    # TODO: Remove once the deprecated function is removed.
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_limits_to_sim_torch(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_torch_dual(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_limits_to_sim", "joint_pos_limits")

    # TODO: Remove once the deprecated function is removed.
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_limits_to_sim_warp_dual(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        self.generic_test_property_writer_warp_dual(device, env_ids, joint_ids, num_instances, num_joints, "write_joint_limits_to_sim", "joint_pos_limits")

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

    def generic_test_property_writer_torch(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int, writer_function_name: str, property_name: str, dtype: type = wp.float32):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_bodies=num_bodies, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_bodies == 1:
            if (body_ids is not None) and (not isinstance(body_ids, slice)):
                body_ids = [0]

        writer_function = getattr(articulation, writer_function_name)
        if dtype == wp.float32:
            ndims = tuple()
        elif dtype == wp.vec3f:
            ndims = (3,)
        elif dtype == wp.mat33f:
            ndims = (3,3,)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        for _ in range(5):
            if (env_ids is None) or (isinstance(env_ids, slice)):
                if (body_ids is None) or (isinstance(body_ids, slice)):
                    # All envs and joints
                    data1 = torch.rand((num_instances, num_bodies, *ndims), device=device)
                    writer_function(data1, env_ids=env_ids, body_ids=body_ids)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected bodies
                    data1 = torch.rand((num_instances, len(body_ids), *ndims), device=device)
                    data_ref = torch.zeros((num_instances, num_bodies, *ndims), device=device)
                    data_ref[:, body_ids] = data1
                    writer_function(data1, env_ids=env_ids, body_ids=body_ids)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
            else:
                if (body_ids is None) or (isinstance(body_ids, slice)):
                    # Selected envs and all bodies
                    data1 = torch.rand((len(env_ids), num_bodies, *ndims), device=device)
                    data_ref = torch.zeros((num_instances, num_bodies, *ndims), device=device)
                    data_ref[env_ids, :] = data1
                    writer_function(data1, env_ids=env_ids, body_ids=body_ids)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and bodies
                    data1 = torch.rand((len(env_ids), len(body_ids), *ndims), device=device)
                    writer_function(data1, env_ids=env_ids, body_ids=body_ids)
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    data_ref = torch.zeros((num_instances, num_bodies, *ndims), device=device)
                    data_ref[env_ids_, body_ids] = data1
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)

    def generic_test_property_writer_warp(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int, writer_function_name: str, property_name: str, dtype: type = wp.float32):
        articulation, mock_view, _ = create_test_articulation(num_instances=num_instances, num_bodies=num_bodies, device=device)
        if num_instances == 1:
            if (env_ids is not None) and (not isinstance(env_ids, slice)):
                env_ids = [0]
        if num_bodies == 1:
            if (body_ids is not None) and (not isinstance(body_ids, slice)):
                body_ids = [0]

        writer_function = getattr(articulation, writer_function_name)
        if dtype == wp.float32:
            ndims = tuple()
        elif dtype == wp.vec3f:
            ndims = (3,)
        elif dtype == wp.mat33f:
            ndims = (3,3,)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        for _ in range(5):
            if env_ids is None:
                if body_ids is None:
                    # All envs and joints
                    data1 = torch.rand((num_instances, num_bodies, *ndims), device=device)
                    data1_warp = wp.from_torch(data1, dtype=dtype)
                    writer_function(
                        data1_warp,
                        env_mask=None,
                        body_mask=None)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data1, atol=1e-6, rtol=1e-6)
                else:
                    # All envs and selected joints
                    data1 = torch.rand((num_instances, len(body_ids), *ndims), device=device)
                    data1_warp = torch.ones((num_instances, num_bodies, *ndims), device=device)
                    data1_warp[:, body_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=dtype)
                    body_mask = torch.zeros((num_bodies,), dtype=torch.bool, device=device)
                    body_mask[body_ids] = True
                    body_mask = wp.from_torch(body_mask, dtype=wp.bool)
                    data_ref = torch.zeros((num_instances, num_bodies, *ndims), device=device)
                    data_ref[:, body_ids] = data1
                    writer_function(
                        data1_warp,
                        env_mask=None,
                        body_mask=body_mask)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
            else:
                if body_ids is None:
                    # Selected envs and all joints
                    data1 = torch.rand((len(env_ids), num_bodies, *ndims), device=device)
                    data1_warp = torch.ones((num_instances, num_bodies, *ndims), device=device)
                    data1_warp[env_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=dtype)
                    data_ref = torch.zeros((num_instances, num_bodies, *ndims), device=device)
                    data_ref[env_ids, :] = data1
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    writer_function(
                        data1_warp,
                        env_mask=env_mask,
                        body_mask=None)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)
                else:
                    # Selected envs and joints
                    env_ids_ = torch.tensor(env_ids, dtype=torch.int32, device=device)
                    env_ids_ = env_ids_[:, None]
                    data1 = torch.rand((len(env_ids), len(body_ids), *ndims), device=device)
                    data1_warp = torch.ones((num_instances, num_bodies, *ndims), device=device)
                    data1_warp[env_ids_, body_ids] = data1
                    data1_warp = wp.from_torch(data1_warp, dtype=dtype)
                    data_ref = torch.zeros((num_instances, num_bodies, *ndims), device=device)
                    data_ref[env_ids_, body_ids] = data1
                    env_mask = torch.zeros((num_instances,), dtype=torch.bool, device=device)
                    env_mask[env_ids] = True
                    env_mask = wp.from_torch(env_mask, dtype=wp.bool)
                    body_mask = torch.zeros((num_bodies,), dtype=torch.bool, device=device)
                    body_mask[body_ids] = True
                    body_mask = wp.from_torch(body_mask, dtype=wp.bool)
                    writer_function(
                        data1_warp,
                        env_mask=env_mask,
                        body_mask=body_mask)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_masses_to_sim_torch(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_torch(device, env_ids, body_ids, num_instances, num_bodies, "set_masses", "body_mass", dtype=wp.float32)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_masses_to_sim_warp(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_warp(device, env_ids, body_ids, num_instances, num_bodies, "set_masses", "body_mass", dtype=wp.float32)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_coms_to_sim_torch(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_torch(device, env_ids, body_ids, num_instances, num_bodies, "set_coms", "body_com_pos_b", dtype=wp.vec3f)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_coms_to_sim_warp(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_warp(device, env_ids, body_ids, num_instances, num_bodies, "set_coms", "body_com_pos_b", dtype=wp.vec3f)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_inertias_to_sim_torch(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_torch(device, env_ids, body_ids, num_instances, num_bodies, "set_inertias", "body_inertia", dtype=wp.mat33f)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_inertias_to_sim_warp(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_warp(device, env_ids, body_ids, num_instances, num_bodies, "set_inertias", "body_inertia", dtype=wp.mat33f)

# TODO: Implement these tests once the Wrench Composers made it to main IsaacLab.
class TestSettersExternalWrench:
    """Tests for setter methods that set external wrench.

    Tests methods:
    - set_external_force_and_torque
    """

    @pytest.mark.skip(reason="Not implemented")
    def test_external_force_and_torque_to_sim_torch(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        raise NotImplementedError()

    @pytest.mark.skip(reason="Not implemented")
    def test_external_force_and_torque_to_sim_warp(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        raise NotImplementedError()


class TestFixedTendonsSetters:
    """Tests for setter methods that set fixed tendon properties.

    Tests methods:
    - set_fixed_tendon_stiffness
    - set_fixed_tendon_damping
    - set_fixed_tendon_limit_stiffness
    - set_fixed_tendon_position_limit
    - set_fixed_tendon_limit (deprecated)
    - set_fixed_tendon_rest_length
    - set_fixed_tendon_offset
    - write_fixed_tendon_properties_to_sim
    """

    def test_set_fixed_tendon_stiffness_not_implemented(self):
        """Test that set_fixed_tendon_stiffness raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        stiffness = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_fixed_tendon_stiffness(stiffness)

    def test_set_fixed_tendon_damping_not_implemented(self):
        """Test that set_fixed_tendon_damping raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        damping = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_fixed_tendon_damping(damping)

    def test_set_fixed_tendon_limit_stiffness_not_implemented(self):
        """Test that set_fixed_tendon_limit_stiffness raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        limit_stiffness = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_fixed_tendon_limit_stiffness(limit_stiffness)

    def test_set_fixed_tendon_position_limit_not_implemented(self):
        """Test that set_fixed_tendon_position_limit raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        limit = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_fixed_tendon_position_limit(limit)

    def test_set_fixed_tendon_limit_not_implemented(self):
        """Test that set_fixed_tendon_limit (deprecated) raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        limit = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_fixed_tendon_limit(limit)

    def test_set_fixed_tendon_rest_length_not_implemented(self):
        """Test that set_fixed_tendon_rest_length raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        rest_length = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_fixed_tendon_rest_length(rest_length)

    def test_set_fixed_tendon_offset_not_implemented(self):
        """Test that set_fixed_tendon_offset raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        offset = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_fixed_tendon_offset(offset)

    def test_write_fixed_tendon_properties_to_sim_not_implemented(self):
        """Test that write_fixed_tendon_properties_to_sim raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        with pytest.raises(NotImplementedError):
            articulation.write_fixed_tendon_properties_to_sim()


class TestSpatialTendonsSetters:
    """Tests for setter methods that set spatial tendon properties.

    Tests methods:
    - set_spatial_tendon_stiffness
    - set_spatial_tendon_damping
    - set_spatial_tendon_limit_stiffness
    - set_spatial_tendon_offset
    - write_spatial_tendon_properties_to_sim
    """

    def test_set_spatial_tendon_stiffness_not_implemented(self):
        """Test that set_spatial_tendon_stiffness raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        stiffness = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_spatial_tendon_stiffness(stiffness)

    def test_set_spatial_tendon_damping_not_implemented(self):
        """Test that set_spatial_tendon_damping raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        damping = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_spatial_tendon_damping(damping)

    def test_set_spatial_tendon_limit_stiffness_not_implemented(self):
        """Test that set_spatial_tendon_limit_stiffness raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        limit_stiffness = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_spatial_tendon_limit_stiffness(limit_stiffness)

    def test_set_spatial_tendon_offset_not_implemented(self):
        """Test that set_spatial_tendon_offset raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        offset = wp.zeros((2, 1), dtype=wp.float32, device="cuda:0")
        with pytest.raises(NotImplementedError):
            articulation.set_spatial_tendon_offset(offset)

    def test_write_spatial_tendon_properties_to_sim_not_implemented(self):
        """Test that write_spatial_tendon_properties_to_sim raises NotImplementedError."""
        articulation, _, _ = create_test_articulation()
        with pytest.raises(NotImplementedError):
            articulation.write_spatial_tendon_properties_to_sim()

#
#class TestSetters:
#    """Tests for setter methods.
#
#    Tests methods like:
#    - set_joint_position_target
#    - set_joint_velocity_target
#    - set_joint_effort_target
#    """
#
#    def test_set_joint_position_target(self):
#        """Test setting joint position targets."""
#        num_instances = 4
#        num_joints = 6
#        device = "cuda:0"
#        articulation, _, _ = create_test_articulation(
#            num_instances=num_instances,
#            num_joints=num_joints,
#            device=device,
#        )
#
#        # Create test targets
#        targets = torch.rand(num_instances, num_joints, device=device)
#
#        # Set targets (accepts torch.Tensor despite type hint saying wp.array)
#        articulation.set_joint_position_target(targets)  # type: ignore[arg-type]
#
#        # Verify
#        result = wp.to_torch(articulation.data.actuator_position_target)
#        torch.testing.assert_close(result, targets, atol=1e-5, rtol=1e-5)
#
#    def test_set_joint_velocity_target(self):
#        """Test setting joint velocity targets."""
#        num_instances = 4
#        num_joints = 6
#        device = "cuda:0"
#        articulation, _, _ = create_test_articulation(
#            num_instances=num_instances,
#            num_joints=num_joints,
#            device=device,
#        )
#
#        # Create test targets
#        targets = torch.rand(num_instances, num_joints, device=device)
#
#        # Set targets (accepts torch.Tensor despite type hint saying wp.array)
#        articulation.set_joint_velocity_target(targets)  # type: ignore[arg-type]
#
#        # Verify
#        result = wp.to_torch(articulation.data.actuator_velocity_target)
#        torch.testing.assert_close(result, targets, atol=1e-5, rtol=1e-5)
#
#    def test_set_joint_effort_target(self):
#        """Test setting joint effort targets."""
#        num_instances = 4
#        num_joints = 6
#        device = "cuda:0"
#        articulation, _, _ = create_test_articulation(
#            num_instances=num_instances,
#            num_joints=num_joints,
#            device=device,
#        )
#
#        # Create test targets
#        targets = torch.rand(num_instances, num_joints, device=device)
#
#        # Set targets (accepts torch.Tensor despite type hint saying wp.array)
#        articulation.set_joint_effort_target(targets)  # type: ignore[arg-type]
#
#        # Verify
#        result = wp.to_torch(articulation.data.actuator_effort_target)
#        torch.testing.assert_close(result, targets, atol=1e-5, rtol=1e-5)
#
#
###
## Test Cases -- Finders
###
#
#
#class TestFinders:
#    """Tests for finder methods.
#
#    Tests methods like:
#    - find_joints
#    - find_bodies
#    """
#
#    def test_find_joints_by_regex(self):
#        """Test finding joints using regex patterns."""
#        joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
#        articulation, _, _ = create_test_articulation(
#            num_joints=6,
#            joint_names=joint_names,
#        )
#
#        # Find all wrist joints
#        mask, names, indices = articulation.find_joints("wrist_.*")
#
#        assert names == ["wrist_1", "wrist_2", "wrist_3"]
#        assert indices == [3, 4, 5]
#
#    def test_find_joints_by_list(self):
#        """Test finding specific joints by name list."""
#        joint_names = ["joint_a", "joint_b", "joint_c", "joint_d"]
#        articulation, _, _ = create_test_articulation(
#            num_joints=4,
#            joint_names=joint_names,
#        )
#
#        # Find specific joints
#        mask, names, indices = articulation.find_joints(["joint_b", "joint_d"])
#
#        assert set(names) == {"joint_b", "joint_d"}
#        assert set(indices) == {1, 3}
#
#    def test_find_bodies_by_regex(self):
#        """Test finding bodies using regex patterns."""
#        body_names = ["base_link", "link_1", "link_2", "gripper_base", "gripper_finger"]
#        articulation, _, _ = create_test_articulation(
#            num_bodies=5,
#            body_names=body_names,
#        )
#
#        # Find all gripper bodies
#        mask, names, indices = articulation.find_bodies("gripper_.*")
#
#        assert names == ["gripper_base", "gripper_finger"]
#        assert indices == [3, 4]
