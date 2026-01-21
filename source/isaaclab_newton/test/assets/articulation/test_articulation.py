# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import torch
from unittest.mock import MagicMock, patch

import pytest
import warp as wp

# Import mock classes from common test utilities
from common.mock_newton import MockNewtonArticulationView, MockNewtonModel
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.kernels import vec13f

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

# TODO: Move these functions to the test utils so they can't be changed in the future.
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_inv

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

    # Call _create_buffers() to initialize temp buffers and wrench composers
    articulation._create_buffers()

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
        assert wp.to_torch(articulation.permanent_wrench_composer.composed_force).allclose(
            torch.ones_like(wp.to_torch(articulation.permanent_wrench_composer.composed_force))
        )
        articulation.reset()
        assert wp.to_torch(articulation.permanent_wrench_composer.composed_force).allclose(
            torch.zeros_like(wp.to_torch(articulation.permanent_wrench_composer.composed_force))
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
        articulation, _, _ = create_test_articulation()
        articulation.update(dt=0.01)
        assert articulation.data._sim_timestamp == 0.01


##
# Test Cases -- Finders
##


class TestFinders:
    """Tests for finder methods."""

    @pytest.mark.parametrize(
        "body_names",
        [["body_0", "body_1", "body_2"], ["body_3", "body_4", "body_5"], ["body_1", "body_3", "body_5"], "body_.*"],
    )
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

    @pytest.mark.parametrize(
        "joint_names",
        [
            ["joint_0", "joint_1", "joint_2"],
            ["joint_3", "joint_4", "joint_5"],
            ["joint_1", "joint_3", "joint_5"],
            "joint_.*",
        ],
    )
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
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids].allclose(
                    data[:, :7], atol=1e-6, rtol=1e-6
                )
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
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids].allclose(
                    data[:, :7], atol=1e-6, rtol=1e-6
                )

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
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids].allclose(
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
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids].allclose(
                    data[:, :7], atol=1e-6, rtol=1e-6
                )
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
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids].allclose(
                    data[:, :7], atol=1e-6, rtol=1e-6
                )

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
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, :].allclose(
                    data[:, 7:13], atol=1e-6, rtol=1e-6
                )
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids].allclose(
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
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids, :].allclose(
                    root_com_velocity, atol=1e-6, rtol=1e-6
                )
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
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids, :].allclose(
                    root_com_velocity, atol=1e-6, rtol=1e-6
                )

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
                assert wp.to_torch(articulation.data.root_com_vel_w)[env_ids, :].allclose(
                    root_com_velocity, atol=1e-6, rtol=1e-6
                )
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
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )
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
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )
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
                assert wp.to_torch(articulation.data.root_link_vel_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )
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
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )
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
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )

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
                assert wp.to_torch(articulation.data.root_com_pose_w)[env_ids, 3:].allclose(
                    data[:, 3:], atol=1e-6, rtol=1e-6
                )

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
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[:, 0, :]
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
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids, :].allclose(
                    root_link_pose, atol=1e-6, rtol=1e-6
                )

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
                com_pos_b = wp.to_torch(articulation.data.body_com_pos_b)[:, 0, :]
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
                assert wp.to_torch(articulation.data.root_link_pose_w)[env_ids, :].allclose(
                    root_link_pose, atol=1e-6, rtol=1e-6
                )


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
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_joints=num_joints, device=device
        )
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
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids_, joint_ids].allclose(
                        data1, atol=1e-6, rtol=1e-6
                    )
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids_, joint_ids].allclose(
                        data2, atol=1e-6, rtol=1e-6
                    )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_state_to_sim_warp(self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_joints=num_joints, device=device
        )
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
                        joint_mask=None,
                    )
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
                    articulation.write_joint_state_to_sim(data1_warp, data2_warp, env_mask=None, joint_mask=joint_mask)
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
                        joint_mask=None,
                    )
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
                    articulation.write_joint_state_to_sim(
                        data1_warp, data2_warp, env_mask=env_mask, joint_mask=joint_mask
                    )
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids_, joint_ids].allclose(
                        data1, atol=1e-6, rtol=1e-6
                    )
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids_, joint_ids].allclose(
                        data2, atol=1e-6, rtol=1e-6
                    )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_position_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_joints=num_joints, device=device
        )
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
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids_, joint_ids].allclose(
                        data1, atol=1e-6, rtol=1e-6
                    )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_position_to_sim_warp(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_joints=num_joints, device=device
        )
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
                        wp.from_torch(data1, dtype=wp.float32), env_mask=None, joint_mask=None
                    )
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
                    articulation.write_joint_position_to_sim(data1_warp, env_mask=None, joint_mask=joint_mask)
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
                    articulation.write_joint_position_to_sim(data1_warp, env_mask=env_mask, joint_mask=None)
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
                    articulation.write_joint_position_to_sim(data1_warp, env_mask=env_mask, joint_mask=joint_mask)
                    assert wp.to_torch(articulation.data.joint_pos)[env_ids_, joint_ids].allclose(
                        data1, atol=1e-6, rtol=1e-6
                    )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], slice(None), [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_velocity_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_joints=num_joints, device=device
        )
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
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids_, joint_ids].allclose(
                        data1, atol=1e-6, rtol=1e-6
                    )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_velocity_to_sim_warp(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_joints=num_joints, device=device
        )
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
                        wp.from_torch(data1, dtype=wp.float32), env_mask=None, joint_mask=None
                    )
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
                    articulation.write_joint_velocity_to_sim(data1_warp, env_mask=None, joint_mask=joint_mask)
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
                    articulation.write_joint_velocity_to_sim(data1_warp, env_mask=env_mask, joint_mask=None)
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
                    articulation.write_joint_velocity_to_sim(data1_warp, env_mask=env_mask, joint_mask=joint_mask)
                    assert wp.to_torch(articulation.data.joint_vel)[env_ids_, joint_ids].allclose(
                        data1, atol=1e-6, rtol=1e-6
                    )


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

    def generic_test_property_writer_torch(
        self,
        device: str,
        env_ids,
        joint_ids,
        num_instances: int,
        num_joints: int,
        writer_function_name: str,
        property_name: str,
    ):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_joints=num_joints, device=device
        )
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
                        assert wp.to_torch(property_data).allclose(
                            data1 * torch.ones((num_instances, num_joints), device=device), atol=1e-6, rtol=1e-6
                        )
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

    def generic_test_property_writer_warp(
        self,
        device: str,
        env_ids,
        joint_ids,
        num_instances: int,
        num_joints: int,
        writer_function_name: str,
        property_name: str,
    ):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_joints=num_joints, device=device
        )
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
                    writer_function(data1_warp, env_mask=None, joint_mask=None)
                    property_data = getattr(articulation.data, property_name)
                    if i % 2 == 0:
                        assert wp.to_torch(property_data).allclose(data1, atol=1e-6, rtol=1e-6)
                    else:
                        assert wp.to_torch(property_data).allclose(
                            data1 * torch.ones((num_instances, num_joints), device=device), atol=1e-6, rtol=1e-6
                        )
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
                    writer_function(data1_warp, env_mask=None, joint_mask=joint_mask)
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
                    writer_function(data1_warp, env_mask=env_mask, joint_mask=None)
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
                    writer_function(data1_warp, env_mask=env_mask, joint_mask=joint_mask)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)

    def generic_test_property_writer_torch_dual(
        self,
        device: str,
        env_ids,
        joint_ids,
        num_instances: int,
        num_joints: int,
        writer_function_name: str,
        property_name: str,
    ):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_joints=num_joints, device=device
        )
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

    def generic_test_property_writer_warp_dual(
        self,
        device: str,
        env_ids,
        joint_ids,
        num_instances: int,
        num_joints: int,
        writer_function_name: str,
        property_name: str,
    ):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_joints=num_joints, device=device
        )
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
                        joint_mask=None,
                    )
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
                    writer_function(data1_warp, data2_warp, env_mask=env_mask, joint_mask=joint_mask)
                    data = torch.cat([data1.unsqueeze(-1), data2.unsqueeze(-1)], dim=-1)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data)[env_ids_, joint_ids].allclose(data, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_stiffness_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_torch(
            device, env_ids, joint_ids, num_instances, num_joints, "write_joint_stiffness_to_sim", "joint_stiffness"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_stiffness_to_sim_warp(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_warp(
            device, env_ids, joint_ids, num_instances, num_joints, "write_joint_stiffness_to_sim", "joint_stiffness"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_damping_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_torch(
            device, env_ids, joint_ids, num_instances, num_joints, "write_joint_damping_to_sim", "joint_damping"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_damping_to_sim_warp(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_warp(
            device, env_ids, joint_ids, num_instances, num_joints, "write_joint_damping_to_sim", "joint_damping"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_velocity_limit_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_torch(
            device,
            env_ids,
            joint_ids,
            num_instances,
            num_joints,
            "write_joint_velocity_limit_to_sim",
            "joint_vel_limits",
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_velocity_limit_to_sim_warp(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_warp(
            device,
            env_ids,
            joint_ids,
            num_instances,
            num_joints,
            "write_joint_velocity_limit_to_sim",
            "joint_vel_limits",
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_effort_limit_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_torch(
            device,
            env_ids,
            joint_ids,
            num_instances,
            num_joints,
            "write_joint_effort_limit_to_sim",
            "joint_effort_limits",
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_effort_limit_to_sim_warp(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_warp(
            device,
            env_ids,
            joint_ids,
            num_instances,
            num_joints,
            "write_joint_effort_limit_to_sim",
            "joint_effort_limits",
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_armature_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_torch(
            device, env_ids, joint_ids, num_instances, num_joints, "write_joint_armature_to_sim", "joint_armature"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_armature_to_sim_warp(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_warp(
            device, env_ids, joint_ids, num_instances, num_joints, "write_joint_armature_to_sim", "joint_armature"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_friction_coefficient_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_torch(
            device,
            env_ids,
            joint_ids,
            num_instances,
            num_joints,
            "write_joint_friction_coefficient_to_sim",
            "joint_friction_coeff",
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_friction_coefficient_to_sim_warp(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_warp(
            device,
            env_ids,
            joint_ids,
            num_instances,
            num_joints,
            "write_joint_friction_coefficient_to_sim",
            "joint_friction_coeff",
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_dynamic_friction_coefficient_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_torch(
            device,
            env_ids,
            joint_ids,
            num_instances,
            num_joints,
            "write_joint_dynamic_friction_coefficient_to_sim",
            "joint_dynamic_friction_coeff",
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_dynamic_friction_coefficient_to_sim_warp(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_warp(
            device,
            env_ids,
            joint_ids,
            num_instances,
            num_joints,
            "write_joint_dynamic_friction_coefficient_to_sim",
            "joint_dynamic_friction_coeff",
        )

    # TODO: Remove once the deprecated function is removed.
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_friction_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_torch(
            device, env_ids, joint_ids, num_instances, num_joints, "write_joint_friction_to_sim", "joint_friction_coeff"
        )

    # TODO: Remove once the deprecated function is removed.
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_friction_to_sim_warp(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_warp(
            device, env_ids, joint_ids, num_instances, num_joints, "write_joint_friction_to_sim", "joint_friction_coeff"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_position_limit_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_torch_dual(
            device,
            env_ids,
            joint_ids,
            num_instances,
            num_joints,
            "write_joint_position_limit_to_sim",
            "joint_pos_limits",
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_position_limit_to_sim_warp_dual(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_warp_dual(
            device,
            env_ids,
            joint_ids,
            num_instances,
            num_joints,
            "write_joint_position_limit_to_sim",
            "joint_pos_limits",
        )

    # TODO: Remove once the deprecated function is removed.
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_limits_to_sim_torch(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_torch_dual(
            device, env_ids, joint_ids, num_instances, num_joints, "write_joint_limits_to_sim", "joint_pos_limits"
        )

    # TODO: Remove once the deprecated function is removed.
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("joint_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_write_joint_limits_to_sim_warp_dual(
        self, device: str, env_ids, joint_ids, num_instances: int, num_joints: int
    ):
        self.generic_test_property_writer_warp_dual(
            device, env_ids, joint_ids, num_instances, num_joints, "write_joint_limits_to_sim", "joint_pos_limits"
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
        num_bodies: int,
        writer_function_name: str,
        property_name: str,
        dtype: type = wp.float32,
    ):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_bodies=num_bodies, device=device
        )
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

    def generic_test_property_writer_warp(
        self,
        device: str,
        env_ids,
        body_ids,
        num_instances: int,
        num_bodies: int,
        writer_function_name: str,
        property_name: str,
        dtype: type = wp.float32,
    ):
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances, num_bodies=num_bodies, device=device
        )
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
                    data1 = torch.rand((num_instances, num_bodies, *ndims), device=device)
                    data1_warp = wp.from_torch(data1, dtype=dtype)
                    writer_function(data1_warp, env_mask=None, body_mask=None)
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
                    writer_function(data1_warp, env_mask=None, body_mask=body_mask)
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
                    writer_function(data1_warp, env_mask=env_mask, body_mask=None)
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
                    writer_function(data1_warp, env_mask=env_mask, body_mask=body_mask)
                    property_data = getattr(articulation.data, property_name)
                    assert wp.to_torch(property_data).allclose(data_ref, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_masses_to_sim_torch(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_torch(
            device, env_ids, body_ids, num_instances, num_bodies, "set_masses", "body_mass", dtype=wp.float32
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_masses_to_sim_warp(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_warp(
            device, env_ids, body_ids, num_instances, num_bodies, "set_masses", "body_mass", dtype=wp.float32
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_coms_to_sim_torch(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_torch(
            device, env_ids, body_ids, num_instances, num_bodies, "set_coms", "body_com_pos_b", dtype=wp.vec3f
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_coms_to_sim_warp(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_warp(
            device, env_ids, body_ids, num_instances, num_bodies, "set_coms", "body_com_pos_b", dtype=wp.vec3f
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_inertias_to_sim_torch(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_torch(
            device, env_ids, body_ids, num_instances, num_bodies, "set_inertias", "body_inertia", dtype=wp.mat33f
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("env_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("body_ids", [None, [0, 1, 2], [0]])
    @pytest.mark.parametrize("num_bodies", [1, 6])
    @pytest.mark.parametrize("num_instances", [1, 4])
    def test_set_inertias_to_sim_warp(self, device: str, env_ids, body_ids, num_instances: int, num_bodies: int):
        self.generic_test_property_writer_warp(
            device, env_ids, body_ids, num_instances, num_bodies, "set_inertias", "body_inertia", dtype=wp.mat33f
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
        num_joints = 6
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Set up joint limits (required for _create_buffers)
        joint_limit_lower = torch.full((num_instances, num_joints), -1.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 1.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Call _create_buffers
        articulation._create_buffers()

        # Verify _ALL_INDICES
        expected_indices = torch.arange(num_instances, dtype=torch.long, device=device)
        assert articulation._ALL_INDICES.shape == (num_instances,)
        assert articulation._ALL_INDICES.dtype == torch.long
        assert articulation._ALL_INDICES.device.type == device.split(":")[0]
        torch.testing.assert_close(articulation._ALL_INDICES, expected_indices)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_create_buffers_soft_joint_limits_factor_1(self, device: str):
        """Test soft_joint_pos_limits with factor=1.0 (limits unchanged)."""
        num_instances = 2
        num_joints = 4
        soft_joint_pos_limit_factor = 1.0
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
            device=device,
        )

        # Set up joint limits: [-2.0, 2.0] for all joints
        joint_limit_lower = torch.full((num_instances, num_joints), -2.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 2.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Call _create_buffers
        articulation._create_buffers()

        # With factor=1.0, soft limits should equal hard limits
        # soft_joint_pos_limits is wp.vec2f (lower, upper)
        soft_limits = wp.to_torch(articulation.data.soft_joint_pos_limits)
        # Shape is (num_instances, num_joints, 2) after conversion
        expected_lower = torch.full((num_instances, num_joints), -2.0, device=device)
        expected_upper = torch.full((num_instances, num_joints), 2.0, device=device)
        torch.testing.assert_close(soft_limits[:, :, 0], expected_lower, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(soft_limits[:, :, 1], expected_upper, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_create_buffers_soft_joint_limits_factor_half(self, device: str):
        """Test soft_joint_pos_limits with factor=0.5 (limits halved around mean)."""
        num_instances = 2
        num_joints = 4
        soft_joint_pos_limit_factor = 0.5
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
            device=device,
        )

        # Set up joint limits: [-2.0, 2.0] for all joints
        # mean = 0.0, range = 4.0
        # soft_lower = 0.0 - 0.5 * 4.0 * 0.5 = -1.0
        # soft_upper = 0.0 + 0.5 * 4.0 * 0.5 = 1.0
        joint_limit_lower = torch.full((num_instances, num_joints), -2.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 2.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Call _create_buffers
        articulation._create_buffers()

        # Verify soft limits are halved
        soft_limits = wp.to_torch(articulation.data.soft_joint_pos_limits)
        expected_lower = torch.full((num_instances, num_joints), -1.0, device=device)
        expected_upper = torch.full((num_instances, num_joints), 1.0, device=device)
        torch.testing.assert_close(soft_limits[:, :, 0], expected_lower, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(soft_limits[:, :, 1], expected_upper, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_create_buffers_soft_joint_limits_asymmetric(self, device: str):
        """Test soft_joint_pos_limits with asymmetric joint limits."""
        num_instances = 2
        num_joints = 3
        soft_joint_pos_limit_factor = 0.8
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
            device=device,
        )

        # Set up asymmetric joint limits
        # Joint 0: [-3.14, 3.14] -> mean=0, range=6.28 -> soft: [-2.512, 2.512]
        # Joint 1: [-1.0, 2.0]   -> mean=0.5, range=3.0 -> soft: [0.5-1.2, 0.5+1.2] = [-0.7, 1.7]
        # Joint 2: [0.0, 1.0]    -> mean=0.5, range=1.0 -> soft: [0.5-0.4, 0.5+0.4] = [0.1, 0.9]
        joint_limit_lower = torch.tensor([[-3.14, -1.0, 0.0]] * num_instances, device=device)
        joint_limit_upper = torch.tensor([[3.14, 2.0, 1.0]] * num_instances, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Call _create_buffers
        articulation._create_buffers()

        # Calculate expected soft limits
        # soft_lower = mean - 0.5 * range * factor
        # soft_upper = mean + 0.5 * range * factor
        expected_lower = torch.tensor([[-2.512, -0.7, 0.1]] * num_instances, device=device)
        expected_upper = torch.tensor([[2.512, 1.7, 0.9]] * num_instances, device=device)

        soft_limits = wp.to_torch(articulation.data.soft_joint_pos_limits)
        torch.testing.assert_close(soft_limits[:, :, 0], expected_lower, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(soft_limits[:, :, 1], expected_upper, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_create_buffers_soft_joint_limits_factor_zero(self, device: str):
        """Test soft_joint_pos_limits with factor=0.0 (limits collapse to mean)."""
        num_instances = 2
        num_joints = 4
        soft_joint_pos_limit_factor = 0.0
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
            device=device,
        )

        # Set up joint limits: [-2.0, 2.0]
        # mean = 0.0, with factor=0.0, soft limits collapse to [0, 0]
        joint_limit_lower = torch.full((num_instances, num_joints), -2.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 2.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Call _create_buffers
        articulation._create_buffers()

        # With factor=0.0, soft limits should collapse to the mean
        soft_limits = wp.to_torch(articulation.data.soft_joint_pos_limits)
        expected_lower = torch.full((num_instances, num_joints), 0.0, device=device)
        expected_upper = torch.full((num_instances, num_joints), 0.0, device=device)
        torch.testing.assert_close(soft_limits[:, :, 0], expected_lower, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(soft_limits[:, :, 1], expected_upper, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_create_buffers_soft_joint_limits_per_joint_different(self, device: str):
        """Test soft_joint_pos_limits with different limits per joint."""
        num_instances = 3
        num_joints = 4
        soft_joint_pos_limit_factor = 0.9
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
            device=device,
        )

        # Each joint has different limits
        joint_limit_lower = torch.tensor([[-1.0, -2.0, -0.5, -3.0]] * num_instances, device=device)
        joint_limit_upper = torch.tensor([[1.0, 2.0, 0.5, 3.0]] * num_instances, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Call _create_buffers
        articulation._create_buffers()

        # Calculate expected: soft_lower/upper = mean ± 0.5 * range * factor
        # Joint 0: mean=0, range=2 -> [0 - 0.9, 0 + 0.9] = [-0.9, 0.9]
        # Joint 1: mean=0, range=4 -> [0 - 1.8, 0 + 1.8] = [-1.8, 1.8]
        # Joint 2: mean=0, range=1 -> [0 - 0.45, 0 + 0.45] = [-0.45, 0.45]
        # Joint 3: mean=0, range=6 -> [0 - 2.7, 0 + 2.7] = [-2.7, 2.7]
        expected_lower = torch.tensor([[-0.9, -1.8, -0.45, -2.7]] * num_instances, device=device)
        expected_upper = torch.tensor([[0.9, 1.8, 0.45, 2.7]] * num_instances, device=device)

        soft_limits = wp.to_torch(articulation.data.soft_joint_pos_limits)
        torch.testing.assert_close(soft_limits[:, :, 0], expected_lower, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(soft_limits[:, :, 1], expected_upper, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_create_buffers_single_environment(self, device: str):
        """Test _create_buffers with a single environment."""
        num_instances = 1
        num_joints = 6
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        joint_limit_lower = torch.full((num_instances, num_joints), -1.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 1.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Call _create_buffers
        articulation._create_buffers()

        # Verify _ALL_INDICES has single element
        assert articulation._ALL_INDICES.shape == (1,)
        assert articulation._ALL_INDICES[0].item() == 0

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_create_buffers_large_number_of_environments(self, device: str):
        """Test _create_buffers with a large number of environments."""
        num_instances = 1024
        num_joints = 12
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        joint_limit_lower = torch.full((num_instances, num_joints), -1.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 1.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Call _create_buffers
        articulation._create_buffers()

        # Verify _ALL_INDICES
        expected_indices = torch.arange(num_instances, dtype=torch.long, device=device)
        assert articulation._ALL_INDICES.shape == (num_instances,)
        torch.testing.assert_close(articulation._ALL_INDICES, expected_indices)

        # Verify soft limits shape
        soft_limits = wp.to_torch(articulation.data.soft_joint_pos_limits)
        assert soft_limits.shape == (num_instances, num_joints, 2)


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
        num_joints = 4
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Set up init_state with specific position and rotation
        # Rotation is in (x, y, z, w) format in the config
        articulation.cfg.init_state.pos = (1.0, 2.0, 3.0)
        articulation.cfg.init_state.rot = (0.0, 0.707, 0.0, 0.707)  # x, y, z, w

        # Call _process_cfg
        articulation._process_cfg()

        # Verify the default root pose
        # Expected: position (1, 2, 3) + quaternion in (x, y, z, w) = (0, 0.707, 0, 0.707)
        expected_pose = torch.tensor(
            [[1.0, 2.0, 3.0, 0.0, 0.707, 0.0, 0.707]] * num_instances,
            device=device,
        )
        result = wp.to_torch(articulation.data.default_root_pose)
        assert result.allclose(expected_pose, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_default_root_velocity(self, device: str):
        """Test that _process_cfg correctly sets default root velocity."""
        num_instances = 2
        num_joints = 4
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Set up init_state with specific velocities
        articulation.cfg.init_state.lin_vel = (1.0, 2.0, 3.0)
        articulation.cfg.init_state.ang_vel = (0.1, 0.2, 0.3)

        # Call _process_cfg
        articulation._process_cfg()

        # Verify the default root velocity
        # Expected: lin_vel + ang_vel = (1, 2, 3, 0.1, 0.2, 0.3)
        expected_vel = torch.tensor(
            [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]] * num_instances,
            device=device,
        )
        result = wp.to_torch(articulation.data.default_root_vel)
        assert result.allclose(expected_vel, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_default_joint_positions_all_joints(self, device: str):
        """Test that _process_cfg correctly sets default joint positions for all joints."""
        num_instances = 2
        num_joints = 4
        joint_names = ["joint_0", "joint_1", "joint_2", "joint_3"]
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            joint_names=joint_names,
            device=device,
        )

        # Set up init_state with joint positions using wildcard pattern
        articulation.cfg.init_state.joint_pos = {".*": 0.5}
        articulation.cfg.init_state.joint_vel = {".*": 0.0}

        # Call _process_cfg
        articulation._process_cfg()

        # Verify the default joint positions
        expected_pos = torch.full((num_instances, num_joints), 0.5, device=device)
        result = wp.to_torch(articulation.data.default_joint_pos)
        assert result.allclose(expected_pos, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_default_joint_positions_specific_joints(self, device: str):
        """Test that _process_cfg correctly sets default joint positions for specific joints."""
        num_instances = 2
        num_joints = 4
        joint_names = ["shoulder", "elbow", "wrist", "gripper"]
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            joint_names=joint_names,
            device=device,
        )

        # Set up init_state with specific joint positions
        articulation.cfg.init_state.joint_pos = {
            "shoulder": 1.0,
            "elbow": 2.0,
            "wrist": 3.0,
            "gripper": 4.0,
        }
        articulation.cfg.init_state.joint_vel = {".*": 0.0}

        # Call _process_cfg
        articulation._process_cfg()

        # Verify the default joint positions
        expected_pos = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * num_instances, device=device)
        result = wp.to_torch(articulation.data.default_joint_pos)
        assert result.allclose(expected_pos, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_default_joint_positions_regex_pattern(self, device: str):
        """Test that _process_cfg correctly handles regex patterns for joint positions."""
        num_instances = 2
        num_joints = 6
        joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "hand_joint_1", "hand_joint_2", "hand_joint_3"]
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            joint_names=joint_names,
            device=device,
        )

        # Set up init_state with regex patterns
        articulation.cfg.init_state.joint_pos = {
            "arm_joint_.*": 1.5,
            "hand_joint_.*": 0.5,
        }
        articulation.cfg.init_state.joint_vel = {".*": 0.0}

        # Call _process_cfg
        articulation._process_cfg()

        # Verify the default joint positions
        # arm joints (indices 0-2) should be 1.5, hand joints (indices 3-5) should be 0.5
        expected_pos = torch.tensor([[1.5, 1.5, 1.5, 0.5, 0.5, 0.5]] * num_instances, device=device)
        result = wp.to_torch(articulation.data.default_joint_pos)
        assert result.allclose(expected_pos, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_default_joint_velocities(self, device: str):
        """Test that _process_cfg correctly sets default joint velocities."""
        num_instances = 2
        num_joints = 4
        joint_names = ["joint_0", "joint_1", "joint_2", "joint_3"]
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            joint_names=joint_names,
            device=device,
        )

        # Set up init_state with joint velocities
        articulation.cfg.init_state.joint_pos = {".*": 0.0}
        articulation.cfg.init_state.joint_vel = {
            "joint_0": 0.1,
            "joint_1": 0.2,
            "joint_2": 0.3,
            "joint_3": 0.4,
        }

        # Call _process_cfg
        articulation._process_cfg()

        # Verify the default joint velocities
        expected_vel = torch.tensor([[0.1, 0.2, 0.3, 0.4]] * num_instances, device=device)
        result = wp.to_torch(articulation.data.default_joint_vel)
        assert result.allclose(expected_vel, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_identity_quaternion(self, device: str):
        """Test that _process_cfg correctly handles identity quaternion."""
        num_instances = 2
        num_joints = 2
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Set up init_state with identity quaternion (x=0, y=0, z=0, w=1)
        articulation.cfg.init_state.pos = (0.0, 0.0, 0.0)
        articulation.cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # Identity: x, y, z, w

        # Call _process_cfg
        articulation._process_cfg()

        # Verify the default root pose
        # Expected: position (0, 0, 0) + quaternion in (x, y, z, w) = (0, 0, 0, 1)
        expected_pose = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * num_instances,
            device=device,
        )
        result = wp.to_torch(articulation.data.default_root_pose)
        assert result.allclose(expected_pose, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_zero_joints(self, device: str):
        """Test that _process_cfg handles articulation with no joints."""
        num_instances = 2
        num_joints = 0
        num_bodies = 1
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            num_bodies=num_bodies,
            device=device,
        )

        # Set up init_state
        articulation.cfg.init_state.pos = (1.0, 2.0, 3.0)
        articulation.cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w
        articulation.cfg.init_state.lin_vel = (0.5, 0.5, 0.5)
        articulation.cfg.init_state.ang_vel = (0.1, 0.1, 0.1)
        articulation.cfg.init_state.joint_pos = {}
        articulation.cfg.init_state.joint_vel = {}

        # Call _process_cfg - should not raise any exception
        articulation._process_cfg()

        # Verify root pose and velocity are still set correctly
        expected_pose = torch.tensor(
            [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]] * num_instances,
            device=device,
        )
        expected_vel = torch.tensor(
            [[0.5, 0.5, 0.5, 0.1, 0.1, 0.1]] * num_instances,
            device=device,
        )
        assert wp.to_torch(articulation.data.default_root_pose).allclose(expected_pose, atol=1e-5, rtol=1e-5)
        assert wp.to_torch(articulation.data.default_root_vel).allclose(expected_vel, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_cfg_mixed_joint_patterns(self, device: str):
        """Test that _process_cfg handles mixed specific and pattern-based joint settings."""
        num_instances = 2
        num_joints = 5
        joint_names = ["base_joint", "arm_1", "arm_2", "hand_1", "hand_2"]
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            joint_names=joint_names,
            device=device,
        )

        # Set up init_state with mixed patterns
        articulation.cfg.init_state.joint_pos = {
            "base_joint": 0.0,
            "arm_.*": 1.0,
            "hand_.*": 2.0,
        }
        articulation.cfg.init_state.joint_vel = {".*": 0.0}

        # Call _process_cfg
        articulation._process_cfg()

        # Verify the default joint positions
        expected_pos = torch.tensor([[0.0, 1.0, 1.0, 2.0, 2.0]] * num_instances, device=device)
        result = wp.to_torch(articulation.data.default_joint_pos)
        assert result.allclose(expected_pos, atol=1e-5, rtol=1e-5)


class TestValidateCfg:
    """Tests for _validate_cfg method.

    Tests that the configuration validation correctly catches:
    - Default joint positions outside of joint limits (lower and upper bounds)
    - Various edge cases with joint limits
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_validate_cfg_positions_within_limits(self, device: str):
        """Test that _validate_cfg passes when all default positions are within limits."""
        num_instances = 2
        num_joints = 6
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Set joint limits: [-1.0, 1.0] for all joints
        joint_limit_lower = torch.full((num_instances, num_joints), -1.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 1.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Set default joint positions within limits
        default_joint_pos = torch.zeros((num_instances, num_joints), device=device)
        articulation.data._default_joint_pos = wp.from_torch(default_joint_pos, dtype=wp.float32)

        # Should not raise any exception
        articulation._validate_cfg()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_validate_cfg_position_below_lower_limit(self, device: str):
        """Test that _validate_cfg raises ValueError when a position is below the lower limit."""
        num_instances = 2
        num_joints = 6
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Set joint limits: [-1.0, 1.0] for all joints
        joint_limit_lower = torch.full((num_instances, num_joints), -1.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 1.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Set default joint position for joint 2 below the lower limit
        default_joint_pos = torch.zeros((num_instances, num_joints), device=device)
        default_joint_pos[:, 2] = -1.5  # Below -1.0 lower limit
        articulation.data._default_joint_pos = wp.from_torch(default_joint_pos, dtype=wp.float32)

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            articulation._validate_cfg()
        assert "joint_2" in str(exc_info.value)
        assert "-1.500" in str(exc_info.value)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_validate_cfg_position_above_upper_limit(self, device: str):
        """Test that _validate_cfg raises ValueError when a position is above the upper limit."""
        num_instances = 2
        num_joints = 6
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Set joint limits: [-1.0, 1.0] for all joints
        joint_limit_lower = torch.full((num_instances, num_joints), -1.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 1.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Set default joint position for joint 4 above the upper limit
        default_joint_pos = torch.zeros((num_instances, num_joints), device=device)
        default_joint_pos[:, 4] = 1.5  # Above 1.0 upper limit
        articulation.data._default_joint_pos = wp.from_torch(default_joint_pos, dtype=wp.float32)

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            articulation._validate_cfg()
        assert "joint_4" in str(exc_info.value)
        assert "1.500" in str(exc_info.value)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_validate_cfg_multiple_positions_out_of_limits(self, device: str):
        """Test that _validate_cfg reports all joints with positions outside limits."""
        num_instances = 2
        num_joints = 6
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Set joint limits: [-1.0, 1.0] for all joints
        joint_limit_lower = torch.full((num_instances, num_joints), -1.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 1.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Set multiple joints out of limits
        default_joint_pos = torch.zeros((num_instances, num_joints), device=device)
        default_joint_pos[:, 0] = -2.0  # Below lower limit
        default_joint_pos[:, 3] = 2.0  # Above upper limit
        default_joint_pos[:, 5] = -1.5  # Below lower limit
        articulation.data._default_joint_pos = wp.from_torch(default_joint_pos, dtype=wp.float32)

        # Should raise ValueError mentioning all violated joints
        with pytest.raises(ValueError) as exc_info:
            articulation._validate_cfg()
        error_msg = str(exc_info.value)
        assert "joint_0" in error_msg
        assert "joint_3" in error_msg
        assert "joint_5" in error_msg

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_validate_cfg_asymmetric_limits(self, device: str):
        """Test that _validate_cfg works with asymmetric joint limits."""
        num_instances = 2
        num_joints = 4
        joint_names = ["shoulder", "elbow", "wrist", "gripper"]
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            joint_names=joint_names,
            device=device,
        )

        # Set asymmetric joint limits for each joint
        joint_limit_lower = torch.tensor([[-3.14, -2.0, -1.5, 0.0]] * num_instances, device=device)
        joint_limit_upper = torch.tensor([[3.14, 0.5, 1.5, 0.1]] * num_instances, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Set positions within asymmetric limits
        default_joint_pos = torch.tensor([[0.0, -1.0, 0.0, 0.05]] * num_instances, device=device)
        articulation.data._default_joint_pos = wp.from_torch(default_joint_pos, dtype=wp.float32)

        # Should not raise any exception
        articulation._validate_cfg()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_validate_cfg_asymmetric_limits_violated(self, device: str):
        """Test that _validate_cfg catches violations with asymmetric limits."""
        num_instances = 2
        num_joints = 4
        joint_names = ["shoulder", "elbow", "wrist", "gripper"]
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            joint_names=joint_names,
            device=device,
        )

        # Set asymmetric joint limits: elbow has range [-2.0, 0.5]
        joint_limit_lower = torch.tensor([[-3.14, -2.0, -1.5, 0.0]] * num_instances, device=device)
        joint_limit_upper = torch.tensor([[3.14, 0.5, 1.5, 0.1]] * num_instances, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Set elbow position above its upper limit (0.5)
        default_joint_pos = torch.tensor([[0.0, 1.0, 0.0, 0.05]] * num_instances, device=device)
        articulation.data._default_joint_pos = wp.from_torch(default_joint_pos, dtype=wp.float32)

        # Should raise ValueError for elbow
        with pytest.raises(ValueError) as exc_info:
            articulation._validate_cfg()
        assert "elbow" in str(exc_info.value)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_validate_cfg_single_joint(self, device: str):
        """Test _validate_cfg with a single joint articulation."""
        num_instances = 2
        num_joints = 1
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Set joint limits
        joint_limit_lower = torch.full((num_instances, num_joints), -0.5, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), 0.5, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Set position outside limits
        default_joint_pos = torch.full((num_instances, num_joints), 1.0, device=device)
        articulation.data._default_joint_pos = wp.from_torch(default_joint_pos, dtype=wp.float32)

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            articulation._validate_cfg()
        assert "joint_0" in str(exc_info.value)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_validate_cfg_negative_range_limits(self, device: str):
        """Test _validate_cfg with limits entirely in the negative range."""
        num_instances = 2
        num_joints = 2
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Set limits entirely in negative range
        joint_limit_lower = torch.full((num_instances, num_joints), -5.0, device=device)
        joint_limit_upper = torch.full((num_instances, num_joints), -2.0, device=device)
        mock_view.set_mock_data(
            joint_limit_lower=wp.from_torch(joint_limit_lower, dtype=wp.float32),
            joint_limit_upper=wp.from_torch(joint_limit_upper, dtype=wp.float32),
        )

        # Set position at zero (outside negative-only limits)
        default_joint_pos = torch.zeros((num_instances, num_joints), device=device)
        articulation.data._default_joint_pos = wp.from_torch(default_joint_pos, dtype=wp.float32)

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            articulation._validate_cfg()
        # Both joints should be reported as violated
        assert "joint_0" in str(exc_info.value)
        assert "joint_1" in str(exc_info.value)


# TODO: Expand these tests when tendons are available in Newton.
#       Currently, tendons are not implemented and _process_tendons only initializes empty lists.
#       When tendon support is added, tests should verify:
#       - Fixed tendon properties are correctly parsed and stored
#       - Spatial tendon properties are correctly parsed and stored
#       - Tendon limits and stiffness values are correctly set
class TestProcessTendons:
    """Tests for _process_tendons method.

    Note: Tendons are not yet implemented in Newton. These tests verify the current
    placeholder behavior. When tendons are implemented, these tests should be expanded.
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_tendons_initializes_empty_lists(self, device: str):
        """Test that _process_tendons initializes empty tendon name lists."""
        num_instances = 2
        num_joints = 4
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Call _process_tendons
        articulation._process_tendons()

        # Verify empty lists are created
        assert articulation._fixed_tendon_names == []
        assert articulation._spatial_tendon_names == []

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_tendons_returns_none(self, device: str):
        """Test that _process_tendons returns None (no tendons implemented)."""
        num_instances = 2
        num_joints = 4
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Call _process_tendons and verify return value
        result = articulation._process_tendons()
        assert result is None


# TODO: Expand these tests when actuator mocking is more mature.
#       Full actuator integration tests would require:
#       - Mocking ActuatorBaseCfg and ActuatorBase classes
#       - Testing implicit vs explicit actuator behavior
#       - Testing stiffness/damping propagation
#       Currently, we test the initialization behavior without actuators configured.
class TestProcessActuatorsCfg:
    """Tests for _process_actuators_cfg method.

    Note: These tests focus on the initialization behavior when no actuators are configured.
    Full actuator integration tests require additional mocking infrastructure.
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_actuators_cfg_initializes_empty_dict(self, device: str):
        """Test that _process_actuators_cfg initializes actuators as empty dict when none configured."""
        num_instances = 2
        num_joints = 4
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # Ensure no actuators are configured
        articulation.cfg.actuators = {}

        # Call _process_actuators_cfg
        articulation._process_actuators_cfg()

        # Verify actuators dict is empty
        assert articulation.actuators == {}
        assert isinstance(articulation.actuators, dict)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_actuators_cfg_sets_implicit_flag_false(self, device: str):
        """Test that _process_actuators_cfg sets _has_implicit_actuators to False initially."""
        num_instances = 2
        num_joints = 4
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        articulation.cfg.actuators = {}

        # Call _process_actuators_cfg
        articulation._process_actuators_cfg()

        # Verify flag is set to False
        assert articulation._has_implicit_actuators is False

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_actuators_cfg_sets_joint_limit_gains(self, device: str):
        """Test that _process_actuators_cfg sets joint_limit_ke and joint_limit_kd."""
        num_instances = 2
        num_joints = 4
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        articulation.cfg.actuators = {}

        # Call _process_actuators_cfg
        articulation._process_actuators_cfg()

        # Verify joint limit gains are set
        joint_limit_ke = wp.to_torch(mock_view.get_attribute("joint_limit_ke", None))
        joint_limit_kd = wp.to_torch(mock_view.get_attribute("joint_limit_kd", None))

        expected_ke = torch.full((num_instances, num_joints), 2500.0, device=device)
        expected_kd = torch.full((num_instances, num_joints), 100.0, device=device)

        torch.testing.assert_close(joint_limit_ke, expected_ke, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(joint_limit_kd, expected_kd, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_actuators_cfg_warns_unactuated_joints(self, device: str):
        """Test that _process_actuators_cfg warns when not all joints have actuators."""
        num_instances = 2
        num_joints = 4
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            device=device,
        )

        # No actuators configured but we have joints
        articulation.cfg.actuators = {}

        # Should warn about unactuated joints
        with pytest.warns(UserWarning, match="Not all actuators are configured"):
            articulation._process_actuators_cfg()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_process_actuators_cfg_no_warning_zero_joints(self, device: str):
        """Test that _process_actuators_cfg does not warn when there are no joints."""
        num_instances = 2
        num_joints = 0
        num_bodies = 1
        articulation, mock_view, _ = create_test_articulation(
            num_instances=num_instances,
            num_joints=num_joints,
            num_bodies=num_bodies,
            device=device,
        )

        articulation.cfg.actuators = {}

        # Should not warn when there are no joints to actuate
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # This should not raise a warning
            articulation._process_actuators_cfg()


##
# Main
##

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
