# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RigidObjectData class comparing Newton implementation against PhysX reference."""

from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

import pytest
import warp as wp

# Import mock classes from common test utilities
from common.mock_newton import MockNewtonArticulationView, MockNewtonModel
from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData

# TODO: Remove this import
from isaaclab.utils import math as math_utils

# Initialize Warp
wp.init()


##
# Test Fixtures
##


@pytest.fixture
def mock_newton_manager():
    """Create mock NewtonManager with necessary methods."""
    mock_model = MockNewtonModel()
    mock_state = MagicMock()
    mock_control = MagicMock()

    # Patch where NewtonManager is used (in the rigid object data module)
    with patch("isaaclab_newton.assets.rigid_object.rigid_object_data.NewtonManager") as MockManager:
        MockManager.get_model.return_value = mock_model
        MockManager.get_state_0.return_value = mock_state
        MockManager.get_control.return_value = mock_control
        MockManager.get_dt.return_value = 0.01
        yield MockManager


##
# Test Cases -- Defaults.
##


class TestDefaults:
    """Tests the following properties:
    - default_root_pose
    - default_root_vel

    Runs the following checks:
    - Checks that by default, the properties are all zero.
    - Checks that the properties are settable.
    - Checks that once the rigid object data is primed, the properties cannot be changed.
    """

    def _setup_method(self, num_instances: int, device: str) -> RigidObjectData:
        mock_view = MockNewtonArticulationView(num_instances, 1, 0, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )

        return rigid_object_data

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_zero_instantiated(self, mock_newton_manager, num_instances: int, device: str):
        """Test zero instantiated rigid object data."""
        # Setup the rigid object data
        rigid_object_data = self._setup_method(num_instances, device)
        # Check the types are correct
        assert rigid_object_data.default_root_pose.dtype is wp.transformf
        assert rigid_object_data.default_root_vel.dtype is wp.spatial_vectorf
        # Check the shapes are correct
        assert rigid_object_data.default_root_pose.shape == (num_instances,)
        assert rigid_object_data.default_root_vel.shape == (num_instances,)
        # Check the values are zero
        assert torch.all(
            wp.to_torch(rigid_object_data.default_root_pose) == torch.zeros(num_instances, 7, device=device)
        )
        assert torch.all(
            wp.to_torch(rigid_object_data.default_root_vel) == torch.zeros(num_instances, 6, device=device)
        )

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_settable(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that the rigid object data is settable."""
        # Setup the rigid object data
        rigid_object_data = self._setup_method(num_instances, device)
        # Set the default values
        rigid_object_data.default_root_pose = wp.ones(num_instances, dtype=wp.transformf, device=device)
        rigid_object_data.default_root_vel = wp.ones(num_instances, dtype=wp.spatial_vectorf, device=device)
        # Check the types are correct
        assert rigid_object_data.default_root_pose.dtype is wp.transformf
        assert rigid_object_data.default_root_vel.dtype is wp.spatial_vectorf
        # Check the shapes are correct
        assert rigid_object_data.default_root_pose.shape == (num_instances,)
        assert rigid_object_data.default_root_vel.shape == (num_instances,)
        # Check the values are set
        assert torch.all(
            wp.to_torch(rigid_object_data.default_root_pose) == torch.ones(num_instances, 7, device=device)
        )
        assert torch.all(wp.to_torch(rigid_object_data.default_root_vel) == torch.ones(num_instances, 6, device=device))
        # Prime the rigid object data
        rigid_object_data.is_primed = True
        # Check that the values cannot be changed
        with pytest.raises(RuntimeError):
            rigid_object_data.default_root_pose = wp.zeros(num_instances, dtype=wp.transformf, device=device)
        with pytest.raises(RuntimeError):
            rigid_object_data.default_root_vel = wp.zeros(num_instances, dtype=wp.spatial_vectorf, device=device)


##
# Test Cases -- Root state properties.
##


class TestRootLinkPoseW:
    """Tests the root link pose property

    This value is read from the simulation. There is no math to check for.

    Runs the following checks:
    - Checks that the returned value is a pointer to the internal data.
    - Checks that the returned value is correct.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )

        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_root_link_pose_w(self, mock_newton_manager, num_instances: int, device: str):
        """Test that the root link pose property returns a pointer to the internal data."""
        rigid_object_data, _ = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.root_link_pose_w.shape == (num_instances,)
        assert rigid_object_data.root_link_pose_w.dtype == wp.transformf

        # Mock data is initialized to zeros
        assert torch.all(
            wp.to_torch(rigid_object_data.root_link_pose_w) == torch.zeros((num_instances, 7), device=device)
        )

        # Get the property
        root_link_pose_w = rigid_object_data.root_link_pose_w

        # Assign a different value to the internal data
        rigid_object_data.root_link_pose_w.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        assert torch.all(
            wp.to_torch(rigid_object_data.root_link_pose_w) == torch.ones((num_instances, 7), device=device)
        )

        # Assign a different value to the pointers
        root_link_pose_w.fill_(2.0)

        # Check that the internal data has been updated
        assert torch.all(
            wp.to_torch(rigid_object_data.root_link_pose_w) == torch.ones((num_instances, 7), device=device) * 2.0
        )


class TestRootLinkVelW:
    """Tests the root link velocity property

    This value is derived from the root center of mass velocity. To ensure that the value is correctly computed,
    we will compare the calculated value to the one currently calculated in the version 2.3.1 of IsaacLab.

    Runs the following checks:
    - Checks that the returned value is a pointer to the internal data.
    - Checks that the returned value is correct.
    - Checks that the timestamp is updated correctly.
    - Checks that the data is invalidated when the timestamp is updated.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that the root link velocity property is correctly computed."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.root_link_vel_w.shape == (num_instances,)
        assert rigid_object_data.root_link_vel_w.dtype == wp.spatial_vectorf

        # Mock data is initialized to zeros
        assert torch.all(
            wp.to_torch(rigid_object_data.root_link_vel_w) == torch.zeros((num_instances, 6), device=device)
        )

        for i in range(10):
            rigid_object_data._sim_timestamp = i + 1.0
            # Generate random com velocity and body com position
            com_vel = torch.rand((num_instances, 6), device=device)
            body_com_pos = torch.rand((num_instances, 1, 3), device=device)
            root_link_pose = torch.zeros((num_instances, 7), device=device)
            root_link_pose[:, 3:] = torch.randn((num_instances, 4), device=device)
            root_link_pose[:, 3:] = torch.nn.functional.normalize(root_link_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)
            mock_view.set_mock_data(
                root_transforms=wp.from_torch(root_link_pose, dtype=wp.transformf),
                root_velocities=wp.from_torch(com_vel, dtype=wp.spatial_vectorf),
                body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
            )

            # Use the original IsaacLab code to compute the root link velocities
            vel = com_vel.clone()
            # TODO: Move the function from math_utils to a test utils file. Decoupling it from changes in math_utils.
            vel[:, :3] += torch.linalg.cross(
                vel[:, 3:], math_utils.quat_apply(root_link_pose[:, 3:], -body_com_pos[:, 0]), dim=-1
            )

            # Compare the computed value to the one from the rigid object data
            assert torch.allclose(wp.to_torch(rigid_object_data.root_link_vel_w), vel, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_update_timestamp(self, mock_newton_manager, device: str):
        """Test that the timestamp is updated correctly."""
        rigid_object_data, mock_view = self._setup_method(1, device)

        # Check that the timestamp is initialized to -1.0
        assert rigid_object_data._root_link_vel_w.timestamp == -1.0

        # Check that the data class timestamp is initialized to 0.0
        assert rigid_object_data._sim_timestamp == 0.0

        # Request the property
        value = wp.to_torch(rigid_object_data.root_link_vel_w).clone()

        # Check that the timestamp is updated. The timestamp should be the same as the data class timestamp.
        assert rigid_object_data._root_link_vel_w.timestamp == rigid_object_data._sim_timestamp

        # Update the root_com_vel_w
        mock_view.set_mock_data(
            root_velocities=wp.from_torch(torch.rand((1, 6), device=device), dtype=wp.spatial_vectorf),
        )

        # Check that the property value was not updated
        assert torch.all(wp.to_torch(rigid_object_data.root_link_vel_w) == value)

        # Update the data class timestamp
        rigid_object_data._sim_timestamp = 1.0

        # Check that the property timestamp was not updated
        assert rigid_object_data._root_link_vel_w.timestamp != rigid_object_data._sim_timestamp

        # Check that the property value was updated
        assert torch.all(wp.to_torch(rigid_object_data.root_link_vel_w) != value)


class TestRootComPoseW:
    """Tests the root center of mass pose property

    This value is derived from the root link pose and the body com position. To ensure that the value is correctly computed,
    we will compare the calculated value to the one currently calculated in the version 2.3.1 of IsaacLab.

    Runs the following checks:
    - Checks that the returned value is a pointer to the internal data.
    - Checks that the returned value is correct.
    - Checks that the timestamp is updated correctly.
    - Checks that the data is invalidated when the timestamp is updated.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_root_com_pose_w(self, mock_newton_manager, num_instances: int, device: str):
        """Test that the root center of mass pose property returns a pointer to the internal data."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.root_com_pose_w.shape == (num_instances,)
        assert rigid_object_data.root_com_pose_w.dtype == wp.transformf

        # Mock data is initialized to zeros
        assert torch.all(
            wp.to_torch(rigid_object_data.root_com_pose_w) == torch.zeros((num_instances, 7), device=device)
        )

        for i in range(10):
            rigid_object_data._sim_timestamp = i + 1.0
            # Generate random root link pose and body com position
            root_link_pose = torch.zeros((num_instances, 7), device=device)
            root_link_pose[:, :3] = torch.rand((num_instances, 3), device=device)
            root_link_pose[:, 3:] = torch.randn((num_instances, 4), device=device)
            root_link_pose[:, 3:] = torch.nn.functional.normalize(root_link_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)
            body_com_pos = torch.rand((num_instances, 1, 3), device=device)
            mock_view.set_mock_data(
                root_transforms=wp.from_torch(root_link_pose, dtype=wp.transformf),
                body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
            )

            # Use the original IsaacLab code to compute the root center of mass pose
            root_link_pos_w = root_link_pose[:, :3]
            root_link_quat_w = root_link_pose[:, 3:]
            body_com_pos_b = body_com_pos.clone()
            body_com_quat_b = torch.zeros((num_instances, 1, 4), device=device)
            body_com_quat_b[:, :, 3] = 1.0
            # --- IL 2.3.1 code ---
            pos, quat = math_utils.combine_frame_transforms(
                root_link_pos_w, root_link_quat_w, body_com_pos_b[:, 0], body_com_quat_b[:, 0]
            )
            # ---
            root_com_pose = torch.cat((pos, quat), dim=-1)

            # Compare the computed value to the one from the rigid object data
            assert torch.allclose(wp.to_torch(rigid_object_data.root_com_pose_w), root_com_pose, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_update_timestamp(self, mock_newton_manager, device: str):
        """Test that the timestamp is updated correctly."""
        rigid_object_data, mock_view = self._setup_method(1, device)

        # Check that the timestamp is initialized to -1.0
        assert rigid_object_data._root_com_pose_w.timestamp == -1.0

        # Check that the data class timestamp is initialized to 0.0
        assert rigid_object_data._sim_timestamp == 0.0

        # Request the property
        value = wp.to_torch(rigid_object_data.root_com_pose_w).clone()

        # Check that the timestamp is updated. The timestamp should be the same as the data class timestamp.
        assert rigid_object_data._root_com_pose_w.timestamp == rigid_object_data._sim_timestamp

        # Update the root_com_vel_w
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(torch.rand((1, 7), device=device), dtype=wp.transformf),
        )

        # Check that the property value was not updated
        assert torch.all(wp.to_torch(rigid_object_data.root_com_pose_w) == value)

        # Update the data class timestamp
        rigid_object_data._sim_timestamp = 1.0

        # Check that the property timestamp was not updated
        assert rigid_object_data._root_com_pose_w.timestamp != rigid_object_data._sim_timestamp

        # Check that the property value was updated
        assert torch.all(wp.to_torch(rigid_object_data.root_com_pose_w) != value)


class TestRootComVelW:
    """Tests the root center of mass velocity property

    This value is read from the simulation. There is no math to check for.

    Checks that the returned value is a pointer to the internal data.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_root_com_vel_w(self, mock_newton_manager, num_instances: int, device: str):
        """Test that the root center of mass velocity property returns a pointer to the internal data."""
        rigid_object_data, _ = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.root_com_vel_w.shape == (num_instances,)
        assert rigid_object_data.root_com_vel_w.dtype == wp.spatial_vectorf

        # Mock data is initialized to zeros
        assert torch.all(
            wp.to_torch(rigid_object_data.root_com_vel_w) == torch.zeros((num_instances, 6), device=device)
        )

        # Get the property
        root_com_vel_w = rigid_object_data.root_com_vel_w

        # Assign a different value to the internal data
        rigid_object_data.root_com_vel_w.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        assert torch.all(wp.to_torch(rigid_object_data.root_com_vel_w) == torch.ones((num_instances, 6), device=device))

        # Assign a different value to the pointers
        root_com_vel_w.fill_(2.0)

        # Check that the internal data has been updated
        assert torch.all(
            wp.to_torch(rigid_object_data.root_com_vel_w) == torch.ones((num_instances, 6), device=device) * 2.0
        )


class TestRootState:
    """Tests the root state properties

    Test the root state properties are correctly updated from the pose and velocity properties.
    Tests the following properties:
    - root_state_w
    - root_link_state_w
    - root_com_state_w

    For each property, we run the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly assembled from pose and velocity.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_root_state_properties(self, mock_newton_manager, num_instances: int, device: str):
        """Test that all root state properties correctly combine pose and velocity."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Generate random mock data
        for i in range(5):
            rigid_object_data._sim_timestamp = i + 1.0

            # Generate random root link pose
            root_link_pose = torch.zeros((num_instances, 7), device=device)
            root_link_pose[:, :3] = torch.rand((num_instances, 3), device=device)
            root_link_pose[:, 3:] = torch.randn((num_instances, 4), device=device)
            root_link_pose[:, 3:] = torch.nn.functional.normalize(root_link_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)

            # Generate random velocities and com position
            com_vel = torch.rand((num_instances, 6), device=device)
            body_com_pos = torch.rand((num_instances, 1, 3), device=device)

            mock_view.set_mock_data(
                root_transforms=wp.from_torch(root_link_pose, dtype=wp.transformf),
                root_velocities=wp.from_torch(com_vel, dtype=wp.spatial_vectorf),
                body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
            )

            # --- Test root_state_w ---
            # Combines root_link_pose_w with root_com_vel_w
            root_state = wp.to_torch(rigid_object_data.root_state_w)
            expected_root_state = torch.cat([root_link_pose, com_vel], dim=-1)

            assert root_state.shape == (num_instances, 13)
            assert torch.allclose(root_state, expected_root_state, atol=1e-6, rtol=1e-6)

            # --- Test root_link_state_w ---
            # Combines root_link_pose_w with root_link_vel_w
            root_link_state = wp.to_torch(rigid_object_data.root_link_state_w)

            # Compute expected root_link_vel from com_vel (same as TestRootLinkVelW)
            root_link_vel = com_vel.clone()
            root_link_vel[:, :3] += torch.linalg.cross(
                root_link_vel[:, 3:], math_utils.quat_apply(root_link_pose[:, 3:], -body_com_pos[:, 0]), dim=-1
            )
            expected_root_link_state = torch.cat([root_link_pose, root_link_vel], dim=-1)

            assert root_link_state.shape == (num_instances, 13)
            assert torch.allclose(root_link_state, expected_root_link_state, atol=1e-6, rtol=1e-6)

            # --- Test root_com_state_w ---
            # Combines root_com_pose_w with root_com_vel_w
            root_com_state = wp.to_torch(rigid_object_data.root_com_state_w)

            # Compute expected root_com_pose from root_link_pose and body_com_pos (same as TestRootComPoseW)
            body_com_quat_b = torch.zeros((num_instances, 4), device=device)
            body_com_quat_b[:, 3] = 1.0
            root_com_pos, root_com_quat = math_utils.combine_frame_transforms(
                root_link_pose[:, :3], root_link_pose[:, 3:], body_com_pos[:, 0], body_com_quat_b
            )
            expected_root_com_state = torch.cat([root_com_pos, root_com_quat, com_vel], dim=-1)

            assert root_com_state.shape == (num_instances, 13)
            assert torch.allclose(root_com_state, expected_root_com_state, atol=1e-6, rtol=1e-6)


##
# Test Cases -- Body state properties.
##


class TestBodyMassInertia:
    """Tests the body mass and inertia properties.

    These values are read directly from the simulation bindings.

    Tests the following properties:
    - body_mass
    - body_inertia

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is a reference to the internal data.
    """

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str
    ) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_body_mass_and_inertia(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that body_mass and body_inertia have correct types, shapes, and reference behavior."""
        rigid_object_data, mock_view = self._setup_method(num_instances, num_bodies, device)

        # --- Test body_mass ---
        # Check the type and shape
        assert rigid_object_data.body_mass.shape == (num_instances, num_bodies)
        assert rigid_object_data.body_mass.dtype == wp.float32

        # Mock data initializes body_mass to ones
        assert torch.all(
            wp.to_torch(rigid_object_data.body_mass) == torch.zeros((num_instances, num_bodies), device=device)
        )

        # Get the property reference
        body_mass_ref = rigid_object_data.body_mass

        # Assign a different value to the internal data via property
        rigid_object_data.body_mass.fill_(2.0)

        # Check that the property returns the new value (reference behavior)
        assert torch.all(
            wp.to_torch(rigid_object_data.body_mass) == torch.ones((num_instances, num_bodies), device=device) * 2.0
        )

        # Assign a different value via reference
        body_mass_ref.fill_(3.0)

        # Check that the internal data has been updated
        assert torch.all(
            wp.to_torch(rigid_object_data.body_mass) == torch.ones((num_instances, num_bodies), device=device) * 3.0
        )

        # --- Test body_inertia ---
        # Check the type and shape
        assert rigid_object_data.body_inertia.shape == (num_instances, num_bodies)
        assert rigid_object_data.body_inertia.dtype == wp.mat33f

        # Mock data initializes body_inertia to zeros
        expected_inertia = torch.zeros((num_instances, num_bodies, 3, 3), device=device)
        assert torch.all(wp.to_torch(rigid_object_data.body_inertia) == expected_inertia)

        # Get the property reference
        body_inertia_ref = rigid_object_data.body_inertia

        # Assign a different value to the internal data via property
        rigid_object_data.body_inertia.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        expected_inertia_ones = torch.ones((num_instances, num_bodies, 3, 3), device=device)
        assert torch.all(wp.to_torch(rigid_object_data.body_inertia) == expected_inertia_ones)

        # Assign a different value via reference
        body_inertia_ref.fill_(2.0)

        # Check that the internal data has been updated
        expected_inertia_twos = torch.ones((num_instances, num_bodies, 3, 3), device=device) * 2.0
        assert torch.all(wp.to_torch(rigid_object_data.body_inertia) == expected_inertia_twos)


class TestBodyLinkPoseW:
    """Tests the body link pose property.

    This value is read directly from the simulation bindings.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is a reference to the internal data.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_body_link_pose_w(self, mock_newton_manager, num_instances: int, device: str):
        """Test that body_link_pose_w has correct type, shape, and reference behavior."""
        rigid_object_data, _ = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.body_link_pose_w.shape == (num_instances, 1)
        assert rigid_object_data.body_link_pose_w.dtype == wp.transformf

        # Mock data is initialized to zeros
        expected = torch.zeros((num_instances, 1, 7), device=device)
        assert torch.all(wp.to_torch(rigid_object_data.body_link_pose_w) == expected)

        # Get the property reference
        body_link_pose_ref = rigid_object_data.body_link_pose_w

        # Assign a different value via property
        rigid_object_data.body_link_pose_w.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        expected_ones = torch.ones((num_instances, 1, 7), device=device)
        assert torch.all(wp.to_torch(rigid_object_data.body_link_pose_w) == expected_ones)

        # Assign a different value via reference
        body_link_pose_ref.fill_(2.0)

        # Check that the internal data has been updated
        expected_twos = torch.ones((num_instances, 1, 7), device=device) * 2.0
        assert torch.all(wp.to_torch(rigid_object_data.body_link_pose_w) == expected_twos)


class TestBodyLinkVelW:
    """Tests the body link velocity property.

    This value is derived from body COM velocity. To ensure correctness,
    we compare against the reference implementation.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that the timestamp is updated correctly.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that body_link_vel_w is correctly computed from COM velocity."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.body_link_vel_w.shape == (num_instances, 1)
        assert rigid_object_data.body_link_vel_w.dtype == wp.spatial_vectorf

        # Mock data is initialized to zeros
        expected = torch.zeros((num_instances, 1, 6), device=device)
        assert torch.all(wp.to_torch(rigid_object_data.body_link_vel_w) == expected)

        for i in range(5):
            rigid_object_data._sim_timestamp = i + 1.0

            # Generate random COM velocity and body COM position
            com_vel = torch.rand((num_instances, 1, 6), device=device)
            body_com_pos = torch.rand((num_instances, 1, 3), device=device)

            # Generate random link poses with normalized quaternions
            link_pose = torch.zeros((num_instances, 1, 7), device=device)
            link_pose[..., :3] = torch.rand((num_instances, 1, 3), device=device)
            link_pose[..., 3:] = torch.randn((num_instances, 1, 4), device=device)
            link_pose[..., 3:] = torch.nn.functional.normalize(link_pose[..., 3:], p=2.0, dim=-1, eps=1e-12)

            mock_view.set_mock_data(
                link_transforms=wp.from_torch(link_pose, dtype=wp.transformf),
                link_velocities=wp.from_torch(com_vel, dtype=wp.spatial_vectorf),
                body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
            )

            # Compute expected link velocity using IsaacLab reference implementation
            # vel[:, :3] += cross(vel[:, 3:], quat_apply(quat, -body_com_pos))
            expected_vel = com_vel.clone()
            expected_vel[..., :3] += torch.linalg.cross(
                expected_vel[..., 3:],
                math_utils.quat_apply(link_pose[..., 3:], -body_com_pos),
                dim=-1,
            )

            # Compare the computed value
            assert torch.allclose(wp.to_torch(rigid_object_data.body_link_vel_w), expected_vel, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        rigid_object_data, mock_view = self._setup_method(1, device)

        # Check initial timestamp
        assert rigid_object_data._body_link_vel_w.timestamp == -1.0
        assert rigid_object_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(rigid_object_data.body_link_vel_w).clone()

        # Check that buffer timestamp matches sim timestamp
        assert rigid_object_data._body_link_vel_w.timestamp == rigid_object_data._sim_timestamp

        # Update mock data without changing sim timestamp
        mock_view.set_mock_data(
            link_velocities=wp.from_torch(torch.rand((1, 1, 6), device=device), dtype=wp.spatial_vectorf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(rigid_object_data.body_link_vel_w) == value)

        # Update sim timestamp
        rigid_object_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert rigid_object_data._body_link_vel_w.timestamp != rigid_object_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(rigid_object_data.body_link_vel_w) == value)


class TestBodyComPoseW:
    """Tests the body center of mass pose property.

    This value is derived from body link pose and body COM position.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that the timestamp is updated correctly.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that body_com_pose_w is correctly computed from link pose and COM position."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.body_com_pose_w.shape == (num_instances, 1)
        assert rigid_object_data.body_com_pose_w.dtype == wp.transformf

        # Mock data is initialized to zeros
        expected = torch.zeros((num_instances, 1, 7), device=device)
        assert torch.all(wp.to_torch(rigid_object_data.body_com_pose_w) == expected)

        for i in range(5):
            rigid_object_data._sim_timestamp = i + 1.0

            # Generate random link poses with normalized quaternions
            link_pose = torch.zeros((num_instances, 1, 7), device=device)
            link_pose[..., :3] = torch.rand((num_instances, 1, 3), device=device)
            link_pose[..., 3:] = torch.randn((num_instances, 1, 4), device=device)
            link_pose[..., 3:] = torch.nn.functional.normalize(link_pose[..., 3:], p=2.0, dim=-1, eps=1e-12)

            # Generate random body COM position in body frame
            body_com_pos = torch.rand((num_instances, 1, 3), device=device)

            mock_view.set_mock_data(
                link_transforms=wp.from_torch(link_pose, dtype=wp.transformf),
                body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
            )

            # Compute expected COM pose using IsaacLab reference implementation
            # combine_frame_transforms(link_pos, link_quat, com_pos_b, identity_quat)
            body_com_quat_b = torch.zeros((num_instances, 1, 4), device=device)
            body_com_quat_b[..., 3] = 1.0  # identity quaternion

            expected_pos, expected_quat = math_utils.combine_frame_transforms(
                link_pose[..., :3], link_pose[..., 3:], body_com_pos, body_com_quat_b
            )
            expected_pose = torch.cat([expected_pos, expected_quat], dim=-1)

            # Compare the computed value
            assert torch.allclose(wp.to_torch(rigid_object_data.body_com_pose_w), expected_pose, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        rigid_object_data, mock_view = self._setup_method(1, device)

        # Check initial timestamp
        assert rigid_object_data._body_com_pose_w.timestamp == -1.0
        assert rigid_object_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(rigid_object_data.body_com_pose_w).clone()

        # Check that buffer timestamp matches sim timestamp
        assert rigid_object_data._body_com_pose_w.timestamp == rigid_object_data._sim_timestamp

        # Update mock data without changing sim timestamp
        mock_view.set_mock_data(
            link_transforms=wp.from_torch(torch.rand((1, 1, 7), device=device), dtype=wp.transformf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(rigid_object_data.body_com_pose_w) == value)

        # Update sim timestamp
        rigid_object_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert rigid_object_data._body_com_pose_w.timestamp != rigid_object_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(rigid_object_data.body_com_pose_w) == value)


class TestBodyComVelW:
    """Tests the body center of mass velocity property.

    This value is read directly from the simulation bindings.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is a reference to the internal data.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_body_com_vel_w(self, mock_newton_manager, num_instances: int, device: str):
        """Test that body_com_vel_w has correct type, shape, and reference behavior."""
        rigid_object_data, _ = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.body_com_vel_w.shape == (num_instances, 1)
        assert rigid_object_data.body_com_vel_w.dtype == wp.spatial_vectorf

        # Mock data is initialized to zeros
        expected = torch.zeros((num_instances, 1, 6), device=device)
        assert torch.all(wp.to_torch(rigid_object_data.body_com_vel_w) == expected)

        # Get the property reference
        body_com_vel_ref = rigid_object_data.body_com_vel_w

        # Assign a different value via property
        rigid_object_data.body_com_vel_w.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        expected_ones = torch.ones((num_instances, 1, 6), device=device)
        assert torch.all(wp.to_torch(rigid_object_data.body_com_vel_w) == expected_ones)

        # Assign a different value via reference
        body_com_vel_ref.fill_(2.0)

        # Check that the internal data has been updated
        expected_twos = torch.ones((num_instances, 1, 6), device=device) * 2.0
        assert torch.all(wp.to_torch(rigid_object_data.body_com_vel_w) == expected_twos)


class TestBodyState:
    """Tests the body state properties.

    Test the body state properties are correctly updated from the pose and velocity properties.
    Tests the following properties:
    - body_state_w
    - body_link_state_w
    - body_com_state_w

    For each property, we run the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly assembled from pose and velocity.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_body_state_properties(self, mock_newton_manager, num_instances: int, device: str):
        """Test that all body state properties correctly combine pose and velocity."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Generate random mock data
        for i in range(5):
            rigid_object_data._sim_timestamp = i + 1.0

            # Generate random body link pose with normalized quaternions
            body_link_pose = torch.zeros((num_instances, 1, 7), device=device)
            body_link_pose[..., :3] = torch.rand((num_instances, 1, 3), device=device)
            body_link_pose[..., 3:] = torch.randn((num_instances, 1, 4), device=device)
            body_link_pose[..., 3:] = torch.nn.functional.normalize(body_link_pose[..., 3:], p=2.0, dim=-1, eps=1e-12)

            # Generate random COM velocities and COM position
            com_vel = torch.rand((num_instances, 1, 6), device=device)
            body_com_pos = torch.rand((num_instances, 1, 3), device=device)

            mock_view.set_mock_data(
                link_transforms=wp.from_torch(body_link_pose, dtype=wp.transformf),
                link_velocities=wp.from_torch(com_vel, dtype=wp.spatial_vectorf),
                body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
            )

            # --- Test body_state_w ---
            # Combines body_link_pose_w with body_com_vel_w
            body_state = wp.to_torch(rigid_object_data.body_state_w)
            expected_body_state = torch.cat([body_link_pose, com_vel], dim=-1)

            assert body_state.shape == (num_instances, 1, 13)
            assert torch.allclose(body_state, expected_body_state, atol=1e-6, rtol=1e-6)

            # --- Test body_link_state_w ---
            # Combines body_link_pose_w with body_link_vel_w
            body_link_state = wp.to_torch(rigid_object_data.body_link_state_w)

            # Compute expected body_link_vel from com_vel (same as TestBodyLinkVelW)
            body_link_vel = com_vel.clone()
            body_link_vel[..., :3] += torch.linalg.cross(
                body_link_vel[..., 3:],
                math_utils.quat_apply(body_link_pose[..., 3:], -body_com_pos),
                dim=-1,
            )
            expected_body_link_state = torch.cat([body_link_pose, body_link_vel], dim=-1)

            assert body_link_state.shape == (num_instances, 1, 13)
            assert torch.allclose(body_link_state, expected_body_link_state, atol=1e-6, rtol=1e-6)

            # --- Test body_com_state_w ---
            # Combines body_com_pose_w with body_com_vel_w
            body_com_state = wp.to_torch(rigid_object_data.body_com_state_w)

            # Compute expected body_com_pose from body_link_pose and body_com_pos (same as TestBodyComPoseW)
            body_com_quat_b = torch.zeros((num_instances, 1, 4), device=device)
            body_com_quat_b[..., 3] = 1.0
            body_com_pos_w, body_com_quat_w = math_utils.combine_frame_transforms(
                body_link_pose[..., :3], body_link_pose[..., 3:], body_com_pos, body_com_quat_b
            )
            expected_body_com_state = torch.cat([body_com_pos_w, body_com_quat_w, com_vel], dim=-1)

            assert body_com_state.shape == (num_instances, 1, 13)
            assert torch.allclose(body_com_state, expected_body_com_state, atol=1e-6, rtol=1e-6)


class TestBodyComAccW:
    """Tests the body center of mass acceleration property.

    This value is derived from velocity finite differencing: (current_vel - previous_vel) / dt

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that the timestamp is updated correctly.
    """

    def _setup_method(
        self, num_instances: int, device: str, initial_vel: torch.Tensor | None = None
    ) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)

        # Set initial velocities (these become _previous_body_com_vel)
        if initial_vel is not None:
            mock_view.set_mock_data(
                link_velocities=wp.from_torch(initial_vel, dtype=wp.spatial_vectorf),
            )
        else:
            mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that body_com_acc_w is correctly computed from velocity finite differencing."""
        # Initial velocity (becomes previous_velocity)
        previous_vel = torch.rand((num_instances, 1, 6), device=device)
        rigid_object_data, mock_view = self._setup_method(num_instances, device, previous_vel)

        # Check the type and shape
        assert rigid_object_data.body_com_acc_w.shape == (num_instances, 1)
        assert rigid_object_data.body_com_acc_w.dtype == wp.spatial_vectorf

        # dt is mocked as 0.01
        dt = 0.01

        for i in range(10):
            rigid_object_data._sim_timestamp = i + 1.0

            # Generate new random velocity
            current_vel = torch.rand((num_instances, 1, 6), device=device)
            mock_view.set_mock_data(
                link_velocities=wp.from_torch(current_vel, dtype=wp.spatial_vectorf),
            )

            # Compute expected acceleration: (current - previous) / dt
            expected_acc = (current_vel - previous_vel) / dt

            # Compare the computed value
            assert torch.allclose(wp.to_torch(rigid_object_data.body_com_acc_w), expected_acc, atol=1e-5, rtol=1e-5)
            # Update previous velocity
            previous_vel = current_vel.clone()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        initial_vel = torch.zeros((1, 1, 6), device=device)
        rigid_object_data, mock_view = self._setup_method(1, device, initial_vel)

        # Check initial timestamp
        assert rigid_object_data._body_com_acc_w.timestamp == -1.0
        assert rigid_object_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(rigid_object_data.body_com_acc_w).clone()

        # Check that buffer timestamp matches sim timestamp
        assert rigid_object_data._body_com_acc_w.timestamp == rigid_object_data._sim_timestamp

        # Update mock data without changing sim timestamp
        mock_view.set_mock_data(
            link_velocities=wp.from_torch(torch.rand((1, 1, 6), device=device), dtype=wp.spatial_vectorf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(rigid_object_data.body_com_acc_w) == value)

        # Update sim timestamp
        rigid_object_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert rigid_object_data._body_com_acc_w.timestamp != rigid_object_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(rigid_object_data.body_com_acc_w) == value)


class TestBodyComPoseB:
    """Tests the body center of mass pose in body frame property.

    This value is generated from COM position with identity quaternion.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value correctly combines position with identity quaternion.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_body_com_pose_b(self, mock_newton_manager, num_instances: int, device: str):
        """Test that body_com_pose_b correctly generates pose from position with identity quaternion."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.body_com_pose_b.shape == (num_instances, 1)
        assert rigid_object_data.body_com_pose_b.dtype == wp.transformf

        # Mock data is initialized to zeros for COM position
        # Expected pose: [0, 0, 0, 0, 0, 0, 1] (position zeros, identity quaternion)
        expected = torch.zeros((num_instances, 1, 7), device=device)
        expected[..., 6] = 1.0  # w component of identity quaternion
        assert torch.all(wp.to_torch(rigid_object_data.body_com_pose_b) == expected)

        # Update COM position and verify
        com_pos = torch.rand((num_instances, 1, 3), device=device)
        mock_view.set_mock_data(
            body_com_pos=wp.from_torch(com_pos, dtype=wp.vec3f),
        )

        # Get the pose
        pose = wp.to_torch(rigid_object_data.body_com_pose_b)

        # Expected: position from mock, identity quaternion
        expected_pose = torch.zeros((num_instances, 1, 7), device=device)
        expected_pose[..., :3] = com_pos
        expected_pose[..., 6] = 1.0  # w component

        assert torch.allclose(pose, expected_pose, atol=1e-6, rtol=1e-6)


##
# Test Cases -- Derived properties.
##


class TestProjectedGravityB:
    """Tests the projected gravity in body frame property.

    This value is derived by projecting the gravity vector onto the body frame.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that the timestamp is updated correctly.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that projected_gravity_b is correctly computed."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.projected_gravity_b.shape == (num_instances,)
        assert rigid_object_data.projected_gravity_b.dtype == wp.vec3f

        # Gravity direction (normalized)
        gravity_dir = torch.tensor([0.0, 0.0, -1.0], device=device)

        for i in range(10):
            rigid_object_data._sim_timestamp = i + 1.0
            # Generate random root pose with normalized quaternion
            root_pose = torch.zeros((num_instances, 7), device=device)
            root_pose[:, :3] = torch.rand((num_instances, 3), device=device)
            root_pose[:, 3:] = torch.randn((num_instances, 4), device=device)
            root_pose[:, 3:] = torch.nn.functional.normalize(root_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)

            mock_view.set_mock_data(
                root_transforms=wp.from_torch(root_pose, dtype=wp.transformf),
            )

            # Compute expected projected gravity: quat_apply(quat, gravity_dir)
            # This rotates gravity from world to body frame
            expected = math_utils.quat_apply_inverse(root_pose[:, 3:], gravity_dir.expand(num_instances, 3))

            # Compare the computed value
            assert torch.allclose(wp.to_torch(rigid_object_data.projected_gravity_b), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        rigid_object_data, mock_view = self._setup_method(1, device)

        # Check initial timestamp
        assert rigid_object_data._projected_gravity_b.timestamp == -1.0
        assert rigid_object_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(rigid_object_data.projected_gravity_b).clone()

        # Check that buffer timestamp matches sim timestamp
        assert rigid_object_data._projected_gravity_b.timestamp == rigid_object_data._sim_timestamp

        # Update mock data without changing sim timestamp
        new_pose = torch.zeros((1, 7), device=device)
        new_pose[:, 3:] = torch.randn((1, 4), device=device)
        new_pose[:, 3:] = torch.nn.functional.normalize(new_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(new_pose, dtype=wp.transformf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(rigid_object_data.projected_gravity_b) == value)

        # Update sim timestamp
        rigid_object_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert rigid_object_data._projected_gravity_b.timestamp != rigid_object_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(rigid_object_data.projected_gravity_b) == value)


class TestHeadingW:
    """Tests the heading in world frame property.

    This value is derived by computing the yaw angle from the forward direction.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that the timestamp is updated correctly.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that heading_w is correctly computed."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.heading_w.shape == (num_instances,)
        assert rigid_object_data.heading_w.dtype == wp.float32

        # Forward direction in body frame
        forward_vec_b = torch.tensor([1.0, 0.0, 0.0], device=device)

        for i in range(10):
            rigid_object_data._sim_timestamp = i + 1.0
            # Generate random root pose with normalized quaternion
            root_pose = torch.zeros((num_instances, 7), device=device)
            root_pose[:, :3] = torch.rand((num_instances, 3), device=device)
            root_pose[:, 3:] = torch.randn((num_instances, 4), device=device)
            root_pose[:, 3:] = torch.nn.functional.normalize(root_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)

            mock_view.set_mock_data(
                root_transforms=wp.from_torch(root_pose, dtype=wp.transformf),
            )
            # Compute expected heading: atan2(rotated_forward.y, rotated_forward.x)
            rotated_forward = math_utils.quat_apply(root_pose[:, 3:], forward_vec_b.expand(num_instances, 3))
            expected = torch.atan2(rotated_forward[:, 1], rotated_forward[:, 0])

            # Compare the computed value
            assert torch.allclose(wp.to_torch(rigid_object_data.heading_w), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        rigid_object_data, mock_view = self._setup_method(1, device)

        # Check initial timestamp
        assert rigid_object_data._heading_w.timestamp == -1.0
        assert rigid_object_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(rigid_object_data.heading_w).clone()

        # Check that buffer timestamp matches sim timestamp
        assert rigid_object_data._heading_w.timestamp == rigid_object_data._sim_timestamp

        # Update mock data without changing sim timestamp
        new_pose = torch.zeros((1, 7), device=device)
        new_pose[:, 3:] = torch.randn((1, 4), device=device)
        new_pose[:, 3:] = torch.nn.functional.normalize(new_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(new_pose, dtype=wp.transformf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(rigid_object_data.heading_w) == value)

        # Update sim timestamp
        rigid_object_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert rigid_object_data._heading_w.timestamp != rigid_object_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(rigid_object_data.heading_w) == value)


class TestRootLinkVelB:
    """Tests the root link velocity in body frame properties.

    Tests the following properties:
    - root_link_vel_b: velocity projected to body frame
    - root_link_lin_vel_b: linear velocity slice (first 3 components)
    - root_link_ang_vel_b: angular velocity slice (last 3 components)

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that lin/ang velocities are correct slices.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that root_link_vel_b and its slices are correctly computed."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Check types and shapes
        assert rigid_object_data.root_link_vel_b.shape == (num_instances,)
        assert rigid_object_data.root_link_vel_b.dtype == wp.spatial_vectorf

        assert rigid_object_data.root_link_lin_vel_b.shape == (num_instances,)
        assert rigid_object_data.root_link_lin_vel_b.dtype == wp.vec3f

        assert rigid_object_data.root_link_ang_vel_b.shape == (num_instances,)
        assert rigid_object_data.root_link_ang_vel_b.dtype == wp.vec3f

        for i in range(5):
            rigid_object_data._sim_timestamp = i + 1.0

            # Generate random root pose with normalized quaternion
            root_pose = torch.zeros((num_instances, 7), device=device)
            root_pose[:, :3] = torch.rand((num_instances, 3), device=device)
            root_pose[:, 3:] = torch.randn((num_instances, 4), device=device)
            root_pose[:, 3:] = torch.nn.functional.normalize(root_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)

            # Generate random COM velocity and body COM position
            com_vel = torch.rand((num_instances, 6), device=device)
            body_com_pos = torch.rand((num_instances, 1, 3), device=device)

            mock_view.set_mock_data(
                root_transforms=wp.from_torch(root_pose, dtype=wp.transformf),
                root_velocities=wp.from_torch(com_vel, dtype=wp.spatial_vectorf),
                body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
            )

            # Compute expected root_link_vel_w (same as TestRootLinkVelW)
            root_link_vel_w = com_vel.clone()
            root_link_vel_w[:, :3] += torch.linalg.cross(
                root_link_vel_w[:, 3:],
                math_utils.quat_apply(root_pose[:, 3:], -body_com_pos[:, 0]),
                dim=-1,
            )

            # Project to body frame using quat_rotate_inv
            # Linear velocity: quat_rotate_inv(quat, lin_vel)
            # Angular velocity: quat_rotate_inv(quat, ang_vel)
            lin_vel_b = math_utils.quat_apply_inverse(root_pose[:, 3:], root_link_vel_w[:, :3])
            ang_vel_b = math_utils.quat_apply_inverse(root_pose[:, 3:], root_link_vel_w[:, 3:])
            expected_vel_b = torch.cat([lin_vel_b, ang_vel_b], dim=-1)

            # Get computed values
            computed_vel_b = wp.to_torch(rigid_object_data.root_link_vel_b)
            computed_lin_vel_b = wp.to_torch(rigid_object_data.root_link_lin_vel_b)
            computed_ang_vel_b = wp.to_torch(rigid_object_data.root_link_ang_vel_b)

            # Compare full velocity
            assert torch.allclose(computed_vel_b, expected_vel_b, atol=1e-6, rtol=1e-6)

            # Check that lin/ang velocities are correct slices
            assert torch.allclose(computed_lin_vel_b, computed_vel_b[:, :3], atol=1e-6, rtol=1e-6)
            assert torch.allclose(computed_ang_vel_b, computed_vel_b[:, 3:], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        rigid_object_data, mock_view = self._setup_method(1, device)

        # Check initial timestamp
        assert rigid_object_data._root_link_vel_b.timestamp == -1.0
        assert rigid_object_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(rigid_object_data.root_link_vel_b).clone()

        # Check that buffer timestamp matches sim timestamp
        assert rigid_object_data._root_link_vel_b.timestamp == rigid_object_data._sim_timestamp

        # Update mock data without changing sim timestamp
        new_pose = torch.zeros((1, 7), device=device)
        new_pose[:, 3:] = torch.randn((1, 4), device=device)
        new_pose[:, 3:] = torch.nn.functional.normalize(new_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(new_pose, dtype=wp.transformf),
            root_velocities=wp.from_torch(torch.rand((1, 6), device=device), dtype=wp.spatial_vectorf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(rigid_object_data.root_link_vel_b) == value)

        # Update sim timestamp
        rigid_object_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert rigid_object_data._root_link_vel_b.timestamp != rigid_object_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(rigid_object_data.root_link_vel_b) == value)


class TestRootComVelB:
    """Tests the root center of mass velocity in body frame properties.

    Tests the following properties:
    - root_com_vel_b: COM velocity projected to body frame
    - root_com_lin_vel_b: linear velocity slice (first 3 components)
    - root_com_ang_vel_b: angular velocity slice (last 3 components)

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that lin/ang velocities are correct slices.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that root_com_vel_b and its slices are correctly computed."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Check types and shapes
        assert rigid_object_data.root_com_vel_b.shape == (num_instances,)
        assert rigid_object_data.root_com_vel_b.dtype == wp.spatial_vectorf

        assert rigid_object_data.root_com_lin_vel_b.shape == (num_instances,)
        assert rigid_object_data.root_com_lin_vel_b.dtype == wp.vec3f

        assert rigid_object_data.root_com_ang_vel_b.shape == (num_instances,)
        assert rigid_object_data.root_com_ang_vel_b.dtype == wp.vec3f

        for i in range(5):
            rigid_object_data._sim_timestamp = i + 1.0

            # Generate random root pose with normalized quaternion
            root_pose = torch.zeros((num_instances, 7), device=device)
            root_pose[:, :3] = torch.rand((num_instances, 3), device=device)
            root_pose[:, 3:] = torch.randn((num_instances, 4), device=device)
            root_pose[:, 3:] = torch.nn.functional.normalize(root_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)

            # Generate random COM velocity (this is root_com_vel_w from simulation)
            com_vel_w = torch.rand((num_instances, 6), device=device)

            mock_view.set_mock_data(
                root_transforms=wp.from_torch(root_pose, dtype=wp.transformf),
                root_velocities=wp.from_torch(com_vel_w, dtype=wp.spatial_vectorf),
            )

            # Project COM velocity to body frame using quat_rotate_inv (quat_conjugate + quat_apply)
            lin_vel_b = math_utils.quat_apply_inverse(root_pose[:, 3:], com_vel_w[:, :3])
            ang_vel_b = math_utils.quat_apply_inverse(root_pose[:, 3:], com_vel_w[:, 3:])
            expected_vel_b = torch.cat([lin_vel_b, ang_vel_b], dim=-1)

            # Get computed values
            computed_vel_b = wp.to_torch(rigid_object_data.root_com_vel_b)
            computed_lin_vel_b = wp.to_torch(rigid_object_data.root_com_lin_vel_b)
            computed_ang_vel_b = wp.to_torch(rigid_object_data.root_com_ang_vel_b)

            # Compare full velocity
            assert torch.allclose(computed_vel_b, expected_vel_b, atol=1e-6, rtol=1e-6)

            # Check that lin/ang velocities are correct slices
            assert torch.allclose(computed_lin_vel_b, computed_vel_b[:, :3], atol=1e-6, rtol=1e-6)
            assert torch.allclose(computed_ang_vel_b, computed_vel_b[:, 3:], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        rigid_object_data, mock_view = self._setup_method(1, device)

        # Check initial timestamp
        assert rigid_object_data._root_com_vel_b.timestamp == -1.0
        assert rigid_object_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(rigid_object_data.root_com_vel_b).clone()

        # Check that buffer timestamp matches sim timestamp
        assert rigid_object_data._root_com_vel_b.timestamp == rigid_object_data._sim_timestamp

        # Update mock data without changing sim timestamp
        new_pose = torch.zeros((1, 7), device=device)
        new_pose[:, 3:] = torch.randn((1, 4), device=device)
        new_pose[:, 3:] = torch.nn.functional.normalize(new_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(new_pose, dtype=wp.transformf),
            root_velocities=wp.from_torch(torch.rand((1, 6), device=device), dtype=wp.spatial_vectorf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(rigid_object_data.root_com_vel_b) == value)

        # Update sim timestamp
        rigid_object_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert rigid_object_data._root_com_vel_b.timestamp != rigid_object_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(rigid_object_data.root_com_vel_b) == value)


##
# Test Cases -- Sliced properties.
##


class TestRootSlicedProperties:
    """Tests the root sliced properties.

    These properties extract position, quaternion, linear velocity, or angular velocity
    from the full pose/velocity arrays.

    Tests the following properties:
    - root_link_pos_w, root_link_quat_w (from root_link_pose_w)
    - root_link_lin_vel_w, root_link_ang_vel_w (from root_link_vel_w)
    - root_com_pos_w, root_com_quat_w (from root_com_pose_w)
    - root_com_lin_vel_w, root_com_ang_vel_w (from root_com_vel_w)

    For each property, we only check that they are the correct slice of the parent property.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_root_sliced_properties(self, mock_newton_manager, num_instances: int, device: str):
        """Test that all root sliced properties are correct slices of their parent properties."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Set up random mock data to ensure non-trivial values
        rigid_object_data._sim_timestamp = 1.0

        root_pose = torch.zeros((num_instances, 7), device=device)
        root_pose[:, :3] = torch.rand((num_instances, 3), device=device)
        root_pose[:, 3:] = torch.randn((num_instances, 4), device=device)
        root_pose[:, 3:] = torch.nn.functional.normalize(root_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)

        com_vel = torch.rand((num_instances, 6), device=device)
        body_com_pos = torch.rand((num_instances, 1, 3), device=device)

        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_pose, dtype=wp.transformf),
            root_velocities=wp.from_torch(com_vel, dtype=wp.spatial_vectorf),
            body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
        )

        # --- Test root_link_pose_w slices ---
        root_link_pose = wp.to_torch(rigid_object_data.root_link_pose_w)
        root_link_pos = wp.to_torch(rigid_object_data.root_link_pos_w)
        root_link_quat = wp.to_torch(rigid_object_data.root_link_quat_w)

        assert root_link_pos.shape == (num_instances, 3)
        assert root_link_quat.shape == (num_instances, 4)
        assert torch.allclose(root_link_pos, root_link_pose[:, :3], atol=1e-6)
        assert torch.allclose(root_link_quat, root_link_pose[:, 3:], atol=1e-6)

        # --- Test root_link_vel_w slices ---
        root_link_vel = wp.to_torch(rigid_object_data.root_link_vel_w)
        root_link_lin_vel = wp.to_torch(rigid_object_data.root_link_lin_vel_w)
        root_link_ang_vel = wp.to_torch(rigid_object_data.root_link_ang_vel_w)

        assert root_link_lin_vel.shape == (num_instances, 3)
        assert root_link_ang_vel.shape == (num_instances, 3)
        assert torch.allclose(root_link_lin_vel, root_link_vel[:, :3], atol=1e-6)
        assert torch.allclose(root_link_ang_vel, root_link_vel[:, 3:], atol=1e-6)

        # --- Test root_com_pose_w slices ---
        root_com_pose = wp.to_torch(rigid_object_data.root_com_pose_w)
        root_com_pos = wp.to_torch(rigid_object_data.root_com_pos_w)
        root_com_quat = wp.to_torch(rigid_object_data.root_com_quat_w)

        assert root_com_pos.shape == (num_instances, 3)
        assert root_com_quat.shape == (num_instances, 4)
        assert torch.allclose(root_com_pos, root_com_pose[:, :3], atol=1e-6)
        assert torch.allclose(root_com_quat, root_com_pose[:, 3:], atol=1e-6)

        # --- Test root_com_vel_w slices ---
        root_com_vel = wp.to_torch(rigid_object_data.root_com_vel_w)
        root_com_lin_vel = wp.to_torch(rigid_object_data.root_com_lin_vel_w)
        root_com_ang_vel = wp.to_torch(rigid_object_data.root_com_ang_vel_w)

        assert root_com_lin_vel.shape == (num_instances, 3)
        assert root_com_ang_vel.shape == (num_instances, 3)
        assert torch.allclose(root_com_lin_vel, root_com_vel[:, :3], atol=1e-6)
        assert torch.allclose(root_com_ang_vel, root_com_vel[:, 3:], atol=1e-6)


class TestBodySlicedProperties:
    """Tests the body sliced properties.

    These properties extract position, quaternion, linear velocity, or angular velocity
    from the full pose/velocity arrays.

    Tests the following properties:
    - body_link_pos_w, body_link_quat_w (from body_link_pose_w)
    - body_link_lin_vel_w, body_link_ang_vel_w (from body_link_vel_w)
    - body_com_pos_w, body_com_quat_w (from body_com_pose_w)
    - body_com_lin_vel_w, body_com_ang_vel_w (from body_com_vel_w)

    For each property, we only check that they are the correct slice of the parent property.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_body_sliced_properties(self, mock_newton_manager, num_instances: int, device: str):
        """Test that all body sliced properties are correct slices of their parent properties."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Set up random mock data to ensure non-trivial values
        rigid_object_data._sim_timestamp = 1.0

        body_pose = torch.zeros((num_instances, 1, 7), device=device)
        body_pose[..., :3] = torch.rand((num_instances, 1, 3), device=device)
        body_pose[..., 3:] = torch.randn((num_instances, 1, 4), device=device)
        body_pose[..., 3:] = torch.nn.functional.normalize(body_pose[..., 3:], p=2.0, dim=-1, eps=1e-12)

        body_vel = torch.rand((num_instances, 1, 6), device=device)
        body_com_pos = torch.rand((num_instances, 1, 3), device=device)

        mock_view.set_mock_data(
            link_transforms=wp.from_torch(body_pose, dtype=wp.transformf),
            link_velocities=wp.from_torch(body_vel, dtype=wp.spatial_vectorf),
            body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
        )

        # --- Test body_link_pose_w slices ---
        body_link_pose = wp.to_torch(rigid_object_data.body_link_pose_w)
        body_link_pos = wp.to_torch(rigid_object_data.body_link_pos_w)
        body_link_quat = wp.to_torch(rigid_object_data.body_link_quat_w)

        assert body_link_pos.shape == (num_instances, 1, 3)
        assert body_link_quat.shape == (num_instances, 1, 4)
        assert torch.allclose(body_link_pos, body_link_pose[..., :3], atol=1e-6)
        assert torch.allclose(body_link_quat, body_link_pose[..., 3:], atol=1e-6)

        # --- Test body_link_vel_w slices ---
        body_link_vel = wp.to_torch(rigid_object_data.body_link_vel_w)
        body_link_lin_vel = wp.to_torch(rigid_object_data.body_link_lin_vel_w)
        body_link_ang_vel = wp.to_torch(rigid_object_data.body_link_ang_vel_w)

        assert body_link_lin_vel.shape == (num_instances, 1, 3)
        assert body_link_ang_vel.shape == (num_instances, 1, 3)
        assert torch.allclose(body_link_lin_vel, body_link_vel[..., :3], atol=1e-6)
        assert torch.allclose(body_link_ang_vel, body_link_vel[..., 3:], atol=1e-6)

        # --- Test body_com_pose_w slices ---
        body_com_pose = wp.to_torch(rigid_object_data.body_com_pose_w)
        body_com_pos_w = wp.to_torch(rigid_object_data.body_com_pos_w)
        body_com_quat_w = wp.to_torch(rigid_object_data.body_com_quat_w)

        assert body_com_pos_w.shape == (num_instances, 1, 3)
        assert body_com_quat_w.shape == (num_instances, 1, 4)
        assert torch.allclose(body_com_pos_w, body_com_pose[..., :3], atol=1e-6)
        assert torch.allclose(body_com_quat_w, body_com_pose[..., 3:], atol=1e-6)

        # --- Test body_com_vel_w slices ---
        body_com_vel = wp.to_torch(rigid_object_data.body_com_vel_w)
        body_com_lin_vel = wp.to_torch(rigid_object_data.body_com_lin_vel_w)
        body_com_ang_vel = wp.to_torch(rigid_object_data.body_com_ang_vel_w)

        assert body_com_lin_vel.shape == (num_instances, 1, 3)
        assert body_com_ang_vel.shape == (num_instances, 1, 3)
        assert torch.allclose(body_com_lin_vel, body_com_vel[..., :3], atol=1e-6)
        assert torch.allclose(body_com_ang_vel, body_com_vel[..., 3:], atol=1e-6)


class TestBodyComPosQuatB:
    """Tests the body center of mass position and quaternion in body frame properties.

    Tests the following properties:
    - body_com_pos_b: COM position in body frame (direct sim binding)
    - body_com_quat_b: COM orientation in body frame (derived from body_com_pose_b)

    Runs the following checks:
    - Checks that the returned values have the correct type and shape.
    - Checks that body_com_pos_b returns the simulation data.
    - Checks that body_com_quat_b is the quaternion slice of body_com_pose_b.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_body_com_pos_and_quat_b(self, mock_newton_manager, num_instances: int, device: str):
        """Test that body_com_pos_b and body_com_quat_b have correct types, shapes, and values."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # --- Test body_com_pos_b ---
        # Check the type and shape
        assert rigid_object_data.body_com_pos_b.shape == (num_instances, 1)
        assert rigid_object_data.body_com_pos_b.dtype == wp.vec3f

        # Mock data is initialized to zeros
        expected_pos = torch.zeros((num_instances, 1, 3), device=device)
        assert torch.all(wp.to_torch(rigid_object_data.body_com_pos_b) == expected_pos)

        # Update with random COM positions
        com_pos = torch.rand((num_instances, 1, 3), device=device)
        mock_view.set_mock_data(
            body_com_pos=wp.from_torch(com_pos, dtype=wp.vec3f),
        )

        # Check that the property returns the mock data
        assert torch.allclose(wp.to_torch(rigid_object_data.body_com_pos_b), com_pos, atol=1e-6)

        # Verify reference behavior
        body_com_pos_ref = rigid_object_data.body_com_pos_b
        rigid_object_data.body_com_pos_b.fill_(1.0)
        expected_ones = torch.ones((num_instances, 1, 3), device=device)
        assert torch.all(wp.to_torch(rigid_object_data.body_com_pos_b) == expected_ones)
        body_com_pos_ref.fill_(2.0)
        expected_twos = torch.ones((num_instances, 1, 3), device=device) * 2.0
        assert torch.all(wp.to_torch(rigid_object_data.body_com_pos_b) == expected_twos)

        # --- Test body_com_quat_b ---
        # Check the type and shape
        assert rigid_object_data.body_com_quat_b.shape == (num_instances, 1)
        assert rigid_object_data.body_com_quat_b.dtype == wp.quatf

        # body_com_quat_b is derived from body_com_pose_b which uses identity quaternion
        # body_com_pose_b = [body_com_pos_b, identity_quat]
        # So body_com_quat_b should be identity quaternion (0, 0, 0, 1)
        body_com_quat = wp.to_torch(rigid_object_data.body_com_quat_b)
        expected_quat = torch.zeros((num_instances, 1, 4), device=device)
        expected_quat[..., 3] = 1.0  # w component of identity quaternion

        assert torch.allclose(body_com_quat, expected_quat, atol=1e-6)


##
# Test Cases -- Backward compatibility.
##


# TODO: Remove this test case in the future.
class TestDefaultRootState:
    """Tests the deprecated default_root_state property.

    This property combines default_root_pose and default_root_vel into a vec13f state.
    It is deprecated in favor of using default_root_pose and default_root_vel directly.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that it correctly combines default_root_pose and default_root_vel.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_default_root_state(self, mock_newton_manager, num_instances: int, device: str):
        """Test that default_root_state correctly combines pose and velocity."""
        rigid_object_data, _ = self._setup_method(num_instances, device)

        # Check the type and shape
        assert rigid_object_data.default_root_state.shape == (num_instances,)

        # Get the combined state
        default_state = wp.to_torch(rigid_object_data.default_root_state)
        assert default_state.shape == (num_instances, 13)

        # Get the individual components
        default_pose = wp.to_torch(rigid_object_data.default_root_pose)
        default_vel = wp.to_torch(rigid_object_data.default_root_vel)

        # Verify the state is the concatenation of pose and velocity
        expected_state = torch.cat([default_pose, default_vel], dim=-1)
        assert torch.allclose(default_state, expected_state, atol=1e-6)

        # Modify default_root_pose and default_root_vel and verify the state updates
        new_pose = torch.zeros((num_instances, 7), device=device)
        new_pose[:, :3] = torch.rand((num_instances, 3), device=device)
        new_pose[:, 3:] = torch.randn((num_instances, 4), device=device)
        new_pose[:, 3:] = torch.nn.functional.normalize(new_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)

        new_vel = torch.rand((num_instances, 6), device=device)

        # Set the new values
        rigid_object_data.default_root_pose.assign(wp.from_torch(new_pose, dtype=wp.transformf))
        rigid_object_data.default_root_vel.assign(wp.from_torch(new_vel, dtype=wp.spatial_vectorf))

        # Verify the state reflects the new values
        updated_state = wp.to_torch(rigid_object_data.default_root_state)
        expected_updated_state = torch.cat([new_pose, new_vel], dim=-1)
        assert torch.allclose(updated_state, expected_updated_state, atol=1e-6)


# TODO: Remove this test case in the future.
class TestDeprecatedRootProperties:
    """Tests the deprecated root pose/velocity properties.

    These are backward compatibility aliases that just return the corresponding new property.

    Tests the following deprecated -> new property mappings:
    - root_pose_w -> root_link_pose_w
    - root_pos_w -> root_link_pos_w
    - root_quat_w -> root_link_quat_w
    - root_vel_w -> root_com_vel_w
    - root_lin_vel_w -> root_com_lin_vel_w
    - root_ang_vel_w -> root_com_ang_vel_w
    - root_lin_vel_b -> root_com_lin_vel_b
    - root_ang_vel_b -> root_com_ang_vel_b
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_deprecated_root_properties(self, mock_newton_manager, num_instances: int, device: str):
        """Test that all deprecated root properties match their replacements."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Set up random mock data to ensure non-trivial values
        rigid_object_data._sim_timestamp = 1.0

        root_pose = torch.zeros((num_instances, 7), device=device)
        root_pose[:, :3] = torch.rand((num_instances, 3), device=device)
        root_pose[:, 3:] = torch.randn((num_instances, 4), device=device)
        root_pose[:, 3:] = torch.nn.functional.normalize(root_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)

        com_vel = torch.rand((num_instances, 6), device=device)
        body_com_pos = torch.rand((num_instances, 1, 3), device=device)

        mock_view.set_mock_data(
            root_transforms=wp.from_torch(root_pose, dtype=wp.transformf),
            root_velocities=wp.from_torch(com_vel, dtype=wp.spatial_vectorf),
            body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
        )

        # --- Test root_pose_w -> root_link_pose_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.root_pose_w),
            wp.to_torch(rigid_object_data.root_link_pose_w),
            atol=1e-6,
        )

        # --- Test root_pos_w -> root_link_pos_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.root_pos_w),
            wp.to_torch(rigid_object_data.root_link_pos_w),
            atol=1e-6,
        )

        # --- Test root_quat_w -> root_link_quat_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.root_quat_w),
            wp.to_torch(rigid_object_data.root_link_quat_w),
            atol=1e-6,
        )

        # --- Test root_vel_w -> root_com_vel_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.root_vel_w),
            wp.to_torch(rigid_object_data.root_com_vel_w),
            atol=1e-6,
        )

        # --- Test root_lin_vel_w -> root_com_lin_vel_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.root_lin_vel_w),
            wp.to_torch(rigid_object_data.root_com_lin_vel_w),
            atol=1e-6,
        )

        # --- Test root_ang_vel_w -> root_com_ang_vel_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.root_ang_vel_w),
            wp.to_torch(rigid_object_data.root_com_ang_vel_w),
            atol=1e-6,
        )

        # --- Test root_lin_vel_b -> root_com_lin_vel_b ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.root_lin_vel_b),
            wp.to_torch(rigid_object_data.root_com_lin_vel_b),
            atol=1e-6,
        )

        # --- Test root_ang_vel_b -> root_com_ang_vel_b ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.root_ang_vel_b),
            wp.to_torch(rigid_object_data.root_com_ang_vel_b),
            atol=1e-6,
        )


class TestDeprecatedBodyProperties:
    """Tests the deprecated body pose/velocity/acceleration properties.

    These are backward compatibility aliases that just return the corresponding new property.

    Tests the following deprecated -> new property mappings:
    - body_pose_w -> body_link_pose_w
    - body_pos_w -> body_link_pos_w
    - body_quat_w -> body_link_quat_w
    - body_vel_w -> body_com_vel_w
    - body_lin_vel_w -> body_com_lin_vel_w
    - body_ang_vel_w -> body_com_ang_vel_w
    - body_acc_w -> body_com_acc_w
    - body_lin_acc_w -> body_com_lin_acc_w
    - body_ang_acc_w -> body_com_ang_acc_w
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_deprecated_body_properties(self, mock_newton_manager, num_instances: int, device: str):
        """Test that all deprecated body properties match their replacements."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Set up random mock data to ensure non-trivial values
        rigid_object_data._sim_timestamp = 1.0

        body_pose = torch.zeros((num_instances, 1, 7), device=device)
        body_pose[..., :3] = torch.rand((num_instances, 1, 3), device=device)
        body_pose[..., 3:] = torch.randn((num_instances, 1, 4), device=device)
        body_pose[..., 3:] = torch.nn.functional.normalize(body_pose[..., 3:], p=2.0, dim=-1, eps=1e-12)

        body_vel = torch.rand((num_instances, 1, 6), device=device)
        body_com_pos = torch.rand((num_instances, 1, 3), device=device)

        mock_view.set_mock_data(
            link_transforms=wp.from_torch(body_pose, dtype=wp.transformf),
            link_velocities=wp.from_torch(body_vel, dtype=wp.spatial_vectorf),
            body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
        )

        # --- Test body_pose_w -> body_link_pose_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.body_pose_w),
            wp.to_torch(rigid_object_data.body_link_pose_w),
            atol=1e-6,
        )

        # --- Test body_pos_w -> body_link_pos_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.body_pos_w),
            wp.to_torch(rigid_object_data.body_link_pos_w),
            atol=1e-6,
        )

        # --- Test body_quat_w -> body_link_quat_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.body_quat_w),
            wp.to_torch(rigid_object_data.body_link_quat_w),
            atol=1e-6,
        )

        # --- Test body_vel_w -> body_com_vel_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.body_vel_w),
            wp.to_torch(rigid_object_data.body_com_vel_w),
            atol=1e-6,
        )

        # --- Test body_lin_vel_w -> body_com_lin_vel_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.body_lin_vel_w),
            wp.to_torch(rigid_object_data.body_com_lin_vel_w),
            atol=1e-6,
        )

        # --- Test body_ang_vel_w -> body_com_ang_vel_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.body_ang_vel_w),
            wp.to_torch(rigid_object_data.body_com_ang_vel_w),
            atol=1e-6,
        )

        # --- Test body_acc_w -> body_com_acc_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.body_acc_w),
            wp.to_torch(rigid_object_data.body_com_acc_w),
            atol=1e-6,
        )

        # --- Test body_lin_acc_w -> body_com_lin_acc_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.body_lin_acc_w),
            wp.to_torch(rigid_object_data.body_com_lin_acc_w),
            atol=1e-6,
        )

        # --- Test body_ang_acc_w -> body_com_ang_acc_w ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.body_ang_acc_w),
            wp.to_torch(rigid_object_data.body_com_ang_acc_w),
            atol=1e-6,
        )


class TestDeprecatedComProperties:
    """Tests the deprecated COM pose properties.

    Tests the following deprecated -> new property mappings:
    - com_pos_b -> body_com_pos_b
    - com_quat_b -> body_com_quat_b
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[RigidObjectData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        rigid_object_data = RigidObjectData(
            mock_view,
            device,
        )
        return rigid_object_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_deprecated_com_properties(self, mock_newton_manager, num_instances: int, device: str):
        """Test that deprecated COM properties match their replacements."""
        rigid_object_data, mock_view = self._setup_method(num_instances, device)

        # Set up random mock data
        com_pos = torch.rand((num_instances, 1, 3), device=device)
        mock_view.set_mock_data(
            body_com_pos=wp.from_torch(com_pos, dtype=wp.vec3f),
        )

        # --- Test com_pos_b -> body_com_pos_b ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.com_pos_b),
            wp.to_torch(rigid_object_data.body_com_pos_b),
            atol=1e-6,
        )

        # --- Test com_quat_b -> body_com_quat_b ---
        assert torch.allclose(
            wp.to_torch(rigid_object_data.com_quat_b),
            wp.to_torch(rigid_object_data.body_com_quat_b),
            atol=1e-6,
        )


##
# Main
##

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
