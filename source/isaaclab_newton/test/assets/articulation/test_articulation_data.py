# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ArticulationData class comparing Newton implementation against PhysX reference."""

from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

import pytest
import warp as wp

# Import mock classes from common test utilities
from common.mock_newton import MockNewtonArticulationView, MockNewtonModel
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData

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

    # Patch where NewtonManager is used (in the articulation_data module)
    with patch("isaaclab_newton.assets.articulation.articulation_data.NewtonManager") as MockManager:
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
    - default_joint_pos
    - default_joint_vel

    Runs the following checks:
    - Checks that by default, the properties are all zero.
    - Checks that the properties are settable.
    - Checks that once the articulation data is primed, the properties cannot be changed.
    """

    def _setup_method(self, num_instances: int, num_dofs: int, device: str) -> ArticulationData:
        mock_view = MockNewtonArticulationView(num_instances, 1, num_dofs, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )

        return articulation_data

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_zero_instantiated(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test zero instantiated articulation data."""
        # Setup the articulation data
        articulation_data = self._setup_method(num_instances, num_dofs, device)
        # Check the types are correct
        assert articulation_data.default_root_pose.dtype is wp.transformf
        assert articulation_data.default_root_vel.dtype is wp.spatial_vectorf
        assert articulation_data.default_joint_pos.dtype is wp.float32
        assert articulation_data.default_joint_vel.dtype is wp.float32
        # Check the shapes are correct
        assert articulation_data.default_root_pose.shape == (num_instances,)
        assert articulation_data.default_root_vel.shape == (num_instances,)
        assert articulation_data.default_joint_pos.shape == (num_instances, num_dofs)
        assert articulation_data.default_joint_vel.shape == (num_instances, num_dofs)
        # Check the values are zero
        assert torch.all(
            wp.to_torch(articulation_data.default_root_pose) == torch.zeros(num_instances, 7, device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.default_root_vel) == torch.zeros(num_instances, 6, device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.default_joint_pos) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.default_joint_vel) == torch.zeros((num_instances, num_dofs), device=device)
        )

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_settable(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that the articulation data is settable."""
        # Setup the articulation data
        articulation_data = self._setup_method(num_instances, num_dofs, device)
        # Set the default values
        articulation_data.default_root_pose = wp.ones(num_instances, dtype=wp.transformf, device=device)
        articulation_data.default_root_vel = wp.ones(num_instances, dtype=wp.spatial_vectorf, device=device)
        articulation_data.default_joint_pos = wp.ones((num_instances, num_dofs), dtype=wp.float32, device=device)
        articulation_data.default_joint_vel = wp.ones((num_instances, num_dofs), dtype=wp.float32, device=device)
        # Check the types are correct
        assert articulation_data.default_root_pose.dtype is wp.transformf
        assert articulation_data.default_root_vel.dtype is wp.spatial_vectorf
        assert articulation_data.default_joint_pos.dtype is wp.float32
        assert articulation_data.default_joint_vel.dtype is wp.float32
        # Check the shapes are correct
        assert articulation_data.default_root_pose.shape == (num_instances,)
        assert articulation_data.default_root_vel.shape == (num_instances,)
        assert articulation_data.default_joint_pos.shape == (num_instances, num_dofs)
        assert articulation_data.default_joint_vel.shape == (num_instances, num_dofs)
        # Check the values are set
        assert torch.all(
            wp.to_torch(articulation_data.default_root_pose) == torch.ones(num_instances, 7, device=device)
        )
        assert torch.all(wp.to_torch(articulation_data.default_root_vel) == torch.ones(num_instances, 6, device=device))
        assert torch.all(
            wp.to_torch(articulation_data.default_joint_pos) == torch.ones((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.default_joint_vel) == torch.ones((num_instances, num_dofs), device=device)
        )
        # Prime the articulation data
        articulation_data.is_primed = True
        # Check that the values cannot be changed
        with pytest.raises(RuntimeError):
            articulation_data.default_root_pose = wp.zeros(num_instances, dtype=wp.transformf, device=device)
        with pytest.raises(RuntimeError):
            articulation_data.default_root_vel = wp.zeros(num_instances, dtype=wp.spatial_vectorf, device=device)
        with pytest.raises(RuntimeError):
            articulation_data.default_joint_pos = wp.zeros((num_instances, num_dofs), dtype=wp.float32, device=device)
        with pytest.raises(RuntimeError):
            articulation_data.default_joint_vel = wp.zeros((num_instances, num_dofs), dtype=wp.float32, device=device)


##
# Test Cases -- Joint Commands (Set into the simulation).
##


class TestJointCommandsSetIntoSimulation:
    """Tests the following properties:
    - joint_pos_target
    - joint_vel_target
    - joint_effort_target

    Runs the following checks:
    - Checks that their types and shapes are correct.
    - Checks that the returned values are pointers to the internal data.
    """

    def _setup_method(self, num_instances: int, num_dofs: int, device: str) -> ArticulationData:
        mock_view = MockNewtonArticulationView(num_instances, 1, num_dofs, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )

        return articulation_data

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_initialized_to_zero(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that the joint commands are initialized to zero."""
        # Setup the articulation data
        articulation_data = self._setup_method(num_instances, num_dofs, device)
        # Check the types is correct
        assert articulation_data.joint_pos_target.dtype is wp.float32
        assert articulation_data.joint_vel_target.dtype is wp.float32
        assert articulation_data.joint_effort.dtype is wp.float32
        # Check the shape is correct
        assert articulation_data.joint_pos_target.shape == (num_instances, num_dofs)
        assert articulation_data.joint_vel_target.shape == (num_instances, num_dofs)
        assert articulation_data.joint_effort.shape == (num_instances, num_dofs)
        # Check the values are zero
        assert torch.all(
            wp.to_torch(articulation_data.joint_pos_target) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_vel_target) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_effort) == torch.zeros((num_instances, num_dofs), device=device)
        )

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_returns_reference(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that the joint commands return a reference to the internal data."""
        # Setup the articulation data
        articulation_data = self._setup_method(num_instances, num_dofs, device)
        # Get the pointers
        joint_pos_target = articulation_data.joint_pos_target
        joint_vel_target = articulation_data.joint_vel_target
        joint_effort = articulation_data.joint_effort
        # Check that they are zeros
        assert torch.all(wp.to_torch(joint_pos_target) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_vel_target) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_effort) == torch.zeros((num_instances, num_dofs), device=device))
        # Assign a different value to the internal data
        articulation_data.joint_pos_target.fill_(1.0)
        articulation_data.joint_vel_target.fill_(1.0)
        articulation_data.joint_effort.fill_(1.0)
        # Check that the joint commands return the new value
        assert torch.all(wp.to_torch(joint_pos_target) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_vel_target) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_effort) == torch.ones((num_instances, num_dofs), device=device))
        # Assign a different value to the pointers
        joint_pos_target.fill_(2.0)
        joint_vel_target.fill_(2.0)
        joint_effort.fill_(2.0)
        # Check that the internal data has been updated
        assert torch.all(
            wp.to_torch(articulation_data.joint_pos_target)
            == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_vel_target)
            == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_effort) == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )


##
# Test Cases -- Joint Commands (Explicit actuators).
##


class TestJointCommandsExplicitActuators:
    """Tests the following properties:
    - computed_effort
    - applied_effort
    - actuator_stiffness
    - actuator_damping
    - actuator_position_target
    - actuator_velocity_target
    - actuator_effort_target

    Runs the following checks:
    - Checks that their types and shapes are correct.
    - Checks that the returned values are pointers to the internal data.
    """

    def _setup_method(self, num_instances: int, num_dofs: int, device: str) -> ArticulationData:
        mock_view = MockNewtonArticulationView(num_instances, 1, num_dofs, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )

        return articulation_data

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_initialized_to_zero(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that the explicit actuator properties are initialized to zero."""
        # Setup the articulation data
        articulation_data = self._setup_method(num_instances, num_dofs, device)
        # Check the types are correct
        assert articulation_data.computed_effort.dtype is wp.float32
        assert articulation_data.applied_effort.dtype is wp.float32
        assert articulation_data.actuator_stiffness.dtype is wp.float32
        assert articulation_data.actuator_damping.dtype is wp.float32
        assert articulation_data.actuator_position_target.dtype is wp.float32
        assert articulation_data.actuator_velocity_target.dtype is wp.float32
        assert articulation_data.actuator_effort_target.dtype is wp.float32
        # Check the shapes are correct
        assert articulation_data.computed_effort.shape == (num_instances, num_dofs)
        assert articulation_data.applied_effort.shape == (num_instances, num_dofs)
        assert articulation_data.actuator_stiffness.shape == (num_instances, num_dofs)
        assert articulation_data.actuator_damping.shape == (num_instances, num_dofs)
        assert articulation_data.actuator_position_target.shape == (num_instances, num_dofs)
        assert articulation_data.actuator_velocity_target.shape == (num_instances, num_dofs)
        assert articulation_data.actuator_effort_target.shape == (num_instances, num_dofs)
        # Check the values are zero
        assert torch.all(
            wp.to_torch(articulation_data.computed_effort) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.applied_effort) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.actuator_stiffness) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.actuator_damping) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.actuator_position_target)
            == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.actuator_velocity_target)
            == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.actuator_effort_target)
            == torch.zeros((num_instances, num_dofs), device=device)
        )

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_returns_reference(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that the explicit actuator properties return a reference to the internal data."""
        # Setup the articulation data
        articulation_data = self._setup_method(num_instances, num_dofs, device)
        # Get the pointers
        computed_effort = articulation_data.computed_effort
        applied_effort = articulation_data.applied_effort
        actuator_stiffness = articulation_data.actuator_stiffness
        actuator_damping = articulation_data.actuator_damping
        actuator_position_target = articulation_data.actuator_position_target
        actuator_velocity_target = articulation_data.actuator_velocity_target
        actuator_effort_target = articulation_data.actuator_effort_target
        # Check that they are zeros
        assert torch.all(wp.to_torch(computed_effort) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(applied_effort) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(actuator_stiffness) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(actuator_damping) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(actuator_position_target) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(actuator_velocity_target) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(actuator_effort_target) == torch.zeros((num_instances, num_dofs), device=device))
        # Assign a different value to the internal data
        articulation_data.computed_effort.fill_(1.0)
        articulation_data.applied_effort.fill_(1.0)
        articulation_data.actuator_stiffness.fill_(1.0)
        articulation_data.actuator_damping.fill_(1.0)
        articulation_data.actuator_position_target.fill_(1.0)
        articulation_data.actuator_velocity_target.fill_(1.0)
        articulation_data.actuator_effort_target.fill_(1.0)
        # Check that the properties return the new value
        assert torch.all(wp.to_torch(computed_effort) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(applied_effort) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(actuator_stiffness) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(actuator_damping) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(actuator_position_target) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(actuator_velocity_target) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(actuator_effort_target) == torch.ones((num_instances, num_dofs), device=device))
        # Assign a different value to the pointers
        computed_effort.fill_(2.0)
        applied_effort.fill_(2.0)
        actuator_stiffness.fill_(2.0)
        actuator_damping.fill_(2.0)
        actuator_position_target.fill_(2.0)
        actuator_velocity_target.fill_(2.0)
        actuator_effort_target.fill_(2.0)
        # Check that the internal data has been updated
        assert torch.all(
            wp.to_torch(articulation_data.computed_effort) == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.applied_effort) == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.actuator_stiffness)
            == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.actuator_damping)
            == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.actuator_position_target)
            == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.actuator_velocity_target)
            == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.actuator_effort_target)
            == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )


##
# Test Cases -- Joint Properties (Set into Simulation).
##


class TestJointPropertiesSetIntoSimulation:
    """Tests the following properties:
    - joint_stiffness
    - joint_damping
    - joint_armature
    - joint_friction_coeff
    - joint_pos_limits_lower
    - joint_pos_limits_upper
    - joint_pos_limits (read-only, computed from lower and upper)
    - joint_vel_limits
    - joint_effort_limits

    Runs the following checks:
    - Checks that their types and shapes are correct.
    - Checks that the returned values are pointers to the internal data.

    .. note:: joint_pos_limits is read-only and does not change the joint position limits.
    """

    def _setup_method(
        self, num_instances: int, num_dofs: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, num_dofs, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )

        # return the mock view, so that it doesn't get garbage collected
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_initialized_to_zero(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that the joint properties are initialized to zero (or ones for limits)."""
        # Setup the articulation data
        articulation_data, _ = self._setup_method(num_instances, num_dofs, device)

        # Check the types are correct
        assert articulation_data.joint_stiffness.dtype is wp.float32
        assert articulation_data.joint_damping.dtype is wp.float32
        assert articulation_data.joint_armature.dtype is wp.float32
        assert articulation_data.joint_friction_coeff.dtype is wp.float32
        assert articulation_data.joint_pos_limits_lower.dtype is wp.float32
        assert articulation_data.joint_pos_limits_upper.dtype is wp.float32
        assert articulation_data.joint_pos_limits.dtype is wp.vec2f
        assert articulation_data.joint_vel_limits.dtype is wp.float32
        assert articulation_data.joint_effort_limits.dtype is wp.float32

        # Check the shapes are correct
        assert articulation_data.joint_stiffness.shape == (num_instances, num_dofs)
        assert articulation_data.joint_damping.shape == (num_instances, num_dofs)
        assert articulation_data.joint_armature.shape == (num_instances, num_dofs)
        assert articulation_data.joint_friction_coeff.shape == (num_instances, num_dofs)
        assert articulation_data.joint_pos_limits_lower.shape == (num_instances, num_dofs)
        assert articulation_data.joint_pos_limits_upper.shape == (num_instances, num_dofs)
        assert articulation_data.joint_pos_limits.shape == (num_instances, num_dofs)
        assert articulation_data.joint_vel_limits.shape == (num_instances, num_dofs)
        assert articulation_data.joint_effort_limits.shape == (num_instances, num_dofs)

        # Check the values are zero
        assert torch.all(
            wp.to_torch(articulation_data.joint_stiffness) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_damping) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_armature) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_friction_coeff) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_pos_limits_lower)
            == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_pos_limits_upper)
            == torch.zeros((num_instances, num_dofs), device=device)
        )
        # joint_pos_limits should be (0, 0) for each joint since both lower and upper are 0
        joint_pos_limits = wp.to_torch(articulation_data.joint_pos_limits)
        assert torch.all(joint_pos_limits == torch.zeros((num_instances, num_dofs, 2), device=device))
        # vel_limits and effort_limits are initialized to zeros in the mock
        assert torch.all(
            wp.to_torch(articulation_data.joint_vel_limits) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_effort_limits) == torch.zeros((num_instances, num_dofs), device=device)
        )

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_returns_reference(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that the joint properties return a reference to the internal data.

        Note: joint_pos_limits is read-only and always returns a new computed array.
        """
        # Setup the articulation data
        articulation_data, _ = self._setup_method(num_instances, num_dofs, device)

        # Get the pointers
        joint_stiffness = articulation_data.joint_stiffness
        joint_damping = articulation_data.joint_damping
        joint_armature = articulation_data.joint_armature
        joint_friction_coeff = articulation_data.joint_friction_coeff
        joint_pos_limits_lower = articulation_data.joint_pos_limits_lower
        joint_pos_limits_upper = articulation_data.joint_pos_limits_upper
        joint_vel_limits = articulation_data.joint_vel_limits
        joint_effort_limits = articulation_data.joint_effort_limits

        # Check that they have initial values (zeros or ones based on mock)
        assert torch.all(wp.to_torch(joint_stiffness) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_damping) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_armature) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_friction_coeff) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_pos_limits_lower) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_pos_limits_upper) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_vel_limits) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_effort_limits) == torch.zeros((num_instances, num_dofs), device=device))

        # Assign a different value to the internal data
        articulation_data.joint_stiffness.fill_(1.0)
        articulation_data.joint_damping.fill_(1.0)
        articulation_data.joint_armature.fill_(1.0)
        articulation_data.joint_friction_coeff.fill_(1.0)
        articulation_data.joint_pos_limits_lower.fill_(-1.0)
        articulation_data.joint_pos_limits_upper.fill_(1.0)
        articulation_data.joint_vel_limits.fill_(2.0)
        articulation_data.joint_effort_limits.fill_(2.0)

        # Check that the properties return the new value (reference behavior)
        assert torch.all(wp.to_torch(joint_stiffness) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_damping) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_armature) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_friction_coeff) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(
            wp.to_torch(joint_pos_limits_lower) == torch.ones((num_instances, num_dofs), device=device) * -1.0
        )
        assert torch.all(wp.to_torch(joint_pos_limits_upper) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(joint_vel_limits) == torch.ones((num_instances, num_dofs), device=device) * 2.0)
        assert torch.all(wp.to_torch(joint_effort_limits) == torch.ones((num_instances, num_dofs), device=device) * 2.0)

        # Check that joint_pos_limits is computed correctly from lower and upper
        joint_pos_limits = wp.to_torch(articulation_data.joint_pos_limits)
        expected_limits = torch.stack(
            [
                torch.ones((num_instances, num_dofs), device=device) * -1.0,
                torch.ones((num_instances, num_dofs), device=device),
            ],
            dim=-1,
        )
        assert torch.all(joint_pos_limits == expected_limits)

        # Assign a different value to the pointers
        joint_stiffness.fill_(3.0)
        joint_damping.fill_(3.0)
        joint_armature.fill_(3.0)
        joint_friction_coeff.fill_(3.0)
        joint_pos_limits_lower.fill_(-2.0)
        joint_pos_limits_upper.fill_(2.0)
        joint_vel_limits.fill_(4.0)
        joint_effort_limits.fill_(4.0)

        # Check that the internal data has been updated
        assert torch.all(
            wp.to_torch(articulation_data.joint_stiffness) == torch.ones((num_instances, num_dofs), device=device) * 3.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_damping) == torch.ones((num_instances, num_dofs), device=device) * 3.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_armature) == torch.ones((num_instances, num_dofs), device=device) * 3.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_friction_coeff)
            == torch.ones((num_instances, num_dofs), device=device) * 3.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_pos_limits_lower)
            == torch.ones((num_instances, num_dofs), device=device) * -2.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_pos_limits_upper)
            == torch.ones((num_instances, num_dofs), device=device) * 2.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_vel_limits)
            == torch.ones((num_instances, num_dofs), device=device) * 4.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_effort_limits)
            == torch.ones((num_instances, num_dofs), device=device) * 4.0
        )

        # Verify joint_pos_limits reflects the updated lower and upper values
        joint_pos_limits_updated = wp.to_torch(articulation_data.joint_pos_limits)
        expected_limits_updated = torch.stack(
            [
                torch.ones((num_instances, num_dofs), device=device) * -2.0,
                torch.ones((num_instances, num_dofs), device=device) * 2.0,
            ],
            dim=-1,
        )
        assert torch.all(joint_pos_limits_updated == expected_limits_updated)

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_joint_pos_limits_is_read_only(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that joint_pos_limits returns a new array each time (not a reference).

        Unlike other joint properties, joint_pos_limits is computed on-the-fly from
        joint_pos_limits_lower and joint_pos_limits_upper. Modifying the returned array
        should not affect the underlying data.
        """
        # Setup the articulation data
        articulation_data, _ = self._setup_method(num_instances, num_dofs, device)

        # Get joint_pos_limits twice
        limits1 = articulation_data.joint_pos_limits
        limits2 = articulation_data.joint_pos_limits

        # They should be separate arrays (not the same reference)
        # Modifying one should not affect the other
        limits1.fill_(2.0)

        # limits2 should be changed to 2.0
        assert torch.all(wp.to_torch(limits2) == torch.ones((num_instances, num_dofs, 2), device=device) * 2.0)

        # The underlying lower and upper should be unchanged
        assert torch.all(
            wp.to_torch(articulation_data.joint_pos_limits_lower)
            == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_pos_limits_upper)
            == torch.zeros((num_instances, num_dofs), device=device)
        )


##
# Test Cases -- Joint Properties (Custom).
##


class TestJointPropertiesCustom:
    """Tests the following properties:
    - joint_dynamic_friction_coeff
    - joint_viscous_friction_coeff
    - soft_joint_pos_limits
    - soft_joint_vel_limits
    - gear_ratio

    Runs the following checks:
    - Checks that their types and shapes are correct.
    - Checks that the returned values are pointers to the internal data.

    .. note:: gear_ratio is initialized to ones (not zeros).
    """

    def _setup_method(self, num_instances: int, num_dofs: int, device: str) -> ArticulationData:
        mock_view = MockNewtonArticulationView(num_instances, 1, num_dofs, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )

        return articulation_data

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_initialized_correctly(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that the custom joint properties are initialized correctly."""
        # Setup the articulation data
        articulation_data = self._setup_method(num_instances, num_dofs, device)

        # Check the types are correct
        assert articulation_data.joint_dynamic_friction_coeff.dtype is wp.float32
        assert articulation_data.joint_viscous_friction_coeff.dtype is wp.float32
        assert articulation_data.soft_joint_pos_limits.dtype is wp.vec2f
        assert articulation_data.soft_joint_vel_limits.dtype is wp.float32
        assert articulation_data.gear_ratio.dtype is wp.float32

        # Check the shapes are correct
        assert articulation_data.joint_dynamic_friction_coeff.shape == (num_instances, num_dofs)
        assert articulation_data.joint_viscous_friction_coeff.shape == (num_instances, num_dofs)
        assert articulation_data.soft_joint_pos_limits.shape == (num_instances, num_dofs)
        assert articulation_data.soft_joint_vel_limits.shape == (num_instances, num_dofs)
        assert articulation_data.gear_ratio.shape == (num_instances, num_dofs)

        # Check the values are initialized correctly
        # Most are zeros, but gear_ratio is initialized to ones
        assert torch.all(
            wp.to_torch(articulation_data.joint_dynamic_friction_coeff)
            == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_viscous_friction_coeff)
            == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.soft_joint_pos_limits)
            == torch.zeros((num_instances, num_dofs, 2), device=device)
        )
        assert torch.all(
            wp.to_torch(articulation_data.soft_joint_vel_limits)
            == torch.zeros((num_instances, num_dofs), device=device)
        )
        # gear_ratio is initialized to ones
        assert torch.all(
            wp.to_torch(articulation_data.gear_ratio) == torch.ones((num_instances, num_dofs), device=device)
        )

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_dofs", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_returns_reference(self, mock_newton_manager, num_instances: int, num_dofs: int, device: str):
        """Test that the custom joint properties return a reference to the internal data."""
        # Setup the articulation data
        articulation_data = self._setup_method(num_instances, num_dofs, device)

        # Get the pointers
        joint_dynamic_friction_coeff = articulation_data.joint_dynamic_friction_coeff
        joint_viscous_friction_coeff = articulation_data.joint_viscous_friction_coeff
        soft_joint_pos_limits = articulation_data.soft_joint_pos_limits
        soft_joint_vel_limits = articulation_data.soft_joint_vel_limits
        gear_ratio = articulation_data.gear_ratio

        # Check that they have initial values
        assert torch.all(
            wp.to_torch(joint_dynamic_friction_coeff) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(joint_viscous_friction_coeff) == torch.zeros((num_instances, num_dofs), device=device)
        )
        assert torch.all(wp.to_torch(soft_joint_pos_limits) == torch.zeros((num_instances, num_dofs, 2), device=device))
        assert torch.all(wp.to_torch(soft_joint_vel_limits) == torch.zeros((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(gear_ratio) == torch.ones((num_instances, num_dofs), device=device))

        # Assign a different value to the internal data
        articulation_data.joint_dynamic_friction_coeff.fill_(1.0)
        articulation_data.joint_viscous_friction_coeff.fill_(1.0)
        articulation_data.soft_joint_pos_limits.fill_(1.0)
        articulation_data.soft_joint_vel_limits.fill_(1.0)
        articulation_data.gear_ratio.fill_(2.0)

        # Check that the properties return the new value (reference behavior)
        assert torch.all(
            wp.to_torch(joint_dynamic_friction_coeff) == torch.ones((num_instances, num_dofs), device=device)
        )
        assert torch.all(
            wp.to_torch(joint_viscous_friction_coeff) == torch.ones((num_instances, num_dofs), device=device)
        )
        assert torch.all(wp.to_torch(soft_joint_pos_limits) == torch.ones((num_instances, num_dofs, 2), device=device))
        assert torch.all(wp.to_torch(soft_joint_vel_limits) == torch.ones((num_instances, num_dofs), device=device))
        assert torch.all(wp.to_torch(gear_ratio) == torch.ones((num_instances, num_dofs), device=device) * 2.0)

        # Assign a different value to the pointers
        joint_dynamic_friction_coeff.fill_(3.0)
        joint_viscous_friction_coeff.fill_(3.0)
        soft_joint_pos_limits.fill_(3.0)
        soft_joint_vel_limits.fill_(3.0)
        gear_ratio.fill_(4.0)

        # Check that the internal data has been updated
        assert torch.all(
            wp.to_torch(articulation_data.joint_dynamic_friction_coeff)
            == torch.ones((num_instances, num_dofs), device=device) * 3.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.joint_viscous_friction_coeff)
            == torch.ones((num_instances, num_dofs), device=device) * 3.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.soft_joint_pos_limits)
            == torch.ones((num_instances, num_dofs, 2), device=device) * 3.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.soft_joint_vel_limits)
            == torch.ones((num_instances, num_dofs), device=device) * 3.0
        )
        assert torch.all(
            wp.to_torch(articulation_data.gear_ratio) == torch.ones((num_instances, num_dofs), device=device) * 4.0
        )


##
# Test Cases -- Fixed Tendon Properties.
##


# TODO: Update these tests when fixed tendon support is added to Newton.
class TestFixedTendonProperties:
    """Tests the following properties:
    - fixed_tendon_stiffness
    - fixed_tendon_damping
    - fixed_tendon_limit_stiffness
    - fixed_tendon_rest_length
    - fixed_tendon_offset
    - fixed_tendon_pos_limits

    Currently, all these properties raise NotImplementedError as fixed tendons
    are not supported in Newton.

    Runs the following checks:
    - Checks that all properties raise NotImplementedError.
    """

    def _setup_method(self, num_instances: int, num_dofs: int, device: str) -> ArticulationData:
        mock_view = MockNewtonArticulationView(num_instances, 1, num_dofs, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )

        return articulation_data

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_fixed_tendon_properties_not_implemented(self, mock_newton_manager, device: str):
        """Test that all fixed tendon properties raise NotImplementedError."""
        articulation_data = self._setup_method(1, 1, device)

        with pytest.raises(NotImplementedError):
            _ = articulation_data.fixed_tendon_stiffness
        with pytest.raises(NotImplementedError):
            _ = articulation_data.fixed_tendon_damping
        with pytest.raises(NotImplementedError):
            _ = articulation_data.fixed_tendon_limit_stiffness
        with pytest.raises(NotImplementedError):
            _ = articulation_data.fixed_tendon_rest_length
        with pytest.raises(NotImplementedError):
            _ = articulation_data.fixed_tendon_offset
        with pytest.raises(NotImplementedError):
            _ = articulation_data.fixed_tendon_pos_limits


##
# Test Cases -- Spatial Tendon Properties.
##


# TODO: Update these tests when spatial tendon support is added to Newton.
class TestSpatialTendonProperties:
    """Tests the following properties:
    - spatial_tendon_stiffness
    - spatial_tendon_damping
    - spatial_tendon_limit_stiffness
    - spatial_tendon_offset

    Currently, all these properties raise NotImplementedError as spatial tendons
    are not supported in Newton.

    Runs the following checks:
    - Checks that all properties raise NotImplementedError.
    """

    def _setup_method(self, num_instances: int, num_dofs: int, device: str) -> ArticulationData:
        mock_view = MockNewtonArticulationView(num_instances, 1, num_dofs, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )

        return articulation_data

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_spatial_tendon_properties_not_implemented(self, mock_newton_manager, device: str):
        """Test that all spatial tendon properties raise NotImplementedError."""
        articulation_data = self._setup_method(1, 1, device)

        with pytest.raises(NotImplementedError):
            _ = articulation_data.spatial_tendon_stiffness
        with pytest.raises(NotImplementedError):
            _ = articulation_data.spatial_tendon_damping
        with pytest.raises(NotImplementedError):
            _ = articulation_data.spatial_tendon_limit_stiffness
        with pytest.raises(NotImplementedError):
            _ = articulation_data.spatial_tendon_offset


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

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )

        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_root_link_pose_w(self, mock_newton_manager, num_instances: int, device: str):
        """Test that the root link pose property returns a pointer to the internal data."""
        articulation_data, _ = self._setup_method(num_instances, device)

        # Check the type and shape
        assert articulation_data.root_link_pose_w.shape == (num_instances,)
        assert articulation_data.root_link_pose_w.dtype == wp.transformf

        # Mock data is initialized to zeros
        assert torch.all(wp.to_torch(articulation_data.root_link_pose_w) == torch.zeros((1, 7), device=device))

        # Get the property
        root_link_pose_w = articulation_data.root_link_pose_w

        # Assign a different value to the internal data
        articulation_data.root_link_pose_w.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        assert torch.all(wp.to_torch(articulation_data.root_link_pose_w) == torch.ones((1, 7), device=device))

        # Assign a different value to the pointers
        root_link_pose_w.fill_(2.0)

        # Check that the internal data has been updated
        assert torch.all(wp.to_torch(articulation_data.root_link_pose_w) == torch.ones((1, 7), device=device) * 2.0)


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

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that the root link velocity property is correctly computed."""
        articulation_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert articulation_data.root_link_vel_w.shape == (num_instances,)
        assert articulation_data.root_link_vel_w.dtype == wp.spatial_vectorf

        # Mock data is initialized to zeros
        assert torch.all(
            wp.to_torch(articulation_data.root_link_vel_w) == torch.zeros((num_instances, 6), device=device)
        )

        for i in range(10):
            articulation_data._sim_timestamp = i + 1.0
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

            # Compare the computed value to the one from the articulation data
            assert torch.allclose(wp.to_torch(articulation_data.root_link_vel_w), vel, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_update_timestamp(self, mock_newton_manager, device: str):
        """Test that the timestamp is updated correctly."""
        articulation_data, mock_view = self._setup_method(1, device)

        # Check that the timestamp is initialized to -1.0
        assert articulation_data._root_link_vel_w.timestamp == -1.0

        # Check that the data class timestamp is initialized to 0.0
        assert articulation_data._sim_timestamp == 0.0

        # Request the property
        value = wp.to_torch(articulation_data.root_link_vel_w).clone()

        # Check that the timestamp is updated. The timestamp should be the same as the data class timestamp.
        assert articulation_data._root_link_vel_w.timestamp == articulation_data._sim_timestamp

        # Update the root_com_vel_w
        mock_view.set_mock_data(
            root_velocities=wp.from_torch(torch.rand((1, 6), device=device), dtype=wp.spatial_vectorf),
        )

        # Check that the property value was not updated
        assert torch.all(wp.to_torch(articulation_data.root_link_vel_w) == value)

        # Update the data class timestamp
        articulation_data._sim_timestamp = 1.0

        # Check that the property timestamp was not updated
        assert articulation_data._root_link_vel_w.timestamp != articulation_data._sim_timestamp

        # Check that the property value was updated
        assert torch.all(wp.to_torch(articulation_data.root_link_vel_w) != value)


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

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_root_com_pose_w(self, mock_newton_manager, num_instances: int, device: str):
        """Test that the root center of mass pose property returns a pointer to the internal data."""
        articulation_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert articulation_data.root_com_pose_w.shape == (num_instances,)
        assert articulation_data.root_com_pose_w.dtype == wp.transformf

        # Mock data is initialized to zeros
        assert torch.all(
            wp.to_torch(articulation_data.root_com_pose_w) == torch.zeros((num_instances, 7), device=device)
        )

        for i in range(10):
            articulation_data._sim_timestamp = i + 1.0
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

            # Compare the computed value to the one from the articulation data
            assert torch.allclose(wp.to_torch(articulation_data.root_com_pose_w), root_com_pose, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_update_timestamp(self, mock_newton_manager, device: str):
        """Test that the timestamp is updated correctly."""
        articulation_data, mock_view = self._setup_method(1, device)

        # Check that the timestamp is initialized to -1.0
        assert articulation_data._root_com_pose_w.timestamp == -1.0

        # Check that the data class timestamp is initialized to 0.0
        assert articulation_data._sim_timestamp == 0.0

        # Request the property
        value = wp.to_torch(articulation_data.root_com_pose_w).clone()

        # Check that the timestamp is updated. The timestamp should be the same as the data class timestamp.
        assert articulation_data._root_com_pose_w.timestamp == articulation_data._sim_timestamp

        # Update the root_com_vel_w
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(torch.rand((1, 7), device=device), dtype=wp.transformf),
        )

        # Check that the property value was not updated
        assert torch.all(wp.to_torch(articulation_data.root_com_pose_w) == value)

        # Update the data class timestamp
        articulation_data._sim_timestamp = 1.0

        # Check that the property timestamp was not updated
        assert articulation_data._root_com_pose_w.timestamp != articulation_data._sim_timestamp

        # Check that the property value was updated
        assert torch.all(wp.to_torch(articulation_data.root_com_pose_w) != value)


class TestRootComVelW:
    """Tests the root center of mass velocity property

    This value is read from the simulation. There is no math to check for.

    Checks that the returned value is a pointer to the internal data.
    """

    def _setup_method(
        self, num_instances: int, device: str, is_fixed_base: bool = False
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device, is_fixed_base=is_fixed_base)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_root_com_vel_w(self, mock_newton_manager, num_instances: int, device: str):
        """Test that the root center of mass velocity property returns a pointer to the internal data."""
        articulation_data, _ = self._setup_method(num_instances, device)

        # Check the type and shape
        assert articulation_data.root_com_vel_w.shape == (num_instances,)
        assert articulation_data.root_com_vel_w.dtype == wp.spatial_vectorf

        # Mock data is initialized to zeros
        assert torch.all(
            wp.to_torch(articulation_data.root_com_vel_w) == torch.zeros((num_instances, 6), device=device)
        )

        # Get the property
        root_com_vel_w = articulation_data.root_com_vel_w

        # Assign a different value to the internal data
        articulation_data.root_com_vel_w.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        assert torch.all(wp.to_torch(articulation_data.root_com_vel_w) == torch.ones((num_instances, 6), device=device))

        # Assign a different value to the pointers
        root_com_vel_w.fill_(2.0)

        # Check that the internal data has been updated
        assert torch.all(
            wp.to_torch(articulation_data.root_com_vel_w) == torch.ones((num_instances, 6), device=device) * 2.0
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_fixed_articulation_zero_velocity(self, mock_newton_manager, device: str):
        """Test that the root center of mass velocity is zero for a fixed articulation."""
        articulation_data, mock_view = self._setup_method(1, device, is_fixed_base=True)
        # Check that the root center of mass velocity is zero.
        assert torch.all(wp.to_torch(articulation_data.root_com_vel_w) == torch.zeros((1, 6), device=device))


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

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_root_state_properties(self, mock_newton_manager, num_instances: int, device: str):
        """Test that all root state properties correctly combine pose and velocity."""
        articulation_data, mock_view = self._setup_method(num_instances, device)

        # Generate random mock data
        for i in range(5):
            articulation_data._sim_timestamp = i + 1.0

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
            root_state = wp.to_torch(articulation_data.root_state_w)
            expected_root_state = torch.cat([root_link_pose, com_vel], dim=-1)

            assert root_state.shape == (num_instances, 13)
            assert torch.allclose(root_state, expected_root_state, atol=1e-6, rtol=1e-6)

            # --- Test root_link_state_w ---
            # Combines root_link_pose_w with root_link_vel_w
            root_link_state = wp.to_torch(articulation_data.root_link_state_w)

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
            root_com_state = wp.to_torch(articulation_data.root_com_state_w)

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
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_body_mass_and_inertia(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that body_mass and body_inertia have correct types, shapes, and reference behavior."""
        articulation_data, mock_view = self._setup_method(num_instances, num_bodies, device)

        # --- Test body_mass ---
        # Check the type and shape
        assert articulation_data.body_mass.shape == (num_instances, num_bodies)
        assert articulation_data.body_mass.dtype == wp.float32

        # Mock data initializes body_mass to ones
        assert torch.all(
            wp.to_torch(articulation_data.body_mass) == torch.zeros((num_instances, num_bodies), device=device)
        )

        # Get the property reference
        body_mass_ref = articulation_data.body_mass

        # Assign a different value to the internal data via property
        articulation_data.body_mass.fill_(2.0)

        # Check that the property returns the new value (reference behavior)
        assert torch.all(
            wp.to_torch(articulation_data.body_mass) == torch.ones((num_instances, num_bodies), device=device) * 2.0
        )

        # Assign a different value via reference
        body_mass_ref.fill_(3.0)

        # Check that the internal data has been updated
        assert torch.all(
            wp.to_torch(articulation_data.body_mass) == torch.ones((num_instances, num_bodies), device=device) * 3.0
        )

        # --- Test body_inertia ---
        # Check the type and shape
        assert articulation_data.body_inertia.shape == (num_instances, num_bodies)
        assert articulation_data.body_inertia.dtype == wp.mat33f

        # Mock data initializes body_inertia to zeros
        expected_inertia = torch.zeros((num_instances, num_bodies, 3, 3), device=device)
        assert torch.all(wp.to_torch(articulation_data.body_inertia) == expected_inertia)

        # Get the property reference
        body_inertia_ref = articulation_data.body_inertia

        # Assign a different value to the internal data via property
        articulation_data.body_inertia.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        expected_inertia_ones = torch.ones((num_instances, num_bodies, 3, 3), device=device)
        assert torch.all(wp.to_torch(articulation_data.body_inertia) == expected_inertia_ones)

        # Assign a different value via reference
        body_inertia_ref.fill_(2.0)

        # Check that the internal data has been updated
        expected_inertia_twos = torch.ones((num_instances, num_bodies, 3, 3), device=device) * 2.0
        assert torch.all(wp.to_torch(articulation_data.body_inertia) == expected_inertia_twos)


class TestBodyLinkPoseW:
    """Tests the body link pose property.

    This value is read directly from the simulation bindings.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is a reference to the internal data.
    """

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_body_link_pose_w(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that body_link_pose_w has correct type, shape, and reference behavior."""
        articulation_data, _ = self._setup_method(num_instances, num_bodies, device)

        # Check the type and shape
        assert articulation_data.body_link_pose_w.shape == (num_instances, num_bodies)
        assert articulation_data.body_link_pose_w.dtype == wp.transformf

        # Mock data is initialized to zeros
        expected = torch.zeros((num_instances, num_bodies, 7), device=device)
        assert torch.all(wp.to_torch(articulation_data.body_link_pose_w) == expected)

        # Get the property reference
        body_link_pose_ref = articulation_data.body_link_pose_w

        # Assign a different value via property
        articulation_data.body_link_pose_w.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        expected_ones = torch.ones((num_instances, num_bodies, 7), device=device)
        assert torch.all(wp.to_torch(articulation_data.body_link_pose_w) == expected_ones)

        # Assign a different value via reference
        body_link_pose_ref.fill_(2.0)

        # Check that the internal data has been updated
        expected_twos = torch.ones((num_instances, num_bodies, 7), device=device) * 2.0
        assert torch.all(wp.to_torch(articulation_data.body_link_pose_w) == expected_twos)


class TestBodyLinkVelW:
    """Tests the body link velocity property.

    This value is derived from body COM velocity. To ensure correctness,
    we compare against the reference implementation.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that the timestamp is updated correctly.
    """

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that body_link_vel_w is correctly computed from COM velocity."""
        articulation_data, mock_view = self._setup_method(num_instances, num_bodies, device)

        # Check the type and shape
        assert articulation_data.body_link_vel_w.shape == (num_instances, num_bodies)
        assert articulation_data.body_link_vel_w.dtype == wp.spatial_vectorf

        # Mock data is initialized to zeros
        expected = torch.zeros((num_instances, num_bodies, 6), device=device)
        assert torch.all(wp.to_torch(articulation_data.body_link_vel_w) == expected)

        for i in range(5):
            articulation_data._sim_timestamp = i + 1.0

            # Generate random COM velocity and body COM position
            com_vel = torch.rand((num_instances, num_bodies, 6), device=device)
            body_com_pos = torch.rand((num_instances, num_bodies, 3), device=device)

            # Generate random link poses with normalized quaternions
            link_pose = torch.zeros((num_instances, num_bodies, 7), device=device)
            link_pose[..., :3] = torch.rand((num_instances, num_bodies, 3), device=device)
            link_pose[..., 3:] = torch.randn((num_instances, num_bodies, 4), device=device)
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
            assert torch.allclose(wp.to_torch(articulation_data.body_link_vel_w), expected_vel, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        articulation_data, mock_view = self._setup_method(1, 1, device)

        # Check initial timestamp
        assert articulation_data._body_link_vel_w.timestamp == -1.0
        assert articulation_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(articulation_data.body_link_vel_w).clone()

        # Check that buffer timestamp matches sim timestamp
        assert articulation_data._body_link_vel_w.timestamp == articulation_data._sim_timestamp

        # Update mock data without changing sim timestamp
        mock_view.set_mock_data(
            link_velocities=wp.from_torch(torch.rand((1, 1, 6), device=device), dtype=wp.spatial_vectorf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(articulation_data.body_link_vel_w) == value)

        # Update sim timestamp
        articulation_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert articulation_data._body_link_vel_w.timestamp != articulation_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(articulation_data.body_link_vel_w) == value)


class TestBodyComPoseW:
    """Tests the body center of mass pose property.

    This value is derived from body link pose and body COM position.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that the timestamp is updated correctly.
    """

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that body_com_pose_w is correctly computed from link pose and COM position."""
        articulation_data, mock_view = self._setup_method(num_instances, num_bodies, device)

        # Check the type and shape
        assert articulation_data.body_com_pose_w.shape == (num_instances, num_bodies)
        assert articulation_data.body_com_pose_w.dtype == wp.transformf

        # Mock data is initialized to zeros
        expected = torch.zeros((num_instances, num_bodies, 7), device=device)
        assert torch.all(wp.to_torch(articulation_data.body_com_pose_w) == expected)

        for i in range(5):
            articulation_data._sim_timestamp = i + 1.0

            # Generate random link poses with normalized quaternions
            link_pose = torch.zeros((num_instances, num_bodies, 7), device=device)
            link_pose[..., :3] = torch.rand((num_instances, num_bodies, 3), device=device)
            link_pose[..., 3:] = torch.randn((num_instances, num_bodies, 4), device=device)
            link_pose[..., 3:] = torch.nn.functional.normalize(link_pose[..., 3:], p=2.0, dim=-1, eps=1e-12)

            # Generate random body COM position in body frame
            body_com_pos = torch.rand((num_instances, num_bodies, 3), device=device)

            mock_view.set_mock_data(
                link_transforms=wp.from_torch(link_pose, dtype=wp.transformf),
                body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
            )

            # Compute expected COM pose using IsaacLab reference implementation
            # combine_frame_transforms(link_pos, link_quat, com_pos_b, identity_quat)
            body_com_quat_b = torch.zeros((num_instances, num_bodies, 4), device=device)
            body_com_quat_b[..., 3] = 1.0  # identity quaternion

            expected_pos, expected_quat = math_utils.combine_frame_transforms(
                link_pose[..., :3], link_pose[..., 3:], body_com_pos, body_com_quat_b
            )
            expected_pose = torch.cat([expected_pos, expected_quat], dim=-1)

            # Compare the computed value
            assert torch.allclose(wp.to_torch(articulation_data.body_com_pose_w), expected_pose, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        articulation_data, mock_view = self._setup_method(1, 1, device)

        # Check initial timestamp
        assert articulation_data._body_com_pose_w.timestamp == -1.0
        assert articulation_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(articulation_data.body_com_pose_w).clone()

        # Check that buffer timestamp matches sim timestamp
        assert articulation_data._body_com_pose_w.timestamp == articulation_data._sim_timestamp

        # Update mock data without changing sim timestamp
        mock_view.set_mock_data(
            link_transforms=wp.from_torch(torch.rand((1, 1, 7), device=device), dtype=wp.transformf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(articulation_data.body_com_pose_w) == value)

        # Update sim timestamp
        articulation_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert articulation_data._body_com_pose_w.timestamp != articulation_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(articulation_data.body_com_pose_w) == value)


class TestBodyComVelW:
    """Tests the body center of mass velocity property.

    This value is read directly from the simulation bindings.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is a reference to the internal data.
    """

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str, is_fixed_base: bool = False
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device, is_fixed_base=is_fixed_base)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_body_com_vel_w(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that body_com_vel_w has correct type, shape, and reference behavior."""
        articulation_data, _ = self._setup_method(num_instances, num_bodies, device)

        # Check the type and shape
        assert articulation_data.body_com_vel_w.shape == (num_instances, num_bodies)
        assert articulation_data.body_com_vel_w.dtype == wp.spatial_vectorf

        # Mock data is initialized to zeros
        expected = torch.zeros((num_instances, num_bodies, 6), device=device)
        assert torch.all(wp.to_torch(articulation_data.body_com_vel_w) == expected)

        # Get the property reference
        body_com_vel_ref = articulation_data.body_com_vel_w

        # Assign a different value via property
        articulation_data.body_com_vel_w.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        expected_ones = torch.ones((num_instances, num_bodies, 6), device=device)
        assert torch.all(wp.to_torch(articulation_data.body_com_vel_w) == expected_ones)

        # Assign a different value via reference
        body_com_vel_ref.fill_(2.0)

        # Check that the internal data has been updated
        expected_twos = torch.ones((num_instances, num_bodies, 6), device=device) * 2.0
        assert torch.all(wp.to_torch(articulation_data.body_com_vel_w) == expected_twos)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_fixed_articulation_zero_velocity(self, mock_newton_manager, device: str):
        """Test that the body center of mass velocity is zero for a fixed articulation."""
        articulation_data, mock_view = self._setup_method(1, 1, device, is_fixed_base=True)
        # Check that the root center of mass velocity is zero.
        assert torch.all(wp.to_torch(articulation_data.body_com_vel_w) == torch.zeros((1, 1, 6), device=device))


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

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_body_state_properties(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that all body state properties correctly combine pose and velocity."""
        articulation_data, mock_view = self._setup_method(num_instances, num_bodies, device)

        # Generate random mock data
        for i in range(5):
            articulation_data._sim_timestamp = i + 1.0

            # Generate random body link pose with normalized quaternions
            body_link_pose = torch.zeros((num_instances, num_bodies, 7), device=device)
            body_link_pose[..., :3] = torch.rand((num_instances, num_bodies, 3), device=device)
            body_link_pose[..., 3:] = torch.randn((num_instances, num_bodies, 4), device=device)
            body_link_pose[..., 3:] = torch.nn.functional.normalize(body_link_pose[..., 3:], p=2.0, dim=-1, eps=1e-12)

            # Generate random COM velocities and COM position
            com_vel = torch.rand((num_instances, num_bodies, 6), device=device)
            body_com_pos = torch.rand((num_instances, num_bodies, 3), device=device)

            mock_view.set_mock_data(
                link_transforms=wp.from_torch(body_link_pose, dtype=wp.transformf),
                link_velocities=wp.from_torch(com_vel, dtype=wp.spatial_vectorf),
                body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
            )

            # --- Test body_state_w ---
            # Combines body_link_pose_w with body_com_vel_w
            body_state = wp.to_torch(articulation_data.body_state_w)
            expected_body_state = torch.cat([body_link_pose, com_vel], dim=-1)

            assert body_state.shape == (num_instances, num_bodies, 13)
            assert torch.allclose(body_state, expected_body_state, atol=1e-6, rtol=1e-6)

            # --- Test body_link_state_w ---
            # Combines body_link_pose_w with body_link_vel_w
            body_link_state = wp.to_torch(articulation_data.body_link_state_w)

            # Compute expected body_link_vel from com_vel (same as TestBodyLinkVelW)
            body_link_vel = com_vel.clone()
            body_link_vel[..., :3] += torch.linalg.cross(
                body_link_vel[..., 3:],
                math_utils.quat_apply(body_link_pose[..., 3:], -body_com_pos),
                dim=-1,
            )
            expected_body_link_state = torch.cat([body_link_pose, body_link_vel], dim=-1)

            assert body_link_state.shape == (num_instances, num_bodies, 13)
            assert torch.allclose(body_link_state, expected_body_link_state, atol=1e-6, rtol=1e-6)

            # --- Test body_com_state_w ---
            # Combines body_com_pose_w with body_com_vel_w
            body_com_state = wp.to_torch(articulation_data.body_com_state_w)

            # Compute expected body_com_pose from body_link_pose and body_com_pos (same as TestBodyComPoseW)
            body_com_quat_b = torch.zeros((num_instances, num_bodies, 4), device=device)
            body_com_quat_b[..., 3] = 1.0
            body_com_pos_w, body_com_quat_w = math_utils.combine_frame_transforms(
                body_link_pose[..., :3], body_link_pose[..., 3:], body_com_pos, body_com_quat_b
            )
            expected_body_com_state = torch.cat([body_com_pos_w, body_com_quat_w, com_vel], dim=-1)

            assert body_com_state.shape == (num_instances, num_bodies, 13)
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
        self, num_instances: int, num_bodies: int, device: str, initial_vel: torch.Tensor | None = None
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)

        # Set initial velocities (these become _previous_body_com_vel)
        if initial_vel is not None:
            mock_view.set_mock_data(
                link_velocities=wp.from_torch(initial_vel, dtype=wp.spatial_vectorf),
            )
        else:
            mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that body_com_acc_w is correctly computed from velocity finite differencing."""
        # Initial velocity (becomes previous_velocity)
        previous_vel = torch.rand((num_instances, num_bodies, 6), device=device)
        articulation_data, mock_view = self._setup_method(num_instances, num_bodies, device, previous_vel)

        # Check the type and shape
        assert articulation_data.body_com_acc_w.shape == (num_instances, num_bodies)
        assert articulation_data.body_com_acc_w.dtype == wp.spatial_vectorf

        # dt is mocked as 0.01
        dt = 0.01

        for i in range(10):
            articulation_data._sim_timestamp = i + 1.0

            # Generate new random velocity
            current_vel = torch.rand((num_instances, num_bodies, 6), device=device)
            mock_view.set_mock_data(
                link_velocities=wp.from_torch(current_vel, dtype=wp.spatial_vectorf),
            )

            # Compute expected acceleration: (current - previous) / dt
            expected_acc = (current_vel - previous_vel) / dt

            # Compare the computed value
            assert torch.allclose(wp.to_torch(articulation_data.body_com_acc_w), expected_acc, atol=1e-5, rtol=1e-5)
            # Update previous velocity
            previous_vel = current_vel.clone()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        initial_vel = torch.zeros((1, 1, 6), device=device)
        articulation_data, mock_view = self._setup_method(1, 1, device, initial_vel)

        # Check initial timestamp
        assert articulation_data._body_com_acc_w.timestamp == -1.0
        assert articulation_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(articulation_data.body_com_acc_w).clone()

        # Check that buffer timestamp matches sim timestamp
        assert articulation_data._body_com_acc_w.timestamp == articulation_data._sim_timestamp

        # Update mock data without changing sim timestamp
        mock_view.set_mock_data(
            link_velocities=wp.from_torch(torch.rand((1, 1, 6), device=device), dtype=wp.spatial_vectorf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(articulation_data.body_com_acc_w) == value)

        # Update sim timestamp
        articulation_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert articulation_data._body_com_acc_w.timestamp != articulation_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(articulation_data.body_com_acc_w) == value)


class TestBodyComPoseB:
    """Tests the body center of mass pose in body frame property.

    This value is generated from COM position with identity quaternion.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value correctly combines position with identity quaternion.
    """

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_body_com_pose_b(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that body_com_pose_b correctly generates pose from position with identity quaternion."""
        articulation_data, mock_view = self._setup_method(num_instances, num_bodies, device)

        # Check the type and shape
        assert articulation_data.body_com_pose_b.shape == (num_instances, num_bodies)
        assert articulation_data.body_com_pose_b.dtype == wp.transformf

        # Mock data is initialized to zeros for COM position
        # Expected pose: [0, 0, 0, 0, 0, 0, 1] (position zeros, identity quaternion)
        expected = torch.zeros((num_instances, num_bodies, 7), device=device)
        expected[..., 6] = 1.0  # w component of identity quaternion
        assert torch.all(wp.to_torch(articulation_data.body_com_pose_b) == expected)

        # Update COM position and verify
        com_pos = torch.rand((num_instances, num_bodies, 3), device=device)
        mock_view.set_mock_data(
            body_com_pos=wp.from_torch(com_pos, dtype=wp.vec3f),
        )

        # Get the pose
        pose = wp.to_torch(articulation_data.body_com_pose_b)

        # Expected: position from mock, identity quaternion
        expected_pose = torch.zeros((num_instances, num_bodies, 7), device=device)
        expected_pose[..., :3] = com_pos
        expected_pose[..., 6] = 1.0  # w component

        assert torch.allclose(pose, expected_pose, atol=1e-6, rtol=1e-6)


# TODO: Update this test when body_incoming_joint_wrench_b support is added to Newton.
class TestBodyIncomingJointWrenchB:
    """Tests the body incoming joint wrench property.

    Currently, this property raises NotImplementedError as joint wrenches
    are not supported in Newton.

    Runs the following checks:
    - Checks that the property raises NotImplementedError.
    """

    def _setup_method(self, num_instances: int, num_bodies: int, device: str) -> ArticulationData:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_not_implemented(self, mock_newton_manager, device: str):
        """Test that body_incoming_joint_wrench_b raises NotImplementedError."""
        articulation_data = self._setup_method(1, 1, device)

        with pytest.raises(NotImplementedError):
            _ = articulation_data.body_incoming_joint_wrench_b


##
# Test Cases -- Joint state properties.
##


class TestJointPosVel:
    """Tests the joint position and velocity properties.

    These values are read directly from the simulation bindings.

    Tests the following properties:
    - joint_pos
    - joint_vel

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is a reference to the internal data.
    """

    def _setup_method(
        self, num_instances: int, num_joints: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, num_joints, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_joint_pos_and_vel(self, mock_newton_manager, num_instances: int, num_joints: int, device: str):
        """Test that joint_pos and joint_vel have correct type, shape, and reference behavior."""
        articulation_data, mock_view = self._setup_method(num_instances, num_joints, device)

        # --- Test joint_pos ---
        # Check the type and shape
        assert articulation_data.joint_pos.shape == (num_instances, num_joints)
        assert articulation_data.joint_pos.dtype == wp.float32

        # Mock data is initialized to zeros
        expected = torch.zeros((num_instances, num_joints), device=device)
        assert torch.all(wp.to_torch(articulation_data.joint_pos) == expected)

        # Get the property reference
        joint_pos_ref = articulation_data.joint_pos

        # Assign a different value via property
        articulation_data.joint_pos.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        expected_ones = torch.ones((num_instances, num_joints), device=device)
        assert torch.all(wp.to_torch(articulation_data.joint_pos) == expected_ones)

        # Assign a different value via reference
        joint_pos_ref.fill_(2.0)

        # Check that the internal data has been updated
        expected_twos = torch.ones((num_instances, num_joints), device=device) * 2.0
        assert torch.all(wp.to_torch(articulation_data.joint_pos) == expected_twos)

        # --- Test joint_vel ---
        # Check the type and shape
        assert articulation_data.joint_vel.shape == (num_instances, num_joints)
        assert articulation_data.joint_vel.dtype == wp.float32

        # Mock data is initialized to zeros
        expected = torch.zeros((num_instances, num_joints), device=device)
        assert torch.all(wp.to_torch(articulation_data.joint_vel) == expected)

        # Get the property reference
        joint_vel_ref = articulation_data.joint_vel

        # Assign a different value via property
        articulation_data.joint_vel.fill_(1.0)

        # Check that the property returns the new value (reference behavior)
        expected_ones = torch.ones((num_instances, num_joints), device=device)
        assert torch.all(wp.to_torch(articulation_data.joint_vel) == expected_ones)

        # Assign a different value via reference
        joint_vel_ref.fill_(2.0)

        # Check that the internal data has been updated
        expected_twos = torch.ones((num_instances, num_joints), device=device) * 2.0
        assert torch.all(wp.to_torch(articulation_data.joint_vel) == expected_twos)


class TestJointAcc:
    """Tests the joint acceleration property.

    This value is derived from velocity finite differencing: (current_vel - previous_vel) / dt

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that the timestamp is updated correctly.
    """

    def _setup_method(
        self, num_instances: int, num_joints: int, device: str, initial_vel: torch.Tensor | None = None
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, num_joints, device)

        # Set initial velocities (these become _previous_joint_vel)
        if initial_vel is not None:
            mock_view.set_mock_data(
                dof_velocities=wp.from_torch(initial_vel, dtype=wp.float32),
            )
        else:
            mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, num_joints: int, device: str):
        """Test that joint_acc is correctly computed from velocity finite differencing."""
        # Initial velocity (becomes previous_velocity)
        previous_vel = torch.rand((num_instances, num_joints), device=device)
        articulation_data, mock_view = self._setup_method(num_instances, num_joints, device, previous_vel)

        # Check the type and shape
        assert articulation_data.joint_acc.shape == (num_instances, num_joints)
        assert articulation_data.joint_acc.dtype == wp.float32

        # dt is mocked as 0.01
        dt = 0.01

        for i in range(5):
            articulation_data._sim_timestamp = i + 1.0

            # Generate new random velocity
            current_vel = torch.rand((num_instances, num_joints), device=device)
            mock_view.set_mock_data(
                dof_velocities=wp.from_torch(current_vel, dtype=wp.float32),
            )

            # Compute expected acceleration: (current - previous) / dt
            expected_acc = (current_vel - previous_vel) / dt

            # Compare the computed value
            assert torch.allclose(wp.to_torch(articulation_data.joint_acc), expected_acc, atol=1e-5, rtol=1e-5)
            # Update previous velocity
            previous_vel = current_vel.clone()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        initial_vel = torch.zeros((1, 1), device=device)
        articulation_data, mock_view = self._setup_method(1, 1, device, initial_vel)

        # Check initial timestamp
        assert articulation_data._joint_acc.timestamp == -1.0
        assert articulation_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(articulation_data.joint_acc).clone()

        # Check that buffer timestamp matches sim timestamp
        assert articulation_data._joint_acc.timestamp == articulation_data._sim_timestamp

        # Update mock data without changing sim timestamp
        mock_view.set_mock_data(
            dof_velocities=wp.from_torch(torch.rand((1, 1), device=device), dtype=wp.float32),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(articulation_data.joint_acc) == value)

        # Update sim timestamp
        articulation_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert articulation_data._joint_acc.timestamp != articulation_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(articulation_data.joint_acc) == value)


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

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that projected_gravity_b is correctly computed."""
        articulation_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert articulation_data.projected_gravity_b.shape == (num_instances,)
        assert articulation_data.projected_gravity_b.dtype == wp.vec3f

        # Gravity direction (normalized)
        gravity_dir = torch.tensor([0.0, 0.0, -1.0], device=device)

        for i in range(10):
            articulation_data._sim_timestamp = i + 1.0
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
            assert torch.allclose(wp.to_torch(articulation_data.projected_gravity_b), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        articulation_data, mock_view = self._setup_method(1, device)

        # Check initial timestamp
        assert articulation_data._projected_gravity_b.timestamp == -1.0
        assert articulation_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(articulation_data.projected_gravity_b).clone()

        # Check that buffer timestamp matches sim timestamp
        assert articulation_data._projected_gravity_b.timestamp == articulation_data._sim_timestamp

        # Update mock data without changing sim timestamp
        new_pose = torch.zeros((1, 7), device=device)
        new_pose[:, 3:] = torch.randn((1, 4), device=device)
        new_pose[:, 3:] = torch.nn.functional.normalize(new_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(new_pose, dtype=wp.transformf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(articulation_data.projected_gravity_b) == value)

        # Update sim timestamp
        articulation_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert articulation_data._projected_gravity_b.timestamp != articulation_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(articulation_data.projected_gravity_b) == value)


class TestHeadingW:
    """Tests the heading in world frame property.

    This value is derived by computing the yaw angle from the forward direction.

    Runs the following checks:
    - Checks that the returned value has the correct type and shape.
    - Checks that the returned value is correctly computed.
    - Checks that the timestamp is updated correctly.
    """

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that heading_w is correctly computed."""
        articulation_data, mock_view = self._setup_method(num_instances, device)

        # Check the type and shape
        assert articulation_data.heading_w.shape == (num_instances,)
        assert articulation_data.heading_w.dtype == wp.float32

        # Forward direction in body frame
        forward_vec_b = torch.tensor([1.0, 0.0, 0.0], device=device)

        for i in range(10):
            articulation_data._sim_timestamp = i + 1.0
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
            assert torch.allclose(wp.to_torch(articulation_data.heading_w), expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        articulation_data, mock_view = self._setup_method(1, device)

        # Check initial timestamp
        assert articulation_data._heading_w.timestamp == -1.0
        assert articulation_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(articulation_data.heading_w).clone()

        # Check that buffer timestamp matches sim timestamp
        assert articulation_data._heading_w.timestamp == articulation_data._sim_timestamp

        # Update mock data without changing sim timestamp
        new_pose = torch.zeros((1, 7), device=device)
        new_pose[:, 3:] = torch.randn((1, 4), device=device)
        new_pose[:, 3:] = torch.nn.functional.normalize(new_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(new_pose, dtype=wp.transformf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(articulation_data.heading_w) == value)

        # Update sim timestamp
        articulation_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert articulation_data._heading_w.timestamp != articulation_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(articulation_data.heading_w) == value)


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

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that root_link_vel_b and its slices are correctly computed."""
        articulation_data, mock_view = self._setup_method(num_instances, device)

        # Check types and shapes
        assert articulation_data.root_link_vel_b.shape == (num_instances,)
        assert articulation_data.root_link_vel_b.dtype == wp.spatial_vectorf

        assert articulation_data.root_link_lin_vel_b.shape == (num_instances,)
        assert articulation_data.root_link_lin_vel_b.dtype == wp.vec3f

        assert articulation_data.root_link_ang_vel_b.shape == (num_instances,)
        assert articulation_data.root_link_ang_vel_b.dtype == wp.vec3f

        for i in range(5):
            articulation_data._sim_timestamp = i + 1.0

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
            computed_vel_b = wp.to_torch(articulation_data.root_link_vel_b)
            computed_lin_vel_b = wp.to_torch(articulation_data.root_link_lin_vel_b)
            computed_ang_vel_b = wp.to_torch(articulation_data.root_link_ang_vel_b)

            # Compare full velocity
            assert torch.allclose(computed_vel_b, expected_vel_b, atol=1e-6, rtol=1e-6)

            # Check that lin/ang velocities are correct slices
            assert torch.allclose(computed_lin_vel_b, computed_vel_b[:, :3], atol=1e-6, rtol=1e-6)
            assert torch.allclose(computed_ang_vel_b, computed_vel_b[:, 3:], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        articulation_data, mock_view = self._setup_method(1, device)

        # Check initial timestamp
        assert articulation_data._root_link_vel_b.timestamp == -1.0
        assert articulation_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(articulation_data.root_link_vel_b).clone()

        # Check that buffer timestamp matches sim timestamp
        assert articulation_data._root_link_vel_b.timestamp == articulation_data._sim_timestamp

        # Update mock data without changing sim timestamp
        new_pose = torch.zeros((1, 7), device=device)
        new_pose[:, 3:] = torch.randn((1, 4), device=device)
        new_pose[:, 3:] = torch.nn.functional.normalize(new_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(new_pose, dtype=wp.transformf),
            root_velocities=wp.from_torch(torch.rand((1, 6), device=device), dtype=wp.spatial_vectorf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(articulation_data.root_link_vel_b) == value)

        # Update sim timestamp
        articulation_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert articulation_data._root_link_vel_b.timestamp != articulation_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(articulation_data.root_link_vel_b) == value)


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

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_correctness(self, mock_newton_manager, num_instances: int, device: str):
        """Test that root_com_vel_b and its slices are correctly computed."""
        articulation_data, mock_view = self._setup_method(num_instances, device)

        # Check types and shapes
        assert articulation_data.root_com_vel_b.shape == (num_instances,)
        assert articulation_data.root_com_vel_b.dtype == wp.spatial_vectorf

        assert articulation_data.root_com_lin_vel_b.shape == (num_instances,)
        assert articulation_data.root_com_lin_vel_b.dtype == wp.vec3f

        assert articulation_data.root_com_ang_vel_b.shape == (num_instances,)
        assert articulation_data.root_com_ang_vel_b.dtype == wp.vec3f

        for i in range(5):
            articulation_data._sim_timestamp = i + 1.0

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
            computed_vel_b = wp.to_torch(articulation_data.root_com_vel_b)
            computed_lin_vel_b = wp.to_torch(articulation_data.root_com_lin_vel_b)
            computed_ang_vel_b = wp.to_torch(articulation_data.root_com_ang_vel_b)

            # Compare full velocity
            assert torch.allclose(computed_vel_b, expected_vel_b, atol=1e-6, rtol=1e-6)

            # Check that lin/ang velocities are correct slices
            assert torch.allclose(computed_lin_vel_b, computed_vel_b[:, :3], atol=1e-6, rtol=1e-6)
            assert torch.allclose(computed_ang_vel_b, computed_vel_b[:, 3:], atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_timestamp_invalidation(self, mock_newton_manager, device: str):
        """Test that data is invalidated when timestamp is updated."""
        articulation_data, mock_view = self._setup_method(1, device)

        # Check initial timestamp
        assert articulation_data._root_com_vel_b.timestamp == -1.0
        assert articulation_data._sim_timestamp == 0.0

        # Request the property to trigger computation
        value = wp.to_torch(articulation_data.root_com_vel_b).clone()

        # Check that buffer timestamp matches sim timestamp
        assert articulation_data._root_com_vel_b.timestamp == articulation_data._sim_timestamp

        # Update mock data without changing sim timestamp
        new_pose = torch.zeros((1, 7), device=device)
        new_pose[:, 3:] = torch.randn((1, 4), device=device)
        new_pose[:, 3:] = torch.nn.functional.normalize(new_pose[:, 3:], p=2.0, dim=-1, eps=1e-12)
        mock_view.set_mock_data(
            root_transforms=wp.from_torch(new_pose, dtype=wp.transformf),
            root_velocities=wp.from_torch(torch.rand((1, 6), device=device), dtype=wp.spatial_vectorf),
        )

        # Value should NOT change (cached)
        assert torch.all(wp.to_torch(articulation_data.root_com_vel_b) == value)

        # Update sim timestamp
        articulation_data._sim_timestamp = 1.0

        # Buffer timestamp should now be stale
        assert articulation_data._root_com_vel_b.timestamp != articulation_data._sim_timestamp

        # Value should now be recomputed (different from cached)
        assert not torch.all(wp.to_torch(articulation_data.root_com_vel_b) == value)


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

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_root_sliced_properties(self, mock_newton_manager, num_instances: int, device: str):
        """Test that all root sliced properties are correct slices of their parent properties."""
        articulation_data, mock_view = self._setup_method(num_instances, device)

        # Set up random mock data to ensure non-trivial values
        articulation_data._sim_timestamp = 1.0

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
        root_link_pose = wp.to_torch(articulation_data.root_link_pose_w)
        root_link_pos = wp.to_torch(articulation_data.root_link_pos_w)
        root_link_quat = wp.to_torch(articulation_data.root_link_quat_w)

        assert root_link_pos.shape == (num_instances, 3)
        assert root_link_quat.shape == (num_instances, 4)
        assert torch.allclose(root_link_pos, root_link_pose[:, :3], atol=1e-6)
        assert torch.allclose(root_link_quat, root_link_pose[:, 3:], atol=1e-6)

        # --- Test root_link_vel_w slices ---
        root_link_vel = wp.to_torch(articulation_data.root_link_vel_w)
        root_link_lin_vel = wp.to_torch(articulation_data.root_link_lin_vel_w)
        root_link_ang_vel = wp.to_torch(articulation_data.root_link_ang_vel_w)

        assert root_link_lin_vel.shape == (num_instances, 3)
        assert root_link_ang_vel.shape == (num_instances, 3)
        assert torch.allclose(root_link_lin_vel, root_link_vel[:, :3], atol=1e-6)
        assert torch.allclose(root_link_ang_vel, root_link_vel[:, 3:], atol=1e-6)

        # --- Test root_com_pose_w slices ---
        root_com_pose = wp.to_torch(articulation_data.root_com_pose_w)
        root_com_pos = wp.to_torch(articulation_data.root_com_pos_w)
        root_com_quat = wp.to_torch(articulation_data.root_com_quat_w)

        assert root_com_pos.shape == (num_instances, 3)
        assert root_com_quat.shape == (num_instances, 4)
        assert torch.allclose(root_com_pos, root_com_pose[:, :3], atol=1e-6)
        assert torch.allclose(root_com_quat, root_com_pose[:, 3:], atol=1e-6)

        # --- Test root_com_vel_w slices ---
        root_com_vel = wp.to_torch(articulation_data.root_com_vel_w)
        root_com_lin_vel = wp.to_torch(articulation_data.root_com_lin_vel_w)
        root_com_ang_vel = wp.to_torch(articulation_data.root_com_ang_vel_w)

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

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_body_sliced_properties(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that all body sliced properties are correct slices of their parent properties."""
        articulation_data, mock_view = self._setup_method(num_instances, num_bodies, device)

        # Set up random mock data to ensure non-trivial values
        articulation_data._sim_timestamp = 1.0

        body_pose = torch.zeros((num_instances, num_bodies, 7), device=device)
        body_pose[..., :3] = torch.rand((num_instances, num_bodies, 3), device=device)
        body_pose[..., 3:] = torch.randn((num_instances, num_bodies, 4), device=device)
        body_pose[..., 3:] = torch.nn.functional.normalize(body_pose[..., 3:], p=2.0, dim=-1, eps=1e-12)

        body_vel = torch.rand((num_instances, num_bodies, 6), device=device)
        body_com_pos = torch.rand((num_instances, num_bodies, 3), device=device)

        mock_view.set_mock_data(
            link_transforms=wp.from_torch(body_pose, dtype=wp.transformf),
            link_velocities=wp.from_torch(body_vel, dtype=wp.spatial_vectorf),
            body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
        )

        # --- Test body_link_pose_w slices ---
        body_link_pose = wp.to_torch(articulation_data.body_link_pose_w)
        body_link_pos = wp.to_torch(articulation_data.body_link_pos_w)
        body_link_quat = wp.to_torch(articulation_data.body_link_quat_w)

        assert body_link_pos.shape == (num_instances, num_bodies, 3)
        assert body_link_quat.shape == (num_instances, num_bodies, 4)
        assert torch.allclose(body_link_pos, body_link_pose[..., :3], atol=1e-6)
        assert torch.allclose(body_link_quat, body_link_pose[..., 3:], atol=1e-6)

        # --- Test body_link_vel_w slices ---
        body_link_vel = wp.to_torch(articulation_data.body_link_vel_w)
        body_link_lin_vel = wp.to_torch(articulation_data.body_link_lin_vel_w)
        body_link_ang_vel = wp.to_torch(articulation_data.body_link_ang_vel_w)

        assert body_link_lin_vel.shape == (num_instances, num_bodies, 3)
        assert body_link_ang_vel.shape == (num_instances, num_bodies, 3)
        assert torch.allclose(body_link_lin_vel, body_link_vel[..., :3], atol=1e-6)
        assert torch.allclose(body_link_ang_vel, body_link_vel[..., 3:], atol=1e-6)

        # --- Test body_com_pose_w slices ---
        body_com_pose = wp.to_torch(articulation_data.body_com_pose_w)
        body_com_pos_w = wp.to_torch(articulation_data.body_com_pos_w)
        body_com_quat_w = wp.to_torch(articulation_data.body_com_quat_w)

        assert body_com_pos_w.shape == (num_instances, num_bodies, 3)
        assert body_com_quat_w.shape == (num_instances, num_bodies, 4)
        assert torch.allclose(body_com_pos_w, body_com_pose[..., :3], atol=1e-6)
        assert torch.allclose(body_com_quat_w, body_com_pose[..., 3:], atol=1e-6)

        # --- Test body_com_vel_w slices ---
        body_com_vel = wp.to_torch(articulation_data.body_com_vel_w)
        body_com_lin_vel = wp.to_torch(articulation_data.body_com_lin_vel_w)
        body_com_ang_vel = wp.to_torch(articulation_data.body_com_ang_vel_w)

        assert body_com_lin_vel.shape == (num_instances, num_bodies, 3)
        assert body_com_ang_vel.shape == (num_instances, num_bodies, 3)
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

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_body_com_pos_and_quat_b(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that body_com_pos_b and body_com_quat_b have correct types, shapes, and values."""
        articulation_data, mock_view = self._setup_method(num_instances, num_bodies, device)

        # --- Test body_com_pos_b ---
        # Check the type and shape
        assert articulation_data.body_com_pos_b.shape == (num_instances, num_bodies)
        assert articulation_data.body_com_pos_b.dtype == wp.vec3f

        # Mock data is initialized to zeros
        expected_pos = torch.zeros((num_instances, num_bodies, 3), device=device)
        assert torch.all(wp.to_torch(articulation_data.body_com_pos_b) == expected_pos)

        # Update with random COM positions
        com_pos = torch.rand((num_instances, num_bodies, 3), device=device)
        mock_view.set_mock_data(
            body_com_pos=wp.from_torch(com_pos, dtype=wp.vec3f),
        )

        # Check that the property returns the mock data
        assert torch.allclose(wp.to_torch(articulation_data.body_com_pos_b), com_pos, atol=1e-6)

        # Verify reference behavior
        body_com_pos_ref = articulation_data.body_com_pos_b
        articulation_data.body_com_pos_b.fill_(1.0)
        expected_ones = torch.ones((num_instances, num_bodies, 3), device=device)
        assert torch.all(wp.to_torch(articulation_data.body_com_pos_b) == expected_ones)
        body_com_pos_ref.fill_(2.0)
        expected_twos = torch.ones((num_instances, num_bodies, 3), device=device) * 2.0
        assert torch.all(wp.to_torch(articulation_data.body_com_pos_b) == expected_twos)

        # --- Test body_com_quat_b ---
        # Check the type and shape
        assert articulation_data.body_com_quat_b.shape == (num_instances, num_bodies)
        assert articulation_data.body_com_quat_b.dtype == wp.quatf

        # body_com_quat_b is derived from body_com_pose_b which uses identity quaternion
        # body_com_pose_b = [body_com_pos_b, identity_quat]
        # So body_com_quat_b should be identity quaternion (0, 0, 0, 1)
        body_com_quat = wp.to_torch(articulation_data.body_com_quat_b)
        expected_quat = torch.zeros((num_instances, num_bodies, 4), device=device)
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

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_default_root_state(self, mock_newton_manager, num_instances: int, device: str):
        """Test that default_root_state correctly combines pose and velocity."""
        articulation_data, _ = self._setup_method(num_instances, device)

        # Check the type and shape
        assert articulation_data.default_root_state.shape == (num_instances,)

        # Get the combined state
        default_state = wp.to_torch(articulation_data.default_root_state)
        assert default_state.shape == (num_instances, 13)

        # Get the individual components
        default_pose = wp.to_torch(articulation_data.default_root_pose)
        default_vel = wp.to_torch(articulation_data.default_root_vel)

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
        articulation_data.default_root_pose.assign(wp.from_torch(new_pose, dtype=wp.transformf))
        articulation_data.default_root_vel.assign(wp.from_torch(new_vel, dtype=wp.spatial_vectorf))

        # Verify the state reflects the new values
        updated_state = wp.to_torch(articulation_data.default_root_state)
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

    def _setup_method(self, num_instances: int, device: str) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_deprecated_root_properties(self, mock_newton_manager, num_instances: int, device: str):
        """Test that all deprecated root properties match their replacements."""
        articulation_data, mock_view = self._setup_method(num_instances, device)

        # Set up random mock data to ensure non-trivial values
        articulation_data._sim_timestamp = 1.0

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
            wp.to_torch(articulation_data.root_pose_w),
            wp.to_torch(articulation_data.root_link_pose_w),
            atol=1e-6,
        )

        # --- Test root_pos_w -> root_link_pos_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.root_pos_w),
            wp.to_torch(articulation_data.root_link_pos_w),
            atol=1e-6,
        )

        # --- Test root_quat_w -> root_link_quat_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.root_quat_w),
            wp.to_torch(articulation_data.root_link_quat_w),
            atol=1e-6,
        )

        # --- Test root_vel_w -> root_com_vel_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.root_vel_w),
            wp.to_torch(articulation_data.root_com_vel_w),
            atol=1e-6,
        )

        # --- Test root_lin_vel_w -> root_com_lin_vel_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.root_lin_vel_w),
            wp.to_torch(articulation_data.root_com_lin_vel_w),
            atol=1e-6,
        )

        # --- Test root_ang_vel_w -> root_com_ang_vel_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.root_ang_vel_w),
            wp.to_torch(articulation_data.root_com_ang_vel_w),
            atol=1e-6,
        )

        # --- Test root_lin_vel_b -> root_com_lin_vel_b ---
        assert torch.allclose(
            wp.to_torch(articulation_data.root_lin_vel_b),
            wp.to_torch(articulation_data.root_com_lin_vel_b),
            atol=1e-6,
        )

        # --- Test root_ang_vel_b -> root_com_ang_vel_b ---
        assert torch.allclose(
            wp.to_torch(articulation_data.root_ang_vel_b),
            wp.to_torch(articulation_data.root_com_ang_vel_b),
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

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_all_deprecated_body_properties(
        self, mock_newton_manager, num_instances: int, num_bodies: int, device: str
    ):
        """Test that all deprecated body properties match their replacements."""
        articulation_data, mock_view = self._setup_method(num_instances, num_bodies, device)

        # Set up random mock data to ensure non-trivial values
        articulation_data._sim_timestamp = 1.0

        body_pose = torch.zeros((num_instances, num_bodies, 7), device=device)
        body_pose[..., :3] = torch.rand((num_instances, num_bodies, 3), device=device)
        body_pose[..., 3:] = torch.randn((num_instances, num_bodies, 4), device=device)
        body_pose[..., 3:] = torch.nn.functional.normalize(body_pose[..., 3:], p=2.0, dim=-1, eps=1e-12)

        body_vel = torch.rand((num_instances, num_bodies, 6), device=device)
        body_com_pos = torch.rand((num_instances, num_bodies, 3), device=device)

        mock_view.set_mock_data(
            link_transforms=wp.from_torch(body_pose, dtype=wp.transformf),
            link_velocities=wp.from_torch(body_vel, dtype=wp.spatial_vectorf),
            body_com_pos=wp.from_torch(body_com_pos, dtype=wp.vec3f),
        )

        # --- Test body_pose_w -> body_link_pose_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.body_pose_w),
            wp.to_torch(articulation_data.body_link_pose_w),
            atol=1e-6,
        )

        # --- Test body_pos_w -> body_link_pos_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.body_pos_w),
            wp.to_torch(articulation_data.body_link_pos_w),
            atol=1e-6,
        )

        # --- Test body_quat_w -> body_link_quat_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.body_quat_w),
            wp.to_torch(articulation_data.body_link_quat_w),
            atol=1e-6,
        )

        # --- Test body_vel_w -> body_com_vel_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.body_vel_w),
            wp.to_torch(articulation_data.body_com_vel_w),
            atol=1e-6,
        )

        # --- Test body_lin_vel_w -> body_com_lin_vel_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.body_lin_vel_w),
            wp.to_torch(articulation_data.body_com_lin_vel_w),
            atol=1e-6,
        )

        # --- Test body_ang_vel_w -> body_com_ang_vel_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.body_ang_vel_w),
            wp.to_torch(articulation_data.body_com_ang_vel_w),
            atol=1e-6,
        )

        # --- Test body_acc_w -> body_com_acc_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.body_acc_w),
            wp.to_torch(articulation_data.body_com_acc_w),
            atol=1e-6,
        )

        # --- Test body_lin_acc_w -> body_com_lin_acc_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.body_lin_acc_w),
            wp.to_torch(articulation_data.body_com_lin_acc_w),
            atol=1e-6,
        )

        # --- Test body_ang_acc_w -> body_com_ang_acc_w ---
        assert torch.allclose(
            wp.to_torch(articulation_data.body_ang_acc_w),
            wp.to_torch(articulation_data.body_com_ang_acc_w),
            atol=1e-6,
        )


class TestDeprecatedComProperties:
    """Tests the deprecated COM pose properties.

    Tests the following deprecated -> new property mappings:
    - com_pos_b -> body_com_pos_b
    - com_quat_b -> body_com_quat_b
    """

    def _setup_method(
        self, num_instances: int, num_bodies: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, num_bodies, 1, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_bodies", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_deprecated_com_properties(self, mock_newton_manager, num_instances: int, num_bodies: int, device: str):
        """Test that deprecated COM properties match their replacements."""
        articulation_data, mock_view = self._setup_method(num_instances, num_bodies, device)

        # Set up random mock data
        com_pos = torch.rand((num_instances, num_bodies, 3), device=device)
        mock_view.set_mock_data(
            body_com_pos=wp.from_torch(com_pos, dtype=wp.vec3f),
        )

        # --- Test com_pos_b -> body_com_pos_b ---
        assert torch.allclose(
            wp.to_torch(articulation_data.com_pos_b),
            wp.to_torch(articulation_data.body_com_pos_b),
            atol=1e-6,
        )

        # --- Test com_quat_b -> body_com_quat_b ---
        assert torch.allclose(
            wp.to_torch(articulation_data.com_quat_b),
            wp.to_torch(articulation_data.body_com_quat_b),
            atol=1e-6,
        )


class TestDeprecatedJointMiscProperties:
    """Tests the deprecated joint and misc properties.

    Tests the following deprecated -> new property mappings:
    - joint_limits -> joint_pos_limits
    - joint_friction -> joint_friction_coeff
    - applied_torque -> applied_effort
    - computed_torque -> computed_effort
    - joint_dynamic_friction -> joint_dynamic_friction_coeff
    - joint_effort_target -> actuator_effort_target
    - joint_viscous_friction -> joint_viscous_friction_coeff
    - joint_velocity_limits -> joint_vel_limits

    Note: fixed_tendon_limit -> fixed_tendon_pos_limits is tested separately
    as it raises NotImplementedError.
    """

    def _setup_method(
        self, num_instances: int, num_joints: int, device: str
    ) -> tuple[ArticulationData, MockNewtonArticulationView]:
        mock_view = MockNewtonArticulationView(num_instances, 1, num_joints, device)
        mock_view.set_mock_data()

        articulation_data = ArticulationData(
            mock_view,
            device,
        )
        return articulation_data, mock_view

    @pytest.mark.parametrize("num_instances", [1, 2])
    @pytest.mark.parametrize("num_joints", [1, 6])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_deprecated_joint_properties(self, mock_newton_manager, num_instances: int, num_joints: int, device: str):
        """Test that deprecated joint properties match their replacements."""
        articulation_data, _ = self._setup_method(num_instances, num_joints, device)

        # --- Test joint_limits -> joint_pos_limits ---
        assert torch.allclose(
            wp.to_torch(articulation_data.joint_limits),
            wp.to_torch(articulation_data.joint_pos_limits),
            atol=1e-6,
        )

        # --- Test joint_friction -> joint_friction_coeff ---
        assert torch.allclose(
            wp.to_torch(articulation_data.joint_friction),
            wp.to_torch(articulation_data.joint_friction_coeff),
            atol=1e-6,
        )

        # --- Test applied_torque -> applied_effort ---
        assert torch.allclose(
            wp.to_torch(articulation_data.applied_torque),
            wp.to_torch(articulation_data.applied_effort),
            atol=1e-6,
        )

        # --- Test computed_torque -> computed_effort ---
        assert torch.allclose(
            wp.to_torch(articulation_data.computed_torque),
            wp.to_torch(articulation_data.computed_effort),
            atol=1e-6,
        )

        # --- Test joint_dynamic_friction -> joint_dynamic_friction_coeff ---
        assert torch.allclose(
            wp.to_torch(articulation_data.joint_dynamic_friction),
            wp.to_torch(articulation_data.joint_dynamic_friction_coeff),
            atol=1e-6,
        )

        # --- Test joint_effort_target -> actuator_effort_target ---
        assert torch.allclose(
            wp.to_torch(articulation_data.joint_effort_target),
            wp.to_torch(articulation_data.actuator_effort_target),
            atol=1e-6,
        )

        # --- Test joint_viscous_friction -> joint_viscous_friction_coeff ---
        assert torch.allclose(
            wp.to_torch(articulation_data.joint_viscous_friction),
            wp.to_torch(articulation_data.joint_viscous_friction_coeff),
            atol=1e-6,
        )

        # --- Test joint_velocity_limits -> joint_vel_limits ---
        assert torch.allclose(
            wp.to_torch(articulation_data.joint_velocity_limits),
            wp.to_torch(articulation_data.joint_vel_limits),
            atol=1e-6,
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_fixed_tendon_limit_not_implemented(self, mock_newton_manager, device: str):
        """Test that fixed_tendon_limit raises NotImplementedError (same as fixed_tendon_pos_limits)."""
        articulation_data, _ = self._setup_method(1, 1, device)

        with pytest.raises(NotImplementedError):
            _ = articulation_data.fixed_tendon_limit


##
# Main
##

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
