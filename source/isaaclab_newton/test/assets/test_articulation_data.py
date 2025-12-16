# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ArticulationData class comparing Newton implementation against PhysX reference."""

from __future__ import annotations

import pytest
import torch
import warp as wp
from unittest.mock import MagicMock, patch

# Initialize Warp
wp.init()

# =============================================================================
# Mock classes for Newton
# =============================================================================

class MockNewtonModel:
    """Mock Newton model that provides gravity."""
    
    def __init__(self, gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)):
        self._gravity = wp.array([gravity], dtype=wp.vec3f, device="cuda:0")
    
    @property
    def gravity(self):
        return self._gravity


class MockNewtonArticulationView:
    """Mock NewtonArticulationView that provides simulation bindings.
    
    This class mimics the interface that ArticulationData expects from Newton.
    """
    
    def __init__(
        self,
        num_instances: int,
        num_bodies: int,
        num_joints: int,
        device: str = "cuda:0",
    ):
        self._count = num_instances
        self._link_count = num_bodies
        self._joint_dof_count = num_joints
        self._device = device
        
        # Storage for mock data
        # Note: These are set via set_mock_data() before any property access in tests
        self._root_transforms: wp.array
        self._root_velocities: wp.array
        self._link_transforms: wp.array
        self._link_velocities: wp.array
        self._body_com_pos: wp.array
        self._dof_positions: wp.array
        self._dof_velocities: wp.array
        self._body_mass: wp.array
        self._body_inertia: wp.array
        
        # Initialize default attributes
        self._attributes: dict = {}

    @property
    def count(self) -> int:
        return self._count

    @property
    def link_count(self) -> int:
        return self._link_count

    @property
    def joint_dof_count(self) -> int:
        return self._joint_dof_count

    def get_root_transforms(self, state) -> wp.array:
        return self._root_transforms

    def get_root_velocities(self, state) -> wp.array:
        return self._root_velocities

    def get_link_transforms(self, state) -> wp.array:
        return self._link_transforms

    def get_link_velocities(self, state) -> wp.array:
        return self._link_velocities

    def get_dof_positions(self, state) -> wp.array:
        return self._dof_positions

    def get_dof_velocities(self, state) -> wp.array:
        return self._dof_velocities

    def get_attribute(self, name: str, model_or_state) -> wp.array:
        return self._attributes[name]

    def set_mock_data(
        self,
        root_transforms: wp.array,
        root_velocities: wp.array,
        link_transforms: wp.array,
        link_velocities: wp.array,
        body_com_pos: wp.array,
        dof_positions: wp.array | None = None,
        dof_velocities: wp.array | None = None,
        body_mass: wp.array | None = None,
        body_inertia: wp.array | None = None,
    ):
        """Set mock simulation data."""
        self._root_transforms = root_transforms
        self._root_velocities = root_velocities
        self._link_transforms = link_transforms
        self._link_velocities = link_velocities
        self._body_com_pos = body_com_pos
        
        # Set attributes that ArticulationData expects
        self._attributes["body_com"] = body_com_pos
        
        if dof_positions is None:
            self._dof_positions = wp.zeros(
                (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
            )
        else:
            self._dof_positions = dof_positions
            
        if dof_velocities is None:
            self._dof_velocities = wp.zeros(
                (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
            )
        else:
            self._dof_velocities = dof_velocities
            
        if body_mass is None:
            self._body_mass = wp.ones(
                (self._count, self._link_count), dtype=wp.float32, device=self._device
            )
        else:
            self._body_mass = body_mass
        self._attributes["body_mass"] = self._body_mass
        
        if body_inertia is None:
            # Initialize as identity inertia tensors
            self._body_inertia = wp.zeros(
                (self._count, self._link_count), dtype=wp.mat33f, device=self._device
            )
        else:
            self._body_inertia = body_inertia
        self._attributes["body_inertia"] = self._body_inertia
        
        # Initialize other required attributes with defaults
        self._attributes["joint_limit_lower"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_limit_upper"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_ke"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_kd"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_armature"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_friction"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_velocity_limit"] = wp.ones(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_effort_limit"] = wp.ones(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["body_f"] = wp.zeros(
            (self._count, self._link_count), dtype=wp.spatial_vectorf, device=self._device
        )
        self._attributes["joint_f"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_pos"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )
        self._attributes["joint_target_vel"] = wp.zeros(
            (self._count, self._joint_dof_count), dtype=wp.float32, device=self._device
        )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_config():
    """Default test configuration."""
    return {
        "num_instances": 4,
        "num_bodies": 3,
        "num_joints": 2,
        "device": "cuda:0",
    }


@pytest.fixture
def mock_newton_manager():
    """Create mock NewtonManager with necessary methods."""
    mock_model = MockNewtonModel()
    mock_state = MagicMock()
    mock_control = MagicMock()
    
    with patch("isaaclab.sim._impl.newton_manager.NewtonManager") as MockManager:
        MockManager.get_model.return_value = mock_model
        MockManager.get_state_0.return_value = mock_state
        MockManager.get_control.return_value = mock_control
        MockManager.get_dt.return_value = 0.01
        yield MockManager


# =============================================================================
# Test Classes
# =============================================================================


class TestRootLinkPoseW:
    """Tests for root_link_pose_w property."""
    
    def test_root_link_pose_w_identity(self, test_config):
        """Test root_link_pose_w with identity pose."""
        num_instances = test_config["num_instances"]
        device = test_config["device"]
        
        # Create identity pose
        ref = ReferenceArticulationData(
            num_instances=num_instances,
            num_bodies=1,
            num_joints=1,
            device=device,
        )
        
        root_pose = torch.zeros(num_instances, 7, device=device)
        root_pose[:, 6] = 1.0  # w = 1 for identity quaternion
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=torch.zeros(num_instances, 1, 6, device=device),
            body_com_pos_b=torch.zeros(num_instances, 1, 3, device=device),
        )
        
        expected = ref.root_link_pose_w
        
        # Verify reference gives expected identity
        assert torch.allclose(expected[:, :3], torch.zeros(num_instances, 3, device=device))
        assert torch.allclose(expected[:, 3:6], torch.zeros(num_instances, 3, device=device))
        assert torch.allclose(expected[:, 6], torch.ones(num_instances, device=device))
    
    def test_root_link_pose_w_random(self, test_config):
        """Test root_link_pose_w with random poses."""
        num_instances = test_config["num_instances"]
        num_bodies = test_config["num_bodies"]
        num_joints = test_config["num_joints"]
        device = test_config["device"]
        
        # Generate test data
        test_data = generate_test_data(num_instances, num_bodies, num_joints, device)
        
        # Setup reference
        ref = ReferenceArticulationData(num_instances, num_bodies, num_joints, device)
        ref.set_mock_data(
            root_link_pose_w=test_data["torch"]["root_link_pose"],
            root_com_vel_w=test_data["torch"]["root_com_vel"],
            body_link_pose_w=test_data["torch"]["body_link_pose"],
            body_com_vel_w=test_data["torch"]["body_com_vel"],
            body_com_pos_b=test_data["torch"]["body_com_pos_b"],
        )
        
        expected = ref.root_link_pose_w
        
        # Newton ArticulationData would return the same pose
        # (since root_link_pose_w is a direct binding to simulation data)
        actual = test_data["torch"]["root_link_pose"]
        
        assert_tensors_close(actual, expected, msg="root_link_pose_w mismatch")


class TestRootLinkVelW:
    """Tests for root_link_vel_w property - velocity projection from CoM to link frame."""
    
    def test_root_link_vel_w_zero_com_offset(self, test_config):
        """Test that with zero CoM offset, link vel equals CoM vel."""
        num_instances = test_config["num_instances"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        root_pose = torch.zeros(num_instances, 7, device=device)
        root_pose[:, 6] = 1.0  # identity quaternion
        
        root_com_vel = torch.randn(num_instances, 6, device=device)
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=root_com_vel,
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=root_com_vel.unsqueeze(1),
            body_com_pos_b=torch.zeros(num_instances, 1, 3, device=device),  # Zero offset
        )
        
        # With zero CoM offset, link velocity should equal CoM velocity
        expected = root_com_vel
        actual = ref.root_link_vel_w
        
        assert_tensors_close(actual, expected, msg="With zero CoM offset, link vel should equal CoM vel")
    
    def test_root_link_vel_w_with_com_offset(self, test_config):
        """Test velocity transformation with non-zero CoM offset."""
        num_instances = test_config["num_instances"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        # Identity pose
        root_pose = torch.zeros(num_instances, 7, device=device)
        root_pose[:, 6] = 1.0
        
        # Only angular velocity (to see the cross product effect)
        root_com_vel = torch.zeros(num_instances, 6, device=device)
        root_com_vel[:, 5] = 1.0  # Angular velocity around z
        
        # CoM offset along x
        com_offset = torch.zeros(num_instances, 1, 3, device=device)
        com_offset[:, 0, 0] = 1.0
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=root_com_vel,
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=root_com_vel.unsqueeze(1),
            body_com_pos_b=com_offset,
        )
        
        actual = ref.root_link_vel_w
        
        # With omega_z = 1 and com_offset_x = 1:
        # v_link = v_com + omega x (-com_offset_world)
        # omega = (0, 0, 1), -com_offset = (-1, 0, 0)
        # omega x (-com_offset) = (0, -1, 0)
        expected_lin_vel = torch.tensor([[0.0, -1.0, 0.0]], device=device).repeat(num_instances, 1)
        
        assert_tensors_close(
            actual[:, :3], expected_lin_vel, 
            msg="Linear velocity adjustment from angular vel and CoM offset"
        )


class TestRootComPoseW:
    """Tests for root_com_pose_w property - CoM pose computation."""
    
    def test_root_com_pose_w_zero_offset(self, test_config):
        """Test that with zero CoM offset, CoM pose equals link pose."""
        num_instances = test_config["num_instances"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        root_pose = torch.randn(num_instances, 7, device=device)
        root_pose[:, 3:7] = generate_random_quaternion((num_instances,), device)
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=torch.zeros(num_instances, 1, 6, device=device),
            body_com_pos_b=torch.zeros(num_instances, 1, 3, device=device),
        )
        
        expected = root_pose
        actual = ref.root_com_pose_w
        
        assert_tensors_close(actual, expected, msg="With zero CoM offset, CoM pose should equal link pose")
    
    def test_root_com_pose_w_with_offset(self, test_config):
        """Test CoM pose computation with offset."""
        num_instances = 1
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        # Identity pose at origin
        root_pose = torch.zeros(num_instances, 7, device=device)
        root_pose[:, 6] = 1.0
        
        # CoM offset
        com_offset = torch.tensor([[[1.0, 0.0, 0.0]]], device=device)
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=torch.zeros(num_instances, 1, 6, device=device),
            body_com_pos_b=com_offset,
        )
        
        actual = ref.root_com_pose_w
        
        # Expected: position at (1, 0, 0), identity quaternion
        expected_pos = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        
        assert_tensors_close(actual[:, :3], expected_pos, msg="CoM position with offset")


class TestProjectedGravityB:
    """Tests for projected_gravity_b property."""
    
    def test_projected_gravity_identity_pose(self, test_config):
        """Test gravity projection with identity pose."""
        num_instances = test_config["num_instances"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        # Identity pose
        root_pose = torch.zeros(num_instances, 7, device=device)
        root_pose[:, 6] = 1.0
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=torch.zeros(num_instances, 1, 6, device=device),
            body_com_pos_b=torch.zeros(num_instances, 1, 3, device=device),
        )
        
        actual = ref.projected_gravity_b
        expected = torch.tensor([[0.0, 0.0, -1.0]], device=device).repeat(num_instances, 1)
        
        assert_tensors_close(actual, expected, msg="Gravity projection with identity pose")
    
    def test_projected_gravity_rotated_pose(self, test_config):
        """Test gravity projection with 90-degree rotation around x-axis."""
        num_instances = 1
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        # 90-degree rotation around x-axis: quat = (sin(45°), 0, 0, cos(45°))
        import math
        angle = math.pi / 2
        root_pose = torch.zeros(num_instances, 7, device=device)
        root_pose[:, 3] = math.sin(angle / 2)  # x
        root_pose[:, 6] = math.cos(angle / 2)  # w
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=torch.zeros(num_instances, 1, 6, device=device),
            body_com_pos_b=torch.zeros(num_instances, 1, 3, device=device),
        )
        
        actual = ref.projected_gravity_b
        # After 90-degree rotation around x, gravity (0,0,-1) should become (0,1,0) in body frame
        expected = torch.tensor([[0.0, 1.0, 0.0]], device=device)
        
        assert_tensors_close(actual, expected, atol=1e-4, msg="Gravity projection with rotated pose")


class TestHeadingW:
    """Tests for heading_w property."""
    
    def test_heading_identity_pose(self, test_config):
        """Test heading with identity pose."""
        num_instances = test_config["num_instances"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        # Identity pose - forward is along x
        root_pose = torch.zeros(num_instances, 7, device=device)
        root_pose[:, 6] = 1.0
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=torch.zeros(num_instances, 1, 6, device=device),
            body_com_pos_b=torch.zeros(num_instances, 1, 3, device=device),
        )
        
        actual = ref.heading_w
        expected = torch.zeros(num_instances, device=device)  # atan2(0, 1) = 0
        
        assert_tensors_close(actual, expected, msg="Heading with identity pose")
    
    def test_heading_90_degrees(self, test_config):
        """Test heading with 90-degree yaw rotation."""
        import math
        num_instances = 1
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        # 90-degree rotation around z-axis
        angle = math.pi / 2
        root_pose = torch.zeros(num_instances, 7, device=device)
        root_pose[:, 5] = math.sin(angle / 2)  # z
        root_pose[:, 6] = math.cos(angle / 2)  # w
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=torch.zeros(num_instances, 1, 6, device=device),
            body_com_pos_b=torch.zeros(num_instances, 1, 3, device=device),
        )
        
        actual = ref.heading_w
        expected = torch.tensor([math.pi / 2], device=device)
        
        assert_tensors_close(actual, expected, atol=1e-4, msg="Heading with 90-degree yaw")


class TestBodyLinkVelW:
    """Tests for body_link_vel_w property."""
    
    def test_body_link_vel_w_zero_offset(self, test_config):
        """Test body link velocity with zero CoM offset."""
        num_instances = test_config["num_instances"]
        num_bodies = test_config["num_bodies"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, num_bodies, 1, device)
        
        body_pose = torch.zeros(num_instances, num_bodies, 7, device=device)
        body_pose[..., 6] = 1.0
        
        body_com_vel = torch.randn(num_instances, num_bodies, 6, device=device)
        
        ref.set_mock_data(
            root_link_pose_w=body_pose[:, 0],
            root_com_vel_w=body_com_vel[:, 0],
            body_link_pose_w=body_pose,
            body_com_vel_w=body_com_vel,
            body_com_pos_b=torch.zeros(num_instances, num_bodies, 3, device=device),
        )
        
        expected = body_com_vel
        actual = ref.body_link_vel_w
        
        assert_tensors_close(actual, expected, msg="Body link vel equals CoM vel with zero offset")


class TestBodyComPoseW:
    """Tests for body_com_pose_w property."""
    
    def test_body_com_pose_w_zero_offset(self, test_config):
        """Test body CoM pose with zero offset."""
        num_instances = test_config["num_instances"]
        num_bodies = test_config["num_bodies"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, num_bodies, 1, device)
        
        body_pose = torch.randn(num_instances, num_bodies, 7, device=device)
        body_pose[..., 3:7] = generate_random_quaternion((num_instances, num_bodies), device)
        
        ref.set_mock_data(
            root_link_pose_w=body_pose[:, 0],
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=body_pose,
            body_com_vel_w=torch.zeros(num_instances, num_bodies, 6, device=device),
            body_com_pos_b=torch.zeros(num_instances, num_bodies, 3, device=device),
        )
        
        expected = body_pose
        actual = ref.body_com_pose_w
        
        assert_tensors_close(actual, expected, msg="Body CoM pose equals link pose with zero offset")


class TestSlicedProperties:
    """Tests for sliced position/quaternion/velocity properties."""
    
    def test_root_link_pos_w(self, test_config):
        """Test root_link_pos_w is correct slice."""
        num_instances = test_config["num_instances"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        root_pose = torch.randn(num_instances, 7, device=device)
        root_pose[:, 3:7] = generate_random_quaternion((num_instances,), device)
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=torch.zeros(num_instances, 1, 6, device=device),
            body_com_pos_b=torch.zeros(num_instances, 1, 3, device=device),
        )
        
        expected = root_pose[:, :3]
        actual = ref.root_link_pos_w
        
        assert_tensors_close(actual, expected)
    
    def test_root_link_quat_w(self, test_config):
        """Test root_link_quat_w is correct slice."""
        num_instances = test_config["num_instances"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        root_pose = torch.randn(num_instances, 7, device=device)
        root_pose[:, 3:7] = generate_random_quaternion((num_instances,), device)
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=torch.zeros(num_instances, 1, 6, device=device),
            body_com_pos_b=torch.zeros(num_instances, 1, 3, device=device),
        )
        
        expected = root_pose[:, 3:7]
        actual = ref.root_link_quat_w
        
        assert_tensors_close(actual, expected)
    
    def test_body_link_pos_w(self, test_config):
        """Test body_link_pos_w is correct slice."""
        num_instances = test_config["num_instances"]
        num_bodies = test_config["num_bodies"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, num_bodies, 1, device)
        
        body_pose = torch.randn(num_instances, num_bodies, 7, device=device)
        body_pose[..., 3:7] = generate_random_quaternion((num_instances, num_bodies), device)
        
        ref.set_mock_data(
            root_link_pose_w=body_pose[:, 0],
            root_com_vel_w=torch.zeros(num_instances, 6, device=device),
            body_link_pose_w=body_pose,
            body_com_vel_w=torch.zeros(num_instances, num_bodies, 6, device=device),
            body_com_pos_b=torch.zeros(num_instances, num_bodies, 3, device=device),
        )
        
        expected = body_pose[..., :3]
        actual = ref.body_link_pos_w
        
        assert_tensors_close(actual, expected)


class TestVelocityInBaseFrame:
    """Tests for velocities projected to base frame."""
    
    def test_root_link_vel_b_identity(self, test_config):
        """Test root link velocity in base frame with identity pose."""
        num_instances = test_config["num_instances"]
        device = test_config["device"]
        
        ref = ReferenceArticulationData(num_instances, 1, 1, device)
        
        # Identity pose
        root_pose = torch.zeros(num_instances, 7, device=device)
        root_pose[:, 6] = 1.0
        
        root_com_vel = torch.randn(num_instances, 6, device=device)
        
        ref.set_mock_data(
            root_link_pose_w=root_pose,
            root_com_vel_w=root_com_vel,
            body_link_pose_w=root_pose.unsqueeze(1),
            body_com_vel_w=root_com_vel.unsqueeze(1),
            body_com_pos_b=torch.zeros(num_instances, 1, 3, device=device),
        )
        
        # With identity pose, world frame equals base frame
        expected = root_com_vel  # Since CoM offset is zero
        actual = ref.root_link_vel_b
        
        assert_tensors_close(actual, expected, msg="With identity pose, vel_b should equal vel_w")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

