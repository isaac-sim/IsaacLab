# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Comprehensive tests for mock data properties - shapes, setters, and device handling."""

import pytest
import torch

from isaaclab.test.mock_interfaces.assets import (
    MockArticulationData,
    MockRigidObjectCollectionData,
    MockRigidObjectData,
)
from isaaclab.test.mock_interfaces.sensors import (
    MockContactSensorData,
    MockFrameTransformerData,
    MockImuData,
)

# ==============================================================================
# IMU Data Property Tests
# ==============================================================================


class TestMockImuDataProperties:
    """Comprehensive tests for all MockImuData properties."""

    @pytest.fixture
    def data(self):
        """Create MockImuData fixture."""
        return MockImuData(num_instances=4, device="cpu")

    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("pos_w", (4, 3)),
            ("quat_w", (4, 4)),
            ("pose_w", (4, 7)),
            ("projected_gravity_b", (4, 3)),
            ("lin_vel_b", (4, 3)),
            ("ang_vel_b", (4, 3)),
            ("lin_acc_b", (4, 3)),
            ("ang_acc_b", (4, 3)),
        ],
    )
    def test_property_shapes(self, data, property_name, expected_shape):
        """Test that all properties return tensors with correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    @pytest.mark.parametrize(
        "setter_name,property_name,shape",
        [
            ("set_pos_w", "pos_w", (4, 3)),
            ("set_quat_w", "quat_w", (4, 4)),
            ("set_projected_gravity_b", "projected_gravity_b", (4, 3)),
            ("set_lin_vel_b", "lin_vel_b", (4, 3)),
            ("set_ang_vel_b", "ang_vel_b", (4, 3)),
            ("set_lin_acc_b", "lin_acc_b", (4, 3)),
            ("set_ang_acc_b", "ang_acc_b", (4, 3)),
        ],
    )
    def test_setters_update_properties(self, data, setter_name, property_name, shape):
        """Test that setters properly update the corresponding properties."""
        test_value = torch.randn(shape)
        setter = getattr(data, setter_name)
        setter(test_value)
        result = getattr(data, property_name)
        assert torch.allclose(result, test_value), f"{setter_name} did not update {property_name}"

    def test_bulk_setter(self, data):
        """Test that set_mock_data updates multiple properties at once."""
        lin_vel = torch.randn(4, 3)
        ang_vel = torch.randn(4, 3)
        lin_acc = torch.randn(4, 3)

        data.set_mock_data(lin_vel_b=lin_vel, ang_vel_b=ang_vel, lin_acc_b=lin_acc)

        assert torch.allclose(data.lin_vel_b, lin_vel)
        assert torch.allclose(data.ang_vel_b, ang_vel)
        assert torch.allclose(data.lin_acc_b, lin_acc)

    def test_default_quaternion_is_identity(self, data):
        """Test that default quaternion is identity (w=1, x=y=z=0)."""
        quat = data.quat_w
        assert torch.allclose(quat[:, 0], torch.ones(4))  # w=1
        assert torch.allclose(quat[:, 1:], torch.zeros(4, 3))  # xyz=0

    def test_default_gravity_points_down(self, data):
        """Test that default gravity points in -z direction."""
        gravity = data.projected_gravity_b
        expected = torch.tensor([[0, 0, -1]] * 4, dtype=torch.float32)
        assert torch.allclose(gravity, expected)


# ==============================================================================
# Contact Sensor Data Property Tests
# ==============================================================================


class TestMockContactSensorDataProperties:
    """Comprehensive tests for all MockContactSensorData properties."""

    @pytest.fixture
    def data(self):
        """Create MockContactSensorData fixture."""
        return MockContactSensorData(
            num_instances=4,
            num_bodies=3,
            device="cpu",
            history_length=2,
            num_filter_bodies=5,
        )

    @pytest.fixture
    def data_no_history(self):
        """Create MockContactSensorData without history."""
        return MockContactSensorData(
            num_instances=4,
            num_bodies=3,
            device="cpu",
            history_length=0,
            num_filter_bodies=0,
        )

    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("pos_w", (4, 3, 3)),
            ("quat_w", (4, 3, 4)),
            ("pose_w", (4, 3, 7)),
            ("net_forces_w", (4, 3, 3)),
            ("last_air_time", (4, 3)),
            ("current_air_time", (4, 3)),
            ("last_contact_time", (4, 3)),
            ("current_contact_time", (4, 3)),
        ],
    )
    def test_basic_property_shapes(self, data, property_name, expected_shape):
        """Test basic properties return tensors with correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("net_forces_w_history", (4, 2, 3, 3)),  # (N, T, B, 3)
            ("force_matrix_w", (4, 3, 5, 3)),  # (N, B, M, 3)
            ("force_matrix_w_history", (4, 2, 3, 5, 3)),  # (N, T, B, M, 3)
            ("contact_pos_w", (4, 3, 5, 3)),  # (N, B, M, 3)
            ("friction_forces_w", (4, 3, 5, 3)),  # (N, B, M, 3)
        ],
    )
    def test_optional_property_shapes_with_history(self, data, property_name, expected_shape):
        """Test optional properties with history/filter enabled."""
        prop = getattr(data, property_name)
        assert prop is not None
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    def test_optional_properties_none_without_config(self, data_no_history):
        """Test optional properties are None when not configured."""
        assert data_no_history.net_forces_w_history is None
        assert data_no_history.force_matrix_w is None
        assert data_no_history.force_matrix_w_history is None
        assert data_no_history.contact_pos_w is None
        assert data_no_history.friction_forces_w is None

    @pytest.mark.parametrize(
        "setter_name,property_name,shape",
        [
            ("set_pos_w", "pos_w", (4, 3, 3)),
            ("set_quat_w", "quat_w", (4, 3, 4)),
            ("set_net_forces_w", "net_forces_w", (4, 3, 3)),
            ("set_last_air_time", "last_air_time", (4, 3)),
            ("set_current_air_time", "current_air_time", (4, 3)),
            ("set_last_contact_time", "last_contact_time", (4, 3)),
            ("set_current_contact_time", "current_contact_time", (4, 3)),
        ],
    )
    def test_setters_update_properties(self, data, setter_name, property_name, shape):
        """Test that setters properly update the corresponding properties."""
        test_value = torch.randn(shape)
        setter = getattr(data, setter_name)
        setter(test_value)
        result = getattr(data, property_name)
        assert torch.allclose(result, test_value), f"{setter_name} did not update {property_name}"


# ==============================================================================
# Frame Transformer Data Property Tests
# ==============================================================================


class TestMockFrameTransformerDataProperties:
    """Comprehensive tests for all MockFrameTransformerData properties."""

    @pytest.fixture
    def data(self):
        """Create MockFrameTransformerData fixture."""
        return MockFrameTransformerData(
            num_instances=4,
            num_target_frames=3,
            target_frame_names=["frame_0", "frame_1", "frame_2"],
            device="cpu",
        )

    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("source_pos_w", (4, 3)),
            ("source_quat_w", (4, 4)),
            ("source_pose_w", (4, 7)),
            ("target_pos_w", (4, 3, 3)),
            ("target_quat_w", (4, 3, 4)),
            ("target_pose_w", (4, 3, 7)),
            ("target_pos_source", (4, 3, 3)),
            ("target_quat_source", (4, 3, 4)),
            ("target_pose_source", (4, 3, 7)),
        ],
    )
    def test_property_shapes(self, data, property_name, expected_shape):
        """Test that all properties return tensors with correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    @pytest.mark.parametrize(
        "setter_name,property_name,shape",
        [
            ("set_source_pos_w", "source_pos_w", (4, 3)),
            ("set_source_quat_w", "source_quat_w", (4, 4)),
            ("set_target_pos_w", "target_pos_w", (4, 3, 3)),
            ("set_target_quat_w", "target_quat_w", (4, 3, 4)),
            ("set_target_pos_source", "target_pos_source", (4, 3, 3)),
            ("set_target_quat_source", "target_quat_source", (4, 3, 4)),
        ],
    )
    def test_setters_update_properties(self, data, setter_name, property_name, shape):
        """Test that setters properly update the corresponding properties."""
        test_value = torch.randn(shape)
        setter = getattr(data, setter_name)
        setter(test_value)
        result = getattr(data, property_name)
        assert torch.allclose(result, test_value), f"{setter_name} did not update {property_name}"

    def test_target_frame_names(self, data):
        """Test that target frame names are correctly stored."""
        assert data.target_frame_names == ["frame_0", "frame_1", "frame_2"]


# ==============================================================================
# Articulation Data Property Tests
# ==============================================================================


class TestMockArticulationDataProperties:
    """Comprehensive tests for all MockArticulationData properties."""

    @pytest.fixture
    def data(self):
        """Create MockArticulationData fixture."""
        return MockArticulationData(
            num_instances=4,
            num_joints=12,
            num_bodies=13,
            num_fixed_tendons=2,
            num_spatial_tendons=1,
            device="cpu",
        )

    # -- Joint State Properties --
    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("joint_pos", (4, 12)),
            ("joint_vel", (4, 12)),
            ("joint_acc", (4, 12)),
            ("joint_pos_target", (4, 12)),
            ("joint_vel_target", (4, 12)),
            ("joint_effort_target", (4, 12)),
            ("computed_torque", (4, 12)),
            ("applied_torque", (4, 12)),
        ],
    )
    def test_joint_state_shapes(self, data, property_name, expected_shape):
        """Test joint state properties have correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    # -- Joint Property Shapes --
    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("joint_stiffness", (4, 12)),
            ("joint_damping", (4, 12)),
            ("joint_armature", (4, 12)),
            ("joint_friction_coeff", (4, 12)),
            ("joint_dynamic_friction_coeff", (4, 12)),
            ("joint_viscous_friction_coeff", (4, 12)),
            ("joint_pos_limits", (4, 12, 2)),
            ("joint_vel_limits", (4, 12)),
            ("joint_effort_limits", (4, 12)),
            ("soft_joint_pos_limits", (4, 12, 2)),
            ("soft_joint_vel_limits", (4, 12)),
            ("gear_ratio", (4, 12)),
        ],
    )
    def test_joint_property_shapes(self, data, property_name, expected_shape):
        """Test joint property shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    # -- Root State Properties --
    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("root_link_pose_w", (4, 7)),
            ("root_link_vel_w", (4, 6)),
            ("root_link_state_w", (4, 13)),
            ("root_link_pos_w", (4, 3)),
            ("root_link_quat_w", (4, 4)),
            ("root_link_lin_vel_w", (4, 3)),
            ("root_link_ang_vel_w", (4, 3)),
            ("root_com_pose_w", (4, 7)),
            ("root_com_vel_w", (4, 6)),
            ("root_com_state_w", (4, 13)),
            ("root_com_pos_w", (4, 3)),
            ("root_com_quat_w", (4, 4)),
            ("root_com_lin_vel_w", (4, 3)),
            ("root_com_ang_vel_w", (4, 3)),
            ("root_state_w", (4, 13)),
        ],
    )
    def test_root_state_shapes(self, data, property_name, expected_shape):
        """Test root state properties have correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    # -- Body State Properties --
    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("body_link_pose_w", (4, 13, 7)),
            ("body_link_vel_w", (4, 13, 6)),
            ("body_link_state_w", (4, 13, 13)),
            ("body_link_pos_w", (4, 13, 3)),
            ("body_link_quat_w", (4, 13, 4)),
            ("body_link_lin_vel_w", (4, 13, 3)),
            ("body_link_ang_vel_w", (4, 13, 3)),
            ("body_com_pose_w", (4, 13, 7)),
            ("body_com_vel_w", (4, 13, 6)),
            ("body_com_state_w", (4, 13, 13)),
            ("body_com_acc_w", (4, 13, 6)),
            ("body_com_pos_w", (4, 13, 3)),
            ("body_com_quat_w", (4, 13, 4)),
            ("body_com_lin_vel_w", (4, 13, 3)),
            ("body_com_ang_vel_w", (4, 13, 3)),
            ("body_com_lin_acc_w", (4, 13, 3)),
            ("body_com_ang_acc_w", (4, 13, 3)),
        ],
    )
    def test_body_state_shapes(self, data, property_name, expected_shape):
        """Test body state properties have correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    # -- Body Properties --
    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("body_mass", (4, 13)),
            ("body_inertia", (4, 13, 3, 3)),
            ("body_incoming_joint_wrench_b", (4, 13, 6)),
        ],
    )
    def test_body_property_shapes(self, data, property_name, expected_shape):
        """Test body property shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    # -- Default State Properties --
    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("default_root_pose", (4, 7)),
            ("default_root_vel", (4, 6)),
            ("default_root_state", (4, 13)),
            ("default_joint_pos", (4, 12)),
            ("default_joint_vel", (4, 12)),
        ],
    )
    def test_default_state_shapes(self, data, property_name, expected_shape):
        """Test default state properties have correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    # -- Derived Properties --
    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("projected_gravity_b", (4, 3)),
            ("heading_w", (4,)),
            ("root_link_lin_vel_b", (4, 3)),
            ("root_link_ang_vel_b", (4, 3)),
            ("root_com_lin_vel_b", (4, 3)),
            ("root_com_ang_vel_b", (4, 3)),
        ],
    )
    def test_derived_property_shapes(self, data, property_name, expected_shape):
        """Test derived properties have correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    # -- Tendon Properties --
    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("fixed_tendon_stiffness", (4, 2)),
            ("fixed_tendon_damping", (4, 2)),
            ("fixed_tendon_limit_stiffness", (4, 2)),
            ("fixed_tendon_rest_length", (4, 2)),
            ("fixed_tendon_offset", (4, 2)),
            ("fixed_tendon_pos_limits", (4, 2, 2)),
            ("spatial_tendon_stiffness", (4, 1)),
            ("spatial_tendon_damping", (4, 1)),
            ("spatial_tendon_limit_stiffness", (4, 1)),
            ("spatial_tendon_offset", (4, 1)),
        ],
    )
    def test_tendon_property_shapes(self, data, property_name, expected_shape):
        """Test tendon properties have correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"

    # -- Setter Tests --
    @pytest.mark.parametrize(
        "setter_name,property_name,shape",
        [
            ("set_joint_pos", "joint_pos", (4, 12)),
            ("set_joint_vel", "joint_vel", (4, 12)),
            ("set_joint_acc", "joint_acc", (4, 12)),
            ("set_root_link_pose_w", "root_link_pose_w", (4, 7)),
            ("set_root_link_vel_w", "root_link_vel_w", (4, 6)),
            ("set_body_link_pose_w", "body_link_pose_w", (4, 13, 7)),
            ("set_body_link_vel_w", "body_link_vel_w", (4, 13, 6)),
            ("set_body_mass", "body_mass", (4, 13)),
        ],
    )
    def test_setters_update_properties(self, data, setter_name, property_name, shape):
        """Test that setters properly update the corresponding properties."""
        test_value = torch.randn(shape)
        setter = getattr(data, setter_name)
        setter(test_value)
        result = getattr(data, property_name)
        assert torch.allclose(result, test_value), f"{setter_name} did not update {property_name}"

    def test_bulk_setter_with_multiple_properties(self, data):
        """Test set_mock_data with multiple properties."""
        joint_pos = torch.randn(4, 12)
        joint_vel = torch.randn(4, 12)
        root_pose = torch.randn(4, 7)

        data.set_mock_data(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_link_pose_w=root_pose,
        )

        assert torch.allclose(data.joint_pos, joint_pos)
        assert torch.allclose(data.joint_vel, joint_vel)
        assert torch.allclose(data.root_link_pose_w, root_pose)

    def test_bulk_setter_unknown_property_raises(self, data):
        """Test that set_mock_data raises on unknown property."""
        with pytest.raises(ValueError, match="Unknown property"):
            data.set_mock_data(unknown_property=torch.zeros(4))


# ==============================================================================
# Rigid Object Data Property Tests
# ==============================================================================


class TestMockRigidObjectDataProperties:
    """Comprehensive tests for MockRigidObjectData properties."""

    @pytest.fixture
    def data(self):
        """Create MockRigidObjectData fixture."""
        return MockRigidObjectData(num_instances=4, device="cpu")

    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("root_link_pose_w", (4, 7)),
            ("root_link_vel_w", (4, 6)),
            ("root_link_state_w", (4, 13)),
            ("root_link_pos_w", (4, 3)),
            ("root_link_quat_w", (4, 4)),
            ("root_link_lin_vel_w", (4, 3)),
            ("root_link_ang_vel_w", (4, 3)),
            ("root_com_pose_w", (4, 7)),
            ("root_com_vel_w", (4, 6)),
            ("root_com_state_w", (4, 13)),
            ("root_state_w", (4, 13)),
            ("body_link_pose_w", (4, 1, 7)),
            ("body_link_vel_w", (4, 1, 6)),
            ("body_link_state_w", (4, 1, 13)),
            ("body_com_pose_w", (4, 1, 7)),
            ("body_com_vel_w", (4, 1, 6)),
            ("body_com_state_w", (4, 1, 13)),
            ("body_com_acc_w", (4, 1, 6)),
            ("body_mass", (4, 1, 1)),
            ("body_inertia", (4, 1, 9)),
            ("projected_gravity_b", (4, 3)),
            ("heading_w", (4,)),
            ("default_root_pose", (4, 7)),
            ("default_root_vel", (4, 6)),
            ("default_root_state", (4, 13)),
        ],
    )
    def test_property_shapes(self, data, property_name, expected_shape):
        """Test that all properties return tensors with correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"


# ==============================================================================
# Rigid Object Collection Data Property Tests
# ==============================================================================


class TestMockRigidObjectCollectionDataProperties:
    """Comprehensive tests for MockRigidObjectCollectionData properties."""

    @pytest.fixture
    def data(self):
        """Create MockRigidObjectCollectionData fixture."""
        return MockRigidObjectCollectionData(
            num_instances=4,
            num_bodies=5,
            device="cpu",
        )

    @pytest.mark.parametrize(
        "property_name,expected_shape",
        [
            ("body_link_pose_w", (4, 5, 7)),
            ("body_link_vel_w", (4, 5, 6)),
            ("body_link_state_w", (4, 5, 13)),
            ("body_link_pos_w", (4, 5, 3)),
            ("body_link_quat_w", (4, 5, 4)),
            ("body_link_lin_vel_w", (4, 5, 3)),
            ("body_link_ang_vel_w", (4, 5, 3)),
            ("body_link_lin_vel_b", (4, 5, 3)),
            ("body_link_ang_vel_b", (4, 5, 3)),
            ("body_com_pose_w", (4, 5, 7)),
            ("body_com_vel_w", (4, 5, 6)),
            ("body_com_state_w", (4, 5, 13)),
            ("body_com_acc_w", (4, 5, 6)),
            ("body_com_pos_w", (4, 5, 3)),
            ("body_com_quat_w", (4, 5, 4)),
            ("body_com_lin_vel_w", (4, 5, 3)),
            ("body_com_ang_vel_w", (4, 5, 3)),
            ("body_com_lin_vel_b", (4, 5, 3)),
            ("body_com_ang_vel_b", (4, 5, 3)),
            ("body_com_lin_acc_w", (4, 5, 3)),
            ("body_com_ang_acc_w", (4, 5, 3)),
            ("body_com_pose_b", (4, 5, 7)),
            ("body_com_pos_b", (4, 5, 3)),
            ("body_com_quat_b", (4, 5, 4)),
            ("body_mass", (4, 5)),
            ("body_inertia", (4, 5, 9)),
            ("projected_gravity_b", (4, 5, 3)),
            ("heading_w", (4, 5)),
            ("default_body_pose", (4, 5, 7)),
            ("default_body_vel", (4, 5, 6)),
            ("default_body_state", (4, 5, 13)),
        ],
    )
    def test_property_shapes(self, data, property_name, expected_shape):
        """Test that all properties return tensors with correct shapes."""
        prop = getattr(data, property_name)
        assert prop.shape == expected_shape, f"{property_name} has wrong shape"


# ==============================================================================
# Device Handling Tests
# ==============================================================================


class TestDeviceHandling:
    """Test that tensors are created on the correct device."""

    def test_imu_data_device(self):
        """Test IMU data tensors are on correct device."""
        data = MockImuData(num_instances=2, device="cpu")
        assert data.pos_w.device.type == "cpu"
        assert data.quat_w.device.type == "cpu"

    def test_contact_sensor_data_device(self):
        """Test contact sensor data tensors are on correct device."""
        data = MockContactSensorData(num_instances=2, num_bodies=3, device="cpu")
        assert data.net_forces_w.device.type == "cpu"

    def test_articulation_data_device(self):
        """Test articulation data tensors are on correct device."""
        data = MockArticulationData(num_instances=2, num_joints=6, num_bodies=7, device="cpu")
        assert data.joint_pos.device.type == "cpu"
        assert data.root_link_pose_w.device.type == "cpu"

    def test_setter_moves_tensor_to_device(self):
        """Test that setters move tensors to the correct device."""
        data = MockImuData(num_instances=2, device="cpu")
        # Create tensor on CPU (default)
        test_tensor = torch.randn(2, 3)
        data.set_pos_w(test_tensor)
        assert data.pos_w.device.type == "cpu"


# ==============================================================================
# Composite Property Tests
# ==============================================================================


class TestCompositeProperties:
    """Test that composite properties are correctly composed from components."""

    def test_imu_pose_composition(self):
        """Test IMU pose_w is correctly composed from pos_w and quat_w."""
        data = MockImuData(num_instances=2, device="cpu")
        pos = torch.randn(2, 3)
        quat = torch.tensor([[1, 0, 0, 0], [0.707, 0.707, 0, 0]], dtype=torch.float32)

        data.set_pos_w(pos)
        data.set_quat_w(quat)

        pose = data.pose_w
        assert torch.allclose(pose[:, :3], pos)
        assert torch.allclose(pose[:, 3:], quat)

    def test_articulation_root_state_composition(self):
        """Test articulation root_link_state_w is correctly composed."""
        data = MockArticulationData(num_instances=2, num_joints=6, num_bodies=7, device="cpu")
        pose = torch.randn(2, 7)
        vel = torch.randn(2, 6)

        data.set_root_link_pose_w(pose)
        data.set_root_link_vel_w(vel)

        state = data.root_link_state_w
        assert torch.allclose(state[:, :7], pose)
        assert torch.allclose(state[:, 7:], vel)

    def test_articulation_default_state_composition(self):
        """Test articulation default_root_state is correctly composed."""
        data = MockArticulationData(num_instances=2, num_joints=6, num_bodies=7, device="cpu")
        pose = torch.randn(2, 7)
        vel = torch.randn(2, 6)

        data.set_default_root_pose(pose)
        data.set_default_root_vel(vel)

        state = data.default_root_state
        assert torch.allclose(state[:, :7], pose)
        assert torch.allclose(state[:, 7:], vel)


# ==============================================================================
# Convenience Alias Property Tests
# ==============================================================================


class TestArticulationConvenienceAliases:
    """Test convenience alias properties for MockArticulationData."""

    @pytest.fixture
    def data(self):
        """Create MockArticulationData fixture."""
        return MockArticulationData(
            num_instances=4,
            num_joints=12,
            num_bodies=13,
            num_fixed_tendons=2,
            device="cpu",
        )

    # -- Root state aliases (without _link_ or _com_ prefix) --
    @pytest.mark.parametrize(
        "alias,source,expected_shape",
        [
            ("root_pos_w", "root_link_pos_w", (4, 3)),
            ("root_quat_w", "root_link_quat_w", (4, 4)),
            ("root_pose_w", "root_link_pose_w", (4, 7)),
            ("root_vel_w", "root_com_vel_w", (4, 6)),
            ("root_lin_vel_w", "root_com_lin_vel_w", (4, 3)),
            ("root_ang_vel_w", "root_com_ang_vel_w", (4, 3)),
            ("root_lin_vel_b", "root_com_lin_vel_b", (4, 3)),
            ("root_ang_vel_b", "root_com_ang_vel_b", (4, 3)),
        ],
    )
    def test_root_aliases(self, data, alias, source, expected_shape):
        """Test root state aliases reference correct properties."""
        alias_prop = getattr(data, alias)
        source_prop = getattr(data, source)
        assert alias_prop.shape == expected_shape, f"{alias} has wrong shape"
        assert torch.equal(alias_prop, source_prop), f"{alias} should equal {source}"

    # -- Body state aliases (without _link_ or _com_ prefix) --
    @pytest.mark.parametrize(
        "alias,source,expected_shape",
        [
            ("body_pos_w", "body_link_pos_w", (4, 13, 3)),
            ("body_quat_w", "body_link_quat_w", (4, 13, 4)),
            ("body_pose_w", "body_link_pose_w", (4, 13, 7)),
            ("body_vel_w", "body_com_vel_w", (4, 13, 6)),
            ("body_state_w", "body_com_state_w", (4, 13, 13)),
            ("body_lin_vel_w", "body_com_lin_vel_w", (4, 13, 3)),
            ("body_ang_vel_w", "body_com_ang_vel_w", (4, 13, 3)),
            ("body_acc_w", "body_com_acc_w", (4, 13, 6)),
            ("body_lin_acc_w", "body_com_lin_acc_w", (4, 13, 3)),
            ("body_ang_acc_w", "body_com_ang_acc_w", (4, 13, 3)),
        ],
    )
    def test_body_aliases(self, data, alias, source, expected_shape):
        """Test body state aliases reference correct properties."""
        alias_prop = getattr(data, alias)
        source_prop = getattr(data, source)
        assert alias_prop.shape == expected_shape, f"{alias} has wrong shape"
        assert torch.equal(alias_prop, source_prop), f"{alias} should equal {source}"

    # -- CoM in body frame --
    @pytest.mark.parametrize(
        "alias,expected_shape",
        [
            ("com_pos_b", (4, 3)),
            ("com_quat_b", (4, 4)),
        ],
    )
    def test_com_body_frame_aliases(self, data, alias, expected_shape):
        """Test CoM in body frame aliases."""
        prop = getattr(data, alias)
        assert prop.shape == expected_shape, f"{alias} has wrong shape"

    # -- Joint property aliases --
    @pytest.mark.parametrize(
        "alias,source,expected_shape",
        [
            ("joint_limits", "joint_pos_limits", (4, 12, 2)),
            ("joint_velocity_limits", "joint_vel_limits", (4, 12)),
            ("joint_friction", "joint_friction_coeff", (4, 12)),
        ],
    )
    def test_joint_aliases(self, data, alias, source, expected_shape):
        """Test joint property aliases."""
        alias_prop = getattr(data, alias)
        source_prop = getattr(data, source)
        assert alias_prop.shape == expected_shape, f"{alias} has wrong shape"
        assert torch.equal(alias_prop, source_prop), f"{alias} should equal {source}"

    # -- Fixed tendon alias --
    def test_fixed_tendon_limit_alias(self, data):
        """Test fixed_tendon_limit is alias for fixed_tendon_pos_limits."""
        assert data.fixed_tendon_limit.shape == (4, 2, 2)
        assert torch.equal(data.fixed_tendon_limit, data.fixed_tendon_pos_limits)


class TestRigidObjectConvenienceAliases:
    """Test convenience alias properties for MockRigidObjectData."""

    @pytest.fixture
    def data(self):
        """Create MockRigidObjectData fixture."""
        return MockRigidObjectData(num_instances=4, device="cpu")

    @pytest.mark.parametrize(
        "alias,source,expected_shape",
        [
            ("root_pos_w", "root_link_pos_w", (4, 3)),
            ("root_quat_w", "root_link_quat_w", (4, 4)),
            ("root_pose_w", "root_link_pose_w", (4, 7)),
            ("root_vel_w", "root_com_vel_w", (4, 6)),
            ("root_lin_vel_w", "root_com_lin_vel_w", (4, 3)),
            ("root_ang_vel_w", "root_com_ang_vel_w", (4, 3)),
            ("root_lin_vel_b", "root_com_lin_vel_b", (4, 3)),
            ("root_ang_vel_b", "root_com_ang_vel_b", (4, 3)),
            ("body_pos_w", "body_link_pos_w", (4, 1, 3)),
            ("body_quat_w", "body_link_quat_w", (4, 1, 4)),
            ("body_pose_w", "body_link_pose_w", (4, 1, 7)),
            ("body_vel_w", "body_com_vel_w", (4, 1, 6)),
            ("body_state_w", "body_com_state_w", (4, 1, 13)),
            ("body_lin_vel_w", "body_com_lin_vel_w", (4, 1, 3)),
            ("body_ang_vel_w", "body_com_ang_vel_w", (4, 1, 3)),
            ("body_acc_w", "body_com_acc_w", (4, 1, 6)),
            ("body_lin_acc_w", "body_com_lin_acc_w", (4, 1, 3)),
            ("body_ang_acc_w", "body_com_ang_acc_w", (4, 1, 3)),
        ],
    )
    def test_aliases(self, data, alias, source, expected_shape):
        """Test convenience aliases reference correct properties."""
        alias_prop = getattr(data, alias)
        source_prop = getattr(data, source)
        assert alias_prop.shape == expected_shape, f"{alias} has wrong shape"
        assert torch.equal(alias_prop, source_prop), f"{alias} should equal {source}"

    @pytest.mark.parametrize(
        "alias,expected_shape",
        [
            ("com_pos_b", (4, 3)),
            ("com_quat_b", (4, 4)),
        ],
    )
    def test_com_body_frame_aliases(self, data, alias, expected_shape):
        """Test CoM in body frame aliases."""
        prop = getattr(data, alias)
        assert prop.shape == expected_shape, f"{alias} has wrong shape"


class TestRigidObjectCollectionConvenienceAliases:
    """Test convenience alias properties for MockRigidObjectCollectionData."""

    @pytest.fixture
    def data(self):
        """Create MockRigidObjectCollectionData fixture."""
        return MockRigidObjectCollectionData(
            num_instances=4,
            num_bodies=5,
            device="cpu",
        )

    def test_body_state_w_alias(self, data):
        """Test body_state_w is alias for body_com_state_w."""
        assert data.body_state_w.shape == (4, 5, 13)
        assert torch.equal(data.body_state_w, data.body_com_state_w)
