# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for mock sensor interfaces."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import pytest
import torch

from isaaclab.test.mock_interfaces.sensors import (
    MockContactSensor,
    MockFrameTransformer,
    MockImu,
    create_mock_foot_contact_sensor,
    create_mock_frame_transformer,
    create_mock_imu,
)
from isaaclab.test.mock_interfaces.utils import MockSensorBuilder

# ==============================================================================
# MockImu Tests
# ==============================================================================


class TestMockImu:
    """Tests for MockImu and MockImuData."""

    @pytest.fixture
    def imu(self):
        """Create a mock IMU fixture."""
        return MockImu(num_instances=4, device="cpu")

    def test_initialization(self, imu):
        """Test that MockImu initializes correctly."""
        assert imu.num_instances == 4
        assert imu.device == "cpu"
        assert imu.data is not None

    def test_lazy_tensor_initialization(self, imu):
        """Test that unset properties return zero tensors with correct shapes."""
        import warp as wp

        # Position
        pos = wp.to_torch(imu.data.pos_w)
        assert pos.shape == (4, 3)
        assert torch.all(pos == 0)

        # Quaternion (should be identity in XYZW format: x=0, y=0, z=0, w=1)
        quat = wp.to_torch(imu.data.quat_w)
        assert quat.shape == (4, 4)
        assert torch.all(quat[:, :3] == 0)  # xyz components
        assert torch.all(quat[:, 3] == 1)  # w component

        # Velocities and accelerations
        assert imu.data.lin_vel_b.shape == (4, 3)
        assert imu.data.ang_vel_b.shape == (4, 3)
        assert imu.data.lin_acc_b.shape == (4, 3)
        assert imu.data.ang_acc_b.shape == (4, 3)

    def test_projected_gravity_default(self, imu):
        """Test default gravity direction."""
        import warp as wp

        gravity = wp.to_torch(imu.data.projected_gravity_b)
        assert gravity.shape == (4, 3)
        # Default gravity should point down: (0, 0, -1)
        assert torch.all(gravity[:, 2] == -1)

    def test_set_mock_data(self, imu):
        """Test bulk data setter."""
        import warp as wp

        lin_vel = torch.randn(4, 3)
        ang_vel = torch.randn(4, 3)

        imu.data.set_mock_data(lin_vel_b=lin_vel, ang_vel_b=ang_vel)

        assert torch.allclose(wp.to_torch(imu.data.lin_vel_b), lin_vel)
        assert torch.allclose(wp.to_torch(imu.data.ang_vel_b), ang_vel)

    def test_per_property_setter(self, imu):
        """Test individual property setters."""
        import warp as wp

        lin_acc = torch.randn(4, 3)
        imu.data.set_lin_acc_b(lin_acc)
        assert torch.allclose(wp.to_torch(imu.data.lin_acc_b), lin_acc)

    def test_pose_composition(self, imu):
        """Test that pose_w combines pos_w and quat_w correctly."""
        import warp as wp

        pos = torch.randn(4, 3)
        quat = torch.tensor([[0, 0, 0, 1]] * 4, dtype=torch.float32)  # XYZW format

        imu.data.set_pos_w(pos)
        imu.data.set_quat_w(quat)

        pose = wp.to_torch(imu.data.pose_w)
        assert pose.shape == (4, 7)
        assert torch.allclose(pose[:, :3], pos)
        assert torch.allclose(pose[:, 3:], quat)


# ==============================================================================
# MockContactSensor Tests
# ==============================================================================


class TestMockContactSensor:
    """Tests for MockContactSensor and MockContactSensorData."""

    @pytest.fixture
    def sensor(self):
        """Create a mock contact sensor fixture."""
        return MockContactSensor(
            num_instances=4,
            num_bodies=4,
            body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            device="cpu",
        )

    def test_initialization(self, sensor):
        """Test that MockContactSensor initializes correctly."""
        assert sensor.num_instances == 4
        assert sensor.num_bodies == 4
        assert sensor.body_names == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        assert sensor.contact_view is None

    def test_lazy_tensor_shapes(self, sensor):
        """Test that unset properties return tensors with correct shapes."""
        forces = sensor.data.net_forces_w
        assert forces.shape == (4, 4, 3)

        contact_time = sensor.data.current_contact_time
        assert contact_time.shape == (4, 4)

        air_time = sensor.data.current_air_time
        assert air_time.shape == (4, 4)

    def test_find_bodies(self, sensor):
        """Test body finding by regex."""
        # Find all bodies matching ".*_foot"
        indices, names = sensor.find_bodies(".*_foot")
        assert len(indices) == 4
        assert set(names) == {"FL_foot", "FR_foot", "RL_foot", "RR_foot"}

        # Find specific body
        indices, names = sensor.find_bodies("FL_foot")
        assert indices == [0]
        assert names == ["FL_foot"]

        # Find front feet
        indices, names = sensor.find_bodies("F._foot")
        assert len(indices) == 2
        assert "FL_foot" in names
        assert "FR_foot" in names

    def test_compute_first_contact(self, sensor):
        """Test first contact computation."""
        # Set contact time to 0.5 for all bodies
        sensor.data.set_current_contact_time(torch.full((4, 4), 0.5))

        # Check with dt=1.0 - should be True (0.5 < 1.0)
        first_contact = sensor.compute_first_contact(dt=1.0)
        assert torch.all(first_contact)

        # Check with dt=0.1 - should be False (0.5 > 0.1)
        first_contact = sensor.compute_first_contact(dt=0.1)
        assert torch.all(~first_contact)

    def test_compute_first_air(self, sensor):
        """Test first air computation."""
        sensor.data.set_current_air_time(torch.full((4, 4), 0.2))

        first_air = sensor.compute_first_air(dt=0.5)
        assert torch.all(first_air)

    def test_history_buffer(self):
        """Test history buffer when enabled."""
        sensor_with_history = MockContactSensor(
            num_instances=2,
            num_bodies=2,
            history_length=3,
            device="cpu",
        )

        history = sensor_with_history.data.net_forces_w_history
        assert history is not None
        assert history.shape == (2, 3, 2, 3)

    def test_no_history_buffer(self, sensor):
        """Test that history buffer is None when not enabled."""
        history = sensor.data.net_forces_w_history
        assert history is None


# ==============================================================================
# MockFrameTransformer Tests
# ==============================================================================


class TestMockFrameTransformer:
    """Tests for MockFrameTransformer and MockFrameTransformerData."""

    @pytest.fixture
    def transformer(self):
        """Create a mock frame transformer fixture."""
        return MockFrameTransformer(
            num_instances=2,
            num_target_frames=3,
            target_frame_names=["target_1", "target_2", "target_3"],
            device="cpu",
        )

    def test_initialization(self, transformer):
        """Test that MockFrameTransformer initializes correctly."""
        assert transformer.num_instances == 2
        assert transformer.num_bodies == 3
        assert transformer.body_names == ["target_1", "target_2", "target_3"]

    def test_data_shapes(self, transformer):
        """Test that data properties have correct shapes."""
        # Source frame
        assert transformer.data.source_pos_w.shape == (2, 3)
        assert transformer.data.source_quat_w.shape == (2, 4)

        # Target frames
        assert transformer.data.target_pos_w.shape == (2, 3, 3)
        assert transformer.data.target_quat_w.shape == (2, 3, 4)

        # Relative poses
        assert transformer.data.target_pos_source.shape == (2, 3, 3)
        assert transformer.data.target_quat_source.shape == (2, 3, 4)

    def test_pose_composition(self, transformer):
        """Test that pose properties combine position and orientation correctly."""
        source_pose = transformer.data.source_pose_w
        assert source_pose.shape == (2, 7)

        target_pose = transformer.data.target_pose_w
        assert target_pose.shape == (2, 3, 7)

    def test_find_bodies(self, transformer):
        """Test frame finding by regex."""
        indices, names = transformer.find_bodies("target_.*")
        assert len(indices) == 3

        indices, names = transformer.find_bodies("target_1")
        assert indices == [0]


# ==============================================================================
# Factory Function Tests
# ==============================================================================


class TestSensorFactories:
    """Tests for sensor factory functions."""

    def test_create_mock_imu(self):
        """Test IMU factory function."""
        imu = create_mock_imu(num_instances=4)
        assert imu.num_instances == 4
        assert imu.data.projected_gravity_b.shape == (4, 3)

    def test_create_mock_foot_contact_sensor(self):
        """Test foot contact sensor factory function."""
        sensor = create_mock_foot_contact_sensor(num_instances=4, num_feet=4)
        assert sensor.num_instances == 4
        assert sensor.num_bodies == 4
        assert "FL_foot" in sensor.body_names

    def test_create_mock_frame_transformer(self):
        """Test frame transformer factory function."""
        transformer = create_mock_frame_transformer(num_instances=2, num_target_frames=5)
        assert transformer.num_instances == 2
        assert transformer.num_bodies == 5


# ==============================================================================
# MockSensorBuilder Tests
# ==============================================================================


class TestMockSensorBuilder:
    """Tests for MockSensorBuilder."""

    def test_build_contact_sensor(self):
        """Test building a contact sensor."""
        sensor = (
            MockSensorBuilder("contact")
            .with_num_instances(4)
            .with_bodies(["foot_1", "foot_2"])
            .with_history_length(3)
            .build()
        )

        assert sensor.num_instances == 4
        assert sensor.num_bodies == 2
        assert sensor.data.net_forces_w_history.shape == (4, 3, 2, 3)

    def test_build_imu_sensor(self):
        """Test building an IMU sensor."""
        sensor = MockSensorBuilder("imu").with_num_instances(2).build()
        assert sensor.num_instances == 2

    def test_build_frame_transformer(self):
        """Test building a frame transformer sensor."""
        sensor = (
            MockSensorBuilder("frame_transformer")
            .with_num_instances(3)
            .with_target_frames(["target_1", "target_2"])
            .build()
        )

        assert sensor.num_instances == 3
        assert sensor.num_bodies == 2
