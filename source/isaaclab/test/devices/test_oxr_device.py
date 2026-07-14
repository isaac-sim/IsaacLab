# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore private usage of variables warning.
# pyright: reportPrivateUsage=none

from __future__ import annotations

from isaaclab.app import AppLauncher

# Can set this to False to see the GUI for debugging.
HEADLESS = True

# Launch omniverse app.
app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import carb
import omni.usd
from isaacsim.core.prims import XFormPrim

from isaaclab.devices import DeviceBase, OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


class NoOpRetargeter(RetargeterBase):
    """A no-op retargeter that requests hand and head tracking but returns empty tensor."""

    def __init__(self, cfg: RetargeterCfg):
        super().__init__(cfg)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        """Request hand and head tracking to trigger data collection."""
        return [
            RetargeterBase.Requirement.HAND_TRACKING,
            RetargeterBase.Requirement.HEAD_TRACKING,
        ]

    def retarget(self, data):
        """Return empty tensor."""
        return torch.tensor([], device=self._sim_device)


class MotionControllerNoOpRetargeter(NoOpRetargeter):
    """A no-op retargeter that requests only motion-controller data."""

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@configclass
class EmptyManagerCfg:
    """Empty manager."""

    pass


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene."""

    pass


@configclass
class EmptyEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the empty test environment."""

    scene: EmptySceneCfg = EmptySceneCfg(num_envs=1, env_spacing=1.0)
    actions: EmptyManagerCfg = EmptyManagerCfg()
    observations: EmptyManagerCfg = EmptyManagerCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 5
        self.episode_length_s = 30.0
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2


@pytest.fixture
def mock_xrcore(mocker):
    """Set up a mock for XRCore and related classes."""
    # Create mock for XRCore and XRPoseValidityFlags
    xr_core_mock = mocker.MagicMock()
    xr_pose_validity_flags_mock = mocker.MagicMock()

    # Set up the validity flags
    xr_pose_validity_flags_mock.POSITION_VALID = 1
    xr_pose_validity_flags_mock.ORIENTATION_VALID = 2

    # Setup the singleton pattern used by XRCore
    singleton_mock = mocker.MagicMock()
    xr_core_mock.get_singleton.return_value = singleton_mock

    # Setup message bus for teleop commands
    message_bus_mock = mocker.MagicMock()
    singleton_mock.get_message_bus.return_value = message_bus_mock
    message_bus_mock.create_subscription_to_pop_by_type.return_value = mocker.MagicMock()

    # Setup input devices (left hand, right hand, head)
    left_hand_mock = mocker.MagicMock()
    right_hand_mock = mocker.MagicMock()
    head_mock = mocker.MagicMock()

    def get_input_device_mock(device_path):
        device_map = {
            "/user/hand/left": left_hand_mock,
            "/user/hand/right": right_hand_mock,
            "/user/head": head_mock,
        }
        return device_map.get(device_path)

    singleton_mock.get_input_device.side_effect = get_input_device_mock

    # Setup the joint poses for hand tracking
    joint_pose_mock = mocker.MagicMock()
    joint_pose_mock.validity_flags = (
        xr_pose_validity_flags_mock.POSITION_VALID | xr_pose_validity_flags_mock.ORIENTATION_VALID
    )

    pose_matrix_mock = mocker.MagicMock()
    pose_matrix_mock.ExtractTranslation.return_value = [0.1, 0.2, 0.3]

    rotation_quat_mock = mocker.MagicMock()
    rotation_quat_mock.GetImaginary.return_value = [0.1, 0.2, 0.3]
    rotation_quat_mock.GetReal.return_value = 0.9

    pose_matrix_mock.ExtractRotationQuat.return_value = rotation_quat_mock
    joint_pose_mock.pose_matrix = pose_matrix_mock

    controller_pose_desc_mock = mocker.MagicMock()
    controller_pose_desc_mock.validity_flags = (
        xr_pose_validity_flags_mock.POSITION_VALID | xr_pose_validity_flags_mock.ORIENTATION_VALID
    )
    controller_pose_desc_mock.pose_matrix = pose_matrix_mock

    joint_poses = {"palm": joint_pose_mock, "wrist": joint_pose_mock}
    left_hand_mock.get_all_virtual_world_poses.return_value = joint_poses
    right_hand_mock.get_all_virtual_world_poses.return_value = joint_poses
    left_hand_mock.get_virtual_world_pose_desc.return_value = controller_pose_desc_mock
    right_hand_mock.get_virtual_world_pose_desc.return_value = controller_pose_desc_mock
    left_hand_mock.has_input_gesture.return_value = False
    right_hand_mock.has_input_gesture.return_value = False

    head_mock.get_virtual_world_pose.return_value = pose_matrix_mock

    # Patch the modules
    device_mod = importlib.import_module("isaaclab.devices.openxr.openxr_device")
    mocker.patch.object(device_mod, "XRCore", xr_core_mock)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", xr_pose_validity_flags_mock)

    return {
        "XRCore": xr_core_mock,
        "XRPoseValidityFlags": xr_pose_validity_flags_mock,
        "singleton": singleton_mock,
        "message_bus": message_bus_mock,
        "left_hand": left_hand_mock,
        "right_hand": right_hand_mock,
        "head": head_mock,
        "controller_pose_desc": controller_pose_desc_mock,
    }


@pytest.fixture
def empty_env():
    """Fixture to create and cleanup an empty environment."""
    # Create a new stage
    omni.usd.get_context().new_stage()
    # Create environment with config
    env_cfg = EmptyEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    yield env, env_cfg

    # Cleanup
    env.close()


@pytest.mark.isaacsim_ci
def test_xr_anchor(empty_env, mock_xrcore):
    """Test XR anchor creation and configuration."""
    env, env_cfg = empty_env
    env_cfg.xr = XrCfg(anchor_pos=(1, 2, 3), anchor_rot=(0, 1, 0, 0))

    device = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=env_cfg.xr))

    # Check that the xr anchor prim is created with the correct pose
    xr_anchor_prim = XFormPrim("/World/XRAnchor")
    assert xr_anchor_prim.is_valid()

    position, orientation = xr_anchor_prim.get_world_poses()
    np.testing.assert_almost_equal(position.tolist(), [[1, 2, 3]])
    np.testing.assert_almost_equal(orientation.tolist(), [[0, 1, 0, 0]])

    # Check that xr anchor mode and custom anchor are set correctly
    assert carb.settings.get_settings().get("/persistent/xr/profile/ar/anchorMode") == "custom anchor"
    assert carb.settings.get_settings().get("/xrstage/profile/ar/customAnchor") == "/World/XRAnchor"

    device.reset()


@pytest.mark.isaacsim_ci
def test_xr_anchor_default(empty_env, mock_xrcore):
    """Test XR anchor creation with default configuration."""
    env, _ = empty_env
    # Create a proper config object with default values
    device = OpenXRDevice(OpenXRDeviceCfg())

    # Check that the xr anchor prim is created with the correct default pose
    xr_anchor_prim = XFormPrim("/World/XRAnchor")
    assert xr_anchor_prim.is_valid()

    position, orientation = xr_anchor_prim.get_world_poses()
    np.testing.assert_almost_equal(position.tolist(), [[0, 0, 0]])
    np.testing.assert_almost_equal(orientation.tolist(), [[1, 0, 0, 0]])

    # Check that xr anchor mode and custom anchor are set correctly
    assert carb.settings.get_settings().get("/persistent/xr/profile/ar/anchorMode") == "custom anchor"
    assert carb.settings.get_settings().get("/xrstage/profile/ar/customAnchor") == "/World/XRAnchor"

    device.reset()


@pytest.mark.isaacsim_ci
def test_xr_anchor_multiple_devices(empty_env, mock_xrcore):
    """Test XR anchor behavior with multiple devices."""
    env, _ = empty_env
    # Create proper config objects with default values
    device_1 = OpenXRDevice(OpenXRDeviceCfg())
    device_2 = OpenXRDevice(OpenXRDeviceCfg())

    # Check that the xr anchor prim is created with the correct default pose
    xr_anchor_prim = XFormPrim("/World/XRAnchor")
    assert xr_anchor_prim.is_valid()

    position, orientation = xr_anchor_prim.get_world_poses()
    np.testing.assert_almost_equal(position.tolist(), [[0, 0, 0]])
    np.testing.assert_almost_equal(orientation.tolist(), [[1, 0, 0, 0]])

    # Check that xr anchor mode and custom anchor are set correctly
    assert carb.settings.get_settings().get("/persistent/xr/profile/ar/anchorMode") == "custom anchor"
    assert carb.settings.get_settings().get("/xrstage/profile/ar/customAnchor") == "/World/XRAnchor"

    device_1.reset()
    device_2.reset()


@pytest.mark.isaacsim_ci
def test_get_raw_data(empty_env, mock_xrcore):
    """Test the _get_raw_data method returns correctly formatted tracking data."""
    env, _ = empty_env
    # Create a proper config object with default values and a no-op retargeter to trigger data collection
    retargeter = NoOpRetargeter(RetargeterCfg())
    device = OpenXRDevice(OpenXRDeviceCfg(), retargeters=[retargeter])

    # Get raw tracking data
    raw_data = device._get_raw_data()

    # Check that the data structure is as expected
    from isaaclab.devices.device_base import DeviceBase

    assert DeviceBase.TrackingTarget.HAND_LEFT in raw_data
    assert DeviceBase.TrackingTarget.HAND_RIGHT in raw_data
    assert DeviceBase.TrackingTarget.HEAD in raw_data

    # Check left hand joints
    left_hand = raw_data[DeviceBase.TrackingTarget.HAND_LEFT]
    assert "palm" in left_hand
    assert "wrist" in left_hand

    # Check that joint pose format is correct
    palm_pose = left_hand["palm"]
    assert len(palm_pose) == 7  # [x, y, z, qw, qx, qy, qz]
    np.testing.assert_almost_equal(palm_pose[:3], [0.1, 0.2, 0.3])  # Position
    np.testing.assert_almost_equal(palm_pose[3:], [0.9, 0.1, 0.2, 0.3])  # Orientation

    # Check head pose
    head_pose = raw_data[DeviceBase.TrackingTarget.HEAD]
    assert len(head_pose) == 7
    np.testing.assert_almost_equal(head_pose[:3], [0.1, 0.2, 0.3])  # Position
    np.testing.assert_almost_equal(head_pose[3:], [0.9, 0.1, 0.2, 0.3])  # Orientation


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    ("validity_flags", "expected_pose_valid"),
    (
        (3, 1.0),
        (1, 0.0),
        (2, 0.0),
        (0, 0.0),
    ),
)
def test_motion_controller_grip_pose_validity(empty_env, mock_xrcore, validity_flags, expected_pose_valid):
    """Controller validity requires both position and orientation flags."""
    env, _ = empty_env
    mock_xrcore["controller_pose_desc"].validity_flags = validity_flags
    device = OpenXRDevice(OpenXRDeviceCfg(), retargeters=[MotionControllerNoOpRetargeter(RetargeterCfg())])

    controller = device._query_controller(mock_xrcore["left_hand"])

    assert controller.shape == (2, 7)
    assert controller.dtype == np.float32
    np.testing.assert_array_equal(controller[0], np.array([0.1, 0.2, 0.3, 0.9, 0.1, 0.2, 0.3], dtype=np.float32))
    assert controller[1, DeviceBase.MotionControllerInputIndex.POSE_VALID.value] == expected_pose_valid
    mock_xrcore["left_hand"].get_virtual_world_pose_desc.assert_called_with("grip")
    assert DeviceBase.MotionControllerInputIndex.PADDING is DeviceBase.MotionControllerInputIndex.POSE_VALID


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("invalid_sample", ("tracking", "degenerate", "nonfinite", "disconnected"))
def test_invalid_controller_sample_is_omitted_and_reported(empty_env, mock_xrcore, invalid_sample):
    """Invalid tracking must not reach retargeters as an actionable sample."""

    env, _ = empty_env
    left = mock_xrcore["left_hand"]
    if invalid_sample == "tracking":
        left.get_virtual_world_pose_desc.return_value = SimpleNamespace(
            validity_flags=1,
            pose_matrix=mock_xrcore["controller_pose_desc"].pose_matrix,
        )
    elif invalid_sample == "degenerate":
        zero_quaternion = SimpleNamespace(GetReal=lambda: 0.0, GetImaginary=lambda: (0.0, 0.0, 0.0))
        left.get_virtual_world_pose_desc.return_value = SimpleNamespace(
            validity_flags=3,
            pose_matrix=SimpleNamespace(
                ExtractTranslation=lambda: (0.1, 0.2, 0.3),
                ExtractRotationQuat=lambda: zero_quaternion,
            ),
        )
    elif invalid_sample == "nonfinite":
        left.get_input_gesture_value.return_value = float("nan")
        left.has_input_gesture.side_effect = lambda gesture, component: (gesture, component) == ("trigger", "value")
    else:
        original_get_input_device = mock_xrcore["singleton"].get_input_device.side_effect
        mock_xrcore["singleton"].get_input_device.side_effect = (
            lambda path: None if path == "/user/hand/left" else original_get_input_device(path)
        )
    device = OpenXRDevice(OpenXRDeviceCfg(), retargeters=[MotionControllerNoOpRetargeter(RetargeterCfg())])

    raw_data = device._get_raw_data()

    assert DeviceBase.TrackingTarget.CONTROLLER_LEFT not in raw_data
    assert DeviceBase.TrackingTarget.CONTROLLER_RIGHT in raw_data
    assert device.controller_faults[DeviceBase.TrackingTarget.CONTROLLER_LEFT] is not None
    assert device.controller_faults[DeviceBase.TrackingTarget.CONTROLLER_RIGHT] is None


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("descriptor_failure", ("missing", "legacy_api"))
def test_missing_controller_descriptor_omits_only_that_side(empty_env, mock_xrcore, descriptor_failure):
    """Unavailable descriptor APIs cannot turn a stale pose into a valid sample."""
    env, _ = empty_env
    left = mock_xrcore["left_hand"]
    if descriptor_failure == "missing":
        left.get_virtual_world_pose_desc.return_value = None
    else:
        left.get_virtual_world_pose_desc.side_effect = AttributeError("legacy mock has only get_virtual_world_pose")
    device = OpenXRDevice(OpenXRDeviceCfg(), retargeters=[MotionControllerNoOpRetargeter(RetargeterCfg())])

    raw_data = device._get_raw_data()

    assert DeviceBase.TrackingTarget.CONTROLLER_LEFT not in raw_data
    assert DeviceBase.TrackingTarget.CONTROLLER_RIGHT in raw_data
    assert raw_data[DeviceBase.TrackingTarget.CONTROLLER_RIGHT].shape == (2, 7)
    assert device.controller_faults[DeviceBase.TrackingTarget.CONTROLLER_LEFT].startswith(
        ("AttributeError:", "TypeError:")
    )
    assert device.controller_faults[DeviceBase.TrackingTarget.CONTROLLER_RIGHT] is None


@pytest.mark.isaacsim_ci
def test_unexpected_motion_controller_errors_propagate(empty_env, mock_xrcore):
    """Do not silently convert an XR API regression into one-sided control."""

    env, _ = empty_env
    mock_xrcore["left_hand"].get_virtual_world_pose_desc.side_effect = RuntimeError("XR runtime failure")
    device = OpenXRDevice(OpenXRDeviceCfg(), retargeters=[MotionControllerNoOpRetargeter(RetargeterCfg())])

    with pytest.raises(RuntimeError, match="XR runtime failure"):
        device._get_raw_data()
