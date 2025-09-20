# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import importlib
import numpy as np
import torch

import pytest

# Import device classes to test
from isaaclab.devices import (
    OpenXRDevice,
    OpenXRDeviceCfg,
    Se2Gamepad,
    Se2GamepadCfg,
    Se2Keyboard,
    Se2KeyboardCfg,
    Se2Phone,
    Se2PhoneCfg,
    Se2SpaceMouse,
    Se2SpaceMouseCfg,
    Se3Gamepad,
    Se3GamepadCfg,
    Se3Keyboard,
    Se3KeyboardCfg,
    Se3Phone,
    Se3PhoneCfg,
    Se3SpaceMouse,
    Se3SpaceMouseCfg,
)
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.openxr.retargeters import GripperRetargeterCfg, Se3AbsRetargeterCfg

# Import teleop device factory for testing
from isaaclab.devices.teleop_device_factory import create_teleop_device


@pytest.fixture
def mock_environment(mocker):
    """Set up common mock objects for tests."""
    # Create mock objects that will be used across tests
    carb_mock = mocker.MagicMock()
    omni_mock = mocker.MagicMock()
    appwindow_mock = mocker.MagicMock()
    keyboard_mock = mocker.MagicMock()
    gamepad_mock = mocker.MagicMock()
    input_mock = mocker.MagicMock()
    settings_mock = mocker.MagicMock()
    hid_mock = mocker.MagicMock()
    device_mock = mocker.MagicMock()

    # Set up the mocks to return appropriate objects
    omni_mock.appwindow.get_default_app_window.return_value = appwindow_mock
    appwindow_mock.get_keyboard.return_value = keyboard_mock
    appwindow_mock.get_gamepad.return_value = gamepad_mock
    carb_mock.input.acquire_input_interface.return_value = input_mock
    carb_mock.settings.get_settings.return_value = settings_mock

    # Mock keyboard event types
    carb_mock.input.KeyboardEventType.KEY_PRESS = 1
    carb_mock.input.KeyboardEventType.KEY_RELEASE = 2

    # Mock the SpaceMouse
    hid_mock.enumerate.return_value = [{"product_string": "SpaceMouse Compact", "vendor_id": 123, "product_id": 456}]
    hid_mock.device.return_value = device_mock

    # Mock OpenXR
    # xr_core_mock = mocker.MagicMock()
    message_bus_mock = mocker.MagicMock()
    singleton_mock = mocker.MagicMock()
    omni_mock.kit.xr.core.XRCore.get_singleton.return_value = singleton_mock
    singleton_mock.get_message_bus.return_value = message_bus_mock
    omni_mock.kit.xr.core.XRPoseValidityFlags.POSITION_VALID = 1
    omni_mock.kit.xr.core.XRPoseValidityFlags.ORIENTATION_VALID = 2

    return {
        "carb": carb_mock,
        "omni": omni_mock,
        "appwindow": appwindow_mock,
        "keyboard": keyboard_mock,
        "gamepad": gamepad_mock,
        "input": input_mock,
        "settings": settings_mock,
        "hid": hid_mock,
        "device": device_mock,
    }


"""
Test keyboard devices.
"""


def test_se2keyboard_constructors(mock_environment, mocker):
    """Test constructor for Se2Keyboard."""
    # Test config-based constructor
    config = Se2KeyboardCfg(
        v_x_sensitivity=0.9,
        v_y_sensitivity=0.5,
        omega_z_sensitivity=1.2,
    )
    device_mod = importlib.import_module("isaaclab.devices.keyboard.se2_keyboard")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    keyboard = Se2Keyboard(config)

    # Verify configuration was applied correctly
    assert keyboard.v_x_sensitivity == 0.9
    assert keyboard.v_y_sensitivity == 0.5
    assert keyboard.omega_z_sensitivity == 1.2

    # Test advance() returns expected type
    result = keyboard.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)  # (v_x, v_y, omega_z)


def test_se3keyboard_constructors(mock_environment, mocker):
    """Test constructor for Se3Keyboard."""
    # Test config-based constructor
    config = Se3KeyboardCfg(
        pos_sensitivity=0.5,
        rot_sensitivity=0.9,
    )
    device_mod = importlib.import_module("isaaclab.devices.keyboard.se3_keyboard")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    keyboard = Se3Keyboard(config)

    # Verify configuration was applied correctly
    assert keyboard.pos_sensitivity == 0.5
    assert keyboard.rot_sensitivity == 0.9

    # Test advance() returns expected type
    result = keyboard.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7,)  # (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper)


"""
Test gamepad devices.
"""


def test_se2gamepad_constructors(mock_environment, mocker):
    """Test constructor for Se2Gamepad."""
    # Test config-based constructor
    config = Se2GamepadCfg(
        v_x_sensitivity=1.1,
        v_y_sensitivity=0.6,
        omega_z_sensitivity=1.2,
        dead_zone=0.02,
    )
    device_mod = importlib.import_module("isaaclab.devices.gamepad.se2_gamepad")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    gamepad = Se2Gamepad(config)

    # Verify configuration was applied correctly
    assert gamepad.v_x_sensitivity == 1.1
    assert gamepad.v_y_sensitivity == 0.6
    assert gamepad.omega_z_sensitivity == 1.2
    assert gamepad.dead_zone == 0.02

    # Test advance() returns expected type
    result = gamepad.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)  # (v_x, v_y, omega_z)


def test_se3gamepad_constructors(mock_environment, mocker):
    """Test constructor for Se3Gamepad."""
    # Test config-based constructor
    config = Se3GamepadCfg(
        pos_sensitivity=1.1,
        rot_sensitivity=1.7,
        dead_zone=0.02,
    )
    device_mod = importlib.import_module("isaaclab.devices.gamepad.se3_gamepad")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    gamepad = Se3Gamepad(config)

    # Verify configuration was applied correctly
    assert gamepad.pos_sensitivity == 1.1
    assert gamepad.rot_sensitivity == 1.7
    assert gamepad.dead_zone == 0.02

    # Test advance() returns expected type
    result = gamepad.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7,)  # (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper)


"""
Test spacemouse devices.
"""


def test_se2spacemouse_constructors(mock_environment, mocker):
    """Test constructor for Se2SpaceMouse."""
    # Test config-based constructor
    config = Se2SpaceMouseCfg(
        v_x_sensitivity=0.9,
        v_y_sensitivity=0.5,
        omega_z_sensitivity=1.2,
    )
    device_mod = importlib.import_module("isaaclab.devices.spacemouse.se2_spacemouse")
    mocker.patch.dict("sys.modules", {"hid": mock_environment["hid"]})
    mocker.patch.object(device_mod, "hid", mock_environment["hid"])

    spacemouse = Se2SpaceMouse(config)

    # Verify configuration was applied correctly
    assert spacemouse.v_x_sensitivity == 0.9
    assert spacemouse.v_y_sensitivity == 0.5
    assert spacemouse.omega_z_sensitivity == 1.2

    # Test advance() returns expected type
    mock_environment["device"].read.return_value = [1, 0, 0, 0, 0]
    result = spacemouse.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)  # (v_x, v_y, omega_z)


def test_se3spacemouse_constructors(mock_environment, mocker):
    """Test constructor for Se3SpaceMouse."""
    # Test config-based constructor
    config = Se3SpaceMouseCfg(
        pos_sensitivity=0.5,
        rot_sensitivity=0.9,
    )
    device_mod = importlib.import_module("isaaclab.devices.spacemouse.se3_spacemouse")
    mocker.patch.dict("sys.modules", {"hid": mock_environment["hid"]})
    mocker.patch.object(device_mod, "hid", mock_environment["hid"])

    spacemouse = Se3SpaceMouse(config)

    # Verify configuration was applied correctly
    assert spacemouse.pos_sensitivity == 0.5
    assert spacemouse.rot_sensitivity == 0.9

    # Test advance() returns expected type
    mock_environment["device"].read.return_value = [1, 0, 0, 0, 0, 0, 0]
    result = spacemouse.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7,)  # (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper)


"""
Test phone devices.
"""


def test_se2phone_constructor(mock_environment, mocker):
    """Test constructor and delta-output behavior for Se2Phone."""

    # --- Fake Teleop that captures the callback and allows us to emit messages ---
    class _FakeTeleop:
        def __init__(self, **kwargs):
            self._cb = None

        def subscribe(self, cb):
            self._cb = cb

        def run(self):
            # No-op: don't start any network/server loop in tests
            return

        def emit(self, msg: dict):
            assert self._cb is not None, "Callback not registered"
            # The device ignores the pose argument; pass a dummy np.array
            self._cb(np.zeros(7), msg)

    # Import the device module and patch Teleop with our fake
    device_mod = importlib.import_module("isaaclab.devices.phone.se2_phone")
    mocker.patch.object(device_mod, "Teleop", _FakeTeleop)

    # Build config with custom sensitivities (we'll verify via output)
    cfg = Se2PhoneCfg(v_x_sensitivity=0.9, v_y_sensitivity=0.5, omega_z_sensitivity=1.2)

    # Create the device
    phone = Se2Phone(cfg)
    assert isinstance(phone, Se2Phone)

    # Grab the fake teleop instance to push messages
    fake = phone._teleop
    assert isinstance(fake, _FakeTeleop)

    # Helper: orientation kept constant so omega_z delta is zero (easier assertion)
    orient = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    # 1) First message initializes internal reference; advance() should return zeros.
    fake.emit({"move": True, "position": {"x": 0.0, "z": 0.0}, "orientation": orient})
    out1 = phone.advance()
    assert isinstance(out1, torch.Tensor)
    assert out1.shape == (3,)
    assert torch.allclose(out1, torch.zeros(3, dtype=torch.float32, device=out1.device))

    # 2) Second message changes position (x, z). Se2Phone maps latest_v_x=-z, latest_v_y=-x.
    #    Deltas: dvx = -(2.0 - 0.0) = -2.0, dvy = -(1.0 - 0.0) = -1.0
    fake.emit({"move": True, "position": {"x": 1.0, "z": 2.0}, "orientation": orient})
    out2 = phone.advance()

    # Expected scaled command:
    #   vx = dvx * 0.9 = -1.8
    #   vy = dvy * 0.5 = -0.5
    #   omega_z = 0.0 (orientation unchanged)
    expected = torch.tensor([-1.8, -0.5, 0.0], dtype=torch.float32, device=out2.device)
    assert out2.shape == (3,)
    assert torch.allclose(out2, expected, atol=1e-5)


def test_se3phone_constructor(mocker):
    """Test constructor and delta-output behavior for Se3Phone."""

    # --- Fake Teleop that captures the callback and lets us emit messages ---
    class _FakeTeleop:
        def __init__(self, **kwargs):
            self._cb = None

        def subscribe(self, cb):
            self._cb = cb

        def run(self):
            return  # no thread loop in tests

        def emit(self, msg: dict):
            assert self._cb is not None, "Callback not registered"
            self._cb(np.zeros(7), msg)  # pose is unused by device

    # Import the device module and patch Teleop with our fake
    device_mod = importlib.import_module("isaaclab.devices.phone.se3_phone")
    mocker.patch.object(device_mod, "Teleop", _FakeTeleop)

    # Build config with custom sensitivities to verify scaling
    cfg = Se3PhoneCfg(pos_sensitivity=0.5, rot_sensitivity=0.4, gripper_term=True)

    # Create the device
    phone = Se3Phone(cfg)
    assert isinstance(phone, Se3Phone)

    fake = phone._teleop
    assert isinstance(fake, _FakeTeleop)

    # Keep gripper explicit; start with OPEN
    # Keep orientation constant on first emit to initialize state
    orient0 = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    # 1) First message initializes internal reference; advance() -> zeros (with gripper)
    fake.emit({
        "move": True,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": orient0,
        "gripper": "open",
    })
    out1 = phone.advance()
    assert isinstance(out1, torch.Tensor)
    assert out1.shape == (7,)
    # first call yields zeros for deltas; gripper=+1
    expected1 = torch.zeros(7, dtype=torch.float32, device=out1.device)
    expected1[6] = 1.0
    assert torch.allclose(out1, expected1, atol=1e-6)

    # 2) Second message changes position & orientation.
    # Position mapping inside device: latest_pos = [-z, -x, y]
    # Use x=1, y=2, z=3  -> latest_pos = [-3, -1, 2], prev was [0,0,0]
    # Orientation: apply yaw rotation by theta about Z:
    # quat (w,x,y,z) = (cos(theta/2), 0, 0, sin(theta/2))
    # axis_angle_from_quat -> [0,0,theta]
    # device remaps: rot[[0,1,2]] = rot[[2,0,1]] * [-1,-1,1]
    # so -> [theta, 0, 0] * [-1,-1,1] = [-theta, 0, 0]
    theta = 0.2  # radians
    w = float(np.cos(theta / 2.0))
    z = float(np.sin(theta / 2.0))
    orient1 = {"x": 0.0, "y": 0.0, "z": z, "w": w}

    fake.emit({
        "move": True,
        "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "orientation": orient1,
        "gripper": "open",
    })
    out2 = phone.advance()

    # Expected scaled deltas
    dpos = torch.tensor([-3.0, -1.0, 2.0], dtype=torch.float32, device=out2.device) * 0.5
    drot = torch.tensor([-theta, 0.0, 0.0], dtype=torch.float32, device=out2.device) * 0.4
    expected2 = torch.cat([dpos, drot, torch.tensor([1.0], dtype=torch.float32, device=out2.device)], dim=0)

    assert out2.shape == (7,)
    assert torch.allclose(out2, expected2, atol=1e-5)

    # 3) Gate OFF should zero deltas and resync reference
    fake.emit({
        "move": False,  # gate off
        "position": {"x": 4.0, "y": 5.0, "z": 6.0},
        "orientation": orient1,  # unchanged quaternion
        "gripper": "close",
    })
    out3 = phone.advance()
    # deltas should be zero; gripper should now be -1
    expected3 = torch.zeros(7, dtype=torch.float32, device=out3.device)
    expected3[6] = -1.0
    assert torch.allclose(out3, expected3, atol=1e-6)


"""
Test OpenXR devices.
"""


def test_openxr_constructors(mock_environment, mocker):
    """Test constructor for OpenXRDevice."""
    # Test config-based constructor with custom XrCfg
    xr_cfg = XrCfg(
        anchor_pos=(1.0, 2.0, 3.0),
        anchor_rot=(0.0, 0.1, 0.2, 0.3),
        near_plane=0.2,
    )
    config = OpenXRDeviceCfg(xr_cfg=xr_cfg)

    # Create mock retargeters
    mock_controller_retargeter = mocker.MagicMock()
    mock_head_retargeter = mocker.MagicMock()
    retargeters = [mock_controller_retargeter, mock_head_retargeter]

    device_mod = importlib.import_module("isaaclab.devices.openxr.openxr_device")
    mocker.patch.dict(
        "sys.modules",
        {
            "carb": mock_environment["carb"],
            "omni.kit.xr.core": mock_environment["omni"].kit.xr.core,
            "isaacsim.core.prims": mocker.MagicMock(),
        },
    )
    mocker.patch.object(device_mod, "XRCore", mock_environment["omni"].kit.xr.core.XRCore)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", mock_environment["omni"].kit.xr.core.XRPoseValidityFlags)
    mock_single_xform = mocker.patch.object(device_mod, "SingleXFormPrim")

    # Configure the mock to return a string for prim_path
    mock_instance = mock_single_xform.return_value
    mock_instance.prim_path = "/XRAnchor"

    # Create the device using the factory
    device = OpenXRDevice(config)

    # Verify the device was created successfully
    assert device._xr_cfg == xr_cfg

    # Test with retargeters
    device = OpenXRDevice(cfg=config, retargeters=retargeters)

    # Verify retargeters were correctly assigned as a list
    assert device._retargeters == retargeters

    # Test with config and retargeters
    device = OpenXRDevice(cfg=config, retargeters=retargeters)

    # Verify both config and retargeters were correctly assigned
    assert device._xr_cfg == xr_cfg
    assert device._retargeters == retargeters

    # Test reset functionality
    device.reset()


"""
Test teleop device factory.
"""


def test_create_teleop_device_basic(mock_environment, mocker):
    """Test creating devices using the teleop device factory."""
    # Create device configuration
    keyboard_cfg = Se3KeyboardCfg(pos_sensitivity=0.8, rot_sensitivity=1.2)

    # Create devices configuration dictionary
    devices_cfg = {"test_keyboard": keyboard_cfg}

    # Mock Se3Keyboard class
    device_mod = importlib.import_module("isaaclab.devices.keyboard.se3_keyboard")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    # Create the device using the factory
    device = create_teleop_device("test_keyboard", devices_cfg)

    # Verify the device was created correctly
    assert isinstance(device, Se3Keyboard)
    assert device.pos_sensitivity == 0.8
    assert device.rot_sensitivity == 1.2


def test_create_teleop_device_with_callbacks(mock_environment, mocker):
    """Test creating device with callbacks."""
    # Create device configuration
    xr_cfg = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(1.0, 0.0, 0.0, 0.0), near_plane=0.15)
    openxr_cfg = OpenXRDeviceCfg(xr_cfg=xr_cfg)

    # Create devices configuration dictionary
    devices_cfg = {"test_xr": openxr_cfg}

    # Create mock callbacks
    button_a_callback = mocker.MagicMock()
    button_b_callback = mocker.MagicMock()
    callbacks = {"button_a": button_a_callback, "button_b": button_b_callback}

    # Mock OpenXRDevice class and dependencies
    device_mod = importlib.import_module("isaaclab.devices.openxr.openxr_device")
    mocker.patch.dict(
        "sys.modules",
        {
            "carb": mock_environment["carb"],
            "omni.kit.xr.core": mock_environment["omni"].kit.xr.core,
            "isaacsim.core.prims": mocker.MagicMock(),
        },
    )
    mocker.patch.object(device_mod, "XRCore", mock_environment["omni"].kit.xr.core.XRCore)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", mock_environment["omni"].kit.xr.core.XRPoseValidityFlags)
    mock_single_xform = mocker.patch.object(device_mod, "SingleXFormPrim")

    # Configure the mock to return a string for prim_path
    mock_instance = mock_single_xform.return_value
    mock_instance.prim_path = "/XRAnchor"

    # Create the device using the factory
    device = create_teleop_device("test_xr", devices_cfg, callbacks)

    # Verify the device was created correctly
    assert isinstance(device, OpenXRDevice)

    # Verify callbacks were registered
    device.add_callback("button_a", button_a_callback)
    device.add_callback("button_b", button_b_callback)
    assert len(device._additional_callbacks) == 2


def test_create_teleop_device_with_retargeters(mock_environment, mocker):
    """Test creating device with retargeters."""
    # Create retargeter configurations
    retargeter_cfg1 = Se3AbsRetargeterCfg()
    retargeter_cfg2 = GripperRetargeterCfg()

    # Create device configuration with retargeters
    xr_cfg = XrCfg()
    device_cfg = OpenXRDeviceCfg(xr_cfg=xr_cfg, retargeters=[retargeter_cfg1, retargeter_cfg2])

    # Create devices configuration dictionary
    devices_cfg = {"test_xr": device_cfg}

    # Mock OpenXRDevice class and dependencies
    device_mod = importlib.import_module("isaaclab.devices.openxr.openxr_device")
    mocker.patch.dict(
        "sys.modules",
        {
            "carb": mock_environment["carb"],
            "omni.kit.xr.core": mock_environment["omni"].kit.xr.core,
            "isaacsim.core.prims": mocker.MagicMock(),
        },
    )
    mocker.patch.object(device_mod, "XRCore", mock_environment["omni"].kit.xr.core.XRCore)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", mock_environment["omni"].kit.xr.core.XRPoseValidityFlags)
    mock_single_xform = mocker.patch.object(device_mod, "SingleXFormPrim")

    # Configure the mock to return a string for prim_path
    mock_instance = mock_single_xform.return_value
    mock_instance.prim_path = "/XRAnchor"

    # Mock retargeter classes
    retargeter_mod = importlib.import_module("isaaclab.devices.openxr.retargeters")
    mocker.patch.object(retargeter_mod, "Se3AbsRetargeter")
    mocker.patch.object(retargeter_mod, "GripperRetargeter")

    # Create the device using the factory
    device = create_teleop_device("test_xr", devices_cfg)

    # Verify retargeters were created
    assert len(device._retargeters) == 2


def test_create_teleop_device_device_not_found():
    """Test error when device name is not found in configuration."""
    # Create devices configuration dictionary
    devices_cfg = {"keyboard": Se3KeyboardCfg()}

    # Try to create a non-existent device
    with pytest.raises(ValueError, match="Device 'gamepad' not found"):
        create_teleop_device("gamepad", devices_cfg)


def test_create_teleop_device_unsupported_config():
    """Test error when device configuration type is not supported."""

    # Create a custom unsupported configuration class
    class UnsupportedCfg:
        pass

    # Create devices configuration dictionary with unsupported config
    devices_cfg = {"unsupported": UnsupportedCfg()}

    # Try to create a device with unsupported configuration
    with pytest.raises(ValueError, match="Unsupported device configuration type"):
        create_teleop_device("unsupported", devices_cfg)
