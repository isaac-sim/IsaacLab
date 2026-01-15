# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import importlib
import json
from typing import cast

import pytest
import torch

# Import device classes to test
from isaaclab.devices import (
    DeviceCfg,
    HaplyDevice,
    HaplyDeviceCfg,
    OpenXRDevice,
    OpenXRDeviceCfg,
    Se2Gamepad,
    Se2GamepadCfg,
    Se2Keyboard,
    Se2KeyboardCfg,
    Se2SpaceMouse,
    Se2SpaceMouseCfg,
    Se3Gamepad,
    Se3GamepadCfg,
    Se3Keyboard,
    Se3KeyboardCfg,
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

    # Mock carb events used by OpenXRDevice
    events_mock = mocker.MagicMock()
    events_mock.type_from_string.return_value = 0
    carb_mock.events = events_mock

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

    # Mock Haply WebSocket
    websockets_mock = mocker.MagicMock()
    websocket_mock = mocker.MagicMock()
    websockets_mock.connect.return_value.__aenter__.return_value = websocket_mock

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
        "websockets": websockets_mock,
        "websocket": websocket_mock,
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
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
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
Test Haply devices.
"""


def test_haply_constructors(mock_environment, mocker):
    """Test constructor for HaplyDevice."""
    # Test config-based constructor
    config = HaplyDeviceCfg(
        websocket_uri="ws://localhost:10001",
        pos_sensitivity=1.5,
        data_rate=250.0,
    )

    # Mock the websockets module and asyncio
    device_mod = importlib.import_module("isaaclab.devices.haply.se3_haply")
    mocker.patch.dict("sys.modules", {"websockets": mock_environment["websockets"]})
    mocker.patch.object(device_mod, "websockets", mock_environment["websockets"])

    # Mock asyncio to prevent actual async operations
    asyncio_mock = mocker.MagicMock()
    mocker.patch.object(device_mod, "asyncio", asyncio_mock)

    # Mock threading to prevent actual thread creation
    threading_mock = mocker.MagicMock()
    thread_instance = mocker.MagicMock()
    threading_mock.Thread.return_value = thread_instance
    thread_instance.is_alive.return_value = False
    mocker.patch.object(device_mod, "threading", threading_mock)

    # Mock time.time() for connection timeout simulation
    time_mock = mocker.MagicMock()
    time_mock.time.side_effect = [0.0, 0.1, 0.2, 0.3, 6.0]  # Will timeout
    mocker.patch.object(device_mod, "time", time_mock)

    # Create sample WebSocket response data
    ws_response = {
        "inverse3": [
            {
                "device_id": "test_inverse3_123",
                "state": {"cursor_position": {"x": 0.1, "y": 0.2, "z": 0.3}},
            }
        ],
        "wireless_verse_grip": [
            {
                "device_id": "test_versegrip_456",
                "state": {
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "buttons": {"a": False, "b": False, "c": False},
                },
            }
        ],
    }

    # Configure websocket mock to return JSON data
    mock_environment["websocket"].recv = mocker.AsyncMock(return_value=json.dumps(ws_response))
    mock_environment["websocket"].send = mocker.AsyncMock()

    # The constructor will raise RuntimeError due to timeout, which is expected in test
    with pytest.raises(RuntimeError, match="Failed to connect both Inverse3 and VerseGrip devices"):
        haply = HaplyDevice(config)

    # Now test successful connection by mocking time to not timeout
    time_mock.time.side_effect = [0.0, 0.1, 0.2, 0.3, 0.4]  # Won't timeout

    # Mock the connection status
    mocker.patch.object(device_mod.HaplyDevice, "_start_websocket_thread")
    haply = device_mod.HaplyDevice.__new__(device_mod.HaplyDevice)
    haply._sim_device = config.sim_device
    haply.websocket_uri = config.websocket_uri
    haply.pos_sensitivity = config.pos_sensitivity
    haply.data_rate = config.data_rate
    haply.limit_force = config.limit_force
    haply.connected = True
    haply.inverse3_device_id = "test_inverse3_123"
    haply.verse_grip_device_id = "test_versegrip_456"
    haply.data_lock = threading_mock.Lock()
    haply.force_lock = threading_mock.Lock()
    haply._connected_lock = threading_mock.Lock()
    haply._additional_callbacks = {}
    haply._prev_buttons = {"a": False, "b": False, "c": False}
    haply._websocket_thread = None  # Initialize to prevent AttributeError in __del__
    haply.running = True
    haply.cached_data = {
        "position": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32).numpy(),
        "quaternion": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32).numpy(),
        "buttons": {"a": False, "b": False, "c": False},
        "inverse3_connected": True,
        "versegrip_connected": True,
    }
    haply.feedback_force = {"x": 0.0, "y": 0.0, "z": 0.0}

    # Verify configuration was applied correctly
    assert haply.websocket_uri == "ws://localhost:10001"
    assert haply.pos_sensitivity == 1.5
    assert haply.data_rate == 250.0

    # Test advance() returns expected type
    result = haply.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (10,)  # (pos_x, pos_y, pos_z, qx, qy, qz, qw, btn_a, btn_b, btn_c)

    # Test push_force with tensor (single force vector)
    forces_within = torch.tensor([[1.0, 1.5, -0.5]], dtype=torch.float32)
    position_zero = torch.tensor([0], dtype=torch.long)
    haply.push_force(forces_within, position_zero)
    assert haply.feedback_force["x"] == pytest.approx(1.0)
    assert haply.feedback_force["y"] == pytest.approx(1.5)
    assert haply.feedback_force["z"] == pytest.approx(-0.5)

    # Test push_force with tensor (force limiting, default limit is 2.0 N)
    forces_exceed = torch.tensor([[5.0, -10.0, 1.5]], dtype=torch.float32)
    haply.push_force(forces_exceed, position_zero)
    assert haply.feedback_force["x"] == pytest.approx(2.0)
    assert haply.feedback_force["y"] == pytest.approx(-2.0)
    assert haply.feedback_force["z"] == pytest.approx(1.5)

    # Test push_force with position tensor (single index)
    forces_multi = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.8, -0.3], [0.1, 0.2, 0.3]], dtype=torch.float32)
    position_single = torch.tensor([1], dtype=torch.long)
    haply.push_force(forces_multi, position=position_single)
    assert haply.feedback_force["x"] == pytest.approx(0.5)
    assert haply.feedback_force["y"] == pytest.approx(0.8)
    assert haply.feedback_force["z"] == pytest.approx(-0.3)

    # Test push_force with position tensor (multiple indices)
    position_multi = torch.tensor([0, 2], dtype=torch.long)
    haply.push_force(forces_multi, position=position_multi)
    # Should sum forces[0] and forces[2]: [1.0+0.1, 2.0+0.2, 3.0+0.3] = [1.1, 2.2, 3.3]
    # But clipped to [-2.0, 2.0]: [1.1, 2.0, 2.0]
    assert haply.feedback_force["x"] == pytest.approx(1.1)
    assert haply.feedback_force["y"] == pytest.approx(2.0)
    assert haply.feedback_force["z"] == pytest.approx(2.0)

    # Test reset functionality
    haply.reset()
    assert haply.feedback_force == {"x": 0.0, "y": 0.0, "z": 0.0}


"""
Test teleop device factory.
"""


def test_create_teleop_device_basic(mock_environment, mocker):
    """Test creating devices using the teleop device factory."""
    # Create device configuration
    keyboard_cfg = Se3KeyboardCfg(pos_sensitivity=0.8, rot_sensitivity=1.2)

    # Create devices configuration dictionary
    devices_cfg: dict[str, DeviceCfg] = {"test_keyboard": keyboard_cfg}

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
    devices_cfg: dict[str, DeviceCfg] = {"test_xr": openxr_cfg}

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
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
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

    # Verify callbacks were registered by the factory
    assert set(device._additional_callbacks.keys()) == {"button_a", "button_b"}


def test_create_teleop_device_with_retargeters(mock_environment, mocker):
    """Test creating device with retargeters."""
    # Create retargeter configurations
    retargeter_cfg1 = Se3AbsRetargeterCfg()
    retargeter_cfg2 = GripperRetargeterCfg()

    # Create device configuration with retargeters
    xr_cfg = XrCfg()
    device_cfg = OpenXRDeviceCfg(xr_cfg=xr_cfg, retargeters=[retargeter_cfg1, retargeter_cfg2])

    # Create devices configuration dictionary
    devices_cfg: dict[str, DeviceCfg] = {"test_xr": device_cfg}

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
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "XRCore", mock_environment["omni"].kit.xr.core.XRCore)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", mock_environment["omni"].kit.xr.core.XRPoseValidityFlags)
    mock_single_xform = mocker.patch.object(device_mod, "SingleXFormPrim")

    # Configure the mock to return a string for prim_path
    mock_instance = mock_single_xform.return_value
    mock_instance.prim_path = "/XRAnchor"

    # Create the device using the factory
    device = create_teleop_device("test_xr", devices_cfg)

    # Verify retargeters were created
    assert len(device._retargeters) == 2


def test_create_teleop_device_device_not_found():
    """Test error when device name is not found in configuration."""
    # Create devices configuration dictionary
    devices_cfg: dict[str, DeviceCfg] = {"keyboard": Se3KeyboardCfg()}

    # Try to create a non-existent device
    with pytest.raises(ValueError, match="Device 'gamepad' not found"):
        create_teleop_device("gamepad", devices_cfg)


def test_create_teleop_device_unsupported_config():
    """Test error when device configuration type is not supported."""

    # Create a custom unsupported configuration class
    class UnsupportedCfg:
        pass

    # Create devices configuration dictionary with unsupported config
    devices_cfg: dict[str, DeviceCfg] = cast(dict[str, DeviceCfg], {"unsupported": UnsupportedCfg()})

    # Try to create a device with unsupported configuration
    with pytest.raises(ValueError, match="does not declare class_type"):
        create_teleop_device("unsupported", devices_cfg)
