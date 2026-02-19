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
from typing import cast

import pytest
from isaaclab_teleop.deprecated.openxr import OpenXRDevice, OpenXRDeviceCfg, XrCfg
from isaaclab_teleop.deprecated.openxr.retargeters import GripperRetargeterCfg, Se3AbsRetargeterCfg

# Import teleop device factory for testing
from isaaclab_teleop.deprecated.teleop_device_factory import create_teleop_device

# Import device classes to test
from isaaclab.devices import (
    DeviceCfg,
    Se3Keyboard,
    Se3KeyboardCfg,
)


@pytest.fixture
def mock_environment(mocker):
    """Set up common mock objects for tests."""
    carb_mock = mocker.MagicMock()
    omni_mock = mocker.MagicMock()
    appwindow_mock = mocker.MagicMock()
    keyboard_mock = mocker.MagicMock()
    gamepad_mock = mocker.MagicMock()
    input_mock = mocker.MagicMock()
    settings_mock = mocker.MagicMock()
    hid_mock = mocker.MagicMock()
    device_mock = mocker.MagicMock()

    omni_mock.appwindow.get_default_app_window.return_value = appwindow_mock
    appwindow_mock.get_keyboard.return_value = keyboard_mock
    appwindow_mock.get_gamepad.return_value = gamepad_mock
    carb_mock.input.acquire_input_interface.return_value = input_mock
    carb_mock.settings.get_settings.return_value = settings_mock

    carb_mock.input.KeyboardEventType.KEY_PRESS = 1
    carb_mock.input.KeyboardEventType.KEY_RELEASE = 2

    events_mock = mocker.MagicMock()
    events_mock.type_from_string.return_value = 0
    carb_mock.events = events_mock

    hid_mock.enumerate.return_value = [{"product_string": "SpaceMouse Compact", "vendor_id": 123, "product_id": 456}]
    hid_mock.device.return_value = device_mock

    message_bus_mock = mocker.MagicMock()
    singleton_mock = mocker.MagicMock()
    omni_mock.kit.xr.core.XRCore.get_singleton.return_value = singleton_mock
    singleton_mock.get_message_bus.return_value = message_bus_mock
    omni_mock.kit.xr.core.XRPoseValidityFlags.POSITION_VALID = 1
    omni_mock.kit.xr.core.XRPoseValidityFlags.ORIENTATION_VALID = 2

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
Test OpenXR devices.
"""


def test_openxr_constructors(mock_environment, mocker):
    """Test constructor for OpenXRDevice."""
    xr_cfg = XrCfg(
        anchor_pos=(1.0, 2.0, 3.0),
        anchor_rot=(0.0, 0.1, 0.2, 0.3),
        near_plane=0.2,
    )
    config = OpenXRDeviceCfg(xr_cfg=xr_cfg)

    mock_controller_retargeter = mocker.MagicMock()
    mock_head_retargeter = mocker.MagicMock()
    retargeters = [mock_controller_retargeter, mock_head_retargeter]

    device_mod = importlib.import_module("isaaclab_teleop.deprecated.openxr.openxr_device")
    mocker.patch.dict(
        "sys.modules",
        {
            "carb": mock_environment["carb"],
            "omni.kit.xr.core": mock_environment["omni"].kit.xr.core,
        },
    )
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "XRCore", mock_environment["omni"].kit.xr.core.XRCore)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", mock_environment["omni"].kit.xr.core.XRPoseValidityFlags)

    mock_stage = mocker.MagicMock()
    mock_prim = mocker.MagicMock()
    mock_prim.IsValid.return_value = False
    mock_stage.GetPrimAtPath.return_value = mock_prim
    mocker.patch.object(device_mod, "sim_utils", mocker.MagicMock())
    device_mod.sim_utils.get_current_stage.return_value = mock_stage
    device_mod.sim_utils.create_prim.return_value = None

    device = OpenXRDevice(config)
    assert device._xr_cfg == xr_cfg

    device = OpenXRDevice(cfg=config, retargeters=retargeters)
    assert device._retargeters == retargeters

    device = OpenXRDevice(cfg=config, retargeters=retargeters)
    assert device._xr_cfg == xr_cfg
    assert device._retargeters == retargeters

    device.reset()


"""
Test teleop device factory.
"""


def test_create_teleop_device_basic(mock_environment, mocker):
    """Test creating devices using the teleop device factory."""
    keyboard_cfg = Se3KeyboardCfg(pos_sensitivity=0.8, rot_sensitivity=1.2)
    devices_cfg: dict[str, DeviceCfg] = {"test_keyboard": keyboard_cfg}

    device_mod = importlib.import_module("isaaclab.devices.keyboard.se3_keyboard")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    device = create_teleop_device("test_keyboard", devices_cfg)

    assert isinstance(device, Se3Keyboard)
    assert device.pos_sensitivity == 0.8
    assert device.rot_sensitivity == 1.2


def test_create_teleop_device_with_callbacks(mock_environment, mocker):
    """Test creating device with callbacks."""
    xr_cfg = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(0.0, 0.0, 0.0, 1.0), near_plane=0.15)
    openxr_cfg = OpenXRDeviceCfg(xr_cfg=xr_cfg)
    devices_cfg: dict[str, DeviceCfg] = {"test_xr": openxr_cfg}

    button_a_callback = mocker.MagicMock()
    button_b_callback = mocker.MagicMock()
    callbacks = {"button_a": button_a_callback, "button_b": button_b_callback}

    device_mod = importlib.import_module("isaaclab_teleop.deprecated.openxr.openxr_device")
    mocker.patch.dict(
        "sys.modules",
        {
            "carb": mock_environment["carb"],
            "omni.kit.xr.core": mock_environment["omni"].kit.xr.core,
        },
    )
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "XRCore", mock_environment["omni"].kit.xr.core.XRCore)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", mock_environment["omni"].kit.xr.core.XRPoseValidityFlags)

    mock_stage = mocker.MagicMock()
    mock_prim = mocker.MagicMock()
    mock_prim.IsValid.return_value = False
    mock_stage.GetPrimAtPath.return_value = mock_prim
    mocker.patch.object(device_mod, "sim_utils", mocker.MagicMock())
    device_mod.sim_utils.get_current_stage.return_value = mock_stage
    device_mod.sim_utils.create_prim.return_value = None

    device = create_teleop_device("test_xr", devices_cfg, callbacks)

    assert isinstance(device, OpenXRDevice)
    assert set(device._additional_callbacks.keys()) == {"button_a", "button_b"}


def test_create_teleop_device_with_retargeters(mock_environment, mocker):
    """Test creating device with retargeters."""
    retargeter_cfg1 = Se3AbsRetargeterCfg()
    retargeter_cfg2 = GripperRetargeterCfg()

    xr_cfg = XrCfg()
    device_cfg = OpenXRDeviceCfg(xr_cfg=xr_cfg, retargeters=[retargeter_cfg1, retargeter_cfg2])
    devices_cfg: dict[str, DeviceCfg] = {"test_xr": device_cfg}

    device_mod = importlib.import_module("isaaclab_teleop.deprecated.openxr.openxr_device")
    mocker.patch.dict(
        "sys.modules",
        {
            "carb": mock_environment["carb"],
            "omni.kit.xr.core": mock_environment["omni"].kit.xr.core,
        },
    )
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "XRCore", mock_environment["omni"].kit.xr.core.XRCore)
    mocker.patch.object(device_mod, "XRPoseValidityFlags", mock_environment["omni"].kit.xr.core.XRPoseValidityFlags)

    mock_stage = mocker.MagicMock()
    mock_prim = mocker.MagicMock()
    mock_prim.IsValid.return_value = False
    mock_stage.GetPrimAtPath.return_value = mock_prim
    mocker.patch.object(device_mod, "sim_utils", mocker.MagicMock())
    device_mod.sim_utils.get_current_stage.return_value = mock_stage
    device_mod.sim_utils.create_prim.return_value = None

    device = create_teleop_device("test_xr", devices_cfg)

    assert len(device._retargeters) == 2


def test_create_teleop_device_device_not_found():
    """Test error when device name is not found in configuration."""
    devices_cfg: dict[str, DeviceCfg] = {"keyboard": Se3KeyboardCfg()}

    with pytest.raises(ValueError, match="Device 'gamepad' not found"):
        create_teleop_device("gamepad", devices_cfg)


def test_create_teleop_device_unsupported_config():
    """Test error when device configuration type is not supported."""

    class UnsupportedCfg:
        pass

    devices_cfg: dict[str, DeviceCfg] = cast(dict[str, DeviceCfg], {"unsupported": UnsupportedCfg()})

    with pytest.raises(ValueError, match="does not declare class_type"):
        create_teleop_device("unsupported", devices_cfg)
