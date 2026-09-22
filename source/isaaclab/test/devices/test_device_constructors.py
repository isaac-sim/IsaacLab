# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constructor and data-flow tests for the teleoperation devices.

The devices talk to hardware through ``carb``/``omni`` (keyboard, gamepad), ``hid`` (SpaceMouse), and
``websockets`` (Haply). Those interfaces are replaced by fakes so the tests run without Kit or hardware.
"""

import importlib
import sys
import threading
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch


def _install_fake_module(name: str) -> list[str]:
    """Register a placeholder module so the device modules import without Kit or hardware libraries."""
    if name in sys.modules:
        return []
    module = ModuleType(name)
    module.__path__ = []
    installed = {name: module}
    if name == "carb":
        module.input = ModuleType("carb.input")
        module.input.KeyboardEventType = MagicMock(KEY_PRESS=1, KEY_RELEASE=2)
        module.input.GamepadInput = MagicMock()
        module.input.acquire_input_interface = MagicMock()
        installed["carb.input"] = module.input
    elif name == "omni":
        module.appwindow = ModuleType("omni.appwindow")
        module.appwindow.get_default_app_window = MagicMock()
        installed["omni.appwindow"] = module.appwindow
    sys.modules.update(installed)
    return list(installed)


_fake_modules = [name for module in ("carb", "omni", "hid", "websockets") for name in _install_fake_module(module)]

from isaaclab.devices import (  # noqa: E402
    HaplyDevice,
    HaplyDeviceCfg,
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

# the device modules keep their own references; drop the placeholders so other tests see the real imports
for _name in _fake_modules:
    sys.modules.pop(_name, None)

pytestmark = pytest.mark.unit

_SPACEMOUSE_COMPACT = {"product_string": "SpaceMouse Compact", "vendor_id": 0x256F, "product_id": 0xC635}


@pytest.fixture
def kit(mocker):
    """Fake carb/omni input interfaces shared by the keyboard and gamepad devices."""
    carb_mock = mocker.MagicMock()
    carb_mock.input.KeyboardEventType.KEY_PRESS = 1
    carb_mock.input.KeyboardEventType.KEY_RELEASE = 2
    omni_mock = mocker.MagicMock()
    mocker.patch("isaaclab.app.settings_manager.get_settings_manager", mocker.MagicMock())
    for module_name in (
        "isaaclab.devices.keyboard.se2_keyboard",
        "isaaclab.devices.keyboard.se3_keyboard",
        "isaaclab.devices.gamepad.se2_gamepad",
        "isaaclab.devices.gamepad.se3_gamepad",
    ):
        module = importlib.import_module(module_name)
        mocker.patch.object(module, "carb", carb_mock)
        mocker.patch.object(module, "omni", omni_mock)
        if hasattr(module, "get_settings_manager"):
            mocker.patch.object(module, "get_settings_manager", mocker.MagicMock())
    return carb_mock


@pytest.fixture
def hid(mocker):
    """Fake ``hid`` backend enumerating one SpaceMouse Compact whose reads return no data."""
    hid_mock = mocker.MagicMock()
    hid_mock.enumerate.return_value = [dict(_SPACEMOUSE_COMPACT)]
    hid_mock.device.return_value.read.return_value = None
    for module_name in ("isaaclab.devices.spacemouse.se2_spacemouse", "isaaclab.devices.spacemouse.se3_spacemouse"):
        module = importlib.import_module(module_name)
        mocker.patch.object(module, "hid", hid_mock)
        # the listener thread would spin on the fake device; detection and command flow are under test
        mocker.patch.object(module, "threading")
        mocker.patch.object(module.time, "sleep")
    return hid_mock


@pytest.mark.parametrize(
    ("device_cls", "cfg", "expected_dim"),
    [
        (Se2Keyboard, Se2KeyboardCfg(v_x_sensitivity=0.9, v_y_sensitivity=0.5, omega_z_sensitivity=1.2), 3),
        (Se3Keyboard, Se3KeyboardCfg(pos_sensitivity=0.5, rot_sensitivity=0.9), 7),
        (
            Se2Gamepad,
            Se2GamepadCfg(v_x_sensitivity=1.1, v_y_sensitivity=0.6, omega_z_sensitivity=1.2, dead_zone=0.02),
            3,
        ),
        (Se3Gamepad, Se3GamepadCfg(pos_sensitivity=1.1, rot_sensitivity=1.7, dead_zone=0.02), 7),
        (Se2SpaceMouse, Se2SpaceMouseCfg(v_x_sensitivity=0.9, v_y_sensitivity=0.5, omega_z_sensitivity=1.2), 3),
        (Se3SpaceMouse, Se3SpaceMouseCfg(pos_sensitivity=0.5, rot_sensitivity=0.9), 7),
    ],
    ids=["se2_keyboard", "se3_keyboard", "se2_gamepad", "se3_gamepad", "se2_spacemouse", "se3_spacemouse"],
)
def test_device_constructors(kit, hid, device_cls, cfg, expected_dim):
    """Devices adopt their configuration and report an idle command of the documented size."""
    device = device_cls(cfg)

    for name in ("v_x_sensitivity", "v_y_sensitivity", "omega_z_sensitivity", "pos_sensitivity", "rot_sensitivity"):
        if hasattr(cfg, name):
            assert getattr(device, name) == getattr(cfg, name)
    if hasattr(cfg, "dead_zone"):
        assert device.dead_zone == cfg.dead_zone

    command = device.advance()
    assert isinstance(command, torch.Tensor)
    assert command.shape == (expected_dim,)
    # idle devices command no motion; SE(3) devices report an open gripper
    torch.testing.assert_close(command[:6], torch.zeros(min(expected_dim, 6)), check_dtype=False)
    if expected_dim == 7:
        assert command[6] == 1.0


@pytest.mark.parametrize(
    ("enumerated", "expected_name", "expected_ids"),
    [
        # some HID backends report the kernel's combined name instead of the bare USB product string
        (
            [dict(_SPACEMOUSE_COMPACT, product_string="3Dconnexion SpaceMouse Compact")],
            "SpaceMouse Compact",
            (0x256F, 0xC635),
        ),
        # the libusb-based backend reports no product string unless the process may open the USB node
        ([dict(_SPACEMOUSE_COMPACT, product_string="")], "SpaceMouse Compact", (0x256F, 0xC635)),
        (
            [{"product_string": "SpaceNavigator", "vendor_id": 0x046D, "product_id": 0xC626}],
            "SpaceNavigator",
            (0x046D, 0xC626),
        ),
        ([{"product_string": "", "vendor_id": 0x046D, "product_id": 0xC626}], "SpaceNavigator", (0x046D, 0xC626)),
    ],
    ids=["prefixed_product_string", "missing_product_string", "spacenavigator", "spacenavigator_usb_id_only"],
)
def test_spacemouse_detection_by_usb_id(hid, enumerated, expected_name, expected_ids):
    """SpaceMouse detection must not rely on an exact product string match."""
    hid.enumerate.return_value = enumerated

    device = Se3SpaceMouse(Se3SpaceMouseCfg())

    # the resolved name selects the report layout used by the listener thread
    assert device._device_name == expected_name
    hid.device.return_value.open.assert_called_with(*expected_ids)


def test_spacemouse_skips_devices_that_cannot_be_opened(hid):
    """An inaccessible SpaceMouse must not hide a second one the user can actually open."""
    hid.enumerate.return_value = [
        {"product_string": "", "vendor_id": 0x256F, "product_id": 0xC635},
        {"product_string": "", "vendor_id": 0x256F, "product_id": 0xC62E},
    ]
    hid.device.return_value.open.side_effect = [OSError("open failed"), None]

    device = Se3SpaceMouse(Se3SpaceMouseCfg())

    assert device._device_name == "SpaceMouse Wireless"
    hid.device.return_value.open.assert_called_with(0x256F, 0xC62E)


@pytest.mark.parametrize(
    ("enumerated", "open_error", "expected_fragments"),
    [
        (
            [dict(_SPACEMOUSE_COMPACT, product_string="")],
            OSError("open failed"),
            ["SpaceMouse Compact", "/dev/bus/usb"],
        ),
        ([{"product_string": "", "vendor_id": 0x046D, "product_id": 0xC52F}], None, ["0x046d:0xc52f", "/dev/bus/usb"]),
    ],
    ids=["open_failure_reports_permissions", "not_found_lists_enumerated_devices"],
)
def test_spacemouse_discovery_errors(hid, enumerated, open_error, expected_fragments):
    """Discovery errors name the inaccessible or unsupported devices and hint at USB permissions."""
    hid.enumerate.return_value = enumerated
    hid.device.return_value.open.side_effect = open_error

    with pytest.raises(OSError) as exc_info:
        Se3SpaceMouse(Se3SpaceMouseCfg())

    for fragment in expected_fragments:
        assert fragment in str(exc_info.value)


@pytest.mark.parametrize("device_cls", [Se2SpaceMouse, Se3SpaceMouse])
def test_spacemouse_destructor_handles_partial_initialization(device_cls):
    """The destructor must tolerate construction failing before the listener thread exists."""
    device_cls.__new__(device_cls).__del__()


def test_haply_device(mocker):
    """The Haply device times out without both peripherals and streams position, orientation, buttons, and forces."""
    cfg = HaplyDeviceCfg(websocket_uri="ws://localhost:10001", pos_sensitivity=1.5, data_rate=250.0)
    device_mod = importlib.import_module("isaaclab.devices.haply.se3_haply")
    # never open a socket: the connection thread is replaced by a thread that is not alive
    mocker.patch.object(device_mod, "websockets")
    mocker.patch.object(device_mod, "asyncio")
    threading_mock = mocker.patch.object(device_mod, "threading")
    threading_mock.Thread.return_value.is_alive.return_value = False
    threading_mock.Lock.side_effect = threading.Lock
    time_mock = mocker.patch.object(device_mod, "time")
    time_mock.time.side_effect = [0.0, 0.1, 0.2, 0.3, 6.0]

    with pytest.raises(RuntimeError, match="Failed to connect both Inverse3 and VerseGrip devices"):
        HaplyDevice(cfg)

    # bypass the connection handshake and feed cached device data directly
    mocker.patch.object(
        device_mod.HaplyDevice, "_start_websocket_thread", lambda self: setattr(self, "connected", True)
    )
    time_mock.time.side_effect = [0.0, 0.1]
    haply = HaplyDevice(cfg)
    assert (haply.websocket_uri, haply.pos_sensitivity, haply.data_rate) == (cfg.websocket_uri, 1.5, 250.0)
    haply.cached_data.update(
        position=torch.tensor([0.1, 0.2, 0.3]).numpy(),
        quaternion=torch.tensor([0.0, 0.0, 1.0, 0.0]).numpy(),
        buttons={"a": True, "b": False, "c": False},
        inverse3_connected=True,
        versegrip_connected=True,
    )
    pressed = []
    haply.add_callback("a", lambda: pressed.append("a"))
    with pytest.raises(ValueError, match="Invalid button key"):
        haply.add_callback("d", lambda: None)

    command = haply.advance()
    # [pos * sensitivity, quaternion, buttons]; the rising edge of button "a" fires its callback once
    torch.testing.assert_close(command, torch.tensor([0.15, 0.3, 0.45, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]))
    haply.advance()
    assert pressed == ["a"]

    forces = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.8, -0.3], [0.1, 0.2, 0.3]])
    haply.push_force(forces, torch.tensor(1))
    assert haply.feedback_force == pytest.approx({"x": 0.5, "y": 0.8, "z": -0.3})
    # the selected forces are summed and clipped to the default 2.0 N limit
    haply.push_force(forces, torch.tensor([0, 2]))
    assert haply.feedback_force == pytest.approx({"x": 1.1, "y": 2.0, "z": 2.0})
    with pytest.raises(ValueError, match="No forces provided"):
        haply.push_force(torch.zeros(0, 3), torch.tensor([0]))

    haply.reset()
    assert haply.feedback_force == {"x": 0.0, "y": 0.0, "z": 0.0}
    haply.__del__()
