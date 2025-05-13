# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import unittest
from unittest.mock import MagicMock, patch

# Import device classes to test
from isaaclab.devices import (
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


class TestDeviceConstructors(unittest.TestCase):
    """Test fixture for device classes constructors and basic functionality."""

    def setUp(self):
        """Set up tests."""
        # Create mock objects that will be used across tests
        self.carb_mock = MagicMock()
        self.omni_mock = MagicMock()
        self.appwindow_mock = MagicMock()
        self.keyboard_mock = MagicMock()
        self.gamepad_mock = MagicMock()
        self.input_mock = MagicMock()
        self.settings_mock = MagicMock()
        self.hid_mock = MagicMock()
        self.device_mock = MagicMock()

        # Set up the mocks to return appropriate objects
        self.omni_mock.appwindow.get_default_app_window.return_value = self.appwindow_mock
        self.appwindow_mock.get_keyboard.return_value = self.keyboard_mock
        self.appwindow_mock.get_gamepad.return_value = self.gamepad_mock
        self.carb_mock.input.acquire_input_interface.return_value = self.input_mock
        self.carb_mock.settings.get_settings.return_value = self.settings_mock

        # Mock keyboard event types
        self.carb_mock.input.KeyboardEventType.KEY_PRESS = 1
        self.carb_mock.input.KeyboardEventType.KEY_RELEASE = 2

        # Mock the SpaceMouse
        self.hid_mock.enumerate.return_value = [
            {"product_string": "SpaceMouse Compact", "vendor_id": 123, "product_id": 456}
        ]
        self.hid_mock.device.return_value = self.device_mock

        # Mock OpenXR
        self.xr_core_mock = MagicMock()
        self.message_bus_mock = MagicMock()
        self.singleton_mock = MagicMock()
        self.omni_mock.kit.xr.core.XRCore.get_singleton.return_value = self.singleton_mock
        self.singleton_mock.get_message_bus.return_value = self.message_bus_mock
        self.omni_mock.kit.xr.core.XRPoseValidityFlags.POSITION_VALID = 1
        self.omni_mock.kit.xr.core.XRPoseValidityFlags.ORIENTATION_VALID = 2

    def tearDown(self):
        """Clean up after tests."""
        # Clean up mock objects if needed
        pass

    """
    Test keyboard devices.
    """

    @patch.dict("sys.modules", {"carb": MagicMock(), "omni": MagicMock()})
    def test_se2keyboard_constructors(self):
        """Test constructor for Se2Keyboard."""
        # Test config-based constructor
        config = Se2KeyboardCfg(
            v_x_sensitivity=0.9,
            v_y_sensitivity=0.5,
            omega_z_sensitivity=1.2,
        )
        with patch("isaaclab.devices.keyboard.se2_keyboard.carb", self.carb_mock):
            with patch("isaaclab.devices.keyboard.se2_keyboard.omni", self.omni_mock):
                keyboard = Se2Keyboard(config)

                # Verify configuration was applied correctly
                self.assertEqual(keyboard.v_x_sensitivity, 0.9)
                self.assertEqual(keyboard.v_y_sensitivity, 0.5)
                self.assertEqual(keyboard.omega_z_sensitivity, 1.2)

                # Test advance() returns expected type
                result = keyboard.advance()
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape, (3,))  # (v_x, v_y, omega_z)

    @patch.dict("sys.modules", {"carb": MagicMock(), "omni": MagicMock()})
    def test_se3keyboard_constructors(self):
        """Test constructor for Se3Keyboard."""
        # Test config-based constructor
        config = Se3KeyboardCfg(
            pos_sensitivity=0.5,
            rot_sensitivity=0.9,
        )
        with patch("isaaclab.devices.keyboard.se3_keyboard.carb", self.carb_mock):
            with patch("isaaclab.devices.keyboard.se3_keyboard.omni", self.omni_mock):
                keyboard = Se3Keyboard(config)

                # Verify configuration was applied correctly
                self.assertEqual(keyboard.pos_sensitivity, 0.5)
                self.assertEqual(keyboard.rot_sensitivity, 0.9)

                # Test advance() returns expected type
                result = keyboard.advance()
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape, (7,))  # (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper)

    """
    Test gamepad devices.
    """

    @patch.dict("sys.modules", {"carb": MagicMock(), "omni": MagicMock()})
    def test_se2gamepad_constructors(self):
        """Test constructor for Se2Gamepad."""
        # Test config-based constructor
        config = Se2GamepadCfg(
            v_x_sensitivity=1.1,
            v_y_sensitivity=0.6,
            omega_z_sensitivity=1.2,
            dead_zone=0.02,
        )
        with patch("isaaclab.devices.gamepad.se2_gamepad.carb", self.carb_mock):
            with patch("isaaclab.devices.gamepad.se2_gamepad.omni", self.omni_mock):
                gamepad = Se2Gamepad(config)

                # Verify configuration was applied correctly
                self.assertEqual(gamepad.v_x_sensitivity, 1.1)
                self.assertEqual(gamepad.v_y_sensitivity, 0.6)
                self.assertEqual(gamepad.omega_z_sensitivity, 1.2)
                self.assertEqual(gamepad.dead_zone, 0.02)

                # Test advance() returns expected type
                result = gamepad.advance()
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape, (3,))  # (v_x, v_y, omega_z)

    @patch.dict("sys.modules", {"carb": MagicMock(), "omni": MagicMock()})
    def test_se3gamepad_constructors(self):
        """Test constructor for Se3Gamepad."""
        # Test config-based constructor
        config = Se3GamepadCfg(
            pos_sensitivity=1.1,
            rot_sensitivity=1.7,
            dead_zone=0.02,
        )
        with patch("isaaclab.devices.gamepad.se3_gamepad.carb", self.carb_mock):
            with patch("isaaclab.devices.gamepad.se3_gamepad.omni", self.omni_mock):
                gamepad = Se3Gamepad(config)

                # Verify configuration was applied correctly
                self.assertEqual(gamepad.pos_sensitivity, 1.1)
                self.assertEqual(gamepad.rot_sensitivity, 1.7)
                self.assertEqual(gamepad.dead_zone, 0.02)

                # Test advance() returns expected type
                result = gamepad.advance()
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape, (7,))  # (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper)

    """
    Test spacemouse devices.
    """

    @patch.dict("sys.modules", {"hid": MagicMock()})
    def test_se2spacemouse_constructors(self):
        """Test constructor for Se2SpaceMouse."""
        # Test config-based constructor
        config = Se2SpaceMouseCfg(
            v_x_sensitivity=0.9,
            v_y_sensitivity=0.5,
            omega_z_sensitivity=1.2,
        )
        with patch("isaaclab.devices.spacemouse.se2_spacemouse.hid", self.hid_mock):
            spacemouse = Se2SpaceMouse(config)

            # Verify configuration was applied correctly
            self.assertEqual(spacemouse.v_x_sensitivity, 0.9)
            self.assertEqual(spacemouse.v_y_sensitivity, 0.5)
            self.assertEqual(spacemouse.omega_z_sensitivity, 1.2)

            # Test advance() returns expected type
            self.device_mock.read.return_value = [1, 0, 0, 0, 0]
            result = spacemouse.advance()
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape, (3,))  # (v_x, v_y, omega_z)

    @patch.dict("sys.modules", {"hid": MagicMock()})
    def test_se3spacemouse_constructors(self):
        """Test constructor for Se3SpaceMouse."""
        # Test config-based constructor
        config = Se3SpaceMouseCfg(
            pos_sensitivity=0.5,
            rot_sensitivity=0.9,
        )
        with patch("isaaclab.devices.spacemouse.se3_spacemouse.hid", self.hid_mock):
            spacemouse = Se3SpaceMouse(config)

            # Verify configuration was applied correctly
            self.assertEqual(spacemouse.pos_sensitivity, 0.5)
            self.assertEqual(spacemouse.rot_sensitivity, 0.9)

            # Test advance() returns expected type
            self.device_mock.read.return_value = [1, 0, 0, 0, 0, 0, 0]
            result = spacemouse.advance()
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape, (7,))  # (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper)

    """
    Test OpenXR devices.
    """

    def test_openxr_constructors(self):
        """Test constructor for OpenXRDevice."""
        # Test config-based constructor with custom XrCfg
        xr_cfg = XrCfg(
            anchor_pos=(1.0, 2.0, 3.0),
            anchor_rot=(0.0, 0.1, 0.2),  # Using 3-tuple for rotation based on type hint
            near_plane=0.2,
        )
        config = OpenXRDeviceCfg(xr_cfg=xr_cfg)

        # Create mock retargeters
        mock_controller_retargeter = MagicMock()
        mock_head_retargeter = MagicMock()
        retargeters = [mock_controller_retargeter, mock_head_retargeter]

        with patch.dict(
            "sys.modules",
            {
                "carb": self.carb_mock,
                "omni.kit.xr.core": self.omni_mock.kit.xr.core,
                "isaacsim.core.prims": MagicMock(),
            },
        ):
            with patch("isaaclab.devices.openxr.openxr_device.XRCore", self.omni_mock.kit.xr.core.XRCore):
                with patch(
                    "isaaclab.devices.openxr.openxr_device.XRPoseValidityFlags",
                    self.omni_mock.kit.xr.core.XRPoseValidityFlags,
                ):
                    with patch("isaaclab.devices.openxr.openxr_device.SingleXFormPrim") as mock_single_xform:
                        # Configure the mock to return a string for prim_path
                        mock_instance = mock_single_xform.return_value
                        mock_instance.prim_path = "/XRAnchor"

                        # Create the device using the factory
                        device = OpenXRDevice(config)

                        # Verify the device was created successfully
                        self.assertEqual(device._xr_cfg, xr_cfg)

                        # Test with retargeters
                        device = OpenXRDevice(cfg=config, retargeters=retargeters)

                        # Verify retargeters were correctly assigned as a list
                        self.assertEqual(device._retargeters, retargeters)

                        # Test with config and retargeters
                        device = OpenXRDevice(cfg=config, retargeters=retargeters)

                        # Verify both config and retargeters were correctly assigned
                        self.assertEqual(device._xr_cfg, xr_cfg)
                        self.assertEqual(device._retargeters, retargeters)

                        # Test reset functionality
                        device.reset()

    """
    Test teleop device factory.
    """

    @patch.dict("sys.modules", {"carb": MagicMock(), "omni": MagicMock()})
    def test_create_teleop_device_basic(self):
        """Test creating devices using the teleop device factory."""
        # Create device configuration
        keyboard_cfg = Se3KeyboardCfg(pos_sensitivity=0.8, rot_sensitivity=1.2)

        # Create devices configuration dictionary
        devices_cfg = {"test_keyboard": keyboard_cfg}

        # Mock Se3Keyboard class
        with patch("isaaclab.devices.keyboard.se3_keyboard.carb", self.carb_mock):
            with patch("isaaclab.devices.keyboard.se3_keyboard.omni", self.omni_mock):
                # Create the device using the factory
                device = create_teleop_device("test_keyboard", devices_cfg)

                # Verify the device was created correctly
                self.assertIsInstance(device, Se3Keyboard)
                self.assertEqual(device.pos_sensitivity, 0.8)
                self.assertEqual(device.rot_sensitivity, 1.2)

    @patch.dict("sys.modules", {"carb": MagicMock(), "omni": MagicMock()})
    def test_create_teleop_device_with_callbacks(self):
        """Test creating device with callbacks."""
        # Create device configuration
        xr_cfg = XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(1.0, 0.0, 0.0, 0.0), near_plane=0.15)
        openxr_cfg = OpenXRDeviceCfg(xr_cfg=xr_cfg)

        # Create devices configuration dictionary
        devices_cfg = {"test_xr": openxr_cfg}

        # Create mock callbacks
        button_a_callback = MagicMock()
        button_b_callback = MagicMock()
        callbacks = {"button_a": button_a_callback, "button_b": button_b_callback}

        # Mock OpenXRDevice class and dependencies
        with patch.dict(
            "sys.modules",
            {
                "carb": self.carb_mock,
                "omni.kit.xr.core": self.omni_mock.kit.xr.core,
                "isaacsim.core.prims": MagicMock(),
            },
        ):
            with patch("isaaclab.devices.openxr.openxr_device.XRCore", self.omni_mock.kit.xr.core.XRCore):
                with patch(
                    "isaaclab.devices.openxr.openxr_device.XRPoseValidityFlags",
                    self.omni_mock.kit.xr.core.XRPoseValidityFlags,
                ):
                    with patch("isaaclab.devices.openxr.openxr_device.SingleXFormPrim") as mock_single_xform:
                        # Configure the mock to return a string for prim_path
                        mock_instance = mock_single_xform.return_value
                        mock_instance.prim_path = "/XRAnchor"

                        # Create the device using the factory
                        device = create_teleop_device("test_xr", devices_cfg, callbacks)

                        # Verify the device was created correctly
                        self.assertIsInstance(device, OpenXRDevice)

                        # Verify callbacks were registered
                        device.add_callback("button_a", button_a_callback)
                        device.add_callback("button_b", button_b_callback)
                        self.assertEqual(len(device._additional_callbacks), 2)

    @patch.dict("sys.modules", {"carb": MagicMock(), "omni": MagicMock()})
    def test_create_teleop_device_with_retargeters(self):
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
        with patch.dict(
            "sys.modules",
            {
                "carb": self.carb_mock,
                "omni.kit.xr.core": self.omni_mock.kit.xr.core,
                "isaacsim.core.prims": MagicMock(),
            },
        ):
            with patch("isaaclab.devices.openxr.openxr_device.XRCore", self.omni_mock.kit.xr.core.XRCore):
                with patch(
                    "isaaclab.devices.openxr.openxr_device.XRPoseValidityFlags",
                    self.omni_mock.kit.xr.core.XRPoseValidityFlags,
                ):
                    with patch("isaaclab.devices.openxr.openxr_device.SingleXFormPrim") as mock_single_xform:
                        # Mock retargeter classes
                        with patch("isaaclab.devices.openxr.retargeters.Se3AbsRetargeter"):
                            with patch("isaaclab.devices.openxr.retargeters.GripperRetargeter"):
                                # Configure the mock to return a string for prim_path
                                mock_instance = mock_single_xform.return_value
                                mock_instance.prim_path = "/XRAnchor"

                                # Create the device using the factory
                                device = create_teleop_device("test_xr", devices_cfg)

                                # Verify retargeters were created
                                self.assertEqual(len(device._retargeters), 2)

    def test_create_teleop_device_device_not_found(self):
        """Test error when device name is not found in configuration."""
        # Create devices configuration dictionary
        devices_cfg = {"keyboard": Se3KeyboardCfg()}

        # Try to create a non-existent device
        with self.assertRaises(ValueError) as context:
            create_teleop_device("gamepad", devices_cfg)

        # Verify the error message
        self.assertIn("Device 'gamepad' not found", str(context.exception))

    def test_create_teleop_device_unsupported_config(self):
        """Test error when device configuration type is not supported."""

        # Create a custom unsupported configuration class
        class UnsupportedCfg:
            pass

        # Create devices configuration dictionary with unsupported config
        devices_cfg = {"unsupported": UnsupportedCfg()}

        # Try to create a device with unsupported configuration
        with self.assertRaises(ValueError) as context:
            create_teleop_device("unsupported", devices_cfg)

        # Verify the error message
        self.assertIn("Unsupported device configuration type", str(context.exception))


if __name__ == "__main__":
    run_tests()
