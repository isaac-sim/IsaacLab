# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Haply device controller for SE3 control with force feedback."""

import asyncio
import json
import numpy as np
import threading
import time
import torch
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import websockets

WEBSOCKETS_AVAILABLE = True

from ..device_base import DeviceBase, DeviceCfg


@dataclass
class HaplyDeviceCfg(DeviceCfg):
    """Configuration for Haply device.

    Attributes:
        websocket_uri: WebSocket URI for Haply SDK connection
        pos_sensitivity: Position sensitivity scaling factor
        connection_timeout: Timeout in seconds for data freshness check
        data_rate: Data exchange rate in Hz
    """

    websocket_uri: str = "ws://localhost:10001"
    pos_sensitivity: float = 1.0
    connection_timeout: float = 2.0
    data_rate: float = 200.0


class HaplyDevice(DeviceBase):
    """A Haply device controller for sending SE(3) commands with force feedback.

    This class provides an interface to Haply robotic devices (Inverse3 + VerseGrip)
    for teleoperation. It communicates via WebSocket and supports:

    - Position tracking from Inverse3 device
    - Orientation and button inputs from VerseGrip device
    - Bidirectional force feedback to Inverse3
    - Real-time data streaming at configurable rates

    The device provides raw data:

    * Position: 3D position (x, y, z) in meters from Inverse3
    * Orientation: Quaternion (x, y, z, w) from VerseGrip
    * Buttons: Three buttons (a, b, c) from VerseGrip with state (pressed/not pressed)

    Note: All button logic (e.g., gripper control, reset, mode switching) should be
    implemented in the application layer using the raw button states from advance().

    Note:
        Requires the Haply SDK to be running and accessible via WebSocket.
        Install dependencies: pip install websockets

    """

    def __init__(self, cfg: HaplyDeviceCfg, retargeters=None):
        """Initialize the Haply device interface.

        Args:
            cfg: Configuration object for Haply device settings.
            retargeters: Optional list of retargeting components (not used in basic implementation).

        Raises:
            ImportError: If websockets module is not installed.
            RuntimeError: If connection to Haply device fails.
        """
        super().__init__(retargeters)

        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets module is required for Haply device. Install with: pip install websockets")

        # Store configuration
        self.websocket_uri = cfg.websocket_uri
        self.pos_sensitivity = cfg.pos_sensitivity
        self.connection_timeout = cfg.connection_timeout
        self.data_rate = cfg.data_rate
        self._sim_device = cfg.sim_device

        # Device status
        self.device_ready = False
        self.connected = False

        # Device IDs (will be set after first message)
        self.inverse3_device_id = None
        self.verse_grip_device_id = None

        # Current data cache
        self.cached_data = {
            "position": np.zeros(3, dtype=np.float32),
            "quaternion": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "buttons": {"a": False, "b": False, "c": False},
            "inverse3_connected": False,
            "versegrip_connected": False,
            "timestamp": 0.0,
        }

        self.data_lock = threading.Lock()
        self.last_data_time = 0.0

        # Force feedback
        self.feedback_force = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.force_lock = threading.Lock()

        self._additional_callbacks = dict()

        self._prev_buttons = {"a": False, "b": False, "c": False}

        # Start WebSocket connection
        self.running = True
        self._start_websocket_thread()

        # Wait for initial connection
        timeout = 5.0
        start_time = time.time()
        while not self.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self.connected:
            self.device_ready = True
            print(f"[INFO] Haply device ready: {self}")
        else:
            raise RuntimeError(
                f"Failed to connect to Haply WebSocket at {self.websocket_uri} "
                f"within {timeout}s. Is the Haply SDK running?"
            )

    def __del__(self):
        """Cleanup on deletion."""
        self.running = False
        with self.force_lock:
            self.feedback_force = {"x": 0.0, "y": 0.0, "z": 0.0}
        print("[INFO] Haply device disconnected")

    def __str__(self) -> str:
        """Returns: A string containing the information of the device."""
        msg = f"Haply Device Controller: {self.__class__.__name__}\n"
        msg += f"\tWebSocket URI: {self.websocket_uri}\n"
        msg += f"\tInverse3 ID: {self.inverse3_device_id}\n"
        msg += f"\tVerseGrip ID: {self.verse_grip_device_id}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tOutput: [x, y, z, qx, qy, qz, qw, btn_a, btn_b, btn_c]\n"
        msg += "\tInverse3: Provides position (x, y, z) and force feedback\n"
        msg += "\tVerseGrip: Provides orientation (quaternion) and buttons (a, b, c)"
        return msg

    """
    Operations
    """

    def reset(self):
        """Reset the device internal state."""
        # Reset force feedback
        with self.force_lock:
            self.feedback_force = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Reset button state tracking
        self._prev_buttons = {"a": False, "b": False, "c": False}

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind to button events.

        Args:
            key: The button to check against. Valid values are "a", "b", "c".
            func: The function to call when button is pressed. The callback function should not
                take any arguments.
        """
        if key not in ["a", "b", "c"]:
            raise ValueError(f"Invalid button key: {key}. Valid keys are 'a', 'b', 'c'.")
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from Haply device state.

        Returns:
            torch.Tensor: A tensor containing the raw device data:
                - 10 elements: [x, y, z, qx, qy, qz, qw, button_a, button_b, button_c]
                    where (x, y, z) is position, (qx, qy, qz, qw) is quaternion orientation,
                    and buttons are 1.0 (pressed) or 0.0 (not pressed)
        """
        if not self.device_ready:
            # Return zero command if device not ready
            return torch.zeros(10, dtype=torch.float32, device=self._sim_device)

        # Get latest cached data
        with self.data_lock:
            position = self.cached_data["position"].copy()
            quaternion = self.cached_data["quaternion"].copy()
            buttons = self.cached_data["buttons"].copy()

        # Apply sensitivity scaling
        position = position * self.pos_sensitivity

        # Check for button press events (rising edge) and call callbacks
        for button_key in ["a", "b", "c"]:
            current_state = buttons.get(button_key, False)
            prev_state = self._prev_buttons.get(button_key, False)

            # Detect rising edge (button just pressed)
            if current_state and not prev_state:
                # Call the callback if registered
                if button_key in self._additional_callbacks:
                    self._additional_callbacks[button_key]()

            # Update previous state
            self._prev_buttons[button_key] = current_state

        # Convert button states to floats
        button_states = np.array(
            [
                1.0 if buttons.get("a", False) else 0.0,
                1.0 if buttons.get("b", False) else 0.0,
                1.0 if buttons.get("c", False) else 0.0,
            ],
            dtype=np.float32,
        )

        # Construct command tensor: [position(3), quaternion(4), buttons(3)]
        command = np.concatenate([position, quaternion, button_states])

        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def set_force_feedback(self, force_x: float, force_y: float, force_z: float):
        """Set force feedback to be sent to Haply Inverse3 device.

        Args:
            force_x: Force in X direction (N)
            force_y: Force in Y direction (N)
            force_z: Force in Z direction (N)
        """
        with self.force_lock:
            self.feedback_force = {
                "x": float(force_x),
                "y": float(force_y),
                "z": float(force_z),
            }

    def get_device_state(self) -> dict[str, Any]:
        """Get current raw device state.

        Returns:
            Dictionary containing:
                - position: [x, y, z] from Inverse3
                - quaternion: [x, y, z, w] from VerseGrip
                - buttons: {'a': bool, 'b': bool, 'c': bool}
                - connected: bool indicating if device is connected
                - data_fresh: bool indicating if data is recent
        """
        with self.data_lock:
            current_data = self.cached_data.copy()

        data_fresh = self._is_data_fresh()
        device_connected = (
            current_data.get("inverse3_connected", False) or current_data.get("versegrip_connected", False)
        ) and data_fresh

        return {
            "position": current_data["position"],
            "quaternion": current_data["quaternion"],
            "buttons": current_data["buttons"],
            "connected": device_connected,
            "data_fresh": data_fresh,
        }

    """
    Internal helpers
    """

    def _is_data_fresh(self) -> bool:
        """Check if data is fresh (connection is normal)."""
        return (time.time() - self.last_data_time) < self.connection_timeout

    def _start_websocket_thread(self):
        """Start WebSocket connection thread."""

        def websocket_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._websocket_loop())

        thread = threading.Thread(target=websocket_thread, daemon=True)
        thread.start()
        print(f"[INFO] Haply WebSocket thread started for {self.websocket_uri}")

    async def _websocket_loop(self):
        """WebSocket data reading and writing loop."""
        print(f"[INFO] Connecting to Haply WebSocket at {self.websocket_uri}...")

        while self.running:
            try:
                async with websockets.connect(self.websocket_uri, ping_interval=None, ping_timeout=None) as ws:
                    print("[INFO] Connected to Haply WebSocket")
                    self.connected = True
                    first_message = True

                    while self.running:
                        try:
                            # Receive data from WebSocket
                            response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(response)

                            # Get devices list from the data
                            inverse3_devices = data.get("inverse3", [])
                            verse_grip_devices = data.get("wireless_verse_grip", [])

                            # Get the first device from each list
                            inverse3_data = inverse3_devices[0] if inverse3_devices else {}
                            verse_grip_data = verse_grip_devices[0] if verse_grip_devices else {}

                            # Handle the first message to get device IDs
                            if first_message:
                                first_message = False

                                if inverse3_data:
                                    self.inverse3_device_id = inverse3_data.get("device_id")
                                    print(f"[INFO] Inverse3 device ID: {self.inverse3_device_id}")
                                else:
                                    print("[WARNING] No Inverse3 device found")

                                if verse_grip_data:
                                    self.verse_grip_device_id = verse_grip_data.get("device_id")
                                    print(f"[INFO] VerseGrip device ID: {self.verse_grip_device_id}")
                                else:
                                    print("[WARNING] No VerseGrip device found")

                            # Update cached data
                            with self.data_lock:
                                # Update position from Inverse3
                                if inverse3_data and "state" in inverse3_data:
                                    cursor_pos = inverse3_data["state"].get("cursor_position", {})
                                    if cursor_pos:
                                        self.cached_data["position"] = np.array(
                                            [
                                                cursor_pos.get("x", 0.0),
                                                cursor_pos.get("y", 0.0),
                                                cursor_pos.get("z", 0.0),
                                            ],
                                            dtype=np.float32,
                                        )
                                        self.cached_data["inverse3_connected"] = True

                                # Update orientation and buttons from VerseGrip
                                if verse_grip_data and "state" in verse_grip_data:
                                    state = verse_grip_data["state"]

                                    # Update buttons
                                    buttons_raw = state.get("buttons", {})
                                    self.cached_data["buttons"] = {
                                        "a": buttons_raw.get("a", False),
                                        "b": buttons_raw.get("b", False),
                                        "c": buttons_raw.get("c", False),
                                    }

                                    # Update quaternion
                                    orientation = state.get("orientation", {})
                                    if orientation:
                                        self.cached_data["quaternion"] = np.array(
                                            [
                                                orientation.get("x", 0.0),
                                                orientation.get("y", 0.0),
                                                orientation.get("z", 0.0),
                                                orientation.get("w", 1.0),
                                            ],
                                            dtype=np.float32,
                                        )

                                    self.cached_data["versegrip_connected"] = True

                                self.cached_data["timestamp"] = time.time()
                                self.last_data_time = time.time()

                            # Send force feedback command
                            if self.inverse3_device_id:
                                # Get current force feedback
                                with self.force_lock:
                                    current_force = self.feedback_force.copy()

                                request_msg = {
                                    "inverse3": [{
                                        "device_id": self.inverse3_device_id,
                                        "commands": {"set_cursor_force": {"values": current_force}},
                                    }]
                                }
                                await ws.send(json.dumps(request_msg))

                            # Control data rate
                            await asyncio.sleep(1.0 / self.data_rate)

                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            print(f"[ERROR] Error in WebSocket receive loop: {e}")
                            break

            except Exception as e:
                print(f"[ERROR] WebSocket connection error: {e}")
                self.connected = False

                if self.running:
                    print("[INFO] Reconnecting in 2 seconds...")
                    await asyncio.sleep(2.0)
                else:
                    break
