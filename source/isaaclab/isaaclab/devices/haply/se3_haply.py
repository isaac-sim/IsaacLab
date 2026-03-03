# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Haply device controller for SE3 control with force feedback."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from ..device_base import DeviceBase, DeviceCfg
from ..retargeter_base import RetargeterBase


class HaplyDevice(DeviceBase):
    """A Haply device controller for sending SE(3) commands with force feedback.

    This class provides an interface to Haply robotic devices (Inverse3 + VerseGrip)
    for teleoperation. It communicates via WebSocket and supports:

    - Position tracking from Inverse3 device
    - Orientation and button inputs from VerseGrip device
    - Directional force feedback to Inverse3
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

    def __init__(self, cfg: HaplyDeviceCfg, retargeters: list[RetargeterBase] | None = None):
        """Initialize the Haply device interface.

        Args:
            cfg: Configuration object for Haply device settings.
            retargeters: Optional list of retargeting components that transform device data
                into robot commands. If None or empty, the device outputs its native data format.

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
        self.data_rate = cfg.data_rate
        self._sim_device = cfg.sim_device
        self.limit_force = cfg.limit_force

        # Device status (True only when both Inverse3 and VerseGrip are connected)
        self.connected = False
        self._connected_lock = threading.Lock()

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
        }

        self.data_lock = threading.Lock()

        # Force feedback
        self.feedback_force = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.force_lock = threading.Lock()

        self._additional_callbacks = dict()

        # Button state tracking
        self._prev_buttons = {"a": False, "b": False, "c": False}

        # Connection monitoring
        self.consecutive_timeouts = 0
        self.max_consecutive_timeouts = 10  # ~10 seconds at 1s timeout
        self.timeout_warning_issued = False

        # Start WebSocket connection
        self.running = True
        self._websocket_thread = None
        self._start_websocket_thread()

        # Wait for both devices to connect
        timeout = 5.0
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            with self._connected_lock:
                if self.connected:
                    break
            time.sleep(0.1)

        with self._connected_lock:
            if not self.connected:
                raise RuntimeError(f"Failed to connect both Inverse3 and VerseGrip devices within {timeout}s. ")

    def __del__(self):
        """Cleanup on deletion: shutdown WebSocket connection and background thread."""
        if not hasattr(self, "running") or not self.running:
            return

        self.running = False

        # Reset force feedback before closing
        if hasattr(self, "force_lock") and hasattr(self, "feedback_force"):
            with self.force_lock:
                self.feedback_force = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Explicitly wait for WebSocket thread to finish
        if hasattr(self, "_websocket_thread") and self._websocket_thread is not None:
            if self._websocket_thread.is_alive():
                self._websocket_thread.join(timeout=2.0)
                if self._websocket_thread.is_alive():
                    self._websocket_thread.daemon = True

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

    def reset(self):
        """Reset the device internal state."""
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
        with self.data_lock:
            if not (self.cached_data["inverse3_connected"] and self.cached_data["versegrip_connected"]):
                raise RuntimeError("Haply devices not connected. Both Inverse3 and VerseGrip must be connected.")

            # Safe copy within lock
            position = self.cached_data["position"].copy() * self.pos_sensitivity
            quaternion = self.cached_data["quaternion"].copy()
            button_a = self.cached_data["buttons"].get("a", False)
            button_b = self.cached_data["buttons"].get("b", False)
            button_c = self.cached_data["buttons"].get("c", False)

        # Button callbacks execute OUTSIDE lock to prevent deadlock
        for button_key, current_state in [("a", button_a), ("b", button_b), ("c", button_c)]:
            prev_state = self._prev_buttons.get(button_key, False)

            if current_state and not prev_state:
                if button_key in self._additional_callbacks:
                    self._additional_callbacks[button_key]()

            self._prev_buttons[button_key] = current_state

        button_states = np.array(
            [
                1.0 if button_a else 0.0,
                1.0 if button_b else 0.0,
                1.0 if button_c else 0.0,
            ],
            dtype=np.float32,
        )

        # Construct command tensor: [position(3), quaternion(4), buttons(3)]
        command = np.concatenate([position, quaternion, button_states])

        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def push_force(self, forces: torch.Tensor, position: torch.Tensor) -> None:
        """Push force vector to Haply Inverse3 device.

        Overrides DeviceBase.push_force() to provide force feedback for Haply Inverse3.
        Forces are clipped to [-limit_force, limit_force] range for safety.

        Args:
            forces: Tensor of shape (N, 3) with forces [fx, fy, fz].
            position: Tensor of shape (N) with indices specifying which forces to use.
        """
        # Check if forces is empty
        if forces.shape[0] == 0:
            raise ValueError("No forces provided")

        # Select forces using position indices
        selected_forces = forces[position] if position.ndim > 0 else forces[position].unsqueeze(0)
        force = selected_forces.sum(dim=0)
        force = force.cpu().numpy() if force.is_cuda else force.numpy()

        fx = np.clip(force[0], -self.limit_force, self.limit_force)
        fy = np.clip(force[1], -self.limit_force, self.limit_force)
        fz = np.clip(force[2], -self.limit_force, self.limit_force)

        with self.force_lock:
            self.feedback_force = {"x": float(fx), "y": float(fy), "z": float(fz)}

    def _start_websocket_thread(self):
        """Start WebSocket connection thread."""

        def websocket_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._websocket_loop())

        self._websocket_thread = threading.Thread(target=websocket_thread, daemon=False)
        self._websocket_thread.start()

    async def _websocket_loop(self):
        """WebSocket data reading and writing loop."""
        while self.running:
            try:
                async with websockets.connect(self.websocket_uri, ping_interval=None, ping_timeout=None) as ws:
                    first_message = True

                    while self.running:
                        try:
                            response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(response)

                            self.consecutive_timeouts = 0
                            if self.timeout_warning_issued:
                                self.timeout_warning_issued = False

                            # Safe array access - no IndexError risk with ternary operator
                            inverse3_list = data.get("inverse3", [])
                            verse_grip_list = data.get("wireless_verse_grip", [])
                            inverse3_data = inverse3_list[0] if inverse3_list else {}
                            verse_grip_data = verse_grip_list[0] if verse_grip_list else {}

                            if first_message:
                                first_message = False
                                if inverse3_data:
                                    self.inverse3_device_id = inverse3_data.get("device_id")
                                if verse_grip_data:
                                    self.verse_grip_device_id = verse_grip_data.get("device_id")

                            with self.data_lock:
                                inverse3_connected = False
                                versegrip_connected = False

                                if inverse3_data and "state" in inverse3_data:
                                    cursor_pos = inverse3_data["state"].get("cursor_position", {})
                                    if cursor_pos:
                                        self.cached_data["position"] = np.array(
                                            [cursor_pos.get(k, 0.0) for k in ("x", "y", "z")], dtype=np.float32
                                        )
                                        inverse3_connected = True

                                if verse_grip_data and "state" in verse_grip_data:
                                    state = verse_grip_data["state"]
                                    self.cached_data["buttons"] = {
                                        k: state.get("buttons", {}).get(k, False) for k in ("a", "b", "c")
                                    }
                                    orientation = state.get("orientation", {})
                                    if orientation:
                                        self.cached_data["quaternion"] = np.array(
                                            [
                                                orientation.get(k, 1.0 if k == "w" else 0.0)
                                                for k in ("x", "y", "z", "w")
                                            ],
                                            dtype=np.float32,
                                        )
                                    versegrip_connected = True

                                self.cached_data["inverse3_connected"] = inverse3_connected
                                self.cached_data["versegrip_connected"] = versegrip_connected
                                # Both devices required (AND logic): Inverse3 for position/force,
                                both_connected = inverse3_connected and versegrip_connected

                            with self._connected_lock:
                                self.connected = both_connected

                            # Send force feedback
                            if self.inverse3_device_id:
                                with self.force_lock:
                                    current_force = self.feedback_force.copy()

                                request_msg = {
                                    "inverse3": [
                                        {
                                            "device_id": self.inverse3_device_id,
                                            "commands": {"set_cursor_force": {"values": current_force}},
                                        }
                                    ]
                                }
                                await ws.send(json.dumps(request_msg))

                            await asyncio.sleep(1.0 / self.data_rate)

                        except asyncio.TimeoutError:
                            self.consecutive_timeouts += 1

                            # Check if timeout
                            if (
                                self.consecutive_timeouts >= self.max_consecutive_timeouts
                                and not self.timeout_warning_issued
                            ):
                                self.timeout_warning_issued = True
                                with self.data_lock:
                                    self.cached_data["inverse3_connected"] = False
                                    self.cached_data["versegrip_connected"] = False
                                with self._connected_lock:
                                    self.connected = False
                            continue
                        except Exception as e:
                            print(f"[ERROR] Error in WebSocket receive loop: {e}")
                            break

            except Exception:
                with self.data_lock:
                    self.cached_data["inverse3_connected"] = False
                    self.cached_data["versegrip_connected"] = False
                with self._connected_lock:
                    self.connected = False
                self.consecutive_timeouts = 0
                self.timeout_warning_issued = False

                if self.running:
                    await asyncio.sleep(2.0)
                else:
                    break


@dataclass
class HaplyDeviceCfg(DeviceCfg):
    """Configuration for Haply device.

    Attributes:
        websocket_uri: WebSocket URI for Haply SDK connection
        pos_sensitivity: Position sensitivity scaling factor
        data_rate: Data exchange rate in Hz
        limit_force: Maximum force magnitude in Newtons (safety limit)
    """

    websocket_uri: str = "ws://localhost:10001"
    pos_sensitivity: float = 1.0
    data_rate: float = 200.0
    limit_force: float = 2.0
    class_type: type[DeviceBase] = HaplyDevice
