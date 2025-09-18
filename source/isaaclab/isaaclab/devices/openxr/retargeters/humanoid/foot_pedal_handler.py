from dataclasses import dataclass
from enum import Enum
import os
import select
import struct
import threading
import time
import torch
from typing import NamedTuple

import omni.log

class PedalMode(Enum):
    """Enumeration of foot pedal operation modes."""

    FORWARD_MODE = 'forward'
    REVERSE_MODE = 'reverse'
    VERTICAL_MODE = 'vertical'


class ClickState(NamedTuple):
    """State tracking for pedal click detection."""

    is_pressed: bool
    press_start_time: float
    max_value_reached: float


class JoystickEvent(NamedTuple):
    """
    Joystick event structure matching Linux js_event.
    See: https://www.kernel.org/doc/html/v6.8/input/joydev/joystick-api.html#event-reading
    """

    # Event timestamp in microseconds
    time: int

    # Event value (signed 16-bit)
    value: int

    # Event type (axis=2, button=1, init=0x80)
    type: int

    # Axis/button number
    number: int


@dataclass
class FootPedalOutput:
    """Output from the foot pedal handler."""

    # Current mode: FORWARD_MODE, REVERSE_MODE, VERTICAL_MODE
    current_mode: PedalMode

    # Raw axis values for [L, R, Rz] in range of [-1.0, 1.0]
    raw_axis_values: torch.Tensor


class FootPedalHandler:
    """
    Handles analog foot pedal inputs and mode switching.

    Reads from a Linux joystick device (e.g., /dev/input/js0) with 3 axes:
    - Axis 0 (L): Left pedal press
    - Axis 1 (R): Right pedal press
    - Axis 2 (Rz): Rudder/Yaw movement

    The foot pedal provides analog input values in range [-32767, 32767].
    """

    # Linux joystick event structure: time(4) + value(2) + type(1) + number(1) = 8 bytes
    # https://www.kernel.org/doc/html/v6.8/input/joydev/joystick-api.html#event-reading
    _JS_EVENT_FMT = 'IhBB'
    _JS_EVENT_SIZE = struct.calcsize(_JS_EVENT_FMT)

    # Event types
    _JS_EVENT_BUTTON = 0x01
    _JS_EVENT_AXIS = 0x02
    _JS_EVENT_INIT = 0x80

    def __init__(
        self,
        *,
        device_path: str = '/dev/input/js0',
        foot_pedal_update_interval: float = 0.02,  # 50Hz default
    ):
        """Initialize the foot pedal handler.

        Args:
            device_path: Path to joystick device (default: '/dev/input/js0')
            foot_pedal_update_interval: Time interval in seconds between input checks
        """
        self.is_running = False
        self.input_thread = None
        self._device_path = device_path
        self._device_fd = None

        self._raw_axis_values = torch.zeros(3)
        self._vel_lock = threading.Lock()  # Lock for velocity tensor access

        # Velocity limits and configuration
        self._update_interval = foot_pedal_update_interval

        # Joystick axis ranges (Linux joystick API uses signed 16-bit)
        self._max_axis_value = 32767.0
        self._min_axis_value = -32767.0

        # Mode and click detection configuration (hard-coded values)
        # Pedal behavior: -1 = fully released, +1 = fully pressed
        self._click_press_threshold = 0.6  # Pedal value above this is considered "pressed"
        self._click_release_threshold = -0.2  # Pedal value below this is considered "released"
        self._click_max_duration = 0.3  # Maximum duration for a valid click (seconds)

        # Mode state tracking
        self._current_mode = PedalMode.FORWARD_MODE
        self._mode_lock = threading.Lock()  # Lock for mode state access

        # Click detection state for left (axis 0) and right (axis 1) pedals
        self._left_click_state = ClickState(
            is_pressed=False, press_start_time=0.0, max_value_reached=0.0
        )
        self._right_click_state = ClickState(
            is_pressed=False, press_start_time=0.0, max_value_reached=0.0
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # pylint: disable=unused-argument
        self.stop()

    def start(self):
        """Start listening for foot pedal input.

        This method opens the joystick device and starts a background thread to
        monitor joystick events.

        If the foot pedal handler is already running, this method does nothing.
        """
        if self.is_running:
            return

        try:
            # Open the joystick device
            if not os.path.exists(self._device_path):
                raise FileNotFoundError(f'Joystick device not found: {self._device_path}')

            self._device_fd = os.open(self._device_path, os.O_RDONLY | os.O_NONBLOCK)
            print(f'Opened joystick device: {self._device_path}')

            # Only mark as running after device is successfully opened
            self.is_running = True

            # Start the input thread
            self.input_thread = threading.Thread(
                target=self._read_input, name='FootPedalInputThread'
            )
            self.input_thread.daemon = True
            self.input_thread.start()
        except (IOError, OSError, PermissionError) as e:
            # Handle device setup errors
            omni.log.error(f'Failed to open joystick device {self._device_path}: {e}')
            # Clean up partially opened resources
            if self._device_fd is not None:
                try:
                    os.close(self._device_fd)
                except OSError:
                    pass
                self._device_fd = None
            raise  # Re-raise the exception after cleanup

    def stop(self):
        """Stop listening for foot pedal input.

        This method is safe to call multiple times and from any thread.
        """
        # Use a flag to prevent multiple threads from trying to stop at once
        if not self.is_running:
            return

        # Mark as not running first to signal threads to exit
        self.is_running = False

        # Close the joystick device
        if self._device_fd is not None:
            try:
                os.close(self._device_fd)
                print(f'Closed joystick device: {self._device_path}')
            except OSError as e:
                omni.log.warning(f'Failed to close joystick device: {e}')
            finally:
                self._device_fd = None

        # Clean up the input thread
        if self.input_thread and self.input_thread != threading.current_thread():
            self.input_thread.join(timeout=1.0)
            self.input_thread = None
        elif self.input_thread == threading.current_thread():
            # Called from within the input thread, just clear the reference
            self.input_thread = None

    def get_raw_axis_values(self) -> FootPedalOutput:
        """Get the current raw axis values and current mode.

        Returns:
            Tuple of (raw_axis_values, current_mode)
        """
        with self._vel_lock:
            raw_values = self._raw_axis_values.clone()
        with self._mode_lock:
            current_mode = self._current_mode

        return FootPedalOutput(raw_axis_values=raw_values, current_mode=current_mode)

    def _detect_click(self, axis_number: int, normalized_value: float) -> bool:
        """Detect if a pedal click occurred.

        A click is defined as: released -> pressed -> released within a time limit

        Args:
            axis_number: 0 for left pedal, 1 for right pedal
            normalized_value: Current normalized axis value [-1.0, 1.0]

        Returns:
            True if a valid click was detected, False otherwise
        """
        if axis_number not in [0, 1]:
            return False

        current_time = time.time()

        # Get the appropriate click state
        if axis_number == 0:
            click_state = self._left_click_state
        else:
            click_state = self._right_click_state

        # State machine for click detection
        if not click_state.is_pressed:
            # Check if pedal started moving from released state
            if normalized_value > self._click_release_threshold:
                # Start tracking press from the moment it leaves the release threshold
                new_state = ClickState(
                    is_pressed=True,
                    press_start_time=current_time,
                    max_value_reached=normalized_value,
                )
                if axis_number == 0:
                    self._left_click_state = new_state
                else:
                    self._right_click_state = new_state
        else:
            # Pedal is currently pressed, update max value
            max_value = max(click_state.max_value_reached, normalized_value)

            # Check if pedal was released (moved back toward released position)
            if normalized_value <= self._click_release_threshold:
                # Check if this constitutes a valid click
                press_duration = current_time - click_state.press_start_time
                valid_click = (
                    press_duration <= self._click_max_duration
                    and max_value >= self._click_press_threshold
                )

                # Reset click state
                reset_state = ClickState(
                    is_pressed=False, press_start_time=0.0, max_value_reached=0.0
                )
                if axis_number == 0:
                    self._left_click_state = reset_state
                else:
                    self._right_click_state = reset_state

                return valid_click
            else:
                # Update max value while still pressed
                updated_state = ClickState(
                    is_pressed=True,
                    press_start_time=click_state.press_start_time,
                    max_value_reached=max_value,
                )
                if axis_number == 0:
                    self._left_click_state = updated_state
                else:
                    self._right_click_state = updated_state

        return False

    def _handle_mode_transition(self, is_left_click: bool):
        """Handle mode transition based on pedal click.

        Args:
            is_left_click: True for left pedal click, False for right pedal click
        """
        with self._mode_lock:
            current_mode = self._current_mode
            new_mode = current_mode  # Default to no change

            if current_mode == PedalMode.FORWARD_MODE:
                if is_left_click:
                    new_mode = PedalMode.REVERSE_MODE
                else:  # right click
                    new_mode = PedalMode.VERTICAL_MODE
            elif current_mode == PedalMode.REVERSE_MODE:
                if is_left_click:
                    new_mode = PedalMode.FORWARD_MODE
                else:  # right click
                    new_mode = PedalMode.VERTICAL_MODE
            elif current_mode == PedalMode.VERTICAL_MODE:
                if is_left_click:
                    new_mode = PedalMode.REVERSE_MODE
                else:  # right click
                    new_mode = PedalMode.FORWARD_MODE

            if new_mode != current_mode:
                self._current_mode = new_mode

    def _read_input(self):
        """Read foot pedal input in a separate thread using non-blocking input."""
        while self.is_running:
            try:
                # Check that device is open
                if self._device_fd is None:
                    omni.log.error('Device file descriptor is None')
                    self.stop()
                    break

                # Use select to check if input is available (non-blocking)
                if select.select([self._device_fd], [], [], self._update_interval)[0]:
                    try:
                        # Read joystick events
                        data = os.read(self._device_fd, self._JS_EVENT_SIZE)
                        if len(data) == self._JS_EVENT_SIZE:
                            event = self._parse_joystick_event(data)
                            if event:
                                self._handle_joystick_event(event)
                    except (OSError, IOError) as e:
                        # Handle input errors gracefully
                        omni.log.error(f'Input error: {e}')
                        self.stop()
                        break
            except (OSError, IOError, ValueError, RuntimeError) as e:
                # Catch specific exceptions to ensure device state is cleaned up
                omni.log.error(f'Error in foot pedal handler: {e}')
                self.stop()
                break

    def _parse_joystick_event(self, data: bytes) -> JoystickEvent | None:
        """Parse raw joystick event data into a JoystickEvent.

        Args:
            data: Raw 8-byte joystick event data

        Returns:
            JoystickEvent if parsing succeeds, None otherwise
        """
        try:
            event_time, value, event_type, number = struct.unpack(self._JS_EVENT_FMT, data)
            return JoystickEvent(time=event_time, value=value, type=event_type, number=number)
        except struct.error as e:
            omni.log.warning(f'Failed to parse joystick event: {e}')
            return None

    def _handle_joystick_event(self, event: JoystickEvent):
        """Handle a parsed joystick event.

        Args:
            event: Parsed joystick event
        """
        if event.type & self._JS_EVENT_INIT:
            # Skip initialization events
            print(f'Initialization event: axis={event.number}, value={event.value}')
            self._handle_axis_event(event.number, event.value)
        elif event.type & self._JS_EVENT_AXIS:
            # Handle axis events
            self._handle_axis_event(event.number, event.value)
        elif event.type & self._JS_EVENT_BUTTON:
            # For foot pedals, we typically don't have buttons, but log if received
            omni.log.warning(f'Unexpected button event: button={event.number}')
        else:
            omni.log.warning(f'Unexpected event: type={event.type}, number={event.number}')

    def _handle_axis_event(self, axis_number: int, raw_value: int):
        """Handle joystick axis movement.

        Args:
            axis_number: Axis number (0=X, 1=Y, 2=Rz for rudder pedals)
            raw_value: Raw axis value in range [-32767, 32767]
        """
        # Normalize axis value to [-1.0, 1.0]
        normalized_value = max(-1.0, min(1.0, raw_value / self._max_axis_value))

        with self._vel_lock:
            if axis_number < 0 or axis_number > 2:
                omni.log.warning(f'Unexpected axis number: {axis_number}')
                return

            # Update raw axis values
            self._raw_axis_values[axis_number] = normalized_value

        # Check for clicks on left (axis 0) and right (axis 1) pedals
        if axis_number in [0, 1]:
            if self._detect_click(axis_number, normalized_value):
                # Handle mode transition
                is_left_click = axis_number == 0
                self._handle_mode_transition(is_left_click)
