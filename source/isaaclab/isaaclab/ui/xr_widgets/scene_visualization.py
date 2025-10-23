# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import inspect
import numpy as np
import threading
import time
import torch
from collections.abc import Callable
from enum import Enum
from typing import Any, Union

import omni.log
from pxr import Gf

from isaaclab.sim import SimulationContext
from isaaclab.ui.xr_widgets import show_instruction


class TriggerType(Enum):
    """Enumeration of trigger types for visualization callbacks.

    Defines when callbacks should be executed:
    - TRIGGER_ON_EVENT: Execute when a specific event occurs
    - TRIGGER_ON_PERIOD: Execute at regular time intervals
    - TRIGGER_ON_CHANGE: Execute when a specific data variable changes
    - TRIGGER_ON_UPDATE: Execute every frame
    """

    TRIGGER_ON_EVENT = 0
    TRIGGER_ON_PERIOD = 1
    TRIGGER_ON_CHANGE = 2
    TRIGGER_ON_UPDATE = 3


class DataCollector:
    """Collects and manages data for visualization purposes.

    This class provides a centralized data store for visualization data,
    with change detection and callback mechanisms for real-time updates.
    """

    def __init__(self):
        """Initialize the data collector with empty data store and callback system."""
        self._data: dict[str, Any] = {}
        self._visualization_callback: Callable | None = None
        self._changed_flags: set[str] = set()

    def _values_equal(self, existing_value: Any, new_value: Any) -> bool:
        """Compare two values using appropriate method based on their types.

        Handles different data types including None, NumPy arrays, PyTorch tensors,
        and standard Python types for accurate change detection.

        Args:
            existing_value: The current value stored in the data collector
            new_value: The new value to compare against

        Returns:
            bool: True if values are equal, False otherwise
        """
        # If both are None or one is None
        if existing_value is None or new_value is None:
            return existing_value is new_value

        # If types are different, they're not equal
        if type(existing_value) is not type(new_value):
            return False

        # Handle NumPy arrays
        if isinstance(existing_value, np.ndarray):
            return np.array_equal(existing_value, new_value)

        # Handle torch tensors (if they exist)
        if hasattr(existing_value, "equal"):
            with contextlib.suppress(Exception):
                return torch.equal(existing_value, new_value)

        # For all other types (int, float, string, bool, list, dict, set), use regular equality
        with contextlib.suppress(Exception):
            return existing_value == new_value
        # If comparison fails for any reason, assume they're different
        return False

    def update_data(self, name: str, value: Any) -> None:
        """Update a data field and trigger change detection.

        This method handles data updates with intelligent change detection.
        It also performs pre-processing and post-processing based on the field name.

        Args:
            name: The name/key of the data field to update
            value: The new value to store (None to remove the field)
        """
        existing_value = self.get_data(name)

        if value is None:
            self._data.pop(name)
            if existing_value is not None:
                self._changed_flags.add(name)
            return

        # Todo: for list or array, the change won't be detected
        # Check if the value has changed using appropriate comparison method
        if self._values_equal(existing_value, value):
            return

        # Save it
        self._data[name] = value
        self._changed_flags.add(name)

    def update_loop(self) -> None:
        """Process pending changes and trigger visualization callbacks.

        This method should be called regularly to ensure visualization updates
        are processed in a timely manner.
        """
        if len(self._changed_flags) > 0:
            if self._visualization_callback:
                self._visualization_callback(self._changed_flags)
            self._changed_flags.clear()

    def get_data(self, name: str) -> Any:
        """Retrieve data by name.

        Args:
            name: The name/key of the data field to retrieve

        Returns:
            The stored value, or None if the field doesn't exist
        """
        return self._data.get(name)

    def set_visualization_callback(self, callback: Callable) -> None:
        """Set the VisualizationManager callback function to be called when data changes.

        Args:
            callback: Function to call when data changes, receives set of changed field names
        """
        self._visualization_callback = callback


class VisualizationManager:
    """Base class for managing visualization rules and callbacks.

    Provides a framework for registering and executing callbacks based on
    different trigger conditions (events, time periods, data changes).
    """

    # Type aliases for different callback signatures
    StandardCallback = Callable[["VisualizationManager", "DataCollector"], None]
    EventCallback = Callable[["VisualizationManager", "DataCollector", Any], None]
    CallbackType = Union[StandardCallback, EventCallback]

    class TimeCountdown:
        """Internal class for managing periodic timer-based callbacks."""

        period: float
        countdown: float
        last_time: float

        def __init__(self, period: float, initial_countdown: float = 0.0):
            """Initialize a countdown timer.

            Args:
                period: Time interval in seconds between callback executions
            """
            self.period = period
            self.countdown = initial_countdown
            self.last_time = time.time()

        def update(self, current_time: float) -> bool:
            """Update the countdown timer and check if callback should be triggered.

            Args:
                current_time: Current time in seconds

            Returns:
                bool: True if callback should be triggered, False otherwise
            """
            self.countdown -= current_time - self.last_time
            self.last_time = current_time
            if self.countdown <= 0.0:
                self.countdown = self.period
                return True
            return False

    # Widget presets for common visualization configurations
    @classmethod
    def message_widget_preset(cls) -> dict[str, Any]:
        """Get the message widget preset configuration.

        Returns:
            dict: Configuration dictionary for message widgets
        """
        return {
            "prim_path_source": "/_xr/stage/xrCamera",
            "translation": Gf.Vec3f(0, 0, -2),
            "display_duration": 3.0,
            "max_width": 2.5,
            "min_width": 1.0,
            "font_size": 0.1,
            "text_color": 0xFF00FFFF,
        }

    @classmethod
    def panel_widget_preset(cls) -> dict[str, Any]:
        """Get the panel widget preset configuration.

        Returns:
            dict: Configuration dictionary for panel widgets
        """
        return {
            "prim_path_source": "/XRAnchor",
            "translation": Gf.Vec3f(0, 2, 2),  # hard-coded temporarily
            "display_duration": 0.0,
            "font_size": 0.13,
            "max_width": 2,
            "min_width": 2,
        }

    def display_widget(self, text: str, name: str, args: dict[str, Any]) -> None:
        """Display a widget with the given text and configuration.

        Args:
            text: Text content to display in the widget
            name: Unique identifier for the widget. If duplicated, the old one will be removed from scene.
            args: Configuration dictionary for widget appearance and behavior
        """
        widget_config = args | {"text": text, "target_prim_path": name}
        show_instruction(**widget_config)

    def __init__(self, data_collector: DataCollector):
        """Initialize the visualization manager.

        Args:
            data_collector: DataCollector instance to access the data for visualization use.
        """
        self.data_collector: DataCollector = data_collector
        data_collector.set_visualization_callback(self.on_change)

        self._rules_on_period: dict[VisualizationManager.TimeCountdown, VisualizationManager.StandardCallback] = {}
        self._rules_on_event: dict[str, list[VisualizationManager.EventCallback]] = {}
        self._rules_on_change: dict[str, list[VisualizationManager.StandardCallback]] = {}
        self._rules_on_update: list[VisualizationManager.StandardCallback] = []

    # Todo: add support to registering same callbacks for different names
    def on_change(self, names: set[str]) -> None:
        """Handle data changes by executing registered callbacks.

        Args:
            names: Set of data field names that have changed
        """
        for name in names:
            callbacks = self._rules_on_change.get(name)
            if callbacks:
                # Create a copy of the list to avoid modification during iteration
                for callback in list(callbacks):
                    callback(self, self.data_collector)
        if len(names) > 0:
            self.on_event("default_event_has_change")

    def update_loop(self) -> None:
        """Update periodic timers and execute callbacks as needed.

        This method should be called regularly to ensure periodic callbacks
        are executed at the correct intervals.
        """

        # Create a copy of the list to avoid modification during iteration
        for callback in list(self._rules_on_update):
            callback(self, self.data_collector)

        current_time = time.time()
        # Create a copy of the items to avoid modification during iteration
        for timer, callback in list(self._rules_on_period.items()):
            triggered = timer.update(current_time)
            if triggered:
                callback(self, self.data_collector)

    def on_event(self, event: str, params: Any = None) -> None:
        """Handle events by executing registered callbacks.

        Args:
            event: Name of the event that occurred
        """
        callbacks = self._rules_on_event.get(event)
        if callbacks is None:
            return
        # Create a copy of the list to avoid modification during iteration
        for callback in list(callbacks):
            callback(self, self.data_collector, params)

    # Todo: better organization of callbacks
    def register_callback(self, trigger: TriggerType, arg: dict, callback: CallbackType) -> Any:
        """Register a callback function to be executed based on trigger conditions.

        Args:
            trigger: Type of trigger that should execute the callback
            arg: Dictionary containing trigger-specific parameters:
                - For TRIGGER_ON_PERIOD: {"period": float}
                - For TRIGGER_ON_EVENT: {"event_name": str}
                - For TRIGGER_ON_CHANGE: {"variable_name": str}
                - For TRIGGER_ON_UPDATE: {}
            callback: Function to execute when trigger condition is met
                - For TRIGGER_ON_EVENT: callback(manager: VisualizationManager, data_collector: DataCollector, event_params: Any)
                - For others: callback(manager: VisualizationManager, data_collector: DataCollector)

        Raises:
            TypeError: If callback signature doesn't match the expected signature for the trigger type
        """
        # Validate callback signature based on trigger type
        self._validate_callback_signature(trigger, callback)

        match trigger:
            case TriggerType.TRIGGER_ON_PERIOD:
                period = arg.get("period")
                initial_countdown = arg.get("initial_countdown", 0.0)
                if isinstance(period, float) and isinstance(initial_countdown, float):
                    timer = VisualizationManager.TimeCountdown(period=period, initial_countdown=initial_countdown)
                    # Type cast since we've validated the signature
                    self._rules_on_period[timer] = callback  # type: ignore
                    return timer
            case TriggerType.TRIGGER_ON_EVENT:
                event = arg.get("event_name")
                if isinstance(event, str):
                    callbacks = self._rules_on_event.get(event)
                    if callbacks is None:
                        # Type cast since we've validated the signature
                        self._rules_on_event[event] = [callback]  # type: ignore
                    else:
                        # Type cast since we've validated the signature
                        self._rules_on_event[event].append(callback)  # type: ignore
                    return event
            case TriggerType.TRIGGER_ON_CHANGE:
                variable_name = arg.get("variable_name")
                if isinstance(variable_name, str):
                    callbacks = self._rules_on_change.get(variable_name)
                    if callbacks is None:
                        # Type cast since we've validated the signature
                        self._rules_on_change[variable_name] = [callback]  # type: ignore
                    else:
                        # Type cast since we've validated the signature
                        self._rules_on_change[variable_name].append(callback)  # type: ignore
                    return variable_name
            case TriggerType.TRIGGER_ON_UPDATE:
                # Type cast since we've validated the signature
                self._rules_on_update.append(callback)  # type: ignore
        return None

    # Todo: better callback-cancel method
    def cancel_rule(self, trigger: TriggerType, arg: str | TimeCountdown, callback: Callable | None = None) -> None:
        """Remove a previously registered callback.

        Periodic callbacks are not supported to be cancelled for now.

        Args:
            trigger: Type of trigger for the callback to remove
            arg: Trigger-specific identifier (event name or variable name)
            callback: The callback function to remove
        """
        callbacks = None
        match trigger:
            case TriggerType.TRIGGER_ON_CHANGE:
                callbacks = self._rules_on_change.get(arg)
            case TriggerType.TRIGGER_ON_EVENT:
                callbacks = self._rules_on_event.get(arg)
            case TriggerType.TRIGGER_ON_PERIOD:
                self._rules_on_period.pop(arg)
            case TriggerType.TRIGGER_ON_UPDATE:
                callbacks = self._rules_on_update
        if callbacks is not None:
            if callback is not None:
                callbacks.remove(callback)
            else:
                callbacks.clear()

    def set_attr(self, name: str, value: Any) -> None:
        """Set an attribute of the visualization manager.

        Args:
            name: Name of the attribute to set
            value: Value to set the attribute to
        """
        setattr(self, name, value)

    def _validate_callback_signature(self, trigger: TriggerType, callback: Callable) -> None:
        """Validate that the callback has the correct signature for the trigger type.

        Args:
            trigger: Type of trigger for the callback
            callback: The callback function to validate

        Raises:
            TypeError: If callback signature doesn't match expected signature
        """
        try:
            sig = inspect.signature(callback)
            params = list(sig.parameters.values())

            # Remove 'self' parameter if it's a bound method
            if params and params[0].name == "self":
                params = params[1:]

            param_count = len(params)

            if trigger == TriggerType.TRIGGER_ON_EVENT:
                # Event callbacks should have 3 parameters: (manager, data_collector, event_params)
                expected_count = 3
                expected_sig = (
                    "callback(manager: VisualizationManager, data_collector: DataCollector, event_params: Any)"
                )
            else:
                # Other callbacks should have 2 parameters: (manager, data_collector)
                expected_count = 2
                expected_sig = "callback(manager: VisualizationManager, data_collector: DataCollector)"

            if param_count != expected_count:
                raise TypeError(
                    f"Callback for {trigger.name} must have {expected_count} parameters, "
                    f"but got {param_count}. Expected signature: {expected_sig}. "
                    f"Actual signature: {sig}"
                )

        except Exception as e:
            if isinstance(e, TypeError):
                raise
            # If we can't inspect the signature (e.g., built-in functions),
            # just log a warning and proceed
            omni.log.warn(f"Could not validate callback signature for {trigger.name}: {e}")


class XRVisualization:
    """Singleton class providing XR visualization functionality.

    This class implements the singleton pattern to ensure only one instance
    of the visualization system exists across the application. It provides
    a centralized API for managing XR visualization features.

    When manage a new event ordata field, please add a comment to the following list.

    Event names:
        "ik_solver_failed"

    Data fields:
        "manipulability_ellipsoid" : list[float]
        "device_raw_data" : dict
        "joints_distance_percentage_to_limit" : list[float]
        "joints_torque" : list[float]
        "joints_torque_limit" : list[float]
        "joints_name" : list[str]
        "wrist_pose" : list[float]
        "approximated_working_space" : list[float]
        "hand_torque_mapping" : list[str]
    """

    _lock = threading.Lock()
    _instance: XRVisualization | None = None
    _registered = False

    def __init__(self):
        """Prevent direct instantiation."""
        raise RuntimeError("Use VisualizationInterface classmethods instead of direct instantiation")

    @classmethod
    def __create_instance(cls, manager: type[VisualizationManager] = VisualizationManager) -> XRVisualization:
        """Get the visualization manager instance.

        Returns:
            VisualizationManager: The visualization manager instance
        """
        with cls._lock:
            if cls._instance is None:
                # Bypass __init__ by calling __new__ directly
                cls._instance = super().__new__(cls)
                cls._instance._initialize(manager)
        return cls._instance

    @classmethod
    def __get_instance(cls) -> XRVisualization:
        """Thread-safe singleton access.

        Returns:
            XRVisualization: The singleton instance of the visualization system
        """
        if cls._instance is None:
            return cls.__create_instance()
        elif not cls._instance._registered:
            cls._instance._register()
        return cls._instance

    def _register(self) -> bool:
        """Register the visualization system.

        Returns:
            bool: True if the visualization system is registered, False otherwise
        """
        if self._registered:
            return True

        sim = SimulationContext.instance()
        if sim is not None:
            sim.add_render_callback("visualization_render_callback", self.update_loop)
            self._registered = True
        return self._registered

    def _initialize(self, manager: type[VisualizationManager]) -> None:
        """Initialize the singleton instance with data collector and visualization manager."""

        self._data_collector = DataCollector()
        self._visualization_manager = manager(self._data_collector)

        self._register()

        self._initialized = True

    # APIs

    def update_loop(self, event) -> None:
        """Update the visualization system.

        This method should be called regularly (e.g., every frame) to ensure
        visualization updates are processed and periodic callbacks are executed.
        """
        self._visualization_manager.update_loop()
        self._data_collector.update_loop()

    @classmethod
    def push_event(cls, name: str, args: Any = None) -> None:
        """Push an event to trigger registered callbacks.

        Args:
            name: Name of the event to trigger
            args: Optional arguments for the event (currently unused)
        """
        instance = cls.__get_instance()
        instance._visualization_manager.on_event(name, args)

    @classmethod
    def push_data(cls, item: dict[str, Any]) -> None:
        """Push data to the visualization system.

        Updates multiple data fields at once. Each key-value pair in the
        dictionary will be processed by the data collector.

        Args:
            item: Dictionary containing data field names and their values
        """
        instance = cls.__get_instance()
        for name, value in item.items():
            instance._data_collector.update_data(name, value)

    @classmethod
    def set_attrs(cls, attributes: dict[str, Any]) -> None:
        """Set configuration data for the visualization system. Not currently used.

        Args:
            attributes: Dictionary containing configuration keys and values
        """

        instance = cls.__get_instance()
        for name, data in attributes.items():
            instance._visualization_manager.set_attr(name, data)

    @classmethod
    def get_attr(cls, name: str) -> Any:
        """Get configuration data for the visualization system. Not currently used.

        Args:
            name: Configuration key
        """
        instance = cls.__get_instance()
        return getattr(instance._visualization_manager, name)

    @classmethod
    def register_callback(cls, trigger: TriggerType, arg: dict, callback: VisualizationManager.CallbackType) -> None:
        """Register a callback function for visualization events.

        Args:
            trigger: Type of trigger that should execute the callback
            arg: Dictionary containing trigger-specific parameters:
                - For TRIGGER_ON_PERIOD: {"period": float}
                - For TRIGGER_ON_EVENT: {"event_name": str}
                - For TRIGGER_ON_CHANGE: {"variable_name": str}
            callback: Function to execute when trigger condition is met
        """
        instance = cls.__get_instance()
        instance._visualization_manager.register_callback(trigger, arg, callback)

    @classmethod
    def assign_manager(cls, manager: type[VisualizationManager]) -> None:
        """Assign a visualization manager type to the visualization system.

        Args:
            manager: Type of the visualization manager to assign
        """
        if cls._instance is not None:
            omni.log.error(
                f"Visualization system already initialized to {type(cls._instance._visualization_manager).__name__},"
                f" cannot assign manager {manager.__name__}"
            )
            return

        cls.__create_instance(manager)
