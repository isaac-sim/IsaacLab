# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Settings manager for Isaac Lab that works with or without Omniverse carb.settings.

This module provides a unified settings interface that can work in two modes:
1. Omniverse mode: Uses carb.settings when SimulationApp is launched
2. Standalone mode: Uses pure Python dictionary when running without Omniverse

This allows Isaac Lab to run visualizers like Rerun and Newton without requiring
the full Omniverse/SimulationApp stack.
"""

import sys
from typing import Any

# Key for storing singleton in sys.modules to survive module reloads (e.g., from Hydra)
_SINGLETON_KEY = "__isaaclab_settings_manager_singleton__"


class SettingsManager:
    """A settings manager that provides a carb.settings-like interface without requiring Omniverse.

    This class can work in two modes:
    - Standalone mode: Uses a Python dictionary to store settings
    - Omniverse mode: Delegates to carb.settings when available

    The interface is designed to be compatible with the carb.settings API.

    This is implemented as a singleton to ensure only one instance exists across the application,
    even when used in different execution contexts (e.g., Hydra). The singleton is stored in
    sys.modules to survive module reloads.
    """

    def __new__(cls):
        """Singleton pattern - always return the same instance, stored in sys.modules to survive reloads."""
        # Check if instance exists in sys.modules (survives module reloads)
        instance = sys.modules.get(_SINGLETON_KEY)

        if instance is None:
            instance = super().__new__(cls)
            sys.modules[_SINGLETON_KEY] = instance
            # Mark that this instance needs initialization
            instance._needs_init = True

        return instance

    def __init__(self):
        """Initialize the settings manager (only runs once due to singleton pattern)."""
        # Check if this instance needs initialization
        needs_init = getattr(self, "_needs_init", False)

        if not needs_init:
            return

        self._standalone_settings: dict[str, Any] = {}
        self._carb_settings = None
        self._use_carb = False
        self._needs_init = False

    @classmethod
    def instance(cls) -> "SettingsManager":
        """Get the singleton instance of the settings manager.

        Returns:
            The singleton SettingsManager instance
        """
        # Get instance from sys.modules (survives module reloads)
        instance = sys.modules.get(_SINGLETON_KEY)

        if instance is None:
            instance = cls()

        return instance

    def initialize_carb_settings(self):
        """Initialize carb.settings if SimulationApp has been launched.

        This should be called after SimulationApp is created to enable
        Omniverse mode. If not called, the manager operates in standalone mode.
        """
        try:
            import carb

            self._carb_settings = carb.settings.get_settings()
            self._use_carb = True
        except (ImportError, AttributeError):
            # carb not available or SimulationApp not launched - use standalone mode
            self._use_carb = False

    def set(self, path: str, value: Any) -> None:
        """Set a setting value at the given path.

        Args:
            path: The settings path (e.g., "/isaaclab/render/offscreen")
            value: The value to set
        """
        if self._use_carb and self._carb_settings is not None:
            # Delegate to carb.settings
            if isinstance(value, bool):
                self._carb_settings.set_bool(path, value)
            elif isinstance(value, int):
                self._carb_settings.set_int(path, value)
            elif isinstance(value, float):
                self._carb_settings.set_float(path, value)
            elif isinstance(value, str):
                self._carb_settings.set_string(path, value)
            else:
                # For other types, try generic set
                self._carb_settings.set(path, value)
        else:
            # Standalone mode - use dictionary
            self._standalone_settings[path] = value

    def get(self, path: str, default: Any = None) -> Any:
        """Get a setting value at the given path.

        Args:
            path: The settings path (e.g., "/isaaclab/render/offscreen")
            default: Default value to return if path doesn't exist

        Returns:
            The value at the path, or default if not found
        """
        if self._use_carb and self._carb_settings is not None:
            # Delegate to carb.settings
            value = self._carb_settings.get(path)
            return value if value is not None else default
        else:
            # Standalone mode - use dictionary
            return self._standalone_settings.get(path, default)

    def set_bool(self, path: str, value: bool) -> None:
        """Set a boolean setting value.

        Args:
            path: The settings path
            value: The boolean value to set
        """
        self.set(path, value)

    def set_int(self, path: str, value: int) -> None:
        """Set an integer setting value.

        Args:
            path: The settings path
            value: The integer value to set
        """
        self.set(path, value)

    def set_float(self, path: str, value: float) -> None:
        """Set a float setting value.

        Args:
            path: The settings path
            value: The float value to set
        """
        self.set(path, value)

    def set_string(self, path: str, value: str) -> None:
        """Set a string setting value.

        Args:
            path: The settings path
            value: The string value to set
        """
        self.set(path, value)

    @property
    def is_omniverse_mode(self) -> bool:
        """Check if the settings manager is using carb.settings (Omniverse mode).

        Returns:
            True if using carb.settings, False if using standalone mode
        """
        return self._use_carb


def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance.

    The SettingsManager is implemented as a singleton, so this function
    always returns the same instance. The singleton is stored in sys.modules
    to survive module reloads (e.g., from Hydra).

    Returns:
        The global SettingsManager instance
    """
    # Get instance from sys.modules (survives module reloads)
    instance = sys.modules.get(_SINGLETON_KEY)
    if instance is None:
        instance = SettingsManager()
    return instance


def initialize_carb_settings():
    """Initialize carb.settings integration for the global settings manager.

    This should be called after SimulationApp is created to enable
    Omniverse mode for the global settings manager.
    """
    manager = get_settings_manager()
    manager.initialize_carb_settings()
