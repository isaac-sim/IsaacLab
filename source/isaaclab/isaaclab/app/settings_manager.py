# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

from typing import Any


class SettingsManager:
    """A settings manager that provides a carb.settings-like interface without requiring Omniverse.
    
    This class can work in two modes:
    - Standalone mode: Uses a Python dictionary to store settings
    - Omniverse mode: Delegates to carb.settings when available
    
    The interface is designed to be compatible with the carb.settings API.
    """
    
    def __init__(self):
        """Initialize the settings manager."""
        self._standalone_settings: dict[str, Any] = {}
        self._carb_settings = None
        self._use_carb = False
        
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


# Global settings manager instance
_global_settings_manager: SettingsManager | None = None


def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance.
    
    Returns:
        The global SettingsManager instance
    """
    global _global_settings_manager
    if _global_settings_manager is None:
        _global_settings_manager = SettingsManager()
    return _global_settings_manager


def initialize_carb_settings():
    """Initialize carb.settings integration for the global settings manager.
    
    This should be called after SimulationApp is created to enable
    Omniverse mode for the global settings manager.
    """
    manager = get_settings_manager()
    manager.initialize_carb_settings()

