# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider abstraction for physics backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pxr import Usd


class SceneDataProvider(ABC):
    """Abstract base class for providing scene data from physics backends to visualizers and renderers.
    
    This abstraction allows visualizers and renderers to work with any physics backend (Newton, PhysX, etc.)
    without directly coupling to specific physics manager implementations.
    """

    @abstractmethod
    def get_model(self) -> Any | None:
        """Get the physics model.
        
        Returns:
            Physics model object, or None if not available. The type depends on the backend
            (e.g., newton.Model for Newton backend).
        """
        pass

    @abstractmethod
    def get_state(self) -> Any | None:
        """Get the current physics state.
        
        Returns:
            Physics state object, or None if not available. The type depends on the backend
            (e.g., newton.State for Newton backend).
        """
        pass

    @abstractmethod
    def get_usd_stage(self) -> Usd.Stage | None:
        """Get the USD stage.
        
        Returns:
            USD stage, or None if not available.
        """
        pass

    def get_metadata(self) -> dict[str, Any]:
        """Get additional metadata about the scene.
        
        Returns:
            Dictionary containing optional metadata such as:
            - "physics_backend": str (e.g., "newton", "physx")
            - "num_envs": int
            - "device": str
            - etc.
        """
        return {}

    def get_scene_data(self) -> dict[str, Any]:
        """Get all available scene data as a dictionary.
        
        This is a convenience method that collects all scene data into a single dict.
        Individual visualizers can extract what they need.
        
        Returns:
            Dictionary containing all available scene data:
            - "model": Physics model
            - "state": Physics state
            - "usd_stage": USD stage
            - "metadata": Additional metadata
        """
        return {
            "model": self.get_model(),
            "state": self.get_state(),
            "usd_stage": self.get_usd_stage(),
            "metadata": self.get_metadata(),
        }

