# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-specific scene data provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .scene_data_provider import SceneDataProvider

if TYPE_CHECKING:
    from pxr import Usd


class NewtonSceneDataProvider(SceneDataProvider):
    """Scene data provider for Newton physics backend.
    
    This class provides access to Newton physics data without directly exposing
    the NewtonManager to downstream consumers.
    """

    def __init__(self, usd_stage: Usd.Stage | None = None):
        """Initialize the Newton scene data provider.
        
        Args:
            usd_stage: USD stage reference, if available.
        """
        self._usd_stage = usd_stage

    def get_model(self) -> Any | None:
        """Get the Newton physics model.
        
        Returns:
            newton.Model instance, or None if not initialized.
        """
        from isaaclab.sim._impl.newton_manager import NewtonManager

        return NewtonManager._model

    def get_state(self) -> Any | None:
        """Get the current Newton physics state.
        
        Returns:
            newton.State instance, or None if not initialized.
        """
        from isaaclab.sim._impl.newton_manager import NewtonManager

        return NewtonManager._state_0

    def get_usd_stage(self) -> Usd.Stage | None:
        """Get the USD stage.
        
        Returns:
            USD stage, or None if not available.
        """
        return self._usd_stage

    def get_metadata(self) -> dict[str, Any]:
        """Get Newton-specific metadata.
        
        Returns:
            Dictionary containing:
            - "physics_backend": "newton"
            - "gravity_vector": tuple[float, float, float]
            - "clone_physics_only": bool
        """
        from isaaclab.sim._impl.newton_manager import NewtonManager

        return {
            "physics_backend": "newton",
            "gravity_vector": NewtonManager._gravity_vector,
            "clone_physics_only": NewtonManager._clone_physics_only,
        }

    def update_stage(self, usd_stage: Usd.Stage | None) -> None:
        """Update the USD stage reference.
        
        Args:
            usd_stage: New USD stage reference.
        """
        self._usd_stage = usd_stage

