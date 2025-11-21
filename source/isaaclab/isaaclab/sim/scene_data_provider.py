# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data provider for visualizers and renderers.

This module provides a unified interface for accessing scene data from different physics backends
and synchronizing that data to rendering backends (USD Fabric, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from newton import Model, State


class SceneDataProvider:
    """Unified scene data provider for all physics and rendering backends.
    
    This class provides:
    - Access to physics state and model for visualizers (Newton OpenGL, Rerun)
    - Synchronization of physics state to USD Fabric for OV visualizer
    - Conditional updates based on active visualizers to avoid unnecessary work
    
    The provider handles both Newton-specific data (state, model) and OV-specific
    synchronization (fabric transforms) within a single class to keep the abstraction
    simple while supporting multiple backends.
    """

    def __init__(self, visualizer_cfgs: list[Any] | None = None):
        """Initialize the scene data provider with visualizer configurations.
        
        Args:
            visualizer_cfgs: List of visualizer configurations to determine which backends are active.
        """
        self._has_newton_visualizer = False
        self._has_rerun_visualizer = False
        self._has_ov_visualizer = False
        self._is_initialized = False
        
        # Determine which visualizers are enabled from configs
        if visualizer_cfgs:
            for cfg in visualizer_cfgs:
                viz_type = getattr(cfg, 'visualizer_type', None)
                if viz_type == 'newton':
                    self._has_newton_visualizer = True
                elif viz_type == 'rerun':
                    self._has_rerun_visualizer = True
                elif viz_type == 'omniverse':
                    self._has_ov_visualizer = True
        
        self._is_initialized = True
    
    def update(self) -> None:
        """Update scene data for visualizers.
        
        This method:
        - Syncs Newton transforms to USD Fabric (if OV visualizer is active)
        - Can be extended for other backend-specific updates
        
        Note:
            Only performs work if necessary visualizers are active to avoid overhead.
        """
        if not self._is_initialized:
            return
        
        # Sync fabric transforms only if OV visualizer is active
        if self._has_ov_visualizer:
            self._sync_fabric_transforms()
    
    def get_state(self) -> State | None:
        """Get physics state for Newton-based visualizers.
        
        Returns:
            Newton State object, or None if not available.
        """
        # State is needed by Newton OpenGL and Rerun visualizers
        if not (self._has_newton_visualizer or self._has_rerun_visualizer):
            return None
        
        # Lazy import to avoid loading Newton if not needed
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager
            return NewtonManager._state_0
        except ImportError:
            return None
    
    def get_model(self) -> Model | None:
        """Get physics model for Newton-based visualizers.
        
        Returns:
            Newton Model object, or None if not available.
        """
        # Model is needed by Newton OpenGL and Rerun visualizers
        if not (self._has_newton_visualizer or self._has_rerun_visualizer):
            return None
        
        # Lazy import to avoid loading Newton if not needed
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager
            return NewtonManager._model
        except ImportError:
            return None
    
    def _sync_fabric_transforms(self) -> None:
        """Sync Newton transforms to USD Fabric for OV visualizer.
        
        This method calls NewtonManager.sync_fabric_transforms() to update the
        USD Fabric with the latest physics transforms. This allows the OV visualizer
        to render the scene without additional data transfer.
        
        Note:
            This is only called when OV visualizer is active to avoid unnecessary GPU work.
        """
        try:
            from isaaclab.sim._impl.newton_manager import NewtonManager
            NewtonManager.sync_fabric_transforms()
        except ImportError:
            pass

