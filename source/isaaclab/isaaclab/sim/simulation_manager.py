# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for simulation managers.

This module defines the interface that any simulation manager implementation must follow.
Backend-specific implementations (e.g., Omniverse) should inherit from SimulationManagerBase.
"""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .simulation_events import SimulationEvent

class SimulationManagerBase(ABC):
    """Abstract base class for simulation managers.

    This class defines the contract for managing physics simulation lifecycle,
    callbacks, and physics views. Backend-specific implementations should inherit
    from this class and implement all abstract methods.

    The manager is designed as a class-level (singleton-like) pattern where all
    methods are classmethods operating on class-level state.

    Attributes:
        _backend: Tensor backend ("torch" or "warp") for physics views.
        _warmup_needed: Whether physics warmup is needed before stepping.
        _view_created: Whether simulation views have been created.
        _assets_loaded: Whether all assets have finished loading.
        _callbacks: Registry of user-registered callbacks (id -> subscription).
        _callback_id: Counter for generating unique callback IDs.
        _handles: Named internal handles for subscriptions.
    """

    # ------------------------------------------------------------------
    # Shared State (backend-agnostic)
    # ------------------------------------------------------------------

    _backend: str = "torch"
    _warmup_needed: bool = True
    _view_created: bool = False
    _assets_loaded: bool = True

    # Callback registry: id -> subscription handle
    _callbacks: dict[int, Any] = {}
    _callback_id: int = 0
    _handles: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Tensor Backend (shared across backends)
    # ------------------------------------------------------------------

    @classmethod
    def get_backend(cls) -> str:
        """Get the tensor backend ("torch" or "warp").

        Returns:
            The current tensor backend name.
        """
        return cls._backend

    @classmethod
    def set_backend(cls, backend: str) -> None:
        """Set the tensor backend.

        Args:
            backend: Either "torch" or "warp".

        Raises:
            ValueError: If backend is not "torch" or "warp".
        """
        if backend not in ("torch", "warp"):
            raise ValueError(f"Backend must be 'torch' or 'warp', got '{backend}'")
        cls._backend = backend

    # ------------------------------------------------------------------
    # State Accessors
    # ------------------------------------------------------------------

    @classmethod
    def is_warmup_needed(cls) -> bool:
        """Check if physics warmup is needed."""
        return cls._warmup_needed

    @classmethod
    def is_view_created(cls) -> bool:
        """Check if simulation views have been created."""
        return cls._view_created

    @classmethod
    def assets_loading(cls) -> bool:
        """Check if assets are currently loading.

        Returns:
            True if assets are still loading, False otherwise.
        """
        return not cls._assets_loaded

    # ------------------------------------------------------------------
    # Lifecycle (Abstract)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def initialize(cls) -> None:
        """Initialize the manager and set up internal callbacks.

        This should set up timeline callbacks, stage tracking, and any
        other backend-specific initialization.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def clear(cls) -> None:
        """Clear all state and callbacks.

        This should:
        - Dispatch prim deletion event for cleanup
        - Clear all callbacks and handles
        - Invalidate and destroy simulation views
        - Reset state flags
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initialize_physics(cls) -> None:
        """Warm-start physics and create simulation views.

        This should:
        - Force load physics from USD/scene
        - Start the simulation engine
        - Perform initial simulation step
        - Dispatch PHYSICS_WARMUP event
        - Create simulation views
        - Dispatch SIMULATION_VIEW_CREATED and PHYSICS_READY events
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Simulation Views (Abstract)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def get_physics_sim_view(cls) -> Any | None:
        """Get the physics simulation view.

        Returns:
            The simulation view for tensor operations, or None if not created.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Physics Configuration (Abstract)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def get_physics_dt(cls) -> float:
        """Get the physics timestep in seconds.

        Returns:
            The fixed timestep used for physics integration.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_physics_sim_device(cls) -> str:
        """Get the physics simulation device.

        Returns:
            Device string like "cpu" or "cuda:0".
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def set_physics_sim_device(cls, device: str) -> None:
        """Set the physics simulation device.

        Args:
            device: Device string like "cpu" or "cuda:0".

        Raises:
            ValueError: If device string is invalid.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Event Dispatch (Abstract)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def dispatch_prim_deletion(cls, prim_path: str) -> None:
        """Dispatch prim deletion event.

        Args:
            prim_path: The USD path of the prim being deleted.
                       Use "/" to signal clearing all prims.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Callback Registration (Abstract)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def register_callback(
        cls,
        callback: Callable,
        event: SimulationEvent,
        order: int = 0,
        name: str | None = None,
    ) -> int:
        """Register a callback for a simulation event.

        Args:
            callback: Function to call when event fires.
            event: The event type to listen for.
            order: Priority (lower = earlier). Defaults to 0.
            name: Optional name for the callback.

        Returns:
            Callback ID for deregistration.
        """
        raise NotImplementedError

    @classmethod
    def deregister_callback(cls, callback_id: int) -> None:
        """Deregister a callback by ID.

        Args:
            callback_id: The callback ID returned from register_callback.
        """
        cls._callbacks.pop(callback_id, None)

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    @classmethod
    def _next_callback_id(cls) -> int:
        """Generate the next unique callback ID.

        Returns:
            A unique callback ID.
        """
        cid = cls._callback_id
        cls._callback_id += 1
        return cid

    @classmethod
    def _wrap_bound_method(cls, callback: Callable) -> Callable:
        """Wrap a bound method with a weak reference to avoid preventing GC.

        Args:
            callback: The callback, possibly a bound method.

        Returns:
            A wrapped callback that uses weak references for bound methods,
            or the original callback if not a bound method.
        """
        if hasattr(callback, "__self__"):
            obj_ref = weakref.proxy(callback.__self__)
            method_name = callback.__name__
            return lambda e, o=obj_ref, m=method_name: getattr(o, m)(e)
        return callback

    # ------------------------------------------------------------------
    # Optional: Stage Open Callback (can be overridden)
    # ------------------------------------------------------------------

    @classmethod
    def enable_stage_open_callback(cls, enable: bool) -> None:
        """Enable or disable stage open tracking.

        This is optional and may be overridden by backend implementations
        that need to track stage/scene changes.

        Args:
            enable: Whether to enable stage open tracking.
        """
        pass  # Default no-op, override in subclass if needed
