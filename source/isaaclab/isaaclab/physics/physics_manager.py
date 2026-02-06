# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for physics managers with unified callback system."""

from __future__ import annotations

import logging
import weakref
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, ClassVar

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class PhysicsEvent(Enum):
    """Physics simulation lifecycle events.

    These are general events that apply across all physics backends.
    Backend-specific events (e.g., PhysX timeline events) are handled
    by the respective manager classes.
    """

    # Initialization
    MODEL_INIT = "model_init"
    """Model/scene is being initialized (before physics starts)."""

    PHYSICS_READY = "physics_ready"
    """Physics is fully initialized and ready to simulate."""

    # Simulation lifecycle
    PRE_STEP = "pre_step"
    """Called before each physics step."""

    POST_STEP = "post_step"
    """Called after each physics step."""

    POST_RESET = "post_reset"
    """Called after simulation reset."""


class CallbackHandle:
    """Handle for a registered callback, allowing deregistration."""

    def __init__(self, callback_id: int, manager: type["PhysicsManager"]):
        self._id = callback_id
        self._manager = manager

    @property
    def id(self) -> int:
        return self._id

    def deregister(self) -> None:
        """Remove this callback from the manager."""
        self._manager.deregister_callback(self._id)


class PhysicsManager(ABC):
    """Abstract base class for physics simulation managers.

    Physics managers handle the lifecycle of a physics simulation backend,
    including initialization, stepping, and cleanup.

    This base class provides a unified callback management system that works
    across different physics backends (PhysX, Newton).

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    # Callback storage: callback_id -> (event, callback, order, name, subscription)
    _callbacks: ClassVar[dict[int, tuple[PhysicsEvent, Callable, int, str | None, Any]]] = {}
    _callback_id: ClassVar[int] = 0

    # ------------------------------------------------------------------
    # Callback Management API
    # ------------------------------------------------------------------

    @classmethod
    def register_callback(
        cls,
        callback: Callable[[Any], None],
        event: PhysicsEvent,
        order: int = 0,
        name: str | None = None,
        wrap_weak_ref: bool = True,
    ) -> CallbackHandle:
        """Register a callback for a physics event.

        Args:
            callback: The callback function. Receives event payload as argument.
            event: The event to listen for.
            order: Priority order (lower = earlier). Default 0.
            name: Optional name for debugging.
            wrap_weak_ref: If True, wrap bound methods with weak references
                to prevent preventing garbage collection. Default True.

        Returns:
            CallbackHandle that can be used to deregister the callback.

        Example:
            >>> def on_physics_ready(payload):
            ...     print("Physics is ready!")
            >>> handle = PhysxManager.register_callback(
            ...     on_physics_ready,
            ...     PhysicsEvent.PHYSICS_READY
            ... )
            >>> # Later, to remove:
            >>> handle.deregister()
        """
        cid = cls._callback_id
        cls._callback_id += 1

        # Wrap bound methods with weak references to prevent leaks
        if wrap_weak_ref:
            callback = cls._wrap_weak_ref(callback)

        # Let subclass handle platform-specific subscription
        subscription = cls._subscribe_to_event(cid, callback, event, order, name)

        cls._callbacks[cid] = (event, callback, order, name, subscription)
        return CallbackHandle(cid, cls)

    @classmethod
    def deregister_callback(cls, callback_id: int) -> None:
        """Remove a registered callback.

        Args:
            callback_id: The ID returned by register_callback().
        """
        if callback_id not in cls._callbacks:
            return

        event, callback, order, name, subscription = cls._callbacks.pop(callback_id)
        cls._unsubscribe_from_event(callback_id, event, subscription)

    @classmethod
    def dispatch_event(cls, event: PhysicsEvent, payload: Any = None) -> None:
        """Dispatch an event to all registered callbacks.

        This is the default implementation using simple callback lists.
        Subclasses may override or extend with platform-specific dispatch.

        Args:
            event: The event to dispatch.
            payload: Optional data to pass to callbacks.
        """
        # Get callbacks for this event, sorted by order
        matching = [
            (cid, cb, order)
            for cid, (ev, cb, order, name, sub) in cls._callbacks.items()
            if ev == event
        ]
        matching.sort(key=lambda x: x[2])

        for cid, callback, order in matching:
            try:
                callback(payload)
            except ReferenceError:
                # Weak reference expired, remove callback
                cls.deregister_callback(cid)
            except Exception as e:
                logger.error(f"Callback {cid} for {event.value} failed: {e}")

    @classmethod
    def clear_callbacks(cls) -> None:
        """Remove all registered callbacks."""
        for cid in list(cls._callbacks.keys()):
            cls.deregister_callback(cid)
        cls._callbacks.clear()
        cls._callback_id = 0

    # ------------------------------------------------------------------
    # Callback Helpers (for subclasses)
    # ------------------------------------------------------------------

    @classmethod
    def _wrap_weak_ref(cls, callback: Callable) -> Callable:
        """Wrap bound methods with weak references to prevent leaks.

        Args:
            callback: The callback to wrap.

        Returns:
            Wrapped callback if it's a bound method, otherwise original.
        """
        if hasattr(callback, "__self__"):
            obj_ref = weakref.proxy(callback.__self__)
            method_name = callback.__name__

            def weak_callback(payload: Any) -> Any:
                return getattr(obj_ref, method_name)(payload)

            return weak_callback
        return callback

    @classmethod
    def _subscribe_to_event(
        cls,
        callback_id: int,
        callback: Callable,
        event: PhysicsEvent,
        order: int,
        name: str | None,
    ) -> Any:
        """Subscribe to a platform-specific event.

        Override in subclasses to integrate with platform event systems
        (e.g., Omniverse event bus, timeline events).

        Args:
            callback_id: Unique ID for this callback.
            callback: The callback function.
            event: The event to subscribe to.
            order: Priority order.
            name: Optional name.

        Returns:
            Platform-specific subscription object (stored for cleanup).
        """
        # Default: no platform subscription, use dispatch_event()
        return None

    @classmethod
    def _unsubscribe_from_event(
        cls,
        callback_id: int,
        event: PhysicsEvent,
        subscription: Any,
    ) -> None:
        """Unsubscribe from a platform-specific event.

        Override in subclasses to clean up platform subscriptions.

        Args:
            callback_id: The callback ID being removed.
            event: The event that was subscribed to.
            subscription: The subscription object from _subscribe_to_event().
        """
        # Default: nothing to clean up
        pass

    # ------------------------------------------------------------------
    # Physics Lifecycle (Abstract)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def initialize(cls, sim_context: "SimulationContext") -> None:
        """Initialize the physics manager with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        pass

    @classmethod
    @abstractmethod
    def reset(cls, soft: bool = False) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        pass

    @classmethod
    @abstractmethod
    def forward(cls) -> None:
        """Update kinematics without stepping physics (for rendering)."""
        pass

    @classmethod
    @abstractmethod
    def step(cls) -> None:
        """Step physics simulation by one timestep (physics only, no rendering)."""
        pass

    @classmethod
    @abstractmethod
    def close(cls) -> None:
        """Clean up physics resources."""
        pass

    @classmethod
    @abstractmethod
    def get_physics_dt(cls) -> float:
        """Get the physics timestep in seconds."""
        pass

    @classmethod
    @abstractmethod
    def get_device(cls) -> str:
        """Get the physics simulation device."""
        pass

    @classmethod
    @abstractmethod
    def get_physics_sim_view(cls):
        """Get the physics simulation view."""
        pass

    @classmethod
    @abstractmethod
    def is_fabric_enabled(cls) -> bool:
        """Check if fabric interface is enabled."""
        pass

    @classmethod
    def play(cls) -> None:
        """Start or resume physics simulation. Default is no-op."""
        pass

    @classmethod
    def pause(cls) -> None:
        """Pause physics simulation. Default is no-op."""
        pass

    @classmethod
    def stop(cls) -> None:
        """Stop physics simulation. Default is no-op."""
        pass
