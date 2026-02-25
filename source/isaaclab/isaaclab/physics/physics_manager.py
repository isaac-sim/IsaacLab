# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for physics managers with unified callback system."""

from __future__ import annotations

import logging
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class PhysicsEvent(Enum):
    """Physics simulation lifecycle events.

    These are general events that apply across all physics backends.
    Backend-specific events (e.g., PhysX step events, timeline events) are handled
    by the respective manager classes via their own event enums (e.g., IsaacEvents).

    Lifecycle order: MODEL_INIT -> PHYSICS_READY -> STOP
    """

    MODEL_INIT = "model_init"
    """Physics model is being constructed.
    Fired during scene building, before simulation can run. Use this to register
    physics representations (rigid bodies, joints, constraints) with the solver.
    """

    PHYSICS_READY = "physics_ready"
    """Physics is initialized and queryable.
    Fired after all physics data structures are created and the simulation is
    ready to step. Assets can now read initial state (positions, velocities).
    """

    STOP = "stop"
    """Simulation is stopping."""


class CallbackHandle:
    """Handle for a registered callback, allowing deregistration."""

    def __init__(self, callback_id: int, manager: type[PhysicsManager]):
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

    This base class provides:
    - Unified callback management system
    - Common state variables (_sim, _cfg, _device)
    - Default accessor implementations

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    _sim: ClassVar[SimulationContext | None] = None
    _cfg: ClassVar[Any] = None
    _device: ClassVar[str] = "cuda:0"
    _sim_time: ClassVar[float] = 0.0
    _callbacks: ClassVar[dict[int, tuple[Any, Callable, int, str | None, Any]]] = {}
    _callback_id: ClassVar[int] = 0

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
            >>> handle = PhysxManager.register_callback(on_physics_ready, PhysicsEvent.PHYSICS_READY)
            >>> # Later, to remove:
            >>> handle.deregister()
        """
        cid = PhysicsManager._callback_id
        PhysicsManager._callback_id += 1

        if wrap_weak_ref:
            callback = cls._wrap_weak_ref(callback)

        subscription = cls._subscribe_to_event(cid, callback, event, order, name)

        PhysicsManager._callbacks[cid] = (event, callback, order, name, subscription)
        return CallbackHandle(cid, cls)

    @classmethod
    def deregister_callback(cls, callback_id: int | CallbackHandle) -> None:
        """Remove a registered callback.

        Args:
            callback_id: The ID or CallbackHandle returned by register_callback().
        """
        cid = callback_id.id if isinstance(callback_id, CallbackHandle) else callback_id
        if cid not in PhysicsManager._callbacks:
            return

        event, callback, order, name, subscription = PhysicsManager._callbacks.pop(cid)
        cls._unsubscribe_from_event(cid, event, subscription)

    @classmethod
    def dispatch_event(cls, event: PhysicsEvent, payload: Any = None) -> None:
        """Dispatch an event to all registered callbacks.

        This is the default implementation using simple callback lists.
        Subclasses may override or extend with platform-specific dispatch.

        Args:
            event: The event to dispatch.
            payload: Optional data to pass to callbacks.
        """
        # All callbacks are stored in PhysicsManager._callbacks (shared across all subclasses)
        matching = [
            (cid, cb, order) for cid, (ev, cb, order, name, sub) in PhysicsManager._callbacks.items() if ev == event
        ]

        matching.sort(key=lambda x: x[2])

        for cid, callback, order in matching:
            try:
                callback(payload)
            except ReferenceError:
                cls.deregister_callback(cid)
            except Exception as e:
                raise ValueError(f"Callback {cid} for {event.value} failed: {e}") from e

    @classmethod
    def clear_callbacks(cls) -> None:
        """Remove all registered callbacks."""
        for cid in list(PhysicsManager._callbacks.keys()):
            cls.deregister_callback(cid)
        PhysicsManager._callbacks.clear()
        PhysicsManager._callback_id = 0

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
        pass

    @classmethod
    @abstractmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the physics manager with simulation context.

        Subclasses should call super().initialize() first, then do backend-specific setup.

        Args:
            sim_context: Parent simulation context.
        """
        # Set on PhysicsManager explicitly so PhysicsManager.get_*() works
        # regardless of which subclass is active (Python class vars are per-class)
        PhysicsManager._sim = sim_context
        PhysicsManager._cfg = sim_context.cfg.physics
        PhysicsManager._device = sim_context.cfg.device
        PhysicsManager._sim_time = 0.0

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
    def close(cls) -> None:
        """Clean up physics resources.

        Subclasses should call super().close() after backend-specific cleanup.
        """
        cls.dispatch_event(PhysicsEvent.STOP)  # notify listeners before cleanup
        cls.clear_callbacks()
        # Reset on PhysicsManager explicitly (matches initialize())
        PhysicsManager._sim = None
        PhysicsManager._cfg = None
        PhysicsManager._sim_time = 0.0

    @classmethod
    def get_physics_dt(cls) -> float:
        """Get the physics timestep in seconds."""
        return PhysicsManager._sim.cfg.dt if PhysicsManager._sim else 1.0 / 60.0

    @classmethod
    def get_device(cls) -> str:
        """Get the physics simulation device."""
        return PhysicsManager._device

    @classmethod
    def get_simulation_time(cls) -> float:
        """Get the current simulation time in seconds."""
        return PhysicsManager._sim_time

    @classmethod
    def get_physics_sim_view(cls) -> Any:
        """Get the physics simulation view. Override in subclasses."""

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

    @classmethod
    def get_backend(cls) -> str:
        """Get the tensor backend being used ("numpy" or "torch")."""
        return "torch" if "cuda" in PhysicsManager._device else "numpy"
