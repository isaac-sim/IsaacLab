# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for physics managers with unified callback system."""

from __future__ import annotations

import functools
import logging
import warnings
import weakref
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from enum import Enum
from types import FunctionType
from typing import TYPE_CHECKING, Any, ClassVar

from ..sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from ..cloner import ReplicateContext
    from ..scene_data import SceneDataBackend
    from ..sim.simulation_context import SimulationContext
    from .physics_manager_cfg import PhysicsCfg

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

    def __init__(self, callback_id: int, manager: PhysicsManager):
        self._id = callback_id
        self._manager = manager

    @property
    def id(self) -> int:
        return self._id

    def deregister(self) -> None:
        """Remove this callback from the manager."""
        self._manager.deregister_callback(self._id)


def _active_manager(cls: type[PhysicsManager]) -> PhysicsManager | None:
    """Return the active simulation's manager if it is an instance of *cls*."""
    from ..sim.simulation_context import SimulationContext  # noqa: PLC0415

    manager = getattr(SimulationContext.instance(), "backend_manager", None)
    return manager if isinstance(manager, cls) else None


def _is_instance_method(cls: type, name: str) -> bool:
    """Whether *name* resolves to a plain function (not a class or static method) on *cls*."""
    for klass in type.__getattribute__(cls, "__mro__"):
        if name in klass.__dict__:
            return isinstance(klass.__dict__[name], FunctionType)
    return False


class _PhysicsManagerMeta(ABCMeta):
    """Forward deprecated class-level runtime access to the active simulation's manager.

    Runtime methods and state are per instance. Class-level calls such as ``NewtonManager.forward()`` keep
    working with a :class:`DeprecationWarning` by binding to ``SimulationContext.instance().backend_manager``.
    Instance access and explicit unbound calls with any manager (``Manager.method(manager, ...)``) are unaffected.
    """

    def __getattribute__(cls, name: str) -> Any:
        attr = super().__getattribute__(name)
        if name[0] == "_" or not isinstance(attr, FunctionType) or not _is_instance_method(cls, name):
            return attr

        @functools.wraps(attr)
        def forward(*args, **kwargs):
            if args and isinstance(args[0], PhysicsManager):
                return attr(*args, **kwargs)
            manager = _active_manager(cls)
            if manager is None:
                raise TypeError(f"{cls.__name__}.{name}() is an instance method; call it on sim.backend_manager.")
            warnings.warn(
                f"Class-level {cls.__name__}.{name}() is deprecated; call it on the simulation's manager instance"
                f" (sim.backend_manager.{name}()).",
                DeprecationWarning,
                stacklevel=2,
            )
            return attr(manager, *args, **kwargs)

        return forward

    def __getattr__(cls, name: str) -> Any:
        # Instance state read on the class, such as ``views``.
        manager = None if name[0] == "_" else _active_manager(cls)
        if manager is None or name not in vars(manager):
            raise AttributeError(f"type object {cls.__name__!r} has no attribute {name!r}")
        warnings.warn(
            f"Class-level {cls.__name__}.{name} is deprecated; read it from the simulation's manager instance"
            f" (sim.backend_manager.{name}).",
            DeprecationWarning,
            stacklevel=2,
        )
        return vars(manager)[name]


class PhysicsManager(metaclass=_PhysicsManagerMeta):
    """Abstract base class for physics simulation managers.

    Physics managers handle the lifecycle of a physics simulation backend,
    including initialization, stepping, and cleanup.

    This base class provides:
    - Unified callback management system
    - Common state variables (_sim, _cfg, _device)
    - Default accessor implementations

    Lifecycle: initialize() -> reset() -> step() (repeated) -> close()
    """

    backend_name: ClassVar[str]
    """Backend implementation key used by asset and sensor factories."""

    def __init__(self, cfg: PhysicsCfg):
        """Store the physics configuration this manager runs.

        Args:
            cfg: Physics configuration whose ``class_type`` selected this manager.
        """
        self._sim: SimulationContext | None = None
        self._cfg = cfg
        self._device: str = "cuda:0"
        self._sim_time: float = 0.0
        self._callbacks: dict[int, tuple[Any, Callable, int, str | None, Any]] = {}
        self._callback_id: int = 0
        self.views: dict[str, Any] = {}
        """Backend views by prim path expression, shared by the consumers of this manager."""

    @classmethod
    def _deprecated_active(cls, owner: str) -> PhysicsManager:
        """Return the active simulation's manager for a caller that omitted it, with a deprecation warning.

        Args:
            owner: Name of the API that now takes the manager, used in the warning.

        Returns:
            The active simulation's manager.

        Raises:
            RuntimeError: If no active simulation uses a manager of this class.
        """
        manager = _active_manager(cls)
        if manager is None:
            raise RuntimeError(f"{owner} requires backend_manager: no active {cls.__name__} simulation.")
        warnings.warn(
            f"Calling {owner} without backend_manager is deprecated; pass the simulation's manager"
            " (sim.backend_manager).",
            DeprecationWarning,
            stacklevel=3,
        )
        return manager
        manager = _active_manager(cls)
        if manager is None:
            raise RuntimeError(f"{owner} requires backend_manager: no active {cls.__name__} simulation.")
        warnings.warn(
            f"Calling {owner} without backend_manager is deprecated; pass the simulation's manager"
            " (sim.backend_manager).",
            DeprecationWarning,
            stacklevel=3,
        )
        return manager

    clone_context_type: ClassVar[type[ReplicateContext] | None] = None

    supports_anim_recording: ClassVar[bool] = False
    """Whether this backend can service ``--anim_recording_enabled`` (OVD Recorder).

    Overridden by backends that implement the recorder (currently PhysX-only).
    """

    @classmethod
    def _prepare_stage_creation(cls) -> None:
        """Perform backend-specific setup required before the USD stage is created."""
        pass

    @classmethod
    def fix_articulation_root(cls, articulation_prim: Any, stage: Any = None) -> Any:
        """Ensure that an articulation root has one enabled world fixed joint.

        The base implementation leaves the root in place. Backends whose parser requires a different
        root topology may relocate it and return the resulting root prim.

        Args:
            articulation_prim: The articulation-root prim to fix.
            stage: The stage containing the prim. Defaults to the current stage.

        Returns:
            The articulation-root prim after backend normalization.

        Raises:
            NotImplementedError: If a new joint is needed and the root is not a rigid body.
        """
        # Keep these imports local. Hoisting the isaaclab.sim ones closes a real import cycle:
        # isaaclab.physics -> sim.schemas.schemas -> sim.utils.prims -> sim.utils.queries ->
        # sim.simulation_context -> isaaclab.physics (partially initialized). Keeping pxr local
        # also keeps USD out of the config-definition path, which env configs import through
        # managers.manager_base before the simulation app starts.
        from pxr import UsdPhysics  # noqa: PLC0415

        from ..sim.schemas.schemas import create_world_fixed_joint  # noqa: PLC0415
        from ..sim.utils import find_global_fixed_joint_prim  # noqa: PLC0415

        if stage is None:
            stage = get_current_stage()
        root_path = articulation_prim.GetPath().pathString
        joint = find_global_fixed_joint_prim(root_path, stage=stage)
        if joint is not None:
            joint.GetJointEnabledAttr().Set(True)
            return articulation_prim
        if not articulation_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise NotImplementedError(f"Cannot fix non-rigid articulation root '{root_path}'.")

        create_world_fixed_joint(articulation_prim, stage)
        return articulation_prim

    @classmethod
    def setup_deformable_body(cls, prim: Any, deformable_type: str, sim_mesh_prim: Any, vis_mesh_prim: Any) -> None:
        """Apply the backend's deformable anchor schemas to a prepared deformable body.

        Called by the deformable family writers after the backend-neutral mesh setup: the
        simulation mesh already exists (a ``UsdGeom.TetMesh`` for volume, a triangle
        ``UsdGeom.Mesh`` for surface) with collision enabled. The backend applies its sim and
        body anchor schemas and any backend-specific authoring (rest state, pose bindings,
        visual-mesh synchronization).

        The base raises: backends with deformable support override.

        Args:
            prim: The deformable-body prim to anchor.
            deformable_type: The deformable type, ``"volume"`` or ``"surface"``.
            sim_mesh_prim: The prepared simulation-mesh prim.
            vis_mesh_prim: The visual-mesh prim.

        Raises:
            NotImplementedError: If the backend does not implement deformable fragment setup.
        """
        raise NotImplementedError(
            f"Physics backend '{cls.__name__}' does not implement deformable fragment setup, so the"
            " 'volume_deformable_props' and 'surface_deformable_props' slots cannot be used with it."
        )

    @staticmethod
    def _relocate_articulation_root(
        articulation_prim: Any,
        companion_schema: str,
        companion_namespace: str,
    ) -> Any:
        """Move root-bearing schemas and authored properties to the root link's parent."""
        # Keep pxr local: this module is imported while environment configs load (via the manager
        # classes), and config loading must not pull USD/omni modules before the simulation app
        # starts.
        from pxr import Usd, UsdPhysics  # noqa: PLC0415

        new_root = articulation_prim.GetParent()
        if new_root.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise RuntimeError(
                f"Cannot relocate '{articulation_prim.GetPath()}' to existing articulation root '{new_root.GetPath()}'."
            )

        # Keep this import local for the same reason as the pxr imports above.
        from ..sim.schemas.backend_hooks import articulation_root_companion_namespace  # noqa: PLC0415

        registry = Usd.SchemaRegistry()
        root_schema = UsdPhysics.Tokens.PhysicsArticulationRootAPI
        schemas_to_move = []
        for schema_name in articulation_prim.GetPrimTypeInfo().GetAppliedAPISchemas():
            definition = registry.FindAppliedAPIPrimDefinition(schema_name)
            companion_namespace_override = articulation_root_companion_namespace(schema_name)
            if schema_name == companion_schema:
                properties = list(articulation_prim.GetAuthoredPropertiesInNamespace(companion_namespace))
            elif companion_namespace_override is not None:
                # a backend-registered schema, possibly an unregistered token the registry cannot
                # describe, so take the namespace the backend declared for it
                properties = list(articulation_prim.GetAuthoredPropertiesInNamespace(companion_namespace_override))
            elif schema_name == root_schema or (
                definition is not None and root_schema in definition.GetAppliedAPISchemas()
            ):
                properties = []
                if definition is not None:
                    for property_name in definition.GetPropertyNames():
                        prop = articulation_prim.GetProperty(property_name)
                        if prop and prop.IsAuthored():
                            properties.append(prop)
            else:
                continue
            schemas_to_move.append((schema_name, properties))

        for schema_name, properties in schemas_to_move:
            if not new_root.AddAppliedSchema(schema_name):
                raise RuntimeError(f"Failed to apply '{schema_name}' to '{new_root.GetPath()}'.")
            for prop in properties:
                if not prop.FlattenTo(new_root):
                    raise RuntimeError(f"Failed to move '{prop.GetPath()}' to '{new_root.GetPath()}'.")
        for schema_name, _ in schemas_to_move:
            if not articulation_prim.RemoveAppliedSchema(schema_name):
                raise RuntimeError(f"Failed to remove '{schema_name}' from '{articulation_prim.GetPath()}'.")
        if articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI) or not new_root.HasAPI(
            UsdPhysics.ArticulationRootAPI
        ):
            raise RuntimeError(
                f"Failed to relocate articulation root '{articulation_prim.GetPath()}' to '{new_root.GetPath()}'."
            )
        return new_root

    def register_callback(
        self,
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
            >>> handle = sim.backend_manager.register_callback(on_physics_ready, PhysicsEvent.PHYSICS_READY)
            >>> # Later, to remove:
            >>> handle.deregister()
        """
        # IDs stay unique for this owner even when callbacks are cleared and registered again.
        cid = self._callback_id
        self._callback_id += 1

        if wrap_weak_ref:
            callback = self._wrap_weak_ref(callback)

        subscription = self._subscribe_to_event(cid, callback, event, order, name)

        self._callbacks[cid] = (event, callback, order, name, subscription)
        return CallbackHandle(cid, self)

    def deregister_callback(self, callback_id: int | CallbackHandle) -> None:
        """Remove a registered callback.

        Args:
            callback_id: The ID or CallbackHandle returned by register_callback().
        """
        cid = callback_id.id if isinstance(callback_id, CallbackHandle) else callback_id
        if cid not in self._callbacks:
            return

        event, callback, order, name, subscription = self._callbacks.pop(cid)
        self._unsubscribe_from_event(cid, event, subscription)

    def dispatch_event(self, event: PhysicsEvent, payload: Any = None) -> None:
        """Dispatch an event to all registered callbacks.

        This is the default implementation using simple callback lists.
        Subclasses may override or extend with platform-specific dispatch.

        Args:
            event: The event to dispatch.
            payload: Optional data to pass to callbacks.
        """
        matching = [(cid, cb, order) for cid, (ev, cb, order, name, sub) in self._callbacks.items() if ev == event]
        matching.sort(key=lambda x: x[2])

        for _, callback, _ in matching:
            callback(payload)

    def clear_callbacks(self) -> None:
        """Remove all registered callbacks.

        IDs remain unique for this manager's lifetime. Otherwise a retained handle could deregister a new callback
        after a close and reinitialization of the same manager.
        """
        for cid in list(self._callbacks.keys()):
            self.deregister_callback(cid)
        self._callbacks.clear()

    def _wrap_weak_ref(self, callback: Callable) -> Callable:
        """Wrap bound methods with weak references to prevent leaks.

        Args:
            callback: The callback to wrap.

        Returns:
            Wrapped callback if it's a bound method, otherwise original.
        """
        owner = getattr(callback, "__self__", None)
        if owner is not None:
            obj_ref = weakref.ref(owner)
            method_name = callback.__name__

            def weak_callback(payload: Any) -> Any:
                obj = obj_ref()
                if obj is None:
                    return None
                return getattr(obj, method_name)(payload)

            return weak_callback
        return callback

    def _subscribe_to_event(
        self,
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
        return None

    def _unsubscribe_from_event(
        self,
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

    def initialize(self, sim_context: SimulationContext) -> None:
        """Attach the manager to its simulation context.

        The context's device becomes the manager's. A manager configured completely at
        construction needs nothing else; subclasses that set up native resources against the context call
        ``super().initialize()`` first.

        Args:
            sim_context: Parent simulation context.
        """
        self._sim = sim_context
        self._device = sim_context.cfg.device
        self._sim_time = 0.0

        # The OVD Recorder (omni.physx.pvd) only records PhysX simulations. On other backends the
        # recording would silently never start, so the process would run until manually killed
        # instead of stopping at `--anim_recording_stop_time` and saving the animation.
        # ``get_setting`` may be absent on lightweight sim_context test doubles that only
        # implement the ``cfg``/``device`` surface this method also reads above.
        get_setting = getattr(sim_context, "get_setting", None)
        if get_setting and get_setting("/isaaclab/anim_recording/enabled") and not self.supports_anim_recording:
            raise ValueError(
                f"'--anim_recording_enabled' was set, but the active physics backend ('{type(self).__name__}') does not"
                " support the OVD Recorder. Select the PhysX backend, e.g. by appending"
                " 'physics=isaacsim_physx' to the command line."
            )

    @abstractmethod
    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        pass

    @abstractmethod
    def forward(self) -> None:
        """Update kinematics without stepping physics (for rendering)."""
        pass

    @abstractmethod
    def get_scene_data_backend(self) -> SceneDataBackend:
        """Return the SceneDataBackend for the SceneDataProvider."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Step physics simulation by one timestep (physics only, no rendering)."""
        pass

    def pre_render(self) -> None:
        """Sync deferred physics state to the rendering backend.

        Called by :meth:`~isaaclab.sim.SimulationContext.render` before cameras
        and visualizers read scene data. The default implementation is a no-op.
        Backends that defer transform writes (e.g. Newton's dirty-flag pattern)
        should override this to flush pending updates.
        """
        pass

    def after_visualizers_render(self) -> None:
        """Hook after visualizers have stepped during :meth:`~isaaclab.sim.SimulationContext.render`.

        Use for physics-backend sync (e.g. fabric) if needed. Default is a no-op.
        """
        pass

    def video_capture_backend(self) -> str | None:
        """Return the video capture backend identifier for this physics manager.

        Used by :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder` to select
        how perspective video frames are captured when no visualizer is active.

        Returns:
            ``"kit"`` for backends that use Kit/Replicator (e.g. :class:`~isaaclab_physx.physics.PhysxManager`),
            ``"newton_gl"`` for backends that use a headless Newton GL viewer
            (e.g. :class:`~isaaclab_newton.physics.NewtonManager`),
            or ``None`` if the backend does not support perspective video capture.
        """
        return None

    def close(self) -> None:
        """Clean up physics resources.

        Subclasses whose STOP listeners own backend handles should call
        ``super().close()`` before backend-specific cleanup so those listeners
        can invalidate their handles while the backend is still live.

        All STOP listeners are given a chance to run. If one or more listeners
        fail, callback and shared simulation state is still cleared before an
        aggregate :class:`RuntimeError` is raised from the first failure.
        """
        callback_errors = self._dispatch_event_collect_errors(PhysicsEvent.STOP)

        try:
            self.clear_callbacks()
        finally:
            self.views.clear()
            self._sim_time = 0.0
            self._sim = None

        if callback_errors:
            raise RuntimeError(
                f"{len(callback_errors)} callback(s) failed during PhysicsEvent.STOP dispatch."
            ) from callback_errors[0]

    def _dispatch_event_collect_errors(self, event: PhysicsEvent, payload: Any = None) -> list[Exception]:
        """Dispatch an event to every listener and collect direct or backend-stored failures."""
        matching = [
            (callback, order)
            for registered_event, callback, order, _name, _subscription in self._callbacks.values()
            if registered_event == event
        ]
        matching.sort(key=lambda item: item[1])
        callback_errors: list[Exception] = []
        raise_stored = getattr(self, "raise_callback_exception_if_any", None)

        def drain_stored_error() -> None:
            if callable(raise_stored):
                try:
                    raise_stored()
                except Exception as exc:
                    callback_errors.append(exc)

        for callback, _order in matching:
            try:
                callback(payload)
            except Exception as exc:
                callback_errors.append(exc)
            drain_stored_error()
        return callback_errors

    def get_physics_dt(self) -> float:
        """Get the physics timestep in seconds."""
        return self._sim.cfg.dt if self._sim else 1.0 / 60.0

    def get_device(self) -> str:
        """Get the physics simulation device."""
        return self._device

    def get_simulation_time(self) -> float:
        """Get the current simulation time in seconds."""
        return self._sim_time

    def get_physics_sim_view(self) -> Any:
        """Get the physics simulation view. Override in subclasses."""
        return None

    def play(self) -> None:
        """Start or resume physics simulation. Default is no-op."""
        pass

    def pause(self) -> None:
        """Pause physics simulation. Default is no-op."""
        pass

    def stop(self) -> None:
        """Stop physics simulation. Default is no-op."""
        pass

    def wait_for_playing(self) -> None:
        """Block until the timeline is playing. Default is no-op."""
        pass

    def set_decimation(self, decimation: int) -> None:
        """Inform the physics backend how many substeps the environment runs per policy step.

        Backends that can fold the full decimation loop into a single
        :meth:`step` call (e.g. Newton with all-graphable actuators) use this
        to size their internal loop / CUDA graph.  The default implementation
        is a no-op.

        Args:
            decimation: Number of physics steps per environment step.
        """
        pass

    def handles_decimation(self) -> bool:
        """``True`` when :meth:`step` executes the full decimation loop internally.

        When this returns ``True`` the environment should call :meth:`step`
        once per policy step instead of looping ``decimation`` times.
        """
        return False

    def get_backend(self) -> str:
        """Get the tensor backend being used ("numpy" or "torch")."""
        return "torch" if "cuda" in self._device else "numpy"

    @staticmethod
    def safe_callback_invoke(
        fn: Callable,
        *args,
        backend_manager: PhysicsManager | None = None,
        physics_manager: PhysicsManager | None = None,
    ) -> None:
        """Invoke a callback, catching exceptions that would be swallowed by external event buses.

        Ignores ``ReferenceError`` (from garbage-collected weakref proxies). All other
        exceptions are forwarded to *backend_manager*.``store_callback_exception`` when
        available (see note below), or re-raised immediately otherwise.

        Args:
            fn: Callback to invoke.
            *args: Positional arguments for *fn*.
            backend_manager: Manager that stores swallowed exceptions, if any.
            physics_manager: Deprecated alias of *backend_manager*.

        Note (Octi):
            The carb event bus used by PhysX/Omniverse silently swallows exceptions raised
            inside callbacks. ``PhysxManager`` works around this by storing the exception
            and re-raising it after event dispatch completes (in ``reset()`` / ``step()``).
            Backends that dispatch events directly (e.g. Newton) don't need this — exceptions
            propagate normally — so ``store_callback_exception`` is not called for them.
            This is a known wart; a cleaner solution is actively being explored.
        """
        if physics_manager is not None:
            warnings.warn(
                "safe_callback_invoke(physics_manager=...) is deprecated; use backend_manager.",
                DeprecationWarning,
                stacklevel=2,
            )
            backend_manager = physics_manager
        try:
            fn(*args)
        except ReferenceError:
            pass
        except Exception as e:
            store_fn = getattr(backend_manager, "store_callback_exception", None)
            if callable(store_fn):
                store_fn(e)
            else:
                raise
