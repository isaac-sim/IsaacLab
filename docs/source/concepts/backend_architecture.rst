.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _backend-architecture:

Backend Architecture
====================

Overview
--------

Isaac Lab supports multiple physics backends while presenting common asset,
sensor, and scene interfaces to environment code. Factories dispatch an object
to the active backend implementation at construction time, so code can use the
same public API without importing backend-specific modules directly. For
choosing a backend or preset in an environment, see
:ref:`backends-and-presets`.

Factory dispatch
----------------

All factories inherit from :class:`~isaaclab.utils.backend_utils.FactoryBase`.
They locate supported backend implementations through a core backend-key
selector followed by package and module-path conventions:

1. The name of ``SimulationContext.physics_manager`` is mapped to one of the
   backend keys recognized by ``FactoryBase._get_backend()``. Adding another
   physics backend requires extending this core selector.
2. The factory module path determines the backend module path. For example,
   ``isaaclab.assets.articulation`` maps to
   ``isaaclab_physx.assets.articulation``,
   ``isaaclab_newton.assets.articulation``, or
   ``isaaclab_ov.assets.articulation``. The OvPhysX backend key uses the shared
   ``isaaclab_ov`` integration package.
3. The factory lazily imports the backend module and caches the implementation
   class in a registry.

.. code-block:: text

    User code: Articulation(cfg)
        │
        ▼
    FactoryBase.__new__()
        │
        ├─ _get_backend()       → "physx", "newton", or "ovphysx"
        │    (reads SimulationContext.physics_manager)
        │
        ├─ _get_module_name()   → "isaaclab_physx.assets.articulation"
        │    (OvPhysX maps to the shared isaaclab_ov package)
        │
        ├─ importlib.import_module()
        │    (lazy load — only on first use)
        │
        └─ Return backend-specific instance

Renderers and visualizers instead select their implementations through their
configuration's ``class_type``. Their selection is independent of physics;
renderer instances are shared through the simulation registry described below.

Physics manager lifecycle
-------------------------

Each backend implements :class:`~isaaclab.physics.PhysicsManager`, the abstract
base class that owns its simulation lifecycle. Implementations initialize their
engine from a :class:`~isaaclab.sim.SimulationContext`, update kinematics with
``forward()``, advance simulation with ``step()``, reset state with ``reset()``,
and release resources with ``close()``.

The manager exposes :class:`~isaaclab.physics.PhysicsEvent` callbacks for
cross-backend lifecycle work. ``MODEL_INIT`` occurs during scene construction,
``PHYSICS_READY`` after physics initialization, and ``STOP`` before native resources are replaced
or shut down.
The concrete ``close()`` implementation dispatches the ``STOP`` event.

``SimulationContext`` owns native resources and renderer instances in one registry.
``get_or_create_backend(backend_cfg)``
reuses one resource for equal configurations of the same concrete type; a cache miss
constructs ``instantiate(backend_cfg)``.
:class:`~isaaclab.sim.BackendCfg` describes resource settings and identity, and
:class:`~isaaclab.renderers.RendererCfg` extends it for renderer instances.
``PhysicsCfg`` selects a physics manager. Finalize configurations before
registration and treat them, including nested values, as read-only afterward.
Use a new configuration for different settings. ``close_backend(backend)`` closes
the exact registered object after all consumers have released their bindings;
it does not compare or hash configurations. Resources declared through ``BackendCfg`` must
implement ``close()``. Plain construction cfgs share Python-owned data without a teardown
operation; removing the registry entry releases its reference. A failed close retains
the entry for retry. After physics shutdown invalidates camera
render data, simulation teardown closes material writers, renderer instances, visualizers,
and remaining native resources, in that order, before closing the stage.

Managers and native renderers expose their borrowed resource through ``backend``.
For example, ``NewtonManager.backend.model`` accesses the finalized native model.
Closing a renderer releases its bindings, not the shared native resource.
Exposing native handles does not replace SDP transport.

Clone preparation resolves the contexts requested by consumer configurations before native
initialization. Contexts derive from :class:`~isaaclab.cloner.ReplicateContext`; its default
``prepare(sim, routing)`` keeps the declared routes. Overrides can merge routes for a shared
representation and acquire resources through the simulation registry. For example, the OV package
can replace the native OVRTX route with an isolated OVStage route. Renderer constructors declare
requirements; they must not select another consumer's cloning path. Preparation also runs before
reset for standalone consumers registered after scene construction.
Consumers registered after physics is ready use the same preparation before their initialization.

The resulting clone contexts are registered separately as ``sim.clone_contexts[Context] = Context(...)``
before plan dispatch. They apply the plan but do not own native runtime resources. Stage population,
clone execution, and physics/render attachment follow the preparation decision.

Newton has two resources with different lifetimes, not two interchangeable backends:

* ``ModelBuilder`` holds mutable construction data. Cloning populates it and sensors declare
  requirements before finalization. It remains available for hard reset. ``NewtonBuilderCfg``
  is a plain construction cfg, not a ``BackendCfg``; the builder needs no native ``close()``.
* ``NewtonBackend`` owns the finalized model and native buffers. Physics and render consumers
  borrow those handles. Closing it releases runtime allocations without closing the builder.

Both resources use the same registry:

.. code-block:: python

    builder_cfg = NewtonBuilderCfg(physics_cfg=sim.cfg.physics)
    builder = sim.get_or_create_backend(builder_cfg)
    # Clone/import populates this builder before model allocation.
    model_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
    backend = sim.get_or_create_backend(model_cfg)

Both configurations use the selected physics cfg; non-Newton physics selects a render-only
representation. ``SimulationContext`` has no backend-specific cfg fields, and consumers do not
access clone contexts. Consumers request body transforms and visual points directly through SDP.
Queries share the native resource's BVHs but keep each consumer's captured work separate.

Portable asset and sensor interfaces
------------------------------------

Assets and sensors use the same layering as the factories:

1. A base class in ``isaaclab`` defines the public contract, such as
   ``BaseArticulation`` or ``BaseContactSensor``.
2. A factory class inherits from both :class:`FactoryBase
   <isaaclab.utils.backend_utils.FactoryBase>` and that base class.
3. Backend packages provide the supported implementations.

Data classes use the same pattern, for example
``ArticulationData(FactoryBase, BaseArticulationData)``. Implementations expose
:class:`~isaaclab.utils.warp.ProxyArray` values through public asset and sensor
data properties. Each proxy wraps the underlying ``wp.array`` and provides
explicit ``.warp`` access to that array and cached, zero-copy ``.torch`` access
to a :class:`torch.Tensor` view. Use those accessors when an API specifically
requires one representation. Passing a ``ProxyArray`` to ``wp.to_torch()`` is
supported only by a deprecated compatibility shim; new code should use
``proxy_array.torch``. Backend-native and internal storage may still use raw
Warp arrays. See :doc:`/source/how-to/proxy_array` for usage and buffer lifetime
guidance.

Portable renderer and scene-data interfaces
-------------------------------------------

Rendering is selected independently from physics. Acquire implementations of the
:class:`~isaaclab.renderers.BaseRenderer` contract through
``sim.get_or_create_backend(renderer_cfg)``. The
:class:`~isaaclab.renderers.RenderContext` coordinates their rendering lifecycle
through a filtered view of that registry, without a separate renderer cache or
renderer ownership. It validates global settings and registration timing, initializes
renderers after physics is ready, and coordinates stage preparation, scene updates,
and material writers. See
:ref:`overview_renderers` for renderer choices and usage.

Physics managers expose live simulation data through
:class:`~isaaclab.scene_data.SceneDataBackend`. The
:class:`~isaaclab.scene_data.SceneDataProvider` owned by the simulation context
converts and remaps that data for backend-independent consumers:

.. code-block:: text

   physics manager -> SceneDataBackend -> SceneDataProvider -> renderer or visualizer

This boundary lets renderers and visualizers consume a common Warp-native data
path without knowing which physics engine owns the state. See
:doc:`/source/developer-tools/scene_data_providers` for the complete
data-flow model.

Native engine access boundary
-----------------------------

The portable interfaces define the stable API boundary. Advanced code can use
each engine's native low-level data API, but those APIs intentionally keep their
own ownership and synchronization semantics. See
:doc:`/source/concepts/native-physics-api/index`
for PhysX typed views, Newton live model/state arrays and generic selections,
and OvPhysX tensor bindings.

Design principles
-----------------

- **Lazy loading:** Backend modules are imported only when first instantiated,
  keeping startup fast and avoiding dependencies on unused backends.
- **Recognized keys plus convention:** Once the core selector recognizes a
  backend key, module paths mirror the ``isaaclab.X.Y`` structure. OvPhysX
  maps to ``isaaclab_ov.X.Y``; other recognized backends use
  ``isaaclab_<backend>.X.Y`` by default.
- **Independent selection:** Physics backend, renderer, and visualizer are
  selected independently.
- **Explicit data interop:** Public asset and sensor data properties return
  :class:`~isaaclab.utils.warp.ProxyArray`; its ``.warp`` and ``.torch``
  accessors expose the required array representation without copying.
- **Zero runtime overhead:** Selection occurs at instantiation time; it does
  not add dispatch logic to the simulation hot path.
