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
They locate backend implementations through convention rather than a central
registration table:

1. The active physics backend comes from
   ``SimulationContext.physics_manager``.
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

Some factories use a different resolution key. For example,
:class:`~isaaclab.renderers.Renderer` selects an implementation from its
renderer configuration because rendering and physics are independent.
Visualizers similarly use their ``visualizer_type`` configuration field.

Physics manager lifecycle
-------------------------

Each backend implements :class:`~isaaclab.physics.PhysicsManager`, the abstract
base class that owns its simulation lifecycle. Implementations initialize their
engine from a :class:`~isaaclab.sim.SimulationContext`, update kinematics with
``forward()``, advance simulation with ``step()``, reset state with ``reset()``,
and release resources with ``close()``.

The manager exposes :class:`~isaaclab.physics.PhysicsEvent` callbacks for
cross-backend lifecycle work. ``MODEL_INIT`` occurs during scene construction,
``PHYSICS_READY`` after physics initialization, and ``STOP`` during shutdown.
The concrete ``close()`` implementation dispatches the ``STOP`` event.

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
Warp ``wp.array`` values for asset and sensor data, allowing portable code to
interoperate with PyTorch through ``wp.to_torch()`` when needed.

Portable renderer and scene-data interfaces
-------------------------------------------

Rendering is selected independently from physics. Renderer configurations
dispatch through :class:`~isaaclab.renderers.Renderer` to implementations that
share the :class:`~isaaclab.renderers.BaseRenderer` contract, with
:class:`~isaaclab.renderers.RenderContext` owning their lifecycle. See
:ref:`overview_renderers` for renderer choices and usage.

Physics managers expose live simulation data through
:class:`~isaaclab.scene_data.SceneDataBackend`. The
:class:`~isaaclab.scene_data.SceneDataProvider` owned by the simulation context
converts and remaps that data for backend-independent consumers:

.. code-block:: text

   physics manager -> SceneDataBackend -> SceneDataProvider -> renderer or visualizer

This boundary lets renderers and visualizers consume a common Warp-native data
path without knowing which physics engine owns the state. See
:doc:`/source/overview/core-concepts/scene_data_providers` for the complete
data-flow model.

Native engine access boundary
-----------------------------

The portable interfaces define the stable API boundary. Advanced code can use
each engine's native low-level data API, but those APIs intentionally keep their
own ownership and synchronization semantics. See
:doc:`/source/how-to/native_physics_api/index`
for PhysX typed views, Newton live model/state arrays and generic selections,
and OvPhysX tensor bindings.

Design principles
-----------------

- **Lazy loading:** Backend modules are imported only when first instantiated,
  keeping startup fast and avoiding dependencies on unused backends.
- **Convention over configuration:** Backend module paths mirror the
  ``isaaclab.X.Y`` structure. OvPhysX maps to ``isaaclab_ov.X.Y``; other
  backends use ``isaaclab_<backend>.X.Y`` packages, so no manual registration
  is required.
- **Independent selection:** Physics backend, renderer, and visualizer are
  selected independently.
- **Warp-native data types:** Backend implementations return ``wp.array`` for
  asset and sensor data.
- **Zero runtime overhead:** Selection occurs at instantiation time; it does
  not add dispatch logic to the simulation hot path.
