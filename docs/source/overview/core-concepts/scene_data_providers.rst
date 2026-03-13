Scene Data Providers
====================

Scene Data Providers bridge physics simulation backends and visualization/rendering systems in
Isaac Lab. They provide a unified interface for accessing scene data (transforms, velocities,
Newton model/state) regardless of which physics backend is active.

Overview
--------

Isaac Lab supports multiple physics backends (PhysX and Newton) and multiple visualizers
(Omniverse Kit, Newton, Rerun, Viser). Each combination requires scene data to flow from the
physics engine to the renderer. Scene Data Providers handle this translation automatically
through a factory pattern.

.. code-block:: python

   from isaaclab.physics import SceneDataProvider

   # Factory auto-selects the correct implementation based on active physics backend
   provider = SceneDataProvider(stage, simulation_context)

Architecture
------------

The system has three layers:

1. **BaseSceneDataProvider** — abstract interface defining the contract:

   - ``update(env_ids)`` — refresh cached scene data
   - ``get_newton_model()`` — return Newton model handle (if available)
   - ``get_newton_state(env_ids)`` — return Newton state handle (if available)
   - ``get_usd_stage()`` — return USD stage handle (if available)
   - ``get_transforms()`` — return body transforms
   - ``get_velocities()`` — return body velocities
   - ``get_contacts()`` — return contact data
   - ``get_camera_transforms()`` — return per-camera, per-env transforms
   - ``get_metadata()`` — return backend metadata (num_envs, gravity, etc.)

2. **SceneDataProvider** — factory that auto-selects the backend-specific implementation
   based on the active :class:`~isaaclab.physics.PhysicsManager`.

3. **Backend implementations:**

   - :class:`~isaaclab_physx.scene_data_providers.PhysxSceneDataProvider`
   - :class:`~isaaclab_newton.scene_data_providers.NewtonSceneDataProvider`

PhysX Scene Data Provider
-------------------------

When PhysX is the active physics backend, the provider **builds and maintains a Newton model
from the USD stage**, then syncs PhysX transforms into it each frame. This is necessary because
Newton-based visualizers (Newton, Rerun, Viser) require a Newton model/state to render.

The sync pipeline:

1. Reads transforms from PhysX ``RigidBodyView`` (fast tensor API)
2. Falls back to ``XformPrimView`` for bodies not covered by the rigid body view
3. Converts and writes merged poses into the Newton state via Warp kernels

Newton Scene Data Provider
--------------------------

When Newton is the active physics backend, the provider **delegates directly to the Newton
manager** — no building or syncing required. Newton already owns the authoritative model and
state.

The only additional work is **optional USD sync**: when an Omniverse Kit visualizer is active,
the provider syncs Newton transforms to the USD stage so Kit can render them. For Newton-only
or Rerun/Viser visualizers, this sync is skipped.

Data Requirements
-----------------

Visualizers and renderers declare their data needs, and the provider is configured accordingly:

.. list-table::
   :header-rows: 1

   * - Component
     - Requires Newton Model
     - Requires USD Stage
   * - Kit visualizer
     - No
     - Yes
   * - Newton visualizer
     - Yes
     - No
   * - Rerun visualizer
     - Yes
     - No
   * - Viser visualizer
     - Yes
     - No
   * - RTX renderer
     - No
     - Yes
   * - Newton Warp renderer
     - Yes
     - No
   * - OVRTX renderer
     - Yes
     - Yes

See Also
--------

- :doc:`renderers` — renderer backends that consume scene data
- :doc:`/source/features/visualization` — visualizer backends that consume scene data
