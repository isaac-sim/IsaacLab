Scene Data Provider
===================

The :class:`~isaaclab.scene_data.SceneDataProvider` bridges physics simulation
backends and the visualizers/renderers that consume scene data. It exposes a single Warp-native
read path for body transforms regardless of which physics backend (PhysX or Newton) is active,
so renderers and visualizers can stay backend-agnostic.

Overview
--------

Isaac Lab supports multiple physics backends (PhysX and Newton) and multiple visualizers
(Omniverse Kit, Newton, Rerun, Viser). Each combination needs scene data to flow from the
physics engine into the renderer or visualizer. The :class:`SceneDataProvider` owns this flow:
the physics manager provides a :class:`~isaaclab.scene_data.SceneDataBackend` that wraps its
native tensor views, and the provider handles format conversion and re-mapping on top of it.

.. code-block:: python

   from isaaclab.sim import SimulationContext

   # The SimulationContext owns the active provider; consumers fetch it instead of
   # constructing one directly.
   provider = SimulationContext.instance().get_scene_data_provider()

Architecture
------------

The system has three layers:

1. :class:`~isaaclab.scene_data.SceneDataBackend` — small interface implemented by each physics
   manager. It exposes the backend's transform array directly as one of the
   :class:`~isaaclab.scene_data.SceneDataFormat` Warp structs, plus the per-transform prim paths
   and total count. There is no per-frame "update" call — the property accessors return live
   views into the underlying tensor each time they're read.

   - :attr:`SceneDataBackend.transforms` — current transforms as a Warp struct (one of
     :class:`SceneDataFormat.Vec3_Quat`, :class:`SceneDataFormat.Transform`,
     :class:`SceneDataFormat.Matrix44`, :class:`SceneDataFormat.Vec3_Matrix33`).
   - :attr:`SceneDataBackend.transform_count` — number of transforms.
   - :attr:`SceneDataBackend.transform_paths` — list of USD prim paths, one per transform.

2. :class:`~isaaclab.scene_data.SceneDataProvider` — wraps a backend and offers
   format conversion plus index re-mapping:

   - :meth:`SceneDataProvider.get_transforms` — write the backend's transforms into a
     consumer-provided :class:`SceneDataFormat` struct, optionally converting format
     (e.g. ``Vec3_Quat`` → ``Transform``) and applying an index mapping. When the backend
     format matches the output format and no mapping is provided, the result is a zero-copy
     passthrough.
   - :meth:`SceneDataProvider.create_mapping` — build a remap array from the backend's prim
     paths to a consumer's desired ordering. Used when a renderer or visualizer wants
     transforms indexed by its own body list rather than by the physics view order.
   - :meth:`SceneDataProvider.get_camera_transforms` — discover per-camera, per-env
     world transforms from the USD stage.
   - :attr:`SceneDataProvider.usd_stage` — USD stage handle for stage-walking consumers.
   - :attr:`SceneDataProvider.num_envs` — environment count inferred from
     ``/World/envs/env_<id>`` prims.

3. Backend implementations:

   - ``PhysxSceneDataBackend`` (internal to :mod:`isaaclab_physx.physics`) wraps PhysX's
     ``RigidBodyView`` and exposes its transforms as :class:`SceneDataFormat.Transform`.
   - ``NewtonSceneDataBackend`` (internal to :mod:`isaaclab_newton.physics`) wraps the
     Newton model's ``body_q`` and exposes it as :class:`SceneDataFormat.Transform`.

PhysX backend
-------------

When PhysX is the active physics backend, the provider reads transforms directly from PhysX's
``RigidBodyView`` (a wildcard-expanded tensor view covering every rigid body across all envs).
The transforms are returned as :class:`SceneDataFormat.Transform` (Warp ``transformf`` array),
so consumers that want this format get them zero-copy.

Newton-native consumers (Newton visualizer, Rerun, Viser, Newton Warp renderer, OVRTX renderer)
additionally need a Newton ``Model``/``State`` to render against. To satisfy that requirement,
:class:`~isaaclab_newton.physics.NewtonManager` builds a **shadow Newton model** from the USD
stage on first access and updates its ``body_q`` from the PhysX backend each render frame.
This is hidden behind :meth:`NewtonManager.get_model` / :meth:`NewtonManager.get_state`, so
renderers don't need to know which physics backend is active.

Newton backend
--------------

When Newton is the active physics backend, the backend wraps the Newton model's ``body_q``
directly. No shadow model or per-frame sync is needed — Newton already owns the authoritative
model and state, and the provider exposes that state as
:class:`SceneDataFormat.Transform`.

Data requirements
-----------------

Visualizers and renderers declare what they need from the scene data path. This is resolved at
simulation-context construction time and is what triggers the shadow-model build for PhysX:

.. list-table::
   :header-rows: 1

   * - Component
     - Requires Newton model
     - Requires USD stage
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
   * - Isaac RTX renderer
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
