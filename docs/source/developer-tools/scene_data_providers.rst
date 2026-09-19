Scene Data Provider
===================

:class:`~isaaclab.scene_data.SceneDataProvider` bridges physics simulation backends and the
visualizers/renderers that consume scene data. It exposes a single Warp-native read path for
body transforms regardless of which physics backend (PhysX or Newton) is active, so renderers
and visualizers can stay backend-agnostic.

Overview
--------

Isaac Lab supports multiple physics backends (PhysX and Newton) and multiple visualizers
(Omniverse Kit, Newton, Rerun, Viser). Each combination needs scene data to flow from the
physics engine into the renderer or visualizer. :class:`SceneDataProvider` owns this flow: the
physics manager provides a :class:`~isaaclab.scene_data.SceneDataBackend` that wraps its native
tensor views, and the provider handles format conversion and re-mapping on top of it.

.. code-block:: python

   from isaaclab.sim import SimulationContext

   # The SimulationContext owns the active provider; consumers fetch it instead of
   # constructing one directly.
   provider = SimulationContext.instance().get_scene_data_provider()

Custom physics integrations must implement ``SceneDataBackend.transform_publication`` with a
``SceneDataPublication(native_format)``. Keep that object alive, update its native pointer when
the solver swaps buffers, and set ``publication.dirty = True`` after state writes or steps.
Render consumers must treat arrays returned by ``request_transforms`` as read-only.

Architecture
------------

The system has three layers:

1. :class:`~isaaclab.scene_data.SceneDataBackend`: a small interface implemented by each physics
   manager. It exposes the backend's transform array directly as one of the
   :class:`~isaaclab.scene_data.SceneDataFormat` Warp structs, plus the per-transform prim paths
   and total count. Its transform publication pairs the native array with a dirty latch.
   Physics marks it dirty after state writes, steps, and buffer swaps; SDP consumes the latch.

   - :attr:`SceneDataBackend.transforms`: current transforms as a Warp struct (one of
     :class:`SceneDataFormat.Vec3_Quat`, :class:`SceneDataFormat.Transform`,
     :class:`SceneDataFormat.Matrix44`, :class:`SceneDataFormat.Vec3_Matrix33`).
   - :attr:`SceneDataBackend.transform_count`: number of transforms.
   - :attr:`SceneDataBackend.transform_paths`: list of USD prim paths, one per transform.
   - :attr:`SceneDataBackend.transform_publication`: native data and its dirty latch.
   - :attr:`SceneDataBackend.points`: flattened deformable nodal positions as
     :class:`SceneDataFormat.Points` (optional; rigid-only backends return an empty buffer).
   - :attr:`SceneDataBackend.point_count`: total number of geometry points.
   - :attr:`SceneDataBackend.geometry_paths`: one USD prim path per deformable body instance.
   - :attr:`SceneDataBackend.geometry_counts`: unpadded nodal count per geometry entity.

2. :class:`~isaaclab.scene_data.SceneDataProvider`: wraps a backend and offers format conversion
   plus index re-mapping.

   - :meth:`SceneDataProvider.request_transforms`: returns an SDP-owned conversion cached by
     format, ordering, and dirty generation. A matching native format and ordering returns
     the producer's array without copying. Remapping and format conversion use one kernel;
     ``TransposedMatrix44d`` also applies static, output-indexed scales in that kernel.
   - :meth:`SceneDataProvider.get_transforms`: writes the backend's transforms into a
     consumer-provided :class:`SceneDataFormat` struct, optionally converting format
     (e.g. ``Vec3_Quat`` to ``Transform``) and applying an index mapping. When the backend
     format matches the output format and no mapping is provided, the result is a zero-copy
     passthrough.
   - :meth:`SceneDataProvider.create_mapping`: builds a remap array from the backend's prim
     paths to a consumer's desired ordering. Used when a renderer or visualizer wants
     transforms indexed by its own body list rather than by the physics view order.
   - :meth:`SceneDataProvider.get_points`: copies backend deformable nodal positions into a
     consumer buffer, optionally remapping entity slices via
     :meth:`SceneDataProvider.create_geometry_mapping`.
   - :meth:`SceneDataProvider.create_geometry_mapping`: maps backend deformable entities to
     consumer particle offsets in a shadow Newton ``particle_q`` buffer.
   - :meth:`SceneDataProvider.get_camera_transforms`: discovers per-camera, per-env world
     transforms from the USD stage.
   - :attr:`SceneDataProvider.usd_stage`: USD stage handle for stage-walking consumers.
   - :attr:`SceneDataProvider.num_envs`: environment count inferred from
     ``/World/envs/env_<id>`` prims.

3. Backend implementations:

   - ``PhysxSceneDataBackend`` (internal to :mod:`isaaclab_physx.physics`) wraps PhysX's
     ``RigidBodyView`` and exposes its transforms as :class:`SceneDataFormat.Transform`. When
     deformable bodies are present it also exposes flattened simulation nodal positions through
     :class:`SceneDataFormat.Points`.
   - ``OvPhysxSceneDataBackend`` (internal to :mod:`isaaclab_ov.physics`) mirrors the PhysX
     contract for rigid transforms and OVPhysX deformable nodal tensors.
   - ``NewtonSceneDataBackend`` (internal to :mod:`isaaclab_newton.physics`) wraps the Newton
     model's ``body_q`` and exposes it as :class:`SceneDataFormat.Transform`.

PhysX backend
-------------

When PhysX is the active physics backend, the provider reads transforms directly from PhysX's
``RigidBodyView`` (a wildcard-expanded tensor view covering every rigid body across all envs).
The transforms are returned as :class:`SceneDataFormat.Transform` (Warp ``transformf`` array),
so consumers that want this format get them zero-copy.

Newton-native consumers (Newton visualizer, Rerun, Viser, Newton Warp renderer) also need a
Newton ``Model``/``State``. Their registered ``NewtonReplicateContext`` builds this resource
from the clone plan during replication, before renderer initialization. Its ``body_q``
references the SDP result directly: native PhysX ordering can pass through, while a different
body ordering requires one cached remap. Consumers resolving the same context share one model
and state; reading them does not build another model or walk the completed stage.

OVRTX consumes rigid transforms directly through SDP's ``TransposedMatrix44d`` format.
Its mutable-geometry integration still requires the Newton resource. When the scene has
PhysX or OVPhysX deformables, that resource also allocates
``particle_q`` render slots for soft/cloth meshes, syncs simulation nodal positions through
:meth:`SceneDataProvider.get_points` with ``allow_passthrough=False`` into a separate
sim-sized buffer, and remaps or copies those positions into the render-sized ``particle_q``
buffer each frame. Volume deformables with mismatched sim and visual vertex counts use a
barycentric sim-to-visual remap so Newton Warp and OVRTX render the paired visual mesh rather
than tet simulation topology. The shadow deformable registry exposes render-slot offsets and
``particles_per_body`` counts for OVRTX point bindings.

The existing point-data API and geometry remapping remain separate from cached rigid-transform
publication. They are not zero-copy guarantees for deformables.

Newton backend
--------------

When Newton is the active physics backend, the backend wraps the Newton model's ``body_q``
directly. No shadow model or per-frame sync is needed: Newton already owns the authoritative
model and state, and the provider exposes that state as :class:`SceneDataFormat.Transform`.

Data requirements
------------------

Visualizers and renderers register their requirements before clone dispatch. Native resources
are built during cloning and consumers initialize after physics:

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

- :doc:`/source/concepts/renderers`: renderer backends that consume scene data
- :doc:`/source/concepts/visualization`: visualizer backends that consume scene data
