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

Architecture
------------

The system has three layers:

1. :class:`~isaaclab.scene_data.SceneDataBackend`: a small interface implemented by each physics
   manager. It exposes the backend's transform array directly as one of the
   :class:`~isaaclab.scene_data.SceneDataFormat` Warp structs, plus the per-transform prim paths
   and total count. Producers mark the publication dirty after native state writes or buffer swaps.

   - :attr:`SceneDataBackend.transform_publication`: a :class:`~isaaclab.scene_data.SceneDataPublication`
     containing the current native-format pointer and dirty flag.
   - :attr:`SceneDataBackend.transforms`: the publication's data as a Warp struct (one of
     :class:`SceneDataFormat.Vec3_Quat`, :class:`SceneDataFormat.Transform`,
     :class:`SceneDataFormat.Matrix44`, :class:`SceneDataFormat.Vec3_Matrix33`).
   - :attr:`SceneDataBackend.transform_count`: number of transforms.
   - :attr:`SceneDataBackend.transform_paths`: list of USD prim paths, one per transform.
   - :attr:`SceneDataBackend.points`: flattened deformable nodal positions as
     :class:`SceneDataFormat.Points` (optional; rigid-only backends return an empty buffer).
   - :attr:`SceneDataBackend.point_count`: total number of geometry points.
   - :attr:`SceneDataBackend.geometry_paths`: one USD prim path per deformable body instance.
   - :attr:`SceneDataBackend.geometry_counts`: unpadded nodal count per geometry entity.

2. :class:`~isaaclab.scene_data.SceneDataProvider`: wraps a backend and offers format conversion
   plus index re-mapping.

   - :meth:`SceneDataProvider.request_transforms`: returns the native pointer when format and
     ordering match, or converts once per dirty generation and destination layout. Converted
     buffers belong to SDP and are shared by repeated requests. Consumers treat them as read-only.
   - :meth:`SceneDataProvider.get_transforms`: retains the caller-owned output-buffer interface
     for tools that explicitly need a copy. Rendering consumers use ``request_transforms``.
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
Newton ``Model``/``State``. Their declared cloning contexts construct that representation from
the shared clone plan before initialization. Its rigid ``body_q`` binds to SDP's requested
``Transform`` array; no intermediate per-frame copy into a second state buffer is required.
OVRTX requests ``TransposedMatrix44d`` directly from SDP, including destination ordering and
static scale in the same conversion. It no longer reads Newton state for rigid transforms.
When the scene has PhysX or OVPhysX deformables, the shadow model also allocates
``particle_q`` render slots for soft/cloth meshes, syncs simulation nodal positions through
:meth:`SceneDataProvider.get_points` with ``allow_passthrough=False`` into a separate
sim-sized buffer, and remaps or copies those positions into the render-sized ``particle_q``
buffer each frame. Volume deformables with mismatched sim and visual vertex counts use a
barycentric sim-to-visual remap so Newton Warp and OVRTX render the paired visual mesh rather
than tet simulation topology. The shadow deformable registry exposes render-slot offsets and
``particles_per_body`` counts for OVRTX point bindings.

The deformable and cable geometry bridge remains separate from this rigid-transform path.
OVRTX still uses Newton geometry metadata for those features.

Native PhysX-to-Fabric updates use the engine-owned Fabric interface through SDP. Other
physics publications convert directly into SDP's bound Fabric matrices. Renderers do not
select a physics-specific synchronization path.

Newton backend
--------------

When Newton is the active physics backend, the backend wraps the Newton model's ``body_q``
directly. No shadow model or per-frame sync is needed: Newton already owns the authoritative
model and state, and the provider exposes that state as :class:`SceneDataFormat.Transform`.

Data requirements
------------------

Visualizers and renderers declare what they need from the scene data path. This is resolved at
consumer construction time, before the shared clone plan is built:

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
