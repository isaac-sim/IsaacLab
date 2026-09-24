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
   and total count. Producers increment ``transforms_version`` after native state writes or buffer swaps;
   SDP calls ``get_transforms(output_format)`` before reading the version, since resolving the pointer
   can itself detect a swap. The default implementation returns the existing ``transforms`` property.
   The version never resets, so independent readers cannot hide changes from one another.

   - :attr:`SceneDataBackend.transforms`: the native data as a Warp struct (one of
     :class:`SceneDataFormat.Vec3_Quat`, :class:`SceneDataFormat.Transform`,
     :class:`SceneDataFormat.Matrix44`, :class:`SceneDataFormat.Vec3_Matrix33`).
   - :attr:`SceneDataBackend.transforms_version`: monotonic version of the native transforms.
   - :attr:`SceneDataBackend.transform_count`: number of transforms.
   - :attr:`SceneDataBackend.transform_paths`: list of USD prim paths, one per transform.
   - :attr:`SceneDataBackend.native_transform_formats`: formats published without conversion.
     PhysX publishes either packed poses or Fabric matrices and refreshes only the requested representation.
   - :attr:`SceneDataBackend.points`: flattened deformable nodal positions as
     :class:`SceneDataFormat.Points` (optional; rigid-only backends return an empty buffer).
   - :attr:`SceneDataBackend.point_count`: total number of geometry points.
   - :attr:`SceneDataBackend.geometry_paths`: one USD prim path per deformable body instance.
   - :attr:`SceneDataBackend.geometry_counts`: unpadded nodal count per geometry entity.

2. :class:`~isaaclab.scene_data.SceneDataProvider`: wraps a backend and offers format conversion
   plus index re-mapping.

   - :meth:`SceneDataProvider.get_transforms`: binds native arrays when format and ordering match,
     or SDP-owned buffers converted once per producer version and destination layout. These shared
     arrays are read-only, including when they replace preallocated output fields. Pass
     ``allow_passthrough=False`` to write directly into caller-owned arrays instead.
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

PhysX owns its native Fabric refresh and publishes the resulting matrices through SDP without
fetching packed poses. For other physics backends, the shared ``RenderContext`` binds Fabric local
matrices and asks SDP to convert directly into them, then propagates the GPU hierarchy.
It binds rigid destinations as Fabric-only reset-stack roots because
physics publishes absolute poses, including for nested bodies. Visual descendants still inherit
their body's transform; authored USD is unchanged. Native source indices and world scales are
bound once. Fabric's selection reuse API reports scene-wide structural changes; ``RenderContext`` refreshes
array views without repeating path matching or scale capture. Otherwise GPU propagation
reuses the hierarchy topology. Clean requests never acquire writable Fabric arrays.
Renderers do not select a physics-specific synchronization path.
``FabricMatrix44`` contains only matrix storage, not bindings or native engine handles.

Newton backend
--------------

When Newton is the active physics backend, the backend wraps the Newton model's ``body_q``
directly. No shadow model or per-frame sync is needed: Newton already owns the authoritative
model and state, and the provider exposes that state as :class:`SceneDataFormat.Transform`.

Native reads reconcile pending authored state writes once. A new physics publication does not
itself request forward kinematics. Kit/RTX requests current Fabric transforms through SDP
before rendering, without issuing an additional physics ``forward()``. Headless viewport
capture requests these transforms on demand rather than on every visualizer step.

Externally replayed CUDA graphs do not call Python write hooks. After writes have been captured,
Newton conservatively republishes transforms when read so an unannounced replay cannot leave
rendering stale. Those reads do not benefit from clean-publication caching.

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
