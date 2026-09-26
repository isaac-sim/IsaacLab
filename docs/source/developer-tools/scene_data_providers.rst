Scene Data Provider
===================

:class:`~isaaclab.scene_data.SceneDataProvider` bridges physics simulation backends and the
visualizers/renderers that consume scene data. It exposes a single Warp-native read path for
body transforms and visual geometry regardless of which physics backend is active, so renderers
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

Lazy-read contract
~~~~~~~~~~~~~~~~~~

Use :class:`~isaaclab.utils.buffers.TimestampedBuffer` to pair cached data with its last successful
update timestamp. Owners allocate storage; the container never allocates or converts it. Same-step
writes invalidate affected asset caches with ``reset_timestamps`` and advance SDP's publication
timestamp. Each consumer compares its own timestamp after requesting data, since resolving a native
pointer can itself detect a change. Consumers tracking only an upload need a timestamp, not another
copy of SDP's data reference.

SDP uses one request-cache lifecycle for converted transforms, point views, and explicit destinations.
Conversion functions bind the requested layout; the cache records its timestamp only after a successful
update. Matching native arrays bypass conversion. Caller-owned destinations are weakly referenced, so
their cached bindings expire when the caller releases them.

Pending work belongs to its executor. Newton's ``forward()`` and PhysX/OVPhysX's
``update_kinematics()`` check ``kinematics_dirty`` internally and clear it after success; readers
call the operation without inspecting its guard. Reordered articulation views still refresh their
own derived arrays. Sensors and Newton's shared BVH similarly use dirty flags, while device masks
select the environments requiring work. Keep eager contact reads to detect contact loss.

Sampling periods and finite differences use simulation time [s]; Fabric geometry cadence uses render
frames. Neither is a publication counter. Python guards do not execute during CUDA graph replay:
captured work must remain in the graph or use device-side invalidation. Newton retains its selective
reset masks and conservative reads after externally replayed writes.

Data flow
~~~~~~~~~

The system has three layers:

1. :class:`~isaaclab.scene_data.SceneDataBackend`: a small interface implemented by each physics
   manager. It exposes the backend's transform array directly as one of the
   :class:`~isaaclab.scene_data.SceneDataFormat` Warp structs, plus the per-transform prim paths
   and total count. Producers increment ``transforms_timestamp`` after native state writes or buffer swaps;
   SDP calls ``get_transforms(output_format)`` before reading the timestamp, since resolving the pointer
   can itself detect a swap. The default implementation returns the existing ``transforms`` property.
   This is a logical publication timestamp, not elapsed time. It never resets within the backend's
   lifetime, so independent readers cannot hide changes from one another.

   - :attr:`SceneDataBackend.transforms`: the native data as a Warp struct (one of
     :class:`SceneDataFormat.Vec3_Quat`, :class:`SceneDataFormat.Transform`,
     :class:`SceneDataFormat.Matrix44`, :class:`SceneDataFormat.Vec3_Matrix33`).
   - :attr:`SceneDataBackend.transforms_timestamp`: logical update timestamp of the native transforms.
   - :attr:`SceneDataBackend.transform_count`: number of transforms.
   - :attr:`SceneDataBackend.transform_paths`: list of USD prim paths, one per transform.
   - :attr:`SceneDataBackend.native_transform_formats`: formats published without conversion.
     PhysX publishes either packed poses or Fabric matrices and refreshes only the requested representation.
   - :meth:`SceneDataBackend.get_geometry_batches`: native point arrays or interpolation inputs,
     paired with exact visual prim paths and ranges compiled during backend construction. It returns
     the requested native representation when available, otherwise the primary representations for
     SDP to convert. The return type is always a list of batches, including native Fabric.
   - :attr:`SceneDataBackend.geometry_timestamp`: logical update timestamp, advanced for same-step
     writes and native pointer swaps. Cached outputs record the timestamp they contain, like asset
     data buffers. This is not elapsed simulation time or a shared dirty flag that a reader clears.
   - :attr:`SceneDataBackend.native_geometry_formats`: geometry formats available without conversion.

2. :class:`~isaaclab.scene_data.SceneDataProvider`: wraps a backend and offers format conversion
   plus index re-mapping.

   - :meth:`SceneDataProvider.get_transforms`: binds native arrays when format and ordering match,
     or SDP-owned buffers converted once per producer timestamp and destination layout. These shared
     arrays are read-only, including when they replace preallocated output fields. Pass
     ``allow_passthrough=False`` to write directly into caller-owned arrays instead.
   - :meth:`SceneDataProvider.create_mapping`: builds a remap array from the backend's prim
     paths to a consumer's desired ordering. Used when a renderer or visualizer wants
     transforms indexed by its own body list rather than by the physics view order.
   - :meth:`SceneDataProvider.get_geometry_points`: read-only world-space point views keyed by
     exact visual prim path. Native point ranges alias the producer; interpolation and destination
     reordering are fused into one cached conversion. Consumers with fixed native storage pass
     that array or ``FabricPoints`` as ``output`` and visual-path offsets as ``offsets``. These
     calls return the supplied destination. Destination caches retain indexing metadata, not the
     consumer's buffers, and expire with the destination.
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
For deformables with different simulation and visual meshes, the producer compiles barycentric
indices and weights from the declared prototype once. SDP applies that interpolation directly into
the Newton representation's final ``particle_q`` slots. There is no intermediate simulation-sized
buffer and no consumer-owned remap. Native PhysX padded nodal arrays are borrowed without packing.

OVRTX receives the same exact visual-path publications through SDP. It neither imports Newton
managers nor requests a Newton model. Meshes, particle clouds, and cable curves use one point-binding
path.

PhysX owns its native Fabric refresh and publishes the resulting matrices through SDP without
fetching packed poses. ``isaaclab_physx.renderers.fabric.FabricBackend`` owns the shared native stage
and hierarchy handles. Its identity is the stage and device, not the SDP source or attribute type.
``SimulationContext`` declares ``fabric_cfg`` when Kit is available. After physics initializes, Kit,
Isaac RTX, and explicit Fabric synchronization obtain the same resource through
``get_or_create_backend(sim.fabric_cfg)``. Transform bindings are state on that resource, not a
separate backend. Consumers pass the simulation's SDP to ``update_transforms(provider)``; for foreign
physics it converts directly into Fabric local matrices, then propagates the GPU hierarchy.
Core ``RenderContext`` owns no Fabric bindings.
It binds rigid destinations as Fabric-only reset-stack roots because
physics publishes absolute poses, including for nested bodies. Visual descendants still inherit
their body's transform; authored USD is unchanged. Native source indices and world scales are
bound once. Fabric's selection reuse API reports scene-wide structural changes; the resource refreshes
array views without repeating path matching or scale capture. Otherwise GPU propagation
reuses the hierarchy topology. Clean requests never acquire writable Fabric arrays.
Renderers do not select a physics-specific synchronization path.
The same Fabric resource receives geometry through ``update_geometries(provider, frame)``.
PhysX publishes its native ``FabricPoints`` without a conversion or rewrite. Foreign mesh points
are interpolated directly into GPU Fabric storage. The current Kit Hydra path requires CPU Fabric
destinations for ``Points`` and ``BasisCurves``; SDP handles their device transfer without USD
attribute writes. Only destinations whose update interval has elapsed are transferred. World-space
point destinations reset their transform stack to avoid applying the environment or body pose twice.
``FabricMatrix44`` and ``FabricPoints`` contain only array storage, not bindings or native engine handles.

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

Geometry publication
--------------------

Newton deformable and MPM positions are direct views of native ``particle_q`` ranges. Cable
publications borrow native body poses and capsule parameters; SDP derives curve endpoints once
per update timestamp. Both camera renderers and viewers consume the same cached result.

``ClonePlan`` remains a generic replication and routing description. Asset construction authors
prototype geometry; native import combines those prototypes with the plan and records native
ranges. Consumers bind to those completed resources, never rediscovering the completed stage.

.. code-block:: python

   # Default: read-only native or converted views, cached by producer timestamp.
   points_by_path = provider.get_geometry_points()

   # A consumer with fixed native storage receives the conversion directly.
   provider.get_geometry_points(output=state.particle_q, offsets=visual_path_offsets)

   # A Fabric consumer supplies native storage and its exact visual-path row indices.
   provider.get_geometry_points(output=fabric_points, offsets=visual_path_rows)

The internal flat-node queries and physics-owned geometry sync methods were removed. Rendering
consumers use ``get_geometry_points``; physics managers no longer run geometry writers from ``pre_render``.

Transform conversion caches use :class:`~isaaclab.utils.buffers.TimestampedBuffer`, the same
data-and-timestamp container used by asset data. Storage is allocated only when conversion is needed;
freshness is committed after conversion succeeds. Native matching formats still pass through without
allocation. Fabric destination replacement invalidates the cached binding even if physics is unchanged.

As in articulation and rigid-object data, the current timestamp and a cached buffer's timestamp
serve different purposes: one identifies current state; the other identifies the state in that buffer.
Asset data advances ``_sim_timestamp`` with time and invalidates dependent buffers on same-step writes.
SDP instead advances ``geometry_timestamp`` on those writes, so independently updated consumers all see
the change. The output's cache owns its freshness check; a downstream upload or BVH may need its own
invalidation, but should not repeat the conversion's cache bookkeeping.

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
     - No
     - Yes

See Also
--------

- :doc:`/source/concepts/renderers`: renderer backends that consume scene data
- :doc:`/source/concepts/visualization`: visualizer backends that consume scene data
