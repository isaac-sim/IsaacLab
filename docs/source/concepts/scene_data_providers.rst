.. _scene-data-provider:

Scene Data Provider
===================

:class:`~isaaclab.scene_data.SceneDataProvider` supplies body transforms and visual
geometry to renderers and visualizers. It reads from the active physics backend
and converts or reorders Warp arrays as needed. Use it when building a renderer,
visualizer, or another consumer of scene transforms and geometry. Task code usually
reads state through the asset and sensor APIs instead.

Overview
--------

The physics manager exposes native data through
:class:`~isaaclab.scene_data.SceneDataBackend`. Get the provider from the simulation
context:

.. code-block:: python

   from isaaclab.sim import SimulationContext

   # The SimulationContext owns the active provider; consumers fetch it instead of
   # constructing one directly.
   provider = SimulationContext.instance().get_scene_data_provider()

Architecture
------------

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
     transforms from the USD stage. Runtime visualizers use bound scene cameras instead.
   - :attr:`SceneDataProvider.num_envs`: world count from :class:`~isaaclab.cloner.ClonePlan`,
     or ``0`` before a plan is published. Native-only clones need no corresponding USD prims.

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

Newton visualizers, Rerun, Viser, and the Newton Warp renderer also need a Newton
``Model``/``State``. Their clone contexts build it before initialization. Its rigid
``body_q`` binds to SDP's ``Transform`` array without an intermediate copy.
OVRTX requests ``TransposedMatrix44d`` directly, with ordering and scale handled
in the same conversion.

For deformables with separate simulation and visual meshes, the producer computes
barycentric indices and weights once. SDP interpolates directly into the Newton
representation's ``particle_q`` slots. It borrows padded PhysX nodal arrays without
packing; consumers need no intermediate buffer or remapping.

OVRTX receives the same exact visual-path publications through SDP. It neither imports Newton
managers nor requests a Newton model. Meshes, particle clouds, and cable curves use one point-binding
path.

PhysX publishes native Fabric matrices through SDP without fetching packed poses.
``isaaclab_physx.renderers.fabric.FabricBackend`` owns the stage and hierarchy
handles, keyed by stage and device. After physics initializes, Kit, Isaac RTX, and
explicit Fabric synchronization share the resource through
``sim.get_or_create_backend(FabricBackendCfg(stage=sim.stage, device=sim.device))``.

Consumers call ``update_transforms(provider)`` with the simulation's SDP. For other
physics backends, it converts poses into Fabric local matrices and updates the GPU
hierarchy. Transform bindings belong to this resource; ``RenderContext`` owns none.
Rigid destinations use Fabric-only reset-stack roots because physics provides
absolute poses. Visual descendants still inherit their body's transform, and USD
is unchanged.

Source indices and world scales are bound once. When Fabric reports a structural
change, the resource refreshes array views without repeating path matching or scale
capture. Otherwise, it reuses the hierarchy. Unchanged requests do not acquire
writable Fabric arrays. Renderers use this same path regardless of physics backend.

``update_geometries(provider, frame)`` handles geometry. PhysX publishes native
``FabricPoints`` directly; other backends interpolate mesh points into GPU Fabric
storage. Kit Hydra requires CPU destinations for ``Points`` and ``BasisCurves``,
so SDP transfers them when their update interval elapses, without writing USD.
World-space points reset their transform stack to avoid applying the body or
environment pose twice. ``FabricMatrix44`` and ``FabricPoints`` hold arrays only,
not bindings or engine handles.

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

Lazy-read contract
------------------

Use :class:`~isaaclab.utils.buffers.TimestampedBuffer` to pair cached data with its last successful
update timestamp. Owners allocate storage; the container never allocates or converts it. Same-step
writes invalidate affected asset caches with ``reset_timestamps`` and advance SDP's publication
timestamp. Each consumer compares its own timestamp after requesting data, since resolving a native
pointer can itself detect a change. Consumers tracking only an upload need a timestamp, not another
copy of SDP's data reference.

SDP readers cache buffers and mapping data with the same timestamp guard used by asset data, without
binding factories or stored update callbacks. Matching native arrays bypass conversion. Caller-owned
destinations are weakly referenced, so their cached mappings expire when the caller releases them.

Pending work belongs to its executor. Newton's ``forward()`` and PhysX/OVPhysX's
``update_kinematics()`` check ``kinematics_dirty`` internally and clear it after success; readers
call the operation without inspecting its guard. Reordered articulation views still refresh their
own derived arrays. Sensors and Newton's shared BVH similarly use dirty flags, while device masks
select the environments requiring work. Keep eager contact reads to detect contact loss.

Sampling periods and finite differences use simulation time [s]; Fabric geometry cadence uses render
frames. Neither is a publication counter. Python guards do not execute during CUDA graph replay:
captured work must remain in the graph or use device-side invalidation. Newton retains its selective
reset masks and conservative reads after externally replayed writes.

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

.. raw:: html

   <span id="see-also"></span>

Where to go next
----------------

- :doc:`/source/concepts/renderers`: renderer backends that consume scene data
- :doc:`/source/concepts/visualization`: visualizer backends that consume scene data
