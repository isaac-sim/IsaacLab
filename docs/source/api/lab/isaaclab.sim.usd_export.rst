isaaclab.sim.usd_export
=======================

.. automodule:: isaaclab.sim.usd_export

Classes
-------

.. currentmodule:: isaaclab.sim.usd_export

.. autosummary::
   :nosignatures:

   ArticulationExporter
   ArticulationPrimPaths

.. autoclass:: ArticulationExporter
   :members:
   :show-inheritance:

.. autoclass:: ArticulationPrimPaths
   :members:
   :show-inheritance:

Functions
---------

.. currentmodule:: isaaclab.sim.usd_export

.. autosummary::
   :nosignatures:

   write_articulation_state_to_stage
   export_articulation_to_usd
   export_environment_to_usd

.. autofunction:: write_articulation_state_to_stage

.. autofunction:: export_articulation_to_usd

.. autofunction:: export_environment_to_usd

Environment snapshots
---------------------

Use the scene-level entry point for deployment::

    from isaaclab.sim import export_environment_to_usd

    # Initialize, reset, apply startup/reset randomization, then refresh observations/data.
    # Pause stepping and complete pending backend writes before taking the snapshot.
    scene.update(0.0)
    export_environment_to_usd(scene, "environment.usda", env_index=1)

``env_index`` selects a ClonePlan environment id, not a tensor row. The export retains that
id's source variants, all registered articulations and rigid objects, static geometry,
shared ground/lights/materials, and authored dependencies. Other replicas are removed.
Source variants supply authored geometry and schemas; they do not supply runtime overrides.
USD changes made during spawning or subsequently authored to the destination are retained.
The backend adapter replaces buffer-only physical values after selection. Missing body
coverage, unresolved non-filter relationships and unsupported physical components fail
before the destination is replaced.

The snapshot point is after initialization and parameter overrides, between simulation
steps, with no concurrent writes. This is a physical scene snapshot, not a training or
solver checkpoint. PhysX-family snapshots also write body poses/velocities and supported
joint positions/velocities. Newton currently records the initialized model placement;
its solver's transient integration state is not exported. Neither format preserves
contact warm-start caches, pending wrenches, random-number-generator state or reset logic.

Supported physical configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PhysX and OVPhysX share the scene writer. It writes articulation, rigid-object and
rigid-object-collection mass, full inertia and COM; joint stiffness, damping, armature,
friction and position/velocity/effort limits; body gravity-disable flags; collision
materials and contact/rest offsets; and scene gravity and timestep. Inertia is expressed
as principal moments and axes in USD, retaining products of inertia. Collision geometry,
joint frames/topology, filtering, combine modes and other authored scene settings remain
in the flattened source composition.

Newton's scene adapter reads all selected-world and shared rigid bodies/shapes/joints,
including runtime mass/inertia/COM, contact parameters, shape geometry, joint properties,
collision exclusions and gravity. It retains source visuals, shader bindings and shared
scene content, and reconstructs effective collision geometry where necessary. Source
visual meshes are kept separately when collision hulls are written. The model-only
``export_model_to_usd`` entry point remains available for callers without a scene, but
cannot preserve source content absent from its model.

Current boundaries
^^^^^^^^^^^^^^^^^^

* Deformable and cable objects, surface-gripper runtime state, and runtime tendon
  properties are not supported by the environment exporter. Newton additionally rejects
  unsupported joint/geometry types, procedural entities without USD identity, and meshes
  authored directly on a rigid-body prim when their visual geometry cannot be separated.
* PhysX-family tensor views do not expose collider prim paths. An object's multiple
  colliders can be written when their exported buffer properties are uniform; distinct
  per-collider values are rejected rather than assigned by an unverified traversal order.
* Environment exports require SI stage units. Newton also exports the live XPBD iteration
  count; its explicit collision pipeline supplies effective candidate pairs. Other solver-specific
  settings beyond the explicitly authored USD timestep/gravity are not reconstructed from the solver instance.
  Arbitrary native solver modifications, plugin state and constraints outside the supported
  property set require separate handling and validation.
* Controllers, policy networks, actuator computation/history (including delays, saturation
  logic and learned actuators), observation/reward/reset code, and sensor runtime state
  are not serialized. Camera or sensor prims may survive in USD; sensor sampling, renderers,
  contact histories and observations needed by a policy must be restored by deployment code.
* USD references and composition are flattened. Referenced asset-valued dependencies such
  as textures remain resolved asset paths; this is not a relocatable asset bundle. Those
  resources must remain accessible when the file is deployed.

Validation and deployment
^^^^^^^^^^^^^^^^^^^^^^^^^

The backend export tests initialize multiple objects in two environments, apply different
runtime overrides, export one id and load its USD into a fresh backend without rebuilding
the original task or reapplying its overrides. Newton compares the physical entity graph
by prim identity, joint endpoints/frames, collision geometry, mass/inertia/COM, joint and
contact properties, collision exclusions and gravity, using exact comparisons for discrete
relationships and tolerances for floating-point values. PhysX-family tests compare loaded
runtime properties and entity coverage, including inertia and per-shape material buffers.

These tests establish the exercised same-backend support boundary; they do not certify
all tasks or unsupported components. A policy deployment must restore its runtime
controllers and observations and validate the complete supported configuration. Deployment
on another backend needs a separate validation of physical semantics, including drive,
friction, collision and gravity behavior. Matching export hashes or a small list of scalar
USD attributes does not establish that equivalence.
