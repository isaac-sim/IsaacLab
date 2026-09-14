isaaclab.sim.usd_export
=======================

.. automodule:: isaaclab.sim.usd_export

Classes
-------

.. currentmodule:: isaaclab.sim.usd_export

.. autosummary::
   :nosignatures:

   SceneExporter
   ArticulationExporter
   ArticulationPrimPaths

.. autoclass:: SceneExporter
   :members:

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

Fixed deployment configurations
-------------------------------

Construct a clean fixed scene for deployment instead of extracting a randomized training scene::

    from isaaclab.sim import SceneExporter

    # Start the requested backend with AppLauncher first. Supply a fixed scene cfg,
    # with num_envs=1, and no active SimulationContext.
    SceneExporter.export_from_cfg(deployment_scene_cfg, deployment_sim_cfg, "environment.usda")

The entry point copies both configurations, constructs ``InteractiveScene`` normally, initializes
physics, applies ``init_state`` through the public asset APIs, and updates kinematics. The snapshot
is taken before the first application physics step and without running task events. Backend
initialization may perform warmup steps; the configured default state is applied afterwards.
``InteractiveScene.reset_to_default`` includes rigid-object collections as well as articulations
and standalone rigid objects. Fixed-base world anchors follow the initialized root pose.

The full flattened stage preserves articulation topology, authored mass/inertia/COM, collision
geometry, per-collider materials and offsets, filtering, tendon schemas, static geometry, terrain,
lights, and shared resources. Spawner schema overrides are already present in that stage. Fixed
actuator initialization additionally writes solver gains, armature, friction and limits into backend
buffers; the exporter writes those resolved values and initial joint/body state. It does not turn
explicit controller gains into implicit USD drives. Unknown asset configuration fields, missing
physical bodies or dependencies, and required unsupported object types fail before replacing a file.

The shared property contract lives in ``isaaclab.assets.physics_properties`` beside the asset data
interfaces. It declares property sources, public ordering, units and frames, and distinguishes
construction, USD, initial-state and runtime-only fields. Backend provenance joins data rows to prim
identities; USD traversal order is not a body or joint index.

Newton fixed exports use the same scene/body/joint writers and retain source geometry. Newton
extensions supply native joint/contact properties and XPBD iteration settings. Non-default XPBD
settings without a USD representation, substeps/decimation, mixed per-joint actuator modes that
cannot be expressed by the importer, and other solver families currently fail explicitly. Newton's
import option for combined position/velocity actuation is stored in the layer because it is an
importer setting, not a USD drive attribute. A Newton deployment loader must consume it::

    stage = Usd.Stage.Open("environment.usda")
    options = stage.GetRootLayer().customLayerData.get("isaaclab:newtonImportOptions", {})
    from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

    info = builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()], **options)

It must also read the exported timestep and supported solver settings when constructing its driver.
The pinned Newton importer uses different units for initial angular velocity from Isaac Sim;
the adapter retains the standard state schema and supplies Newton's native state attributes too.
Those attributes currently use Newton's non-schema compatibility path and require revalidation
when upgrading Newton.
This does not certify Newton solver behavior in Isaac Sim; cross-backend semantics require a
separate validation.

Controllers and observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported native explicit actuators retain their ``NewtonActuator`` schemas. Isaac Sim's
``isaacsim.core.experimental.actuators`` extension provides an ``ArticulationActuators`` loader
and an OmniGraph node that attach these pipelines. Loading a USD stage alone does not attach them.
This interface was checked against GitLab Isaac Sim develop at
``b023e77b0534f11fc6e2f1a0e500d46524f5e10d``; it is experimental. Python-only controllers without
USD representations fail fixed export rather than silently losing their control law.

Policies, observation/reward/reset code, controller history, external wrenches, and sensor runtime
are outside this physical configuration export. Sensor prims may survive, but deployment code must
recreate sampling schedules, renderers, contact histories and every sensor needed by policy
observations. Referenced textures and model files must remain accessible; flattening is not asset
packaging. Isaac Sim supplies the standard OmniPBR/OmniGlass/OmniSurface MDL modules.

Runtime environment snapshots
-----------------------------

The separate runtime entry point remains available for supported snapshots::

    from isaaclab.sim import export_environment_to_usd

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

PhysX and OVPhysX share the runtime scene writer. It writes articulation, rigid-object and
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

Runtime snapshot boundaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
* Controller histories, policy networks, observation/reward/reset code, and sensor runtime state
  are not serialized; authored native actuator schemas are preserved. Camera or sensor prims may survive in USD; sensor sampling, renderers,
  contact histories and observations needed by a policy must be restored by deployment code.
* USD references and composition are flattened. Referenced asset-valued dependencies such
  as textures remain resolved asset paths; this is not a relocatable asset bundle. Those
  resources must remain accessible when the file is deployed.

Validation and deployment
^^^^^^^^^^^^^^^^^^^^^^^^^

Lightweight tests under ``source/isaaclab/test/sim`` exercise field coverage, complete authored
content preservation, selection, dependency failures, unit/frame conversion, and atomic saving.
Fixed-scene backend tests use local assets, articulation links, standalone and collected bodies,
static/shared geometry, non-default initial conditions and fixed actuator properties. Their
references come from the live backend and original USD, independently of exporter selection.
The fresh loader receives only the exported file and its supported simulation/import settings.
Isaac Sim's reset performs two integration warmup steps; the test compares the exact authored
initial conditions and then advances its independent reference through the same warmup before
comparing native body/joint state. Native actuator schema preservation is checked separately
from controller execution, which deployment code must attach.

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
