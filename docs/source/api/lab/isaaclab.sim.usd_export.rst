isaaclab.sim.usd_export
=======================

.. automodule:: isaaclab.sim.usd_export

Initialization boundary
-----------------------

Use ``--export_deployment_usd`` with the unified RSL-RL, RL-Games, SKRL or SB3 training
entrypoint to write ``deployment.usda`` into the run log directory. The option is off
by default; only global rank zero exports. For example::

    uv run --extra rsl-rl isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
        physics=newton_mjwarp env.scene.num_envs=1 --headless --export_deployment_usd

Environment construction calls ``scene.export_to_usd`` after physics and asset
initialization and before startup events, reset or stepping. Set
``env_cfg.scene.export_usd_path`` before construction to use the same boundary outside
training. Prestartup changes already authored to USD are retained. Startup randomization
and task-only inertia corrections are excluded; training executes these events normally.
The flag does not create a second environment or a cached nominal snapshot. No metrics
sidecar is produced. RLINF and the experimental Warp frontend are outside this integration.

A standalone scene can call ``scene.export_to_usd("environment.usda", env_id=0,
preserve_source_contacts=True)`` after ``sim.reset()`` initializes its assets, before
applying startup/task changes. A call after task construction cannot recover pre-startup
values. The method rejects export after the first physics step. It retains body placement
and authored joint defaults, omits explicit live joint-state samples, and clears initial
body, joint and nodal velocities.

Fixed materials belong in spawn configuration. The locomotion preset migration preserves
backend friction semantics: Newton has one dynamic-friction coefficient, while PhysX
retains static and dynamic values and combine modes. Removing fixed event terms changes
random-number consumption compared with the earlier task configuration. Startup inertia
events remain unchanged.

Selection and ownership
-----------------------

The scene selects registered articulations, rigid objects, collections and cables using
ClonePlan queries, retains static geometry and shared resources, and removes other
environments. ``env_id`` defaults to zero; one environment is sufficient. ClonePlan
provides layout and source variants, not subsequent property overrides. Physics-only
clones are materialized from their authored sources before selected-instance values are
written. Replicated layouts with different property values have fixture coverage;
arbitrary heterogeneous scenes are not certified.

Assets supply stable prim paths, public row indices and joint axes. Revolute/prismatic
axes are resolved through schema types; Cartesian multi-axis joints provide one axis
per public DOF. Multiple DOFs may address one USD joint prim. This does not imply that
SphericalJoint cone limits equal independent Cartesian limits; unsupported mappings fail.

The existing ``UsdWriter`` owns copying, property writes, frame conversion, conditional
material copying/binding, dependency checks and atomic saving. Backend owners interpret
native buffers and resolve collider identities. Source material bindings remain when
they already express the effective values; changed shared materials receive independent
copies with connections preserved. Before PhysX buffer overrides,
``preserve_source_contacts=True`` retains source materials and automatic contact offsets.
Distinct per-shape buffer overrides without stable collider identities are rejected.
Newton contact defaults applied during asset initialization may require explicit material
values even when no startup event has run.

Property declarations and units
-------------------------------

Data getters declare physical source units and USD targets in the same place::

    @property
    @source_units(angular="rad", linear="m")
    @usd_field(
        UsdAttribute("physics:lowerLimit", component=LimitComponent.LOWER,
                     axes=("angular", "linear")),
        UsdAttribute("limit:{axis}:physics:low", "PhysicsLimitAPI:{axis}",
                     component=LimitComponent.LOWER,
                     axes=("rotX", "rotY", "rotZ", "transX", "transY", "transZ")),
    )
    def joint_pos_limits(self):
        ...

``UsdPhysicsUnits`` maintains target units from the schema semantics; it does not infer
units from names alone or claim that USD supplies complete unit metadata. The generic
converter checks length, mass, time and angle dimensions, supports compound units, and
uses the stage's length/mass scales. Thus angular stiffness declares ``N*m/rad`` and
converts to the USD angular-drive unit, independently of joint-limit conversion.
Unknown unit rules fail explicitly.

``usd_fields`` discovers inherited getter declarations without invoking the getters.
The declaration scope distinguishes body, joint, collider, material and array properties
on shared data objects. A binding may name a boolean data property for applicability;
Newton uses this to retain native MuJoCo limits instead of authoring fallback force gains.
Inertia uses an explicit multi-output representation transform:
``principal_inertia`` reads the tensor and declared COM quaternion input, preserves an
existing principal frame where possible, and returns principal moments and axes. Each
output has its own source unit and follows the same generic USD write path. This
representation change is separate from unit conversion.

Cable properties use the same declarations on ``CableObjectData``. The owner resolves
per-environment shape indices; the writer receives complete array rows. Declared native
defaults may be omitted. The cable reader discovers those declarations at class level,
without constructing live backend data. There is no separate export field table.

Native loading and optional solver settings
-------------------------------------------

Ordinary scene/asset export does not depend on solver metadata. Load Newton rigid assets
with its native builder and schema resolvers::

    import newton
    from pxr import Usd
    from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

    stage = Usd.Stage.Open("deployment.usda")
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    info = builder.add_usd(
        stage,
        schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton(), SchemaResolverPhysx()],
        **stage.GetRootLayer().customLayerData.get("isaaclab:newtonImportOptions", {}),
    )
    model = builder.finalize(device="cuda:0")

The optional import dictionary describes joint actuation modes. It is independent of
solver settings. For cables, request ``return_deformable_results=True`` from ``add_usd``
and call ``CableObject.restore_fixed_configuration(stage, builder,
info.get("path_cable_map", {}))`` before finalization to restore contact supplements
that native curve import does not read. No duplicate deployment model/solver loader is
provided. Source terrain meshes remain in USD; a consumer's heightfield substitution
requires separate validation.

Set ``include_solver_settings=True`` on ``scene.export_to_usd`` to request available
Newton MJWarp settings. Other Newton solvers still export scene/assets; VBD, XPBD,
Kamino and coupled solver settings are not serialized or reconstructed. Determinism is
excluded. PhysX and OVPhysX keep their supported scene settings and integer timestep
frequency; Newton asset export does not require that PhysX representation. Missing optional
settings never prevent scene/asset export or loading.

MJWarp iterations use the native ``mjc:option:iterations`` attribute. Remaining supported
constructor settings are stored as JSON in ``isaaclab:newtonDriver.options``; a caller
may pass these options to ``newton.solvers.SolverMuJoCo`` and obtain iterations from
the exported physics scene. There is no automatic task reconstruction or event replay.
A deployment loop must choose its own stepping schedule.

Limits and validation
---------------------

Deformable export and its dedicated loading are deferred. The existing deformation and
coupling simulation implementations remain unchanged; exporting a scene containing a
registered deformable fails explicitly. Surface grippers are also unsupported. Source
geometry, schemas, camera prims and external resources are retained where supported,
but flattening does not package external textures or MDL dependencies.

Fresh same-backend tests compare stable entity identities, exact discrete topology and
collision relationships, and floating-point mass/inertia/COM, joint, geometry/material
and gravity values with tolerances. They verify selection from multiple environments,
source USD preservation and the initialization boundary. Non-MJWarp solver configuration
reconstruction is not a test gate. Physical round-trip results do not certify controllers,
policy observations, sensor execution or cross-backend semantics.

Policies, explicit controller code/history, observation processing, sensor sampling and
rendering, warm-start caches, active contacts, transient forces and runtime solver choices
require deployment integration. Export is an initialized physical configuration, not a
training-state checkpoint.
