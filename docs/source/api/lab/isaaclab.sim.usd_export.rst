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
values. The method rejects export after the first physics step. It retains root placement
and source joint defaults. It neither samples nor clears body, joint or nodal velocities;
existing source state remains untouched. Arbitrary post-initialization buffer mutations
are outside this fixed-configuration export contract.

Fixed materials belong in spawn configuration. The locomotion preset migration preserves
backend friction semantics: Newton has one dynamic-friction coefficient, while PhysX
retains static and dynamic values and combine modes. Removing fixed event terms changes
random-number consumption compared with the earlier task configuration. Startup inertia
events remain unchanged.

Selection and ownership
-----------------------

The scene selects registered articulations, rigid objects, collections, deformables and cables using
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
native buffers and select fixed overrides missing from USD. Imported properties,
automatic defaults and source material bindings are preserved without buffer writeback.
``preserve_source_contacts`` remains accepted for compatibility; source contact settings
are now always retained. Newton builder defaults require writeback only when configured
differently from native defaults and not superseded by source shape or material opinions.
Material copies are shared between consumers requesting the same necessary override.

Data-property ``@usd_field(actuator_config="...")`` declarations bind actuator configuration
keys to public data fields. The actuator framework discovers these inherited bindings for
imported-default reads and initialization override selection. ``ActuatorControl.get_joint_property_overrides``
reports affected public fields and instance/joint masks before backend assignment;
``ActuatorCollection`` records the identities after successful backend assignment, without
additional value snapshots. The exporter
consumes those identities and the existing data-property USD decorators, without a separate
actuator configuration table.
``None`` inherits the source; scalars and dictionaries select the whole actuator group,
preserving the existing unmatched-dictionary zero-fill behavior. Explicit control can
also require disabled solver gains. Only selected fields are read during export, and
equivalent values are skipped before changing schemas, bindings or instanceability.
Body mass/inertia/COM and articulation child transforms remain as authored in the source.

Property declarations and units
-------------------------------

Data getters declare USD targets and any angular conversion in the same place::

    @property
    @usd_field(
        UsdAttribute("physics:lowerLimit", component=LimitComponent.LOWER,
                     axes=("angular", "linear")),
        UsdAttribute("limit:{axis}:physics:low", "PhysicsLimitAPI:{axis}",
                     component=LimitComponent.LOWER,
                     axes=("rotX", "rotY", "rotZ", "transX", "transY", "transZ")),
        angular_conversion=radians_to_degrees,
    )
    def joint_pos_limits(self):
        ...

Actuator configuration is declared alongside its USD target, rather than in a second
export mapping::

    @property
    @usd_field(
        UsdAttribute("drive:{axis}:physics:stiffness", "PhysicsDriveAPI:{axis}"),
        angular_conversion=per_radian_to_per_degree,
        actuator_config="stiffness",
    )
    def joint_stiffness(self):
        ...

``usd_actuator_fields`` discovers these bindings without evaluating getters. Backend
overrides inherit the configuration binding even when they replace or extend USD targets.

Deployment requires the meter/kilogram stage convention set by ``SimulationContext``.
Angular limits and rates use ``radians_to_degrees``; angular gains use
``per_radian_to_per_degree``. Linear properties and other SI values pass through.
These two conversions apply only to angular axes. Non-SI stages fail explicitly.

``usd_fields`` discovers inherited getter declarations without invoking the getters.
The declaration scope distinguishes body, joint, collider, material and array properties
on shared data objects. A binding may name a boolean data property for applicability;
Newton uses this to retain native MuJoCo limits instead of authoring fallback force gains.
Inertia uses an explicit multi-output representation transform:
``principal_inertia`` reads the tensor and declared COM quaternion input, preserves an
existing principal frame where possible, and returns principal moments and axes. Each
output follows the same generic USD write path. This
representation change is separate from unit conversion.

Cable properties use the same declarations on ``CableObjectData``. The owner resolves
per-environment shape indices; the writer receives only array fields selected by fixed
contact-default overrides. The cable reader discovers those declarations at class level,
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
provided. Source terrain meshes remain in USD. Newton's native USD reader does not
interpret the terrain heightfield marker. A consumer matching Isaac Lab's terrain
representation can reuse its existing geometry adapter before ``add_usd``::

    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics.contact_data import NewtonContactData

    terrain_paths = NewtonManager._inject_terrain_heightfields(stage, builder, root_paths=("/",), device="cuda:0")
    for row, shape_path in enumerate(builder.shape_label):
        NewtonContactData.restore_fixed_configuration(stage.GetPrimAtPath(shape_path), builder, row)
    # Pass ignore_paths=terrain_paths to add_usd to avoid importing the terrain twice.

This reuses normal terrain construction and reads contacts from the data declarations;
it does not depend on solver metadata. It is
an Isaac Lab adapter, not a native Newton/USD heightfield schema. Loading the mesh directly
is a different geometry representation and needs its own physical-semantics validation.

Set ``include_solver_settings=True`` on ``scene.export_to_usd`` to request available
Newton MJWarp, VBD, XPBD, Kamino and proxy-coupled settings. Each manager owns its
constructor provenance and restoration; VBD collision scheduling and Kamino nested
configuration use their manager's ``load_exported_solver`` method. Determinism is
excluded. PhysX and OVPhysX keep their supported scene settings, including the frequency
used to cook automatic contacts. The integration timestep [s] is recorded separately in
``stage.GetRootLayer().customLayerData["isaaclab:physicsDt"]``; use it when stepping the
fresh backend rather than inferring it from the cooking frequency. Missing optional
settings never prevent scene/asset export or loading.

MJWarp iterations use the native ``mjc:option:iterations`` attribute. Remaining supported
constructor settings are stored as JSON in ``isaaclab:newtonDriver.options``; a caller
may pass these options to ``newton.solvers.SolverMuJoCo`` and obtain iterations from
the exported physics scene. There is no automatic task reconstruction or event replay.
A deployment loop must choose its own stepping schedule.

Limits and validation
---------------------

Registered deformables use their owning backend adapters; unregistered deformables and
surface grippers are rejected. Source
geometry, schemas, camera prims and external resources are retained where supported,
but flattening does not package external textures or MDL dependencies.

Fresh same-backend tests compare stable entity identities, exact discrete topology and
collision relationships, and floating-point mass/inertia/COM, joint, geometry/material
and gravity values with tolerances. They verify selection from multiple environments,
source USD preservation and the initialization boundary. Optional solver settings and
proxy ownership are compared after reconstruction. Physical round-trip results do not certify controllers,
policy observations, sensor execution or cross-backend semantics.

Policies, explicit controller code/history, observation processing, sensor sampling and
rendering, warm-start caches, active contacts, transient forces and runtime solver choices
require deployment integration. Export is an initialized physical configuration, not a
training-state checkpoint.

Deformable loading
------------------

Deformable source points, topology, rest geometry and materials remain in USD. Newton
records its fixed initialization override of edge rest angles through the same selected
array writer used by cables. Derived particle masses/radii and unmodified defaults are
not written back. PhysX preserves its cooked schemas and authored materials. Later
particle mutations, pinning and runtime material randomization are outside this boundary.
Existing initialization is unchanged.

Newton consumers can reuse ``add_exported_deformables_to_builder(stage, builder)`` from
``isaaclab_contrib.deformable.deformable_object`` before native ``add_usd``. Pass the
returned ``ignore_paths`` to native import, color a VBD builder, and restore each
entry's ``inverse_masses`` after finalization (which otherwise derives them from mass).
This adapter shares the normal asset reader and construction path; no task events run.
The fresh-loader recipe in ``isaaclab_newton/test/sim/test_usd_export.py`` demonstrates
terrain, cable and deformable supplements together.

Proxy coupling additionally records entry ownership, substeps and proxy relationships
by prim identity and local node index. ``NewtonCouplerManager.load_exported_solver``
accepts the exported JSON options and a ``(mesh_path, node_index)`` to native particle
index map derived from the returned particle ranges. ADMM coupling and arbitrary
user-defined solver factories have no export representation. Global Newton soft contact
constants live in ``isaaclab:newtonSoftContacts`` and must be applied to the fresh model.
Controllers, sensors and observation execution remain external deployment components.
