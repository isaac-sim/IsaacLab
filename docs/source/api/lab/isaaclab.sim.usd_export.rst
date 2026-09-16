isaaclab.sim.usd_export
=======================

.. automodule:: isaaclab.sim.usd_export

Fixed deployment export
-----------------------

Use ``--export_deployment_usd`` with the unified RSL-RL, RL-Games, SKRL or SB3 training
entrypoint to write ``deployment.usda`` beside the run's configuration files in ``log_dir``.
The option is off by default. Only global rank zero exports; RLINF and the experimental
Warp task frontend are not supported by this integration.

The normal task construction exports environment zero after backend and asset initialization,
before ``startup`` events and the first reset or step. Training then runs its startup events
normally. There is no second environment, saved nominal snapshot or export-only script.
Set ``env_cfg.scene.export_usd_path`` to use the same boundary outside the training entrypoint.
``prestartup`` USD edits are already present and are retained; this path does not undo them.

Fixed physical properties belong in asset configuration. Material overrides are authored by
the spawner and actuator settings are resolved during initialization. Startup inertia
corrections remain task events and are excluded from the deployment artifact.
Moving material settings from events requires preserving backend semantics: Newton's importer
uses dynamic friction for its single coefficient, whereas its material randomizer uses the
static-friction range. Referenced instance colliders must also be reachable by material binding.
Removing fixed randomizers changes random-number consumption, so the same training seed may
produce different samples than before the configuration migration.

The exporter reads initialized physical properties, retains body placement and authored joint
defaults, and writes zero initial body/joint velocities. Backend warmup velocities are not
deployment initial conditions. It does not reset the scene or apply
``default_*`` buffers. Newton delivers queued property-change notifications to its native solver
without advancing physics. Tasks must finish and register their physical scene before this
boundary; unregistered enabled bodies fail completeness checks. Physics added or changed later
in a task constructor is not part of the automatic artifact and must move into scene/asset setup.
A task that steps before this boundary cannot use initialization-only export.

Direct calls to ``InteractiveScene.export_to_usd`` still export the current scene at the time of
the call, with the same body-placement/zero-velocity policy. To export nominal properties,
call before randomization; a manual call after startup
retains its effective changes. Neither mode serializes controllers, observation processing or
sensor execution for policy deployment.

For a standalone robot and box, launch the backend normally and use the scene's export method.
Here ``sim_cfg`` is the resolved simulation configuration for that backend::

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

    @configclass
    class DeploymentSceneCfg(InteractiveSceneCfg):
        robot = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.5)),
        )

    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(DeploymentSceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        scene.export_to_usd("environment.usda", preserve_source_contacts=True)

In this standalone example, no backend-buffer contact overrides have run, so
``preserve_source_contacts=True`` retains authored PhysX materials and automatic offsets.
Automatic pre-startup export uses the same setting. Leave it false when exporting actual
buffer contact overrides; ambiguous per-shape overrides fail explicitly. Newton uses its
native shape identities in either case. The exporter never executes event functions.

Preservation and authoring
--------------------------

``InteractiveScene.export_to_usd`` copies the stage and calls each registered asset's
``author_fixed_configuration(writer)``. In this example the robot supplies its
mass and joint properties; the box supplies its mass properties. Both write into the
same copy through a shared ``UsdWriter``. Articulation resolves and traverses its own joints;
the writer discovers data declarations and performs the common attribute writes. Collections use that same body writer for
all members. Scene-wide settings, dependencies and completeness are checked before saving.
Concrete assets map their native view identities into public data order. Physics managers own
global driver settings and backend collision tables, including static colliders without registered assets.
There is no scene export adapter or backend factory. ``UsdWriter.from_stage`` owns the isolated copy.

One environment is sufficient for export. ``env_id`` defaults to zero and selects a single
environment from an existing replicated scene; for example, ``env_id=37`` selects environment
37 when that environment exists. Differences in other environments do not change the selected
environment's effective configuration. The replicated-scene regression covers matching
object layouts with different runtime overrides; this does not certify heterogeneous scenes.
ClonePlan queries supply
its source variants and instance membership, not runtime property values. Physics-only
clones missing from USD are materialized from their authored sources; objects supply the
selected instance's effective buffers. Other environments are removed, shared scene content
and dependencies retained, and the selected environment keeps its world frame.

The in-memory USD stage carries geometry, mass/inertia/COM, topology, collision materials,
filtering, static objects, terrain, shared resources, tendon schemas and other authored settings.
Native Flatten/Stage.Export preserve that content. Effective body/joint state, mass, inertia, COM, actuator properties and supported native
contact/gravity values are written onto the isolated copy.
The live stage and caller's configuration are not authored by the exporter. Newton's
queued property notifications synchronize solver buffers at the snapshot boundary.

Data properties declare targets alongside their getters with ``@usd_field(UsdAttribute(...))``.
Source names come from the decorated properties, without a second exporter field list. For example::

    @property
    @usd_field(UsdAttribute("drive:{axis}:physics:stiffness", "PhysicsDriveAPI:{axis}", angular_power=-1))
    def joint_stiffness(self):
        ...

``usd_fields`` discovers declarations through the class MRO without evaluating getters.
Backend getter overrides inherit declarations; an explicit decorator replaces them, or appends
with ``extend=True``. Empty declarations require a backend definition. Joint friction is declared
by each concrete backend: OVPhysX's raw binding values are not converted into another backend's
effort units. Abstract-property and existing observation metadata are preserved.

Registered schemas supply exact attribute names and types through ``GetSchemaAttributeNames``
and ``Prim.GetAttribute``. Single/multi-apply and typed schemas are distinguished. Generic writing
supports scalar, vector and scalar-array attributes, and declared components such as lower/upper limits.
Matrices and relationship semantics need dedicated operations; body placement,
fixed-root frames and Newton material bindings remain explicit. Unregistered extensions require
an explicit target type. Schema discovery cannot infer source fields, units or backend semantics.

The common writer targets standard USD Physics plus the
PhysX extension dialect used by Isaac Sim. Concrete assets and physics managers supply required
native extensions; this does not make PhysX-specific friction or drive semantics backend-neutral.
Explicit controller gains never become implicit solver gains.

Missing required objects, unsupported driven joint types and
unresolved dependencies fail before replacing the destination. The implementation currently
requires SI stage units and a representable timestep. Newton cloth, soft bodies and cables,
and Isaac Sim surface/volume deformables retain their authored geometry and effective physical
properties. Surface grippers are rejected. Native schemas unsupported by a deployment consumer still
need separate validation. Flattening is not packaging: referenced textures, MDL modules and other
external assets must remain accessible.

Newton loading
--------------

Newton preserves the original geometry and adds its fixed contact/joint properties. Colliders get
independent physics-material bindings when native values are authored, preserving differences even
when the source material was shared. Missing bindings receive explicit native material values.

XPBD, MJWarp, Kamino and VBD managers own their driver exports. The proxy coupler delegates
to its MJWarp/VBD children and preserves solver ownership, proxy relationships and substeps.
MJWarp additionally reads native body, joint and contact buffers; Kamino stores its resolved
nested driver configuration. Unrepresentable per-joint actuation modes fail explicitly.
Load with the deployment entry points to consume the physical extensions and driver settings::

    from pxr import Usd
    from isaaclab_newton.physics.deployment import create_deployment_model, create_deployment_solver

    path = "deployment.usda"
    metadata = Usd.Stage.Open(path).GetRootLayer().customLayerData
    model, mappings = create_deployment_model(path, device="cuda:0")
    solver = create_deployment_solver(
        model, mappings["driver"], particle_paths=mappings["particle_paths"]
    )
    timing = metadata["isaaclab:newtonSimulation"]
    dt = timing["dt"]

The loader registers the selected solver schemas and restores physical configuration without
constructing a task or replaying its events. Cloth and volume meshes retain connectivity and
stress-free geometry separately from nodal placement. Particle masses use ``physics:masses``.
Named ``newton:export:*`` attributes supplement particle pinning, radii and flags, edge rest
angles, and cable contact properties that native import cannot reproduce. In particular,
initialization can overwrite rest angles without changing mesh geometry; export preserves
those effective values without changing the live simulation. These extensions require this
loader; a generic USD importer does not
provide equivalent physical semantics. Proxy coupling supports MJWarp/VBD children; ADMM and
custom collision-pipeline factories are rejected.

The deployment stepping loop must honor ``dt``, ``num_substeps`` and ``collision_decimation``
in ``isaaclab:newtonSimulation``.

The metadata consumers are the deployment loader (illustrated above) and the independent
fresh-load tests. Isaac Sim/OVPhysX do not consume these Newton driver options. The descriptive
``isaaclab:configuration`` marker does not execute task code.

The fresh-load tests compare complete fixture entity coverage, topology, geometry,
materials, collision relationships, body/joint properties and gravity. MJWarp and
Kamino also compare native solver configuration after loading without task overrides.
Initial velocities are checked as zero; post-warmup poses and velocities are not compared.

Runtime integration and cost
----------------------------

The artifact is a physical deployment configuration, not a policy or controller checkpoint.
Native ``NewtonActuator`` schemas are retained. Isaac Sim's experimental
``isaacsim.core.experimental.actuators.ArticulationActuators`` loader requires explicit attachment;
loading USD alone does not start controllers. This interface was checked at Isaac Sim develop
``b023e77b0534f11fc6e2f1a0e500d46524f5e10d``; loader execution is not certified here. Python-only
controllers without a native representation are rejected.

Deployment must restore policies, observations, sensor sampling/rendering, controller histories,
and task logic. Complete same-backend fresh-load tests are the first gate; use on a different
backend requires a separate physical-semantics validation.

``deployment.metrics.json`` reports selection/materialization, configuration authoring,
validation and saving durations, total export wall time, source environment count, selected
id and output size. It measures incremental export in the already initialized process.
Same-seed integration tests compare event results and subsequent training with the flag off/on.

PhysX tensor interfaces expose body identities but not per-collider identities. Before
buffer overrides, source bindings, geometry and automatic offset semantics are preserved,
including distinct source materials. Manual buffer-override export supports single colliders
and uniform per-body values; distinct per-shape overrides are rejected. Uniform values also
cover cooked meshes with multiple convex pieces. Newton retains explicit shape labels,
including static colliders. These support boundaries are separate from task/preset availability.

Additional support boundaries
-----------------------------

Newton, OVPhysX and Isaac Sim PhysX Cartesian multi-axis joints retain axis-specific limits
and drives. MJWarp force-based joint limits are exported as per-axis stiffness/damping;
the fresh solver derives its inertia-dependent ``solreflimit`` values. Explicit native
joint-wide limit settings are retained when representable. Non-Cartesian Newton axes,
unrepresentable joint-wide native overrides, and OVPhysX spherical-joint drives are rejected.

OVPhysX cannot export different contact/material values for individual cooked convex pieces
until its tensor API exposes stable piece-to-USD identity and cooked geometry. Body identity
alone is insufficient. Native warmup can generate motion even before a task takes its first
step; this transient state is excluded from deployment export. OVPhysX scene frequency is
authored from the simulation timestep so native contact-default calculation uses the configured rate.

Newton terrain heightfields retain the source mesh, conversion resolution and effective
contact properties. The deployment loader repeats the same mesh-to-heightfield conversion.
Other native colliders without an authored USD identity or a supported curve representation
are rejected.

The exporter does not reconstruct arbitrary edits to private solver topology or geometry
buffers, Kamino-only body/material buffer mutations, solver warm-start caches, active contacts,
or transient applied forces. Use supported object-data setters for physical randomization.
Authored tendon/actuator schemas are retained, but runtime tendon/control buffers require
separate validation. A successful physical export does not certify policy observation support:
cameras, ray/contact sensors, sampling schedules and controller histories need deployment
integration even when their authored scene prims are present.

Unreachable external resources and dangling dependencies fail completeness checks.
Missing direct material targets are cleared only when all resolved materials on the
affected subtree remain unchanged, including inherited bindings and material purposes.
Bindings whose removal would change material resolution still fail completeness checks.
A reachable authored URL is checked in its original form when USD dependency discovery
normalizes its scheme incorrectly. Missing source materials are not invented or replaced.
