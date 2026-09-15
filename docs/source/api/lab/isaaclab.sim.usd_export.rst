isaaclab.sim.usd_export
=======================

.. automodule:: isaaclab.sim.usd_export

Fixed deployment export
-----------------------

Use ``--export_deployment_usd`` with the unified RSL-RL, RL-Games, SKRL or SB3 training
entrypoint to write ``deployment.usda`` beside the run's configuration files in ``log_dir``.
The option is off by default. Only global rank zero exports; RLINF and the experimental
Warp task frontend are not supported by this integration.

The normal task constructor completes first, including subclass setup and one-time
``prestartup``/``startup`` events. The opt-in call then exports environment zero before
training wrappers perform their first reset or step. There is no second task construction,
pre-event callback or exception used to interrupt initialization. Sampled one-time results
are part of the artifact; later reset, interval and training changes are excluded.

The snapshot uses current backend state. It does not apply ``default_*`` buffers: defaults
are inputs to future resets and may differ from the current state or be changed by events.
Calling reset or applying defaults during export would erase startup results. Tasks must
register physical assets in the scene; unregistered enabled bodies fail completeness checks.
A task that steps inside its constructor cannot use this initialization-only export.

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
        scene.reset_to_default()
        sim.forward()
        scene.update(0.0)
        scene.export_to_usd("environment.usda")

In this standalone example, default state is applied explicitly before the snapshot because
there is no task event lifecycle. For task environments, export after the constructor returns;
do not apply defaults again after startup events. The exporter never executes event functions.

Preservation and authoring
--------------------------

``InteractiveScene.export_to_usd`` copies the stage and calls each registered asset's
``author_fixed_configuration(writer)``. In this example the robot supplies its
link state and joint properties; the box supplies its body state. Both write into the
same copy through a shared ``UsdWriter``. Articulation resolves and traverses its own joints;
the writer discovers data declarations and performs the common attribute writes. Collections use that same body writer for
all members. Scene-wide settings, dependencies and completeness are checked before saving.
Concrete assets map their native view identities into public data order. Physics managers own
global driver settings and backend collision tables, including static colliders without registered assets.
There is no scene export adapter or backend factory. ``UsdWriter.from_stage`` owns the isolated copy.

One environment is sufficient for export. ``env_id`` defaults to zero and selects a single
environment from an existing replicated scene; for example, ``env_id=37`` selects environment
37 when that environment exists. Differences in other environments do not change the selected
environment's effective configuration. ClonePlan queries supply
its source variants and instance membership, not runtime property values. Physics-only
clones missing from USD are materialized from their authored sources; objects supply the
selected instance's effective buffers. Other environments are removed, shared scene content
and dependencies retained, and the selected environment keeps its world frame.

The in-memory USD stage carries geometry, mass/inertia/COM, topology, collision materials,
filtering, static objects, terrain, shared resources, tendon schemas and other authored settings.
Native Flatten/Stage.Export preserve that content. Effective body/joint state, mass, inertia, COM, actuator properties and supported native
contact/gravity values are written onto the isolated copy.
The live stage, backend buffers and caller's configuration are not authored by the exporter.

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
supports scalar and vector attributes, and declared components such as lower/upper limits.
Arrays, matrices, transforms and relationship semantics need dedicated operations; body transforms,
fixed-root frames and Newton material bindings remain explicit. Unregistered extensions require
an explicit target type. Schema discovery cannot infer source fields, units or backend semantics.

The common writer targets standard USD Physics plus the
PhysX extension dialect used by Isaac Sim. Concrete assets and physics managers supply required
native extensions; this does not make PhysX-specific friction or drive semantics backend-neutral.
Explicit controller gains never become implicit solver gains.

Missing required objects, unsupported driven joint types and
unresolved dependencies fail before replacing the destination. The implementation currently
requires SI stage units, representable timestep and supported rigid assets. Deformables, cables
and surface grippers are rejected. Native schemas unsupported by a deployment consumer still
need separate validation. Flattening is not packaging: referenced textures, MDL modules and other
external assets must remain accessible.

Newton loading
--------------

Newton preserves the original geometry and adds its fixed contact/joint properties. Colliders get
independent physics-material bindings when native values are authored, preserving differences even
when the source material was shared. Missing bindings receive explicit native material values.

The supported Newton driver is XPBD. Unsupported solver families, non-default unrepresentable
options, substeps/decimation and unrepresentable per-joint actuation modes fail explicitly. The
loader must consume the exported import options and driver metadata, rather than silently using
its own defaults::

    from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

    stage = Usd.Stage.Open("deployment.usda")
    metadata = stage.GetRootLayer().customLayerData
    builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
                    **metadata["isaaclab:newtonImportOptions"])
    model = builder.finalize()
    driver = dict(metadata["isaaclab:newtonDriver"])
    assert driver.pop("solver") == "xpbd"
    solver = newton.solvers.SolverXPBD(model, **driver)
    dt = 1 / stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:timeStepsPerSecond").Get()

The metadata consumers are the deployment loader (illustrated above) and the independent
fresh-load tests. Isaac Sim/OVPhysX do not consume these Newton driver options. The descriptive
``isaaclab:configuration`` marker does not execute task code.

The fresh-load test constructs this driver and checks a non-default iteration count. The pinned
Newton importer uses radians/second for native initial angular velocity; its compatibility
attributes are emitted alongside standard USD degree-based state and currently produce warnings.
That behavior requires revalidation on Newton upgrades.

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

PhysX tensor interfaces expose body identities but not per-collider identities. Single
colliders and equal per-body contact values can be authored unambiguously; distinct per-shape
values are rejected. Uniform values also cover cooked meshes with multiple convex pieces. Newton retains explicit shape labels,
including static colliders. These support boundaries are separate from task/preset availability.
