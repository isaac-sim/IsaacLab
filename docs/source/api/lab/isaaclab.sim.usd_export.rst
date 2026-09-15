isaaclab.sim.usd_export
=======================

.. automodule:: isaaclab.sim.usd_export

Fixed deployment export
-----------------------

Use ``--export_deployment_usd`` with the unified RSL-RL, RL-Games, SKRL or SB3 training
entrypoint to write ``deployment.usda`` beside the run's configuration files in ``log_dir``.
The option is off by default. Only global rank zero exports; RLINF and the experimental
Warp task frontend are not supported by this integration.

A separate process constructs the actual task with one environment using a copy of the resolved
configuration. It includes Direct tasks' ``_setup_scene`` content, then stops before creating
``EventManager`` or executing ``prestartup``/``startup`` events. Physics initialization and
warmup complete, configured default body/joint states are applied, and the exporter takes its
snapshot. The parent subsequently constructs the normal training environment, with its original
environment count, configuration, seed and event behavior. Custom task constructors must use the
standard scene boundary; physical assets created only after that boundary are outside this API.
Direct tasks must register their physical assets in the scene. Unregistered rigid bodies fail
coverage checks. Unimportable task classes/configurations cannot be sent to the isolated worker.

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

There is no runtime snapshot
mode, model-only reconstruction, articulation-only export or replicated-environment extraction.
A task that requires randomized parameters for deployment must first express them as a fixed
configuration. Event functions are not evaluated by the export process.

Preservation and authoring
--------------------------

``InteractiveScene.export_to_usd`` copies the stage and calls each registered asset's
``author_fixed_configuration(writer)``. In this example the robot supplies its
link state and joint properties; the box supplies its body state. Both write into the
same copy through a shared ``UsdWriter``. Articulation resolves and traverses its own joints;
the writer discovers data declarations and performs the common attribute writes. Collections use that same body writer for
all members. Scene-wide settings, dependencies and completeness are checked before saving.
Backend managers provide the adapter; the scene does not select a backend by package name.

Task construction and the pre-event callback belong to the training worker. Because a task
constructor has no normal return at this boundary, a worker-local exception stops its remaining
controller/observation setup. This hook is a construction constraint, not part of USD authoring.

The in-memory USD stage carries geometry, mass/inertia/COM, topology, collision materials,
filtering, static objects, terrain, shared resources, tendon schemas and other authored settings.
Native Flatten/Stage.Export preserve that content. Only initialized values absent from USD are
written onto the isolated copy: body/joint initial state and resolved actuator solver properties.
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
PhysX extension dialect used by Isaac Sim. Backend adapters supply source identities and required
native extensions; this does not make PhysX-specific friction or drive semantics backend-neutral.
Explicit controller gains never become implicit solver gains.

Missing required objects, unsupported driven joint types, unknown configuration fields and
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

``deployment.metrics.json`` reports scene construction, initialization, flattening,
fixed configuration, dependency/coverage validation and file saving separately, in seconds.
The configuration phase includes object/backend data reading and authoring. It also
reports the child process's peak resident memory, output size and total process wall time including
startup/shutdown. Peak resident memory includes imported libraries and backend initialization;
it is not incremental GPU memory. The extra process and scene have a measurable startup cost,
so the flag remains opt-in. Same-seed integration tests compare training with the flag off/on.
