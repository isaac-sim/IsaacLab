isaaclab.sim.usd_export
=======================

.. automodule:: isaaclab.sim.usd_export

.. autoclass:: isaaclab.sim.usd_export.SceneExporter
   :members:

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

For a standalone fixed scene, launch the backend normally and call::

    from isaaclab.sim import SceneExporter

    SceneExporter.export_from_cfg(scene_cfg, sim_cfg, "environment.usda")

This requires ``num_envs=1`` and no active ``SimulationContext``. There is no runtime snapshot
mode, model-only reconstruction, articulation-only export or replicated-environment extraction.
A task that requires randomized parameters for deployment must first express them as a fixed
configuration. Event functions are not evaluated by the export process.

Preservation and authoring
--------------------------

The in-memory USD stage carries geometry, mass/inertia/COM, topology, collision materials,
filtering, static objects, terrain, shared resources, tendon schemas and other authored settings.
Native Flatten/Stage.Export preserve that content. Only initialized values absent from USD are
written onto the isolated copy: body/joint initial state and resolved actuator solver properties.
The live stage, backend buffers and caller's configuration are not authored by the exporter.

``isaaclab.assets.physics_properties`` declares public data sources, target attributes/schemas,
DOF identity and angular conversions. The common writer targets standard USD Physics plus the
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

``deployment.metrics.json`` reports scene construction, initialization, data reading, flattening,
fixed authoring, dependency/coverage validation and file saving separately, in seconds. It also
reports the child process's peak resident memory, output size and total process wall time including
startup/shutdown. Peak resident memory includes imported libraries and backend initialization;
it is not incremental GPU memory. The extra process and scene have a measurable startup cost,
so the flag remains opt-in. Same-seed integration tests compare training with the flag off/on.
