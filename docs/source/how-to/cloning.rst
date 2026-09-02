.. _cloning-environments:

Cloning Environments
====================

.. currentmodule:: isaaclab

Parallel simulation at scale needs many environments stepping side by side —
hundreds, sometimes tens of thousands per GPU — and authoring each of those envs
by hand would be hopelessly slow. Cloning is Isaac Lab's answer: you author a
small representative scene under ``/World/envs/env_n`` and the cloner expands it
across the rest of the env population for you, optionally with per-env variation.

Each backend receives the same :class:`~isaaclab.cloner.ClonePlan`. Its replicate
context reads the rows routed to it and applies them to USD, PhysX, Newton, or
another native runtime. No backend rebuilds the layout from parallel arguments.

.. contents:: On this page
   :local:
   :depth: 2


The Backend Layer
-----------------

At the bottom of the stack, each backend exposes a replicate context with one
execution method:

.. code-block:: text

    context.replicate(plan)

The context constructor receives its owning runtime, such as a USD stage or
:class:`~isaaclab.sim.SimulationContext`; the plan is its only execution input.


Standalone Examples
~~~~~~~~~~~~~~~~~~~

Raw :func:`~isaaclab.cloner.usd_replicate` remains available for low-level tooling and
the :class:`~isaaclab.scene.InteractiveScene` environment-root bootstrap. Direct PhysX,
Newton, and OvPhysX control uses their public replicate contexts; the examples below
illustrate the common contract. Normal scene construction reaches for one of the ways in
`Cloning in a Backend-Agnostic Way`_ instead.

Start with one explicit plan. This one routes a visual cube to USD:

.. code-block:: python

    from dataclasses import replace

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab import cloner

    num_envs = 128
    device = "cuda:0"
    stage = sim_utils.get_current_stage()
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/envs/env_0/Cube", cube_cfg)

    plan = cloner.ClonePlan(
        sources=("/World/envs/env_0/Cube",),
        destinations=("/World/envs/env_{}/Cube",),
        clone_mask=torch.ones((1, num_envs), dtype=torch.bool, device=device),
        env_ids=torch.arange(num_envs, device=device),
        positions=cloner.grid_transforms(num_envs, device=device)[0],
        context_rows={cloner.UsdReplicateContext: (0,)},
    )
    cloner.UsdReplicateContext(stage).replicate(plan)

For PhysX, route the same row to both contexts. USD must author the destination
topology before native PhysX consumes it:

.. code-block:: python

    from isaaclab_physx.cloner import PhysxReplicateContext

    plan = replace(
        plan,
        context_rows={cloner.UsdReplicateContext: (0,), PhysxReplicateContext: (0,)},
    )
    cloner.UsdReplicateContext(stage).replicate(plan)
    physx_context = PhysxReplicateContext(stage)
    physx_context.replicate(plan)

The PhysX context owns a native registration. Retain it while the simulation
uses that registration, then clear it during application teardown:

.. code-block:: python

    physx_context.clear()

Newton receives the owning simulation context rather than a USD stage:

.. code-block:: python

    from isaaclab_newton.cloner import NewtonReplicateContext

    sim = sim_utils.SimulationContext.instance()
    assert sim is not None
    plan = replace(plan, context_rows={NewtonReplicateContext: (0,)})
    NewtonReplicateContext(sim).replicate(plan)


Cloning in a Backend-Agnostic Way
---------------------------------

Authoring every prim in every env by hand would be prohibitively slow and would
also tie scene code to whichever physics engine happens to be active. Isaac Lab
sidesteps both problems with a single central abstraction:
:class:`~isaaclab.cloner.ClonePlan` — a compact description of how a small set of
prim-level prototypes maps onto the full population of envs, with each prototype
free to land in some envs and not others. A plan is built once, fed to each backend, and
lets every engine take its own fastest replication path: USD instancing for
visuals, PhysX's native replicator for rigid bodies and articulations, Newton's
world system for its parallel pipeline. The same plan drives all of them, so user
code never branches on the backend.

ClonePlan
~~~~~~~~~

A plan holds the parallel arrays every backend consumes in one place. Conceptually
it is a small table
where each row describes one distinct prototype-to-destination mapping; the
fields listed below are that table's columns. Backend-neutral planning and
replication entry points pass this value between them:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``sources``
     - Exact source prim paths, one per replication row.
   * - ``destinations``
     - Destination templates with ``"{}"`` for the env id, or an exact path for a shared asset.
   * - ``clone_mask``
     - Bool tensor ``[len(sources), num_envs]``; ``True`` when env ``j`` comes from row ``i``.
   * - ``env_ids``
     - Required long tensor of target env ids; mask column ``j`` represents ``env_ids[j]``.
   * - ``positions``
     - Required per-env world positions [m], shape ``[num_envs, 3]``.
   * - ``cfg_rows``
     - Configuration identities mapped to the rows they own.
   * - ``context_rows``
     - Replicate-context types mapped to the rows they consume.
   * - ``env_template``
     - Environment path template used to resolve environment roots.

``clone_mask``, ``env_ids``, and ``positions`` live on one device. Environment ids are
unique, nonnegative values; mask column ``j`` and position row ``j`` both describe
``env_ids[j]``.

When every env is a copy of env_0 and the scene has one shared ground plane:

.. code-block:: text

    sources      = ("/World/envs/env_0", "/World/Ground")
    destinations = ("/World/envs/env_{}", "/World/Ground")
    clone_mask   = [[1, 1, ..., 1],
                    [0, 0, ..., 0]]

The matching exact source and destination plus the all-zero mask identify the
ground as an authored-once shared row. It stays in the same plan instead of
traveling through a separate ``global_paths`` field.

Homogeneous construction retains the environment-root row so manually authored
siblings are covered alongside cfg-owned assets. Heterogeneous construction uses
asset-wise rows because different environments may select different prototypes.

When envs differ — say a cartpole in every env plus a 2-variant obstacle (box into
envs 0/1, sphere into envs 2/3):

.. code-block:: text

    sources      = ("/World/envs/env_0/Cartpole",
                    "/World/envs/env_0/Obstacle_0",     # box prototype
                    "/World/envs/env_0/Obstacle_1")     # sphere prototype
    destinations = ("/World/envs/env_{}/Cartpole",
                    "/World/envs/env_{}/Obstacle",
                    "/World/envs/env_{}/Obstacle")
    clone_mask   = [[1, 1, 1, 1],
                    [1, 1, 0, 0],
                    [0, 0, 1, 1]]

Querying a plan
~~~~~~~~~~~~~~~

Anything that has to follow an asset between the two sides of that table — a sensor
resolving its ``prim_path`` back to the prototype it should read, a ray caster
loading one mesh per variant — asks :mod:`isaaclab.cloner.query` rather than
manipulating path strings itself:

.. code-block:: python

    from isaaclab import cloner

    # where does this prototype land in env 2?
    cloner.query.path_to_clone(plan, "/World/envs/env_0/Obstacle_1", env_id=2)
    # -> "/World/envs/env_2/Obstacle"

    # which envs does this prototype reach at all?
    cloner.query.path_env_ids(plan, "/World/envs/env_0/Obstacle_1")
    # -> (2, 3)

    # which prototype is env 2's obstacle cloned from?
    cloner.query.path_to_source(plan, "/World/envs/env_2/Obstacle")
    # -> ("/World/envs/env_0/Obstacle_1", "/World/envs/env_*/Obstacle", "")

Two obstacle variants share one destination template, so the template alone does not
identify a prototype — the environment does. A concrete path carries it in the clone
slot; a ``env_.*`` wildcard does not, and resolves to one representative variant
unless you pass ``env_id``. Use :func:`~isaaclab.cloner.query.iter_sources` when you
need every variant behind a template. Note that environment ids are not mask columns:
column ``j`` stands for ``env_ids[j]``, and the queries speak ids throughout.

A plan is the *what*. Putting one together and handing it to the backends is
the *how*, and Isaac Lab exposes three idiomatic ways to do that. All three end
in the same ``cloner.replicate(plan)`` call, so the choice between
them is purely about ergonomics:

* The first wraps both phases in a context manager and is what
  :class:`~isaaclab.scene.InteractiveScene` runs under the hood. Reach for it
  when you want the lifecycle hidden and you are authoring assets through a
  scene config.
* The second spells the same flow out as plain function calls, leaving a moment
  between the build and the drain where you can inspect the plan.
  Reach for it when you are assembling a scene outside
  :class:`~isaaclab.scene.InteractiveScene` or want fine control over timing.
* The third is a one-shot shortcut for the case where every env is just a copy
  of env_0. Reach for it in :class:`~isaaclab.envs.DirectRLEnv` and standalone
  scripts that hand-build the env-0 prototype prim by prim.

``ReplicateSession``
~~~~~~~~~~~~~~~~~~~~

:class:`~isaaclab.cloner.ReplicateSession` is a context manager that brackets the
whole cloning lifecycle. Entering the block builds the fully routed plan, the body
constructs the prototype assets, and exiting the block clears constructor registrations
and dispatches that same plan. Those registrations are consumed only by the separate,
post-construction :func:`~isaaclab.cloner.clone_plan_from_env_0` workflow:

.. code-block:: python

    with cloner.ReplicateSession(cfgs, num_clones=N, env_spacing=2.0, device=device):
        for cfg in cfgs:
            cfg.class_type(cfg)

This is what :class:`~isaaclab.scene.InteractiveScene` runs when you declare assets
in an :class:`~isaaclab.scene.InteractiveSceneCfg`:

.. code-block:: python

    @configclass
    class MySceneCfg(InteractiveSceneCfg):
        robot = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0),
        )

    scene = InteractiveScene(MySceneCfg(num_envs=128, env_spacing=2.0))

When envs need to differ across the population, use
:class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg` or
:class:`~isaaclab.sim.spawners.wrappers.MultiUsdFileCfg`; see
:doc:`multi_asset_spawning`.

``make_clone_plan`` + ``replicate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same two phases as the session, written as separate function calls. The plan
is built first, asset construction happens in between, and the drain runs
explicitly at the end. The gap lets tooling inspect or log the plan before
replication:

.. code-block:: python

    plan = cloner.make_clone_plan(cfgs, num_clones=N, env_spacing=2.0, device=device)
    for cfg in cfgs:
        cfg.class_type(cfg)
    cloner.replicate(plan)

``clone_plan_from_env_0`` + ``replicate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shortcut for the case where every env is just a copy of env_0.
:func:`~isaaclab.cloner.clone_plan_from_env_0` builds the single-source plan in
one line by pointing at the prototype, and :func:`~isaaclab.cloner.replicate`
finishes the setup. This is the pattern most :class:`~isaaclab.envs.DirectRLEnv`
subclasses use — they author the env-0 prototype prim by prim in
``_setup_scene`` and end the method with this sequence:

.. code-block:: python

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # ... any other assets ...

        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        global_paths = ("/World/ground",)
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, self.device, pos, global_paths=global_paths)
        cloner.replicate(plan)

Every env receives the same prototype. When envs need to differ, use one of the
other two. Hand-built scenes pass every shared asset root through the
``global_paths`` builder argument; the builder records each root as an exact,
all-zero-mask plan row. Use ``()`` when there are none.


Under the Hood
--------------

To see how the backend-agnostic surface works, follow one asset through a
:class:`~isaaclab.cloner.ReplicateSession`. Entering the session builds the plan
from its cfgs. The plan records both the rows each cfg owns in ``cfg_rows`` and
the rows each backend consumes in ``context_rows``. Asset constructors inside
the block author the prototypes; they do not clone them inline. Exiting the
session sends that same plan to every routed backend context.

The story has to look like this because the engines underneath disagree about
*when* and *how* replication actually happens:

* **PhysX** registers its plan projection with the native physics replicator.
* **USD** is declarative and immediate — its context materializes the clones in
  place.
* **Newton** is also declarative and immediate, but it insists on replicating
  the whole world in one shot rather than asset by asset, so the framework
  assembles every routed row first.

Every backend supplies a :class:`~isaaclab.cloner.UsdReplicateContext`,
``PhysxReplicateContext``, ``NewtonReplicateContext``, or equivalent with the
same ``replicate(plan)`` execution interface.

Planning and registration
~~~~~~~~~~~~~~~~~~~~~~~~~

For :func:`~isaaclab.cloner.make_clone_plan` and
:class:`~isaaclab.cloner.ReplicateSession`, cfg-to-context routing is compiled
before asset construction. The post-construction
:func:`~isaaclab.cloner.clone_plan_from_env_0` shortcut instead reads
:data:`~isaaclab.cloner.REPLICATION_QUEUE`, which contains cfgs registered by
their constructors.

The physics context comes from the cfg's
:attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` when set, otherwise the
context registered for the active simulation's physics clone role (PhysX and
Newton replicate natively; OvPhysX replays its own clones). Planning adds
:class:`~isaaclab.cloner.UsdReplicateContext` automatically whenever a cfg has a
spawner and Kit is available, so USD clones accompany physics replication under
Kit and are skipped by default in headless runs. An explicit cfg override may
still request USD replication without Kit. With
:attr:`~isaaclab.cloner.CloneCfg.replicate_physics` disabled, cloning is
USD-only: every physics context is dropped and the physics engine parses the
per-env USD prims directly.

This batches every asset's request into one call per context while keeping asset
code free of backend branches.

:attr:`~isaaclab.scene.InteractiveSceneCfg.replicate_physics` is piped into
:attr:`~isaaclab.cloner.CloneCfg.replicate_physics` and applied at dispatch;
an asset whose only cloning mechanism is physics replication is then simply
not cloned.

Backend contexts
~~~~~~~~~~~~~~~~

Each backend ships a small adapter class — its *replicate context* — owned by
the active :class:`~isaaclab.sim.SimulationContext` backend registry:

.. code-block:: text

    UsdReplicateContext      # replicates USD prim subtrees
    PhysxReplicateContext    # replicates PhysX rigid bodies and articulations
    NewtonReplicateContext   # replicates Newton bodies in its parallel pipeline

A row may resolve to more than one context. PhysX pairs its context with USD so
physics and visuals both follow; Newton's default stack includes USD only under
Kit, so kitless runs skip the authoring cost. Each context reads its own routed
rows from the same plan.

Running replication
~~~~~~~~~~~~~~~~~~~

:func:`~isaaclab.cloner.replicate` is what actually runs the registered work.
The dispatch shape is roughly:

.. code-block:: python

    def replicate(plan):
        clear_registration_queue()
        contexts = [simulation_backend_registry[context_type] for context_type in plan.context_rows]
        for context in sorted(contexts, key=lambda item: item.replicate_priority):
            context.replicate(plan)
        publish(plan)

USD has priority ``-100`` and therefore authors destinations before native
physics contexts at priority ``0`` consume them. After successful dispatch, the
plan is published to :class:`~isaaclab.sim.SimulationContext` so the rest of the
framework can read the per-env layout back.

Collision Filtering
-------------------

PhysX models per-env isolation through collision groups, so PhysX scenes need a
filtering pass after cloning to keep envs from colliding with each other while
still letting them collide with global prims (terrain, ground planes, lights).

:class:`~isaaclab.scene.InteractiveScene` runs that pass automatically when
``filter_collisions=True`` and the backend is PhysX. For direct PhysX pipelines,
call :func:`~isaaclab.cloner.filter_collisions` after the replicate:

.. code-block:: python

    from isaaclab.cloner import filter_collisions

    filter_collisions(
        stage=stage,
        physicsscene_path="/physicsScene",
        collision_root_path="/World/collisions",
        prim_paths=[f"/World/envs/env_{i}" for i in range(num_envs)],
        global_paths=["/World/ground"],
    )

Newton isolates envs through its world system and does not need this pass.
