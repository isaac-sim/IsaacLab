.. _cloning-environments:

Cloning Environments
====================

.. currentmodule:: isaaclab

Parallel simulation at scale needs many environments stepping side by side —
hundreds, sometimes tens of thousands per GPU — and authoring each of those envs
by hand would be hopelessly slow. Cloning is Isaac Lab's answer: you author a
small representative scene under ``/World/envs/env_n`` and the cloner expands it
across the rest of the env population for you, optionally with per-env variation.

The expansion itself is performed by each physics backend's native replicator —
USD, PhysX, or Newton — wrapped by Isaac Lab's core :mod:`isaaclab.cloner` module
behind a single uniform surface so the same user code works regardless of which
backend is active.

.. contents:: On this page
   :local:
   :depth: 2


The Backend Layer
-----------------

At the bottom of the stack, each backend exposes a single function that takes a
flat description of the world layout and materializes it on its runtime. The
signatures are deliberately parallel so the layers above can target every backend
through one interface:

.. code-block:: text

    backend_replicate(stage, sources, destinations, env_ids, mask, positions=None, quaternions=None, ...)

The arguments are parallel arrays describing the layout:

* ``sources`` — source prim paths already authored on the stage.
* ``destinations`` — destination templates containing ``"{}"``, formatted with each env id.
* ``env_ids`` — long tensor of target env indices.
* ``mask`` — bool tensor of shape ``[len(sources), num_envs]``; ``mask[i, j]`` is
  ``True`` when env ``j`` should be populated from source ``i``.
* ``positions`` / ``quaternions`` — optional per-env world transforms.


Standalone Examples
~~~~~~~~~~~~~~~~~~~

Direct calls into the backend functions, for tooling or tests that need full
control. Production code reaches for one of the ways in
`Cloning in a Backend-Agnostic Way`_ instead.

**USD** — clone a visual cube across envs:

.. code-block:: python

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.cloner import usd_replicate

    num_envs = 128
    stage = sim_utils.get_current_stage()
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/envs/env_0/Cube", cube_cfg)

    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Cube"],
        destinations=["/World/envs/env_{}/Cube"],
        env_ids=torch.arange(num_envs, device="cuda:0"),
        mask=torch.ones((1, num_envs), dtype=torch.bool, device="cuda:0"),
    )

**PhysX** — call PhysX and USD on the same sources and destinations (either order):

.. code-block:: python

    from isaaclab_physx.cloner import physx_replicate

    physx_replicate(stage, sources, destinations, env_ids, mask)
    usd_replicate(stage, sources, destinations, env_ids, mask)

**Newton**:

.. code-block:: python

    from isaaclab_newton.cloner import newton_physics_replicate

    newton_physics_replicate(stage, sources, destinations, env_ids, mapping=mask)


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

A plan holds the parallel arrays a backend replicate consumes — sources,
destinations, mask, env ids — in one place. Conceptually it is a small table
where each row describes one distinct prototype-to-destination mapping; the
fields listed below are that table's columns. Every entry point in
:mod:`isaaclab.cloner` either produces a plan, consumes a plan, or both, so a
quick look at the fields is the fastest way to build intuition for the rest of
this page:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``sources``
     - Source prim paths, one per replication row.
   * - ``destinations``
     - Destination templates with ``"{}"`` for the env id, one per row.
   * - ``clone_mask``
     - Bool tensor ``[len(sources), num_envs]``; ``True`` when env ``j`` comes from row ``i``.
   * - ``env_ids``
     - Long tensor of target env ids.
   * - ``positions``
     - Optional per-env world positions [m], shape ``[num_envs, 3]``.

The plan is stage-agnostic by design — the same instance can be replayed against a
different stage, inspected by tooling, or serialized.

When every env is a copy of env_0:

.. code-block:: text

    sources      = ("/World/envs/env_0",)
    destinations = ("/World/envs/env_{}",)
    clone_mask   = [[True, True, ..., True]]

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

A plan is the *what*. Putting one together and handing it to the backends is
the *how*, and Isaac Lab exposes three idiomatic ways to do that. All three end
in the same ``cloner.replicate(plan, stage=...)`` call, so the choice between
them is purely about ergonomics:

* The first wraps both phases in a context manager and is what
  :class:`~isaaclab.scene.InteractiveScene` runs under the hood. Reach for it
  when you want the lifecycle hidden and you are authoring assets through a
  scene config.
* The second spells the same flow out as plain function calls, leaving a moment
  between the build and the drain where you can inspect or mutate the plan.
  Reach for it when you are assembling a scene outside
  :class:`~isaaclab.scene.InteractiveScene` or want fine control over timing.
* The third is a one-shot shortcut for the case where every env is just a copy
  of env_0. Reach for it in :class:`~isaaclab.envs.DirectRLEnv` and standalone
  scripts that hand-build the env-0 prototype prim by prim.

``ReplicateSession``
~~~~~~~~~~~~~~~~~~~~

:class:`~isaaclab.cloner.ReplicateSession` is a context manager that brackets the
whole cloning lifecycle. Entering the block builds the plan, the body is where
you construct your assets (each one registers itself as part of its constructor),
and exiting the block drains every registration against the plan:

.. code-block:: python

    with cloner.ReplicateSession(cfgs, num_clones=N, env_spacing=2.0,
                                 device=device, stage=stage):
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
explicitly at the end. The gap between the build and the drain is the point —
that is where you can read the plan back, mutate it, log it, or otherwise
intervene before replication actually happens:

.. code-block:: python

    plan = cloner.make_clone_plan(cfgs, num_clones=N, env_spacing=2.0, device=device)
    for cfg in cfgs:
        cfg.class_type(cfg)
    cloner.replicate(plan, stage=stage)

``ClonePlan.from_env_0`` + ``replicate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shortcut for the case where every env is just a copy of env_0.
:meth:`~isaaclab.cloner.ClonePlan.from_env_0` builds the single-source plan in
one line by pointing at the prototype, and :func:`~isaaclab.cloner.replicate`
finishes the setup. This is the pattern most :class:`~isaaclab.envs.DirectRLEnv`
subclasses use — they author the env-0 prototype prim by prim in
``_setup_scene`` and end the method with these four lines:

.. code-block:: python

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # ... any other assets ...

        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.ClonePlan.from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)

Every env receives the same prototype. When envs need to differ, use one of the
other two.


Under the Hood
--------------

To see how the backend-agnostic surface works, follow one asset through the
system. Suppose you write ``Articulation(cfg)`` for a PhysX articulation
somewhere inside a :class:`~isaaclab.cloner.ReplicateSession`. The constructor
does not actually clone anything yet — at that moment the plan describing how
the full env population should be laid out may not even exist. Instead the
constructor *registers* the asset with the cloner, the cloner files the
registration into a queue, and later — when the session exits and the cloner
runs replication — that registration is handed to the backend code that knows
how to replicate a PhysX articulation, with the plan telling it where each
clone goes.

The story has to look like this because the engines underneath disagree about
*when* and *how* replication actually happens:

* **PhysX** defers the real work to physics runtime. At construction time the
  only thing user code can do is register intent; PhysX replays those
  registrations entity by entity when the simulation comes up.
* **USD** is declarative and immediate — calling :func:`~isaaclab.cloner.usd_replicate`
  materializes the clones in place, right then and there.
* **Newton** is also declarative and immediate, but it insists on replicating
  the whole world in one shot rather than asset by asset, so the framework
  cannot just hand it one cfg at a time — everything Newton-related has to be
  assembled first.

Isaac Lab reconciles these into one surface with two small pieces of plumbing.
Every backend supplies its own :class:`~isaaclab.cloner.UsdReplicateContext` /
``PhysxReplicateContext`` / ``NewtonReplicateContext``, a class that hides the
timing and granularity differences above behind a single uniform interface. A
shared :data:`~isaaclab.cloner.REPLICATION_QUEUE` then remembers which asset
belongs to which backend's context until it is time to run. The three
subsections below explain the queue, the contexts, and the function that joins
them against a plan.

The registration queue
~~~~~~~~~~~~~~~~~~~~~~

Asset constructors do not replicate inline. They register their intent with
:data:`~isaaclab.cloner.REPLICATION_QUEUE` and the framework defers the actual
work to the drain. The queue ends up holding one entry per ``(asset, backend)``
pair:

.. code-block:: text

    REPLICATION_QUEUE
        (cartpole_cfg, PhysxReplicateContext)
        (cartpole_cfg, UsdReplicateContext)
        (cube_cfg,     UsdReplicateContext)
        (light_cfg,    UsdReplicateContext)
        ...

Deferring the work like this buys three things at once:

* Replication can wait until the plan is fully built, so the final layout is
  known before any prims are spawned.
* Every asset's request is batched into a single backend call instead of one
  call per asset.
* Asset code stays free of any branching on which backend is active — it just
  registers and lets the framework take it from there.

Backend contexts
~~~~~~~~~~~~~~~~

Each backend ships a small adapter class — its *replicate context* — that
knows how to take a registered cfg and replicate it on the backend's specific
runtime:

.. code-block:: text

    UsdReplicateContext      # replicates USD prim subtrees
    PhysxReplicateContext    # replicates PhysX rigid bodies and articulations
    NewtonReplicateContext   # replicates Newton bodies in its parallel pipeline

A single asset can register more than one context — a PhysX articulation
registers a PhysX context and a USD context so physics and visuals both follow,
a Newton articulation registers a Newton context plus a USD context only if it
owns visual prims. This is where backend differences are absorbed: swapping a
scene from PhysX to Newton swaps which context an asset registers with, while
the cfgs and the rest of the user code stay unchanged.

Running replication
~~~~~~~~~~~~~~~~~~~

:func:`~isaaclab.cloner.replicate` is what actually runs the registered work.
The dispatch shape is roughly:

.. code-block:: python

    def replicate(plan, stage):
        for context_cls, rows in group_queue_by_context(plan):
            context_cls().replicate(rows=rows, stage=stage)
        publish(plan)

Contexts run in a priority order that puts physics ahead of visuals, and the
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
