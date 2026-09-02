.. _cloning-environments:

Cloning Environments
====================

.. currentmodule:: isaaclab

Parallel simulation at scale needs many environments stepping side by side —
hundreds, sometimes tens of thousands per GPU — and authoring each of those envs
by hand would be hopelessly slow. Cloning is Isaac Lab's answer: you author a
small representative scene under ``/World/envs/env_n`` and the cloner expands it
across the rest of the env population for you, optionally with per-env variation.

The expansion itself is performed by USD and the active physics backend's native
replicator, wrapped by Isaac Lab's core :mod:`isaaclab.cloner` module behind a
single uniform surface.

.. contents:: On this page
   :local:
   :depth: 2


The Backend Layer
-----------------

At the bottom of the stack, each backend exposes a raw function that takes a flat
description of the world layout. These functions are useful for standalone tools
and tests and deliberately have parallel signatures:

.. code-block:: text

    backend_replicate(stage, sources, destinations, env_ids, selection, positions=None, quaternions=None, ...)

The arguments are parallel arrays describing the layout:

* ``sources`` — source prim paths already authored on the stage.
* ``destinations`` — destination templates containing ``"{}"``, formatted with each env id.
* ``env_ids`` — NumPy integer array of target env indices.
* ``selection`` — NumPy boolean array of shape ``[len(sources), num_envs]``;
  ``selection[i, j]`` is ``True`` when env ``j`` should be populated from source ``i``.
  The raw USD function names this argument ``mask``; physics functions name it ``mapping``.
* ``positions`` / ``quaternions`` — optional per-env world transforms.

Production scene construction stores those arrays once in a
:class:`~isaaclab.cloner.ClonePlan`. Simulation-owned backend contexts consume the
same value through ``context.replicate(plan)``; no backend rebuilds the mapping
from a second queue of array arguments.


Standalone Examples
~~~~~~~~~~~~~~~~~~~

Direct calls into the backend functions, for tooling or tests that need full
control. Production code reaches for one of the ways in
`Cloning in a Backend-Agnostic Way`_ instead.

**USD** — clone a visual cube across envs:

.. code-block:: python

    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.cloner import usd_replicate

    num_envs = 128
    stage = sim_utils.get_current_stage()
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/envs/env_0/Cube", cube_cfg)

    usd_replicate(
        stage,
        sources=("/World/envs/env_0/Cube",),
        destinations=("/World/envs/env_{}/Cube",),
        env_ids=np.arange(num_envs),
        mask=np.ones((1, num_envs), dtype=np.bool_),
    )

**PhysX** — call PhysX and USD on the same sources and destinations (either order):

.. code-block:: python

    from isaaclab_physx.cloner import physx_replicate

    sources = ("/World/envs/env_0/Cube",)
    destinations = ("/World/envs/env_{}/Cube",)
    env_ids = np.arange(num_envs)
    mapping = np.ones((1, num_envs), dtype=np.bool_)
    physx_replicate(stage, sources, destinations, env_ids, mapping=mapping)
    usd_replicate(stage, sources, destinations, env_ids, mask=mapping)

**Newton**:

.. code-block:: python

    from isaaclab_newton.cloner import newton_physics_replicate

    newton_physics_replicate(stage, sources, destinations, env_ids, mapping=mapping)

**OvPhysX**:

.. code-block:: python

    from isaaclab_ov.cloner import ovphysx_replicate

    ovphysx_replicate(stage, sources, destinations, env_ids, mapping=mapping)


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

A plan holds the parallel arrays used by production clone contexts — sources,
destinations, mask, env ids — in one place. Conceptually it is a small table
where each row describes one distinct prototype-to-destination mapping; the
fields listed below are that table's columns:

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
     - NumPy boolean array ``[len(sources), num_envs]``; ``True`` when env ``j`` comes from row ``i``.
   * - ``env_ids``
     - Optional NumPy integer array of target env ids; execution requires it.
   * - ``positions``
     - Optional per-env world positions [m], shape ``[num_envs, 3]``.
   * - ``global_paths``
     - Unique prim paths for scene assets shared by every env and therefore not replicated.
   * - ``context_rows``
     - Clone-context types mapped to the rows they consume.

The plan does not own a stage. Simulation-owned contexts supply their own runtime
when they consume it.

When every env is a copy of env_0:

.. code-block:: text

    sources      = ("/World/envs/env_0",)
    destinations = ("/World/envs/env_{}",)
    clone_mask   = [[True, True, ..., True]]
    global_paths = ("/World/Ground", "/World/Light")

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
  between the build and the drain where you can inspect or mutate the plan.
  Reach for it when you are assembling a scene outside
  :class:`~isaaclab.scene.InteractiveScene` or want fine control over timing.
* The third is a one-shot shortcut for the case where every env is just a copy
  of env_0. Reach for it in :class:`~isaaclab.envs.DirectRLEnv` and standalone
  scripts that hand-build the env-0 prototype prim by prim.

``ReplicateSession``
~~~~~~~~~~~~~~~~~~~~

:class:`~isaaclab.cloner.ReplicateSession` is a context manager that brackets the
whole cloning lifecycle. Entering the block builds and publishes the plan, the body
constructs assets at their planned source paths, and exiting dispatches that same plan:

.. code-block:: python

    with cloner.ReplicateSession(cfgs, num_clones=N, env_spacing=2.0):
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

The same lifecycle as the session, written as separate function calls. Publish the
plan before construction so every participant observes the same layout, then
dispatch it after the prototypes exist:

.. code-block:: python

    plan = cloner.make_clone_plan(cfgs, num_clones=N, env_spacing=2.0)
    sim.set_clone_plan(plan)
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
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing)[0]
        global_paths = ("/World/ground",)
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, pos, global_paths=global_paths)
        cloner.replicate(plan)

Every env receives the same prototype. When envs need to differ, use one of the
other two. Hand-built scenes must pass every shared asset root in ``global_paths``;
use ``()`` when there are none.


Under the Hood
--------------

Planning maps each cfg to rows in ``cfg_rows`` and each participating backend to
its subset in ``context_rows``. The active physics manager registers its clone
context during simulation initialization. Assets use that context by default;
:attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` can select an explicitly
registered context instead. Planning also registers
:class:`~isaaclab.cloner.UsdReplicateContext` for spawned assets when Kit is
available.

The backend packages expose different context implementations behind one
execution contract:

.. code-block:: text

    UsdReplicateContext      # replicates USD prim subtrees
    PhysxReplicateContext    # replicates PhysX rigid bodies and articulations
    NewtonReplicateContext   # replicates Newton bodies in its parallel pipeline

:func:`~isaaclab.cloner.replicate` resolves these types through the
:class:`~isaaclab.sim.SimulationContext` backend registry, orders them by
``replicate_priority``, and passes the published plan to each one:

.. code-block:: python

    simulation.set_clone_plan(plan)
    construct_prototypes()
    for context_type in plan.context_rows:
        simulation_backends[context_type].replicate(plan)

The cfg-first lifecycle publishes before ``construct_prototypes()``. The direct
single-source workflow remains post-construction and is published by
:func:`~isaaclab.cloner.replicate` immediately before dispatch. In either form,
the simulation accepts one plan and each backend receives that exact object once.

USD runs before native physics contexts so the destination topology exists when
they consume it. No fallback context is constructed during dispatch.

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
