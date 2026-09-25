:orphan:

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

Production scene construction retains the asset declarations and selected variants in a
:class:`~isaaclab.cloner.ClonePlan`. Simulation-owned contexts consume that same plan through
``context.replicate(plan)`` and derive these execution arrays with
:func:`~isaaclab.cloner.query.replication_mapping`.


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

Newton startup requires a builder; it no longer imports the USD stage implicitly.
``InteractiveScene`` handles planning and replication internally; use it for maintained
demos, tutorials, and asset previews. The explicit cloner examples below are for tests
and code that teaches the cloner API. Native tools can instead supply a builder with
``NewtonManager.set_builder(builder)``.

Require a plan where a consumer uses it, not merely because simulation initializes.
Empty PhysX simulations and tools that supply a native Newton builder need no dummy plan.

ClonePlan
~~~~~~~~~

A plan retains each asset or sensor declaration once and selects its variant for each
environment. Source and destination paths are derived from those declarations, not stored
in a second manifest:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``sources``
     - Original asset and sensor cfg references, including shared assets.
   * - ``destinations``
     - Integer array ``[len(sources), num_envs]`` selecting a variant per asset and environment;
       ``-1`` means no replicated instance.
   * - ``env_ids``
     - Optional NumPy integer array of target env ids; execution requires it.
   * - ``positions``
     - Optional per-env world positions [m], shape ``[num_envs, 3]``.
   * - ``global_paths``
     - Derived shared-asset roots outside the environment namespace; imported once.
   * - ``clone_template``
     - Environment namespace template, such as ``/World/envs/env_{}`` or ``/Lab/Cell{}``.
   * - ``context_source_indices``
     - Clone-context types mapped to indices of the source declarations they consume.

The plan describes replication and routing, not asset geometry or native state. Asset
construction authors the prototypes. Each backend imports its declared roots and records
the geometry-to-native mappings needed by its consumers; consumers bind after native
initialization. Clone contexts apply the plan but do not own native runtime resources.

Deformable imports read prototype meshes and expand their paths through the plan without
copying vertex arrays per environment. Newton cable imports retain ordered native segment
bindings. MPM spawners author render points under the asset before cloning, then bind the
importer's particle ranges. None requires geometry-specific fields on ``ClonePlan`` or
discovery of the completed replicated scene.

For a robot in four environments and a shared ground plane:

.. code-block:: text

    sources      = (robot_cfg, ground_cfg)
    destinations = [[ 0,  0,  0,  0],   # same robot variant in every environment
                    [-1, -1, -1, -1]]   # ground is shared, not replicated

Only declared subtrees are cloned. Declaring ``env_0/Robot`` does not authorize cloning
an undeclared sibling camera. Newton combines homogeneous native prototypes before one
batched replication; that optimization does not expand the USD import scope.

When envs differ — say a cartpole in every env plus a 2-variant obstacle (box into
envs 0/1, sphere into envs 2/3):

.. code-block:: text

    sources      = (cartpole_cfg, obstacle_cfg)
    destinations = [[0, 0, 0, 0],
                    [0, 0, 1, 1]]

The obstacle cfg retains its two spawner variants. Planning assigns their prototype
spawn paths in the first environment that uses each: ``env_0/Obstacle`` and ``env_2/Obstacle``.

Querying a plan
~~~~~~~~~~~~~~~

Anything that has to follow an asset between the two sides of that table — a sensor
resolving its ``prim_path`` back to the prototype it should read, a ray caster
loading one mesh per variant — asks :mod:`isaaclab.cloner.query` rather than
manipulating path strings itself:

.. code-block:: python

    from isaaclab import cloner

    # where does this prototype land in env 3?
    cloner.query.path_to_clone(plan, "/World/envs/env_2/Obstacle", env_id=3)
    # -> "/World/envs/env_3/Obstacle"

    # which envs does this prototype reach at all?
    cloner.query.path_env_ids(plan, "/World/envs/env_2/Obstacle")
    # -> (2, 3)

    # which prototype is env 2's obstacle cloned from?
    cloner.query.path_to_source(plan, "/World/envs/env_2/Obstacle")
    # -> ("/World/envs/env_2/Obstacle", "/World/envs/env_[^/]+/Obstacle", "")

Two obstacle variants share one destination template, so the template alone does not
identify a prototype — the environment does. A concrete path carries it in the clone
slot; a ``env_.*`` wildcard does not, and resolves to one representative variant
unless you pass ``env_id``. Use :func:`~isaaclab.cloner.query.iter_sources` when you
need every variant behind a template. Note that environment ids are not mask columns:
column ``j`` stands for ``env_ids[j]``, and the queries speak ids throughout.

A plan is the *what*. Putting one together and handing it to the backends is
the *how*. Both Manager-based and Direct environments normally declare their
assets on :class:`~isaaclab.scene.InteractiveSceneCfg`; the scene owns the one
clone lifecycle. The lower-level APIs remain available to standalone tools and
tests that deliberately do not depend on :class:`~isaaclab.scene.InteractiveScene`.

``ReplicateSession``
~~~~~~~~~~~~~~~~~~~~

:class:`~isaaclab.cloner.ReplicateSession` is the context manager used by
:class:`~isaaclab.scene.InteractiveScene` to bracket the whole cloning lifecycle.
Entering the block builds and publishes the plan, the body constructs assets at
their planned source paths, and exiting dispatches that same plan:

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

``clone_plan_from_env_0`` + ``replicate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a standalone homogeneous workflow where every env is one copy of env_0,
pass a :class:`~isaaclab.cloner.CloneCfg` and a flat tuple of asset and sensor
cfgs. :func:`~isaaclab.cloner.clone_plan_from_env_0` publishes the plan and
assigns prototype spawn paths before construction:

.. code-block:: python

    asset_cfgs = (robot_cfg, ground_cfg, light_cfg)
    plan = cloner.clone_plan_from_env_0(clone_cfg, asset_cfgs, num_envs=128, env_spacing=2.0)
    robot, _, _ = [cfg.class_type(cfg) for cfg in asset_cfgs]
    cloner.replicate(plan, replicate_physics=clone_cfg.replicate_physics)

Every env receives the same prototype. The tuple is deliberately flat: the
cloner does not inspect a task or scene cfg tree. Prefer
:class:`~isaaclab.scene.InteractiveSceneCfg` for environment implementations and
heterogeneous scenes.


Under the Hood
--------------

Planning retains cfgs in ``sources`` and maps each participating backend to
its subset in ``context_source_indices``. The active physics manager registers its clone
context during simulation initialization. Assets use that context by default;
:attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` can select an explicitly
registered context instead. Renderer and visualizer cfgs declare their required
representations through ``cloning_contexts``. They are registered before planning,
so spawned assets also route to those contexts and to
:class:`~isaaclab.cloner.UsdReplicateContext` when Kit is available.

The backend packages expose different context implementations behind one
execution contract:

.. code-block:: text

    UsdReplicateContext      # replicates USD prim subtrees
    PhysxReplicateContext    # replicates PhysX rigid bodies and articulations
    NewtonReplicateContext   # replicates Newton bodies in its parallel pipeline

:func:`~isaaclab.cloner.replicate` resolves these types through the
:class:`~isaaclab.sim.SimulationContext` clone-context registry, orders them by
``replicate_priority``, and passes the published plan to each one:

.. code-block:: python

    plan = published_clone_plan
    for context_type in plan.context_source_indices:
        sim.clone_contexts[context_type].replicate(plan)

Every maintained lifecycle publishes its plan before asset construction. The
simulation accepts one plan and each backend receives that exact object.
Newton rendering under another physics backend imports only the plan's routed
sources and shared roots, then expands them using the plan's mapping. Its native
model is allocated at ``PHYSICS_READY``, before camera renderers initialize;
model getters do not build representations or discover a finished stage.

Construct visualizers through ``SimulationCfg`` and declare camera renderer cfgs
before cloning. Their constructors register requirements without reading a native
model; initialization binds the realized resources afterward. Interactive scenes
handle this ordering automatically. With the direct cloner API, include cameras
in the asset cfgs supplied to the planner.

Standalone previews with no replicated environments can declare their authored
roots as a global-only plan. This keeps their existing native physics initialization:

.. code-block:: python

    # These cfgs declare their existing prims outside the environment namespace.
    plan = cloner.make_clone_plan((robot_cfg, light_cfg), 1, 0.0)
    sim.set_clone_plan(plan)
    # Author the declared robot and light here.
    cloner.replicate(plan, replicate_physics=False)
    sim.reset()

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
