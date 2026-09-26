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

Production scene construction keeps asset definitions and world membership in a
:class:`~isaaclab.cloner.ClonePlan`. Each simulation-owned clone context receives that
plan and the asset-prototype indices routed to it. Native names and placement belong
to the context, not the plan. Clone contexts consume native instance groups and their
destination world IDs directly, without constructing a dense source-to-world mask.
Only standalone calls to the raw APIs above accept caller-supplied masks.


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

A plan stores topology only. Asset prototypes are reusable cfg definitions; each world
prototype lists the asset-prototype indices it contains. Repeating an index creates
another instance with the prototype's default pose. Backends assign native names.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``asset_prototypes``
     - Concrete asset cfg references, one per reusable prototype.
   * - ``world_prototypes``
     - Flat array of asset-prototype indices, including repeated instances.
   * - ``world_prototype_starts``
     - Slice boundaries into ``world_prototypes``. The first slice is the shared world ``-1``.
   * - ``world_prototype_layout``
     - World-prototype index selected for each world, indexed by world ID.

For two reusable assets, four compositions, and sixteen destination worlds:

.. code-block:: python

    plan = cloner.make_clone_plan(
        (banana_cfg, franka_cfg),
        ((0, 1), (0, 1, 1), (0, 0, 1), (1,)),
        16,
    )

.. code-block:: text

    asset_prototypes       = (banana_cfg, franka_cfg)
    world_prototypes       = [0,1, 0,1,1, 0,0,1, 1]
    world_prototype_starts = [0,0,2,5,8,9]
    world_prototype_layout = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]

The leading ``[0, 0]`` describes an empty shared world. To include a shared ground,
append its cfg to ``asset_prototypes`` and pass ``shared_assets=(2,)``.
The shared slice then contains index ``2``; its membership is never sampled.

Optional ``weights`` assign relative probabilities to world prototypes. The default
sequential strategy allocates contiguous groups of worlds in those proportions.
The random strategy samples world prototypes independently. Neither strategy
duplicates prototype definitions to represent weights.

``make_clone_plan`` does not mutate cfgs or construct a stage. In the normal workflow,
``InteractiveScene`` resolves spawner variants into concrete asset prototypes and owns
authoring and replication. Its USD context holds native source paths, destination
templates, and environment origins. Other backends import those declared prototypes
and realize the same topology. Clone contexts do not own native runtime resources.

Only declared subtrees are cloned. Declaring ``env_0/Robot`` does not authorize cloning
an undeclared sibling camera. Newton composes each selected world prototype once
before batched native replication; this does not expand its USD import scope.

Querying topology and native paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Topology queries depend only on the plan and return 1-D NumPy arrays of prototype IDs with dtype ``int32``:

.. code-block:: python

    cloner.query.get_asset_prototypes(plan, banana_cfg.prim_path)  # array([0], dtype=int32)
    cloner.query.get_world_prototypes(plan, banana_cfg.prim_path)  # array([0, 1, 2], dtype=int32)

    cfg = plan.asset_prototypes[asset_id]
    start, end = plan.world_prototype_starts[world_prototype_id + 1 : world_prototype_id + 3]
    asset_ids = plan.world_prototypes[start:end]

Omitting ``path_expr`` selects all definitions, including unused prototypes; world-prototype IDs
include ``-1`` for the shared world, even when empty. A filter is either the exact declared
cfg ``prim_path`` or a regular expression matched against that complete path string.
World filtering selects compositions containing matching assets, without removing their
other members or repeated instances. These IDs identify prototypes, not destination worlds.

The complementary queries return actual world indices, also as 1-D NumPy ``int32`` arrays:

.. code-block:: python

    # World prototype 1 occupies worlds 4 through 7 in the sixteen-world example above.
    cloner.query.get_world_prototype_world_index(plan, 1)  # array([4, 5, 6, 7], dtype=int32)

    # Franka is asset prototype 1 and occurs twice in each of those worlds.
    worlds = cloner.query.get_asset_prototype_world_index(plan, 1)
    # array([0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=int32)
    worlds = cloner.query.get_asset_prototype_world_index(plan, franka_cfg.prim_path, unique=True)
    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=int32)

``get_asset_prototype_world_index`` accepts an integer prototype index or the same declared-path
expression as ``get_asset_prototypes``. It preserves one entry per selected asset instance by
default; ``unique=True`` returns each containing world once. Shared instances use world index
``-1``. ``get_world_prototype_world_index(plan, -1)`` returns ``[-1]``, even for an empty shared world.
An unused world prototype or an asset with no instances returns an empty array.

Native-path queries instead resolve backend-assigned instance names and descendants.
They take plain native mapping data, not a clone context or a plan. For example,
a USD-backed consumer gets its source/destination mappings from the USD context:

.. code-block:: python

    usd = sim.clone_contexts[cloner.UsdReplicateContext]
    instances = usd.instances
    # Each entry: asset-prototype ID, source path, destination template, world IDs.
    matches = cloner.query.get_matched_sources(instances, "/World/envs/env_[^/]+/Obstacle")
    for source_root, destination, source_path, world_ids in matches:
        ...

:func:`~isaaclab.cloner.query.get_matched_sources` returns a list of all populated
instance groups behind the nearest matching destination declaration, not a generator.
The existing :func:`~isaaclab.cloner.query.path_to_source` selects one representative
prototype; a concrete path or explicit ``env_id`` selects its world.
:func:`~isaaclab.cloner.query.path_to_clone` rejects ambiguous single-instance requests
when a world contains the same asset more than once. The generic query module never
imports a backend context.

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

Planning retains cfgs in ``asset_prototypes``. Dispatch derives each backend's
participating asset indices from those declarations. The active physics manager
registers its clone context during simulation initialization. Assets use that context by default;
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
``replicate_priority``, and passes the published plan with the selected asset-prototype IDs:

.. code-block:: python

    # A backend receives the same topology, restricted to its routed asset definitions.
    context.replicate(plan, asset_prototype_ids)

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
    plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), (robot_cfg, light_cfg), 1, 0.0)
    # Author the declared robot and light here.
    cloner.replicate(plan, replicate_physics=False)
    sim.reset()

USD runs before native physics contexts so the destination topology exists when
they consume it. Dispatch derives routing from the asset and consumer cfgs; the plan does not store a second routing map.

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
