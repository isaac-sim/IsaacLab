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

``ClonePlan`` holds numeric ``topology``, host ``asset_cfgs``, an ``env_template``, and
optional world ``positions`` [m]. :class:`~isaaclab.cloner.PrototypeWorldTopology`
contains only the prototype count and numeric relationships below. Each world
prototype lists the asset-prototype indices it contains. Repeating an index creates
another instance with the prototype's default pose. Backends assign native names.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Topology field
     - Meaning
   * - ``num_asset_prototypes``
     - Number of reusable asset definitions, including unused prototypes.
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
        positions=cloner.grid_transforms(16, spacing=2.0)[0],
    )
    topology = plan.topology

.. code-block:: text

    num_asset_prototypes   = 2
    world_prototypes       = [0,1, 0,1,1, 0,0,1, 1]
    world_prototype_starts = [0,0,2,5,8,9]
    world_prototype_layout = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]

The leading ``[0, 0]`` describes an empty shared world. To include a shared ground,
append its cfg to the input ``asset_cfgs`` and pass ``shared_assets=(2,)``.
The shared slice then contains index ``2``; its membership is never sampled.

Optional ``weights`` assign relative probabilities to world prototypes. The default
sequential strategy allocates contiguous groups of worlds in those proportions.
The random strategy samples world prototypes independently. Neither strategy
duplicates prototype definitions to represent weights.

``make_clone_plan`` does not mutate cfgs or construct a stage. In the normal workflow,
``InteractiveScene`` resolves spawner variants into concrete asset prototypes and owns
authoring and replication. The plan holds four fields: ``topology``, ``asset_cfgs``,
``env_template``, and ``positions``. Numeric topology indexes the host cfg table;
it contains neither cfg objects nor naming or placement. Path utilities derive source
paths and destination names from the plan. Consumers use these utilities without
accessing clone contexts. Clone contexts execute replication; they own neither plan
metadata nor native runtime resources.

Only declared subtrees are cloned. Declaring ``env_0/Robot`` does not authorize cloning
an undeclared sibling camera. Newton composes each selected world prototype once
before batched native replication; this does not expand its USD import scope.

Querying topology and native paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Path matching runs on the host and returns NumPy ``int32`` prototype IDs:

.. code-block:: python

    cloner.path.get_asset_prototypes(plan, banana_cfg.prim_path)  # array([0])
    cloner.path.get_world_prototypes(plan, banana_cfg.prim_path)  # array([0, 1, 2])

    cfg = plan.asset_cfgs[asset_id]
    start, end = plan.topology.world_prototype_starts[world_prototype_id + 1 : world_prototype_id + 3]
    asset_ids = plan.topology.world_prototypes[start:end]

Omitting ``path_expr`` selects all definitions, including unused prototypes and shared
world prototype ``-1``. A filter matches the exact declared cfg ``prim_path`` or a regular
expression against that complete string. Generated native paths are not matched.

Numeric queries accept prototype IDs, not strings. All three return
``(world_indices, world_starts)``, with one independent result per input ID:

* ``get_asset_prototype_world_index``: one world index per asset instance.
* ``get_asset_prototype_unique_world_index``: each containing world once per queried asset.
* ``get_world_prototype_world_index``: worlds using each queried world prototype.

For a compact version of the Banana/Franka example, select one of each world prototype:

.. code-block:: python

    plan = cloner.make_clone_plan(
        (banana_cfg, franka_cfg), ((0, 1), (0, 1, 1), (0, 0, 1), (1,)), num_worlds=4
    )
    worlds, starts = cloner.query.get_asset_prototype_world_index(plan.topology, np.array([0, 1, 0]))
    # worlds = [0, 1, 2, 2,   0, 1, 1, 2, 3,   0, 1, 2, 2]
    # starts = [[0, 0, 1, 2, 4, 4],
    #           [4, 4, 5, 7, 8, 9],
    #           [9, 9, 10, 11, 13, 13]]
    begin, end = starts[1, 2:4]  # Query 1, world 1: worlds[5:7] == [1, 1].

``world_indices`` is flat ``int32``; ``world_starts`` is ``int64`` with shape
``[num_queries, num_worlds + 2]``. Query ``q``, world ``w`` uses
``world_starts[q, w + 1 : w + 3]``. The leading slice belongs to shared world ``-1``;
empty worlds have equal start/end offsets. A row's first and last offsets delimit the
whole query, so no additional query-offset array is needed. Repeated query IDs retain
separate results. A scalar integer is a batch of length one. These offsets address
selected plan instances, not native body or particle buffers.

Unused IDs produce empty slices. Querying world prototype ``-1`` returns the shared
world even when it contains no assets. Resolve multiple path matches first with
``cloner.path``, then pass the resulting integer array to a numeric query.

Explicit Warp materialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Planning and initialization queries stay in NumPy. When runtime code needs device-side
queries, explicitly materialize the topology's three numeric arrays. The returned
:class:`~isaaclab.cloner.PrototypeWorldTopology` contains only the prototype count and numeric arrays:

.. code-block:: python

    import warp as wp

    topology = cloner.to_warp(plan.topology, device="cuda:0")
    query_ids = wp.array([0, 1, 0], dtype=wp.int32, device="cuda:0")

    # A safe capacity for any IDs: each query can select at most every instance.
    counts = np.diff(plan.topology.world_prototype_starts)
    num_instances = counts[0] + counts[plan.topology.world_prototype_layout + 1].sum()
    out = (
        wp.empty(len(query_ids) * int(num_instances), dtype=wp.int32, device="cuda:0"),
        wp.empty((len(query_ids), len(plan.topology.world_prototype_layout) + 2), dtype=wp.int64, device="cuda:0"),
    )
    cloner.query.get_asset_prototype_world_index(topology, query_ids, out=out)

``to_warp`` has no device cache: the lifecycle owner calls it once and shares the
returned topology. Matching contiguous NumPy storage is borrowed on CPU; CUDA
materialization copies it. Both keep their arrays alive after the host plan is released.
Treat topology as read-only after planning. Cfgs stay on the host in ``plan.asset_cfgs``;
the naming template and positions are not part of this materialization, and
there is no host/device synchronization of later edits.

Warp queries require resident ``int32`` IDs and preallocated output arrays. Warm the query
once before CUDA graph capture, then replay with changed ID contents and the same buffers.
No query converts the plan, allocates result buffers, or reads results back to the CPU.
The output prefix ends at ``world_starts[-1, -1]``; device consumers use the offsets
directly. Allocate for every selection allowed during replay. If capacity is insufficient,
the query reports the required offsets but leaves indices untouched, never a partial result.
For the world-prototype query, ``num_queries * (num_worlds + 1)`` is a safe capacity.
An empty batch returns empty indices and zero rows of boundaries.

When both topology and selection are fixed, compute the result once in NumPy and upload
that result instead of repeating the query during training.

Native names are separate from topology. Compose path primitives with plan indexing
when a consumer starts with a concrete destination path rather than an asset cfg:

.. code-block:: python

    plan = sim.get_clone_plan()
    path = "/World/envs/env_2/Robot/hand"
    world, _ = cloner.path.match(path, plan.env_template)
    world_id = int(world)
    world_prototype_id = plan.topology.world_prototype_layout[world_id]
    start, end = plan.topology.world_prototype_starts[world_prototype_id + 1 : world_prototype_id + 3]
    asset_ids = plan.topology.world_prototypes[start:end]

    # Names distinguish repeated occurrences of the same asset prototype.
    matches = []
    for asset_id, source, destination, worlds in cloner.path.get_instance_paths(plan):
        if asset_id in asset_ids and world_id in worlds:
            suffix = cloner.path.relative_to(path, destination.format(world_id))
            if suffix is not None:
                matches.append((source, suffix))
    # A separately declared child owns its descendants instead of its parent.
    source, suffix = min(matches, key=lambda item: len(item[1]))
    hand = sim.stage.GetPrimAtPath(source + suffix)

Segment primitives such as ``match``, ``relative_to``, and ``rebase`` do not inspect a plan or choose a world.
For a world expression, select the matching world IDs and process each relevant world
prototype. Preserve member positions when matching repeated assets; prototype IDs alone
do not distinguish their native names. A consumer that already knows its asset cfg can
select its prototype IDs directly and use their native bindings without parsing a path.
Shared assets use the leading world ``-1`` slice, not a negative index into
``world_prototype_layout``. No lookup requires walking cloned USD destinations.

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

Planning retains cfgs in ``plan.asset_cfgs``. Dispatch derives each backend's
participating asset indices from those declarations and constructs the required clone
contexts. The active physics manager declares the default context type;
:attr:`~isaaclab.assets.AssetBaseCfg.cloning_contexts` can override it per asset.
Renderer and visualizer cfgs declare their required representations before planning.
Spawned assets also route to those contexts and to
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
