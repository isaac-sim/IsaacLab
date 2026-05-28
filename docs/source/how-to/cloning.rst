.. _cloning-environments:

Cloning Environments
====================

.. currentmodule:: isaaclab

Isaac Lab creates many parallel environments by spawning representative source prims and
then cloning them to the remaining environment paths. This guide starts with direct cloning
so the primitive contract is clear, then shows how :class:`~isaaclab.cloner.ClonePlan` and
:class:`~isaaclab.scene.InteractiveScene` build on top of that contract.

.. contents:: On this page
   :local:
   :depth: 2


Direct Cloning
--------------

Use direct cloning for custom scene pipelines, tooling, or tests that need explicit
control over the replication contract.

The cloner operates on three pieces of data:

1. **Source prims** that already exist on the stage.
2. **Destination templates** containing ``{}``, which is formatted with each environment id.
3. **A boolean mask** with shape ``[len(sources), num_envs]`` that selects which source
   populates each environment.

The direct flow is:

1. Create the environment namespace prims.
2. Spawn representative source prims.
3. Call the physics replicate function for your backend.
4. Call :func:`~isaaclab.cloner.usd_replicate` with the same source-to-environment mapping.

.. code-block:: python

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.cloner import usd_replicate
    from isaaclab_physx.cloner import physx_replicate

    num_envs = 128
    stage = sim_utils.get_current_stage()
    env_ids = torch.arange(num_envs, device="cuda:0")

    sim_utils.create_prim("/World/envs", "Xform")
    for env_id in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{env_id}", "Xform")

    source = "/World/envs/env_0/Cube"
    destination = "/World/envs/env_{}/Object"

    cube_cfg = sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5))
    cube_cfg.func(source, cube_cfg)

    mask = torch.ones((1, num_envs), dtype=torch.bool, device="cuda:0")

    physx_replicate(stage, [source], [destination], env_ids, mask, device="cuda:0")
    usd_replicate(stage, [source], [destination], env_ids, mask)

This creates one source cube at ``/World/envs/env_0/Cube`` and clones it to
``/World/envs/env_1/Object`` through ``/World/envs/env_127/Object``. When a source path is
the same as the destination for an environment, ``usd_replicate`` skips the self-copy.

Direct heterogeneous cloning uses the same API with more source rows. Each row in ``mask``
selects the environments that receive the matching source. For example, this explicit mask
clones a cone into environments 0 and 2, and a sphere into environments 1 and 3:

.. code-block:: python

    env_ids = torch.arange(4, device="cuda:0")
    sources = ["/World/envs/env_0/Cone", "/World/envs/env_1/Sphere"]
    destinations = ["/World/envs/env_{}/Object", "/World/envs/env_{}/Object"]

    cone_cfg = sim_utils.ConeCfg(radius=0.25, height=0.5)
    sphere_cfg = sim_utils.SphereCfg(radius=0.25)
    cone_cfg.func(sources[0], cone_cfg)
    sphere_cfg.func(sources[1], sphere_cfg)

    mask = torch.tensor([[True, False, True, False], [False, True, False, True]], dtype=torch.bool)

    physx_replicate(stage, sources, destinations, env_ids, mask, device="cuda:0")
    usd_replicate(stage, sources, destinations, env_ids, mask)

The mask above reads as:

.. list-table::
   :header-rows: 1
   :widths: 15 40 20 25

   * - Source row
     - Source path
     - Env ids
     - Destination path
   * - ``0``
     - ``/World/envs/env_0/Cone``
     - ``0, 2``
     - ``/World/envs/env_{}/Object``
   * - ``1``
     - ``/World/envs/env_1/Sphere``
     - ``1, 3``
     - ``/World/envs/env_{}/Object``

``usd_replicate`` copies parent paths before children and supports optional ``positions``
and ``quaternions`` buffers. If ``positions`` is provided, it authors
``xformOp:translate`` on each destination using the environment id. The helper
:func:`~isaaclab.cloner.grid_transforms` creates the same grid layout used by
:class:`~isaaclab.scene.InteractiveScene`.

.. code-block:: python

    from isaaclab.cloner import grid_transforms

    positions, orientations = grid_transforms(
        N=num_envs,
        spacing=2.0,
        up_axis="z",
        device="cuda:0",
    )
    usd_replicate(stage, [source], [destination], env_ids, mask, positions=positions)


Clone Plans
-----------

For one source row, passing ``sources``, ``destinations``, and ``mask`` by hand is simple.
For heterogeneous scenes, the mapping is easier to build with
:func:`~isaaclab.cloner.make_clone_plan`, which returns the raw flat components. Composing
those components into a :class:`~isaaclab.cloner.ClonePlan` together with the per-environment
pose buffer is the caller's responsibility — keeping pose authority on the side that owns the
buffer (typically the scene) avoids duplicating tensors.

:class:`~isaaclab.cloner.ClonePlan` stores the same flat contract used by direct cloning:

.. code-block:: text

    sources      = [source_0, source_1, ...]
    destinations = [destination_0, destination_1, ...]
    clone_mask   = bool tensor, shape [len(sources), num_envs]

``clone_mask[i, j]`` is ``True`` when environment ``j`` should receive source row ``i``.
The same plan can be passed to USD replication, physics replication, and scene-data
providers.

Homogeneous Plans
~~~~~~~~~~~~~~~~~

In a homogeneous scene, every environment receives the same asset layout. The default plan
is:

.. code-block:: text

    sources      = ["/World/envs/env_0"]
    destinations = ["/World/envs/env_{}"]
    clone_mask   = all True, shape [1, num_envs]

This means the scene spawns everything for ``env_0`` and replicates that environment to
``env_1`` through ``env_N``.

Heterogeneous Plans
~~~~~~~~~~~~~~~~~~~

Heterogeneous cloning is used when different environments receive different prototypes.
For example, an object with three variants may have representative source prims at:

.. code-block:: text

    /World/envs/env_0/Object
    /World/envs/env_1/Object
    /World/envs/env_2/Object

These paths have the same leaf name because each variant will be cloned to
``/World/envs/env_{}/Object``, but their authored contents are different. For example,
``env_0/Object`` could be a cone, ``env_1/Object`` a cuboid, and ``env_2/Object`` a sphere.

The plan maps those source rows to all environments:

.. code-block:: python

    from isaaclab.cloner import make_clone_plan, sequential

    sources, destinations, clone_mask = make_clone_plan(
        sources=[
            [
                "/World/envs/env_0/Object",
                "/World/envs/env_1/Object",
                "/World/envs/env_2/Object",
            ]
        ],
        destinations=["/World/envs/env_{}/Object"],
        num_clones=8,
        clone_strategy=sequential,
        device="cuda:0",
    )

    # source row used by env: 0, 1, 2, 0, 1, 2, 0, 1

Direct code can use the components exactly like the hand-written direct example:

.. code-block:: python

    physx_replicate(stage, sources, destinations, env_ids, clone_mask, device="cuda:0")
    usd_replicate(stage, sources, destinations, env_ids, clone_mask)

When variants span multiple groups, such as robot variants and object variants,
``make_clone_plan`` enumerates the Cartesian product of the groups and assigns one
combination per environment. Unused prototype rows may still appear in the plan with an
all-false mask row.

.. _cloning-strategies:

Clone Strategies
~~~~~~~~~~~~~~~~

A clone strategy chooses prototype combinations for the environments:

* :func:`~isaaclab.cloner.random` samples combinations randomly and is the default.
* :func:`~isaaclab.cloner.sequential` assigns combinations in round-robin order, which is
  useful for reproducible tests and balanced coverage.

Custom strategies are callables with this signature:

.. code-block:: python

    def my_strategy(combinations: torch.Tensor, num_clones: int, device: str) -> torch.Tensor:
        ...

``combinations`` has shape ``[num_combinations, num_groups]`` and the return value must have
shape ``[num_clones, num_groups]``.


Common Workflow: ``InteractiveScene``
-------------------------------------

:class:`~isaaclab.scene.InteractiveScene` automates the direct cloning flow for task scenes.
It inspects scene configuration, builds a :class:`~isaaclab.cloner.ClonePlan`, rewrites
spawner paths to the representative sources, spawns those sources, runs physics and USD
replication, and filters inter-environment collisions for PhysX when configured.

Put per-environment assets under ``{ENV_REGEX_NS}`` and global assets under normal USD
paths:

.. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


    @configclass
    class MySceneCfg(InteractiveSceneCfg):
        # Cloned once per environment.
        robot = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Authored once globally, not cloned per environment.
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0),
        )


    scene_cfg = MySceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)
    scene = InteractiveScene(cfg=scene_cfg)

For heterogeneous scenes, use :class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg`
or :class:`~isaaclab.sim.spawners.wrappers.MultiUsdFileCfg`. ``InteractiveScene`` assigns
representative source paths to the spawner and lets the clone strategy choose which
prototype each environment receives. See :doc:`multi_asset_spawning` for the asset
configuration details.

The most important scene options are on :class:`~isaaclab.scene.InteractiveSceneCfg`:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - When to change it
   * - ``replicate_physics``
     - ``True``
     - Keep enabled for homogeneous environments and fast startup. Disable it when each
       environment needs independently authored physics or USD randomization.
   * - ``filter_collisions``
     - ``True``
     - Keep enabled for parallel RL so cloned environments do not collide with each other.
       This is automatic for PhysX-backed scene cloning.
   * - ``clone_in_fabric``
     - ``False``
     - Enables the PhysX Fabric cloning path for faster scene creation. Use USDRT for stage
       inspection when Fabric cloning is enabled.


Choosing an API
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Goal
     - Recommended API
     - Notes
   * - Build a custom cloning pipeline
     - :func:`~isaaclab.cloner.usd_replicate` and a backend physics replicate function
     - Useful for tests, tooling, or advanced scene construction.
   * - Build complex direct mappings
     - :func:`~isaaclab.cloner.make_clone_plan`
     - Produces the same ``sources``, ``destinations``, and ``clone_mask`` used by direct cloning.
   * - Build normal task scenes
     - :class:`~isaaclab.scene.InteractiveScene`
     - Preferred path. Configure assets with ``{ENV_REGEX_NS}`` and let the scene clone them.
   * - Randomize which asset each environment receives
     - ``InteractiveScene`` with :class:`~isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg` or
       :class:`~isaaclab.sim.spawners.wrappers.MultiUsdFileCfg`
     - See :doc:`multi_asset_spawning` for the asset configuration details.
   * - Use Isaac Sim's ``GridCloner``
     - Isaac Sim API
     - Isaac Lab's tested path is the ``isaaclab.cloner`` API described here.


Migrating From Template Cloning
-------------------------------

The template-root discovery API has been removed. Replace
``clone_from_template(...)`` calls with explicit source prims plus
:func:`~isaaclab.cloner.make_clone_plan`, a backend physics replicate function, and
:func:`~isaaclab.cloner.usd_replicate`. Replace ``TemplateCloneCfg`` with
:class:`~isaaclab.cloner.CloneCfg` for execution settings such as clone strategy,
Fabric cloning, and backend replication.


Collision Filtering and Isolation
---------------------------------

Some prims, such as terrain, are intentionally shared across environments and should collide
with every environment. These are modeled as global collision paths. The workaround is only
the per-environment filtering: when cloning is fully isolated per world, cloned environments
should not collide with each other and no manual per-environment filter should be needed.
Some PhysX cloning paths still rely on USD collision groups for that isolation fallback. In
the scene workflow this is handled by ``InteractiveScene`` when ``filter_collisions=True``
and the backend is PhysX.

For direct PhysX usage, call :func:`~isaaclab.cloner.filter_collisions` after cloning if
per-environment isolation is not already provided by the cloning backend:

.. code-block:: python

    from isaaclab.cloner import filter_collisions

    filter_collisions(
        stage=stage,
        physicsscene_path="/physicsScene",
        collision_root_path="/World/collisions",
        prim_paths=[f"/World/envs/env_{i}" for i in range(num_envs)],
        global_paths=["/World/ground"],
    )

.. note::

    Collision filtering uses PhysX collision groups. Newton handles per-environment isolation
    through its own world system.


Backend and Option Notes
------------------------

**Physics replication.** :class:`~isaaclab.scene.InteractiveScene` selects the backend
replication function automatically. Direct PhysX users call
:func:`~isaaclab_physx.cloner.physx_replicate`; Newton users call
:func:`~isaaclab_newton.cloner.newton_physics_replicate`.

**``replicate_physics=False``.** Disable physics replication when environments need
independent authored USD or physics state, such as some scale, texture, or color
randomization workflows. Startup and physics parsing are slower because the backend cannot
assume every environment is a clone of the same source.

**``copy_from_source``.** ``InteractiveScene`` calls
``clone_environments(copy_from_source=True)`` when ``replicate_physics=False``. This skips
backend physics replication and leaves physics parsing to the backend. Spawner-level
``copy_from_source`` is a separate setting used by spawn functions that clone from a source
path matched by a regex.

**Fabric cloning.** ``clone_in_fabric=True`` applies to PhysX replication. It can reduce
scene-creation time for large PhysX scenes, especially when many replicated rigid bodies are
authored. Fabric-backed stage data must be inspected through USDRT rather than normal USD
APIs.


See Also
--------

* :doc:`multi_asset_spawning` -- configuring multi-asset and multi-USD spawners.
* :doc:`optimize_stage_creation` -- Fabric cloning and stage-in-memory optimizations.
