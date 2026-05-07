.. _cloning-environments:

Cloning Environments
====================

.. currentmodule:: isaaclab

Isaac Lab uses a **clone-plan-based** system to efficiently replicate environments for
parallel simulation. Instead of authoring each environment independently, the scene first
builds a :class:`~isaaclab.cloner.ClonePlan` from asset configuration, spawns the
representative source prims directly in their selected environments, and then replicates those
sources to every destination selected by the plan.

This guide covers the cloning API and how to customize environment creation.

How Cloning Works
-----------------

The cloning pipeline has three stages:

1. **Plan** -- The scene inspects asset configuration and computes a
   :class:`~isaaclab.cloner.ClonePlan`. Homogeneous scenes use the default
   ``env_0`` source plan. Heterogeneous scenes add one source row for each spawned variant.

2. **Spawn sources** -- Spawner configuration is rewritten to point at representative source
   paths such as ``/World/envs/env_0/Object`` or ``/World/envs/env_5/Object``. Multi-asset
   spawners receive an explicit ``spawn_paths`` list.

3. **Replicate** -- The selected source prims are replicated to per-environment prim paths via
   USD spec copying and physics-backend-specific replication.

Most users interact with cloning indirectly through
:class:`~isaaclab.scene.InteractiveScene`. For advanced use cases, you can call the
lower-level planning and replication utilities directly.


Basic Usage
-----------

The simplest case is homogeneous cloning -- every environment gets the same assets:

.. code-block:: python

    import torch
    from isaaclab.cloner import usd_replicate
    from isaaclab.sim import SimulationContext

    sim = SimulationContext()
    stage = sim.stage

    # Spawn the representative source in env_0.
    import isaaclab.sim as sim_utils

    spawn_cfg = sim_utils.UsdFileCfg(usd_path="path/to/robot.usd")
    spawn_cfg.func("/World/envs/env_0/Robot", spawn_cfg)

    # Replicate env_0 to every environment.
    usd_replicate(
        stage,
        sources=["/World/envs/env_0"],
        destinations=["/World/envs/env_{}"],
        env_ids=torch.arange(128),
        mask=torch.ones((1, 128), dtype=torch.bool),
    )

This creates 128 environments at ``/World/envs/env_0`` through ``/World/envs/env_127``,
each containing a copy of the robot.


Configuration Reference
-----------------------

:class:`~isaaclab.cloner.CloneCfg` controls scene replication behavior:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``clone_regex``
     - ``"/World/envs/env_.*"``
     - Destination path template. The ``.*`` is replaced with the environment index.
   * - ``clone_usd``
     - ``True``
     - Whether to replicate USD prim specs to destination paths.
   * - ``clone_physics``
     - ``True``
     - Whether to perform physics-backend-specific replication.
   * - ``physics_clone_fn``
     - ``None``
     - Backend-specific physics replication function. Set automatically by
       :class:`~isaaclab.scene.InteractiveScene`.
   * - ``clone_strategy``
     - ``random``
     - Strategy function for assigning source variants to environments. See
       :ref:`cloning-strategies` below.
   * - ``device``
     - ``"cpu"``
     - Torch device for mapping buffers.
   * - ``clone_in_fabric``
     - ``False``
     - Enable cloning in Fabric (PhysX only, experimental).


.. _cloning-strategies:

Cloning Strategies
------------------

When multiple source variants exist, the **clone strategy** determines which variant each
environment receives. Isaac Lab provides two built-in strategies:

**Random** (default)

Each environment receives a randomly sampled prototype combination:

.. code-block:: python

    from isaaclab.cloner import CloneCfg, random

    clone_cfg = CloneCfg(
        clone_strategy=random,
        device="cuda:0",
    )

This is useful for domain randomization and curriculum learning where you want diverse
environments.

**Sequential**

Prototypes are assigned in round-robin order (``env_id % num_combinations``):

.. code-block:: python

    from isaaclab.cloner import CloneCfg, sequential

    clone_cfg = CloneCfg(
        clone_strategy=sequential,
        device="cuda:0",
    )

This produces a deterministic, balanced distribution -- useful for reproducible experiments.

**Custom strategies** can be written as any callable matching the signature
``(combinations: torch.Tensor, num_clones: int, device: str) -> torch.Tensor``,
where ``combinations`` has shape ``(num_combinations, num_groups)`` and the return
value has shape ``(num_clones, num_groups)``.


Heterogeneous Environments
--------------------------

To create environments with different assets, use a multi-asset spawner or provide multiple
source groups to :func:`~isaaclab.cloner.make_clone_plan`:

.. code-block:: python

    from isaaclab.cloner import make_clone_plan, sequential, usd_replicate
    import isaaclab.sim as sim_utils
    import torch

    plan = make_clone_plan(
        sources=[[
            "/World/envs/env_0/Object",
            "/World/envs/env_1/Object",
            "/World/envs/env_2/Object",
        ]],
        destinations=["/World/envs/env_{}/Object"],
        num_clones=128,
        clone_strategy=sequential,
        device="cuda:0",
    )

    sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5)).func(plan.sources[0], sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5)))
    sim_utils.ConeCfg(radius=0.25, height=0.5).func(plan.sources[1], sim_utils.ConeCfg(radius=0.25, height=0.5))
    sim_utils.SphereCfg(radius=0.25).func(plan.sources[2], sim_utils.SphereCfg(radius=0.25))

    usd_replicate(stage, plan.sources, plan.destinations, torch.arange(128), plan.clone_mask)
    # env_0 gets Cuboid, env_1 gets Cone, env_2 gets Sphere, env_3 gets Cuboid, ...

When variants span multiple groups (e.g., different robots *and* different objects),
the cloner enumerates the Cartesian product of all groups and assigns combinations
using the selected strategy.


Environment Positioning
-----------------------

Environments are arranged in a grid layout using :func:`~isaaclab.cloner.grid_transforms`:

.. code-block:: python

    from isaaclab.cloner import grid_transforms

    positions, orientations = grid_transforms(
        N=128,       # number of environments
        spacing=2.0, # meters between neighbors
        up_axis="Z",
        device="cuda:0",
    )
    # positions: (128, 3), orientations: (128, 4) identity quaternions

:class:`~isaaclab.scene.InteractiveScene` calls this automatically based on
``InteractiveSceneCfg.env_spacing``.


Collision Filtering
-------------------

By default, assets in different environments can collide with each other. To prevent
cross-environment collisions (the typical setup for parallel RL), use
:func:`~isaaclab.cloner.filter_collisions`:

.. code-block:: python

    from isaaclab.cloner import filter_collisions

    filter_collisions(
        stage=stage,
        physicsscene_path="/physicsScene",
        collision_root_path="/World/collisions",
        prim_paths=[f"/World/envs/env_{i}" for i in range(128)],
        global_paths=["/World/defaultGroundPlane"],  # collides with all envs
    )

.. note::

    Collision filtering uses PhysX collision groups and is only applicable to the PhysX backend.
    The Newton backend handles per-environment isolation through its world system.


Physics Backend Replication
---------------------------

Each physics backend has its own replication function that registers cloned prims with the
physics engine:

- **PhysX**: :func:`~isaaclab_physx.cloner.physx_replicate` -- Uses the PhysX replicator
  interface for fast physics body registration.
- **Newton**: :func:`~isaaclab_newton.cloner.newton_physics_replicate` -- Builds a Newton
  ``ModelBuilder`` with per-environment worlds, supporting heterogeneous spawning.

These functions are set automatically when using :class:`~isaaclab.scene.InteractiveScene`.
For direct usage:

.. code-block:: python

    import torch
    from isaaclab_physx.cloner import physx_replicate

    physx_replicate(
        stage=stage,
        sources=["/World/envs/env_0/Robot"],
        destinations=["/World/envs/env_{}/Robot"],  # {} is replaced with env index
        env_ids=torch.arange(128),
        mapping=torch.ones(1, 128, dtype=torch.bool),
        device="cuda:0",
    )


See Also
--------

- :doc:`multi_asset_spawning` -- spawning different assets per environment
- :doc:`optimize_stage_creation` -- fabric cloning and stage-in-memory optimizations
