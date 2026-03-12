.. _cloning-environments:

Cloning Environments
====================

.. currentmodule:: isaaclab

Isaac Lab uses a **template-based cloning** system to efficiently replicate environments for
parallel simulation. Instead of authoring each environment individually on the USD stage,
you define a single template and let the cloner stamp out copies with optional per-environment
variation.

This guide covers the cloning API and how to customize environment creation.

How Cloning Works
-----------------

The cloning pipeline has three stages:

1. **Template authoring** -- You place one or more *prototype* prims under a template root
   (default ``/World/template``). Each prototype is a variant of an asset (e.g., different robot
   configurations or object meshes).

2. **Clone plan** -- The cloner discovers prototypes, enumerates all possible combinations (one
   per prototype group), and assigns a combination to each environment using a *strategy*.

3. **Replication** -- The selected prototypes are replicated to per-environment prim paths via
   USD spec copying and physics-backend-specific replication.

Most users interact with cloning indirectly through
:class:`~isaaclab.scene.InteractiveScene`, which calls
:func:`~isaaclab.cloner.clone_from_template` during ``clone_environments()``.
For advanced use cases, you can call the cloning utilities directly.


Basic Usage
-----------

The simplest case is homogeneous cloning -- every environment gets the same assets:

.. code-block:: python

    from isaaclab.cloner import TemplateCloneCfg, clone_from_template
    from isaaclab.sim import SimulationContext

    sim = SimulationContext()
    stage = sim.stage

    # Spawn a single prototype under the template root using a spawner
    import isaaclab.sim as sim_utils

    spawn_cfg = sim_utils.UsdFileCfg(usd_path="path/to/robot.usd")
    spawn_cfg.func("/World/template/Robot/proto_asset_0", spawn_cfg)

    # Configure and clone
    clone_cfg = TemplateCloneCfg(device=sim.cfg.device)
    clone_from_template(stage, num_clones=128, template_clone_cfg=clone_cfg)

This creates 128 environments at ``/World/envs/env_0`` through ``/World/envs/env_127``,
each containing a copy of the robot.


Configuration Reference
-----------------------

:class:`~isaaclab.cloner.TemplateCloneCfg` controls the cloning behavior:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``template_root``
     - ``"/World/template"``
     - Root path under which prototype prims are authored.
   * - ``template_prototype_identifier``
     - ``"proto_asset"``
     - Name prefix used to discover prototype prims. The cloner finds all prims whose
       base name starts with this identifier (e.g., ``proto_asset_0``, ``proto_asset_1``).
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
   * - ``visualizer_clone_fn``
     - ``None``
     - Optional callback to prebuild visualizer artifacts from the clone plan.
   * - ``clone_strategy``
     - ``random``
     - Strategy function for assigning prototypes to environments. See
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

When multiple prototypes exist in the template, the **clone strategy** determines which
prototype each environment receives. Isaac Lab provides two built-in strategies:

**Random** (default)

Each environment receives a randomly sampled prototype combination:

.. code-block:: python

    from isaaclab.cloner import TemplateCloneCfg, random

    clone_cfg = TemplateCloneCfg(
        clone_strategy=random,
        device="cuda:0",
    )

This is useful for domain randomization and curriculum learning where you want diverse
environments.

**Sequential**

Prototypes are assigned in round-robin order (``env_id % num_combinations``):

.. code-block:: python

    from isaaclab.cloner import TemplateCloneCfg, sequential

    clone_cfg = TemplateCloneCfg(
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

To create environments with different assets, place multiple prototypes under the same
group in the template:

.. code-block:: python

    # Spawn three different object prototypes under the same group
    import isaaclab.sim as sim_utils

    sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5)).func(
        "/World/template/Object/proto_asset_0", sim_utils.CuboidCfg(size=(0.5, 0.5, 0.5))
    )
    sim_utils.ConeCfg(radius=0.25, height=0.5).func(
        "/World/template/Object/proto_asset_1", sim_utils.ConeCfg(radius=0.25, height=0.5)
    )
    sim_utils.SphereCfg(radius=0.25).func(
        "/World/template/Object/proto_asset_2", sim_utils.SphereCfg(radius=0.25)
    )

    clone_cfg = TemplateCloneCfg(
        clone_strategy=sequential,
        device="cuda:0",
    )
    clone_from_template(stage, num_clones=128, template_clone_cfg=clone_cfg)
    # env_0 gets Cuboid, env_1 gets Cone, env_2 gets Sphere, env_3 gets Cuboid, ...

When prototypes span multiple groups (e.g., different robots *and* different objects),
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
- :doc:`/source/overview/core-concepts/multi_backend_architecture` -- how backends are selected
  and how cloning integrates with the factory pattern
