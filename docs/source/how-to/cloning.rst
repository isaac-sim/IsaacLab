:orphan:

.. _cloning-environments:

Cloning Environments
====================

.. currentmodule:: isaaclab

Cloning builds many simulation environments from a small set of reusable asset
prototypes. A :class:`~isaaclab.cloner.ClonePlan` describes which assets belong in
each environment. Physics and rendering use the same plan.


The cloning lifecycle
---------------------

Scene construction follows one sequence:

1. Declare assets, sensors, physics, renderers, and visualizers in configuration.
   Consumers register their required representations before planning.
2. Build and publish the clone plan, then construct assets at the planned prototype paths.
3. Replicate the declared prototypes through the required backends using that same plan.
4. Initialize consumers and bind them to the resulting native resources.

For tasks and demos, :class:`~isaaclab.scene.InteractiveScene` handles planning,
asset construction, and replication. Given existing ``banana_cfg`` and ``franka_cfg``
asset configurations and an active simulation context ``sim``:

.. code-block:: python

    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    @configclass
    class BananaFrankaSceneCfg(InteractiveSceneCfg):
        banana = banana_cfg.replace(prim_path="{ENV_REGEX_NS}/Banana")
        franka = franka_cfg.replace(prim_path="{ENV_REGEX_NS}/Franka")

    scene = InteractiveScene(BananaFrankaSceneCfg(num_envs=16, env_spacing=2.0))
    sim.reset()

This creates sixteen environments, each with one banana and one Franka.
Declare cameras alongside the assets so they participate in the same lifecycle.
For varying assets across environments, see :doc:`multi_asset_spawning`.


Walking through a ClonePlan
---------------------------

An **asset prototype** is one reusable asset definition. A **world prototype** is
a composition of those assets. For example, two asset prototypes can describe
four different world prototypes:

.. code-block:: python

    from isaaclab import cloner

    asset_cfgs = (banana_cfg, franka_cfg)  # Asset IDs: banana = 0, Franka = 1.
    world_prototypes = (
        (0, 1),     # One banana, one Franka.
        (0, 1, 1),  # One banana, two Frankas.
        (0, 0, 1),  # Two bananas, one Franka.
        (1,),       # One Franka.
    )
    positions = cloner.grid_transforms(16, spacing=2.0)[0]
    plan = cloner.make_clone_plan(asset_cfgs, world_prototypes, 16, positions=positions)

Repeating an asset index creates another instance, not another prototype definition.
With the default equal weights and sequential allocation, each world prototype
is used by four of the sixteen destination worlds.

``ClonePlan`` keeps four things:

* ``asset_cfgs``: the asset definitions, stored once.
* ``topology``: numeric asset membership and world-prototype selection.
* ``env_template``: the destination naming template, with one ``{}`` slot for the world ID.
* ``positions``: optional destination-world origins [m], separate from topology.

The example's topology is stored as flat arrays:

.. code-block:: text

    num_asset_prototypes   = 2
    world_prototypes       = [0,1, 0,1,1, 0,0,1, 1]
    world_prototype_starts = [0,0,2,5,8,9]
    world_prototype_layout = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]

``world_prototype_starts`` separates the asset lists. Its first slice belongs to
the shared world ``-1``; the leading ``[0, 0]`` means there are no shared assets.
For a shared ground, append its cfg to ``asset_cfgs`` and pass ``shared_assets=(2,)``.
That asset is instantiated once, outside the replicated environments.

``world_prototype_layout`` selects a prototype for each destination world: worlds
0–3 use prototype 0, worlds 4–7 use prototype 1, and so on. Optional ``weights``
change how worlds are distributed among prototypes.

``make_clone_plan`` only describes the scene; it does not spawn or replicate assets.
``InteractiveScene`` prepares and executes the plan as part of scene construction.
Only declared asset subtrees participate, not undeclared siblings found on the stage.


Explicit cloning for tests and tools
------------------------------------

Without ``InteractiveScene``, a homogeneous scene can use the same lifecycle
explicitly. Set the asset paths to ``{ENV_REGEX_NS}/Banana`` and
``{ENV_REGEX_NS}/Franka``, then plan before constructing them:

.. code-block:: python

    asset_cfgs = (banana_cfg, franka_cfg)
    plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), asset_cfgs, 16, 2.0)
    banana, franka = [cfg.class_type(cfg) for cfg in asset_cfgs]
    cloner.replicate(plan)
    sim.reset()

Pass a flat collection of asset and sensor cfgs, including shared assets such as
ground and lights. Prefer ``InteractiveScene`` for maintained tasks and demos;
it also handles backend-specific setup such as PhysX collision filtering.

Setting ``replicate_physics=False`` skips the active physics replication context.
Other declared contexts still run, including USD replication when configured, so
they can create the per-environment prims without replicating the physics model.
The physics backend can then parse those prims independently.

See the :doc:`cloner API reference <../api/lab/isaaclab.cloner>` for individual
functions, topology queries, and path utilities.
