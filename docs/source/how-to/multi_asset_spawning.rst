:orphan:

Spawning Multiple Assets
========================

.. currentmodule:: isaaclab

Typical spawning configurations (introduced in the :ref:`tutorial-spawn-prims` tutorial) copy one asset across all
prim paths resolved from an expression. Multi-asset workflows cover two related composition needs:

1. A rigid object collection batches several rigid objects in every environment behind one data and command API.
2. A multi-asset spawner declares several variants for one scene asset binding, allowing environments to contain
   different geometry or robot variants.

This guide demonstrates both mechanisms and explains how their execution differs between PhysX and Newton.

The packaged ``multi-asset`` example provides the reference script, ``multi_asset.py``.

.. dropdown:: Code for multi_asset.py
   :icon: code

   .. literalinclude:: ../../../examples/multi_asset.py
      :language: python
      :linenos:

With the PhysX configuration (``--physics isaacsim_physx``), this script creates multiple environments containing:

* a rigid object collection containing a sphere, a cube, and a cylinder
* a rigid object selected from nine geometry and material variants by the clone plan
* an articulation selected from the ANYmal-C and ANYmal-D variants by the clone plan

.. image:: ../_static/demos/multi_asset.jpg
  :width: 100%
  :alt: result of multi_asset.py


Rigid object collections
------------------------

Use a rigid object collection when every environment contains the same set of independently moving rigid bodies and you
want to access them as one batch. The collection exposes data with an ``(env, object, ...)`` layout and accepts
``(env_ids, obj_ids)`` selections for commands. Compared with managing each object separately, the collection uses one
batched physics view.

.. literalinclude:: ../../../examples/multi_asset.py
   :language: python
   :start-at: object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
   :end-before: # articulation
   :dedent: 4

The :class:`~assets.RigidObjectCollectionCfg` configuration owns a dictionary of :class:`~assets.RigidObjectCfg`
instances. Each dictionary key is the object's stable identifier within the collection.

The example resets all collection members through the same API used by both physics backends:

.. literalinclude:: ../../../examples/multi_asset.py
   :language: python
   :start-at: default_pose_w = rigid_object_collection.data.default_body_pose.torch.clone()
   :end-at: rigid_object_collection.write_body_com_velocity_to_sim_index(body_velocities=default_vel_w)
   :dedent: 12


Spawning variants for one scene asset
-------------------------------------

Use :class:`~sim.spawners.wrappers.MultiAssetSpawnerCfg` and :class:`~sim.spawners.wrappers.MultiUsdFileCfg` to declare
the available variants for one scene asset binding. :class:`~scene.InteractiveScene` includes these variants in its clone
plan and assigns one valid prototype combination to each environment.

For configuration-based assets, assign :class:`~sim.spawners.wrappers.MultiAssetSpawnerCfg` to the
:class:`~assets.RigidObjectCfg` spawn configuration:

.. literalinclude:: ../../../examples/multi_asset.py
   :language: python
   :start-at: object: RigidObjectCfg = RigidObjectCfg(
   :end-before: # object collection
   :dedent: 4

The ``assets_cfg`` list defines the prototypes available to the clone plan. Variant assignment is controlled by
:attr:`~cloner.CloneCfg.clone_strategy`; the default :func:`~cloner.sequential` strategy assigns combinations in
round-robin order. To sample combinations randomly instead, set the strategy before constructing the scene:

.. code-block:: python

   from isaaclab import cloner

   scene_cfg.clone_cfg.clone_strategy = cloner.random

For USD assets, assign :class:`~sim.spawners.wrappers.MultiUsdFileCfg` to the
:class:`~assets.ArticulationCfg` spawn configuration:

.. literalinclude:: ../../../examples/multi_asset.py
   :language: python
   :start-at: robot: ArticulationCfg = ArticulationCfg(
   :end-before: ##
   :dedent: 4

To spawn SimReady props found by a search, assign :class:`~sim.spawners.wrappers.SimReadyUsdFileCfg`, a
:class:`~sim.spawners.wrappers.MultiUsdFileCfg` whose USD paths come from
:func:`~utils.assets.search_simready_usd_paths`. The search result is cached in a JSON file, and each prop's
rigid body is moved onto the spawned prim so that all props share one rigid-body layout. The ``simready``
object preset of the lift environments uses it, for example in the Kuka-Allegro lift environment:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/lift/lift_env_cfg.py
   :language: python
   :start-at: simready = sim_utils.SimReadyUsdFileCfg(
   :end-before: default = shapes
   :dedent: 4

Variant compatibility
~~~~~~~~~~~~~~~~~~~~~

All variants behind one batched asset interface must have a compatible structure. Articulation variants must have the
same links, joints, collision-body count, and names. Rigid object variants can differ in geometry and material while
retaining a compatible rigid-body layout. Model structurally different assets as separate scene bindings.

Clone planning and physics replication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~scene.InteractiveScene` represents multi-asset variants as clone-plan prototypes and dispatches each
prototype only to its assigned environments. Physics assets use their declared clone contexts, including native
physics replication by default. Declare per-environment stage differences as
:attr:`~isaaclab.cloner.CloneCfg.clone_combinations` prototypes.

The example relies on physics replication. For Newton, it also narrows the standalone object and articulation to one
variant because their batched Newton views currently require a uniform body layout across worlds:

.. literalinclude:: ../../../examples/multi_asset.py
   :language: python
   :start-at: scene_cfg = MultiObjectSceneCfg(num_envs=args_cli.num_envs
   :end-at: scene_cfg.robot.spawn.usd_path = scene_cfg.robot.spawn.usd_path[0]
   :dedent: 8

An :class:`~isaaclab.cloner.InclusionSet` declares which assets are active and which semantic tag
values must match. No subclass or callback is needed. For a scene with ``gear`` and ``fixture`` assets:

.. code-block:: python

   scene_cfg.clone_cfg.clone_combinations = [
       InclusionSet(assets=["gear", "fixture"], match_tags=["size", "assembly"]),
   ]

Tag the individual spawner configurations using the existing semantic API:

.. code-block:: python

   small_gear_cfg.semantic_tags = [("size", "small"), ("assembly", "drive")]
   small_fixture_cfg.semantic_tags = [("size", "small"), ("assembly", "drive")]
   universal_fixture_cfg.semantic_tags = [
       ("size", "small"), ("size", "medium"), ("size", "large"),
       ("assembly", "drive"),
   ]

Every selected gear and fixture must share a value for each requested tag type. The universal fixture
can therefore match small, medium, or large gears in the drive assembly. Missing requested tags do not
match. Tag values are compared exactly; there are no reserved wildcard values or implicit regex rules.
Tags inherited from ``MultiAssetSpawnerCfg`` also participate. ``MultiUsdFileCfg`` candidates inherit
the wrapper's tags and retain their individual USD paths.

Repeat an asset name to request multiple instances:

.. code-block:: python

   InclusionSet(assets=["gear", "gear", "fixture"], match_tags=["size", "assembly"])

This requests two gear instances and one fixture, all sharing a size and assembly. It does not express
separate pairwise constraints. Repeated instances may use the same variant. Assets not referenced by
any inclusion definition remain active independently and do not participate in tag matching. Collection
members retain their separate candidate slots even when they share a scene asset name.

Matching definitions remain compact until clone-plan construction. For example, 1,000 candidates across
five categories and eight asset slots can be represented as five compatible groups, rather than
expanding every valid concrete world. The scene's ``num_envs`` determines how many worlds to instantiate.
Its :attr:`~isaaclab.cloner.CloneCfg.clone_strategy` selects compatible groups according to their weights,
then selects each slot's concrete variant within the chosen group. Identical concrete worlds share a
prototype. The inclusion weight is divided equally across its compatible groups; this gives equal
weight to tag-value groups, rather than to every possible concrete combination. Multi-valued tags can
make groups overlap.

The existing ``sequential`` strategy distributes environments across groups and candidate lists in order.
The ``random`` strategy draws groups by weight and candidates uniformly, using the existing NumPy random
state. ``InclusionSet`` does not own an environment count or random seed. Planning cost depends on
candidate metadata and selected worlds, without expanding the product within each matching group;
loading and simulating concrete assets still has its own cost.

With ``match_tags=[]`` (the default), the existing independent Cartesian product and assignment behavior
are preserved. Inclusion interpretation stays in scene planning. ``ReplicateSession`` forwards the
resolved definitions without applying tag matching or task-specific rules.

A task bank uses one flat scene and :class:`~isaaclab.managers.HeterogeneousTaskCfg` records. Declare
objects once in the scene; each record selects those names and supplies only task-specific data:

.. code-block:: python

   tasks = {
       "place_gear": HeterogeneousTaskCfg(
           inclusion=InclusionSet(assets=["gear", "fixture"]),
           init_states={"gear": RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.1, 0.1))},
           instruction="put the gear in the fixture",
           episode_length_s=30.0,
           terminations={"success": success_term},
       ),
   }
   HeterogeneousTaskManager.compose_tasks(env_cfg, tasks)

The shared scene must already declare ``gear`` and ``fixture`` as rigid objects. Requested semantic
``match_tags`` are validated against those concrete source assets. Initial-state overrides and success
terms use the original asset names. Other scene entities, such as the robot and cameras, stay shared.

The task manager moves selected source assets into one catalog owned by its reset configuration, then
compiles them into the fixed slots required by reset-time mesh switching. Task records contain no scene
or environment configuration. Equal spawn definitions share a candidate; pose differences remain in the
records. The composer derives ``task`` tags on candidate copies and marks the first task's candidates
with ``initial_task``. Its default clone inclusion matches ``initial_task``, retaining one initial task
world while keeping every candidate available for later resets.

The 256 VLA benchmarks use one inline table and one shared environment class in
``contrib/heterogeneous_tasks/heterogeneous_tasks_env_cfg.py``. Individual benchmark IDs select a record
from that table; there are no per-task modules or generated environment subclasses. Instructions,
durations, poses and success predicates remain explicit data, independent of asset compatibility.

Caller-supplied clone inclusions remain authoritative. Matching ``task`` can describe other legal task
families, but does not bind runtime task IDs, poses, instructions, or success terms: custom initial
layouts must still agree with the first task. Selecting several initial task families and binding their
runtime task IDs is a separate capability.

For more detail on prototype assignment and replication, see :doc:`cloning`.

Reset-time selection
~~~~~~~~~~~~~~~~~~~~

A rigid object's multi-asset spawner defines the variants and their IDs in list order. Clone composition chooses
the initial variant for each world, using those same IDs. A reset event can change it before writing the
episode's pose and velocity:

.. code-block:: python

   part_cfg = RigidObjectCfg(
       prim_path="{ENV_REGEX_NS}/Part",
       spawn=sim_utils.MultiUsdFileCfg(usd_path=["small_part.usd", "large_part.usd"], static=False),
   )

Call ``scene["part"].write_mesh_variant_to_sim(variant_ids, env_ids)`` to select variants. A task with matching
fixed and movable objects selects the same ID on both. Newton owns the selected IDs, geometry, and mass properties.

Multi-asset spawning keeps its initial choice unless ``static=False`` is set. Reset-time selection uses Newton
body variants with XPBD or MJWarp with external contacts and sleeping disabled, and at least two alternatives. Every
alternative must contain one rigid body with the same fixed/free root-joint type and kinematic flags. Newton permits
different shape types and counts, but shapes sharing a row need compatible collision groups. Select before
randomizing contact materials because Newton restores the variant's materials. Initial world selections are applied
after solver construction so MuJoCo compiles a shared layout even when alternatives differ in shape type or count.
MJWarp with MuJoCo contacts rejects
body variants when the solver initializes. Other physics backends keep the initial choice.

With supported Newton physics, ``static=False`` authors every alternative under ``/World/IsaacLabPrototypes`` and
clones the chosen source into each world. The source parent is hidden after cloning. The default ``static=True``
authors each alternative at its first assigned world and uses the same source-to-destination clone pass.

Run the example
---------------

The physics backend and visualizer are selected independently. Run one of these commands from the repository root:

.. tab-set::

   .. tab-item:: PhysX with Kit

      .. code-block:: bash

         uv run --extra isaacsim isaaclab example multi-asset \
             --physics isaacsim_physx --visualizer kit --num_envs 2048

   .. tab-item:: Newton MJWarp with Kit

      .. code-block:: bash

         uv run --extra isaacsim isaaclab example multi-asset \
             --physics newton_mjwarp --visualizer kit --num_envs 2048

   .. tab-item:: Newton MJWarp with Newton

      .. code-block:: bash

         uv run isaaclab example multi-asset \
             --physics newton_mjwarp --visualizer newton_gl --num_envs 2048

The Newton commands exercise the same :class:`~assets.RigidObjectCollectionCfg` and ``(env_ids, obj_ids)`` APIs as the
PhysX command. They do not demonstrate per-environment object or articulation variants because of the uniform-layout
restriction described above; use the PhysX command to inspect that part of the example. See the
:doc:`installation guide </source/isaaclab/installation>` before running
the kitless command.

To stop the simulation, you can close the window, or press ``Ctrl+C`` in the terminal.
