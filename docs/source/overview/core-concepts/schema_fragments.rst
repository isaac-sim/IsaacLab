.. _schema-fragments:

Schema Fragments
================

Isaac Lab authors physics properties onto USD prims through *schema fragments*: small
configuration classes that each mirror exactly one USD applied schema and write into a
single attribute namespace. Because fragments compose in lists, one asset configuration
can carry OpenUSD physics (``physics:*``), PhysX (``physx*:*``), and Newton
(``newton:*`` / ``mjc:*``) attributes side by side and run on any backend.

This page explains the fragment model, the prim-path expressions that target fragments
at prims, and the spawner-level configuration surface. For the solver-common vs.
backend-specific class tiers, see :ref:`schema-cfgs`. For the full class and function
reference, see :doc:`/source/api/lab/isaaclab.sim.schemas`.

The fragment model
------------------

Every fragment subclasses :class:`~isaaclab.sim.schemas.SchemaFragment` and declares
which USD namespace its fields write to and which applied schema, if any, it owns. A
fragment's :attr:`~isaaclab.sim.schemas.SchemaFragment.func` names the callable that
applies it to a prim; the default applier
(:func:`~isaaclab.sim.schemas.apply_namespaced`) writes each non-``None`` field as
``<namespace>:<camelCase(field)>`` and leaves ``None`` fields untouched (partial
update). Irregular APIs override ``func`` — for example
:class:`~isaaclab.sim.schemas.UsdPhysicsDriveCfg` dispatches through
:func:`~isaaclab.sim.schemas.apply_drive` to handle the multi-instance
``UsdPhysics.DriveAPI``.

Fragments are grouped into *families*, one per spawner slot. Each family has a writer
that resolves target prims from an expression and dispatches every fragment via its
``func``. Backend fragments carry backend-specific appliers, so the core package never
imports a backend:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Spawner field
     - Family writer
     - Valid targets
   * - ``rigid_props``
     - :func:`~isaaclab.sim.schemas.apply_rigid_body_properties`
     - prims with ``UsdPhysics.RigidBodyAPI``
   * - ``collision_props``
     - :func:`~isaaclab.sim.schemas.apply_collision_properties`
     - prims with ``UsdPhysics.CollisionAPI``
   * - ``mass_props``
     - :func:`~isaaclab.sim.schemas.apply_mass_properties`
     - prims with ``UsdPhysics.MassAPI``
   * - ``articulation_props``
     - :func:`~isaaclab.sim.schemas.apply_articulation_root_properties`
     - prims with ``UsdPhysics.ArticulationRootAPI``
   * - ``joint_drive_props``
     - :func:`~isaaclab.sim.schemas.apply_joint_drive_properties`
     - revolute / prismatic joint prims
   * - ``fixed_tendons_props``
     - :func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties`
     - tendon-bearing prims (existing tendon instances)
   * - ``spatial_tendons_props``
     - :func:`~isaaclab.sim.schemas.apply_spatial_tendon_properties`
     - tendon attachment root / leaf prims

The tendon families are *tune-not-apply*: the tendon topology is authored in the source
asset, so their writers only tune existing instances and never create them.

Targeting expressions
---------------------

Target prims are resolved with :func:`~isaaclab.sim.utils.queries.find_matching_prims`.
Each ``/``-separated token of the expression is a regular expression matched against
prim names at the corresponding depth. A trailing ``**`` token selects the *anchor*
prim (the prim matched by the preceding tokens) itself together with all its
descendants at any depth; it is only valid as the final token. The recursive expansion
traverses into instanceable prims (instance proxies are included) and includes inactive
prims.

The matched set is then filtered to valid family targets (see the table above): API
carriers for the rigid-body, collision, mass, and articulation families; revolute and
prismatic joint prims for the joint-drive family; tendon-bearing prims for the tendon
families. Non-joint matches of a joint-drive expression are ignored silently, since a
``**`` expression legitimately sweeps whole subtrees.

Edge cases behave as follows:

* **Instanced matches** cannot be authored on (prototypes are read-only) and are
  skipped with a warning.
* **Zero targets** emit a warning and the writer returns ``False`` without authoring
  anything.
* **An empty fragment list** is an authoring no-op and returns ``True``.

Configuring fragments on spawners
---------------------------------

Spawner configurations (:class:`~isaaclab.sim.spawners.from_files.UsdFileCfg`,
:class:`~isaaclab.sim.spawners.shapes.CuboidCfg`, ...) expose one field per family.
Each field accepts either a mapping from target pattern to a list of fragments (the
only fragment spelling), a single legacy dataclass cfg (e.g.
:class:`~isaaclab.sim.schemas.RigidBodyBaseCfg` or a backend ``*PropertiesCfg``, routed
to the legacy writers), or ``None``.

Mapping keys are prim-path patterns *relative to the prim the spawner authors that
family on*: the spawn prim for USD, URDF, and MJCF assets; for shape and mesh spawners,
the geometry prim for the collision family and the container prim for the rigid-body
and mass families. The empty string ``""`` selects the anchor prim itself. Entries
apply in insertion order, so when two patterns match the same prim, fragments from
later entries override attributes authored by earlier ones.

A robot spawned from USD, with a broad rule and a narrowing override:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.sim.schemas import UsdPhysicsDriveCfg, UsdPhysicsRigidBodyCfg
   from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
   from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
   from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

   spawn = sim_utils.UsdFileCfg(
       usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
       rigid_props={
           # every rigid body: universal + PhysX + MuJoCo attributes side by side
           "**": [
               UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
               PhysxRigidBodyCfg(max_depenetration_velocity=5.0),
               MujocoRigidBodyCfg(gravcomp=1.0),
           ],
           # hand links (and their subtrees) get a tighter depenetration limit
           ".*_hand/**": [PhysxRigidBodyCfg(max_depenetration_velocity=1.0)],
       },
       joint_drive_props={
           "**": [UsdPhysicsDriveCfg(drive_type="force", stiffness=40.0, damping=4.0)],
       },
   )

A primitive shape, where ``""`` targets the family's anchor prim directly:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
   from isaaclab_newton.sim.schemas import NewtonCollisionCfg

   cuboid = sim_utils.CuboidCfg(
       size=(0.1, 0.1, 0.1),
       rigid_props={"": [UsdPhysicsRigidBodyCfg()]},
       mass_props={"": [MassCfg(mass=0.5)]},
       collision_props={
           "": [UsdPhysicsCollisionCfg(collision_enabled=True), NewtonCollisionCfg(contact_margin=0.001)],
       },
   )

Creating missing APIs
---------------------

By default, the family writers only *modify* prims that already carry the family's
defining USD API. Three per-family spawner flags — ``mass_props_create_if_missing``,
``articulation_props_create_if_missing``, and ``joint_drive_props_create_if_missing`` —
additionally apply the defining API to matched prims that lack it before the fragments
are authored (for the joint-drive family, the axis-appropriate ``UsdPhysics.DriveAPI``
instance). Shape and mesh spawners always create the APIs on the bare prims they
author, since freshly created geometry carries no physics APIs yet.

The writers trust the expression as written: with creation enabled, every matched prim
receives the API, so a too-broad pattern can, for example, give every mesh in a subtree
its own mass. Which bodies participate in an articulation is still decided by the
asset's joints, not by the expression. Scope creation patterns deliberately.

See also
--------

* :ref:`schema-cfgs` — solver-common vs. backend-specific configuration tiers
* :doc:`/source/api/lab/isaaclab.sim.schemas` — fragment base classes and family writers
* :doc:`/source/api/lab_physx/isaaclab_physx.sim.schemas` — PhysX fragments
* :doc:`/source/api/lab_newton/isaaclab_newton.sim.schemas` — Newton / MuJoCo fragments
