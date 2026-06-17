.. _schema-cfgs:

Schema Configuration Classes
============================

Isaac Lab's spawners author USD physics attributes onto prims via a layered set of
configuration classes. The layering separates **universal-physics** parameters
from **backend-specific** parameters, so the same asset cfg can be authored once
and target any backend that supports it.

This page explains the class hierarchy, when to use each tier, and how parameters
route to the underlying USD attributes.

Migrating from 2.x? See :ref:`schemas-cfg-refactor` in the 3.0 migration guide.

Quick example
-------------

Add MuJoCo (MJC) gravity compensation to an articulated asset:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab_newton.sim.schemas import (
       MujocoRigidBodyPropertiesCfg,
       MujocoJointDrivePropertiesCfg,
   )

   spawn = sim_utils.UsdFileCfg(
       usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
       rigid_props=MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
       joint_drive_props=MujocoJointDrivePropertiesCfg(actuatorgravcomp=True),
   )

The Mujoco-specific fields land under ``mjc:*`` on the prim; any
``RigidBodyBaseCfg`` / ``JointDriveBaseCfg`` fields you set on the same instance
land under ``physics:*``. See :ref:`schema-cfgs-mixed` for the full routing rules.

Class hierarchy
---------------

For each property group (rigid body, joint drive, collision, articulation root,
material, mesh collision), Isaac Lab defines a single base class in core
``isaaclab.sim.schemas`` and one subclass per backend in the corresponding
extension package:

.. code-block:: text

   isaaclab.sim.schemas
   ├── RigidBodyBaseCfg
   │   ├── isaaclab_physx.sim.schemas.PhysxRigidBodyPropertiesCfg
   │   └── isaaclab_newton.sim.schemas.NewtonRigidBodyPropertiesCfg
   │       └── isaaclab_newton.sim.schemas.MujocoRigidBodyPropertiesCfg
   │
   ├── JointDriveBaseCfg
   │   ├── isaaclab_physx.sim.schemas.PhysxJointDrivePropertiesCfg
   │   └── isaaclab_newton.sim.schemas.NewtonJointDrivePropertiesCfg
   │       └── isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg
   │
   ├── CollisionBaseCfg
   │   ├── isaaclab_physx.sim.schemas.PhysxCollisionPropertiesCfg
   │   └── isaaclab_newton.sim.schemas.NewtonCollisionPropertiesCfg
   │       ├── isaaclab_newton.sim.schemas.NewtonMeshCollisionPropertiesCfg
   │       └── isaaclab_newton.sim.schemas.NewtonSDFCollisionPropertiesCfg
   │
   ├── ArticulationRootBaseCfg
   │   ├── isaaclab_physx.sim.schemas.PhysxArticulationRootPropertiesCfg
   │   └── isaaclab_newton.sim.schemas.NewtonArticulationRootPropertiesCfg
   │
   ├── MeshCollisionBaseCfg
   │   ├── isaaclab_physx.sim.schemas.{PhysxConvexHull, PhysxConvexDecomposition,
   │   │                                PhysxTriangleMesh, PhysxSDFMesh, ...}PropertiesCfg
   │   └── isaaclab_newton.sim.schemas.NewtonMeshCollisionPropertiesCfg
   │       (also inherits NewtonCollisionPropertiesCfg — multi-namespace)
   │
   └── isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg
       ├── isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg
       └── isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg

:class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionPropertiesCfg` uses
multiple inheritance: it extends both
:class:`~isaaclab_newton.sim.schemas.NewtonCollisionPropertiesCfg` (for
``contact_margin`` / ``contact_gap``) and
:class:`~isaaclab.sim.schemas.MeshCollisionBaseCfg` (for
``mesh_approximation_name``). This is the textbook case for the per-declaring-
class MRO routing described under :ref:`schema-cfgs-mixed` — each inherited
field is written under the namespace of the class that declared it.
:class:`~isaaclab_newton.sim.schemas.NewtonSDFCollisionPropertiesCfg` is
collision-rooted, so it authors ``NewtonSDFCollisionAPI`` without also applying
``UsdPhysics.MeshCollisionAPI`` or setting ``physics:approximation``.

The hierarchy is **single-rooted per spawner slot**: every spawner has a single
field for each property group (``rigid_props``, ``joint_drive_props``,
``collision_props``, etc.), and Python's polymorphism allows any subclass to be
passed where the base type is expected.

When to use which class
-----------------------

The choice depends on which backends you target and which fields you need.

**Use a base class** (``RigidBodyBaseCfg``, ``JointDriveBaseCfg``, etc.)
   when you only need universal-physics fields and you want your asset cfg to be
   backend-portable. Importing the base class does not pull in
   :mod:`isaaclab_physx` or :mod:`isaaclab_newton`.

**Use a PhysX subclass** (``PhysxRigidBodyPropertiesCfg``, etc.)
   when your asset uses PhysX-specific knobs (per-body damping, TGS solver
   iterations, sleep / stabilization thresholds, torsional patch friction,
   compliant-contact materials, etc.) and you target the PhysX backend. Inherits
   all base-class fields, so you can set both universal and PhysX fields on the
   same instance.

**Use a Newton subclass** (``NewtonRigidBodyPropertiesCfg``, etc.)
   when you target Newton and need Newton-native attributes
   (``newton:contactMargin``, ``newton:torsionalFriction``,
   ``newton:selfCollisionEnabled``, etc.). The empty Newton base classes
   (``NewtonRigidBodyPropertiesCfg``, ``NewtonJointDrivePropertiesCfg``) reserve
   the ``newton:*`` namespace for future native fields and act as the parent for
   solver-specific subclasses.

**Use a MuJoCo subclass** (``MujocoRigidBodyPropertiesCfg``, ``MujocoJointDrivePropertiesCfg``)
   when you specifically use Newton's **MuJoCo** solver and need MuJoCo-only
   knobs (gravity compensation via ``mjc:gravcomp`` /
   ``mjc:actuatorgravcomp``). Inherits from the Newton base, so
   ``isinstance(cfg, NewtonRigidBodyPropertiesCfg)`` is True.

What parameters live where
--------------------------

.. note::

   The tables below summarize which fields live on which cfg classes. The
   canonical source is the auto-generated API reference — see
   :doc:`/source/api/lab/isaaclab.sim.schemas`,
   :doc:`/source/api/lab_physx/isaaclab_physx.sim.schemas`, and
   :doc:`/source/api/lab_newton/isaaclab_newton.sim.schemas`, which render
   the cfg class docstrings directly. Treat these tables as a navigation aid;
   if they drift from the source, the API docs win.

Universal physics (declared on the base class)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lives on the **base class**. Most fields write to ``physics:*`` (the standard
``UsdPhysics.*API`` namespace), but a small set of "exception" fields are
declared on the base for backend-portability yet route to a non-``physics:*``
namespace because that is the only USD path honored today (e.g.,
``disable_gravity`` writes ``physxRigidBody:disableGravity`` because both PhysX
and Newton's importer consume the PhysX attribute). The "USD attribute" column
below is the actual emitted attribute, not the namespace family.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Base class
     - Field
     - USD attribute
   * - ``RigidBodyBaseCfg``
     - ``rigid_body_enabled``, ``kinematic_enabled``
     - ``physics:rigidBodyEnabled``, ``physics:kinematicEnabled``
   * - ``RigidBodyBaseCfg``
     - ``disable_gravity``
     - ``physxRigidBody:disableGravity`` (per-body on PhysX; scene-level partial honor on Newton)
   * - ``CollisionBaseCfg``
     - ``collision_enabled``
     - ``physics:collisionEnabled``
   * - ``CollisionBaseCfg``
     - ``contact_offset``, ``rest_offset``
     - ``physxCollision:contactOffset``, ``physxCollision:restOffset`` (Newton consumes via PhysX bridge)
   * - ``ArticulationRootBaseCfg``
     - ``articulation_enabled``
     - ``physxArticulation:articulationEnabled``
   * - ``ArticulationRootBaseCfg``
     - ``fix_root_link``
     - synthesizes ``UsdPhysics.FixedJoint`` (writer-side, not a USD attribute)
   * - ``JointDriveBaseCfg``
     - ``drive_type``, ``max_force``, ``stiffness``, ``damping``
     - ``drive:<axis>:physics:type/maxForce/stiffness/damping``
   * - ``JointDriveBaseCfg``
     - ``max_joint_velocity``
     - ``physxJoint:maxJointVelocity`` (sole USD path; Newton consumes via PhysX bridge today)
   * - ``JointDriveBaseCfg``
     - ``ensure_drives_exist``
     - writer-side only — when ``True``, ensures any drive with ``stiffness=0`` and
       ``damping=0`` gets a minimal ``stiffness=1e-3`` so backends like Newton recognize
       the joint as actively driven; not a USD attribute on its own
   * - ``MassPropertiesCfg``
     - ``mass``, ``density``
     - ``physics:mass``, ``physics:density``
   * - ``RigidBodyMaterialBaseCfg``
     - ``static_friction``, ``dynamic_friction``, ``restitution``
     - ``physics:staticFriction``, ``physics:dynamicFriction``, ``physics:restitution``
   * - ``MeshCollisionBaseCfg``
     - ``mesh_approximation_name``
     - ``physics:approximation``

PhysX-specific (``physx*:*`` namespace, ``Physx*API`` schemas)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lives on the PhysX subclass. Only authored when the user opts in by setting the
field on a PhysX cfg.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - PhysX subclass
     - Fields (selection)
     - USD namespace / schema
   * - ``PhysxRigidBodyPropertiesCfg``
     - ``linear_damping``, ``angular_damping``, ``max_linear_velocity``, ``max_angular_velocity``, ``solver_position_iteration_count``, ``sleep_threshold``, ``enable_gyroscopic_forces``, …
     - ``physxRigidBody:*`` / ``PhysxRigidBodyAPI``
   * - ``PhysxJointDrivePropertiesCfg``
     - (currently empty; reserved for future PhysX-only drive knobs)
     - ``physxJoint:*`` / ``PhysxJointAPI``
   * - ``PhysxCollisionPropertiesCfg``
     - ``torsional_patch_radius``, ``min_torsional_patch_radius``
     - ``physxCollision:*`` / ``PhysxCollisionAPI``
   * - ``PhysxArticulationRootPropertiesCfg``
     - ``enabled_self_collisions``, ``solver_position_iteration_count``, ``sleep_threshold``, ``stabilization_threshold``
     - ``physxArticulation:*`` / ``PhysxArticulationAPI``
   * - ``PhysxRigidBodyMaterialCfg``
     - ``compliant_contact_stiffness``, ``compliant_contact_damping``, ``friction_combine_mode``, ``restitution_combine_mode``
     - ``physxMaterial:*`` / ``PhysxMaterialAPI``
   * - ``PhysxConvexHullPropertiesCfg`` (and other mesh-cooking subclasses)
     - ``hull_vertex_limit``, ``min_thickness``, …
     - ``physxConvexHullCollision:*`` / ``PhysxConvexHullCollisionAPI``

Newton-targeted (``newton:*`` namespace, ``Newton*API`` schemas)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lives on the Newton subclass. Authored only when the user opts in.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Newton subclass
     - Fields
     - USD namespace / schema
   * - ``NewtonRigidBodyPropertiesCfg``
     - (empty — reserved for future Newton-native rigid-body fields)
     - ``newton:*``
   * - ``NewtonJointDrivePropertiesCfg``
     - (empty — reserved for future Newton-native joint-drive fields)
     - ``newton:*``
   * - ``NewtonCollisionPropertiesCfg``
     - ``contact_margin``, ``contact_gap``
     - ``newton:*`` / ``NewtonCollisionAPI``
   * - ``NewtonMeshCollisionPropertiesCfg``
     - ``max_hull_vertices``
     - ``newton:*`` / ``NewtonMeshCollisionAPI``
   * - ``NewtonSDFCollisionPropertiesCfg``
     - ``sdf_max_resolution``, ``sdf_target_voxel_size``, ``sdf_texture_format``, ``hydroelastic_enabled``, ...
     - ``newton:*`` / ``NewtonSDFCollisionAPI``
   * - ``NewtonMaterialPropertiesCfg``
     - ``torsional_friction``, ``rolling_friction``
     - ``newton:*`` / ``NewtonMaterialAPI``
   * - ``NewtonArticulationRootPropertiesCfg``
     - ``self_collision_enabled``
     - ``newton:*`` / ``NewtonArticulationRootAPI``

MuJoCo-solver-specific (``mjc:*`` namespace)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lives on a MuJoCo subclass that extends a Newton subclass. Only consumed when
running Newton's MuJoCo solver.

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - MuJoCo subclass
     - Field
     - USD attribute / schema
   * - ``MujocoRigidBodyPropertiesCfg``
     - ``gravcomp``
     - ``mjc:gravcomp`` (raw attribute, no applied schema)
   * - ``MujocoJointDrivePropertiesCfg``
     - ``actuatorgravcomp``
     - ``mjc:actuatorgravcomp`` via ``MjcJointAPI``

.. note::

   The two MuJoCo rows differ in their USD applied-schema requirement:
   ``mjc:actuatorgravcomp`` is part of the registered ``MjcJointAPI`` applied
   schema (so the writer calls ``prim.AddAppliedSchema("MjcJointAPI")`` when
   the field is non-None). ``mjc:gravcomp`` has no registered Mjc applied
   schema for body-level gravity compensation, so the writer authors it as a
   raw USD attribute. Newton's MuJoCo solver consumes both via the same
   resolver path; the schema-application difference is purely a USD-side
   detail.

.. _schema-cfgs-mixed:

Mixed-namespace authoring on a single instance
----------------------------------------------

Because each cfg field is routed to its **declaring class's** namespace (not
the instance's class), a subclass instance can author attributes across multiple
namespaces on the same prim. For example:

.. code-block:: python

   from isaaclab_newton.sim.schemas import MujocoRigidBodyPropertiesCfg

   cfg = MujocoRigidBodyPropertiesCfg(
       rigid_body_enabled=True,        # declared on RigidBodyBaseCfg → physics:rigidBodyEnabled
       disable_gravity=True,            # declared on RigidBodyBaseCfg (exception) → physxRigidBody:disableGravity
       gravcomp=1.0,                    # declared on MujocoRigidBodyPropertiesCfg → mjc:gravcomp
   )

The writer applies each field to the namespace of the class where the field is
declared. The applied schemas (``PhysxRigidBodyAPI`` for ``disable_gravity``,
none for the Mjc raw attribute) are added only when the corresponding
fields are non-None.

Spawner usage
-------------

Spawners (``UsdFileCfg``, ``MeshCuboidCfg``, ``MeshSphereCfg``, …) accept
the base class type for each slot and use polymorphism to dispatch to the
correct subclass at write time:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
   from isaaclab_newton.sim.schemas import MujocoJointDrivePropertiesCfg

   spawn = sim_utils.UsdFileCfg(
       usd_path="...",
       rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=True, linear_damping=0.1),
       joint_drive_props=MujocoJointDrivePropertiesCfg(
           drive_type="acceleration",
           stiffness=10.0,
           damping=0.1,
           actuatorgravcomp=True,
       ),
   )

.. _schema-cfgs-gravcomp:

Gravity compensation (MuJoCo solver)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gravity compensation has two halves and you typically need both:

* **Body-level**:
  :attr:`~isaaclab_newton.sim.schemas.MujocoRigidBodyPropertiesCfg.gravcomp`
  on each rigid body (writes ``mjc:gravcomp``). This is what *computes* the
  compensation force.
* **Joint-level**:
  :attr:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg.actuatorgravcomp`
  on each joint (writes ``mjc:actuatorgravcomp``). This routes the compensation
  force through the actuator channel (``qfrc_actuator``) so it counts against
  ``actuatorfrcrange``; otherwise it goes to ``qfrc_passive``.

``actuatorgravcomp=True`` alone is a no-op — without body-level ``gravcomp``
there are no forces to route. To prevent this footgun, the spawner
**auto-enables** ``MujocoRigidBodyPropertiesCfg(gravcomp=1.0)`` whenever
``joint_drive_props`` is a Mujoco cfg with ``actuatorgravcomp=True`` and
``rigid_props`` is not already a Mujoco cfg. If you want a different
``gravcomp`` value (or want to disable the auto-enable), pass an explicit
``MujocoRigidBodyPropertiesCfg`` in ``rigid_props``.

Naming convention
-----------------

Cfg field names use ``snake_case``; the writer converts them to ``camelCase``
USD attribute names (``contact_margin`` → ``newton:contactMargin``). For
single-token fields (``gravcomp``, ``actuatorgravcomp``), the conversion is
identity, which matches MuJoCo's lowercase convention.

Field renames preserve backward compatibility via deprecation aliases. Two such
aliases live on ``JointDriveBaseCfg`` today:

* ``max_velocity`` → ``max_joint_velocity`` (USD attribute is ``physxJoint:maxJointVelocity``)
* ``max_effort`` → ``max_force`` (USD attribute is ``drive:<axis>:physics:maxForce``)

The old names remain as real dataclass fields (so ``dataclasses.fields()``
sees them), defaulting to ``None``. ``__post_init__`` runs
``_deprecate_field_alias`` which, when the old field is set: emits a
``DeprecationWarning``, copies the value into the canonical field if the
canonical is ``None``, then nulls the old field. Setting **both** in the same
constructor is silent — the canonical wins; the old name's value is discarded.
Both aliases are scheduled for removal in 4.0.

See also
--------

* :doc:`/source/migration/migrating_to_isaaclab_3-0` — migration guide
* :doc:`/source/api/lab/isaaclab.sim.schemas` — solver-common base class API
* :doc:`/source/api/lab_physx/isaaclab_physx.sim.schemas` — PhysX subclass API
* :doc:`/source/api/lab_newton/isaaclab_newton.sim.schemas` — Newton/MuJoCo subclass API
