.. _schema-cfgs:

Physics Schemas
===============

A robot model needs more than geometry to simulate. Its links need mass, its shapes
need collision properties, and its joints need physical constraints. USD stores
these properties in **schemas**: named groups of attributes attached to objects in
the scene. An object in a USD scene is called a **prim**; a robot link, collision
mesh, or joint can each be a prim.

Isaac Lab's schema configurations let you set these properties from Python when an
asset is spawned. Use them to tune an existing asset or give a new shape its physical
properties, without editing the source USD file by hand. An imported asset may
already contain everything you need; override only the properties you want to change.

These are asset settings. Choose the simulation backend and its scene-wide solver
settings separately through :doc:`backends_and_presets`.


What is a fragment?
-------------------

A **schema fragment** is a small Python configuration object that sets properties
from one USD schema or attribute namespace. For example, ``MassCfg(mass=0.5)``
sets a mass property, while ``UsdPhysicsCollisionCfg(collision_enabled=True)``
sets a collision property.

It is called a fragment because it describes just one part of an asset's physics
configuration. Combine fragments to describe the properties you need, then pass
them to the appropriate spawner fields. A fragment is configuration data, not a
separate USD file or a piece of geometry.

The distinction is:

* The **schema** defines the available properties and how they are stored in USD.
* The **fragment** supplies the values you want to set from Python.
* The **spawner** applies those values to the asset's prims when it creates the asset.


Give an object physical properties
----------------------------------

This configuration creates a small box with a mass of 0.5 kg and enabled collisions:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg

   cuboid = sim_utils.CuboidCfg(
       size=(0.1, 0.1, 0.1),
       rigid_props=UsdPhysicsRigidBodyCfg(),
       mass_props=MassCfg(mass=0.5),
       collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
   )

Pass this configuration to an asset's ``spawn`` field, such as
``RigidObjectCfg(spawn=cuboid, ...)``. The spawner creates the geometry and applies
the schema configurations; the physics backend then reads the resulting properties.

Each spawner field controls a different part of the asset:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - What it controls
   * - ``rigid_props``
     - Rigid-body behavior, such as whether a body is kinematic.
   * - ``mesh_collision_props``
     - Mesh approximation and backend cooking settings for existing colliders.
   * - ``mass_props``
     - Mass or density for rigid bodies.
   * - ``collision_props``
     - Collision behavior and contact settings.
   * - ``articulation_props``
     - Properties of a connected robot, such as its fixed root.
   * - ``joint_drive_props``
     - Joint drive settings stored on the asset.
   * - ``physics_material``
     - Surface properties such as friction and restitution.

For joints controlled by Isaac Lab actuators, also read :doc:`actuators`: actuator
configuration can override drive settings after spawning.

A field set to ``None`` leaves the existing attribute unchanged. An empty fragment,
such as ``UsdPhysicsRigidBodyCfg()``, still requests the corresponding schema; it
does not replace every attribute with a Python default.


.. _schema-fragments:

Add backend-specific settings
-----------------------------

A spawner field can accept a list of fragments for the same property family. Use
a list to combine shared USD properties with settings for a particular backend:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.sim.schemas import MassCfg, UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
   from isaaclab_newton.sim.schemas import NewtonCollisionCfg

   cuboid = sim_utils.CuboidCfg(
       size=(0.1, 0.1, 0.1),
       rigid_props=UsdPhysicsRigidBodyCfg(),
       mass_props=MassCfg(mass=0.5),
       collision_props=[UsdPhysicsCollisionCfg(collision_enabled=True), NewtonCollisionCfg(contact_margin=0.001)],
   )

Here, ``UsdPhysicsCollisionCfg`` enables collisions and ``NewtonCollisionCfg`` sets
a Newton contact margin of 1 mm. They describe the same collision shape, so they
belong together in ``collision_props``.

Import shared fragments from ``isaaclab.sim.schemas``, PhysX fragments from
``isaaclab_physx.sim.schemas``, and Newton or MuJoCo fragments from
``isaaclab_newton.sim.schemas``. Material fragments live in the corresponding
``sim.spawners.materials`` modules.

You can store settings for several backends on one asset. This does not activate
those backends or make their parameters interchangeable: a solver must support a
property for it to affect the simulation. For example, MuJoCo gravity compensation
settings apply to Newton's MuJoCo solver, not to PhysX.


Configure an existing robot
---------------------------

``UsdFileCfg`` accepts the same property fields. Pass a fragment or list directly
to update existing matching bodies, collisions, or joints throughout the imported
asset. Use a dictionary when different parts need different settings:

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
           "/.*": [
               UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
               PhysxRigidBodyCfg(max_depenetration_velocity=5.0),
               MujocoRigidBodyCfg(gravcomp=1.0),
           ],
           # hand links (and their subtrees) get a tighter depenetration limit
           "/.*_hand(/.*)?": [PhysxRigidBodyCfg(max_depenetration_velocity=1.0)],
       },
       joint_drive_props={
           "/.*": [UsdPhysicsDriveCfg(drive_type="force", stiffness=40.0, damping=4.0)],
       },
   )

Dictionary keys are regular-expression suffixes appended to the spawn prim's path.
For example, if the robot is spawned at ``/World/Robot``, ``"/.*"`` searches beneath
``/World/Robot``. Rules apply in order, so a later rule can override an earlier one.
In this example, the hand subtree receives a smaller PhysX depenetration limit.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Pattern
     - Selection relative to the anchor
   * - ``""``
     - The anchor itself.
   * - ``"/[^/]+"``
     - Direct children only.
   * - ``"/.*"``
     - All descendants, at any depth.
   * - ``"(/.*)?"``
     - The anchor and all descendants.

For USD, URDF, and MJCF imports, the anchor is the spawn prim. Shape and mesh spawners
use the geometry prim for collision and mesh-collision properties, and the container
prim for rigid-body and mass
properties. Deformable settings use the spawn prim for every spawner type.

The search includes inactive and undefined prims as well as instance proxies.
Patterns select candidate prims; the property family then filters them. For example,
rigid-body settings apply to rigid bodies, and joint-drive settings apply to revolute
or prismatic joints. A broad pattern does not normally turn every mesh into a body.


When a setting does not apply
-----------------------------

First check that the selected backend supports the property and that the pattern
matches the intended prim. Matching uses the whole path: ``.*`` can cross ``/``
boundaries, while ``[^/]+`` matches just one path segment.

Most fragment writers modify prims that already have the relevant USD API, meaning
the schema that identifies them as a body, collision shape, or other physics object.
For mass, articulation, and joint-drive settings, the spawner flags
``mass_props_create_if_missing``, ``articulation_props_create_if_missing``, and
``joint_drive_props_create_if_missing`` can create missing APIs on matched targets.

.. hint::

   Keep patterns narrow when enabling API creation. A broad mass pattern can give
   each matching mesh its own mass, rather than setting one mass for the body.

Shape and mesh spawners create the APIs needed by their new geometry. With a bare
fragment or list, file spawners search for existing schema carriers; if none exist
for the family, they apply it at the spawn prim, treating the asset as a single body.
Mesh-collision fragments are an exception: they only modify existing colliders.
Explicit dictionaries give you control over the target selection.

The writers warn when no valid targets are found or when a target is an instance
proxy that cannot be edited. An empty fragment list makes no changes. Joint-drive
patterns silently skip non-joint prims, since a subtree can contain both joints and
meshes.


.. _schema-cfgs-gravcomp:

Gravity compensation with MuJoCo
--------------------------------

Set body-level ``gravcomp`` to compute gravity compensation. Set joint-level
``actuatorgravcomp`` to route that force through the actuator channel, where actuator
force limits apply; otherwise it goes through the passive-force channel.

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
   from isaaclab_newton.sim.schemas import MujocoJointCfg, MujocoRigidBodyCfg

   spawn = sim_utils.UsdFileCfg(
       usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
       rigid_props=MujocoRigidBodyCfg(gravcomp=1.0),
       joint_drive_props=MujocoJointCfg(actuatorgravcomp=True),
   )

``gravcomp=1.0`` provides full compensation. The joint fragment also enables full
compensation on its child body if that body's value is unset or zero; an existing
nonzero value is kept. To disable compensation, turn off ``actuatorgravcomp`` as
well as setting the body's ``gravcomp`` to zero.

The older ``MujocoJointDrivePropertiesCfg`` also auto-enables body compensation
when requested, unless an explicit body ``gravcomp`` value is supplied. Prefer the
fragment example above for new configurations.


Mesh collision settings
-----------------------

Mesh-collision fragments use the same anchor as collision fragments and only modify
existing colliders; they never create colliders. For example, to cook every
collider of a USD asset as a convex hull with a PhysX vertex limit:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab_physx.sim.schemas import PhysxConvexHullCfg

   spawn = sim_utils.UsdFileCfg(
       usd_path="/path/to/asset.usd",
       mesh_collision_props=[
           sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
           PhysxConvexHullCfg(hull_vertex_limit=32),
       ],
   )


.. _schema-fragments-deformable:

Deformable bodies
-----------------

Choose ``volume_deformable_props`` for a solid deformable or
``surface_deformable_props`` for a cloth-like surface. Set only one of these fields
on a spawner. Both accept ``OmniPhysicsDeformableBodyCfg`` for shared properties such
as mass and kinematic behavior; PhysX also provides tuning fragments.

Unlike most rigid-body writers, deformable writers create the deformable setup if
it is missing. Set deformable mass with ``OmniPhysicsDeformableBodyCfg.mass``;
``UsdPhysics.MassAPI`` does not control it.

This example creates a volume deformable with PhysX tuning, a collision offset on
its simulation mesh, and material settings for both PhysX and Newton:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyCfg
   from isaaclab.sim.spawners.materials import OmniPhysicsDeformableMaterialCfg
   from isaaclab_newton.sim.spawners.materials import NewtonVolumeDeformableMaterialCfg

   sim_utils.MeshCuboidCfg(
       size=(0.2, 0.2, 0.2),
       volume_deformable_props=[
           sim_utils.schemas.OmniPhysicsDeformableBodyCfg(mass=0.5),
           PhysxDeformableBodyCfg(solver_position_iteration_count=32),
       ],
       collision_props={"/sim_mesh": [PhysxCollisionCfg(contact_offset=0.01)]},
       physics_material=[
           OmniPhysicsDeformableMaterialCfg(youngs_modulus=1.0e6, poissons_ratio=0.45),
           NewtonVolumeDeformableMaterialCfg(k_mu=1.0e5),
       ],
   )

The simulation mesh is a child named ``sim_mesh`` by default. Target it through
``collision_props`` to set collision offsets. Deformable and simulation-mesh collision
patterns are relative to the spawn prim.

``physics_material`` accepts deformable material fragments and lists that mix rigid
and deformable material settings. Each fragment applies its own material schema;
a material used only by deformables does not need the rigid-body material API.


Using the property configuration classes
----------------------------------------

You will also see classes such as ``RigidBodyBaseCfg`` and
``PhysxRigidBodyPropertiesCfg`` in existing tasks. These combine a property group's
shared and backend-specific fields in one object. These compatibility classes are
deprecated and scheduled for removal in 3.2. Existing configurations still work in
the same spawner fields, but use
fragments for new code. Pass a compatibility object directly, rather than placing
it in a fragment list.

* Base classes in ``isaaclab.sim.schemas`` contain shared fields and do not import a
  backend package.
* PhysX subclasses add fields such as damping and solver iterations.
* Newton subclasses add Newton settings; MuJoCo subclasses add solver-specific fields
  such as gravity compensation.

When updating an existing task, split its compatibility object into fragments for
the shared and backend-specific fields. See :ref:`the fragment migration examples <schema-fragments-migration>` for
examples.

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

This example writes both PhysX and MuJoCo settings into the asset. Each backend uses
the fields it supports. For a full list of classes and fields, expand the reference
sections below.


Schema reference and implementation
-----------------------------------

The sections below explain how configuration fields become USD attributes. You do
not need these details for the examples above.

.. dropdown:: Property configuration class hierarchy

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
    ``mesh_approximation_name``). Each inherited field is written to the namespace of its declaring class;
    see :ref:`field-to-namespace mapping <schema-cfgs-mixed>`.
    :class:`~isaaclab_newton.sim.schemas.NewtonSDFCollisionPropertiesCfg` is
    collision-rooted, so it authors ``NewtonSDFCollisionAPI`` without also applying
    ``UsdPhysics.MeshCollisionAPI`` or setting ``physics:approximation``.

    Each spawner has one field per property group, such as ``rigid_props`` or
    ``joint_drive_props``. Each field accepts the base class or any of its subclasses.

.. dropdown:: Shared and backend-specific field reference

    .. note::

       These tables summarize common fields. For the full field lists and defaults, see
       :doc:`/source/api/lab/isaaclab.sim.schemas`,
       :doc:`/source/api/lab_physx/isaaclab_physx.sim.schemas`, and
       :doc:`/source/api/lab_newton/isaaclab_newton.sim.schemas`, which are generated
       from the configuration classes.

    .. rubric:: Universal physics (declared on the base class)

    Most shared fields write to ``physics:*`` (the standard
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

    .. rubric:: PhysX-specific (``physx*:*`` namespace, ``Physx*API`` schemas)

    These fields are written when set on a PhysX configuration.

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

    .. rubric:: Newton-targeted (``newton:*`` namespace, ``Newton*API`` schemas)

    These fields are written when set on a Newton configuration.

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

    .. rubric:: MuJoCo-solver-specific (``mjc:*`` namespace)

    These fields are consumed by Newton's MuJoCo solver.

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

.. dropdown:: Fragment writers and target families

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

    Fragments are grouped into *families*, one per spawner slot. Most families have a writer
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
       * - ``mesh_collision_props``
         - :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`, once per collider
         - existing collision prims
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
         - tendon attachment root prims
       * - ``volume_deformable_props``
         - :func:`~isaaclab.sim.schemas.apply_volume_deformable_properties`
         - volume deformable-body prims (``UsdGeom.TetMesh`` simulation mesh)
       * - ``surface_deformable_props``
         - :func:`~isaaclab.sim.schemas.apply_surface_deformable_properties`
         - surface deformable-body prims (triangle-mesh simulation mesh)

    The tendon families are *tune-not-apply*: the tendon topology is authored in the source
    asset, so their writers only tune existing instances and never create them. The deformable
    families sit at the other end of that spectrum: see :ref:`schema-fragments-deformable`.
    PhysX tendon
    fragments use ``instance_names`` to select one or more instances on each matching prim;
    ``None`` (the default) broadcasts to all existing instances.

    A fixed tendon is split at the same boundary as the PhysX schemas: the root fragment owns
    whole-tendon dynamics and limits, while the axis fragment owns each joint's contribution.
    Prim-path matching chooses the joints and ``instance_names`` chooses the tendon on those
    joints:

    .. code-block:: python

       from isaaclab_physx.sim.schemas import PhysxTendonAxisCfg, PhysxTendonAxisRootCfg

       fixed_tendons_props = {
           "/joints/index_root": [
               PhysxTendonAxisRootCfg(instance_names="index_finger", stiffness=10.0),
               PhysxTendonAxisCfg(instance_names="index_finger", gearing=[1.0], joint_axis=["rotX"]),
           ],
           "/joints/index_distal": [
               PhysxTendonAxisCfg(instance_names="index_finger", gearing=[-0.5], joint_axis=["rotX"]),
           ],
       }

.. dropdown:: How the deformable setup is created

    Creating the deformable setup on a matched prim without the anchor happens in three steps:

    #. **Simulation mesh.** A ``sim_mesh`` child prim is authored under the target (the name is
       configurable via the writer's ``sim_mesh_name`` argument). For surface deformables this is
       a triangle-mesh copy of the visual mesh; for volume deformables it is a tetrahedralized
       ``UsdGeom.TetMesh`` — reused in place, without re-tetrahedralizing, when the source asset
       already ships a pre-tetrahedralized ``UsdGeom.TetMesh`` child.
    #. **Backend anchor.** The active physics backend applies its own anchor schemas through
       ``PhysicsManager.setup_deformable_body``: PhysX applies the real ``OmniPhysicsDeformableBodyAPI``
       to the body prim (the same single schema for both deformable types) and the type-dependent
       ``OmniPhysicsVolumeDeformableSimAPI`` / ``OmniPhysicsSurfaceDeformableSimAPI`` to the
       simulation-mesh prim, while Newton applies its own (currently token-only) ``Physics*`` schemas.
       Core fragments own no applied schema of their own —
       :class:`~isaaclab.sim.schemas.OmniPhysicsDeformableBodyCfg` only writes into the
       ``omniphysics:*`` namespace once the anchor exists.
    #. **Fragment dispatch.** Every fragment in the matched slot is applied to the target, in list
       order, exactly like the other families.

.. _schema-cfgs-mixed:

.. dropdown:: How fields map to USD namespaces

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

.. dropdown:: Field names and compatibility aliases

    Python fields use ``snake_case``; USD attributes generally use ``camelCase``
    (for example, ``contact_margin`` becomes ``newton:contactMargin``). Single-word
    MuJoCo fields such as ``gravcomp`` keep their spelling.

    ``JointDriveBaseCfg`` still accepts ``max_velocity`` as an alias for
    ``max_joint_velocity`` and ``max_effort`` as an alias for ``max_force``. Prefer the
    new names: the aliases are scheduled for removal in 4.0. Using an old name emits a
    deprecation warning and copies its value if the new field is unset. If both are
    provided, the new field wins without a warning. The old dataclass fields are then
    reset to ``None``.

.. raw:: html

   <span id="see-also"></span>

Where to go next
----------------

* :doc:`/source/how-to/write_articulation_cfg` — configure a complete robot asset.
* :doc:`actuators` — understand how actuators resolve joint drive properties.
* :ref:`schemas-cfg-refactor` — migrate older schema configurations to 3.0.
* :doc:`/source/api/lab/isaaclab.sim.schemas` — shared configurations and writers.
* :doc:`/source/api/lab_physx/isaaclab_physx.sim.schemas` — PhysX configurations.
* :doc:`/source/api/lab_newton/isaaclab_newton.sim.schemas` — Newton and MuJoCo configurations.
