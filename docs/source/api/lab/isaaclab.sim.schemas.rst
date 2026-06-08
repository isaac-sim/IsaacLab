isaaclab.sim.schemas
====================

.. automodule:: isaaclab.sim.schemas

  .. rubric:: Solver-common base classes

  These base classes carry the universal-physics fields that every backend honors.
  They live in core ``isaaclab`` and have no backend dependency. For backend-specific
  knobs, use the matching subclass in :mod:`isaaclab_physx.sim.schemas` or
  :mod:`isaaclab_newton.sim.schemas`. See :doc:`/source/overview/core-concepts/schema_cfgs`
  for the full design.

  .. autosummary::

    ArticulationRootBaseCfg
    RigidBodyBaseCfg
    CollisionBaseCfg
    JointDriveBaseCfg
    MeshCollisionBaseCfg
    MassPropertiesCfg
    JointDrivePropertiesCfg
    FixedTendonPropertiesCfg
    DeformableBodyPropertiesBaseCfg

  .. rubric:: Mesh collision approximations (USD-only, no PhysX schema)

  .. autosummary::

    BoundingCubePropertiesCfg
    BoundingSpherePropertiesCfg

  .. rubric:: Functions

  .. autosummary::

    define_articulation_root_properties
    modify_articulation_root_properties
    define_rigid_body_properties
    modify_rigid_body_properties
    activate_contact_sensors
    define_collision_properties
    modify_collision_properties
    define_mass_properties
    modify_mass_properties
    modify_joint_drive_properties
    define_mesh_collision_properties
    modify_mesh_collision_properties
    modify_fixed_tendon_properties
    define_deformable_body_properties
    modify_deformable_body_properties

Articulation Root
-----------------

.. autoclass:: ArticulationRootBaseCfg
    :members:
    :exclude-members: __init__

.. autofunction:: define_articulation_root_properties
.. autofunction:: modify_articulation_root_properties

For PhysX-specific articulation properties (self-collisions, TGS solver iterations,
sleep/stabilization thresholds), see
:class:`~isaaclab_physx.sim.schemas.PhysxArticulationRootPropertiesCfg`. For
Newton-native self-collisions, see
:class:`~isaaclab_newton.sim.schemas.NewtonArticulationRootPropertiesCfg`.

Rigid Body
----------

.. autoclass:: RigidBodyBaseCfg
    :members:
    :exclude-members: __init__

.. autofunction:: define_rigid_body_properties
.. autofunction:: modify_rigid_body_properties
.. autofunction:: activate_contact_sensors

For PhysX-specific rigid body properties (damping, max velocities, solver iterations,
sleep/stabilization), see :class:`~isaaclab_physx.sim.schemas.PhysxRigidBodyPropertiesCfg`.
For MuJoCo-specific gravity compensation, see
:class:`~isaaclab_newton.sim.schemas.MujocoRigidBodyPropertiesCfg`.

Collision
---------

.. autoclass:: CollisionBaseCfg
    :members:
    :exclude-members: __init__

.. autofunction:: define_collision_properties
.. autofunction:: modify_collision_properties

For PhysX torsional patch friction, see
:class:`~isaaclab_physx.sim.schemas.PhysxCollisionPropertiesCfg`. For Newton-native
contact margin/gap, see
:class:`~isaaclab_newton.sim.schemas.NewtonCollisionPropertiesCfg`.

Mass
----

.. autoclass:: MassPropertiesCfg
    :members:
    :exclude-members: __init__

.. autofunction:: define_mass_properties
.. autofunction:: modify_mass_properties

Joint Drive
-----------

.. autoclass:: JointDriveBaseCfg
    :members:
    :exclude-members: __init__

.. autofunction:: modify_joint_drive_properties

For PhysX-specific drive properties, see
:class:`~isaaclab_physx.sim.schemas.PhysxJointDrivePropertiesCfg`. For MuJoCo
actuator gravity compensation, see
:class:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg`.

Mesh Collision
--------------

.. autoclass:: MeshCollisionBaseCfg
    :members:
    :exclude-members: __init__

.. autoclass:: BoundingCubePropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: BoundingSpherePropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autofunction:: define_mesh_collision_properties
.. autofunction:: modify_mesh_collision_properties

For PhysX cooking schemas (convex hull / decomposition / triangle mesh / SDF),
see the ``Physx*PropertiesCfg`` family in :mod:`isaaclab_physx.sim.schemas`.
For Newton hull-vertex limit, see
:class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionPropertiesCfg`.

Tendon
------

.. autofunction:: modify_fixed_tendon_properties
.. autofunction:: modify_spatial_tendon_properties

Tendon cfg classes are PhysX-only and live in
:mod:`isaaclab_physx.sim.schemas`
(:class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonPropertiesCfg`,
:class:`~isaaclab_physx.sim.schemas.PhysxSpatialTendonPropertiesCfg`).

Deformable Body
---------------

.. autoclass:: DeformableBodyPropertiesBaseCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autofunction:: define_deformable_body_properties
.. autofunction:: modify_deformable_body_properties
