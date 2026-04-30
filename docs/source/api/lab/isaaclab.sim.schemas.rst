isaaclab.sim.schemas
====================

.. automodule:: isaaclab.sim.schemas

  .. rubric:: Classes

  .. autosummary::

    ArticulationRootPropertiesCfg
    RigidBodyPropertiesCfg
    CollisionPropertiesCfg
    MassPropertiesCfg
    JointDrivePropertiesCfg
    FixedTendonPropertiesCfg

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
    modify_fixed_tendon_properties

Articulation Root
-----------------

.. autoclass:: ArticulationRootPropertiesCfg
    :members:
    :exclude-members: __init__

.. autofunction:: define_articulation_root_properties
.. autofunction:: modify_articulation_root_properties

Rigid Body
----------

.. autoclass:: RigidBodyPropertiesCfg
    :members:
    :exclude-members: __init__

.. autofunction:: define_rigid_body_properties
.. autofunction:: modify_rigid_body_properties
.. autofunction:: activate_contact_sensors

Collision
---------

.. autoclass:: CollisionPropertiesCfg
    :members:
    :exclude-members: __init__

.. autofunction:: define_collision_properties
.. autofunction:: modify_collision_properties

Mass
----

.. autoclass:: MassPropertiesCfg
    :members:
    :exclude-members: __init__

.. autofunction:: define_mass_properties
.. autofunction:: modify_mass_properties

Joint Drive
-----------

.. autoclass:: JointDrivePropertiesCfg
    :members:
    :exclude-members: __init__

.. autofunction:: modify_joint_drive_properties

Fixed Tendon
------------

.. autoclass:: FixedTendonPropertiesCfg
    :members:
    :exclude-members: __init__

.. autofunction:: modify_fixed_tendon_properties

Deformable Body
---------------

.. note::

   Deformable body schemas have moved to the PhysX backend extension. See
   :class:`isaaclab_physx.sim.schemas.DeformableBodyPropertiesCfg`,
   :func:`isaaclab_physx.sim.schemas.define_deformable_body_properties`, and
   :func:`isaaclab_physx.sim.schemas.modify_deformable_body_properties`.

   For migration details, see :ref:`migrating-deformables`.
