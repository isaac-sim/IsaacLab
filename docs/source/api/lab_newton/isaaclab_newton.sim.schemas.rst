isaaclab_newton.sim.schemas
===========================

.. automodule:: isaaclab_newton.sim.schemas

  Newton-targeted schema configuration classes. Each cfg below extends a
  solver-common base in :mod:`isaaclab.sim.schemas` with Newton-namespaced
  attributes (``newton:*``) or solver-specific attributes (``mjc:*`` for
  Newton's MuJoCo solver). MuJoCo cfgs subclass their Newton counterpart
  because MuJoCo is one of Newton's solver options.

  See :doc:`/source/overview/core-concepts/schema_cfgs` for the design and
  when to use each class.

  .. rubric:: Newton-targeted (family roots)

  .. autosummary::

    NewtonRigidBodyPropertiesCfg
    NewtonJointDrivePropertiesCfg
    NewtonCollisionPropertiesCfg
    NewtonMeshCollisionPropertiesCfg
    NewtonMaterialPropertiesCfg
    NewtonArticulationRootPropertiesCfg

  .. rubric:: MuJoCo-solver-specific

  .. autosummary::

    MujocoRigidBodyPropertiesCfg
    MujocoJointDrivePropertiesCfg

.. currentmodule:: isaaclab_newton.sim.schemas

Rigid Body
----------

.. autoclass:: NewtonRigidBodyPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: MujocoRigidBodyPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Joint Drive
-----------

.. autoclass:: NewtonJointDrivePropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: MujocoJointDrivePropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Collision
---------

.. autoclass:: NewtonCollisionPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: NewtonMeshCollisionPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Material
--------

.. autoclass:: NewtonMaterialPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Articulation Root
-----------------

.. autoclass:: NewtonArticulationRootPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__
