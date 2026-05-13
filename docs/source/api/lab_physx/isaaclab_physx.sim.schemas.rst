isaaclab_physx.sim.schemas
==========================

.. automodule:: isaaclab_physx.sim.schemas

  PhysX-specific schema configuration classes. Each cfg below extends a
  solver-common base in :mod:`isaaclab.sim.schemas` with PhysX-namespaced
  attributes (``physx*:*``) and applies the corresponding ``Physx*API``
  applied schema. See :doc:`/source/overview/core-concepts/schema_cfgs`
  for the design.

  .. rubric:: Rigid body and joint drive

  .. autosummary::

    PhysxRigidBodyPropertiesCfg
    PhysxJointDrivePropertiesCfg

  .. rubric:: Collision

  .. autosummary::

    PhysxCollisionPropertiesCfg

  .. rubric:: Articulation root

  .. autosummary::

    PhysxArticulationRootPropertiesCfg

  .. rubric:: Mesh collision (PhysX cooking)

  .. autosummary::

    PhysxConvexHullPropertiesCfg
    PhysxConvexDecompositionPropertiesCfg
    PhysxTriangleMeshPropertiesCfg
    PhysxTriangleMeshSimplificationPropertiesCfg
    PhysxSDFMeshPropertiesCfg

  .. rubric:: Tendon

  .. autosummary::

    PhysxFixedTendonPropertiesCfg
    PhysxSpatialTendonPropertiesCfg

  .. rubric:: Deformable body

  .. autosummary::

    DeformableBodyPropertiesCfg

  .. rubric:: Functions

  .. autosummary::

    define_deformable_body_properties
    modify_deformable_body_properties

.. currentmodule:: isaaclab_physx.sim.schemas

Rigid Body
----------

.. autoclass:: PhysxRigidBodyPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Joint Drive
-----------

.. autoclass:: PhysxJointDrivePropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Collision
---------

.. autoclass:: PhysxCollisionPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Articulation Root
-----------------

.. autoclass:: PhysxArticulationRootPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Mesh Collision (PhysX cooking)
-------------------------------

.. autoclass:: PhysxConvexHullPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxConvexDecompositionPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxTriangleMeshPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxTriangleMeshSimplificationPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxSDFMeshPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Tendon
------

.. autoclass:: PhysxFixedTendonPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: PhysxSpatialTendonPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Deformable Body
---------------

.. autoclass:: DeformableBodyPropertiesCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autofunction:: define_deformable_body_properties
.. autofunction:: modify_deformable_body_properties
