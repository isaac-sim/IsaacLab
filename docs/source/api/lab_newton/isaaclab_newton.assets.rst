isaaclab\_newton.assets
=======================

.. automodule:: isaaclab_newton.assets

  .. rubric:: Classes

  .. autosummary::

    Articulation
    ArticulationData
    RigidObject
    RigidObjectData
    RigidObjectCollection
    RigidObjectCollectionData
    DeformableObject
    DeformableObjectData
    MPMObject
    MPMObjectCfg
    MPMObjectData

.. currentmodule:: isaaclab_newton.assets

Articulation
------------

.. autoclass:: Articulation
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: ArticulationData
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__

Rigid Object
------------

.. autoclass:: RigidObject
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: RigidObjectData
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__

Rigid Object Collection
-----------------------

.. autoclass:: RigidObjectCollection
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: RigidObjectCollectionData
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__

Deformable Object
-----------------

.. note::

  :class:`isaaclab.assets.DeformableObjectCfg` is the shared configuration
  class for deformable objects. The Newton extension exposes the Newton
  implementation of :class:`isaaclab.assets.DeformableObject`, while
  deformable schema and material cfgs referenced by ``spawn`` remain
  backend-specific.

.. autoclass:: DeformableObject
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: DeformableObjectData
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__

MPM Object
----------

.. autoclass:: MPMObject
  :members:
  :inherited-members:
  :show-inheritance:

.. autoclass:: MPMObjectCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

.. autoclass:: MPMObjectData
  :members:
  :inherited-members:
  :show-inheritance:
  :exclude-members: __init__
