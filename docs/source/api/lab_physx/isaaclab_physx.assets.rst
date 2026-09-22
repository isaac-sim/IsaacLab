isaaclab_physx.assets
=====================

.. automodule:: isaaclab_physx.assets
  :noindex:

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
    SurfaceGripper
    SurfaceGripperCfg

.. currentmodule:: isaaclab_physx.assets

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

.. autoclass:: DeformableObject
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: DeformableObjectData
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__

.. note::

    :class:`isaaclab.assets.DeformableObjectCfg` is the shared configuration
    class for deformable objects. The PhysX extension provides the PhysX
    implementation of :class:`isaaclab.assets.DeformableObject`, while
    deformable schema and material cfgs referenced by ``spawn`` remain
    backend-specific.

Surface Gripper
---------------

.. autoclass:: SurfaceGripper
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: SurfaceGripperCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type, InitialStateCfg
