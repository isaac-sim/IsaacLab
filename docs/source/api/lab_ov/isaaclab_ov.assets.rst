isaaclab\_ov.assets
====================

.. automodule:: isaaclab_ov.assets
  :noindex:

  .. rubric:: Classes

  .. autosummary::

    DeformableObject
    DeformableObjectData

.. currentmodule:: isaaclab_ov.assets

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
    class for deformable objects. The ``isaaclab_ov`` extension provides the OVPhysX
    implementation of :class:`isaaclab.assets.DeformableObject`, while
    deformable schema and material cfgs referenced by ``spawn`` remain
    backend-specific.
