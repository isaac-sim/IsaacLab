isaaclab.assets
===============

.. automodule:: isaaclab.assets

  .. rubric:: Classes

  .. autosummary::

    AssetBase
    AssetBaseCfg
    BaseCableObject
    BaseCableObjectData
    CableObject
    CableObjectData
    CableObjectCfg
    RigidObject
    RigidObjectData
    RigidObjectCfg
    RigidObjectCollection
    RigidObjectCollectionData
    RigidObjectCollectionCfg
    BaseDeformableObject
    BaseDeformableObjectData
    DeformableObject
    DeformableObjectData
    DeformableObjectCfg
    BaseArticulation
    BaseArticulationData
    Articulation
    ArticulationData
    ArticulationCfg
    ArticulationOrderingConvention
    ArticulationNameMap

  .. rubric:: Functions

  .. autosummary::

    apply_articulation_ordering_preset
    parse_articulation_ordering_convention
    get_articulation_name_ordering

.. currentmodule:: isaaclab.assets

Asset Base
----------

.. autoclass:: AssetBase
    :members:

.. autoclass:: AssetBaseCfg
    :members:
    :exclude-members: __init__, class_type, InitialStateCfg

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

.. autoclass:: RigidObjectCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

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

.. autoclass:: RigidObjectCollectionCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Cable Object
------------

Cable object dynamics are currently supported only by the Newton backend.

.. autoclass:: CableObject
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: BaseCableObject
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: CableObjectData
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: BaseCableObjectData
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: CableObjectCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type, InitialStateCfg

Deformable Object
-----------------

.. autoclass:: DeformableObject
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: BaseDeformableObject
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: DeformableObjectData
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: BaseDeformableObjectData
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: DeformableObjectCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type, InitialStateCfg

Articulation
------------

.. autoclass:: Articulation
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: BaseArticulation
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ArticulationData
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: BaseArticulationData
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: ArticulationCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Articulation Ordering
---------------------

.. autoclass:: ArticulationOrderingConvention
    :members:

.. autoclass:: ArticulationNameMap
    :members:

.. autofunction:: apply_articulation_ordering_preset

.. autofunction:: parse_articulation_ordering_convention

.. autofunction:: get_articulation_name_ordering
