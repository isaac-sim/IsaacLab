isaaclab.assets
===============

.. automodule:: isaaclab.assets

  .. rubric:: Classes

  .. autosummary::

    AssetBase
    AssetBaseCfg
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
    Articulation
    ArticulationData
    ArticulationCfg
    ArticulationOrderingConvention
    ArticulationNameMap

  .. rubric:: Functions

  .. autosummary::

    apply_articulation_ordering_preset
    build_articulation_name_map
    parse_articulation_ordering_convention
    get_mjwarp_articulation_name_ordering
    get_physx_articulation_name_ordering
    get_robot_schema_articulation_name_ordering
    resolve_articulation_convention_name_ordering
    resolve_articulation_ordering_names

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

.. autoclass:: ArticulationData
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

.. autofunction:: build_articulation_name_map

.. autofunction:: parse_articulation_ordering_convention

.. autofunction:: get_mjwarp_articulation_name_ordering

.. autofunction:: get_physx_articulation_name_ordering

.. autofunction:: get_robot_schema_articulation_name_ordering

.. autofunction:: resolve_articulation_convention_name_ordering

.. autofunction:: resolve_articulation_ordering_names
