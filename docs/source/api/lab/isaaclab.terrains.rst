isaaclab.terrains
=================

.. automodule:: isaaclab.terrains

  .. rubric:: Classes

  .. autosummary::

    TerrainImporter
    TerrainImporterCfg
    TerrainGenerator
    TerrainGeneratorCfg
    SubTerrainBaseCfg


Terrain importer
----------------

.. autoclass:: TerrainImporter
    :members:
    :show-inheritance:

.. autoclass:: TerrainImporterCfg
    :members:
    :exclude-members: __init__, class_type

Terrain generator
-----------------

.. autoclass:: TerrainGenerator
    :members:

.. autoclass:: TerrainGeneratorCfg
    :members:
    :exclude-members: __init__

.. autoclass:: SubTerrainBaseCfg
    :members:
    :exclude-members: __init__

Height fields
-------------

.. automodule:: isaaclab.terrains.height_field

All sub-terrains must inherit from the :class:`HfTerrainBaseCfg` class which contains the common
parameters for all terrains generated from height fields.

.. autoclass:: isaaclab.terrains.height_field.HfTerrainBaseCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Random Uniform Terrain
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.height_field.HfRandomUniformTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Pyramid Sloped Terrain
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.height_field.HfPyramidSlopedTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

.. autoclass:: isaaclab.terrains.height_field.HfInvertedPyramidSlopedTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Pyramid Stairs Terrain
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.height_field.HfPyramidStairsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

.. autoclass:: isaaclab.terrains.height_field.HfInvertedPyramidStairsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Discrete Obstacles Terrain
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.height_field.HfDiscreteObstaclesTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Wave Terrain
^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.height_field.HfWaveTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Stepping Stones Terrain
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.height_field.HfSteppingStonesTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Trimesh terrains
----------------

.. automodule:: isaaclab.terrains.trimesh


Flat terrain
^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshPlaneTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Pyramid terrain
^^^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshPyramidStairsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Inverted pyramid terrain
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshInvertedPyramidStairsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Random grid terrain
^^^^^^^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshRandomGridTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Rails terrain
^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshRailsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Pit terrain
^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshPitTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Box terrain
^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshBoxTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Gap terrain
^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshGapTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Floating ring terrain
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshFloatingRingTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Star terrain
^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshStarTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Repeated Objects Terrain
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: isaaclab.terrains.trimesh.MeshRepeatedPyramidsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

.. autoclass:: isaaclab.terrains.trimesh.MeshRepeatedBoxesTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

.. autoclass:: isaaclab.terrains.trimesh.MeshRepeatedCylindersTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.terrains` API.

.. currentmodule:: isaaclab.terrains

.. autosummary::
   :nosignatures:

   FlatPatchSamplingCfg

.. autoclass:: FlatPatchSamplingCfg
   :show-inheritance:
