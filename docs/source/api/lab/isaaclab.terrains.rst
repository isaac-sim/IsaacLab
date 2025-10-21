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

.. autoclass:: isaaclab.terrains.height_field.hf_terrains_cfg.HfTerrainBaseCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Random Uniform Terrain
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.height_field.hf_terrains.random_uniform_terrain

.. autoclass:: isaaclab.terrains.height_field.hf_terrains_cfg.HfRandomUniformTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Pyramid Sloped Terrain
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.height_field.hf_terrains.pyramid_sloped_terrain

.. autoclass:: isaaclab.terrains.height_field.hf_terrains_cfg.HfPyramidSlopedTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

.. autoclass:: isaaclab.terrains.height_field.hf_terrains_cfg.HfInvertedPyramidSlopedTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Pyramid Stairs Terrain
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.height_field.hf_terrains.pyramid_stairs_terrain

.. autoclass:: isaaclab.terrains.height_field.hf_terrains_cfg.HfPyramidStairsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

.. autoclass:: isaaclab.terrains.height_field.hf_terrains_cfg.HfInvertedPyramidStairsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Discrete Obstacles Terrain
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.height_field.hf_terrains.discrete_obstacles_terrain

.. autoclass:: isaaclab.terrains.height_field.hf_terrains_cfg.HfDiscreteObstaclesTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Wave Terrain
^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.height_field.hf_terrains.wave_terrain

.. autoclass:: isaaclab.terrains.height_field.hf_terrains_cfg.HfWaveTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Stepping Stones Terrain
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.height_field.hf_terrains.stepping_stones_terrain

.. autoclass:: isaaclab.terrains.height_field.hf_terrains_cfg.HfSteppingStonesTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Trimesh terrains
----------------

.. automodule:: isaaclab.terrains.trimesh


Flat terrain
^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.flat_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshPlaneTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Pyramid terrain
^^^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.pyramid_stairs_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshPyramidStairsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Inverted pyramid terrain
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.inverted_pyramid_stairs_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshInvertedPyramidStairsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Random grid terrain
^^^^^^^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.random_grid_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshRandomGridTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Rails terrain
^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.rails_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshRailsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Pit terrain
^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.pit_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshPitTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Box terrain
^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.box_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshBoxTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Gap terrain
^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.gap_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshGapTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Floating ring terrain
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.floating_ring_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshFloatingRingTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Star terrain
^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.star_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshStarTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Repeated Objects Terrain
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: isaaclab.terrains.trimesh.mesh_terrains.repeated_objects_terrain

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshRepeatedObjectsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshRepeatedPyramidsTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshRepeatedBoxesTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

.. autoclass:: isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshRepeatedCylindersTerrainCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, function

Utilities
---------

.. automodule:: isaaclab.terrains.utils
    :members:
    :undoc-members:
