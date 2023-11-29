orbit.command\_generators
=========================

.. automodule:: omni.isaac.orbit.command_generators

  .. rubric:: Classes

  .. autosummary::

    CommandGeneratorBase
    CommandGeneratorBaseCfg
    NullCommandGenerator
    NullCommandGeneratorCfg
    UniformVelocityCommandGenerator
    UniformVelocityCommandGeneratorCfg
    NormalVelocityCommandGenerator
    NormalVelocityCommandGeneratorCfg
    TerrainBasedPositionCommandGenerator
    TerrainBasedPositionCommandGeneratorCfg

Command Generator Base
----------------------

.. autoclass:: CommandGeneratorBase
    :members:

.. autoclass:: CommandGeneratorBaseCfg
    :members:
    :exclude-members: __init__, class_type

Null Command Generator
----------------------

.. autoclass:: NullCommandGenerator
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: NullCommandGeneratorCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type, resampling_time_range

Uniform SE(2) Velocity Command Generator
----------------------------------------

.. autoclass:: UniformVelocityCommandGenerator
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: UniformVelocityCommandGeneratorCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Normal SE(2) Velocity Command Generator
---------------------------------------

.. autoclass:: NormalVelocityCommandGenerator
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: NormalVelocityCommandGeneratorCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Uniform SE(3) Pose Command Generator
------------------------------------

.. autoclass:: UniformPoseCommandGenerator
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: UniformPoseCommandGeneratorCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type


Terrain-based SE(2) Position Command Generator
----------------------------------------------

.. note::
    This command generator is currently not tested. It may not work as expected.

.. autoclass:: TerrainBasedPositionCommandGenerator
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: TerrainBasedPositionCommandGeneratorCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type
