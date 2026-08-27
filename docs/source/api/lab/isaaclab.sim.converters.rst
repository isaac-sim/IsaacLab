isaaclab.sim.converters
=======================

.. automodule:: isaaclab.sim.converters

  .. rubric:: Classes

  .. autosummary::

    AssetConverterBase
    AssetConverterBaseCfg
    MeshConverter
    MeshConverterCfg
    UrdfConverter
    UrdfConverterCfg
    MjcfConverter
    MjcfConverterCfg

Asset Converter Base
--------------------

.. autoclass:: AssetConverterBase
    :members:

.. autoclass:: AssetConverterBaseCfg
    :members:
    :exclude-members: __init__

Mesh Converter
--------------

.. autoclass:: MeshConverter
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: MeshConverterCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, PhysicsVariant


URDF Converter
--------------

.. note::

    Xacro files are not accepted directly by :class:`UrdfConverter`. Expand the Xacro description to a plain
    URDF first, then pass the generated ``.urdf`` file to Isaac Lab. With the ROS ``xacro`` command installed:

    .. code-block:: bash

       xacro path/to/robot.urdf.xacro > path/to/robot.urdf
       uv run python scripts/tools/convert_urdf.py path/to/robot.urdf path/to/output_dir

    Xacro expansion resolves macros and Xacro arguments, but it does not generally resolve ``package://`` mesh
    URLs in the generated URDF. Rewrite those URLs to resolvable filesystem paths before using the CLI, or use
    the Python API and provide package mappings through :attr:`UrdfConverterCfg.ros_package_paths`. The resulting
    URDF can then use the normal :class:`UrdfConverterCfg` options for collision geometry, joint drives,
    fixed-joint merging, and USD output.

.. autoclass:: UrdfConverter
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: UrdfConverterCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, PhysicsVariant

MJCF Converter
--------------

.. autoclass:: MjcfConverter
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: MjcfConverterCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, PhysicsVariant
