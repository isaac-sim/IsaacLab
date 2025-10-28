isaaclab.sim.spawners
=====================

.. automodule:: isaaclab.sim.spawners

  .. rubric:: Submodules

  .. autosummary::

    shapes
    meshes
    lights
    sensors
    from_files
    materials
    wrappers

  .. rubric:: Classes

  .. autosummary::

    SpawnerCfg
    RigidObjectSpawnerCfg
    DeformableObjectSpawnerCfg

Spawners
--------

.. autoclass:: SpawnerCfg
    :members:
    :exclude-members: __init__

.. autoclass:: RigidObjectSpawnerCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

.. autoclass:: DeformableObjectSpawnerCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__

Shapes
------

.. automodule:: isaaclab.sim.spawners.shapes

  .. rubric:: Classes

  .. autosummary::

    ShapeCfg
    CapsuleCfg
    ConeCfg
    CuboidCfg
    CylinderCfg
    SphereCfg

.. autoclass:: ShapeCfg
    :members:
    :exclude-members: __init__, func

.. autofunction:: spawn_capsule

.. autoclass:: CapsuleCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autofunction:: spawn_cone

.. autoclass:: ConeCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autofunction:: spawn_cuboid

.. autoclass:: CuboidCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autofunction:: spawn_cylinder

.. autoclass:: CylinderCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autofunction:: spawn_sphere

.. autoclass:: SphereCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

Meshes
------

.. automodule:: isaaclab.sim.spawners.meshes

  .. rubric:: Classes

  .. autosummary::

    MeshCfg
    MeshCapsuleCfg
    MeshConeCfg
    MeshCuboidCfg
    MeshCylinderCfg
    MeshSphereCfg

.. autoclass:: MeshCfg
    :members:
    :exclude-members: __init__, func

.. autofunction:: spawn_mesh_capsule

.. autoclass:: MeshCapsuleCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autofunction:: spawn_mesh_cone

.. autoclass:: MeshConeCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autofunction:: spawn_mesh_cuboid

.. autoclass:: MeshCuboidCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autofunction:: spawn_mesh_cylinder

.. autoclass:: MeshCylinderCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autofunction:: spawn_mesh_sphere

.. autoclass:: MeshSphereCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

Lights
------

.. automodule:: isaaclab.sim.spawners.lights

  .. rubric:: Classes

  .. autosummary::

    LightCfg
    CylinderLightCfg
    DiskLightCfg
    DistantLightCfg
    DomeLightCfg
    SphereLightCfg

.. autofunction:: spawn_light

.. autoclass:: LightCfg
    :members:
    :exclude-members: __init__, func

.. autoclass:: CylinderLightCfg
    :members:
    :exclude-members: __init__, func

.. autoclass:: DiskLightCfg
    :members:
    :exclude-members: __init__, func

.. autoclass:: DistantLightCfg
    :members:
    :exclude-members: __init__, func

.. autoclass:: DomeLightCfg
    :members:
    :exclude-members: __init__, func

.. autoclass:: SphereLightCfg
    :members:
    :exclude-members: __init__, func

Sensors
-------

.. automodule:: isaaclab.sim.spawners.sensors

  .. rubric:: Classes

  .. autosummary::

    PinholeCameraCfg
    FisheyeCameraCfg

.. autofunction:: spawn_camera

.. autoclass:: PinholeCameraCfg
    :members:
    :exclude-members: __init__, func

.. autoclass:: FisheyeCameraCfg
    :members:
    :exclude-members: __init__, func

From Files
----------

.. automodule:: isaaclab.sim.spawners.from_files

  .. rubric:: Classes

  .. autosummary::

    UrdfFileCfg
    UsdFileCfg
    GroundPlaneCfg

.. autofunction:: spawn_from_urdf

.. autoclass:: UrdfFileCfg
    :members:
    :exclude-members: __init__, func

.. autofunction:: spawn_from_usd

.. autoclass:: UsdFileCfg
    :members:
    :exclude-members: __init__, func

.. autofunction:: spawn_ground_plane

.. autoclass:: GroundPlaneCfg
    :members:
    :exclude-members: __init__, func

Materials
---------

.. automodule:: isaaclab.sim.spawners.materials

  .. rubric:: Classes

  .. autosummary::

    VisualMaterialCfg
    PreviewSurfaceCfg
    MdlFileCfg
    GlassMdlCfg
    PhysicsMaterialCfg
    RigidBodyMaterialCfg
    DeformableBodyMaterialCfg

Visual Materials
~~~~~~~~~~~~~~~~

.. autoclass:: VisualMaterialCfg
    :members:
    :exclude-members: __init__, func

.. autofunction:: spawn_preview_surface

.. autoclass:: PreviewSurfaceCfg
    :members:
    :exclude-members: __init__, func

.. autofunction:: spawn_from_mdl_file

.. autoclass:: MdlFileCfg
    :members:
    :exclude-members: __init__, func

.. autoclass:: GlassMdlCfg
    :members:
    :exclude-members: __init__, func

Physical Materials
~~~~~~~~~~~~~~~~~~

.. autoclass:: PhysicsMaterialCfg
    :members:
    :exclude-members: __init__, func

.. autofunction:: spawn_rigid_body_material

.. autoclass:: RigidBodyMaterialCfg
    :members:
    :exclude-members: __init__, func

.. autofunction:: spawn_deformable_body_material

.. autoclass:: DeformableBodyMaterialCfg
    :members:
    :exclude-members: __init__, func

Wrappers
--------

.. automodule:: isaaclab.sim.spawners.wrappers

  .. rubric:: Classes

  .. autosummary::

    MultiAssetSpawnerCfg
    MultiUsdFileCfg

.. autofunction:: spawn_multi_asset

.. autoclass:: MultiAssetSpawnerCfg
    :members:
    :exclude-members: __init__, func

.. autofunction:: spawn_multi_usd_file

.. autoclass:: MultiUsdFileCfg
    :members:
    :exclude-members: __init__, func
