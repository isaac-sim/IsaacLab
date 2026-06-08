isaaclab_physx.sim.spawners
===========================

.. automodule:: isaaclab_physx.sim.spawners.materials

  .. rubric:: Classes

  .. autosummary::

    PhysxDeformableBodyMaterialCfg
    PhysxSurfaceDeformableBodyMaterialCfg
    PhysXDeformableMaterialCfg
    DeformableBodyMaterialCfg
    SurfaceDeformableBodyMaterialCfg

Deformable Materials
--------------------

PhysX provides the backend-specific deformable material cfgs. Deformable material spawning is unified in
:func:`isaaclab.sim.spawners.materials.spawn_deformable_body_material`.

.. autoclass:: PhysxDeformableBodyMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: PhysxSurfaceDeformableBodyMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: PhysXDeformableMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

Deprecated Aliases
------------------

.. autoclass:: DeformableBodyMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: SurfaceDeformableBodyMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func
