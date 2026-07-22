isaaclab_physx.sim.spawners
===========================

.. automodule:: isaaclab_physx.sim.spawners.materials

  .. rubric:: Classes

  .. autosummary::

    PhysxRigidBodyMaterialCfg
    PhysxMaterialCfg
    PhysxDeformableMaterialCfg
    PhysxSurfaceDeformableMaterialCfg
    PhysxDeformableBodyMaterialCfg
    PhysxSurfaceDeformableBodyMaterialCfg
    PhysXDeformableMaterialCfg
    DeformableBodyMaterialCfg
    SurfaceDeformableBodyMaterialCfg

Rigid Materials
---------------

.. autoclass:: PhysxRigidBodyMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: PhysxMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

Deformable Materials
--------------------

PhysX provides the backend-specific deformable material cfgs. Deformable material spawning is unified in
:func:`isaaclab.sim.spawners.materials.spawn_deformable_body_material`.

.. autoclass:: PhysxDeformableMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: PhysxSurfaceDeformableMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

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
