isaaclab_newton.sim.spawners
============================

.. automodule:: isaaclab_newton.sim.spawners.materials

  .. rubric:: Classes

  .. autosummary::

    NewtonDeformableBodyMaterialCfg
    NewtonDeformableMaterialCfg
    NewtonSurfaceDeformableBodyMaterialCfg

Deformable Materials
--------------------

Newton provides the backend-specific deformable material cfgs. Deformable material spawning is unified in
:func:`isaaclab.sim.spawners.materials.spawn_deformable_body_material`.

.. autoclass:: NewtonDeformableBodyMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: NewtonDeformableMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: NewtonSurfaceDeformableBodyMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func
