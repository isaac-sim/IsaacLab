isaaclab_newton.sim.spawners
============================

.. automodule:: isaaclab_newton.sim.spawners.materials

  .. rubric:: Classes

  .. autosummary::

    NewtonDeformableBodyMaterialCfg
    NewtonDeformableMaterialCfg
    NewtonSurfaceDeformableBodyMaterialCfg
    NewtonCableMaterialCfg

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

Cable Material
--------------

Cable rod material parameters for :class:`~isaaclab.sim.spawners.shapes.CableCfg`
and :class:`~isaaclab_contrib.cable.CableObject`. Authored as a
``UsdShade.Material`` with ``newton:*`` attributes via the same
:func:`isaaclab.sim.spawners.materials.spawn_deformable_body_material` helper.

.. autoclass:: NewtonCableMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func
