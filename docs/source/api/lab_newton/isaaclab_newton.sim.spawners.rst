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

.. automodule:: isaaclab_newton.sim.spawners.mpm

  .. rubric:: Classes

  .. autosummary::

    MPMParticleSpawnerCfg
    MPMGridCfg
    MPMPointsCfg
    MPMParticleMaterialCfg

MPM Particles
-------------

Declarative particle generation for :class:`~isaaclab_newton.assets.MPMObject`.
The spawner creates a placeholder ``Xform`` prim; the particles themselves are
emitted into the Newton model builder during replication.

.. autoclass:: MPMParticleSpawnerCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: MPMGridCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: MPMPointsCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__, func

.. autoclass:: MPMParticleMaterialCfg
    :members:
    :show-inheritance:
    :exclude-members: __init__
