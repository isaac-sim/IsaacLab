Added
^^^^^

* Added common deformable property and material base cfgs in :mod:`isaaclab.sim`.

Changed
^^^^^^^

* Changed deformable spawners to accept backend-specific deformable property and
  material cfgs. Use PhysX cfgs from :mod:`isaaclab_physx.sim` or Newton cfgs
  from :mod:`isaaclab_newton.sim`.

Deprecated
^^^^^^^^^^

* Deprecated generic deformable property and material cfgs in favor of
  ``PhysxDeformableBodyPropertiesCfg``, ``PhysxDeformableBodyMaterialCfg``,
  ``PhysxSurfaceDeformableBodyMaterialCfg``, ``NewtonDeformableBodyPropertiesCfg``,
  ``NewtonDeformableBodyMaterialCfg``, and ``NewtonSurfaceDeformableBodyMaterialCfg``.
