Added
^^^^^

* Added :attr:`~isaaclab.sim.converters.AssetConverterBaseCfg.physics_variant` to choose which
  ``"Physics"`` variant the URDF and MJCF converters select on the generated USD file. Defaults to
  the backend-portable ``"physics"``; pass ``"physx"`` or ``"mujoco"`` for solver-specific tuning, or
  ``"none"`` to convert without physics.

Fixed
^^^^^

* Fixed URDF and MJCF conversion producing assets with no joints, articulation roots, or mass
  properties when the importer does not emit the requested physics variant, as happens for a URDF
  whose joints are all fixed. These assets now fall back to the ``"physics"`` variant.
