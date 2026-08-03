Added
^^^^^

* Added :attr:`~isaaclab.sim.converters.AssetConverterBaseCfg.physics_variant` to choose which
  ``"Physics"`` variant the URDF and MJCF converters select on the generated USD file. Defaults to
  ``"physx"``; pass ``"mujoco"`` for the MuJoCo actuators or ``"none"`` to convert without physics.

Fixed
^^^^^

* Fixed URDF and MJCF conversion producing assets with no joints, articulation roots, or mass
  properties when the importer offers no ``"physx"`` variant, as happens for a URDF whose joints are
  all fixed. These assets now fall back to the backend-neutral ``"physics"`` variant.
