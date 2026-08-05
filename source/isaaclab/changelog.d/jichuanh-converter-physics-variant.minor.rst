Added
^^^^^

* Added :attr:`~isaaclab.sim.converters.AssetConverterBaseCfg.physics_variant` to choose which
  ``"Physics"`` variant the URDF and MJCF converters select on the generated USD file. Defaults to
  the backend-portable :attr:`~isaaclab.sim.converters.AssetConverterBaseCfg.PhysicsVariant.PHYSICS`;
  select :attr:`~isaaclab.sim.converters.AssetConverterBaseCfg.PhysicsVariant.PHYSX` or
  :attr:`~isaaclab.sim.converters.AssetConverterBaseCfg.PhysicsVariant.MUJOCO` for solver-specific
  tuning, or :attr:`~isaaclab.sim.converters.AssetConverterBaseCfg.PhysicsVariant.NONE` to convert
  without physics.

Fixed
^^^^^

* Fixed URDF and MJCF conversion producing assets with no joints, articulation roots, or mass
  properties when the importer does not emit the requested physics variant, as happens for a URDF
  whose joints are all fixed. These assets now fall back to the ``"physics"`` variant.

* Fixed :meth:`~isaaclab.utils.dict.class_to_dict` expanding enum values into their internal
  members, which wrote unusable entries into serialized configurations.
